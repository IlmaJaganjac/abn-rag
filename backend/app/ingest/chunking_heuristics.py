from __future__ import annotations

import html
import re
from collections.abc import Iterable

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$")
YEAR_RE = re.compile(r"\b20\d{2}\b")
VALUE_RE = re.compile(
    r"^\s*(?:[€$£]?\s*)?(?:[<>]?\s*)?\d[\d,.]*(?:\s?(?:%|bn|m|kt|mt|million|billion))?\s*$",
    re.IGNORECASE,
)
READ_MORE_RE = re.compile(r"^read more on page \d+\s*>?$", re.IGNORECASE)
PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")

SENTENCE_HEADING_PREFIXES = (
    "we ",
    "our ",
    "the ",
    "this ",
    "these ",
    "as ",
    "in ",
    "while ",
    "today",
    "at the ",
    "closer to ",
    "turning ",
    "finally",
    "i ",
)

BOILERPLATE_PHRASES: frozenset[str] = frozenset(
    {
        "strategic report",
        "corporate governance",
        "sustainability",
        "financials",
        "at a glance",
        "q&#x26;a with the ceo",
        "q&a with the ceo",
        "our business",
        "financial performance",
        "risk and security",
        "general disclosures",
        "environmental",
        "social",
        "governance",
    }
)


def clean_cell(text: str) -> str:
    """Normalize one table cell into a single cleaned text value."""
    return " ".join(html.unescape(text).replace("\\&", "&").split())


def clean_heading(text: str) -> str:
    """Clean a heading line and return the heading text without markdown markers."""
    return clean_cell(text).strip("# ")


def clean_block(text: str) -> str:
    """Trim leading and trailing blank lines and return the cleaned block text."""
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def normalize_line(line: str) -> str:
    """Normalize one line for case-insensitive comparisons and boilerplate checks."""
    line = line.strip()
    heading = HEADING_RE.match(line)
    if heading:
        line = heading.group(2)
    line = clean_cell(line)
    return line.casefold()


def is_table_line(line: str) -> bool:
    """Return whether a line looks like a markdown table row."""
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def parse_table_cells(row: str) -> list[str]:
    """Split one markdown table row into cleaned cell values."""
    stripped = row.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [clean_cell(cell) for cell in stripped.split("|")]


def looks_like_value(text: str) -> bool:
    """Return whether text looks like a numeric value or standalone year."""
    clean = text.replace("&nbsp;", " ").strip()
    return bool(VALUE_RE.match(clean)) or bool(YEAR_RE.fullmatch(clean))


def starts_with_value(text: str) -> bool:
    """Return whether text begins with a numeric-looking value."""
    clean = text.replace("&nbsp;", " ").strip()
    return bool(re.match(r"^[€$£]?\s*[<>]?\s*\d", clean))


def looks_like_heading(text: str) -> bool:
    """Heuristically decide whether a text line should be treated as a heading."""
    clean = clean_heading(text)
    if not clean:
        return False
    normalized = clean.casefold()
    if READ_MORE_RE.match(normalized):
        return False
    if clean.startswith("Q:") or clean.startswith("Q "):
        return True
    if len(clean) <= 60 and clean[-1] not in ".!?":
        return True
    if len(clean) > 100:
        return False
    if clean.endswith("."):
        return False
    if any(normalized.startswith(prefix) for prefix in SENTENCE_HEADING_PREFIXES):
        return False
    if "," in clean and len(clean) > 60:
        return False
    return True


def is_noise_line(line: str) -> bool:
    """Return whether a line is layout noise such as page numbers or read-more hints."""
    if not line:
        return False
    normalized = normalize_line(line)
    return bool(READ_MORE_RE.match(normalized) or PAGE_NUMBER_RE.match(normalized))


def is_likely_boilerplate(line: str) -> bool:
    """Return whether a normalized line is likely repeated report boilerplate."""
    if len(line) <= 2:
        return False
    if "annual report" in line:
        return True
    return line in BOILERPLATE_PHRASES


def year_period(text: str) -> str | None:
    """Return the extracted year when text represents a year-like period label."""
    match = re.fullmatch(r"(?:FY\s*|Year\s*)?(20\d{2})", text.strip(), re.IGNORECASE)
    return match.group(1) if match else None



def remove_boilerplate(text: str, boilerplate: set[str]) -> str:
    """Remove normalized boilerplate lines from a block and return the cleaned text."""
    lines = [
        line
        for line in text.splitlines()
        if normalize_line(line) not in boilerplate
    ]
    return clean_block("\n".join(lines))


def find_boilerplate_lines(texts: Iterable[str]) -> set[str]:
    """Find repeated boilerplate lines across pages and return them as normalized text."""
    from collections import Counter
    per_page: list[set[str]] = []
    for text in texts:
        lines = {
            norm
            for line in text.splitlines()
            if (norm := normalize_line(line)) and not is_table_line(line.strip())
        }
        per_page.append(lines)
    if not per_page:
        return set()
    counts = Counter(line for lines in per_page for line in lines)
    threshold = max(3, len(per_page) // 8)
    return {
        line
        for line, count in counts.items()
        if count >= threshold and is_likely_boilerplate(line)
    }
