from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from backend.app.schemas import Chunk

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$")
YEAR_RE = re.compile(r"\b20\d{2}\b")
VALUE_RE = re.compile(
    r"^\s*(?:[€$£]?\s*)?(?:[<>]?\s*)?\d[\d,.]*(?:\s?(?:%|bn|m|kt|mt|million|billion))?\s*$",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class ChunkDraft:
    page: int
    text: str
    chunk_kind: str
    section_path: str | None
    embedding_text: str


def build_semantic_chunks(
    pages: Iterable[tuple[int, str]],
    *,
    source: str,
    company: str | None,
    year: int | None,
    parser: str | None,
    max_tokens: int,
    overlap: int,
    token_counter: Callable[[str], int],
    split_oversize: Callable[[str, int, int], list[str]],
) -> list[Chunk]:
    page_list = list(pages)
    boilerplate = _find_boilerplate_lines(text for _, text in page_list)
    drafts: list[ChunkDraft] = []
    heading_stack: list[tuple[int, str]] = []
    for page, text in page_list:
        drafts.extend(
            _page_drafts(
                page=page,
                text=text,
                heading_stack=heading_stack,
                company=company,
                year=year,
                parser=parser,
                boilerplate=boilerplate,
                max_tokens=max_tokens,
                overlap=overlap,
                token_counter=token_counter,
                split_oversize=split_oversize,
            )
        )

    chunks: list[Chunk] = []
    idx_by_page: defaultdict[int, int] = defaultdict(int)
    for draft in drafts:
        idx = idx_by_page[draft.page]
        idx_by_page[draft.page] += 1
        chunks.append(
            Chunk(
                id=f"{source}:{draft.page}:{idx}",
                source=source,
                company=company,
                year=year,
                page=draft.page,
                text=draft.text,
                token_count=token_counter(draft.text),
                parser=parser,
                chunk_kind=draft.chunk_kind,
                section_path=draft.section_path,
                embedding_text=draft.embedding_text,
            )
        )
    return chunks


def _page_drafts(
    *,
    page: int,
    text: str,
    heading_stack: list[tuple[int, str]],
    company: str | None,
    year: int | None,
    parser: str | None,
    boilerplate: set[str],
    max_tokens: int,
    overlap: int,
    token_counter: Callable[[str], int],
    split_oversize: Callable[[str, int, int], list[str]],
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    narrative: list[str] = []
    table: list[str] = []

    def section_path() -> str | None:
        headings = [
            clean
            for _, clean in heading_stack
            if _normalize_line(clean) not in boilerplate
        ]
        return " > ".join(headings) if headings else None

    def flush_narrative() -> None:
        nonlocal narrative
        body = _remove_boilerplate("\n".join(narrative), boilerplate)
        narrative = []
        if not body:
            return
        for part in split_oversize(body, max_tokens, overlap):
            drafts.append(
                _draft(
                    page=page,
                    text=part,
                    chunk_kind="section",
                    section_path=section_path(),
                    company=company,
                    year=year,
                    parser=parser,
                    boilerplate=boilerplate,
                )
            )

    def flush_table() -> None:
        nonlocal table
        rows = table
        table = []
        if not rows:
            return
        drafts.extend(
            _table_drafts(
                page=page,
                rows=rows,
                section_path=section_path(),
                company=company,
                year=year,
                parser=parser,
                boilerplate=boilerplate,
                max_tokens=max_tokens,
                overlap=overlap,
                token_counter=token_counter,
                split_oversize=split_oversize,
            )
        )

    for line in text.splitlines():
        stripped = line.strip()
        heading = HEADING_RE.match(stripped)
        if heading:
            flush_table()
            flush_narrative()
            level = len(heading.group(1))
            title = _clean_heading(heading.group(2))
            if _normalize_line(title) in boilerplate:
                continue
            heading_stack[:] = [(lvl, val) for lvl, val in heading_stack if lvl < level]
            heading_stack.append((level, title))
            continue

        if _normalize_line(stripped) in boilerplate:
            continue

        if _is_table_line(stripped):
            flush_narrative()
            table.append(stripped)
            continue

        if table:
            flush_table()
        narrative.append(line)

    flush_table()
    flush_narrative()

    return drafts


def _table_drafts(
    *,
    page: int,
    rows: list[str],
    section_path: str | None,
    company: str | None,
    year: int | None,
    parser: str | None,
    boilerplate: set[str],
    max_tokens: int,
    overlap: int,
    token_counter: Callable[[str], int],
    split_oversize: Callable[[str, int, int], list[str]],
) -> list[ChunkDraft]:
    drafts: list[ChunkDraft] = []
    table_kind = _classify_table(rows)
    full_table = _clean_block("\n".join(rows))
    if full_table:
        table_context = _table_context(section_path, rows, table_kind)
        for part in _split_table_on_rows(
            rows=rows,
            max_tokens=max_tokens,
            token_counter=token_counter,
        ):
            drafts.append(
                _draft(
                    page=page,
                    text=part,
                    chunk_kind="table",
                    section_path=section_path,
                    company=company,
                    year=year,
                    parser=parser,
                    boilerplate=boilerplate,
                    extra_embedding_context=table_context,
                )
            )

    data_rows = [row for row in rows if not TABLE_SEPARATOR_RE.match(row)]
    headers, body_rows = _table_headers_and_body(data_rows)

    if table_kind == "header_table":
        for row in body_rows:
            cells = _parse_table_cells(row)
            row_text = _format_header_aware_row(headers, cells)
            if not row_text:
                continue
            drafts.append(
                _draft(
                    page=page,
                    text=row_text,
                    chunk_kind="table_row",
                    section_path=section_path,
                    company=company,
                    year=year,
                    parser=parser,
                    boilerplate=boilerplate,
                    extra_embedding_context=_table_context(section_path, rows, table_kind),
                )
            )
    return drafts


def _split_table_on_rows(
    *,
    rows: list[str],
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> list[str]:
    full_table = _clean_block("\n".join(rows))
    if not full_table or token_counter(full_table) <= max_tokens:
        return [full_table] if full_table else []

    header_rows: list[str] = []
    body_start = 0
    if rows:
        header_rows.append(rows[0])
        body_start = 1
    if len(rows) > 1 and TABLE_SEPARATOR_RE.match(rows[1]):
        header_rows.append(rows[1])
        body_start = 2

    header_text = "\n".join(header_rows)
    body_rows = rows[body_start:]
    if not body_rows:
        return [full_table]

    parts: list[str] = []
    current: list[str] = []

    def current_text() -> str:
        return "\n".join(header_rows + current)

    for row in body_rows:
        candidate_rows = current + [row]
        candidate = "\n".join(header_rows + candidate_rows)
        if current and token_counter(candidate) > max_tokens:
            parts.append(_clean_block(current_text()))
            current = [row]
        else:
            current = candidate_rows

    if current:
        parts.append(_clean_block(current_text()))

    return parts or [full_table]


def _draft(
    *,
    page: int,
    text: str,
    chunk_kind: str,
    section_path: str | None,
    company: str | None,
    year: int | None,
    parser: str | None,
    boilerplate: set[str],
    extra_embedding_context: str | None = None,
) -> ChunkDraft:
    body = _remove_boilerplate(text, boilerplate).strip()
    context = _embedding_context(
        company=company,
        year=year,
        parser=parser,
        section_path=section_path,
        extra=extra_embedding_context,
    )
    embedding_text = "\n".join(part for part in (context, body) if part).strip()
    return ChunkDraft(
        page=page,
        text=body,
        chunk_kind=chunk_kind,
        section_path=section_path,
        embedding_text=embedding_text,
    )


def _embedding_context(
    *,
    company: str | None,
    year: int | None,
    parser: str | None,
    section_path: str | None,
    extra: str | None = None,
) -> str:
    parts = [
        company,
        str(year) if year is not None else None,
        f"Section: {section_path}" if section_path else None,
        extra,
    ]
    return "\n".join(part for part in parts if part)


def _table_context(section_path: str | None, rows: list[str], table_kind: str) -> str | None:
    data_rows = [row for row in rows if not TABLE_SEPARATOR_RE.match(row)]
    headers, _ = _table_headers_and_body(data_rows)
    parts: list[str] = [f"Table type: {table_kind}"]
    if section_path:
        parts.append(f"Table caption: {section_path}")
    if headers:
        columns = [header for header in headers if header.strip()]
        if columns:
            parts.append("Columns: " + " | ".join(columns))
    return "\n".join(parts) if parts else None


def _classify_table(rows: list[str]) -> str:
    data_rows = [row for row in rows if not TABLE_SEPARATOR_RE.match(row)]
    headers, body_rows = _table_headers_and_body(data_rows)
    if headers is not None and body_rows and _looks_like_header_row(headers):
        return "header_table"
    if data_rows and _mostly_metric_pairs(_parse_table_cells(data_rows[0])):
        return "kpi_pairs"
    if headers is not None and body_rows:
        return "header_table"
    return "generic_table"


def _looks_like_header_row(cells: list[str]) -> bool:
    normalized = {_normalize_line(cell) for cell in cells if cell.strip()}
    if any(_year_period(cell) is not None for cell in cells):
        return True
    return bool(normalized & {"notes", "note", "description", "metric", "topic"})


def _mostly_metric_pairs(cells: list[str]) -> bool:
    non_empty = [cell for cell in cells if cell.strip()]
    if len(non_empty) < 2:
        return False
    return len(_metric_pairs(non_empty)) * 2 >= len(non_empty)


def _metric_pairs(cells: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    i = 0
    while i + 1 < len(cells):
        value = cells[i].strip()
        label = cells[i + 1].strip()
        if value and label and _looks_like_value(value) and not _looks_like_value(label):
            pairs.append((value, label))
            i += 2
        elif (
            value
            and label
            and YEAR_RE.search(value)
            and _starts_with_value(label)
            and not _looks_like_value(label)
        ):
            pairs.append((label, value))
            i += 2
        else:
            i += 1
    return pairs


def _year_period(text: str) -> str | None:
    match = re.fullmatch(r"(?:FY\s*|Year\s*)?(20\d{2})", text.strip(), re.IGNORECASE)
    return match.group(1) if match else None


def _table_headers_and_body(data_rows: list[str]) -> tuple[list[str] | None, list[str]]:
    if len(data_rows) < 2:
        return None, data_rows
    headers = _parse_table_cells(data_rows[0])
    if not headers or not any(header.strip() for header in headers):
        return None, data_rows
    return headers, data_rows[1:]


def _format_header_aware_row(headers: list[str] | None, cells: list[str]) -> str | None:
    if headers is None or len(headers) != len(cells):
        return None
    lines = [
        f"{header.strip()}: {cell.strip()}"
        for header, cell in zip(headers, cells, strict=True)
        if header.strip() and cell.strip()
    ]
    return "\n".join(lines) if lines else None


def _parse_table_cells(row: str) -> list[str]:
    stripped = row.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [_clean_cell(cell) for cell in stripped.split("|")]


def _format_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _looks_like_value(text: str) -> bool:
    clean = text.replace("&nbsp;", " ").strip()
    return bool(VALUE_RE.match(clean)) or bool(YEAR_RE.fullmatch(clean))


def _starts_with_value(text: str) -> bool:
    clean = text.replace("&nbsp;", " ").strip()
    return bool(re.match(r"^[€$£]?\s*[<>]?\s*\d", clean))


def _is_table_line(line: str) -> bool:
    return line.startswith("|") and line.endswith("|") and line.count("|") >= 2


def _find_boilerplate_lines(texts: Iterable[str]) -> set[str]:
    per_page: list[set[str]] = []
    for text in texts:
        lines = {
            norm
            for line in text.splitlines()
            if (norm := _normalize_line(line)) and not _is_table_line(line.strip())
        }
        per_page.append(lines)
    if not per_page:
        return set()
    counts = Counter(line for lines in per_page for line in lines)
    threshold = max(3, len(per_page) // 8)
    return {
        line
        for line, count in counts.items()
        if count >= threshold and _is_likely_boilerplate(line)
    }


def _is_likely_boilerplate(line: str) -> bool:
    if len(line) <= 2:
        return False
    if "annual report" in line:
        return True
    if line in {
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
    }:
        return True
    return False


def _remove_boilerplate(text: str, boilerplate: set[str]) -> str:
    lines = [
        line
        for line in text.splitlines()
        if _normalize_line(line) not in boilerplate
    ]
    return _clean_block("\n".join(lines))


def _clean_block(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines()]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _clean_heading(text: str) -> str:
    return _clean_cell(text).strip("# ")


def _clean_cell(text: str) -> str:
    return " ".join(text.replace("\\&", "&").split())


def _normalize_line(line: str) -> str:
    line = line.strip()
    heading = HEADING_RE.match(line)
    if heading:
        line = heading.group(2)
    line = _clean_cell(line)
    return line.casefold()
