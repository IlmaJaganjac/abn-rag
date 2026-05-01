from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Signal patterns
# ---------------------------------------------------------------------------

_FTE_HIGH = re.compile(
    r"total\s+employees|total\s+employees\s*\(fte|"
    r"full[- ]time\s+equivalents?|average\s+number\s+of\s+employees|"
    r"average\s+number\s+of\s+payroll\s+employees",
    re.IGNORECASE,
)
_FTE_MED = re.compile(
    r"\bftes?\b|payroll\s+employees?|payroll\s+fte|"
    r"number\s+of\s+employees|total\s+headcount|total\s+workforce",
    re.IGNORECASE,
)
_FTE_LOW = re.compile(r"\bworkforce\b|\bheadcount\b|\bemployees?\b", re.IGNORECASE)
_FTE_SPECIFIC = re.compile(
    r"dedicated\s+ftes?|team\s+ftes?|program\s+ftes?|project\s+ftes?",
    re.IGNORECASE,
)

_SUST_TARGET = re.compile(
    r"\btarget\b|\bgoal\b|\bambition\b|\bcommitment\b|\bcommitted\b|\baim\b",
    re.IGNORECASE,
)
_SUST_ESG = re.compile(
    r"net[- ]zero|greenhouse\s+gas|\bghg\b|co2e?|co₂e?|\bemissions?\b|"
    r"\bscope\s+[123]\b|\bclimate\b|\benergy\b|\brenewable\b|"
    r"\bcircularity\b|\bwaste\b|\brecycl",
    re.IGNORECASE,
)
_SUST_FUTURE_YEAR = re.compile(r"\b20[3-9]\d\b")

_KPI_TITLE = re.compile(
    r"at\s+a\s+glance|overview|highlights?\b|"
    r"key\s+performance\s+indicators?|\bkpis?\b",
    re.IGNORECASE,
)
_KPI_METRICS = re.compile(
    r"net\s+sales|gross\s+margin|research\s+and\s+development|\br&d\b|"
    r"system\s+sales|nationalities?|number\s+of\s+suppliers|"
    r"returned\s+to\s+shareholders|customer\s+satisfaction",
    re.IGNORECASE,
)
_TABLE_LINE = re.compile(r"^\|", re.MULTILINE)
_NUMBER_TOKEN = re.compile(
    r"(?:€|\$|£)?\s*>?\s*\d[\d,.]*(?:\s?(?:bn|m|%|kt|Mt|FTEs?|,000))?",
)

CATEGORIES = ("fte", "sustainability", "kpi_highlights")


@dataclass
class PageScore:
    page: int
    score: float


@dataclass
class PageScanResult:
    fte: list[PageScore] = field(default_factory=list)
    sustainability: list[PageScore] = field(default_factory=list)
    kpi_highlights: list[PageScore] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_fte(text: str) -> float:
    score = 0.0
    score += len(_FTE_HIGH.findall(text)) * 5.0
    score += len(_FTE_MED.findall(text)) * 2.0
    score += min(len(_FTE_LOW.findall(text)), 5) * 0.5
    score -= len(_FTE_SPECIFIC.findall(text)) * 3.0
    return max(score, 0.0)


def _score_sustainability(text: str) -> float:
    score = 0.0
    n_target = len(_SUST_TARGET.findall(text))
    n_esg = len(_SUST_ESG.findall(text))
    score += min(n_target, 6) * 1.0
    score += min(n_esg, 10) * 2.0
    if n_target > 0 and n_esg > 0:
        score += 4.0
    score += len(_SUST_FUTURE_YEAR.findall(text)) * 1.5
    return score


def _score_kpi(text: str) -> float:
    score = 0.0
    score += len(_KPI_TITLE.findall(text)) * 8.0
    score += len(_KPI_METRICS.findall(text)) * 2.0
    score += min(len(_TABLE_LINE.findall(text)), 10) * 0.5
    score += min(len(_NUMBER_TOKEN.findall(text)), 20) * 0.3
    return score


def score_page(text: str) -> dict[str, float]:
    return {
        "fte": _score_fte(text),
        "sustainability": _score_sustainability(text),
        "kpi_highlights": _score_kpi(text),
    }


# ---------------------------------------------------------------------------
# Range helpers
# ---------------------------------------------------------------------------

def _expand_with_context(pages: list[int], context: int, all_pages: set[int]) -> list[int]:
    expanded: set[int] = set()
    for p in pages:
        for delta in range(-context, context + 1):
            candidate = p + delta
            if candidate in all_pages:
                expanded.add(candidate)
    return sorted(expanded)


def _to_range_string(pages: list[int]) -> str:
    if not pages:
        return ""
    pages = sorted(set(pages))
    groups: list[list[int]] = []
    current = [pages[0]]
    for p in pages[1:]:
        if p == current[-1] + 1:
            current.append(p)
        else:
            groups.append(current)
            current = [p]
    groups.append(current)
    parts = []
    for g in groups:
        parts.append(str(g[0]) if len(g) == 1 else f"{g[0]}-{g[-1]}")
    return ",".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pages_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def scan_pages(
    pages: list[dict],
    *,
    context_window: int = 1,
    max_pages_per_category: int = 20,
) -> tuple[PageScanResult, dict[str, str]]:
    """Score pages and return top candidates per category.

    Returns (PageScanResult, page_range_strings).
    page_range_strings maps category → range string suitable for LlamaExtract.
    """
    all_page_nos: set[int] = {p["page"] for p in pages}
    scored: dict[str, list[PageScore]] = {c: [] for c in CATEGORIES}

    for record in pages:
        text = record.get("text", "")
        scores = score_page(text)
        for cat in CATEGORIES:
            if scores[cat] > 0:
                scored[cat].append(PageScore(page=record["page"], score=scores[cat]))

    result = PageScanResult()
    ranges: dict[str, str] = {}

    for cat in CATEGORIES:
        ranked = sorted(scored[cat], key=lambda s: s.score, reverse=True)
        top = ranked[:max_pages_per_category]
        top_pages = sorted(p.page for p in top)
        expanded = _expand_with_context(top_pages, context_window, all_page_nos)
        page_scores = sorted(scored[cat], key=lambda s: s.score, reverse=True)
        getattr(result, cat).extend(page_scores)
        ranges[cat] = _to_range_string(expanded)

    return result, ranges
