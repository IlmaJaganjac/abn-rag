from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# FTE signal patterns
# ---------------------------------------------------------------------------

_FTE_HIGH = re.compile(
    r"total\s+employees?(?:\s*\(fte)?|full[- ]time\s+equivalents?|"
    r"average\s+number\s+of\s+(?:payroll\s+)?employees?|"
    r"total\s+number\s+of\s+(?:payroll\s+)?employees?|"
    r"internal\s+employees?|permanent\s+employees?|temporary\s+employees?",
    re.IGNORECASE,
)
_FTE_MED = re.compile(
    r"\bftes?\b|payroll\s+employees?|payroll\s+fte|"
    r"number\s+of\s+employees|total\s+headcount|total\s+workforce|"
    r"full[- ]time\s+employees?|part[- ]time\s+employees?|"
    r"external\s+employees?|non[- ]guaranteed\s+hours|"
    r"year[- ]end\s+employees?|average\s+employees?|"
    r"employee\s+turnover|attrition\s+rate|employees\s+by\s+(?:gender|region|country)",
    re.IGNORECASE,
)
_FTE_LOW = re.compile(
    r"\bworkforce\b|\bheadcount\b|\bemployees?\b|\bpersonnel\b|\bstaff\b|\bcontractors?\b",
    re.IGNORECASE,
)
_FTE_SPECIFIC = re.compile(
    r"dedicated\s+ftes?|team\s+ftes?|program\s+ftes?|project\s+ftes?",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Sustainability signal patterns
# ---------------------------------------------------------------------------

_SUST_TARGET = re.compile(
    r"\btarget\b|\bgoal\b|\bambition\b|\bcommitment\b|\bcommitted\b|\baim\b|"
    r"\bplan\s+to\b|\bintend\s+to\b|\bwe\s+will\b|\breduce\s+by\b|\bachieve\s+by\b|"
    r"net[- ]zero|transition\s+plan|decarbonisati|decarbonizati|"
    r"science\s+based\s+targets?|\bsbti\b|1\.5\s*°?\s*c",
    re.IGNORECASE,
)
_SUST_ESG = re.compile(
    r"net[- ]zero|greenhouse\s+gas|\bghg\b|co2e?|co₂e?|\bemissions?\b|"
    r"\bscope\s+[123]\b|\bclimate\b|\benergy\b|\brenewable\b|"
    r"\bcircularity\b|\bwaste\b|\brecycl|carbon\s+intensity|emission\s+intensity|"
    r"climate\s+change\s+mitigation",
    re.IGNORECASE,
)
_SUST_FUTURE_YEAR = re.compile(r"\b20[3-9]\d\b")

# ---------------------------------------------------------------------------
# Financial highlight signal patterns
# ---------------------------------------------------------------------------

_FIN_TITLE = re.compile(
    r"financial\s+performance|operating\s+results?|performance\s+kpis?|"
    r"at\s+a\s+glance|financial\s+highlights?|key\s+financials?",
    re.IGNORECASE,
)
_FIN_METRICS = re.compile(
    r"net\s+sales|total\s+(?:net\s+)?(?:sales|revenue|income)|\brevenue\b|"
    r"operating\s+income|operating\s+profit|\bebit(?:da)?\b|adjusted\s+ebitda|"
    r"gross\s+profit|gross\s+margin|net\s+income|net\s+profit|profit\s+for\s+the\s+period|"
    r"earnings\s+per\s+share|\beps\b|diluted\s+eps|"
    r"research\s+and\s+development|\br&d\b|"
    r"free\s+cash\s+flow|operating\s+cash\s+flow|cash\s+flow\s+from\s+operat|"
    r"cash\s+and\s+cash\s+equivalents|\bcapex\b|capital\s+expenditure|"
    r"return\s+on\s+(?:equity|invested\s+capital)|\broe\b|\broic\b|"
    r"\bcet1\b|capital\s+ratio|liquidity\s+coverage|net\s+interest\s+margin|\bnim\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Business performance signal patterns
# ---------------------------------------------------------------------------

_BIZ_TITLE = re.compile(
    r"at\s+a\s+glance|business\s+performance|operational\s+highlights?|"
    r"segment\s+performance|divisional\s+performance|regional\s+performance|"
    r"key\s+(?:operational|performance)\s+indicators?|\bkpis?\b",
    re.IGNORECASE,
)
_BIZ_METRICS = re.compile(
    r"systems?\s+sold|lithography\s+systems?|net\s+system\s+sales|units?\s+sold|"
    r"installed\s+base|active\s+systems?|new\s+systems?|used\s+systems?|"
    r"order\s+intake|order\s+book|\bbacklog\b|\bbookings\b|"
    r"number\s+of\s+(?:suppliers?|customers?|clients?)|customer\s+satisfaction|"
    r"reuse\s+rate|market\s+share|production\s+volume|(?:volume\s+of\s+)?deliveries|"
    r"\bloans?\b|\bdeposits?\b|\bmortgages?\b|"
    r"\blng\b|barrels?\s+per\s+day|refining\s+throughput|"
    r"beer\s+volume|\bhectoliters?\b|\bstores?\b|\bbranches?\b|\blocations?\b|"
    r"assets\s+under\s+management|\baum\b|transaction\s+volume",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Shareholder return signal patterns
# ---------------------------------------------------------------------------

_SH_METRICS = re.compile(
    r"returned?\s+to\s+shareholders?|return\s+to\s+shareholders?|"
    r"shareholder\s+(?:returns?|distributions?)|distributions?\s+to\s+shareholders?|"
    r"capital\s+return|\bdividend\b|dividends?\s+paid|ordinary\s+dividend|"
    r"special\s+dividend|final\s+dividend|interim\s+dividend|"
    r"dividend\s+per\s+share|proposed\s+dividend|payout\s+ratio|"
    r"share\s+buybacks?|share\s+repurchases?|\brepurchases?\b|"
    r"treasury\s+shares?|cancellation\s+of\s+shares?",
    re.IGNORECASE,
)

_TABLE_LINE = re.compile(r"^\|", re.MULTILINE)
_NUMBER_TOKEN = re.compile(
    r"(?:€|\$|£)?\s*>?\s*\d[\d,.]*(?:\s?(?:bn|m|%|kt|Mt|FTEs?|,000))?",
)

CATEGORIES = (
    "fte",
    "sustainability",
    "financial_highlight",
    "business_performance",
    "shareholder_return",
)


@dataclass
class PageScore:
    page: int
    score: float


@dataclass
class PageScanResult:
    fte: list[PageScore] = field(default_factory=list)
    sustainability: list[PageScore] = field(default_factory=list)
    financial_highlight: list[PageScore] = field(default_factory=list)
    business_performance: list[PageScore] = field(default_factory=list)
    shareholder_return: list[PageScore] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring functions
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
    # bonus when target language co-occurs with climate/ESG terms
    if n_target > 0 and n_esg > 0:
        score += 4.0
    score += len(_SUST_FUTURE_YEAR.findall(text)) * 1.5
    return score


def _score_financial_highlight(text: str) -> float:
    score = 0.0
    n_title = len(_FIN_TITLE.findall(text))
    n_metrics = len(_FIN_METRICS.findall(text))
    score += n_title * 5.0
    score += min(n_metrics, 8) * 2.0
    # bonus when title and metric terms appear together
    if n_title > 0 and n_metrics > 0:
        score += 3.0
    score += min(len(_TABLE_LINE.findall(text)), 10) * 0.3
    score += min(len(_NUMBER_TOKEN.findall(text)), 20) * 0.2
    return score


def _score_business_performance(text: str) -> float:
    score = 0.0
    n_title = len(_BIZ_TITLE.findall(text))
    n_metrics = len(_BIZ_METRICS.findall(text))
    score += n_title * 4.0
    score += min(n_metrics, 8) * 3.0
    # bonus when title and metric terms co-occur
    if n_title > 0 and n_metrics > 0:
        score += 3.0
    score += min(len(_NUMBER_TOKEN.findall(text)), 20) * 0.2
    return score


def _score_shareholder_return(text: str) -> float:
    score = 0.0
    n_metrics = len(_SH_METRICS.findall(text))
    score += min(n_metrics, 6) * 4.0
    # bonus when shareholder terms appear alongside numeric values
    if n_metrics > 0 and _NUMBER_TOKEN.search(text):
        score += 2.0
    return score


def score_page(text: str) -> dict[str, float]:
    return {
        "fte": _score_fte(text),
        "sustainability": _score_sustainability(text),
        "financial_highlight": _score_financial_highlight(text),
        "business_performance": _score_business_performance(text),
        "shareholder_return": _score_shareholder_return(text),
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
    max_pages_per_category: int = 50,
) -> tuple[PageScanResult, dict[str, str]]:
    """Score pages and return top candidates per category.

    Returns (PageScanResult, page_range_strings).
    page_range_strings maps category → range string suitable for LlamaExtract.
    """
    all_page_nos: set[int] = {p["page"] for p in pages}
    scored: dict[str, list[PageScore]] = {c: [] for c in CATEGORIES}

    for record in pages:
        text = record.get("enhanced_text") or record.get("text", "")
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
