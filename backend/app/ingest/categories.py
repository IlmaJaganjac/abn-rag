from __future__ import annotations

import re

FTE_PATTERNS = [
    re.compile(
        r"\bFTEs?\b|\bfull[-\s]time equivalents?\b|"
        r"total\s+employees?|number\s+of\s+employees|headcount|workforce|"
        r"payroll\s+employees?|internal\s+employees?|external\s+employees?|"
        r"permanent\s+employees?|temporary\s+employees?|part[-\s]time\s+employees?|"
        r"employee\s+turnover|attrition\s+rate",
        re.IGNORECASE,
    ),
]

SUSTAINABILITY_PATTERNS = [
    re.compile(
        r"\b(target|goal|aim|ambition|commit(?:ment|ted)?|plan|intend|achieve|reduce|reduction)\b|"
        r"net[-\s]?zero|science\s+based\s+targets?|\bsbti\b|transition\s+plan",
        re.IGNORECASE,
    ),
    re.compile(
        r"greenhouse\s+gas|\bghg\b|co2e?|co₂e?|\bscope\s+[123]\b|emissions?|"
        r"climate|carbon|methane|flaring|renewable|energy|waste|recycl|circular|"
        r"water|biodiversity|supplier|diversity|inclusion|safety|human\s+rights",
        re.IGNORECASE,
    ),
    re.compile(r"\b20[3-9]\d\b", re.IGNORECASE),
]

ESG_PATTERNS = [
    re.compile(
        r"\b(ESG|environmental|social|governance)\b|"
        r"greenhouse\s+gas|\bghg\b|co2e?|co₂e?|\bscope\s+[123]\b|emissions?|"
        r"renewable|energy|water|waste|recycl|circular|biodiversity|supplier|"
        r"diversity|inclusion|safety|ethics",
        re.IGNORECASE,
    ),
]

FINANCIAL_HIGHLIGHT_PATTERNS = [
    re.compile(
        r"financial\s+highlights?|financial\s+performance|at\s+a\s+glance|"
        r"\brevenue\b|net\s+sales|total\s+income|net\s+income|net\s+profit|"
        r"operating\s+(?:income|profit)|\bebit(?:da)?\b|gross\s+margin|gross\s+profit|"
        r"earnings\s+per\s+share|\beps\b|free\s+cash\s+flow|operating\s+cash\s+flow|"
        r"cash\s+flow\s+from\s+operat|r&d|research\s+and\s+development|"
        r"return\s+on\s+equity|\broe\b|\bcet1\b|capital\s+ratio|"
        r"liquidity\s+coverage|net\s+interest\s+margin|\bnim\b",
        re.IGNORECASE,
    ),
]

BUSINESS_PERFORMANCE_PATTERNS = [
    re.compile(
        r"business\s+performance|operational\s+highlights?|segment\s+performance|"
        r"systems?\s+sold|lithography\s+systems?|installed\s+base|order\s+intake|"
        r"order\s+book|backlog|bookings|customers?|clients?|suppliers?|"
        r"customer\s+satisfaction|market\s+share|production\s+volume|deliveries|"
        r"\bloans?\b|\bdeposits?\b|\bmortgages?\b|\blng\b|barrels?\s+per\s+day|"
        r"refining\s+throughput|assets\s+under\s+management|\baum\b",
        re.IGNORECASE,
    ),
]

SHAREHOLDER_RETURN_PATTERNS = [
    re.compile(
        r"returned?\s+to\s+shareholders?|shareholder\s+(?:returns?|distributions?)|"
        r"capital\s+return|dividends?|dividend\s+per\s+share|payout\s+ratio|"
        r"share\s+buybacks?|share\s+repurchases?|repurchased|treasury\s+shares?|"
        r"shares?\s+cancelled",
        re.IGNORECASE,
    ),
]

CATEGORY_MAX_PAGES: dict[str, int] = {
    "sustainability": 30,
    "fte": 20,
    "esg": 20,
    "financial_highlight": 20,
    "business_performance": 20,
    "shareholder_return": 20,
}

CATEGORY_PATTERNS: dict[str, list[re.Pattern]] = {
    "fte": FTE_PATTERNS,
    "sustainability": SUSTAINABILITY_PATTERNS,
    "esg": ESG_PATTERNS,
    "financial_highlight": FINANCIAL_HIGHLIGHT_PATTERNS,
    "business_performance": BUSINESS_PERFORMANCE_PATTERNS,
    "shareholder_return": SHAREHOLDER_RETURN_PATTERNS,
}

DATAPOINT_CATEGORIES: tuple[str, ...] = (
    "fte",
    "sustainability",
    "esg",
    "financial_highlight",
    "business_performance",
    "shareholder_return",
)


def category_page_score(category: str, text: str) -> int:
    """Return a regex-based relevance score for one category on one page of text."""
    return sum(len(pattern.findall(text)) for pattern in CATEGORY_PATTERNS.get(category, []))
