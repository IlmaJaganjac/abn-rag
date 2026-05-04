from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from backend.app._openai import openai_client
from backend.app.config import settings

log = logging.getLogger(__name__)

_CATEGORY_RULES: dict[str, str] = {
    "fte": """
A valid fte datapoint must describe company workforce size or workforce breakdowns.
Valid: total employees, FTE/full-time equivalent, headcount, payroll employees, average employees,
year-end employees, internal employees, permanent/temporary/part-time/full-time employees,
contractors/external workers if clearly workforce-related, portfolio company employees,
employees by gender, country, region, age, contract type, or employment type.
Invalid: board/executive diversity percentages, employee engagement survey scores, training hours,
safety incident rates, customer counts, suppliers, financial values, sustainability goals,
general HR text without a numeric workforce datapoint.
""",
    "sustainability": """
A valid sustainability_goal must be a forward-looking goal, target, ambition, aim, or commitment
about environmental, social, or governance impact.
Valid: net-zero emissions by 2050, halve Scope 1 and 2 emissions by 2030, maintain methane emissions
intensity below 0.2%, achieve near-zero methane emissions intensity by 2030, eliminate routine flaring
by a target year, reduce NCI by a target amount/year, reduce customer emissions by a target amount/year,
reduce water consumption by a target amount/year, increase recycled/reusable/recyclable packaging by a
target amount/year, biodiversity/deforestation commitments if they state a clear commitment/aim/target.
Invalid: actual ESG performance values, progress/status/performance against a target (e.g. "70% achieved",
"9% reduction in 2025", "18% reduction by end of 2025"), reported emissions values, financial/FTE/
operational/production/revenue/shareholder data, general sustainability strategy text without a specific
goal/target/ambition/commitment.
If a candidate is progress against a target, mark invalid even if it mentions the target.
""",
    "esg": """
A valid esg datapoint must be an actual reported ESG performance value for the reporting year or period.
Valid: Scope 1/2/3 emissions actuals, GHG emissions, methane intensity actual, routine flaring actual,
renewable electricity percentage actual, energy consumption, water use, waste generated, recycling/reuse
actuals, safety metrics, supplier audit results, workforce/social ESG metrics if reported as performance values.
Invalid: forward-looking targets/goals/ambitions/commitments, financial KPIs, FTE/headcount totals unless
explicitly an ESG workforce metric, business production/operational metrics unless explicitly ESG,
general ESG narrative without a numeric reported value.
""",
    "financial_highlight": """
A valid financial_highlight must be a financial KPI/value from the annual report.
Valid: revenue/net sales/total income, operating profit/operating income/EBIT/EBITDA, net income/net profit/
profit for period, gross profit/gross margin, EPS, R&D expense, free cash flow/operating cash flow,
capex/capital expenditure, cash and cash equivalents, ROE/ROIC, CET1/capital ratio/liquidity coverage/NIM.
Invalid: workforce/FTE, sustainability goals, ESG actuals, operational KPIs like systems sold/LNG sales/
barrels per day/customers/suppliers, shareholder distributions like dividends or buybacks.
""",
    "business_performance": """
A valid business_performance datapoint must be an operational/business KPI rather than a financial statement line.
Valid: systems sold/units sold, installed base, order intake/bookings/backlog, customers/clients/suppliers,
customer satisfaction/NPS, production volume, LNG sales/barrels per day/refining throughput, loans/deposits/
mortgages, beer volume/hectoliters, stores/branches/locations, AUM/transaction volume, segment operational KPIs.
Invalid: FTE/headcount, financial statement lines like revenue/net income/gross margin/EPS, sustainability goals,
ESG actuals unless explicitly operational KPI requested, shareholder distributions.
""",
    "shareholder_return": """
A valid shareholder_return datapoint must describe distributions or capital returns to shareholders.
Valid: dividends paid, ordinary/special/final/interim/proposed dividend, dividend per share, payout ratio,
share buybacks/repurchases, treasury shares purchased, shares cancelled, total returned/distributed to
shareholders, capital return.
Invalid: generic shareholders' equity, share price, market capitalization, EPS unless directly tied to
payout/dividend, revenue/net income/cash flow, workforce, sustainability/ESG/business operational KPIs.
""",
}

_SYSTEM_PROMPT_TEMPLATE = """You are a financial data quality validator for annual report datapoints.

Category: {category}

Category validation rules:
{rules}

Deduplication rules:
- Identify duplicates within the candidate list. If multiple datapoints represent the same real-world
  datapoint/goal/KPI, keep only the best one.
- Prefer the record with: (1) the clearest exact quote, (2) the most complete fields, (3) the most
  specific metric name, (4) the clearest page/source, (5) the highest confidence if available.
- Mark duplicate records as is_valid=false and set duplicate_of_index to the kept record's index.
- Do NOT mark genuinely different breakdowns as duplicates (e.g. total employees vs employees by gender,
  Scope 1+2 target vs Scope 3 target, NCI 2030 target vs NCI 2050 target).

For each candidate datapoint (identified by index), return a ValidationItem with:
- index: the candidate index
- is_valid: true only if it belongs to this category and is not invalid or duplicate
- reason: brief explanation if invalid or duplicate
- confidence: optional float 0-1 for your confidence in the validity judgment
- corrected_type: optional correct category if misclassified
- duplicate_of_index: index of the kept record when marking a duplicate as invalid

Return a ValidationResult with all items.
"""


class ValidationItem(BaseModel):
    index: int
    is_valid: bool
    reason: str
    confidence: float | None = None
    corrected_type: str | None = None
    duplicate_of_index: int | None = None


class ValidationResult(BaseModel):
    items: list[ValidationItem]


def _datapoint_to_candidate(index: int, dp: Any) -> dict[str, Any]:
    fields = ["datapoint_type", "metric", "label", "value", "period", "basis",
              "scope", "target_year", "baseline", "page", "quote", "extractor"]
    candidate: dict[str, Any] = {"index": index}
    for f in fields:
        v = getattr(dp, f, None)
        if v is not None:
            candidate[f] = v
    return candidate


def validate_datapoints_openai(
    *,
    category: str,
    datapoints: list[Any],
    company: str | None,
    year: int | None,
) -> list[ValidationItem]:
    if not datapoints:
        return []

    import json

    rules = _CATEGORY_RULES.get(category, "Validate that each datapoint belongs to the requested category.")
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(category=category, rules=rules.strip())

    candidates = [_datapoint_to_candidate(i, dp) for i, dp in enumerate(datapoints)]
    user_msg = (
        f"Company: {company or ''}\n"
        f"Year: {year or ''}\n"
        f"Category: {category}\n\n"
        f"Candidate datapoints:\n{json.dumps(candidates, indent=2)}"
    )

    model = settings.openai_validate_model
    supports_temp_zero = "gpt-5" not in model

    client = openai_client()
    completion = client.beta.chat.completions.parse(
        model=model,
        **({"temperature": 0} if supports_temp_zero else {}),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format=ValidationResult,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return []
    return parsed.items
