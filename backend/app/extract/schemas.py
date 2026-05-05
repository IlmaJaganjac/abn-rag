from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

_BASE_INSTRUCTIONS = """\
Only extract information explicitly present in the document.
Do not infer, calculate, convert, round, or normalize values.
Preserve exact values and qualifiers, including >, <, approximately, more than, %, €, kt, Mt, CO₂e, FTEs.
Preserve exact value-label relationships. Do not swap labels between nearby tiles or columns.
If not present, leave the list empty. Do not guess.

For every item, classify:
- fact_kind: "actual" (reported/historical for the period), "target" (forward-looking goal/ambition), "progress" (status update against a target), or "forecast" (outlook/guidance).
- scope_type: "company_wide" if total/group-level; otherwise "segment", "geography", "product", "customer", or "project". For greenhouse-gas / emissions / climate datapoints, use "scope_1_2", "scope_3", or "supply_chain" only when the report explicitly mentions that scope. Use "unknown" only when the text gives no scope clue.

For every extracted item, quote is REQUIRED. The quote must be a verbatim evidence snippet from the page, table row, or nearby text that contains or directly supports both the metric label and the value. If no such quote exists, omit the item.

Keep value and unit separate. Put the numeric amount or exact reported value in value, and the measurement unit/currency in unit. Do not duplicate the unit in both value and unit. Example: value="1,073", unit="ktCO₂e", not value="1,073 ktCO₂e", unit="ktCO₂e".

Reject placeholder values ("N/A", "—", "xx%", "TBD") and status-only values ("On track", "TBC", "approved", "executed", "in progress", "ongoing").\
"""

_CATEGORY_PROMPTS: dict[str, str] = {
    "fte": f"""\
You extract workforce SIZE/COUNT datapoints from annual reports. Only populate fte_datapoints. Leave all other lists empty.

Extract ONLY actual reported workforce size/count datapoints for the reporting period. Valid metrics are limited to:
- Total employees, total workforce, FTE / full-time equivalents, headcount
- Payroll employees, average number of employees, year-end employees
- Internal employees, external employees, contractors / external workers
- Permanent / temporary / full-time / part-time / non-guaranteed hours employees — only when the value is a count, headcount, or FTE
- Employee breakdowns by gender, country, region, age, contract type, or employment type — only when the reported value is a count, headcount, or FTE

The unit must be a count-style unit: FTEs, headcount, employees, persons, or a plain integer count. The value must be a count.

Do NOT extract any of the following as fte:
- Percentages, rates, ratios of any kind (gender diversity %, turnover rate %, pay gap %, disability %, engagement participation %, training-hours rate, safety rates, etc.)
- Remuneration / compensation amounts, sign-on / severance / bonus / salary / wages / remuneration table values
- Employee engagement survey scores, engagement index, engagement drivers, employee engagement metrics, or participation rates
- Contract-type rows if values are percentages or the table does not clearly state headcount/FTE/count
- Executive Board, Supervisory Board, CLA+, Identified Staff, remuneration/governance table rows
- Training hours or safety incident rates
- Workforce reduction plans, hiring plans, future FTE targets, dedicated FTEs for a specific project, program, or team
- Board, supervisory-board, or executive diversity percentages
- Financial, sustainability/ESG, operational, customer, or shareholder data

If no exact quote supports both the metric label and the value, omit the item. fact_kind must be "actual".

{_BASE_INSTRUCTIONS}""",

    "sustainability": f"""\
You extract sustainability goals, targets, ambitions, and commitments from annual reports. Only populate sustainability_goals. Leave all other lists empty.

EXTRACT every distinct forward-looking sustainability goal, target, ambition, or commitment. Cover all of the following categories when present:
- Climate/GHG: net-zero 2050, Scope 1 emissions reduction, Scope 2 emissions reduction, Scope 3 (upstream / downstream / customer) emissions reduction, net carbon intensity (NCI), carbon intensity, absolute emissions halving
- Methane: methane intensity target, near-zero methane emissions
- Routine flaring: eliminate routine flaring target year
- Energy: renewable electricity target, energy efficiency target
- Circular economy / waste / water / biodiversity: recycling targets, circularity targets, water use targets, biodiversity commitments
- Supplier: supplier sustainability audits, supplier GHG commitments
- Social: gender diversity / inclusion targets, safety targets (TRIR, fatalities), human rights commitments
- Governance / ethics: ethics policies, anti-corruption commitments

If one sentence, paragraph, bullet, or table row contains multiple distinct goals, targets, ambitions, or commitments, ALWAYS split them into separate sustainability_goals items. Do not combine multiple goals in one item.

Examples:
- "Maintain methane emissions intensity below 0.2% and achieve near-zero methane emissions intensity by 2030" must become two items:
  1. metric: "Methane emissions intensity maintenance target"; value_or_target: "below 0.2%"
  2. metric: "Near-zero methane emissions intensity target"; value_or_target: "near-zero methane emissions intensity"; target_year: "2030"
- "Reduce NCI by 15-20% by 2030 and become net-zero by 2050" must become two items:
  1. metric: "Net carbon intensity (NCI) reduction target"; value_or_target: "15-20% reduction"; target_year: "2030"
  2. metric: "Net-zero emissions target"; value_or_target: "net-zero"; target_year: "2050"

For each goal use a SPECIFIC metric name (not generic names like "% reduction"). Examples: "Scope 1 and 2 emissions reduction target", "Methane intensity target", "Net carbon intensity (NCI) 2030 target", "Routine flaring elimination target", "Net-zero Scope 1, 2 and 3 by 2050".

Always populate:
- goal: short description of the goal/target/ambition/commitment
- metric: specific metric name as described above
- value_or_target: the reduction %, absolute level, or qualitative commitment
- target_year: if stated
- baseline: if stated (e.g., "compared to 2016")
- scope: Scope 1 / Scope 2 / Scope 3 / value chain / upstream / downstream if stated
- page: exact page number
- quote: exact verbatim sentence(s) from the report

REJECT: actual historical ESG performance values (reported emissions for the reporting year), progress/status-only updates without a target value, business growth or production targets (LNG sales, liquids production, barrels per day, refining throughput, revenue, market share), financial / FTE / operational / shareholder data, general sustainability context or strategy text that states no specific measurable target or commitment, and governance, risk or policy text without an explicit sustainability target.

{_BASE_INSTRUCTIONS}""",

    "esg": f"""\
You extract actual ESG performance values from annual reports. Only populate esg_datapoints. Leave all other lists empty.

Extract actual reported performance values for the reporting year (not targets): GHG emissions (Scope 1, Scope 2, Scope 3, combined), renewable electricity percentage, total energy consumption, water use, waste generated, recycling or reuse rate (as ESG metric), supplier sustainability audit results.

ESG MAY include social/workforce percentages and rates when reported as ESG / S1 / social performance, such as: gender diversity %, pay gap %, disability %, turnover rate %, employee engagement participation %, safety incident rates. Treat such values as ESG performance — do NOT classify them as fte.

ESG must NOT include:
- Forward-looking targets, ambitions, commitments, target years, or progress against a target — those belong in sustainability_goals.
- Plain FTE / headcount / employee count totals — those belong in fte. Only include workforce counts here when they appear in an explicit ESG / S1 workforce performance table.
- Business production volumes, financial KPIs, shareholder data, or general operational KPIs unless the value is explicitly reported as an ESG performance metric.

Include value, unit, period/year, scope if present, page, and verbatim quote. fact_kind must be "actual".

{_BASE_INSTRUCTIONS}""",

    "financial_highlight": f"""\
You extract financial KPI values from annual reports. Only populate financial_highlights. Leave all other lists empty.

Extract financial performance values from summaries, income statements, banking KPI tables, or "at a glance" / highlights pages. Target metrics: total net sales / revenue / total income, net interest income, fee and commission income, operating result / operating income / operating profit / EBIT / EBITDA / adjusted EBITDA, gross profit, gross margin (%), net income / net profit / profit for the period, earnings per share (EPS) / diluted EPS, R&D spend / research and development expense, free cash flow, operating cash flow / cash flow from operating activities, cash and cash equivalents, capex / capital expenditure, return on equity (ROE), return on invested capital (ROIC), CET1 ratio, capital ratio, liquidity coverage ratio, net interest margin (NIM), cost/income ratio, impairment charges. Include metric, value, unit (€m / €bn / %), period (year), page, and a short verbatim quote.

Do not extract: workforce/FTE data, sustainability or ESG targets, actual ESG performance, operational non-financial KPIs (systems sold, suppliers, satisfaction, LNG sales, barrels per day, production volume), customer counts, or shareholder return/distribution amounts such as dividends and buybacks.

{_BASE_INSTRUCTIONS}""",

    "business_performance": f"""\
You extract business and operational performance KPI values from annual reports. Only populate business_performance. Leave all other lists empty.

Extract operational/business performance values from summaries, segment tables, or "at a glance" pages. Target metrics: systems sold / lithography systems / net system sales in units, new systems sold, used systems sold, installed base / active systems, order intake / bookings, order book / backlog, number of customers or clients, customer satisfaction score / NPS, number of suppliers, reuse rate (operational/circularity KPI), market share, production volume, delivery volume, client assets / assets under management (AUM), transaction volume, number of branches or locations. For banks, include operational balance-sheet/customer metrics such as customer loans, deposits, mortgages, client assets, and number of clients only when reported as business scale/performance KPIs, not as income-statement profit metrics. Include metric, value, unit, period, page, and a short verbatim quote.

Do not extract: FTE/headcount, employee demographics, financial statement lines (revenue/net sales as money, net interest income, fee income, net income, gross margin, operating profit), capital/liquidity ratios, sustainability or ESG targets, actual ESG performance values, or shareholder distributions.

{_BASE_INSTRUCTIONS}""",

    "shareholder_return": f"""\
You extract shareholder return and capital distribution values from annual reports. Only populate shareholder_returns. Leave all other lists empty.

Extract: total returned to shareholders, total shareholder distributions, capital return, dividends paid, ordinary dividend, special dividend, final dividend, interim dividend, dividend per share, proposed dividend, payout ratio, share buybacks, share repurchases, treasury shares purchased, shares cancelled, number of shares repurchased, total cash returned. Include metric, value, unit (€ per share / €bn / €m / %), period (year), page, and a short verbatim quote.

Do not extract: financial performance KPIs (net income, gross margin, revenue, cash flow, EPS unless explicitly tied to dividend payout), workforce data, sustainability targets, business operational KPIs, or general share price/market capitalization values unless they are explicitly part of dividends, buybacks, capital returns, or shareholder distributions.

{_BASE_INSTRUCTIONS}""",
}

# Fallback for unknown categories
_DEFAULT_PROMPT = f"""\
You extract structured datapoints from annual reports.
{_BASE_INSTRUCTIONS}
"""

class ExtractedFTEDatapoint(BaseModel):
    """Structured FTE or workforce datapoint extracted from a report."""
    label: str
    value: str
    quote: str
    fact_kind: Literal["actual", "target", "progress", "forecast"]
    scope_type: Literal["company_wide", "segment", "geography", "product", "customer", "project", "scope_1_2", "scope_3", "supply_chain", "unknown"]
    unit: str | None = None
    basis: str | None = None
    period: str | None = None
    page: int | None = None
    confidence: float | None = None


class ExtractedSustainabilityGoal(BaseModel):
    """Structured sustainability goal, target, or commitment extracted from a report."""
    goal: str
    quote: str
    fact_kind: Literal["actual", "target", "progress", "forecast"]
    scope_type: Literal["company_wide", "segment", "geography", "product", "customer", "project", "scope_1_2", "scope_3", "supply_chain", "unknown"]
    target_year: str | None = None
    metric: str | None = None
    baseline: str | None = None
    value_or_target: str | None = None
    scope: str | None = None
    page: int | None = None
    confidence: float | None = None


class ExtractedESGDatapoint(BaseModel):
    """Structured ESG performance datapoint extracted from a report."""
    metric: str
    value: str
    quote: str
    fact_kind: Literal["actual", "target", "progress", "forecast"]
    scope_type: Literal["company_wide", "segment", "geography", "product", "customer", "project", "scope_1_2", "scope_3", "supply_chain", "unknown"]
    unit: str | None = None
    period: str | None = None
    scope: str | None = None
    page: int | None = None
    confidence: float | None = None


class ExtractedFinancialHighlight(BaseModel):
    """Structured financial KPI extracted from a report."""
    metric: str
    value: str
    quote: str
    fact_kind: Literal["actual", "target", "progress", "forecast"]
    scope_type: Literal["company_wide", "segment", "geography", "product", "customer", "project", "scope_1_2", "scope_3", "supply_chain", "unknown"]
    unit: str | None = None
    period: str | None = None
    page: int | None = None
    confidence: float | None = None
    basis: str | None = None


class ExtractedBusinessPerformance(BaseModel):
    """Structured operational or business KPI extracted from a report."""
    metric: str
    value: str
    quote: str
    fact_kind: Literal["actual", "target", "progress", "forecast"]
    scope_type: Literal["company_wide", "segment", "geography", "product", "customer", "project", "scope_1_2", "scope_3", "supply_chain", "unknown"]
    unit: str | None = None
    period: str | None = None
    page: int | None = None
    confidence: float | None = None
    basis: str | None = None


class ExtractedShareholderReturn(BaseModel):
    """Structured shareholder return or capital distribution datapoint."""
    metric: str
    value: str
    quote: str
    fact_kind: Literal["actual", "target", "progress", "forecast"]
    scope_type: Literal["company_wide", "segment", "geography", "product", "customer", "project", "scope_1_2", "scope_3", "supply_chain", "unknown"]
    unit: str | None = None
    period: str | None = None
    page: int | None = None
    confidence: float | None = None
    basis: str | None = None


class AnnualReportDatapoints(BaseModel):
    """Top-level grouped extraction result for one annual report."""
    company: str | None = None
    year: int | None = None
    fte_datapoints: list[ExtractedFTEDatapoint] = []
    sustainability_goals: list[ExtractedSustainabilityGoal] = []
    esg_datapoints: list[ExtractedESGDatapoint] = []
    financial_highlights: list[ExtractedFinancialHighlight] = []
    business_performance: list[ExtractedBusinessPerformance] = []
    shareholder_returns: list[ExtractedShareholderReturn] = []


def category_prompt(category: str | None) -> str:
    """Return the extraction prompt for one category or the generic fallback prompt."""
    if not category:
        return _DEFAULT_PROMPT
    return _CATEGORY_PROMPTS.get(category, _DEFAULT_PROMPT)
