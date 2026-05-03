from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from backend.app.config import settings

logger = logging.getLogger(__name__)

_BASE_INSTRUCTIONS = """\
Only extract information explicitly present in the document.
Do not infer, calculate, convert, round, or normalize values.
Preserve exact values and qualifiers, including >, <, approximately, more than, %, €, kt, Mt, CO₂e, FTEs.
For every extracted item, include page and a short verbatim quote if available.
Preserve exact value-label relationships. Do not swap labels between nearby tiles or columns.
If not present, leave the list empty. Do not guess.

For every item, classify:
- fact_kind: "actual" (reported/historical for the period), "target" (forward-looking goal/ambition), "progress" (status update against a target), or "forecast" (outlook/guidance).
- scope_type: "company_wide" if total/group-level; otherwise "segment", "geography", "product", "customer", or "project". For greenhouse-gas / emissions / climate datapoints, use "scope_1_2", "scope_3", or "supply_chain" only when the report explicitly mentions that scope. Use "unknown" only when the text gives no scope clue.

quote is REQUIRED: a verbatim evidence snippet from the page, table row, or nearby text that contains or directly supports both the metric label and the value. If no such evidence exists, omit the item.

Reject placeholder values ("N/A", "—", "xx%", "TBD") and status-only values ("On track", "TBC", "approved", "executed", "in progress", "ongoing").\
"""

_CATEGORY_PROMPTS: dict[str, str] = {
    "fte": f"""\
You extract workforce and headcount datapoints from annual reports. Only populate fte_datapoints. Leave all other lists empty.

Extract workforce/headcount datapoints: total employees (FTEs or headcount), average number of payroll employees, number of internal employees, permanent employees, temporary employees, part-time employees, full-time employees, non-guaranteed hours employees, external employees, contractors, year-end employee count, average employee count, employees by gender, employees by region or country, employee turnover or attrition rate. Distinguish FTE vs headcount, payroll vs temporary, average vs year-end if stated. Include label, value, unit (FTEs or headcount), basis (FTE/headcount/payroll/etc.), period (year or date), page, and verbatim quote.

Do not extract: dedicated FTEs for a specific project, program, or team; employee engagement survey scores; training hours; safety incident rates; board or executive diversity percentages unless they are explicitly reported as employee/headcount counts. Do not extract financial, sustainability, operational, or shareholder data.

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

REJECT: actual historical ESG performance values (reported emissions for the reporting year), business growth or production targets (LNG sales, liquids production, barrels per day, refining throughput, revenue, market share), financial / FTE / operational / shareholder data, general sustainability context or strategy text that states no specific target, and governance or risk text without an explicit target.

{_BASE_INSTRUCTIONS}""",

    "esg": f"""\
You extract actual ESG performance values from annual reports. Only populate esg_datapoints. Leave all other lists empty.

Extract actual reported performance values for the reporting year (not targets): GHG emissions (Scope 1, Scope 2, Scope 3, combined), renewable electricity percentage, total energy consumption, water use, waste generated, recycling or reuse rate (as ESG metric), supplier sustainability audit results, social/workforce ESG metrics. Include value, unit, period/year, scope if present, page, and verbatim quote.

Do not extract forward-looking targets or commitments. Do not extract business production volumes, financial KPIs, shareholder data, FTE/headcount totals, or general operational KPIs unless the value is explicitly reported as an ESG performance metric.

{_BASE_INSTRUCTIONS}""",

    "financial_highlight": f"""\
You extract financial KPI values from annual reports. Only populate financial_highlights. Leave all other lists empty.

Extract financial performance values from summaries, income statements, tables, or "at a glance" / highlights pages. Target metrics: total net sales / revenue / total income, gross profit, gross margin (%), operating income / operating profit / EBIT / EBITDA / adjusted EBITDA, net income / net profit / profit for the period, earnings per share (EPS) / diluted EPS, R&D spend / research and development expense, free cash flow, operating cash flow / cash flow from operating activities, cash and cash equivalents, capex / capital expenditure, return on equity (ROE), return on invested capital (ROIC), CET1 ratio, capital ratio, liquidity coverage ratio, net interest margin (NIM). Include metric, value, unit (€m / €bn / %), period (year), page, and a short verbatim quote.

Do not extract: workforce/FTE data, sustainability or ESG targets, operational KPIs (systems sold, suppliers, satisfaction, LNG sales, barrels per day, production volume), or shareholder return/distribution amounts such as dividends and buybacks.

{_BASE_INSTRUCTIONS}""",

    "business_performance": f"""\
You extract business and operational performance KPI values from annual reports. Only populate business_performance. Leave all other lists empty.

Extract operational/business performance values from summaries, segment tables, or "at a glance" pages. Target metrics: systems sold / lithography systems / net system sales in units, new systems sold, used systems sold, installed base / active systems, order intake / bookings, order book / backlog, number of customers or clients, customer satisfaction score / NPS, number of suppliers, reuse rate (operational/circularity KPI), market share, production volume, delivery volume, loans / deposits / mortgages (banking), LNG sales / barrels per day / refining throughput (energy), beer volume / hectoliters (consumer goods), number of stores / branches / locations (retail), assets under management (AUM), transaction volume. Include metric, value, unit, period, page, and a short verbatim quote.

Do not extract: FTE/headcount, employee demographics, financial statement lines (revenue/net sales as money, net income, gross margin), sustainability or ESG targets, actual ESG performance values, or shareholder distributions.

{_BASE_INSTRUCTIONS}""",

    "shareholder_return": f"""\
You extract shareholder return and capital distribution values from annual reports. Only populate shareholder_returns. Leave all other lists empty.

Extract: total returned to shareholders, total shareholder distributions, capital return, dividends paid, ordinary dividend, special dividend, final dividend, interim dividend, dividend per share, proposed dividend, payout ratio, share buybacks, share repurchases, treasury shares purchased, shares cancelled, number of shares repurchased, total cash returned. Include metric, value, unit (€ per share / €bn / €m / %), period (year), page, and a short verbatim quote.

Do not extract: financial performance KPIs (net income, gross margin, revenue, cash flow), workforce data, sustainability targets, business operational KPIs, or general share price/market capitalization values unless they are explicitly part of dividends, buybacks, capital returns, or shareholder distributions.

{_BASE_INSTRUCTIONS}""",
}

# Fallback for unknown categories
_DEFAULT_PROMPT = f"""\
You extract structured datapoints from annual reports.
{_BASE_INSTRUCTIONS}
"""

_POLL_INTERVAL = 5
_MAX_WAIT = 600


class ExtractedFTEDatapoint(BaseModel):
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
    company: str | None = None
    year: int | None = None
    fte_datapoints: list[ExtractedFTEDatapoint] = []
    sustainability_goals: list[ExtractedSustainabilityGoal] = []
    esg_datapoints: list[ExtractedESGDatapoint] = []
    financial_highlights: list[ExtractedFinancialHighlight] = []
    business_performance: list[ExtractedBusinessPerformance] = []
    shareholder_returns: list[ExtractedShareholderReturn] = []


def _json_schema() -> dict[str, Any]:
    return AnnualReportDatapoints.model_json_schema()


def category_prompt(category: str | None) -> str:
    if not category:
        return _DEFAULT_PROMPT
    return _CATEGORY_PROMPTS.get(category, _DEFAULT_PROMPT)


def _poll_until_done(client_extract: Any, job_id: str) -> str:
    from llama_cloud import ExtractJobStatus

    deadline = time.monotonic() + _MAX_WAIT
    while time.monotonic() < deadline:
        job = client_extract.get_job(job_id)
        status = job.status
        if status == ExtractJobStatus.SUCCESS:
            return "success"
        if status in (ExtractJobStatus.ERROR, ExtractJobStatus.CANCELLED):
            error = getattr(job, "error", None)
            raise RuntimeError(f"LlamaExtract job {job_id} ended with status {status}: {error}")
        time.sleep(_POLL_INTERVAL)
    raise TimeoutError(f"LlamaExtract job {job_id} did not complete within {_MAX_WAIT}s")


def extract_annual_report_datapoints(
    pdf_path: Path,
    *,
    company: str | None,
    year: int | None,
    page_range: str | None = None,
    category: str | None = None,
) -> AnnualReportDatapoints:
    api_key = settings.llama_cloud_api_key.get_secret_value()
    if not api_key:
        raise RuntimeError(
            "LLAMA_CLOUD_API_KEY is not set. Add it to .env."
        )

    try:
        from llama_cloud.client import LlamaCloud
        from llama_cloud import ExtractConfig, FileData
        from llama_cloud.types.extract_mode import ExtractMode
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"llama-cloud SDK not installed. Run `pip install llama-cloud`. Missing: {exc}"
        ) from exc

    pdf_bytes = pdf_path.read_bytes()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")

    client = LlamaCloud(token=api_key)
    le = client.llama_extract

    system_prompt = category_prompt(category)
    schema = _json_schema()
    config = ExtractConfig(
        cite_sources=True,
        confidence_scores=True,
        extraction_mode=ExtractMode.FAST,
        system_prompt=system_prompt,
        page_range=page_range,
        use_reasoning=True,
    )

    logger.info("submitting LlamaExtract job for %s", pdf_path.name)
    job = le.extract_stateless(
        config=config,
        data_schema=schema,
        file=FileData(data=pdf_b64, mime_type="application/pdf"),
    )
    job_id = job.id
    logger.info("LlamaExtract job submitted: %s", job_id)

    _poll_until_done(le, job_id)
    logger.info("LlamaExtract job %s complete", job_id)

    resultset = le.get_job_result(job_id)
    raw: Any = resultset.data

    if raw is None:
        logger.warning("LlamaExtract returned null data for job %s", job_id)
        return AnnualReportDatapoints(company=company, year=year)

    if isinstance(raw, list):
        raw = raw[0] if raw else {}

    try:
        result = AnnualReportDatapoints.model_validate(raw)
    except Exception as exc:
        logger.error("failed to parse LlamaExtract result: %s\nraw=%s", exc, json.dumps(raw, indent=2)[:500])
        return AnnualReportDatapoints(company=company, year=year)

    result = AnnualReportDatapoints(
        company=company or result.company,
        year=year or result.year,
        fte_datapoints=result.fte_datapoints,
        sustainability_goals=result.sustainability_goals,
        esg_datapoints=result.esg_datapoints,
        financial_highlights=result.financial_highlights,
        business_performance=result.business_performance,
        shareholder_returns=result.shareholder_returns,
    )
    return result
