from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from backend.app.config import settings

logger = logging.getLogger(__name__)

_BASE_INSTRUCTIONS = """\
Only extract information explicitly present in the document.
Do not infer, calculate, convert, round, or normalize values.
Preserve exact values and qualifiers, including >, <, approximately, more than, %, €, kt, Mt, CO₂e, FTEs.
For every extracted item, include page and a short verbatim quote if available.
Preserve exact value-label relationships. Do not swap labels between nearby tiles or columns.
If not present, leave the list empty. Do not guess.\
"""

_CATEGORY_PROMPTS: dict[str, str] = {
    "fte": f"""\
You extract workforce and headcount datapoints from annual reports. Only populate fte_datapoints. Leave all other lists empty.

Extract workforce/headcount datapoints: total employees (FTEs or headcount), average number of payroll employees, number of internal employees, permanent employees, temporary employees, part-time employees, full-time employees, non-guaranteed hours employees, external employees, contractors, year-end employee count, average employee count, employees by gender, employees by region or country, employee turnover or attrition rate. Distinguish FTE vs headcount, payroll vs temporary, average vs year-end if stated. Include label, value, unit (FTEs or headcount), basis (FTE/headcount/payroll/etc.), period (year or date), page, and verbatim quote.

Do not extract: dedicated FTEs for a specific project, program, or team. Do not extract financial, sustainability, operational, or shareholder data.

{_BASE_INSTRUCTIONS}""",

    "sustainability": f"""\
You extract sustainability goals, targets, ambitions, and commitments from annual reports. Only populate sustainability_goals. Leave all other lists empty.

Extract explicit forward-looking targets, ambitions, commitments, or goals related to: climate change mitigation, greenhouse gas (GHG) emission reductions, net zero, gross Scope 1 / Scope 2 / Scope 3 emissions reduction, carbon intensity, renewable electricity or energy, energy efficiency, circularity, waste, recycling, water, biodiversity, supplier sustainability. Look for language such as: "target", "goal", "ambition", "commitment", "aim", "we will", "we plan to", "reduce by [X%] by [year]", "achieve net zero by [year]", "transition plan", "SBTi", "Science Based Targets", "1.5°C". Include target_year, scope (Scope 1/2/3 or value chain), value_or_target (the reduction percentage or absolute target), and a verbatim quote.

Do not extract: actual historical ESG performance values (e.g., reported GHG emissions for last year). Do not extract FTE, financial, business performance, or shareholder data.

{_BASE_INSTRUCTIONS}""",

    "esg": f"""\
You extract actual ESG performance values from annual reports. Only populate esg_datapoints. Leave all other lists empty.

Extract actual reported performance values for the reporting year (not targets): GHG emissions (Scope 1, Scope 2, Scope 3, combined), renewable electricity percentage, total energy consumption, water use, waste generated, recycling or reuse rate (as ESG metric), supplier sustainability audit results, social/workforce ESG metrics. Include value, unit, period/year, scope if present, page, and verbatim quote.

Do not extract forward-looking targets or commitments. Do not extract FTE headcount, financial KPIs, business performance, or shareholder data.

{_BASE_INSTRUCTIONS}""",

    "financial_highlight": f"""\
You extract financial KPI values from annual reports. Only populate financial_highlights. Leave all other lists empty.

Extract financial performance values from summaries, income statements, tables, or "at a glance" / highlights pages. Target metrics: total net sales / revenue / total income, gross profit, gross margin (%), operating income / operating profit / EBIT / EBITDA / adjusted EBITDA, net income / net profit / profit for the period, earnings per share (EPS) / diluted EPS, R&D spend / research and development expense, free cash flow, operating cash flow / cash flow from operating activities, cash and cash equivalents, capex / capital expenditure, return on equity (ROE), return on invested capital (ROIC), CET1 ratio, capital ratio, liquidity coverage ratio, net interest margin (NIM). Include metric, value, unit (€m / €bn / %), period (year), page, and a short verbatim quote.

Do not extract: workforce/FTE data, sustainability targets, operational KPIs (systems sold, suppliers, satisfaction), or shareholder return amounts.

{_BASE_INSTRUCTIONS}""",

    "business_performance": f"""\
You extract business and operational performance KPI values from annual reports. Only populate business_performance. Leave all other lists empty.

Extract operational/business performance values from summaries, segment tables, or "at a glance" pages. Target metrics: systems sold / lithography systems / net system sales in units, new systems sold, used systems sold, installed base / active systems, order intake / bookings, order book / backlog, number of customers or clients, customer satisfaction score / NPS, number of suppliers, reuse rate (operational/circularity KPI), market share, production volume, delivery volume, loans / deposits / mortgages (banking), LNG sales / barrels per day / refining throughput (energy), beer volume / hectoliters (consumer goods), number of stores / branches / locations (retail), assets under management (AUM), transaction volume. Include metric, value, unit, period, page, and a short verbatim quote.

Do not extract: FTE/headcount, financial statement lines (net sales, net income, gross margin), sustainability targets, or shareholder distributions.

{_BASE_INSTRUCTIONS}""",

    "shareholder_return": f"""\
You extract shareholder return and capital distribution values from annual reports. Only populate shareholder_returns. Leave all other lists empty.

Extract: total returned to shareholders, total shareholder distributions, capital return, dividends paid, ordinary dividend, special dividend, final dividend, interim dividend, dividend per share, proposed dividend, payout ratio, share buybacks, share repurchases, treasury shares purchased, shares cancelled, number of shares repurchased, total cash returned. Include metric, value, unit (€ per share / €bn / €m / %), period (year), page, and a short verbatim quote.

Do not extract: financial performance KPIs (net income, gross margin), workforce data, sustainability targets, or business operational KPIs.

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
    unit: str | None = None
    basis: str | None = None
    period: str | None = None
    page: int | None = None
    quote: str | None = None
    confidence: float | None = None


class ExtractedSustainabilityGoal(BaseModel):
    goal: str
    target_year: str | None = None
    metric: str | None = None
    baseline: str | None = None
    value_or_target: str | None = None
    scope: str | None = None
    page: int | None = None
    quote: str | None = None
    confidence: float | None = None


class ExtractedESGDatapoint(BaseModel):
    metric: str
    value: str
    unit: str | None = None
    period: str | None = None
    scope: str | None = None
    page: int | None = None
    quote: str | None = None
    confidence: float | None = None


class ExtractedFinancialHighlight(BaseModel):
    metric: str
    value: str
    unit: str | None = None
    period: str | None = None
    page: int | None = None
    quote: str | None = None
    confidence: float | None = None
    basis: str | None = None


class ExtractedBusinessPerformance(BaseModel):
    metric: str
    value: str
    unit: str | None = None
    period: str | None = None
    page: int | None = None
    quote: str | None = None
    confidence: float | None = None
    basis: str | None = None


class ExtractedShareholderReturn(BaseModel):
    metric: str
    value: str
    unit: str | None = None
    period: str | None = None
    page: int | None = None
    quote: str | None = None
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
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"llama-cloud SDK not installed. Run `pip install llama-cloud`. Missing: {exc}"
        ) from exc

    pdf_bytes = pdf_path.read_bytes()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")

    client = LlamaCloud(token=api_key)
    le = client.llama_extract

    system_prompt = _CATEGORY_PROMPTS.get(category, _DEFAULT_PROMPT) if category else _DEFAULT_PROMPT
    schema = _json_schema()
    config = ExtractConfig(
        cite_sources=True,
        confidence_scores=True,
        system_prompt=system_prompt,
        page_range=page_range,
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
