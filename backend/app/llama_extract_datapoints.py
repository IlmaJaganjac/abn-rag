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

_SYSTEM_PROMPT = """\
You extract structured datapoints from annual reports.
Only extract information explicitly present in the document.
Do not infer, calculate, convert, round, or normalize values.
Preserve exact values and qualifiers, including >, <, approximately, more than, %, €, kt, Mt, CO₂e, FTEs.

For FTE, extract workforce datapoints only. Distinguish FTE vs headcount, payroll vs temporary, average vs year-end if stated.

For sustainability goals (sustainability_goals): extract explicit targets, ambitions, commitments, or goals related to climate, GHG, emissions, energy, net-zero, scope 1/2/3, circularity, or sustainability. These are future-oriented and often contain words like "target", "aim", "ambition", "commitment", "by 2030", "by 2040", "we will", "we plan to".

For ESG datapoints (esg_datapoints): extract actual reported performance values for the reporting year. These are historical, measured values — not targets. Examples: actual GHG emissions (scope 1, 2, 3), renewable electricity percentage, energy consumption, water use, waste, recycling rate, supplier sustainability metrics, social/workforce ESG metrics. Extract the actual reported value, unit, period/year, scope if present, page, and quote.

Do not confuse targets (sustainability_goals) with actuals (esg_datapoints). A goal says what will happen; an ESG datapoint says what was measured.

Also extract visually prominent KPI highlights from dashboard, tile, overview, or "at a glance" pages.
Preserve exact value-label relationships. Do not swap labels between nearby tiles or columns.
Only extract values explicitly shown. Do not infer, calculate, convert, round, or normalize.
For every KPI, include metric, value, unit if explicit, period, page, and quote if available.
For every extracted item, include page and a short verbatim quote if available.
If not present, leave the list empty. Do not guess.\
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


class ExtractedKPIHighlight(BaseModel):
    metric: str
    value: str
    unit: str | None = None
    period: str | None = None
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


class AnnualReportDatapoints(BaseModel):
    company: str | None = None
    year: int | None = None
    fte_datapoints: list[ExtractedFTEDatapoint] = []
    sustainability_goals: list[ExtractedSustainabilityGoal] = []
    esg_datapoints: list[ExtractedESGDatapoint] = []
    kpi_highlights: list[ExtractedKPIHighlight] = []


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

    schema = _json_schema()
    config = ExtractConfig(
        cite_sources=True,
        confidence_scores=True,
        system_prompt=_SYSTEM_PROMPT,
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
        kpi_highlights=result.kpi_highlights,
    )
    return result
