from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from backend.app.llama_extract_datapoints import AnnualReportDatapoints, ExtractedESGDatapoint
from backend.app.schemas import Chunk

_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")
_SYMBOL_SPACE_RE = re.compile(r"\s*([€$£%><=,.])\s*")

# FTE priority helpers
_FTE_COMPANY_WIDE = re.compile(
    r"total\s+employees?|total\s+workforce|all\s+employees?|"
    r"total\s+number\s+of\s+payroll|total\s+number\s+of\s+employees",
    re.IGNORECASE,
)
_FTE_AVG_PAYROLL = re.compile(
    r"\baverage\b|\bpayroll\b|\btemporary\b",
    re.IGNORECASE,
)
_FTE_HEADCOUNT = re.compile(r"\bheadcount\b", re.IGNORECASE)
_FTE_SPECIFIC = re.compile(
    r"dedicated\s+fte|team\s+fte|program\s+fte|project\s+fte",
    re.IGNORECASE,
)


class NormalizedDatapoint(BaseModel):
    source: str
    company: str | None = None
    year: int | None = None
    datapoint_type: str
    metric: str
    value: str | None = None
    unit: str | None = None
    period: str | None = None
    page: int | None = None
    quote: str | None = None
    basis: str | None = None
    scope: str | None = None
    target_year: str | None = None
    extractor: str = "llamaextract"
    priority: int = 50
    confidence: float | None = None


class NormalizedDatapointSet(BaseModel):
    source: str
    company: str | None = None
    year: int | None = None
    datapoints: list[NormalizedDatapoint] = []


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _norm_text(s: str | None) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = _PUNCT_RE.sub(" ", s).lower()
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def _norm_value(s: str | None) -> str:
    if not s:
        return ""
    s = s.lower().strip()
    s = _SYMBOL_SPACE_RE.sub(r"\1", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def _fte_priority(metric: str, basis: str | None) -> int:
    combined = f"{metric} {basis or ''}"
    if _FTE_SPECIFIC.search(combined):
        return 30
    if _FTE_HEADCOUNT.search(combined):
        return 70
    if _FTE_AVG_PAYROLL.search(combined):
        return 85
    if _FTE_COMPANY_WIDE.search(combined):
        return 100
    # generic FTE mention
    return 60


def _sustainability_priority(goal: str, quote: str | None, target_year: str | None) -> int:
    if quote and target_year:
        return 95
    if target_year:
        return 80
    if quote:
        return 75
    return 70


def _kpi_priority(page: int | None, quote: str | None) -> int:
    if page is not None and quote:
        return 90
    if page is not None or quote:
        return 75
    return 60


def _esg_priority(quote: str | None, period: str | None) -> int:
    if quote and period:
        return 95
    if quote:
        return 85
    return 70


# ---------------------------------------------------------------------------
# Public normalization
# ---------------------------------------------------------------------------

def normalize_llamaextract_result(
    result: AnnualReportDatapoints,
    *,
    source: str,
    company: str | None,
    year: int | None,
) -> list[NormalizedDatapoint]:
    out: list[NormalizedDatapoint] = []

    for dp in result.fte_datapoints:
        priority = _fte_priority(dp.label, dp.basis)
        out.append(NormalizedDatapoint(
            source=source,
            company=company,
            year=year,
            datapoint_type="fte",
            metric=dp.label,
            value=dp.value,
            unit=dp.unit,
            period=dp.period,
            page=dp.page,
            quote=dp.quote,
            basis=dp.basis,
            priority=priority,
            confidence=dp.confidence,
        ))

    for sg in result.sustainability_goals:
        priority = _sustainability_priority(sg.goal, sg.quote, sg.target_year)
        out.append(NormalizedDatapoint(
            source=source,
            company=company,
            year=year,
            datapoint_type="sustainability_goal",
            metric=sg.metric or sg.goal[:80],
            value=sg.value_or_target,
            unit=None,
            period=None,
            page=sg.page,
            quote=sg.quote,
            scope=sg.scope,
            target_year=sg.target_year,
            priority=priority,
            confidence=sg.confidence,
        ))

    for esg in result.esg_datapoints:
        priority = _esg_priority(esg.quote, esg.period)
        out.append(NormalizedDatapoint(
            source=source,
            company=company,
            year=year,
            datapoint_type="esg_datapoint",
            metric=esg.metric,
            value=esg.value,
            unit=esg.unit,
            period=esg.period,
            scope=esg.scope,
            page=esg.page,
            quote=esg.quote,
            priority=priority,
            confidence=esg.confidence,
        ))

    for kpi in result.kpi_highlights:
        priority = _kpi_priority(kpi.page, kpi.quote)
        out.append(NormalizedDatapoint(
            source=source,
            company=company,
            year=year,
            datapoint_type="kpi_highlight",
            metric=kpi.metric,
            value=kpi.value,
            unit=kpi.unit,
            period=kpi.period,
            page=kpi.page,
            quote=kpi.quote,
            priority=priority,
            confidence=kpi.confidence,
        ))

    return out


def deduplicate_datapoints(
    datapoints: list[NormalizedDatapoint],
) -> list[NormalizedDatapoint]:
    seen: dict[tuple, NormalizedDatapoint] = {}
    for dp in datapoints:
        key = (
            dp.source,
            dp.page,
            dp.datapoint_type,
            _norm_text(dp.metric),
            _norm_value(dp.value),
        )
        existing = seen.get(key)
        if existing is None:
            seen[key] = dp
            continue
        # keep higher priority
        if dp.priority > existing.priority:
            seen[key] = dp
        elif dp.priority == existing.priority:
            # prefer higher confidence
            dp_conf = dp.confidence or 0.0
            ex_conf = existing.confidence or 0.0
            if dp_conf > ex_conf:
                seen[key] = dp
            elif dp_conf == ex_conf and dp.quote and not existing.quote:
                seen[key] = dp
    return list(seen.values())


_SECTION_PATH = {
    "fte": "Pre-extracted > FTE",
    "sustainability_goal": "Pre-extracted > Sustainability goals",
    "esg_datapoint": "Pre-extracted > ESG datapoints",
    "kpi_highlight": "Pre-extracted > KPI highlights",
}

_FTE_SYNONYMS = "employees workforce FTE full-time equivalents headcount payroll"
_SUST_SYNONYMS = "sustainability target goal climate GHG CO2 CO₂ emissions scope net zero"
_ESG_SYNONYMS = "ESG sustainability performance actual reported emissions GHG CO2 CO₂ scope energy waste recycling renewable water circularity"


def _build_text(dp: NormalizedDatapoint) -> str:
    lines = [f"Datapoint type: {dp.datapoint_type}", f"Metric: {dp.metric}"]
    if dp.datapoint_type == "fte":
        lines.append(f"Period: {dp.period or dp.year or ''}")
        lines.append(f"Value: {dp.value or ''}")
        if dp.unit:
            lines.append(f"Unit: {dp.unit}")
        if dp.basis:
            lines.append(f"Basis: {dp.basis}")
    elif dp.datapoint_type == "sustainability_goal":
        if dp.target_year:
            lines.append(f"Target year: {dp.target_year}")
        lines.append(f"Value/target: {dp.value or ''}")
        if dp.scope:
            lines.append(f"Scope: {dp.scope}")
    elif dp.datapoint_type == "esg_datapoint":
        lines.append(f"Period: {dp.period or dp.year or ''}")
        lines.append(f"Value: {dp.value or ''}")
        if dp.unit:
            lines.append(f"Unit: {dp.unit}")
        if dp.scope:
            lines.append(f"Scope: {dp.scope}")
    else:
        lines.append(f"Period: {dp.period or dp.year or ''}")
        lines.append(f"Value: {dp.value or ''}")
        if dp.unit:
            lines.append(f"Unit: {dp.unit}")
    if dp.quote:
        lines.append(f"Quote: {dp.quote}")
    lines.append(f"Extractor: llamaextract")
    lines.append(f"Priority: {dp.priority}")
    return "\n".join(lines)


def _build_embedding_text(dp: NormalizedDatapoint) -> str:
    parts = [
        dp.company or "",
        str(dp.year) if dp.year is not None else "",
        dp.datapoint_type,
        dp.metric,
        dp.value or "",
        dp.unit or "",
    ]
    if dp.datapoint_type == "fte":
        parts += [dp.basis or "", dp.period or "", _FTE_SYNONYMS]
    elif dp.datapoint_type == "sustainability_goal":
        parts += [dp.scope or "", dp.target_year or "", _SUST_SYNONYMS]
    elif dp.datapoint_type == "esg_datapoint":
        parts += [dp.scope or "", dp.period or "", _ESG_SYNONYMS]
    else:
        parts += [dp.period or ""]
    if dp.quote:
        parts.append(dp.quote)
    return "\n".join(p for p in parts if p)


def datapoints_to_chunks(datapoint_set: NormalizedDatapointSet) -> list[Chunk]:
    chunks: list[Chunk] = []
    for idx, dp in enumerate(datapoint_set.datapoints):
        text = _build_text(dp)
        embedding_text = _build_embedding_text(dp)
        section_path = _SECTION_PATH.get(dp.datapoint_type, "Pre-extracted")
        chunks.append(Chunk(
            id=f"{dp.source}:extracted:{idx}",
            source=dp.source,
            company=dp.company,
            year=dp.year,
            page=dp.page or 1,
            text=text,
            token_count=len(text.split()),
            parser="llamaextract",
            chunk_kind="extracted_datapoint",
            section_path=section_path,
            embedding_text=embedding_text,
        ))
    return chunks


def save_datapoint_set(
    datapoints: list[NormalizedDatapoint],
    *,
    source: str,
    company: str | None,
    year: int | None,
    out_path: Path,
) -> Path:
    ds = NormalizedDatapointSet(
        source=source,
        company=company,
        year=year,
        datapoints=datapoints,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    return out_path
