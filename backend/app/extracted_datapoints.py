from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from backend.app.llama_extract_datapoints import AnnualReportDatapoints

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
