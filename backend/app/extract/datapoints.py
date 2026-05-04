from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from pydantic import BaseModel

from backend.app.extract.schemas import AnnualReportDatapoints
from backend.app.extract.signals import (
    FTE_AVG_PAYROLL,
    FTE_COMPANY_WIDE,
    FTE_HEADCOUNT,
    FTE_SPECIFIC,
    PLACEHOLDER_VALUE,
    PUNCT_RE,
    SPACE_RE,
    STATUS_ONLY_VALUE,
    SYMBOL_SPACE_RE,
)


class NormalizedDatapoint(BaseModel):
    """Canonical datapoint shape used after extraction cleanup."""
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
    fact_kind: str | None = None
    scope_type: str | None = None
    quality: str | None = None
    validation_status: str | None = None
    canonical_metric: str | None = None


class NormalizedDatapointSet(BaseModel):
    """Container for one source file and the normalized datapoints extracted from it."""
    source: str
    company: str | None = None
    year: int | None = None
    datapoints: list[NormalizedDatapoint] = []


def _norm_text(s: str | None) -> str:
    """Normalize free text for deduplication and heuristic matching."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = PUNCT_RE.sub(" ", s).lower()
    s = SPACE_RE.sub(" ", s).strip()
    return s


def _norm_value(s: str | None) -> str:
    """Normalize a datapoint value string for value-level comparisons."""
    if not s:
        return ""
    s = s.lower().strip()
    s = SYMBOL_SPACE_RE.sub(r"\1", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def _fte_priority(metric: str, basis: str | None) -> int:
    """Score FTE datapoints so broader company-wide metrics outrank weaker variants."""
    combined = f"{metric} {basis or ''}"
    if FTE_SPECIFIC.search(combined):
        return 30
    if FTE_HEADCOUNT.search(combined):
        return 70
    if FTE_AVG_PAYROLL.search(combined):
        return 85
    if FTE_COMPANY_WIDE.search(combined):
        return 100
    return 60


def _sustainability_priority(goal: str, quote: str | None, target_year: str | None) -> int:
    """Score sustainability goals so more explicit targets rank above vague ones."""
    if quote and target_year:
        return 95
    if target_year:
        return 80
    if quote:
        return 75
    return 70


def _highlight_priority(page: int | None, quote: str | None) -> int:
    """Score generic highlight datapoints based on evidence completeness."""
    if page is not None and quote:
        return 90
    if page is not None or quote:
        return 75
    return 60


def _esg_priority(quote: str | None, period: str | None) -> int:
    """Score ESG datapoints so quoted period-specific facts rank highest."""
    if quote and period:
        return 95
    if quote:
        return 85
    return 70




def _normalized_contains_value(quote: str, value: str) -> bool:
    """Return whether the quote still contains the datapoint value after normalization."""
    if not quote or not value:
        return True
    if value in quote:
        return True
    compact_value = re.sub(r"[^\w.%><=-]+", "", value.casefold())
    compact_quote = re.sub(r"[^\w.%><=-]+", "", quote.casefold())
    if compact_value and compact_value in compact_quote:
        return True
    numeric_value = re.sub(r"[^\d.%-]+", "", value).lstrip("+-")
    numeric_quote = re.sub(r"[^\d.%-]+", "", quote)
    return bool(numeric_value and numeric_value in numeric_quote)



def _is_plausible_datapoint(dp: NormalizedDatapoint) -> bool:
    """Reject obvious placeholders and ungrounded values; leave category checks to the LLM validator."""
    if dp.extractor != "openai":
        return True
    value = (dp.value or "").strip()
    quote = dp.quote or ""
    if not quote:
        return False
    if value and (PLACEHOLDER_VALUE.fullmatch(value) or STATUS_ONLY_VALUE.fullmatch(value)):
        return False
    if value and re.search(r"[\d%<>]", value) and not _normalized_contains_value(quote, value):
        return False
    return True


def normalize_llamaextract_result(
    result: AnnualReportDatapoints,
    *,
    source: str,
    company: str | None,
    year: int | None,
    extractor: str = "llamaextract",
) -> list[NormalizedDatapoint]:
    """Convert raw extraction output into cleaned normalized datapoints."""
    out: list[NormalizedDatapoint] = []

    def append_if_plausible(dp: NormalizedDatapoint) -> None:
        """Append a datapoint only when it passes plausibility checks."""
        if _is_plausible_datapoint(dp):
            out.append(dp)

    for dp in result.fte_datapoints:
        priority = _fte_priority(dp.label, dp.basis)
        append_if_plausible(NormalizedDatapoint(
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
            fact_kind=getattr(dp, "fact_kind", None),
            scope_type=getattr(dp, "scope_type", None),
            extractor=extractor,
            priority=priority,
            confidence=dp.confidence,
        ))

    for sg in result.sustainability_goals:
        priority = _sustainability_priority(sg.goal, sg.quote, sg.target_year)
        append_if_plausible(NormalizedDatapoint(
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
            fact_kind=getattr(sg, "fact_kind", None),
            scope_type=getattr(sg, "scope_type", None),
            extractor=extractor,
            priority=priority,
            confidence=sg.confidence,
        ))

    for esg in result.esg_datapoints:
        priority = _esg_priority(esg.quote, esg.period)
        append_if_plausible(NormalizedDatapoint(
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
            fact_kind=getattr(esg, "fact_kind", None),
            scope_type=getattr(esg, "scope_type", None),
            extractor=extractor,
            priority=priority,
            confidence=esg.confidence,
        ))

    for fh in result.financial_highlights:
        priority = _highlight_priority(fh.page, fh.quote)
        append_if_plausible(NormalizedDatapoint(
            source=source,
            company=company,
            year=year,
            datapoint_type="financial_highlight",
            metric=fh.metric,
            value=fh.value,
            unit=fh.unit,
            period=fh.period,
            page=fh.page,
            quote=fh.quote,
            basis=fh.basis,
            fact_kind=getattr(fh, "fact_kind", None),
            scope_type=getattr(fh, "scope_type", None),
            extractor=extractor,
            priority=priority,
            confidence=fh.confidence,
        ))

    for bp in result.business_performance:
        priority = _highlight_priority(bp.page, bp.quote)
        append_if_plausible(NormalizedDatapoint(
            source=source,
            company=company,
            year=year,
            datapoint_type="business_performance",
            metric=bp.metric,
            value=bp.value,
            unit=bp.unit,
            period=bp.period,
            page=bp.page,
            quote=bp.quote,
            basis=bp.basis,
            fact_kind=getattr(bp, "fact_kind", None),
            scope_type=getattr(bp, "scope_type", None),
            extractor=extractor,
            priority=priority,
            confidence=bp.confidence,
        ))

    for sr in result.shareholder_returns:
        priority = _highlight_priority(sr.page, sr.quote)
        append_if_plausible(NormalizedDatapoint(
            source=source,
            company=company,
            year=year,
            datapoint_type="shareholder_return",
            metric=sr.metric,
            value=sr.value,
            unit=sr.unit,
            period=sr.period,
            page=sr.page,
            quote=sr.quote,
            basis=sr.basis,
            fact_kind=getattr(sr, "fact_kind", None),
            scope_type=getattr(sr, "scope_type", None),
            extractor=extractor,
            priority=priority,
            confidence=sr.confidence,
        ))

    return out


def deduplicate_datapoints(
    datapoints: list[NormalizedDatapoint],
) -> list[NormalizedDatapoint]:
    """Return the best unique datapoints after applying category-specific deduplication rules."""
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
        if dp.priority > existing.priority:
            seen[key] = dp
        elif dp.priority == existing.priority:
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
    """Persist a normalized datapoint set to disk and return the written JSON path."""
    ds = NormalizedDatapointSet(
        source=source,
        company=company,
        year=year,
        datapoints=datapoints,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(ds.model_dump_json(indent=2), encoding="utf-8")
    return out_path
