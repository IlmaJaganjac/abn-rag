from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel

from backend.app.extract.schemas import AnnualReportDatapoints
from backend.app.extract.signals import (
    ACTUAL_METRIC,
    BIZ_SIGNAL,
    DEFINITION_SIGNAL,
    ESG_SIGNAL,
    FIN_SIGNAL,
    FTE_AVG_PAYROLL,
    FTE_COMPANY_WIDE,
    FTE_HEADCOUNT,
    FTE_NON_EMPLOYEE,
    FTE_SIGNAL,
    FTE_SPECIFIC,
    PLACEHOLDER_VALUE,
    PUNCT_RE,
    RATE_UNIT_SIGNAL,
    SH_SIGNAL,
    SPACE_RE,
    STATUS_ONLY_VALUE,
    SUST_BUSINESS_ONLY,
    SUST_SIGNAL,
    SUST_TARGET_SIGNAL,
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
    extractor: str = "openai"
    priority: int = 50
    confidence: float | None = None
    fact_kind: str | None = None
    scope_type: str | None = None
    quality: str | None = None
    validation_status: str | None = None
    canonical_metric: str | None = None


def norm_text(s: str | None) -> str:
    """Normalize free text for deduplication and heuristic matching."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = PUNCT_RE.sub(" ", s).lower()
    s = SPACE_RE.sub(" ", s).strip()
    return s


def norm_value(s: str | None) -> str:
    """Normalize a datapoint value string for value-level comparisons."""
    if not s:
        return ""
    s = s.lower().strip()
    s = SYMBOL_SPACE_RE.sub(r"\1", s)
    s = SPACE_RE.sub(" ", s).strip()
    return s


def fte_priority(metric: str, basis: str | None) -> int:
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


def sustainability_priority(quote: str | None, target_year: str | None) -> int:
    """Score sustainability goals so more explicit targets rank above vague ones."""
    if quote and target_year:
        return 95
    if target_year:
        return 80
    if quote:
        return 75
    return 70


def highlight_priority(page: int | None, quote: str | None) -> int:
    """Score generic highlight datapoints based on evidence completeness."""
    if page is not None and quote:
        return 90
    if page is not None or quote:
        return 75
    return 60


def esg_priority(quote: str | None, period: str | None) -> int:
    """Score ESG datapoints so quoted period-specific facts rank highest."""
    if quote and period:
        return 95
    if quote:
        return 85
    return 70


def combined_dp_text(dp: NormalizedDatapoint) -> str:
    """Concatenate the key fields of one datapoint into one searchable text string."""
    return " ".join(
        part for part in (
            dp.datapoint_type,
            dp.metric,
            dp.value,
            dp.unit,
            dp.period,
            dp.quote,
            dp.basis,
            dp.scope,
            dp.target_year,
        )
        if part
    )


def normalized_contains_value(quote: str, value: str) -> bool:
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


def is_percentage_or_rate(dp: NormalizedDatapoint) -> bool:
    """Return whether a datapoint reports a percentage, rate, or ratio rather than a count."""
    value = dp.value or ""
    unit = dp.unit or ""
    metric_basis = " ".join(
        part for part in (dp.metric, dp.unit, dp.basis) if part
    )
    return "%" in value or "%" in unit or bool(RATE_UNIT_SIGNAL.search(metric_basis))


def is_definition_without_numeric_value(dp: NormalizedDatapoint) -> bool:
    """Return whether a datapoint looks like a textual definition with no numeric value."""
    text = combined_dp_text(dp)
    value = dp.value or ""
    return bool(DEFINITION_SIGNAL.search(text)) and not bool(re.search(r"\d|%|€|\$|£", value))


def strip_duplicated_unit(value: str | None, unit: str | None) -> str | None:
    """Return value with the trailing unit removed when value duplicates unit."""
    if not value or not unit:
        return value
    v = value.rstrip()
    u = unit.strip()
    if not u or not v.lower().endswith(u.lower()):
        return value
    head = v[: -len(u)].rstrip()
    if not head or not re.search(r"[\d%]", head):
        return value
    return head


def is_ambiguous_fte_table_row(dp: NormalizedDatapoint) -> bool:
    """Reject FTE-looking rows from governance, remuneration, or ambiguous contract tables."""
    text = combined_dp_text(dp)
    metric = dp.metric or ""
    quote = dp.quote or ""
    value = dp.value or ""
    unit = (dp.unit or "").lower()

    if re.search(
        r"executive\s+board|supervisory\s+board|identified\s+staff|\bcla\+?\b|"
        r"remuneration|compensation|severance|sign[-\s]?on|bonus|salary|wages",
        text,
        re.IGNORECASE,
    ):
        return True

    if re.search(
        r"contract\s+types?|temporary|permanent|full[-\s]?time|part[-\s]?time",
        metric,
        re.IGNORECASE,
    ):
        if unit not in {"fte", "ftes", "headcount", "employees", "persons"}:
            return True
        if not re.search(
            r"headcount|fte|full[-\s]?time\s+equivalent|number\s+of\s+employees|employees\s+\(",
            quote,
            re.IGNORECASE,
        ):
            return True
        if re.fullmatch(r"\d{1,2}", value.strip()) and not re.search(
            r"headcount|fte|number\s+of",
            quote,
            re.IGNORECASE,
        ):
            return True

    return False


def is_plausible_datapoint(dp: NormalizedDatapoint) -> bool:
    """Reject placeholder, mis-scoped, or category-inconsistent datapoints."""
    text = combined_dp_text(dp)
    value = (dp.value or "").strip()
    quote = dp.quote or ""
    metric = dp.metric or ""
    strict_openai = dp.extractor == "openai"
    if strict_openai:
        if not quote:
            return False
        if value and (PLACEHOLDER_VALUE.fullmatch(value) or STATUS_ONLY_VALUE.fullmatch(value)):
            return False
        if value and re.search(r"[\d%<>]", value) and not normalized_contains_value(quote, value):
            return False
        if is_definition_without_numeric_value(dp):
            return False
    if dp.datapoint_type == "fte":
        if strict_openai and dp.fact_kind != "actual":
            return False
        if is_percentage_or_rate(dp):
            return False
        if is_ambiguous_fte_table_row(dp):
            return False
        return bool(FTE_SIGNAL.search(text)) and not bool(FTE_NON_EMPLOYEE.search(text))

    if dp.datapoint_type == "sustainability_goal":
        if strict_openai and dp.fact_kind not in {"target", "forecast"}:
            return False
        if strict_openai and ACTUAL_METRIC.search(metric):
            return False
        has_sust_signal = bool(SUST_SIGNAL.search(text))
        has_target_signal = bool(SUST_TARGET_SIGNAL.search(text) or dp.target_year)
        business_only = bool(SUST_BUSINESS_ONLY.search(text)) and not has_sust_signal
        return has_sust_signal and has_target_signal and not business_only

    if dp.datapoint_type == "esg_datapoint":
        if strict_openai and dp.fact_kind != "actual":
            return False
        return bool(ESG_SIGNAL.search(text)) and not bool(SUST_TARGET_SIGNAL.search(text))

    if dp.datapoint_type == "financial_highlight":
        if strict_openai and dp.fact_kind != "actual":
            return False
        if SH_SIGNAL.search(text) or FTE_SIGNAL.search(text) or SUST_SIGNAL.search(text):
            return False
        return bool(FIN_SIGNAL.search(text))

    if dp.datapoint_type == "business_performance":
        if strict_openai and dp.fact_kind != "actual":
            return False
        if FTE_SIGNAL.search(text) or SH_SIGNAL.search(text):
            return False
        if SUST_SIGNAL.search(text) and SUST_TARGET_SIGNAL.search(text):
            return False
        return bool(BIZ_SIGNAL.search(text))

    if dp.datapoint_type == "shareholder_return":
        return bool(SH_SIGNAL.search(text))

    return True


def normalize_llamaextract_result(
    result: AnnualReportDatapoints,
    *,
    source: str,
    company: str | None,
    year: int | None,
    extractor: str = "openai",
) -> list[NormalizedDatapoint]:
    """Convert raw extraction output into cleaned normalized datapoints."""
    out: list[NormalizedDatapoint] = []

    def append_if_plausible(dp: NormalizedDatapoint) -> None:
        """Append a datapoint only when it passes plausibility checks."""
        if is_plausible_datapoint(dp):
            out.append(dp)

    for dp in result.fte_datapoints:
        priority = fte_priority(dp.label, dp.basis)
        append_if_plausible(NormalizedDatapoint(
            source=source,
            company=company,
            year=year,
            datapoint_type="fte",
            metric=dp.label,
            value=strip_duplicated_unit(dp.value, dp.unit),
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
        priority = sustainability_priority(sg.quote, sg.target_year)
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
        priority = esg_priority(esg.quote, esg.period)
        append_if_plausible(NormalizedDatapoint(
            source=source,
            company=company,
            year=year,
            datapoint_type="esg_datapoint",
            metric=esg.metric,
            value=strip_duplicated_unit(esg.value, esg.unit),
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
        priority = highlight_priority(fh.page, fh.quote)
        append_if_plausible(NormalizedDatapoint(
            source=source,
            company=company,
            year=year,
            datapoint_type="financial_highlight",
            metric=fh.metric,
            value=strip_duplicated_unit(fh.value, fh.unit),
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
        priority = highlight_priority(bp.page, bp.quote)
        append_if_plausible(NormalizedDatapoint(
            source=source,
            company=company,
            year=year,
            datapoint_type="business_performance",
            metric=bp.metric,
            value=strip_duplicated_unit(bp.value, bp.unit),
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
        priority = highlight_priority(sr.page, sr.quote)
        append_if_plausible(NormalizedDatapoint(
            source=source,
            company=company,
            year=year,
            datapoint_type="shareholder_return",
            metric=sr.metric,
            value=strip_duplicated_unit(sr.value, sr.unit),
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
            norm_text(dp.metric),
            norm_value(dp.value),
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
