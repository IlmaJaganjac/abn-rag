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
_PLACEHOLDER_VALUE = re.compile(r"^(?:unknown|n/?a|none|null|[-—]+|[€$£]?\s*x+x+%?|x+x+%?)$", re.IGNORECASE)
_STATUS_ONLY_VALUE = re.compile(
    r"^(?:approved|executed|suspended|modified|discontinued|may\s+(?:suspend|not declare|pay).*|"
    r"on\s+track|tbc|tbd|in\s+progress|ongoing)$",
    re.IGNORECASE,
)
_ACTUAL_METRIC = re.compile(r"\bactual\b|\breported\b|\bperformance\b", re.IGNORECASE)
_PROGRESS_SIGNAL = re.compile(r"\bprogress\b|achieved|completion|against\s+(?:the\s+)?target", re.IGNORECASE)
_FORECAST_SIGNAL = re.compile(r"\bforecast\b|outlook|guidance|expect(?:ed|s|ation)?|anticipate|project(?:ed|ion)?", re.IGNORECASE)
_DEFINITION_SIGNAL = re.compile(r"\bdefinition\b|defined as|means|refers to", re.IGNORECASE)
_SCOPE_CUSTOMER_SIGNAL = re.compile(r"\bcustomer\b|client", re.IGNORECASE)
_SCOPE_SEGMENT_SIGNAL = re.compile(r"\bsegment\b|division|business unit|product line|product-specific", re.IGNORECASE)
_SCOPE_GEOGRAPHY_SIGNAL = re.compile(r"\bgeograph|region|country|emea|asia|europe|united states|china|japan", re.IGNORECASE)
_SCOPE_PRODUCT_SIGNAL = re.compile(r"\bproduct\b|system|unit model", re.IGNORECASE)
_TOTAL_COMPANY_SIGNAL = re.compile(r"\btotal\b|company[- ]wide|group", re.IGNORECASE)
_COUNT_METRIC_SIGNAL = re.compile(r"\b(?:count|number|units?|systems?|employees?|fte?s?|headcount|sold|shipped)\b", re.IGNORECASE)
_RATE_UNIT_SIGNAL = re.compile(r"%|per[-\s]?hour|per\s+\w+|rate|ratio|intensity|throughput|efficiency", re.IGNORECASE)

# FTE priority helpers
_FTE_COMPANY_WIDE = re.compile(
    r"total\s+employees?|total\s+workforce|all\s+employees?|"
    r"total\s+number\s+of\s+(?:payroll\s+)?employees|internal\s+employees?",
    re.IGNORECASE,
)
_FTE_AVG_PAYROLL = re.compile(
    r"\baverage\b|\bpayroll\b|\btemporary\b|\bpermanent\b",
    re.IGNORECASE,
)
_FTE_HEADCOUNT = re.compile(r"\bheadcount\b", re.IGNORECASE)
_FTE_SPECIFIC = re.compile(
    r"dedicated\s+fte|team\s+fte|program\s+fte|project\s+fte",
    re.IGNORECASE,
)
_FTE_SIGNAL = re.compile(
    r"\bfte?s?\b|full[- ]time\s+equivalents?|headcount|employees?|workforce|"
    r"payroll|permanent|temporary|internal|external|contractors?|turnover|attrition",
    re.IGNORECASE,
)
_FTE_NON_EMPLOYEE = re.compile(
    r"survey\s+score|engagement\s+score|training\s+hours?|lost\s+time|incident\s+rate|"
    r"revenue|net\s+sales|dividend|buyback|emissions?|scope\s+[123]|net[- ]zero",
    re.IGNORECASE,
)
_SUST_SIGNAL = re.compile(
    r"emissions?|greenhouse\s+gas|\bghg\b|co2e?|co₂e?|scope\s+[123]|net[- ]zero|"
    r"carbon|methane|flaring|renewable|energy\s+(?:efficien|savings?|use)|"
    r"electricity|power\s+consumption|wafer|waste|reuse|re-?use|recycl|circular|water|"
    r"biodiversity|supplier\s+sustainability|"
    r"diversity|inclusion|safety|ethics|governance",
    re.IGNORECASE,
)
_SUST_TARGET_SIGNAL = re.compile(
    r"target|goal|ambition|commitment|committed|aim|reduce|reduction|halve|"
    r"achieve|become|maintain|eliminate|by\s+20[2-9]\d|20[3-9]\d",
    re.IGNORECASE,
)
_SUST_BUSINESS_ONLY = re.compile(
    r"lng\s+sales|liquids?\s+production|barrels?\s+per\s+day|refining\s+throughput|"
    r"production\s+growth|volume\s+growth|market\s+share|revenue\s+growth|"
    r"customer\s+growth|stores?|branches?|locations?",
    re.IGNORECASE,
)
_ESG_SIGNAL = re.compile(
    r"emissions?|greenhouse\s+gas|\bghg\b|co2e?|co₂e?|scope\s+[123]|renewable|"
    r"energy|waste|reuse|re-?use|recycl|water|circular|biodiversity|supplier|diversity|"
    r"inclusion|safety|ethics|governance",
    re.IGNORECASE,
)
_FIN_SIGNAL = re.compile(
    r"net\s+sales|revenue|total\s+income|gross\s+profit|gross\s+margin|"
    r"operating\s+(?:income|profit)|income\s+from\s+operations|\bebit(?:da)?\b|"
    r"net\s+(?:income|profit)|income\s+tax|effective\s+tax\s+rate|\betr\b|"
    r"earnings\s+per\s+share|\beps\b|r&d|research\s+and\s+development|"
    r"free\s+cash\s+flow|operating\s+cash\s+flow|cash\s+flow\s+from\s+operat|"
    r"net\s+cash\s+provided\s+by\s+operating\s+activities|"
    r"cash\s+and\s+cash\s+equivalents|short[- ]term\s+investments?|"
    r"capex|capital\s+expenditure|property,\s+plant\s+and\s+equipment|"
    r"intangible\s+assets|return\s+on\s+(?:equity|invested\s+capital)|"
    r"\broe\b|\broic\b|\bcet1\b|capital\s+ratio|liquidity\s+coverage|"
    r"net\s+interest\s+margin|\bnim\b",
    re.IGNORECASE,
)
_BIZ_SIGNAL = re.compile(
    r"systems?\s+sold|systems?\s+recognized|lithography\s+systems?|euv\s+systems?|"
    r"units?\s+sold|installed\s+base|"
    r"order\s+intake|order\s+book|backlog|bookings|customers?|clients?|"
    r"customer\s+satisfaction|suppliers?|reuse\s+rate|market\s+share|"
    r"production\s+volume|deliveries|loans?|deposits?|mortgages?|lng|"
    r"barrels?\s+per\s+day|refining\s+throughput|beer\s+volume|hectoliters?|"
    r"stores?|branches?|locations?|assets\s+under\s+management|\baum\b|"
    r"transaction\s+volume",
    re.IGNORECASE,
)
_SH_SIGNAL = re.compile(
    r"dividend|share\s+buybacks?|share\s+repurchases?|repurchased|"
    r"returned?\s+to\s+shareholders?|shareholder\s+(?:returns?|distributions?)|"
    r"capital\s+return|payout\s+ratio|treasury\s+shares?|shares?\s+cancelled|"
    r"cash\s+returned",
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
    fact_kind: str | None = None
    scope_type: str | None = None
    quality: str | None = None
    validation_status: str | None = None
    canonical_metric: str | None = None


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
    return 60


def _sustainability_priority(goal: str, quote: str | None, target_year: str | None) -> int:
    if quote and target_year:
        return 95
    if target_year:
        return 80
    if quote:
        return 75
    return 70


def _highlight_priority(page: int | None, quote: str | None) -> int:
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


def _combined_dp_text(dp: NormalizedDatapoint) -> str:
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


def _normalized_contains_value(quote: str, value: str) -> bool:
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


def _is_scoped_total(dp: NormalizedDatapoint) -> bool:
    text = _combined_dp_text(dp)
    return bool(
        _TOTAL_COMPANY_SIGNAL.search(dp.metric or "")
        and (
            _SCOPE_CUSTOMER_SIGNAL.search(text)
            or _SCOPE_SEGMENT_SIGNAL.search(text)
            or _SCOPE_GEOGRAPHY_SIGNAL.search(text)
            or _SCOPE_PRODUCT_SIGNAL.search(text)
            or _FTE_SPECIFIC.search(text)
        )
    )


def _is_plausible_datapoint(dp: NormalizedDatapoint) -> bool:
    text = _combined_dp_text(dp)
    value = (dp.value or "").strip()
    quote = dp.quote or ""
    metric = dp.metric or ""
    unit = dp.unit or ""
    strict_openai = dp.extractor == "openai"
    if strict_openai:
        if not quote:
            return False
        if value and (_PLACEHOLDER_VALUE.fullmatch(value) or _STATUS_ONLY_VALUE.fullmatch(value)):
            return False
        if value and re.search(r"[\d%<>]", value) and not _normalized_contains_value(quote, value):
            return False
    if dp.datapoint_type == "fte":
        return bool(_FTE_SIGNAL.search(text)) and not bool(_FTE_NON_EMPLOYEE.search(text))

    if dp.datapoint_type == "sustainability_goal":
        if strict_openai and _ACTUAL_METRIC.search(metric):
            return False
        has_sust_signal = bool(_SUST_SIGNAL.search(text))
        has_target_signal = bool(_SUST_TARGET_SIGNAL.search(text) or dp.target_year)
        business_only = bool(_SUST_BUSINESS_ONLY.search(text)) and not has_sust_signal
        return has_sust_signal and has_target_signal and not business_only

    if dp.datapoint_type == "esg_datapoint":
        return bool(_ESG_SIGNAL.search(text)) and not bool(_SUST_TARGET_SIGNAL.search(text))

    if dp.datapoint_type == "financial_highlight":
        if _SH_SIGNAL.search(text) or _FTE_SIGNAL.search(text) or _SUST_SIGNAL.search(text):
            return False
        return bool(_FIN_SIGNAL.search(text))

    if dp.datapoint_type == "business_performance":
        if _FTE_SIGNAL.search(text) or _SH_SIGNAL.search(text):
            return False
        if _SUST_SIGNAL.search(text) and _SUST_TARGET_SIGNAL.search(text):
            return False
        return bool(_BIZ_SIGNAL.search(text))

    if dp.datapoint_type == "shareholder_return":
        return bool(_SH_SIGNAL.search(text))

    return True


# ---------------------------------------------------------------------------
# Public normalization
# ---------------------------------------------------------------------------

def normalize_llamaextract_result(
    result: AnnualReportDatapoints,
    *,
    source: str,
    company: str | None,
    year: int | None,
    extractor: str = "llamaextract",
) -> list[NormalizedDatapoint]:
    out: list[NormalizedDatapoint] = []

    def append_if_plausible(dp: NormalizedDatapoint) -> None:
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
