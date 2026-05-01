from __future__ import annotations

import pytest

from backend.app.extracted_datapoints import (
    NormalizedDatapoint,
    deduplicate_datapoints,
    normalize_llamaextract_result,
)
from backend.app.llama_extract_datapoints import (
    AnnualReportDatapoints,
    ExtractedFTEDatapoint,
    ExtractedKPIHighlight,
    ExtractedSustainabilityGoal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(**kwargs) -> AnnualReportDatapoints:
    return AnnualReportDatapoints(company="ACME", year=2025, **kwargs)


def _norm(datapoints: list[NormalizedDatapoint], dtype: str) -> list[NormalizedDatapoint]:
    return [d for d in datapoints if d.datapoint_type == dtype]


# ---------------------------------------------------------------------------
# Priority rules
# ---------------------------------------------------------------------------

def test_total_employees_fte_gets_priority_100():
    result = _make_result(fte_datapoints=[
        ExtractedFTEDatapoint(label="Total employees (FTEs)", value="> 44,000", basis="FTE", page=5),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fte = _norm(dps, "fte")
    assert len(fte) == 1
    assert fte[0].priority == 100


def test_average_payroll_gets_priority_85():
    result = _make_result(fte_datapoints=[
        ExtractedFTEDatapoint(label="Average number of payroll employees in FTEs", value="43,267", basis="FTE average", page=131),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fte = _norm(dps, "fte")
    assert fte[0].priority == 85


def test_headcount_gets_priority_70():
    result = _make_result(fte_datapoints=[
        ExtractedFTEDatapoint(label="Women in our workforce (headcount)", value="21%", basis="headcount", page=5),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fte = _norm(dps, "fte")
    assert fte[0].priority == 70


def test_dedicated_fte_gets_priority_30():
    result = _make_result(fte_datapoints=[
        ExtractedFTEDatapoint(label="Dedicated FTEs for sustainability team", value="12", basis="dedicated FTE", page=50),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fte = _norm(dps, "fte")
    assert fte[0].priority == 30


def test_company_wide_outprioritizes_program_specific():
    result = _make_result(fte_datapoints=[
        ExtractedFTEDatapoint(label="Total employees (FTEs)", value="> 44,000", basis="FTE", page=5),
        ExtractedFTEDatapoint(label="Dedicated FTEs for sustainability team", value="12", page=50),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fte = sorted(_norm(dps, "fte"), key=lambda d: d.priority, reverse=True)
    assert fte[0].priority == 100
    assert fte[1].priority == 30


def test_sustainability_goal_with_year_and_quote_priority_95():
    result = _make_result(sustainability_goals=[
        ExtractedSustainabilityGoal(
            goal="Achieve net-zero GHG emissions",
            target_year="2040",
            quote="We aim to achieve net-zero GHG by 2040.",
            page=160,
        ),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    sg = _norm(dps, "sustainability_goal")
    assert sg[0].priority == 95


def test_sustainability_goal_no_quote_priority_70():
    result = _make_result(sustainability_goals=[
        ExtractedSustainabilityGoal(goal="Reduce GHG emissions", page=160),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    sg = _norm(dps, "sustainability_goal")
    assert sg[0].priority == 70


def test_kpi_with_page_and_quote_priority_90():
    result = _make_result(kpi_highlights=[
        ExtractedKPIHighlight(metric="System sales in units", value="535", page=5, quote="535 System sales in units"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    kpi = _norm(dps, "kpi_highlight")
    assert kpi[0].priority == 90


def test_kpi_without_quote_priority_60():
    result = _make_result(kpi_highlights=[
        ExtractedKPIHighlight(metric="System sales in units", value="535"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    kpi = _norm(dps, "kpi_highlight")
    assert kpi[0].priority == 60


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dp(metric: str, value: str, priority: int, confidence: float | None = None, quote: str | None = None, page: int = 5) -> NormalizedDatapoint:
    return NormalizedDatapoint(
        source="test.pdf",
        company="ACME",
        year=2025,
        datapoint_type="kpi_highlight",
        metric=metric,
        value=value,
        page=page,
        priority=priority,
        confidence=confidence,
        quote=quote,
    )


def test_dedup_keeps_higher_priority():
    dps = [
        _dp("System sales in units", "535", priority=60),
        _dp("System sales in units", "535", priority=90, quote="535 System sales"),
    ]
    result = deduplicate_datapoints(dps)
    assert len(result) == 1
    assert result[0].priority == 90


def test_dedup_keeps_higher_confidence_on_tie():
    dps = [
        _dp("Gross margin", "52.8%", priority=90, confidence=0.7),
        _dp("Gross margin", "52.8%", priority=90, confidence=0.95),
    ]
    result = deduplicate_datapoints(dps)
    assert len(result) == 1
    assert result[0].confidence == 0.95


def test_dedup_prefers_quote_on_tie():
    dps = [
        _dp("Total net sales", "32.7", priority=90, confidence=0.8),
        _dp("Total net sales", "32.7", priority=90, confidence=0.8, quote="€32.7bn Total net sales"),
    ]
    result = deduplicate_datapoints(dps)
    assert len(result) == 1
    assert result[0].quote is not None


def test_dedup_different_values_not_deduped():
    dps = [
        _dp("Net sales", "32.7", priority=90),
        _dp("Net sales", "28.3", priority=90),  # different value (prior year)
    ]
    result = deduplicate_datapoints(dps)
    assert len(result) == 2


def test_dedup_different_pages_not_deduped():
    dps = [
        _dp("Total employees (FTEs)", "> 44,000", priority=100, page=5),
        _dp("Total employees (FTEs)", "> 44,000", priority=100, page=227),
    ]
    result = deduplicate_datapoints(dps)
    assert len(result) == 2


def test_dedup_normalizes_metric_case():
    dps = [
        _dp("System Sales In Units", "535", priority=60),
        _dp("system sales in units", "535", priority=90),
    ]
    result = deduplicate_datapoints(dps)
    assert len(result) == 1
    assert result[0].priority == 90


def test_normalize_preserves_fields():
    result = _make_result(fte_datapoints=[
        ExtractedFTEDatapoint(
            label="Total employees (FTEs)",
            value="> 44,000",
            unit="FTEs",
            basis="FTE",
            period="2025",
            page=5,
            quote="> 44,000 Total employees (FTEs)",
            confidence=0.98,
        ),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    dp = dps[0]
    assert dp.value == "> 44,000"
    assert dp.unit == "FTEs"
    assert dp.basis == "FTE"
    assert dp.period == "2025"
    assert dp.page == 5
    assert dp.confidence == 0.98
    assert dp.extractor == "llamaextract"
