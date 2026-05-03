from __future__ import annotations

import pytest

from backend.app.extracted_datapoints import (
    NormalizedDatapoint,
    NormalizedDatapointSet,
    deduplicate_datapoints,
    normalize_llamaextract_result,
)
from backend.app.llama_extract_datapoints import (
    AnnualReportDatapoints,
    ExtractedBusinessPerformance,
    ExtractedESGDatapoint,
    ExtractedFinancialHighlight,
    ExtractedFTEDatapoint,
    ExtractedShareholderReturn,
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
# Priority rules — FTE
# ---------------------------------------------------------------------------

def test_total_employees_fte_gets_priority_100():
    result = _make_result(fte_datapoints=[
        ExtractedFTEDatapoint(label="Total employees (FTEs)", value="> 44,000", basis="FTE", page=5, quote="Total employees (FTEs): > 44,000", fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fte = _norm(dps, "fte")
    assert len(fte) == 1
    assert fte[0].priority == 100


def test_average_payroll_gets_priority_85():
    result = _make_result(fte_datapoints=[
        ExtractedFTEDatapoint(label="Average number of payroll employees in FTEs", value="43,267", basis="FTE average", page=131, quote="Average number of payroll employees in FTEs: 43,267", fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fte = _norm(dps, "fte")
    assert fte[0].priority == 85


def test_headcount_gets_priority_70():
    result = _make_result(fte_datapoints=[
        ExtractedFTEDatapoint(label="Women in our workforce (headcount)", value="21%", basis="headcount", page=5, quote="Women in our workforce (headcount): 21%", fact_kind="actual", scope_type="segment"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fte = _norm(dps, "fte")
    assert fte[0].priority == 70


def test_dedicated_fte_gets_priority_30():
    result = _make_result(fte_datapoints=[
        ExtractedFTEDatapoint(label="Dedicated FTEs for sustainability team", value="12", basis="dedicated FTE", page=50, quote="Dedicated FTEs for sustainability team: 12", fact_kind="actual", scope_type="project"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fte = _norm(dps, "fte")
    assert fte[0].priority == 30


def test_company_wide_outprioritizes_program_specific():
    result = _make_result(fte_datapoints=[
        ExtractedFTEDatapoint(label="Total employees (FTEs)", value="> 44,000", basis="FTE", page=5, quote="Total employees (FTEs): > 44,000", fact_kind="actual", scope_type="company_wide"),
        ExtractedFTEDatapoint(label="Dedicated FTEs for sustainability team", value="12", page=50, quote="Dedicated FTEs for sustainability team: 12", fact_kind="actual", scope_type="project"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fte = sorted(_norm(dps, "fte"), key=lambda d: d.priority, reverse=True)
    assert fte[0].priority == 100
    assert fte[1].priority == 30


# ---------------------------------------------------------------------------
# Priority rules — sustainability_goal
# ---------------------------------------------------------------------------

def test_sustainability_goal_with_year_and_quote_priority_95():
    result = _make_result(sustainability_goals=[
        ExtractedSustainabilityGoal(
            goal="Achieve net-zero GHG emissions",
            target_year="2040",
            quote="We aim to achieve net-zero GHG by 2040.",
            page=160,
            fact_kind="target",
            scope_type="company_wide",
        ),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    sg = _norm(dps, "sustainability_goal")
    assert sg[0].priority == 95


def test_sustainability_goal_no_quote_priority_70():
    result = _make_result(sustainability_goals=[
        ExtractedSustainabilityGoal(goal="Reduce GHG emissions", page=160, quote="", fact_kind="target", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    sg = _norm(dps, "sustainability_goal")
    assert sg[0].priority == 70


# ---------------------------------------------------------------------------
# Priority rules — financial_highlight, business_performance, shareholder_return
# ---------------------------------------------------------------------------

def test_financial_highlight_with_page_and_quote_priority_90():
    result = _make_result(financial_highlights=[
        ExtractedFinancialHighlight(metric="Total net sales", value="28.3", page=5, quote="€28.3bn Total net sales", fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fh = _norm(dps, "financial_highlight")
    assert fh[0].priority == 90


def test_financial_highlight_without_quote_priority_75():
    result = _make_result(financial_highlights=[
        ExtractedFinancialHighlight(metric="Gross margin", value="51.3%", page=56, quote="", fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fh = _norm(dps, "financial_highlight")
    assert fh[0].priority == 75


def test_business_performance_with_page_and_quote_priority_90():
    result = _make_result(business_performance=[
        ExtractedBusinessPerformance(metric="Systems sold", value="418", page=55, quote="418 systems sold in 2024", fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    bp = _norm(dps, "business_performance")
    assert bp[0].priority == 90


def test_shareholder_return_with_page_and_quote_priority_90():
    result = _make_result(shareholder_returns=[
        ExtractedShareholderReturn(metric="Total returned to shareholders", value="3.0", page=5, quote="€3.0bn returned", fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    sr = _norm(dps, "shareholder_return")
    assert sr[0].priority == 90


def test_shareholder_return_no_page_no_quote_priority_60():
    result = _make_result(shareholder_returns=[
        ExtractedShareholderReturn(metric="Dividends paid", value="1.2", quote="", fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    sr = _norm(dps, "shareholder_return")
    assert sr[0].priority == 60


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _dp(metric: str, value: str, priority: int, confidence: float | None = None, quote: str | None = None, page: int = 5) -> NormalizedDatapoint:
    return NormalizedDatapoint(
        source="test.pdf",
        company="ACME",
        year=2025,
        datapoint_type="financial_highlight",
        metric=metric,
        value=value,
        page=page,
        priority=priority,
        confidence=confidence,
        quote=quote,
    )


def test_dedup_keeps_higher_priority():
    dps = [
        _dp("Total net sales", "28.3", priority=75),
        _dp("Total net sales", "28.3", priority=90, quote="€28.3bn"),
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
        _dp("Net sales", "28.3", priority=90),
    ]
    result = deduplicate_datapoints(dps)
    assert len(result) == 2


def test_dedup_different_pages_not_deduped():
    dps = [
        _dp("Total net sales", "28.3", priority=90, page=5),
        _dp("Total net sales", "28.3", priority=90, page=56),
    ]
    result = deduplicate_datapoints(dps)
    assert len(result) == 2


def test_dedup_normalizes_metric_case():
    dps = [
        _dp("Total Net Sales", "28.3", priority=75),
        _dp("total net sales", "28.3", priority=90),
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
            fact_kind="actual",
            scope_type="company_wide",
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


# ---------------------------------------------------------------------------
# Normalization — new categories
# ---------------------------------------------------------------------------

def test_financial_highlight_normalize():
    result = _make_result(financial_highlights=[
        ExtractedFinancialHighlight(
            metric="Total net sales", value="28.3", unit="€bn",
            period="2024", page=56, quote="€28.3bn net sales",
            fact_kind="actual", scope_type="company_wide",
        ),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    assert len(dps) == 1
    assert dps[0].datapoint_type == "financial_highlight"
    assert dps[0].priority == 90


def test_business_performance_normalize():
    result = _make_result(business_performance=[
        ExtractedBusinessPerformance(
            metric="Lithography systems sold", value="418", unit="units",
            period="2024", page=55, quote="418 systems sold",
            fact_kind="actual", scope_type="company_wide",
        ),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    assert len(dps) == 1
    assert dps[0].datapoint_type == "business_performance"
    assert dps[0].priority == 90


def test_shareholder_return_normalize():
    result = _make_result(shareholder_returns=[
        ExtractedShareholderReturn(
            metric="Total returned to shareholders", value="3.0", unit="€bn",
            period="2024", page=5, quote="€3.0bn returned to shareholders",
            fact_kind="actual", scope_type="company_wide",
        ),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    assert len(dps) == 1
    assert dps[0].datapoint_type == "shareholder_return"
    assert dps[0].priority == 90


def test_new_categories_do_not_bleed_into_each_other():
    result = _make_result(
        financial_highlights=[
            ExtractedFinancialHighlight(metric="Net income", value="7.6", page=58, quote="net income 7.6", fact_kind="actual", scope_type="company_wide"),
        ],
        business_performance=[
            ExtractedBusinessPerformance(metric="Systems sold", value="418", page=55, quote="418 systems", fact_kind="actual", scope_type="company_wide"),
        ],
        shareholder_returns=[
            ExtractedShareholderReturn(metric="Dividends paid", value="1.2", page=5, quote="dividends 1.2", fact_kind="actual", scope_type="company_wide"),
        ],
    )
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    types = {d.datapoint_type for d in dps}
    assert types == {"financial_highlight", "business_performance", "shareholder_return"}


# ---------------------------------------------------------------------------
# ESG datapoints
# ---------------------------------------------------------------------------

def test_esg_normalizes_to_correct_type():
    result = _make_result(esg_datapoints=[
        ExtractedESGDatapoint(
            metric="Gross scope 1 and 2 GHG emissions",
            value="149 kt CO₂e",
            unit="kt CO₂e",
            period="2025",
            scope="Scope 1 and 2",
            page=160,
            quote="Gross scope 1 and 2 GHG emissions: 149 kt CO₂e",
            fact_kind="actual",
            scope_type="company_wide",
        ),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    esg = _norm(dps, "esg_datapoint")
    assert len(esg) == 1
    assert esg[0].datapoint_type == "esg_datapoint"
    assert esg[0].metric == "Gross scope 1 and 2 GHG emissions"
    assert esg[0].scope == "Scope 1 and 2"


def test_esg_priority_quote_and_period():
    result = _make_result(esg_datapoints=[
        ExtractedESGDatapoint(metric="Scope 3 emissions", value="4,200 kt", period="2025", quote="Scope 3: 4,200 kt", page=161, fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    esg = _norm(dps, "esg_datapoint")
    assert esg[0].priority == 95


def test_esg_priority_quote_only():
    result = _make_result(esg_datapoints=[
        ExtractedESGDatapoint(metric="Renewable electricity", value="85%", quote="85% renewable electricity", page=160, fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    esg = _norm(dps, "esg_datapoint")
    assert esg[0].priority == 85


def test_esg_priority_no_quote_no_period():
    result = _make_result(esg_datapoints=[
        ExtractedESGDatapoint(metric="Water use", value="1,200 ML", page=160, quote="", fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    esg = _norm(dps, "esg_datapoint")
    assert esg[0].priority == 70


def test_sustainability_category_keeps_goals_and_esg():
    result = AnnualReportDatapoints(
        company="ACME", year=2025,
        sustainability_goals=[
            ExtractedSustainabilityGoal(goal="Achieve net-zero by 2040", target_year="2040", quote="net-zero by 2040", page=5, fact_kind="target", scope_type="company_wide"),
        ],
        esg_datapoints=[
            ExtractedESGDatapoint(metric="Scope 1 emissions", value="50 kt", period="2025", quote="50 kt scope 1", page=10, fact_kind="actual", scope_type="company_wide"),
        ],
    )
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    assert any(d.datapoint_type == "sustainability_goal" for d in dps)
    assert any(d.datapoint_type == "esg_datapoint" for d in dps)


# ---------------------------------------------------------------------------
# fact_kind and scope_type propagation
# ---------------------------------------------------------------------------

def test_fact_kind_propagates_for_fte():
    result = _make_result(fte_datapoints=[
        ExtractedFTEDatapoint(label="Total employees (FTEs)", value="> 44,000", basis="FTE", page=5, quote="Total employees: > 44,000", fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    assert dps[0].fact_kind == "actual"
    assert dps[0].scope_type == "company_wide"


def test_fact_kind_propagates_for_sustainability_goal():
    result = _make_result(sustainability_goals=[
        ExtractedSustainabilityGoal(goal="Reach net-zero by 2050", target_year="2050", quote="net-zero by 2050", page=5, fact_kind="target", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    sg = [d for d in dps if d.datapoint_type == "sustainability_goal"]
    assert sg[0].fact_kind == "target"
    assert sg[0].scope_type == "company_wide"


def test_fact_kind_propagates_for_esg():
    result = _make_result(esg_datapoints=[
        ExtractedESGDatapoint(metric="Scope 1 and 2 GHG emissions", value="149 kt", period="2025", quote="Scope 1 and 2 GHG emissions: 149 kt", page=10, fact_kind="actual", scope_type="scope_1_2"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    esg = [d for d in dps if d.datapoint_type == "esg_datapoint"]
    assert esg[0].fact_kind == "actual"
    assert esg[0].scope_type == "scope_1_2"


def test_fact_kind_propagates_for_financial_highlight():
    result = _make_result(financial_highlights=[
        ExtractedFinancialHighlight(metric="Net income", value="7.6", unit="€bn", period="2024", page=58, quote="net income 7.6", fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    assert dps[0].fact_kind == "actual"
    assert dps[0].scope_type == "company_wide"


def test_fact_kind_propagates_for_business_performance():
    result = _make_result(business_performance=[
        ExtractedBusinessPerformance(metric="Systems sold", value="418", period="2024", page=55, quote="418 systems sold", fact_kind="actual", scope_type="segment"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    assert dps[0].fact_kind == "actual"
    assert dps[0].scope_type == "segment"


def test_fact_kind_propagates_for_shareholder_return():
    result = _make_result(shareholder_returns=[
        ExtractedShareholderReturn(metric="Total returned to shareholders", value="3.0", unit="€bn", period="2024", page=5, quote="€3.0bn returned", fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    assert dps[0].fact_kind == "actual"
    assert dps[0].scope_type == "company_wide"


def test_scope_type_scope_3_propagates():
    result = _make_result(esg_datapoints=[
        ExtractedESGDatapoint(metric="Scope 3 emissions", value="4,200 kt", period="2025", quote="Scope 3: 4,200 kt", page=11, fact_kind="actual", scope_type="scope_3"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    assert dps[0].scope_type == "scope_3"


# ---------------------------------------------------------------------------
# OpenAI validation — quote, placeholder, status-only, value-in-quote
# ---------------------------------------------------------------------------

def _openai_fin(**kwargs) -> AnnualReportDatapoints:
    defaults = dict(
        metric="Net income", value="7.6", unit="€bn", period="2024", page=5,
        quote="Net income was €7.6bn in 2024.", fact_kind="actual", scope_type="company_wide",
    )
    defaults.update(kwargs)
    return _make_result(financial_highlights=[ExtractedFinancialHighlight(**defaults)])


def _norm_openai(result: AnnualReportDatapoints) -> list:
    return normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025, extractor="openai")


def test_openai_blank_quote_rejected():
    dps = _norm_openai(_openai_fin(quote=""))
    assert len(dps) == 0


def test_openai_valid_quote_accepted():
    dps = _norm_openai(_openai_fin(quote="Net income was €7.6bn in 2024."))
    assert len(dps) == 1


def test_openai_placeholder_unknown_rejected():
    dps = _norm_openai(_openai_fin(value="unknown", quote="Net income unknown"))
    assert len(dps) == 0


def test_openai_placeholder_na_rejected():
    dps = _norm_openai(_openai_fin(value="N/A", quote="Net income N/A"))
    assert len(dps) == 0


def test_openai_placeholder_dash_rejected():
    dps = _norm_openai(_openai_fin(value="-", quote="Net income -"))
    assert len(dps) == 0


def test_openai_placeholder_xx_percent_rejected():
    dps = _norm_openai(_openai_fin(value="xx%", quote="reduction of xx%"))
    assert len(dps) == 0


def test_openai_status_on_track_rejected():
    dps = _norm_openai(_openai_fin(value="On track", quote="Net income on track"))
    assert len(dps) == 0


def test_openai_status_tbc_rejected():
    dps = _norm_openai(_openai_fin(value="TBC", quote="Net income TBC"))
    assert len(dps) == 0


def test_openai_status_tbd_rejected():
    dps = _norm_openai(_openai_fin(value="TBD", quote="Net income TBD"))
    assert len(dps) == 0


def test_openai_status_in_progress_rejected():
    dps = _norm_openai(_openai_fin(value="in progress", quote="Net income in progress"))
    assert len(dps) == 0


def test_openai_status_ongoing_rejected():
    dps = _norm_openai(_openai_fin(value="ongoing", quote="Net income ongoing"))
    assert len(dps) == 0


def test_openai_value_not_in_quote_rejected():
    dps = _norm_openai(_openai_fin(value="7.6", quote="Net income was strong last year."))
    assert len(dps) == 0


def test_openai_percent_value_not_in_quote_rejected():
    dps = _norm_openai(_openai_fin(value="51.3%", quote="Gross margin improved year on year."))
    assert len(dps) == 0


def test_openai_less_than_value_not_in_quote_rejected():
    dps = _norm_openai(_openai_fin(value="<5%", quote="Gross margin was strong."))
    assert len(dps) == 0


def test_openai_percent_value_in_quote_accepted():
    dps = _norm_openai(_openai_fin(value="51.3%", quote="Gross margin was 51.3% in 2024."))
    assert len(dps) == 1


def test_llamaextract_blank_quote_not_rejected():
    result = _make_result(financial_highlights=[
        ExtractedFinancialHighlight(metric="Net income", value="7.6", unit="€bn", period="2024", page=5,
                                    quote="", fact_kind="actual", scope_type="company_wide"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025, extractor="llamaextract")
    assert len(dps) == 1


# ---------------------------------------------------------------------------
# Raw audit payload
# ---------------------------------------------------------------------------

from scripts.run_pre_extraction import _build_raw_audit_payload  # noqa: E402


def test_raw_audit_payload_structure():
    raw_batch = {"company": "ACME", "year": 2024, "fte_datapoints": [{"label": "FTE", "value": "44000"}]}
    payload = _build_raw_audit_payload(
        source="acme_2024.pdf",
        company="ACME",
        year=2024,
        extractor="openai",
        categories=["fte", "esg"],
        raw_by_cat={"fte": [raw_batch]},
    )
    assert payload["source"] == "acme_2024.pdf"
    assert payload["company"] == "ACME"
    assert payload["year"] == 2024
    assert payload["extractor"] == "openai"
    assert payload["categories"] == ["fte", "esg"]
    assert payload["raw"]["fte"] == [raw_batch]


def test_raw_audit_payload_does_not_mutate_input():
    raw_batch = {"company": "ACME", "fte_datapoints": []}
    original = {"fte": [raw_batch]}
    _build_raw_audit_payload("x.pdf", "ACME", 2024, "openai", ["fte"], original)
    assert original == {"fte": [raw_batch]}


def test_raw_audit_payload_empty_categories():
    payload = _build_raw_audit_payload("x.pdf", None, None, "openai", [], {})
    assert payload["categories"] == []
    assert payload["raw"] == {}
