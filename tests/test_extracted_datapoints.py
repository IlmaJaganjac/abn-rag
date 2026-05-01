from __future__ import annotations

import pytest

from backend.app.extracted_datapoints import (
    NormalizedDatapoint,
    NormalizedDatapointSet,
    datapoints_to_chunks,
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


# ---------------------------------------------------------------------------
# Priority rules — financial_highlight, business_performance, shareholder_return
# ---------------------------------------------------------------------------

def test_financial_highlight_with_page_and_quote_priority_90():
    result = _make_result(financial_highlights=[
        ExtractedFinancialHighlight(metric="Total net sales", value="28.3", page=5, quote="€28.3bn Total net sales"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fh = _norm(dps, "financial_highlight")
    assert fh[0].priority == 90


def test_financial_highlight_without_quote_priority_75():
    result = _make_result(financial_highlights=[
        ExtractedFinancialHighlight(metric="Gross margin", value="51.3%", page=56),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    fh = _norm(dps, "financial_highlight")
    assert fh[0].priority == 75


def test_business_performance_with_page_and_quote_priority_90():
    result = _make_result(business_performance=[
        ExtractedBusinessPerformance(metric="Systems sold", value="418", page=55, quote="418 systems sold in 2024"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    bp = _norm(dps, "business_performance")
    assert bp[0].priority == 90


def test_shareholder_return_with_page_and_quote_priority_90():
    result = _make_result(shareholder_returns=[
        ExtractedShareholderReturn(metric="Total returned to shareholders", value="3.0", page=5, quote="€3.0bn returned"),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    sr = _norm(dps, "shareholder_return")
    assert sr[0].priority == 90


def test_shareholder_return_no_page_no_quote_priority_60():
    result = _make_result(shareholder_returns=[
        ExtractedShareholderReturn(metric="Dividends paid", value="1.2"),
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
# datapoints_to_chunks — helpers
# ---------------------------------------------------------------------------

def _make_ds(datapoints: list[NormalizedDatapoint]) -> NormalizedDatapointSet:
    return NormalizedDatapointSet(
        source="test.pdf", company="ACME", year=2025, datapoints=datapoints
    )


def _fte_dp(**kwargs) -> NormalizedDatapoint:
    defaults = dict(
        source="test.pdf", company="ACME", year=2025,
        datapoint_type="fte", metric="Total employees (FTEs)",
        value="> 44,000", unit="FTEs", basis="FTE", period="2025",
        page=5, quote="> 44,000 Total employees (FTEs)", priority=100,
    )
    defaults.update(kwargs)
    return NormalizedDatapoint(**defaults)


def _sust_dp(**kwargs) -> NormalizedDatapoint:
    defaults = dict(
        source="test.pdf", company="ACME", year=2025,
        datapoint_type="sustainability_goal",
        metric="Absolute gross scope 1 and 2 GHG emissions",
        value="-75%", target_year="2030",
        quote="Reduce absolute gross scope 1 and 2 GHG by 75% by 2030",
        page=160, priority=95,
    )
    defaults.update(kwargs)
    return NormalizedDatapoint(**defaults)


def _fin_dp(**kwargs) -> NormalizedDatapoint:
    defaults = dict(
        source="test.pdf", company="ACME", year=2025,
        datapoint_type="financial_highlight",
        metric="Total net sales",
        value="28.3", unit="€bn", period="2024",
        page=56, quote="Total net sales €28.3bn", priority=90,
    )
    defaults.update(kwargs)
    return NormalizedDatapoint(**defaults)


def _biz_dp(**kwargs) -> NormalizedDatapoint:
    defaults = dict(
        source="test.pdf", company="ACME", year=2025,
        datapoint_type="business_performance",
        metric="Lithography systems sold",
        value="418", unit="units", period="2024",
        page=55, quote="418 systems sold in 2024", priority=90,
    )
    defaults.update(kwargs)
    return NormalizedDatapoint(**defaults)


def _sh_dp(**kwargs) -> NormalizedDatapoint:
    defaults = dict(
        source="test.pdf", company="ACME", year=2025,
        datapoint_type="shareholder_return",
        metric="Total returned to shareholders",
        value="3.0", unit="€bn", period="2024",
        page=5, quote="€3.0bn returned to shareholders", priority=90,
    )
    defaults.update(kwargs)
    return NormalizedDatapoint(**defaults)


# ---------------------------------------------------------------------------
# Chunk kind and section_path
# ---------------------------------------------------------------------------

def test_fte_chunk_kind():
    chunks = datapoints_to_chunks(_make_ds([_fte_dp()]))
    assert len(chunks) == 1
    assert chunks[0].chunk_kind == "extracted_datapoint"
    assert chunks[0].parser == "llamaextract"


def test_sustainability_chunk_kind():
    chunks = datapoints_to_chunks(_make_ds([_sust_dp()]))
    assert chunks[0].chunk_kind == "extracted_datapoint"


def test_financial_highlight_chunk_kind_and_section_path():
    chunks = datapoints_to_chunks(_make_ds([_fin_dp()]))
    assert chunks[0].chunk_kind == "extracted_datapoint"
    assert chunks[0].parser == "llamaextract"
    assert "Financial highlights" in chunks[0].section_path


def test_business_performance_chunk_kind_and_section_path():
    chunks = datapoints_to_chunks(_make_ds([_biz_dp()]))
    assert chunks[0].chunk_kind == "extracted_datapoint"
    assert chunks[0].parser == "llamaextract"
    assert "Business performance" in chunks[0].section_path


def test_shareholder_return_chunk_kind_and_section_path():
    chunks = datapoints_to_chunks(_make_ds([_sh_dp()]))
    assert chunks[0].chunk_kind == "extracted_datapoint"
    assert chunks[0].parser == "llamaextract"
    assert "Shareholder returns" in chunks[0].section_path


# ---------------------------------------------------------------------------
# Text format
# ---------------------------------------------------------------------------

def test_fte_text_contains_required_fields():
    chunks = datapoints_to_chunks(_make_ds([_fte_dp()]))
    text = chunks[0].text
    assert "Metric:" in text
    assert "Value:" in text
    assert "Quote:" in text
    assert "Priority:" in text
    assert "Total employees (FTEs)" in text
    assert "> 44,000" in text


def test_sustainability_text_contains_target_year():
    chunks = datapoints_to_chunks(_make_ds([_sust_dp()]))
    text = chunks[0].text
    assert "Target year: 2030" in text
    assert "Value/target:" in text


def test_financial_highlight_text_format():
    chunks = datapoints_to_chunks(_make_ds([_fin_dp()]))
    text = chunks[0].text
    assert "Datapoint type: financial_highlight" in text
    assert "Metric:" in text
    assert "Value:" in text
    assert "Period:" in text
    assert "Quote:" in text


def test_business_performance_text_format():
    chunks = datapoints_to_chunks(_make_ds([_biz_dp()]))
    text = chunks[0].text
    assert "Datapoint type: business_performance" in text
    assert "Metric:" in text
    assert "Value:" in text


def test_shareholder_return_text_format():
    chunks = datapoints_to_chunks(_make_ds([_sh_dp()]))
    text = chunks[0].text
    assert "Datapoint type: shareholder_return" in text
    assert "Metric:" in text
    assert "Value:" in text


# ---------------------------------------------------------------------------
# Embedding text synonyms
# ---------------------------------------------------------------------------

def test_fte_embedding_text_has_synonyms():
    chunks = datapoints_to_chunks(_make_ds([_fte_dp()]))
    emb = chunks[0].embedding_text or ""
    assert "workforce" in emb
    assert "headcount" in emb
    assert "payroll" in emb


def test_sustainability_embedding_text_has_synonyms():
    chunks = datapoints_to_chunks(_make_ds([_sust_dp()]))
    emb = chunks[0].embedding_text or ""
    assert "sustainability" in emb
    assert "emissions" in emb
    assert "net zero" in emb


def test_financial_highlight_embedding_synonyms():
    chunks = datapoints_to_chunks(_make_ds([_fin_dp()]))
    emb = chunks[0].embedding_text or ""
    assert "revenue" in emb
    assert "gross margin" in emb
    assert "net income" in emb
    assert "free cash flow" in emb


def test_business_performance_embedding_synonyms():
    chunks = datapoints_to_chunks(_make_ds([_biz_dp()]))
    emb = chunks[0].embedding_text or ""
    assert "systems sold" in emb
    assert "suppliers" in emb
    assert "customer satisfaction" in emb


def test_shareholder_return_embedding_synonyms():
    chunks = datapoints_to_chunks(_make_ds([_sh_dp()]))
    emb = chunks[0].embedding_text or ""
    assert "dividend" in emb
    assert "buyback" in emb
    assert "returned to shareholders" in emb


# ---------------------------------------------------------------------------
# Normalization — new categories
# ---------------------------------------------------------------------------

def test_financial_highlight_normalize():
    result = _make_result(financial_highlights=[
        ExtractedFinancialHighlight(
            metric="Total net sales", value="28.3", unit="€bn",
            period="2024", page=56, quote="€28.3bn net sales",
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
        ),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    assert len(dps) == 1
    assert dps[0].datapoint_type == "shareholder_return"
    assert dps[0].priority == 90


def test_new_categories_do_not_bleed_into_each_other():
    result = _make_result(
        financial_highlights=[
            ExtractedFinancialHighlight(metric="Net income", value="7.6", page=58, quote="net income 7.6"),
        ],
        business_performance=[
            ExtractedBusinessPerformance(metric="Systems sold", value="418", page=55, quote="418 systems"),
        ],
        shareholder_returns=[
            ExtractedShareholderReturn(metric="Dividends paid", value="1.2", page=5, quote="dividends 1.2"),
        ],
    )
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    types = {d.datapoint_type for d in dps}
    assert types == {"financial_highlight", "business_performance", "shareholder_return"}


def test_category_order_preserved_in_combined_list():
    fin = _fin_dp()
    biz = _biz_dp()
    sh = _sh_dp()
    ds = _make_ds([fin, biz, sh])
    chunks = datapoints_to_chunks(ds)
    assert chunks[0].section_path == "Pre-extracted > Financial highlights"
    assert chunks[1].section_path == "Pre-extracted > Business performance"
    assert chunks[2].section_path == "Pre-extracted > Shareholder returns"


# ---------------------------------------------------------------------------
# Chunk IDs
# ---------------------------------------------------------------------------

def test_chunk_ids_are_unique_and_stable():
    dps = [_fte_dp(), _sust_dp()]
    chunks = datapoints_to_chunks(_make_ds(dps))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids[0] == "test.pdf:extracted:0"
    assert ids[1] == "test.pdf:extracted:1"


def test_section_path_by_type():
    dps = [_fte_dp(), _sust_dp()]
    chunks = datapoints_to_chunks(_make_ds(dps))
    assert "FTE" in chunks[0].section_path
    assert "Sustainability" in chunks[1].section_path


def test_page_fallback_to_one():
    dp = _fte_dp(page=None)
    chunks = datapoints_to_chunks(_make_ds([dp]))
    assert chunks[0].page == 1


# ---------------------------------------------------------------------------
# ESG datapoints
# ---------------------------------------------------------------------------

def _esg_dp(**kwargs) -> NormalizedDatapoint:
    defaults = dict(
        source="test.pdf", company="ACME", year=2025,
        datapoint_type="esg_datapoint",
        metric="Gross scope 1 and 2 GHG emissions",
        value="149 kt CO₂e", unit="kt CO₂e", period="2025",
        scope="Scope 1 and 2", page=160, priority=95,
        quote="Gross scope 1 and 2 GHG emissions: 149 kt CO₂e",
    )
    defaults.update(kwargs)
    return NormalizedDatapoint(**defaults)


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
        ExtractedESGDatapoint(metric="Scope 3 emissions", value="4,200 kt", period="2025", quote="Scope 3: 4,200 kt", page=161),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    esg = _norm(dps, "esg_datapoint")
    assert esg[0].priority == 95


def test_esg_priority_quote_only():
    result = _make_result(esg_datapoints=[
        ExtractedESGDatapoint(metric="Renewable electricity", value="85%", quote="85% renewable electricity", page=160),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    esg = _norm(dps, "esg_datapoint")
    assert esg[0].priority == 85


def test_esg_priority_no_quote_no_period():
    result = _make_result(esg_datapoints=[
        ExtractedESGDatapoint(metric="Water use", value="1,200 ML", page=160),
    ])
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    esg = _norm(dps, "esg_datapoint")
    assert esg[0].priority == 70


def test_esg_chunk_kind():
    chunks = datapoints_to_chunks(_make_ds([_esg_dp()]))
    assert chunks[0].chunk_kind == "extracted_datapoint"
    assert chunks[0].parser == "llamaextract"


def test_esg_section_path():
    chunks = datapoints_to_chunks(_make_ds([_esg_dp()]))
    assert "ESG" in chunks[0].section_path


def test_esg_text_format():
    chunks = datapoints_to_chunks(_make_ds([_esg_dp()]))
    text = chunks[0].text
    assert "Datapoint type: esg_datapoint" in text
    assert "Metric:" in text
    assert "Value:" in text
    assert "Scope:" in text
    assert "Quote:" in text
    assert "Period:" in text


def test_esg_embedding_text_has_synonyms():
    chunks = datapoints_to_chunks(_make_ds([_esg_dp()]))
    emb = chunks[0].embedding_text or ""
    assert "emissions" in emb
    assert "renewable" in emb
    assert "recycling" in emb


def test_sustainability_category_keeps_goals_and_esg():
    result = AnnualReportDatapoints(
        company="ACME", year=2025,
        sustainability_goals=[
            ExtractedSustainabilityGoal(goal="Achieve net-zero by 2040", target_year="2040", quote="net-zero by 2040", page=5),
        ],
        esg_datapoints=[
            ExtractedESGDatapoint(metric="Scope 1 emissions", value="50 kt", period="2025", quote="50 kt scope 1", page=10),
        ],
    )
    dps = normalize_llamaextract_result(result, source="test.pdf", company="ACME", year=2025)
    assert any(d.datapoint_type == "sustainability_goal" for d in dps)
    assert any(d.datapoint_type == "esg_datapoint" for d in dps)
