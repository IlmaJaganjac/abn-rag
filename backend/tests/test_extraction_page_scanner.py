from __future__ import annotations

import pytest

from backend.app.extract.page_scanner import (
    _expand_with_context,
    _to_range_string,
    scan_pages,
    score_page,
    _score_financial_highlight,
    _score_business_performance,
    _score_shareholder_return,
)


# ---------------------------------------------------------------------------
# score_page — FTE
# ---------------------------------------------------------------------------

def test_fte_high_signal():
    text = "Total employees (FTEs) as of December 31 was 44,209."
    scores = score_page(text)
    assert scores["fte"] >= 5.0


def test_fte_low_signal_for_program_specific():
    text = "We allocated 12 dedicated FTEs to the program this quarter."
    scores = score_page(text)
    total_text = "Total employees (FTEs) 44,209. Average number of payroll employees: 43,267."
    assert scores["fte"] < score_page(total_text)["fte"]


def test_fte_program_specific_penalized():
    text = "dedicated FTEs team FTEs program FTEs"
    scores = score_page(text)
    total_text = "Total employees (FTEs) 44,209"
    assert scores["fte"] < score_page(total_text)["fte"]


def test_fte_broader_terms():
    text = "Internal employees: 23,126. Permanent employees: 21,885. Temporary employees: 1,241."
    score = score_page(text)["fte"]
    assert score >= 5.0


# ---------------------------------------------------------------------------
# score_page — sustainability
# ---------------------------------------------------------------------------

def test_sustainability_boosted_by_combo():
    text_combo = "Our target is to achieve net-zero GHG emissions by 2030."
    text_esg_only = "We track GHG emissions and renewable energy usage."
    assert score_page(text_combo)["sustainability"] > score_page(text_esg_only)["sustainability"]


def test_sustainability_future_year_boost():
    text_with_year = "We aim to reach net-zero by 2040 in line with scope 1 and 2 targets."
    text_no_year = "We aim to reach net-zero in line with scope 1 and 2 targets."
    assert score_page(text_with_year)["sustainability"] > score_page(text_no_year)["sustainability"]


def test_sustainability_sbti_terms():
    # SBTi + "targets" matches 3× target patterns → score 3.0 (no ESG terms so no combo bonus)
    text = "Our targets are aligned with the Science Based Targets initiative (SBTi) and 1.5°C pathway."
    score = score_page(text)["sustainability"]
    assert score >= 2.5


def test_sustainability_decarbonisation():
    # decarbonisation + transition plan + 2040 future year → score 3.5
    text = "Our decarbonisation transition plan covers all scopes by 2040."
    score = score_page(text)["sustainability"]
    assert score >= 3.0


# ---------------------------------------------------------------------------
# score_page — financial_highlight
# ---------------------------------------------------------------------------

def test_financial_highlight_at_a_glance():
    text = "At a glance\n€28.3bn Total net sales\n51.3% Gross margin\n€7.6bn Net income"
    score = _score_financial_highlight(text)
    assert score >= 5.0


def test_financial_highlight_financial_performance_title():
    text = "Financial performance\nNet sales: €28,262.9m\nR&D expense: €4,303.7m\nFree cash flow: €9,083.1m"
    score = _score_financial_highlight(text)
    assert score >= 5.0


def test_financial_highlight_broader_terms():
    text = "Operating income: €9,023m. EBITDA: €11,200m. Return on equity: 18.3%. CET1 ratio: 15.4%."
    score = _score_financial_highlight(text)
    assert score >= 4.0


def test_financial_highlight_low_for_plain_prose():
    text = "Our company continues to grow. We focus on customer relationships."
    score = _score_financial_highlight(text)
    assert score < 3.0


def test_financial_highlight_in_score_page():
    text = "Financial highlights\n€28.3bn Net sales\n51.3% Gross margin\nFree cash flow €9bn"
    scores = score_page(text)
    assert "financial_highlight" in scores
    assert scores["financial_highlight"] >= 5.0


# ---------------------------------------------------------------------------
# score_page — business_performance
# ---------------------------------------------------------------------------

def test_business_performance_detects_systems_sold():
    text = "Net system sales in units: 418. Number of suppliers: 5,150. Customer satisfaction: 86%."
    score = _score_business_performance(text)
    assert score >= 6.0


def test_business_performance_at_a_glance():
    text = "At a glance\n418 systems sold\n5,150 number of suppliers\n86% customer satisfaction"
    score = _score_business_performance(text)
    assert score >= 5.0


def test_business_performance_broader_terms():
    text = "Order intake: €36.2bn. Backlog: €44.5bn. Installed base: 6,200 active systems."
    score = _score_business_performance(text)
    assert score >= 3.0


def test_business_performance_banking_terms():
    text = "Loans outstanding: €180bn. Deposits: €220bn. Mortgages: €45bn."
    score = _score_business_performance(text)
    assert score >= 3.0


def test_business_performance_low_for_financial_only():
    text = "Net income was €7.6bn. Gross margin reached 51.3%. Free cash flow was €9bn."
    score = _score_business_performance(text)
    assert score < 3.0


def test_business_performance_in_score_page():
    text = "At a glance\n418 systems sold\nInstalled base 6,200\nOrder intake €36bn"
    scores = score_page(text)
    assert "business_performance" in scores
    assert scores["business_performance"] >= 5.0


# ---------------------------------------------------------------------------
# score_page — shareholder_return
# ---------------------------------------------------------------------------

def test_shareholder_return_detects_dividend():
    text = "We returned €3.0bn to shareholders through dividends and share buybacks."
    score = _score_shareholder_return(text)
    assert score >= 4.0


def test_shareholder_return_detects_repurchase():
    text = "Share repurchase programme: €2.5bn. Dividend per share: €6.40 proposed dividend."
    score = _score_shareholder_return(text)
    assert score >= 4.0


def test_shareholder_return_broader_terms():
    text = "Final dividend: €4.80. Interim dividend: €1.60. Payout ratio: 58%. Treasury shares cancelled: 200,000."
    score = _score_shareholder_return(text)
    assert score >= 8.0


def test_shareholder_return_low_for_fte_text():
    text = "Total employees (FTEs): 44,209. Average payroll employees: 43,267."
    score = _score_shareholder_return(text)
    assert score < 2.0


def test_shareholder_return_in_score_page():
    text = "We returned €3.0bn to shareholders through dividends of €1.80 per share and share buybacks."
    scores = score_page(text)
    assert "shareholder_return" in scores
    assert scores["shareholder_return"] > 0


# ---------------------------------------------------------------------------
# score_page — no kpi_highlights key
# ---------------------------------------------------------------------------

def test_score_page_does_not_contain_kpi_highlights():
    scores = score_page("At a glance\n€32.7bn Total net sales")
    assert "kpi_highlights" not in scores


def test_score_page_does_not_contain_operational_highlight():
    scores = score_page("Systems sold 418 units")
    assert "operational_highlight" not in scores


# ---------------------------------------------------------------------------
# Range helpers
# ---------------------------------------------------------------------------

def test_to_range_string_single():
    assert _to_range_string([5]) == "5"


def test_to_range_string_range():
    assert _to_range_string([5, 6, 7]) == "5-7"


def test_to_range_string_mixed():
    assert _to_range_string([5, 7, 8, 10]) == "5,7-8,10"


def test_to_range_string_empty():
    assert _to_range_string([]) == ""


def test_expand_with_context():
    all_pages = set(range(1, 20))
    result = _expand_with_context([5, 10], context=1, all_pages=all_pages)
    assert 4 in result
    assert 5 in result
    assert 6 in result
    assert 9 in result
    assert 10 in result
    assert 11 in result


def test_expand_clamps_to_available():
    all_pages = {3, 4, 5}
    result = _expand_with_context([3], context=2, all_pages=all_pages)
    assert 1 not in result
    assert 2 not in result
    assert 3 in result
    assert 4 in result
    assert 5 in result


# ---------------------------------------------------------------------------
# scan_pages integration
# ---------------------------------------------------------------------------

def _make_pages(entries: list[tuple[int, str]]) -> list[dict]:
    return [{"page": p, "text": t, "source": "test.pdf"} for p, t in entries]


def test_scan_detects_fte_page():
    pages = _make_pages([
        (1, "Introduction to our company."),
        (5, "Total employees (FTEs) as of December 31: 44,209. Average payroll FTEs: 43,267."),
        (10, "Revenue grew 15% year over year."),
    ])
    _, ranges = scan_pages(pages, context_window=0, max_pages_per_category=5)
    assert "5" in ranges["fte"]


def test_scan_detects_sustainability_page():
    pages = _make_pages([
        (1, "Our business overview."),
        (20, "We target net-zero GHG emissions by 2030. Scope 1 and 2 targets aligned with SBTi."),
        (30, "Financial statements."),
    ])
    _, ranges = scan_pages(pages, context_window=0, max_pages_per_category=5)
    assert "20" in ranges["sustainability"]


def test_scan_detects_financial_highlight_page():
    pages = _make_pages([
        (1, "Introduction."),
        (5, "Financial performance\n€28.3bn Total net sales\n51.3% Gross margin\n€7.6bn Net income\nR&D €4.3bn free cash flow"),
        (10, "Workforce overview."),
    ])
    _, ranges = scan_pages(pages, context_window=0, max_pages_per_category=5)
    assert "5" in ranges["financial_highlight"]


def test_scan_detects_business_performance_page():
    pages = _make_pages([
        (1, "Introduction."),
        (5, "At a glance\n418 systems sold\n5,150 number of suppliers\n86% customer satisfaction installed base"),
        (10, "Financials."),
    ])
    _, ranges = scan_pages(pages, context_window=0, max_pages_per_category=5)
    assert "5" in ranges["business_performance"]


def test_scan_detects_shareholder_return_page():
    pages = _make_pages([
        (1, "Introduction."),
        (5, "We returned €3.0bn to shareholders through dividends and share buybacks in 2024."),
        (10, "Financial statements."),
    ])
    _, ranges = scan_pages(pages, context_window=0, max_pages_per_category=5)
    assert "5" in ranges["shareholder_return"]


def test_scan_context_window_expands_range():
    pages = _make_pages([
        (1, "intro"),
        (5, "Total employees (FTEs) 44,209"),
        (6, "continued payroll details"),
        (7, "other content"),
    ])
    _, ranges = scan_pages(pages, context_window=1, max_pages_per_category=5)
    assert "4" not in ranges["fte"]
    assert "5" in ranges["fte"]
    assert "6" in ranges["fte"]


def test_scan_ranges_has_all_active_categories():
    pages = _make_pages([(1, "intro")])
    _, ranges = scan_pages(pages, context_window=0, max_pages_per_category=5)
    assert "fte" in ranges
    assert "sustainability" in ranges
    assert "financial_highlight" in ranges
    assert "business_performance" in ranges
    assert "shareholder_return" in ranges
    assert "kpi_highlights" not in ranges
    assert "operational_highlight" not in ranges
