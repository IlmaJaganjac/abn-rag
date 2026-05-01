from __future__ import annotations

import pytest

from backend.app.extraction_page_scanner import (
    _expand_with_context,
    _to_range_string,
    scan_pages,
    score_page,
)


# ---------------------------------------------------------------------------
# score_page unit tests
# ---------------------------------------------------------------------------

def test_fte_high_signal():
    text = "Total employees (FTEs) as of December 31 was 44,209."
    scores = score_page(text)
    assert scores["fte"] >= 5.0


def test_fte_low_signal_for_program_specific():
    text = "We allocated 12 dedicated FTEs to the program this quarter."
    scores = score_page(text)
    # Should score lower than a total-employees page
    total_text = "Total employees (FTEs) 44,209. Average number of payroll employees: 43,267."
    assert scores["fte"] < score_page(total_text)["fte"]


def test_fte_program_specific_penalized():
    text = "dedicated FTEs team FTEs program FTEs"
    scores = score_page(text)
    total_text = "Total employees (FTEs) 44,209"
    assert scores["fte"] < score_page(total_text)["fte"]


def test_sustainability_boosted_by_combo():
    text_combo = "Our target is to achieve net-zero GHG emissions by 2030."
    text_esg_only = "We track GHG emissions and renewable energy usage."
    assert score_page(text_combo)["sustainability"] > score_page(text_esg_only)["sustainability"]


def test_sustainability_future_year_boost():
    text_with_year = "We aim to reach net-zero by 2040 in line with scope 1 and 2 targets."
    text_no_year = "We aim to reach net-zero in line with scope 1 and 2 targets."
    assert score_page(text_with_year)["sustainability"] > score_page(text_no_year)["sustainability"]


def test_kpi_highlights_at_a_glance():
    text = "At a glance\n€32.7bn Total net sales\n52.8% Gross margin\n535 System sales in units"
    scores = score_page(text)
    assert scores["kpi_highlights"] >= 8.0


def test_kpi_plain_prose_low_score():
    text = "The company focuses on customer relationships and operational excellence."
    scores = score_page(text)
    assert scores["kpi_highlights"] < 3.0


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


def test_scan_detects_kpi_page():
    pages = _make_pages([
        (1, "Introduction."),
        (5, "At a glance\n€32.7bn Total net sales\n52.8% Gross margin\n535 System sales in units"),
        (10, "Detailed financials."),
    ])
    _, ranges = scan_pages(pages, context_window=0, max_pages_per_category=5)
    assert "5" in ranges["kpi_highlights"]


def test_scan_context_window_expands_range():
    pages = _make_pages([
        (1, "intro"),
        (5, "Total employees (FTEs) 44,209"),
        (6, "continued payroll details"),
        (7, "other content"),
    ])
    _, ranges = scan_pages(pages, context_window=1, max_pages_per_category=5)
    # p4 doesn't exist but p5 and p6 should be in range
    assert "4" not in ranges["fte"]
    assert "5" in ranges["fte"]
    assert "6" in ranges["fte"]
