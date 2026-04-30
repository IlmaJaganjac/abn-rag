from __future__ import annotations

from backend.app.answer import (
    _fragments_in_order,
    _ground_citations,
    _normalize_for_grounding,
    _split_on_ellipsis,
)
from backend.app.schemas import Citation, RetrievedChunk


def _chunk(*, source: str, page: int, text: str, idx: int = 0) -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{source}:{page}:{idx}",
        source=source,
        company="ASML",
        year=2024,
        page=page,
        text=text,
        token_count=len(text.split()),
        score=0.9,
    )


def test_normalize_collapses_whitespace_and_lowercases() -> None:
    assert _normalize_for_grounding("  Hello   World\n\nFoo  ") == "hello world foo"


def test_normalize_folds_nbsp_via_nfkc() -> None:
    # U+00A0 between digits → regular space → collapsed
    assert _normalize_for_grounding("44\u00a0027") == "44 027"


def test_normalize_replaces_dash_variants() -> None:
    assert _normalize_for_grounding("2024\u20132025") == "2024-2025"
    assert _normalize_for_grounding("a\u2014b") == "a-b"


def test_normalize_replaces_unicode_ellipsis() -> None:
    assert _normalize_for_grounding("Total\u2026 44,027") == "total... 44,027"


def test_normalize_replaces_curly_quotes() -> None:
    assert _normalize_for_grounding("\u201chello\u201d") == '"hello"'


def test_split_on_ellipsis() -> None:
    assert _split_on_ellipsis("total ... 44,027") == ["total", "44,027"]
    assert _split_on_ellipsis("net sales increased by 0.7 billion") == [
        "net sales increased by 0.7 billion"
    ]
    assert _split_on_ellipsis("a....b....c") == ["a", "b", "c"]


def test_fragments_in_order_walks_cursor() -> None:
    hay = "total 39,086 42,416 44,027 less: temporary"
    assert _fragments_in_order(["total", "44,027"], hay) is True
    # reversed order must fail
    assert _fragments_in_order(["44,027", "total"], hay) is False


def test_ellipsis_quote_grounds_against_table() -> None:
    chunk = _chunk(
        source="asml-2024.pdf",
        page=358,
        text="Total\n\n39,086\n42,416\n44,027\nLess: Temporary employees",
    )
    cite = Citation(
        source="asml-2024.pdf",
        page=358,
        quote="Total ... 44,027",
    )
    grounded, drops, failure = _ground_citations([cite], [chunk])
    assert failure is None
    assert drops == []
    assert len(grounded) == 1
    assert grounded[0].source == "asml-2024.pdf"


def test_invented_source_is_rewritten_when_page_quote_match() -> None:
    chunk = _chunk(
        source="asml.pdf",
        page=5,
        text="Gross margin 52.8% — strong year for the company.",
    )
    cite = Citation(
        source="ASML Annual Report 2025",  # invented; comes from the page footer
        page=5,
        quote="Gross margin 52.8%",
    )
    grounded, drops, failure = _ground_citations([cite], [chunk])
    assert failure is None
    assert drops == []
    assert len(grounded) == 1
    assert grounded[0].source == "asml.pdf"  # rewritten from chunk
    assert grounded[0].page == 5


def test_nbsp_in_quote_grounds_against_regular_space() -> None:
    chunk = _chunk(
        source="asml.pdf",
        page=10,
        text="Total employees 44 027 worldwide",
    )
    # quote uses NBSP between the digits
    cite = Citation(source="asml.pdf", page=10, quote="44\u00a0027")
    grounded, drops, failure = _ground_citations([cite], [chunk])
    assert failure is None
    assert drops == []
    assert len(grounded) == 1


def test_en_dash_quote_grounds_against_hyphen_text() -> None:
    chunk = _chunk(
        source="asml.pdf",
        page=11,
        text="The 2024-2025 reporting period saw growth.",
    )
    cite = Citation(source="asml.pdf", page=11, quote="2024\u20132025 reporting period")
    grounded, _, failure = _ground_citations([cite], [chunk])
    assert failure is None
    assert len(grounded) == 1


def test_trivial_short_fragment_is_rejected() -> None:
    chunk = _chunk(
        source="asml.pdf",
        page=12,
        text="a quick brown fox jumps over a lazy dog",
    )
    cite = Citation(source="asml.pdf", page=12, quote="a ... b")
    grounded, drops, failure = _ground_citations([cite], [chunk])
    assert failure is not None
    assert grounded == []
    assert drops[0].reason == "quote_not_found_verbatim"


def test_page_not_in_retrieved_set_falls_back_to_quote_match() -> None:
    """If the LLM cited a wrong page but the quote matches another retrieved
    chunk, rewrite page+source to the matching chunk."""
    chunk = _chunk(source="asml.pdf", page=10, text="some unique sentence here")
    cite = Citation(source="asml.pdf", page=99, quote="unique sentence")
    grounded, drops, failure = _ground_citations([cite], [chunk])
    assert failure is None
    assert drops == []
    assert len(grounded) == 1
    assert grounded[0].page == 10  # rewritten


def test_quote_nowhere_with_unretrieved_page_uses_page_drop_reason() -> None:
    chunk = _chunk(source="asml.pdf", page=10, text="some text")
    cite = Citation(source="asml.pdf", page=99, quote="completely different text")
    grounded, drops, failure = _ground_citations([cite], [chunk])
    assert grounded == []
    assert failure == "no citations could be grounded in the retrieved chunks"
    assert drops[0].reason == "source_page_not_in_retrieved_set"


def test_fabricated_quote_is_dropped() -> None:
    chunk = _chunk(source="asml.pdf", page=5, text="Total net sales 28,262.9")
    cite = Citation(
        source="asml.pdf",
        page=5,
        quote="Net income was 99,999 billion which is impossible",
    )
    grounded, drops, failure = _ground_citations([cite], [chunk])
    assert grounded == []
    assert failure == "no citations could be grounded in the retrieved chunks"
    assert drops[0].reason == "quote_not_found_verbatim"


def test_non_verbatim_quote_can_repair_to_exact_chunk_text_when_tokens_match() -> None:
    chunk = _chunk(
        source="asml.pdf",
        page=145,
        text=(
            "Target: Commitment from our top-80% suppliers (based on CO₂e emissions) "
            "to reduce their CO₂e footprint by 2030 75% commitment from top 80% "
            "suppliers by 2026\nPerformance: 32%"
        ),
    )
    cite = Citation(
        source="asml.pdf",
        page=166,
        quote=(
            "Commitment from top-80% suppliers (based on CO₂e emissions) "
            "to reduce their CO₂e footprint by 2030 75% commitment from top 80% "
            "suppliers by 2026"
        ),
    )

    grounded, drops, failure = _ground_citations([cite], [chunk])

    assert failure is None
    assert drops == []
    assert len(grounded) == 1
    assert grounded[0].source == "asml.pdf"
    assert grounded[0].page == 145
    assert grounded[0].quote == chunk.text


def test_token_repair_does_not_ground_across_long_unrelated_span() -> None:
    filler = " ".join(f"filler{i}" for i in range(60))
    chunk = _chunk(
        source="asml.pdf",
        page=10,
        text=f"alpha {filler} beta gamma delta epsilon zeta eta theta",
    )
    cite = Citation(
        source="asml.pdf",
        page=10,
        quote="alpha beta gamma delta epsilon zeta eta theta",
    )

    grounded, drops, failure = _ground_citations([cite], [chunk])

    assert grounded == []
    assert failure == "no citations could be grounded in the retrieved chunks"
    assert drops[0].reason == "quote_not_found_verbatim"


def test_token_repair_does_not_drop_meaningful_qualifier() -> None:
    chunk = _chunk(
        source="asml.pdf",
        page=5,
        text="Metric: Total employees\nPeriod: 2025\nValue: more than 44,000\nUnit: FTEs",
    )
    cite = Citation(
        source="asml.pdf",
        page=5,
        quote="Metric: Total employees\nPeriod: 2025\nValue: 44,000\nUnit: FTEs",
    )

    grounded, drops, failure = _ground_citations([cite], [chunk])

    assert grounded == []
    assert failure == "no citations could be grounded in the retrieved chunks"
    assert drops[0].reason == "quote_not_found_verbatim"


def test_no_citations_returned_yields_failure() -> None:
    grounded, drops, failure = _ground_citations([], [])
    assert grounded == []
    assert drops == []
    assert failure == "no citations returned"


def test_empty_quote_after_normalization_is_dropped() -> None:
    chunk = _chunk(source="asml.pdf", page=5, text="real text here")
    cite = Citation(source="asml.pdf", page=5, quote="   ")
    grounded, drops, failure = _ground_citations([cite], [chunk])
    assert grounded == []
    assert failure is not None
    assert drops[0].reason == "empty_quote"


def test_mixed_grounded_and_dropped_citations() -> None:
    chunk = _chunk(source="asml.pdf", page=5, text="Net income €7,571.6 million")
    good = Citation(source="asml.pdf", page=5, quote="Net income \u20ac7,571.6 million")
    bad = Citation(source="asml.pdf", page=99, quote="this never appears anywhere")
    grounded, drops, failure = _ground_citations([good, bad], [chunk])
    assert failure is None
    assert len(grounded) == 1
    assert len(drops) == 1
    assert drops[0].reason == "source_page_not_in_retrieved_set"


def test_ellipsis_with_only_one_fragment_after_split() -> None:
    chunk = _chunk(source="asml.pdf", page=5, text="hello world done")
    # quote is just ellipses → no fragments → rejected
    cite = Citation(source="asml.pdf", page=5, quote="......")
    grounded, drops, failure = _ground_citations([cite], [chunk])
    assert grounded == []
    assert failure is not None


def test_page_match_picks_correct_source_when_multiple() -> None:
    chunk_a = _chunk(source="asml-2024.pdf", page=5, text="net sales 28.3 billion")
    chunk_b = _chunk(source="other.pdf", page=5, text="entirely different content here")
    cite = Citation(source="ASML Annual Report 2024", page=5, quote="28.3 billion")
    grounded, drops, failure = _ground_citations([cite], [chunk_a, chunk_b])
    assert failure is None
    assert len(grounded) == 1
    assert grounded[0].source == "asml-2024.pdf"


def test_prefers_cited_page_when_quote_appears_on_multiple_pages() -> None:
    chunk_a = _chunk(source="asml.pdf", page=7, text="Christophe Fouquet, CEO")
    chunk_b = _chunk(source="asml.pdf", page=78, text="Christophe Fouquet biography text")
    cite = Citation(source="asml.pdf", page=7, quote="Christophe Fouquet")
    grounded, _, failure = _ground_citations([cite], [chunk_a, chunk_b])
    assert failure is None
    assert grounded[0].page == 7  # cited page wins among ties
