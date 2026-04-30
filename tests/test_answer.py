from __future__ import annotations

from types import SimpleNamespace

from backend.app.answer import (
    SYSTEM_PROMPT,
    _fragments_in_order,
    _ground_citations,
    _normalize_for_grounding,
    _split_on_ellipsis,
    answer_question,
)
from backend.app.schemas import Citation, LLMAnswer, RetrievedChunk


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


class _FakeCompletions:
    def __init__(self, parsed: LLMAnswer) -> None:
        self.parsed = parsed
        self.messages = None

    def parse(self, **kwargs):  # noqa: ANN003
        self.messages = kwargs["messages"]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=self.parsed))]
        )


def _answer_with_fake_llm(
    monkeypatch,
    *,
    question: str,
    chunks: list[RetrievedChunk],
    parsed: LLMAnswer,
) -> tuple[SimpleNamespace, object]:
    completions = _FakeCompletions(parsed)
    fake_client = SimpleNamespace(
        beta=SimpleNamespace(
            chat=SimpleNamespace(
                completions=completions,
            )
        )
    )
    monkeypatch.setattr("backend.app.answer.openai_client", lambda: fake_client)
    return answer_question(question, chunks), completions


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


def test_answer_prompt_tells_rd_spend_to_choose_monetary_unit() -> None:
    assert "For spend, cost, expense, R&D spend, or capex questions" in SYSTEM_PROMPT
    assert "Never answer" in SYSTEM_PROMPT
    assert "FTE" in SYSTEM_PROMPT


def test_rd_spend_answer_uses_monetary_chunk_over_fte(monkeypatch) -> None:
    fte = _chunk(
        source="asml.pdf",
        page=301,
        text="Metric: Research and Development\nPeriod: 2025\nValue: 16,448\nUnit: in FTE",
        idx=1,
    )
    money = _chunk(
        source="asml.pdf",
        page=30,
        text="Metric: R&D Investment\nPeriod: 2025\nValue: 4.7\nUnit: in € billions",
        idx=2,
    )
    parsed = LLMAnswer(
        answer="ASML spent €4.7 billion on R&D in 2025.",
        verbatim="4.7",
        citations=[Citation(source="asml.pdf", page=30, quote=money.text)],
    )

    answer, completions = _answer_with_fake_llm(
        monkeypatch,
        question="How much did ASML spend financially on Research and Development in 2025?",
        chunks=[fte, money],
        parsed=parsed,
    )

    assert answer.refused is False
    assert answer.citations[0].page == 30
    assert answer.verbatim == "4.7"
    assert "Never answer" in completions.messages[0]["content"]
    assert "FTE" in completions.messages[0]["content"]


def test_average_payroll_fte_answer_uses_average_payroll_chunk(monkeypatch) -> None:
    average = _chunk(
        source="asml.pdf",
        page=131,
        text="Metric: Average number of payroll employees in FTEs\nPeriod: 2025\nValue: 43,267",
        idx=1,
    )
    year_end = _chunk(
        source="asml.pdf",
        page=301,
        text="Metric: Payroll employees\nPeriod: 2025\nValue: 43,520\nUnit: in FTE",
        idx=2,
    )
    parsed = LLMAnswer(
        answer="ASML's average number of payroll employees in FTEs was 43,267.",
        verbatim="43,267",
        citations=[Citation(source="asml.pdf", page=131, quote=average.text)],
    )

    answer, completions = _answer_with_fake_llm(
        monkeypatch,
        question=(
            "In ASML's pay-ratio/remuneration table, what was the average number "
            "of payroll employees in FTEs for 2025?"
        ),
        chunks=[average, year_end],
        parsed=parsed,
    )

    assert answer.refused is False
    assert answer.citations[0].page == 131
    assert answer.verbatim == "43,267"
    assert "preserve average vs year-end" in completions.messages[0]["content"]


def test_gross_margin_answer_uses_explicit_margin_percentage(monkeypatch) -> None:
    unrelated = _chunk(
        source="asml.pdf",
        page=199,
        text="Metric: Percentage of non-recycled waste\nPeriod: 2025\nValue: 26.3%",
        idx=1,
    )
    margin = _chunk(
        source="asml.pdf",
        page=53,
        text="Metric: Gross margin\nPeriod: 2025\nValue: 52.8\nUnit: %",
        idx=2,
    )
    parsed = LLMAnswer(
        answer="ASML's reported gross margin in 2025 was 52.8%.",
        verbatim="52.8",
        citations=[Citation(source="asml.pdf", page=53, quote=margin.text)],
    )

    answer, completions = _answer_with_fake_llm(
        monkeypatch,
        question="What reported gross margin percentage did ASML state for 2025?",
        chunks=[unrelated, margin],
        parsed=parsed,
    )

    assert answer.refused is False
    assert answer.citations[0].page == 53
    assert answer.verbatim == "52.8"
    assert "Do not calculate" in completions.messages[0]["content"]
    assert "unless the user asks to calculate" in completions.messages[0]["content"]


def test_spend_question_refuses_when_only_incompatible_unit_available(monkeypatch) -> None:
    fte = _chunk(
        source="asml.pdf",
        page=301,
        text="Metric: Research and Development\nPeriod: 2025\nValue: 16,448\nUnit: in FTE",
    )
    parsed = LLMAnswer(
        answer="The answer is not available in the provided reports.",
        refused=True,
        refusal_reason="Only an FTE metric is available, not monetary spend.",
    )

    answer, completions = _answer_with_fake_llm(
        monkeypatch,
        question="How much did ASML spend financially on Research and Development in 2025?",
        chunks=[fte],
        parsed=parsed,
    )

    assert answer.refused is True
    assert answer.citations == []
    assert "If label and unit do not clearly match the question" in completions.messages[0]["content"]


def test_cross_company_question_is_refused(monkeypatch) -> None:
    asml_chunk = _chunk(
        source="asml.pdf",
        page=5,
        text="Metric: Total net sales\nPeriod: 2025\nValue: €32.7bn\nUnit: EUR billion",
    )
    parsed = LLMAnswer(
        answer="The answer is not available in the provided reports.",
        refused=True,
        refusal_reason="question is about a different company than the retrieved reports",
    )

    answer, completions = _answer_with_fake_llm(
        monkeypatch,
        question="What was Intel's total net sales in 2025?",
        chunks=[asml_chunk],
        parsed=parsed,
    )

    assert answer.refused is True
    assert answer.citations == []
    assert (
        answer.refusal_reason
        == "question is about a different company than the retrieved reports"
    )
    assert "Active company in context" in completions.messages[1]["content"]


def test_active_company_question_is_not_refused(monkeypatch) -> None:
    chunk = _chunk(
        source="asml-2024.pdf",
        page=6,
        text="| 5,150 | Total suppliers |",
    )
    parsed = LLMAnswer(
        answer="ASML had 5,150 total suppliers in 2024.",
        verbatim="5,150",
        citations=[
            Citation(source="asml-2024.pdf", page=6, quote="| 5,150 | Total suppliers |")
        ],
        refused=False,
    )

    answer, _completions = _answer_with_fake_llm(
        monkeypatch,
        question="How many total suppliers does ASML have in 2024?",
        chunks=[chunk],
        parsed=parsed,
    )

    assert answer.refused is False
    assert any(c.source == "asml-2024.pdf" for c in answer.citations)
