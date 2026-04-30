from __future__ import annotations

import pytest

from backend.app.schemas import Citation, EvalQuestion, GroundingDrop, RetrievedChunk, VerbatimAnswer
from evals.runner import build_diagnostics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _q(
    *,
    expected_answer_contains_any: list[str] | None = None,
    expected_answer_contains_all: list[str] | None = None,
    expected_page: int | None = None,
    expected_source: str | None = None,
    expected_behavior: str | None = None,
    category: str = "pre_extracted_fact",
) -> EvalQuestion:
    return EvalQuestion(
        id="test_q",
        question="Test question?",
        category=category,  # type: ignore[arg-type]
        difficulty="easy",
        expected_answer_contains_any=expected_answer_contains_any,
        expected_answer_contains_all=expected_answer_contains_all,
        expected_page=expected_page,
        expected_source=expected_source,
        expected_behavior=expected_behavior,  # type: ignore[arg-type]
    )


def _ans(
    *,
    answer: str = "The answer is 44,209.",
    verbatim: str | None = None,
    refused: bool = False,
    refusal_reason: str | None = None,
    citations: list[Citation] | None = None,
    grounding_drops: list[GroundingDrop] | None = None,
    raw_evidence: list | None = None,
    raw_citations: list | None = None,
) -> VerbatimAnswer:
    if not refused and citations is None:
        citations = [Citation(source="asml.pdf", page=5, quote="some quote")]
    return VerbatimAnswer(
        question="Test question?",
        answer=answer,
        verbatim=verbatim,
        refused=refused,
        refusal_reason=refusal_reason,
        citations=citations or [],
        evidence=[],
        raw_evidence=raw_evidence or [],
        raw_citations=raw_citations or [],
        grounding_drops=grounding_drops or [],
    )


def _chunk(*, page: int, text: str, source: str = "asml.pdf") -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{source}:{page}:0",
        source=source,
        company="ASML",
        year=2025,
        page=page,
        text=text,
        token_count=len(text.split()),
        score=0.9,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_diagnosis_pass() -> None:
    q = _q(expected_answer_contains_any=["44,209"], expected_page=5, expected_source="asml.pdf")
    ans = _ans(
        answer="ASML had 44,209 employees.",
        citations=[Citation(source="asml.pdf", page=5, quote="44,209")],
    )
    chunks = [_chunk(page=5, text="Total employees: 44,209")]
    d = build_diagnostics(q, ans, chunks, passed=True)
    assert d.diagnosis == "pass"
    assert d.value_correct is True
    assert d.citation_page_correct is True


def test_diagnosis_expected_refusal_pass() -> None:
    q = _q(expected_behavior="refuse", category="hallucination_check")
    ans = _ans(refused=True, refusal_reason="not in corpus", citations=[])
    d = build_diagnostics(q, ans, [], passed=True)
    assert d.diagnosis == "expected_refusal_pass"
    assert d.expected_refusal is True
    assert d.answer_refused is True


def test_diagnosis_answer_refused_unexpectedly() -> None:
    q = _q(expected_answer_contains_any=["44,209"])
    ans = _ans(refused=True, refusal_reason="no retrieved context", citations=[])
    chunks = [_chunk(page=5, text="Total employees: 44,209")]
    d = build_diagnostics(q, ans, chunks, passed=False)
    assert d.diagnosis == "answer_refused_unexpectedly"
    assert d.answer_refused is True
    assert d.expected_refusal is False


def test_diagnosis_grounding_failed() -> None:
    q = _q(expected_answer_contains_any=["44,209"])
    ans = _ans(
        refused=True,
        refusal_reason="ungrounded evidence: no evidence could be grounded",
        citations=[],
    )
    d = build_diagnostics(q, ans, [], passed=False)
    assert d.diagnosis == "grounding_failed"


def test_diagnosis_retrieval_miss() -> None:
    # Expected value not present in any retrieved chunk
    q = _q(expected_answer_contains_any=["44,209"], expected_page=301, expected_source="asml.pdf")
    ans = _ans(
        answer="ASML had many employees.",
        citations=[Citation(source="asml.pdf", page=5, quote="many employees")],
    )
    chunks = [_chunk(page=5, text="At a glance: highlights only, no exact figure")]
    d = build_diagnostics(q, ans, chunks, passed=False)
    assert d.diagnosis == "retrieval_miss"
    assert d.expected_value_in_chunks is False
    assert d.value_correct is False
    assert d.expected_page_retrieved is False


def test_diagnosis_answer_value_wrong() -> None:
    # Expected value IS in retrieved chunks but NOT in the answer
    q = _q(expected_answer_contains_any=["44,209"])
    ans = _ans(
        answer="ASML had approximately 43,000 employees.",
        citations=[Citation(source="asml.pdf", page=5, quote="43,000 employees")],
    )
    chunks = [_chunk(page=5, text="Total employees: 44,209 in the full table")]
    d = build_diagnostics(q, ans, chunks, passed=False)
    assert d.diagnosis == "answer_value_wrong"
    assert d.expected_value_in_chunks is True
    assert d.value_correct is False


def test_diagnosis_citation_source_wrong() -> None:
    q = _q(
        expected_answer_contains_any=["44,209"],
        expected_source="asml.pdf",
        expected_page=5,
    )
    ans = _ans(
        answer="ASML had 44,209 employees.",
        citations=[Citation(source="wrong-report.pdf", page=5, quote="44,209")],
    )
    chunks = [_chunk(page=5, text="Total employees: 44,209")]
    d = build_diagnostics(q, ans, chunks, passed=False)
    assert d.diagnosis == "citation_source_wrong"
    assert d.citation_source_correct is False
    assert d.value_correct is True


def test_diagnosis_citation_page_wrong_lm_error() -> None:
    # Value correct, source correct, cited page is wrong AND cited chunk lacks the value
    q = _q(
        expected_answer_contains_any=["44,209"],
        expected_source="asml.pdf",
        expected_page=301,
    )
    # LLM cited page 5, but page 5 chunk doesn't contain "44,209"
    ans = _ans(
        answer="ASML had 44,209 employees.",
        citations=[Citation(source="asml.pdf", page=5, quote="44,209")],
    )
    chunks = [
        _chunk(page=5, text="This is a highlights page with no exact figure"),
        _chunk(page=301, text="Total FTE: 44,209"),
    ]
    d = build_diagnostics(q, ans, chunks, passed=False)
    assert d.diagnosis == "citation_page_wrong"
    assert d.value_correct is True
    assert d.citation_page_correct is False


def test_diagnosis_eval_expected_page_maybe_stale() -> None:
    # Value correct, source correct, cited page is wrong BUT cited chunk DOES contain the value
    # → the expected_page in the YAML may be stale
    q = _q(
        expected_answer_contains_any=["44,209"],
        expected_source="asml.pdf",
        expected_page=301,
    )
    # LLM cited page 200 which actually contains the value
    ans = _ans(
        answer="ASML had 44,209 employees.",
        citations=[Citation(source="asml.pdf", page=200, quote="44,209")],
    )
    chunks = [_chunk(page=200, text="Full headcount: 44,209 employees total")]
    d = build_diagnostics(q, ans, chunks, passed=False)
    assert d.diagnosis == "eval_expected_page_maybe_stale"
    assert d.value_correct is True
    assert d.citation_page_correct is False


def test_diagnosis_unknown_failure_for_expected_refusal_violated() -> None:
    # Expected refusal but got an answer — no specific code, falls to unknown_failure
    q = _q(expected_behavior="refuse", category="hallucination_check")
    ans = _ans(
        answer="The answer is 99,999.",
        citations=[Citation(source="asml.pdf", page=5, quote="99,999")],
    )
    d = build_diagnostics(q, ans, [], passed=False)
    assert d.diagnosis == "unknown_failure"
    assert d.expected_refusal is True
    assert d.answer_refused is False


def test_diagnosis_unknown_failure_for_exception() -> None:
    q = _q(expected_answer_contains_any=["44,209"])
    d = build_diagnostics(q, None, [], passed=False)
    assert d.diagnosis == "unknown_failure"


def test_grounding_signals_populated() -> None:
    drop = GroundingDrop(
        source="asml.pdf",
        page=5,
        quote="not found",
        reason="quote_not_found_verbatim",
    )
    ans = _ans(
        refused=True,
        refusal_reason="ungrounded evidence: no evidence returned",
        citations=[],
        grounding_drops=[drop],
    )
    q = _q(expected_answer_contains_any=["44,209"])
    d = build_diagnostics(q, ans, [], passed=False)
    assert d.grounding_failure_reasons == ["quote_not_found_verbatim"]
    assert d.has_final_citation is False
    assert d.diagnosis == "grounding_failed"


def test_retrieval_signals_expected_page_found() -> None:
    q = _q(expected_answer_contains_any=["52.8%"], expected_page=5, expected_source="asml.pdf")
    ans = _ans(
        answer="Gross margin was 52.8%.",
        citations=[Citation(source="asml.pdf", page=5, quote="52.8%")],
    )
    chunks = [_chunk(page=5, text="Gross margin 52.8% — strong year")]
    d = build_diagnostics(q, ans, chunks, passed=True)
    assert d.expected_page_retrieved is True
    assert d.expected_value_in_chunks is True
    assert d.diagnosis == "pass"


def test_value_correct_requires_both_any_and_all() -> None:
    # Question requires contains_any AND contains_all both satisfied
    q = _q(
        expected_answer_contains_any=["75%"],
        expected_answer_contains_all=["2026"],
    )
    # Answer has "2026" but not "75%" — value_correct should be False
    ans = _ans(
        answer="Target deadline is 2026.",
        citations=[Citation(source="asml.pdf", page=5, quote="2026")],
    )
    chunks = [_chunk(page=5, text="75% of suppliers by 2026")]
    d = build_diagnostics(q, ans, chunks, passed=False)
    assert d.value_correct is False
    assert d.expected_value_in_chunks is True
    assert d.diagnosis == "answer_value_wrong"


def test_no_needle_constraints_value_correct_true() -> None:
    # Question with no value constraints: value_correct=True when answer not refused
    q = _q()
    ans = _ans(answer="The CEO is Christophe Fouquet.")
    d = build_diagnostics(q, ans, [], passed=True)
    assert d.value_correct is True
    assert d.diagnosis == "pass"
