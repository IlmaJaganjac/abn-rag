from __future__ import annotations

from types import SimpleNamespace

from backend.app.schemas import Citation, EvalQuestion, RetrievedChunk, VerbatimAnswer
from backend.evals.runner import _citation_grounded, _retrieval_hit, score_one


def _chunk(source: str, page: int, text: str = "evidence") -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{source}:{page}:0",
        source=source,
        company="ASML",
        year=2025,
        page=page,
        text=text,
        token_count=len(text.split()),
        score=1.0,
    )


def _answer(source: str, page: int, answer: str = "answer") -> VerbatimAnswer:
    return VerbatimAnswer(
        question="question",
        answer=answer,
        citations=[Citation(source=source, page=page, quote=answer)],
    )


def test_eval_page_matching_falls_back_to_expected_page() -> None:
    question = SimpleNamespace(expected_page=10, expected_source="asml.pdf")

    assert _retrieval_hit(question, [_chunk("asml.pdf", 11)]) is True
    assert _citation_grounded(question, _answer("asml.pdf", 9)) is True


def test_eval_page_matching_supports_accepted_pages() -> None:
    question = EvalQuestion(
        id="accepted_pages_example",
        question="question",
        category="verbatim_financial",
        difficulty="easy",
        expected_page=10,
        accepted_pages=[30, 40],
        expected_source="asml.pdf",
    )

    assert _retrieval_hit(question, [_chunk("asml.pdf", 39)]) is True
    assert _citation_grounded(question, _answer("asml.pdf", 31)) is True


def test_score_one_citation_fallback_only_uses_cited_chunks() -> None:
    question = SimpleNamespace(
        expected_behavior=None,
        expected_answer_contains_any=["needle"],
        expected_answer_contains_all=None,
        expected_page=10,
        expected_source="asml.pdf",
    )
    answer = _answer("asml.pdf", 50, "needle")
    chunks = [
        _chunk("asml.pdf", 50, "this cited chunk contains needle"),
        _chunk("asml.pdf", 10, "expected page without the answer"),
    ]

    passed, reasons, _, answer_correct, citation_grounded = score_one(
        question,
        answer,
        chunks,
    )

    assert passed is True
    assert answer_correct is True
    assert citation_grounded is True
    assert not any("no citation page within" in reason for reason in reasons)
