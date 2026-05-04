from __future__ import annotations

import pytest

from backend.app.pipeline import answer_with_context
from backend.app.schemas import RetrievalQuery, RetrievalResult, RetrievedChunk, VerbatimAnswer


def test_answer_with_context_runs_retrieval_and_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_query: RetrievalQuery | None = None
    chunk = RetrievedChunk(
        id="asml.pdf:1:0",
        source="asml.pdf",
        company="ASML",
        year=2025,
        page=1,
        text="Revenue was 32.7 billion.",
        token_count=4,
        score=0.9,
    )

    def fake_retrieve(query: RetrievalQuery) -> RetrievalResult:
        nonlocal captured_query
        captured_query = query
        return RetrievalResult(query=query, chunks=[chunk])

    def fake_answer(question: str, chunks: list[RetrievedChunk]) -> VerbatimAnswer:
        assert question == "What was ASML revenue in 2025?"
        assert chunks == [chunk]
        return VerbatimAnswer(
            question=question,
            answer="ASML revenue was 32.7 billion.",
            verbatim="32.7 billion",
            citations=[],
            refused=True,
            refusal_reason="stubbed in test",
        )

    monkeypatch.setattr("backend.app.pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("backend.app.pipeline.answer_question", fake_answer)

    result = answer_with_context(
        "What was ASML revenue in 2025?",
        top_k=5,
        company="ASML",
        year=2025,
    )

    assert captured_query is not None
    assert captured_query.question == "What was ASML revenue in 2025?"
    assert captured_query.company == "ASML"
    assert captured_query.year == 2025
    assert captured_query.top_k == 5
    assert result.retrieved_chunks == [chunk]
    assert result.answer.answer == "ASML revenue was 32.7 billion."
