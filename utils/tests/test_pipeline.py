from __future__ import annotations

import pytest

from backend.app.pipeline import _company_mismatch_refusal, answer_with_context


def test_company_mismatch_refuses_before_retrieval(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_retrieve(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("retrieve should not run for company mismatch")

    monkeypatch.setattr("backend.app.pipeline.retrieve", fail_retrieve)

    result = answer_with_context(
        "What was Tesla revenue in 2025?",
        top_k=5,
        company="ASML",
        year=2025,
    )

    assert result.retrieved_chunks == []
    assert result.answer.refused is True
    assert (
        result.answer.refusal_reason
        == "Question mentions TESLA, but the active company filter is ASML."
    )


@pytest.mark.parametrize(
    ("question", "company"),
    [
        ("What were ASML total net sales in 2025?", "ASML"),
        ("What was ABN AMRO's CET1 ratio in 2025?", "ABN AMRO"),
        ("How many employees did Shell have in 2025?", "SHELL"),
        ("What was the dividend in 2025?", "ASML"),
    ],
)
def test_company_mismatch_guard_allows_matching_or_unmentioned_company(
    question: str,
    company: str,
) -> None:
    assert _company_mismatch_refusal(question, company) is None
