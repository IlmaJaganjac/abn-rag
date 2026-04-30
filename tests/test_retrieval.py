from __future__ import annotations

from typing import Any

from backend.app.retrieval import retrieve
from backend.app.schemas import RetrievalQuery


def _kind_in_where(where: dict[str, Any] | None, kind: str) -> bool:
    if where is None:
        return False
    if where.get("chunk_kind") == kind:
        return True
    return any(part.get("chunk_kind") == kind for part in where.get("$and", []))


class FakeHybridCollection:
    def get(self, *, where, include):
        if not _kind_in_where(where, "metric"):
            return {"ids": [], "documents": [], "metadatas": []}
        return {
            "ids": ["asml.pdf:54:8", "asml.pdf:5:2"],
            "documents": [
                (
                    "Metric: Total net sales\n"
                    "Period: 2025\n"
                    "Value: 32,667.3\n"
                    "Unit: €, in millions, except per share data"
                ),
                "| €32.7bn | Total net sales |",
            ],
            "metadatas": [
                {
                    "source": "asml.pdf",
                    "company": "ASML",
                    "year": 2025,
                    "page": 54,
                    "token_count": 18,
                    "chunk_kind": "metric",
                },
                {
                    "source": "asml.pdf",
                    "company": "ASML",
                    "year": 2025,
                    "page": 5,
                    "token_count": 11,
                    "chunk_kind": "metric",
                },
            ],
        }

    def query(self, **kwargs):
        return {
            "ids": [["asml.pdf:5:2"]],
            "documents": [["| €32.7bn | Total net sales |"]],
            "metadatas": [[
                {
                    "source": "asml.pdf",
                    "company": "ASML",
                    "year": 2025,
                    "page": 5,
                    "token_count": 11,
                    "chunk_kind": "metric",
                }
            ]],
            "distances": [[0.05]],
        }


class FakeNonMetricCollection(FakeHybridCollection):
    def query(self, **kwargs):
        return {
            "ids": [["asml.pdf:7:1"]],
            "documents": [["Christophe Fouquet, President, Chief Executive Officer"]],
            "metadatas": [[
                {
                    "source": "asml.pdf",
                    "company": "ASML",
                    "year": 2025,
                    "page": 7,
                    "token_count": 8,
                    "chunk_kind": "section",
                }
            ]],
            "distances": [[0.2]],
        }


def test_retrieve_boosts_exact_financial_metric_chunk(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.retrieval.get_collection", lambda: FakeHybridCollection())
    monkeypatch.setattr("backend.app.retrieval.embed_texts", lambda texts: [[0.1, 0.2]])

    result = retrieve(
        RetrievalQuery(
            question="What were ASML total net sales in million euros in 2025?",
            company="ASML",
            year=2025,
            top_k=12,
        )
    )

    assert result.chunks[0].id == "asml.pdf:54:8"
    assert "Metric: Total net sales" in result.chunks[0].text
    assert "Period: 2025" in result.chunks[0].text
    assert "Value: 32,667.3" in result.chunks[0].text
    assert "Unit: €, in millions, except per share data" in result.chunks[0].text
    assert [chunk.id for chunk in result.chunks].count("asml.pdf:5:2") == 1


def test_retrieve_keeps_vector_ranking_for_non_metric_question(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.retrieval.get_collection", lambda: FakeNonMetricCollection())
    monkeypatch.setattr("backend.app.retrieval.embed_texts", lambda texts: [[0.1, 0.2]])

    result = retrieve(
        RetrievalQuery(
            question="Who is ASML CEO?",
            company="ASML",
            year=2025,
            top_k=8,
        )
    )

    assert result.chunks[0].id == "asml.pdf:7:1"
