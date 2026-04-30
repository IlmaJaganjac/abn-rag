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


class FakeCollection:
    def get(self, *, where, include):
        if _kind_in_where(where, "metric"):
            return {
                "ids": ["asml.pdf:5:7"],
                "documents": ["535\nSystem sales in units"],
                "metadatas": [
                    {
                        "source": "asml.pdf",
                        "company": "ASML",
                        "year": 2025,
                        "page": 5,
                        "token_count": 5,
                        "chunk_kind": "metric",
                        "metric_name": "System sales in units",
                        "metric_value": "535",
                        "metric_period": "2025",
                    }
                ],
            }
        return {"ids": [], "documents": [], "metadatas": []}

    def query(self, **kwargs):
        return {
            "ids": [["asml.pdf:191:2"]],
            "documents": [["27 refurbished lithography systems"]],
            "metadatas": [[
                {
                    "source": "asml.pdf",
                    "company": "ASML",
                    "year": 2025,
                    "page": 191,
                    "token_count": 4,
                    "chunk_kind": "section",
                }
            ]],
            "distances": [[0.3]],
        }


class FakeFteCollection:
    def get(self, *, where, include):
        if not _kind_in_where(where, "datapoint"):
            return {"ids": [], "documents": [], "metadatas": []}
        text = (
            "Average number of payroll employees in FTEs\n"
            "2023\n2024\n2025\nWorldwide (including Netherlands)\n38,805\n41,697\n43,267"
        )
        return {
            "ids": ["asml.pdf:301:9", "asml.pdf:131:6"],
            "documents": [text, text],
            "metadatas": [
                {
                    "source": "asml.pdf",
                    "company": "ASML",
                    "year": 2025,
                    "page": 301,
                    "token_count": 20,
                    "chunk_kind": "datapoint",
                },
                {
                    "source": "asml.pdf",
                    "company": "ASML",
                    "year": 2025,
                    "page": 131,
                    "token_count": 20,
                    "chunk_kind": "datapoint",
                },
            ],
        }

    def query(self, **kwargs):
        return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}


def test_retrieve_boosts_matching_metric_chunk(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.retrieval.get_collection", lambda: FakeCollection())
    monkeypatch.setattr("backend.app.retrieval.embed_texts", lambda texts: [[0.1, 0.2]])

    result = retrieve(
        RetrievalQuery(
            question="How many lithography systems did ASML sell in 2025?",
            company="ASML",
            year=2025,
            top_k=8,
        )
    )

    assert result.chunks[0].id == "asml.pdf:5:7"
    assert result.chunks[0].metric_name == "System sales in units"


def test_retrieve_prefers_earlier_exact_datapoint_page(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.retrieval.get_collection", lambda: FakeFteCollection())
    monkeypatch.setattr("backend.app.retrieval.embed_texts", lambda texts: [[0.1, 0.2]])

    result = retrieve(
        RetrievalQuery(
            question="What was ASML's average number of payroll employees in FTEs for 2025?",
            company="ASML",
            year=2025,
            top_k=8,
        )
    )

    assert result.chunks[0].page == 131
