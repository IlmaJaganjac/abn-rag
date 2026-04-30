from __future__ import annotations

from typing import Any

from backend.app.answer import _ground_citations
from backend.app.retrieval import retrieve
from backend.app.schemas import Citation, RetrievalQuery, RetrievedChunk


def _kind_in_where(where: dict[str, Any] | None, kind: str) -> bool:
    if where is None:
        return False
    if where.get("chunk_kind") == kind:
        return True
    return any(part.get("chunk_kind") == kind for part in where.get("$and", []))


class FakeHybridCollection:
    metric_docs = [
        (
            "abn-amro-2025.pdf:metric:cet1",
            "Metric: Common Equity Tier 1 (CET1) ratio\nPeriod: 2025\nValue: 15.4%\nUnit: %",
            120,
        ),
        (
            "abn-amro-2025.pdf:metric:fte",
            "Metric: Number of internal employees\nPeriod: 2025\nValue: 23,126\nUnit: FTEs",
            220,
        ),
        (
            "abn-amro-2025.pdf:metric:profit",
            "Metric: Profit/(loss) for the period\nPeriod: 2025\nValue: 2,252\nUnit: EUR million",
            88,
        ),
        (
            "abn-amro-2025.pdf:metric:kpi",
            "| €32.7bn | Total net sales |",
            5,
        ),
    ]

    def get(self, *, where, include):
        if not _kind_in_where(where, "metric"):
            return {"ids": [], "documents": [], "metadatas": []}
        return {
            "ids": [item[0] for item in self.metric_docs],
            "documents": [item[1] for item in self.metric_docs],
            "metadatas": [
                {
                    "source": "abn-amro-2025.pdf",
                    "company": "ABN AMRO",
                    "year": 2025,
                    "page": item[2],
                    "token_count": 12,
                    "chunk_kind": "metric",
                }
                for item in self.metric_docs
            ],
        }

    def query(self, **kwargs):
        return {
            "ids": [["abn-amro-2025.pdf:dense:generic"]],
            "documents": [["Generic annual report text"]],
            "metadatas": [[
                {
                    "source": "abn-amro-2025.pdf",
                    "company": "ABN AMRO",
                    "year": 2025,
                    "page": 10,
                    "token_count": 4,
                    "chunk_kind": "section",
                }
            ]],
            "distances": [[0.2]],
        }


def _chunk(cid: str, text: str, page: int, kind: str = "section") -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        source="abn-amro-2025.pdf",
        company="ABN AMRO",
        year=2025,
        page=page,
        text=text,
        token_count=len(text.split()),
        chunk_kind=kind,
        score=1.0,
    )


def test_retrieve_uses_bm25_for_ceo_text_question(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.retrieval.get_collection", lambda: FakeHybridCollection())
    monkeypatch.setattr("backend.app.retrieval.embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(
        "backend.app.retrieval._bm25_candidates",
        lambda query: [
            _chunk(
                "abn-amro-2025.pdf:ceo",
                "Marguerite Bérard was appointed Chief Executive Officer.",
                14,
            )
        ],
    )

    result = retrieve(
        RetrievalQuery(
            question="Who was ABN AMRO's CEO in 2025?",
            company="ABN AMRO",
            year=2025,
            top_k=12,
        )
    )

    assert result.chunks[0].id == "abn-amro-2025.pdf:ceo"
    assert "Chief Executive Officer" in result.chunks[0].text


def test_retrieve_cet1_metric_for_numeric_query(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.retrieval.get_collection", lambda: FakeHybridCollection())
    monkeypatch.setattr("backend.app.retrieval.embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr("backend.app.retrieval._bm25_candidates", lambda query: [])

    result = retrieve(
        RetrievalQuery(
            question="What was ABN AMRO's CET1 ratio in 2025?",
            company="ABN AMRO",
            year=2025,
            top_k=12,
        )
    )

    assert result.chunks[0].id == "abn-amro-2025.pdf:metric:cet1"
    assert "Metric: Common Equity Tier 1 (CET1) ratio" in result.chunks[0].text
    assert "Value: 15.4%" in result.chunks[0].text


def test_retrieve_fte_metric_and_metric_quote_grounds(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.retrieval.get_collection", lambda: FakeHybridCollection())
    monkeypatch.setattr("backend.app.retrieval.embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr("backend.app.retrieval._bm25_candidates", lambda query: [])

    result = retrieve(
        RetrievalQuery(
            question="How many internal employees in FTEs did ABN AMRO have in 2025?",
            company="ABN AMRO",
            year=2025,
            top_k=12,
        )
    )

    quote = "Metric: Number of internal employees\nPeriod: 2025\nValue: 23,126\nUnit: FTEs"
    assert result.chunks[0].id == "abn-amro-2025.pdf:metric:fte"
    assert quote in result.chunks[0].text
    grounded, drops, failure = _ground_citations(
        [Citation(source="abn-amro-2025.pdf", page=result.chunks[0].page, quote=quote)],
        result.chunks,
    )
    assert len(grounded) == 1
    assert drops == []
    assert failure is None


def test_retrieve_net_profit_metric_for_numeric_query(monkeypatch) -> None:
    monkeypatch.setattr("backend.app.retrieval.get_collection", lambda: FakeHybridCollection())
    monkeypatch.setattr("backend.app.retrieval.embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr("backend.app.retrieval._bm25_candidates", lambda query: [])

    result = retrieve(
        RetrievalQuery(
            question="What was ABN AMRO's net profit in 2025?",
            company="ABN AMRO",
            year=2025,
            top_k=12,
        )
    )

    assert result.chunks[0].id == "abn-amro-2025.pdf:metric:profit"
    assert "Metric: Profit/(loss) for the period" in result.chunks[0].text
    assert "Value: 2,252" in result.chunks[0].text
