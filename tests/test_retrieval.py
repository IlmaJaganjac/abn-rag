from __future__ import annotations

from typing import Any

from backend.app.answer import _ground_citations
from backend.app.retrieval import expand_query_for_retrieval, rerank_chunks_cross_encoder, retrieve
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


def test_expand_fte_question():
    expanded = expand_query_for_retrieval("How many FTEs did ASML have in 2025?")
    assert expanded.startswith("How many FTEs did ASML have in 2025?")
    assert "employees" in expanded
    assert "headcount" in expanded
    assert "workforce" in expanded
    assert "full-time equivalents" in expanded


def test_expand_sustainability_question():
    expanded = expand_query_for_retrieval("What are ASML's sustainability goals for 2030?")
    assert expanded.startswith("What are ASML's sustainability goals for 2030?")
    assert "climate" in expanded
    assert "emissions" in expanded
    assert "scope" in expanded
    assert "net zero" in expanded


def test_expand_financial_question():
    expanded = expand_query_for_retrieval("What was ASML's revenue in 2025?")
    assert expanded.startswith("What was ASML's revenue in 2025?")
    assert "net sales" in expanded
    assert "gross margin" in expanded


def test_expand_unrelated_question_unchanged():
    q = "Who was the CEO of ASML in 2025?"
    expanded = expand_query_for_retrieval(q)
    assert expanded == q


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


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------

def test_reranker_reorders_by_mock_scores(monkeypatch):
    chunks = [
        _chunk("id:low", "low relevance text", 1),
        _chunk("id:mid", "medium relevance text", 2),
        _chunk("id:high", "most relevant text", 3),
    ]
    monkeypatch.setattr("backend.app.retrieval._get_reranker", lambda: type("M", (), {"predict": lambda self, pairs: [0.1, 0.5, 0.9]})())
    result = rerank_chunks_cross_encoder("test question", chunks, top_k=3)
    assert result[0].id == "id:high"
    assert result[1].id == "id:mid"
    assert result[2].id == "id:low"
    assert result[0].score == 0.9


def test_reranker_returns_top_k(monkeypatch):
    chunks = [_chunk(f"id:{i}", f"text {i}", i + 1) for i in range(10)]
    scores = list(range(10))
    monkeypatch.setattr("backend.app.retrieval._get_reranker", lambda: type("M", (), {"predict": lambda self, pairs: scores})())
    result = rerank_chunks_cross_encoder("question", chunks, top_k=3)
    assert len(result) == 3
    assert result[0].id == "id:9"


def test_reranker_fallback_on_exception(monkeypatch):
    def bad_predict(pairs):
        raise RuntimeError("model failed")
    monkeypatch.setattr("backend.app.retrieval._get_reranker", lambda: type("M", (), {"predict": bad_predict})())
    chunks = [_chunk("id:a", "text a", 1), _chunk("id:b", "text b", 2)]
    result = rerank_chunks_cross_encoder("question", chunks, top_k=2)
    assert [c.id for c in result] == ["id:a", "id:b"]


def test_rerank_candidate_k_env_var(monkeypatch):
    captured = {}
    def fake_rerank(question, chunks, top_k, text_chars=3000):
        captured["n_chunks"] = len(chunks)
        return chunks[:top_k]
    monkeypatch.setenv("ENABLE_RERANKER", "1")
    monkeypatch.setenv("RERANK_CANDIDATE_K", "15")
    monkeypatch.setattr("backend.app.retrieval.rerank_chunks_cross_encoder", fake_rerank)
    monkeypatch.setattr("backend.app.retrieval.get_collection", lambda: FakeHybridCollection())
    monkeypatch.setattr("backend.app.retrieval.embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr("backend.app.retrieval._bm25_candidates", lambda query: [])
    retrieve(RetrievalQuery(question="How many FTEs?", company="ABN AMRO", year=2025, top_k=12))
    assert captured.get("n_chunks", 999) <= 15


def test_rerank_text_chars_env_var(monkeypatch):
    captured = {}
    original_predict = lambda self, pairs: [0.5] * len(pairs)

    def fake_get_reranker():
        class FakeModel:
            def predict(self, pairs):
                captured["max_len"] = max(len(p[1]) for p in pairs) if pairs else 0
                return [0.5] * len(pairs)
        return FakeModel()

    monkeypatch.setattr("backend.app.retrieval._get_reranker", fake_get_reranker)
    long_text = "x" * 5000
    chunks = [_chunk("id:1", long_text, 1), _chunk("id:2", long_text, 2)]
    rerank_chunks_cross_encoder("question", chunks, top_k=2, text_chars=1200)
    assert captured["max_len"] <= 1200


def test_reranker_not_called_without_flag(monkeypatch):
    called = []
    monkeypatch.setattr("backend.app.retrieval.rerank_chunks_cross_encoder",
        lambda q, chunks, top_k: called.append(True) or chunks[:top_k])
    monkeypatch.setattr("backend.app.retrieval.get_collection", lambda: FakeHybridCollection())
    monkeypatch.setattr("backend.app.retrieval.embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr("backend.app.retrieval._bm25_candidates", lambda query: [])
    monkeypatch.delenv("ENABLE_RERANKER", raising=False)
    retrieve(RetrievalQuery(question="What was ABN AMRO's net profit?", company="ABN AMRO", year=2025, top_k=5))
    assert called == []
