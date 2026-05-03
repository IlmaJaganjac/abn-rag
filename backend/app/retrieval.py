from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)

from backend.app.config import settings
from backend.app.ingestion import embed_texts, get_collection
from backend.app.schemas import RetrievalQuery, RetrievalResult, RetrievedChunk

RRF_K = 60
DENSE_TOP_N = 30
BM25_TOP_N = 30

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_CANDIDATE_K = 40
DEFAULT_RERANK_TEXT_CHARS = 1000
_reranker_model = None
_reranker_model_name: str | None = None


def _get_reranker():
    global _reranker_model, _reranker_model_name
    model_name = os.environ.get("RERANKER_MODEL") or DEFAULT_RERANKER_MODEL
    if _reranker_model is None or _reranker_model_name != model_name:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(model_name)
        _reranker_model_name = model_name
    return _reranker_model


def rerank_chunks_cross_encoder(
    question: str,
    chunks: list[RetrievedChunk],
    top_k: int,
    text_chars: int = 3000,
) -> list[RetrievedChunk]:
    try:
        model = _get_reranker()
        pairs = [(question, chunk.text[:text_chars]) for chunk in chunks]
        scores = model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk.model_copy(update={"score": float(score)}) for score, chunk in ranked[:top_k]]
    except Exception as exc:
        logger.warning("reranker failed, falling back to original order: %s", exc)
        return chunks[:top_k]


_FTE_TERMS = frozenset({"fte", "ftes", "employee", "employees", "headcount", "workforce", "staff", "personnel", "people"})
_FTE_EXPANSION = "employees workforce headcount staff personnel FTE full-time equivalents payroll temporary internal external year-end average"

_SUST_TERMS = frozenset({"sustainability", "sustainable", "climate", "emissions", "emission", "ghg", "co2", "co₂", "scope", "net zero", "target", "goal"})
_SUST_EXPANSION = "sustainability climate targets goals ambition commitment GHG CO2 CO₂ CO2e CO₂e emissions scope 1 scope 2 scope 3 net zero renewable energy waste recycling carbon intensity"

_FIN_TERMS = frozenset({"revenue", "sales", "margin", "profit", "income", "dividend", "cash", "flow", "interest", "nii", "fee", "commission"})
_FIN_EXPANSION = "financial performance revenue net sales gross margin operating income net income net interest income NII fee commission dividend cash flow R&D research and development"


def expand_query_for_retrieval(question: str) -> str:
    q = question.casefold()
    tokens = set(re.findall(r"[a-z0-9₂&]+", q))
    parts = [question]
    if tokens & _FTE_TERMS:
        parts.append(_FTE_EXPANSION)
    if tokens & _SUST_TERMS or "net zero" in q:
        parts.append(_SUST_EXPANSION)
    if tokens & _FIN_TERMS or "cash flow" in q or "r&d" in q or "research and development" in q:
        parts.append(_FIN_EXPANSION)
    return "\n".join(parts)


def _build_where(company: str | None, year: int | None) -> dict[str, Any] | None:
    clauses: list[dict[str, Any]] = []
    if company is not None:
        clauses.append({"company": company})
    if year is not None:
        clauses.append({"year": year})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def _retrieved_chunk(
    *,
    cid: str,
    doc: str,
    meta: dict[str, Any],
    score: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        id=cid,
        source=str(meta.get("source", "")),
        company=meta.get("company"),
        year=meta.get("year"),
        page=int(meta.get("page", 1)),
        text=doc or "",
        token_count=int(meta.get("token_count", 0)),
        parser=meta.get("parser"),
        chunk_kind=meta.get("chunk_kind"),
        section_path=meta.get("section_path"),
        fact_kind=meta.get("fact_kind"),
        basis=meta.get("basis"),
        scope_type=meta.get("scope_type"),
        quality=meta.get("quality"),
        validation_status=meta.get("validation_status"),
        canonical_metric=meta.get("canonical_metric"),
        score=score,
    )


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 1
    ]


def _chunk_search_text(record: dict[str, Any]) -> str:
    parts = [
        record.get("text"),
        record.get("section_path"),
        record.get("company"),
        str(record.get("year")) if record.get("year") is not None else None,
        record.get("chunk_kind"),
    ]
    return "\n".join(str(part) for part in parts if part)


def _iter_processed_chunk_records() -> list[dict[str, Any]]:
    chunks_dir = settings.get_processed_path() / "chunks"
    records: list[dict[str, Any]] = []
    for path in sorted(chunks_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
    return records


def _passes_filters(record: dict[str, Any], query: RetrievalQuery) -> bool:
    if query.company is not None and record.get("company") != query.company:
        return False
    if query.year is not None and record.get("year") != query.year:
        return False
    return True


def _bm25_candidates(query: RetrievalQuery) -> list[RetrievedChunk]:
    records = [record for record in _iter_processed_chunk_records() if _passes_filters(record, query)]
    query_terms = _tokenize(query.question)
    if not records or not query_terms:
        return []

    doc_tokens = [_tokenize(_chunk_search_text(record)) for record in records]
    doc_freq: Counter[str] = Counter()
    for tokens in doc_tokens:
        doc_freq.update(set(tokens))

    avg_len = sum(len(tokens) for tokens in doc_tokens) / len(doc_tokens)
    k1 = 1.5
    b = 0.75
    scored: list[tuple[float, dict[str, Any]]] = []
    for record, tokens in zip(records, doc_tokens, strict=True):
        if not tokens:
            continue
        counts = Counter(tokens)
        score = 0.0
        for term in query_terms:
            if term not in counts:
                continue
            idf = math.log(1 + (len(records) - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            tf = counts[term]
            denom = tf + k1 * (1 - b + b * len(tokens) / avg_len)
            score += idf * (tf * (k1 + 1) / denom)
        if score > 0:
            scored.append((score, record))

    scored.sort(key=lambda item: item[0], reverse=True)
    chunks: list[RetrievedChunk] = []
    for score, record in scored[:BM25_TOP_N]:
        chunks.append(
            _retrieved_chunk(
                cid=str(record.get("id", "")),
                doc=str(record.get("text", "")),
                meta=record,
                score=score,
            )
        )
    return chunks


def _rrf_merge(
    *,
    dense_chunks: list[RetrievedChunk],
    bm25_chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}

    def add_ranked(chunks: list[RetrievedChunk], weight: float) -> None:
        for rank, chunk in enumerate(chunks, start=1):
            chunks_by_id.setdefault(chunk.id, chunk)
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight / (RRF_K + rank)

    add_ranked(dense_chunks, 1.0)
    add_ranked(bm25_chunks, 1.2)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    out: list[RetrievedChunk] = []
    for cid, score in ranked:
        chunk = chunks_by_id[cid].model_copy(update={"score": score})
        out.append(chunk)
    return out


def retrieve(query: RetrievalQuery) -> RetrievalResult:
    collection = get_collection()
    reranker_enabled = os.environ.get("ENABLE_RERANKER") == "1"
    if reranker_enabled:
        _ck = os.environ.get("RERANK_CANDIDATE_K")
        candidate_k = int(_ck) if _ck else max(query.top_k * 4, DEFAULT_RERANK_CANDIDATE_K)
    else:
        candidate_k = query.top_k

    retrieval_text = expand_query_for_retrieval(query.question)

    [embedding] = embed_texts([retrieval_text])

    where = _build_where(query.company, query.year)
    raw = collection.query(
        query_embeddings=[embedding],
        n_results=max(candidate_k, DENSE_TOP_N),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    dense_chunks: list[RetrievedChunk] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
        meta = meta or {}
        dense_chunks.append(_retrieved_chunk(cid=cid, doc=doc or "", meta=meta, score=1.0 - float(dist)))

    retrieval_query = query.model_copy(update={"question": retrieval_text}) if retrieval_text != query.question else query

    chunks = _rrf_merge(
        dense_chunks=dense_chunks,
        bm25_chunks=_bm25_candidates(retrieval_query),
        top_k=candidate_k,
    )

    if reranker_enabled:
        _tc = os.environ.get("RERANK_TEXT_CHARS")
        text_chars = int(_tc) if _tc else DEFAULT_RERANK_TEXT_CHARS
        chunks = rerank_chunks_cross_encoder(query.question, chunks, query.top_k, text_chars=text_chars)

    return RetrievalResult(query=query, chunks=chunks)


__all__ = ["retrieve", "settings"]
