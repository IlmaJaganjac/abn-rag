from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from typing import Any

from backend.app.config import settings
from backend.app.ingestion import embed_texts, get_collection
from backend.app.schemas import RetrievalQuery, RetrievalResult, RetrievedChunk

RRF_K = 60
DENSE_TOP_N = 30
BM25_TOP_N = 30
METRIC_TOP_N = 30

_FTE_TERMS = frozenset({"fte", "ftes", "employee", "employees", "headcount", "workforce", "staff", "personnel", "people"})
_FTE_EXPANSION = "employees workforce headcount staff personnel FTE full-time equivalents payroll temporary internal external year-end average"

_SUST_TERMS = frozenset({"sustainability", "sustainable", "climate", "emissions", "emission", "ghg", "co2", "co₂", "scope", "net zero", "target", "goal"})
_SUST_EXPANSION = "sustainability climate targets goals ambition commitment GHG CO2 CO₂ CO2e CO₂e emissions scope 1 scope 2 scope 3 net zero renewable energy waste recycling carbon intensity"

_FIN_TERMS = frozenset({"revenue", "sales", "margin", "profit", "income", "dividend", "cash flow", "r&d", "research and development"})
_FIN_EXPANSION = "financial performance revenue net sales gross margin operating income net income dividend cash flow R&D research and development"


def expand_query_for_retrieval(question: str) -> str:
    q = question.casefold()
    tokens = set(re.findall(r"[a-z0-9₂&]+", q))
    parts = [question]
    if tokens & _FTE_TERMS:
        parts.append(_FTE_EXPANSION)
    if tokens & _SUST_TERMS or "net zero" in q or "cash flow" in q or "research and development" in q:
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


def _build_metric_where(company: str | None, year: int | None) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = [{"chunk_kind": "metric"}]
    if company is not None:
        clauses.append({"company": company})
    if year is not None:
        clauses.append({"year": year})
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
        score=score,
    )


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 1
    ]


def _is_numeric_query(question: str) -> bool:
    query = question.casefold()
    query_without_years = re.sub(r"\b20\d{2}\b", " ", query)
    if re.search(r"\d", query_without_years):
        return True
    return any(
        term in query
        for term in (
            "ratio",
            "profit",
            "income",
            "revenue",
            "sales",
            "employees",
            "fte",
            "capital",
            "assets",
            "margin",
            "dividend",
            "cost",
            "cet1",
            "roe",
            "nim",
            "how many",
            "how much",
        )
    )


def _query_years(question: str) -> set[str]:
    return set(re.findall(r"\b20\d{2}\b", question))


def _metric_candidate_score(question: str, document: str) -> float:
    query = question.casefold()
    doc = document.casefold()
    score = 0.0

    if "total net sales" in query and "metric: total net sales" in doc:
        score += 3.0

    for year in _query_years(question):
        if f"period: {year}" in doc:
            score += 2.0

    asks_millions = "million" in query or "million euros" in query
    has_million_euro_unit = "unit:" in doc and "in millions" in doc and "€" in document
    if asks_millions and has_million_euro_unit:
        score += 2.0

    query_terms = {
        term
        for term in re.findall(r"[a-z0-9]+", query)
        if len(term) > 2 and term not in {"what", "were", "was", "the", "asml", "did"}
    }
    doc_terms = set(re.findall(r"[a-z0-9]+", doc))
    score += 0.1 * len(query_terms & doc_terms)
    return score


def _metric_candidates(collection, query: RetrievalQuery) -> list[RetrievedChunk]:
    raw = collection.get(
        where=_build_metric_where(query.company, query.year),
        include=["documents", "metadatas"],
    )
    ids = raw.get("ids") or []
    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    scored: list[tuple[float, RetrievedChunk]] = []
    for cid, doc, meta in zip(ids, docs, metas, strict=True):
        score = _metric_candidate_score(query.question, doc or "")
        if score <= 0:
            continue
        scored.append((
            score,
            _retrieved_chunk(
                cid=cid,
                doc=doc or "",
                meta=meta or {},
                score=score,
            ),
        ))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:METRIC_TOP_N]]


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
    metric_chunks: list[RetrievedChunk],
    metric_weight: float,
    top_k: int,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}

    def add_ranked(chunks: list[RetrievedChunk], weight: float, use_metric_score: bool = False) -> None:
        for rank, chunk in enumerate(chunks, start=1):
            chunks_by_id.setdefault(chunk.id, chunk)
            scores[chunk.id] = scores.get(chunk.id, 0.0) + weight / (RRF_K + rank)
            if use_metric_score:
                scores[chunk.id] += 0.01 * chunk.score

    add_ranked(dense_chunks, 1.0)
    add_ranked(bm25_chunks, 1.2)
    add_ranked(metric_chunks, metric_weight, use_metric_score=metric_weight > 1.0)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    out: list[RetrievedChunk] = []
    for cid, score in ranked:
        chunk = chunks_by_id[cid].model_copy(update={"score": score})
        out.append(chunk)
    return out


def retrieve(query: RetrievalQuery) -> RetrievalResult:
    collection = get_collection()

    if os.environ.get("ENABLE_QUERY_EXPANSION") == "1":
        retrieval_text = expand_query_for_retrieval(query.question)
    else:
        retrieval_text = query.question

    [embedding] = embed_texts([retrieval_text])

    where = _build_where(query.company, query.year)
    raw = collection.query(
        query_embeddings=[embedding],
        n_results=max(query.top_k, DENSE_TOP_N),
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

    metric_weight = 1.5 if _is_numeric_query(query.question) else 0.5
    if os.environ.get("DISABLE_METRIC_CANDIDATES") == "1":
        metric_chunks: list[RetrievedChunk] = []
    else:
        metric_chunks = _metric_candidates(collection, query)
    chunks = _rrf_merge(
        dense_chunks=dense_chunks,
        bm25_chunks=_bm25_candidates(retrieval_query),
        metric_chunks=metric_chunks,
        metric_weight=metric_weight,
        top_k=query.top_k,
    )
    return RetrievalResult(query=query, chunks=chunks)


__all__ = ["retrieve", "settings"]
