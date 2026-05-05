from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

_REWRITE_SYSTEM = (
    "You are a query planner for a RAG system over annual reports. "
    "Given optional conversation history and the latest user question, produce 1–4 standalone atomic sub-questions. "
    "Each sub-question must be fully self-contained (resolve all pronouns and references using history). "
    "Return ONLY a JSON array of strings, no explanation. "
    "If the question is already atomic and standalone, return a single-element array."
)

logger = logging.getLogger(__name__)


from backend.app.config import settings
from backend.app.ingestion import embed_texts, get_collection
from backend.app.schemas import RetrievalQuery, RetrievalResult, RetrievedChunk

RRF_K = 60
DENSE_TOP_N = 40
BM25_TOP_N = 40

_reranker = None
_reranker_lock = threading.Lock()
_reranker_state: dict[str, str | None] = {
    "status": "idle",
    "model": settings.reranker_model,
    "error": None,
}


def get_reranker_status() -> dict[str, str | None]:
    """Return the current runtime load status of the cross-encoder reranker."""
    return dict(_reranker_state)


def warmup_reranker() -> None:
    """Trigger the runtime model download/load path without changing retrieval behavior."""
    try:
        get_reranker()
    except Exception:
        logger.exception("reranker warmup failed")


def get_reranker():
    """Load and cache the cross-encoder reranker model, returning the singleton."""
    global _reranker
    if _reranker is not None:
        return _reranker

    with _reranker_lock:
        if _reranker is not None:
            return _reranker
        _reranker_state.update({
            "status": "loading",
            "model": settings.reranker_model,
            "error": None,
        })
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(settings.reranker_model)
            _reranker_state.update({"status": "ready", "error": None})
        except Exception as exc:
            _reranker_state.update({"status": "error", "error": str(exc)})
            raise
    return _reranker


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Re-score chunks with the cross-encoder and return the top-k by descending score."""
    reranker = get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c.model_copy(update={"score": float(s)}) for s, c in ranked[:top_k]]


def build_where(company: str | None, year: int | None) -> dict[str, Any] | None:
    """Build a Chroma metadata filter for the requested company and year."""
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


def retrieved_chunk(
    *,
    cid: str,
    doc: str,
    meta: dict[str, Any],
    score: float,
) -> RetrievedChunk:
    """Convert raw vector-store fields into one `RetrievedChunk` object."""
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


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms for BM25-style scoring."""
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 1
    ]


def chunk_search_text(record: dict[str, Any]) -> str:
    """Concatenate searchable record fields into one BM25 text string."""
    parts = [
        record.get("text"),
        record.get("section_path"),
        record.get("company"),
        str(record.get("year")) if record.get("year") is not None else None,
        record.get("chunk_kind"),
    ]
    return "\n".join(str(part) for part in parts if part)


_BM25_CACHE: dict[str, Any] = {"sig": None, "records": [], "tokens": [], "bm25": None}


def chunks_dir_signature() -> tuple[tuple[str, float, int], ...]:
    """Return a signature of the chunks dir so cache invalidates on file changes."""
    chunks_dir = settings.get_processed_path() / "chunks"
    if not chunks_dir.exists():
        return ()
    return tuple(
        (p.name, p.stat().st_mtime, p.stat().st_size)
        for p in sorted(chunks_dir.glob("*.jsonl"))
    )


def iter_processed_chunk_records() -> list[dict[str, Any]]:
    """Load all persisted chunk records from disk; cached until files change."""
    sig = chunks_dir_signature()
    if _BM25_CACHE["sig"] == sig:
        return _BM25_CACHE["records"]

    chunks_dir = settings.get_processed_path() / "chunks"
    records: list[dict[str, Any]] = []
    for path in sorted(chunks_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))

    _BM25_CACHE["sig"] = sig
    _BM25_CACHE["records"] = records
    _BM25_CACHE["tokens"] = [tokenize(chunk_search_text(r)) for r in records]
    _BM25_CACHE["bm25"] = None
    return records


def passes_filters(record: dict[str, Any], query: RetrievalQuery) -> bool:
    """Return whether one persisted record matches the active retrieval filters."""
    if query.company is not None and record.get("company") != query.company:
        return False
    if query.year is not None and record.get("year") != query.year:
        return False
    return True


def bm25_candidates(query: RetrievalQuery) -> list[RetrievedChunk]:
    """Score persisted chunks with BM25 and return candidate chunks."""
    all_records = iter_processed_chunk_records()
    all_tokens = _BM25_CACHE["tokens"]
    query_terms = tokenize(query.question)
    if not all_records or not query_terms:
        return []

    if _BM25_CACHE["bm25"] is None:
        from rank_bm25 import BM25Okapi
        _BM25_CACHE["bm25"] = BM25Okapi(all_tokens)

    scores = _BM25_CACHE["bm25"].get_scores(query_terms)
    scored = [
        (float(score), record)
        for score, record in zip(scores, all_records, strict=True)
        if score > 0 and passes_filters(record, query)
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    chunks: list[RetrievedChunk] = []
    for score, record in scored[:BM25_TOP_N]:
        chunks.append(
            retrieved_chunk(
                cid=str(record.get("id", "")),
                doc=str(record.get("text", "")),
                meta=record,
                score=score,
            )
        )
    return chunks


def rrf_merge(
    *,
    dense_chunks: list[RetrievedChunk],
    bm25_chunks: list[RetrievedChunk],
    top_k: int,
) -> list[RetrievedChunk]:
    """Fuse dense and BM25 rankings with reciprocal-rank fusion and return merged chunks."""
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, RetrievedChunk] = {}

    def add_ranked(chunks: list[RetrievedChunk], weight: float) -> None:
        """Accumulate reciprocal-rank scores for one ranked result list."""
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


def retrieve(query: RetrievalQuery, *, rerank_results: bool = True) -> RetrievalResult:
    """Run the retrieval pipeline and return ranked chunks plus the effective query."""
    collection = get_collection()

    # BM25 has no dependency on the embedding — run it in parallel with embed+chroma.
    with ThreadPoolExecutor(max_workers=1) as ex:
        bm25_future = ex.submit(bm25_candidates, query)

        [embedding] = embed_texts([query.question])

        where = build_where(query.company, query.year)
        raw = collection.query(
            query_embeddings=[embedding],
            n_results=max(query.top_k, DENSE_TOP_N),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        bm25_chunks = bm25_future.result()

    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    dense_chunks: list[RetrievedChunk] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
        meta = meta or {}
        dense_chunks.append(retrieved_chunk(cid=cid, doc=doc or "", meta=meta, score=1.0 - float(dist)))

    candidates = rrf_merge(
        dense_chunks=dense_chunks,
        bm25_chunks=bm25_chunks,
        top_k=settings.rerank_top_n,
    )
    chunks = rerank(query.question, candidates, query.top_k) if rerank_results else candidates

    return RetrievalResult(query=query, chunks=chunks)


def rewrite_and_decompose(question: str, history: list[dict]) -> list[str]:
    """Rewrite a follow-up question using conversation history, decompose into atomic sub-questions."""
    from backend.app.config import openai_client
    client = openai_client()
    user_content = question
    if history:
        lines = []
        for h in history[-3:]:
            lines.append(f"Q: {h['question']}\nA: {h['answer']}")
        user_content = "Conversation history:\n" + "\n\n".join(lines) + f"\n\nCurrent question: {question}"
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": _REWRITE_SYSTEM},
            {"role": "user", "content": user_content},
        ],
    )
    raw = (resp.choices[0].message.content or "").strip()
    try:
        parts = json.loads(raw)
        if isinstance(parts, list) and parts:
            return [str(q) for q in parts]
    except Exception:
        pass
    return [question]


def merge_chunks(chunk_lists: list[list[RetrievedChunk]], top_k: int) -> list[RetrievedChunk]:
    """Deduplicate chunks from multiple retrievals, keeping highest score per id."""
    best: dict[str, RetrievedChunk] = {}
    for chunks in chunk_lists:
        for chunk in chunks:
            if chunk.id not in best or chunk.score > best[chunk.id].score:
                best[chunk.id] = chunk
    return sorted(best.values(), key=lambda c: c.score, reverse=True)[:top_k]


def retrieve_decomposed(query: RetrievalQuery, history: list[dict] | None = None) -> RetrievalResult:
    """Decompose questions with history, retrieve for each sub-question, merge results."""
    history = history or []
    sub_questions = rewrite_and_decompose(query.question, history) if history else [query.question]
    logger.info("decomposed into %d sub-questions", len(sub_questions))
    chunk_lists = [
        retrieve(query.model_copy(update={"question": q}), rerank_results=False).chunks
        for q in sub_questions
    ]
    candidates = merge_chunks(chunk_lists, settings.rerank_top_n)
    chunks = rerank(query.question, candidates, query.top_k)
    return RetrievalResult(query=query, chunks=chunks)


__all__ = [
    "retrieve",
    "retrieve_decomposed",
    "rewrite_and_decompose",
    "settings",
]
