from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter
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


def get_reranker():
    """Load and cache the cross-encoder reranker model, returning the singleton."""
    global _reranker
    if _reranker is None:
        import os
        # Skip ~25 HF HEAD requests on every cold load when model is already cached.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(settings.reranker_model)
    return _reranker


def rerank(query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """Re-score chunks with the cross-encoder and return the top-k by descending score."""
    reranker = get_reranker()
    pairs = [(query, c.text) for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c.model_copy(update={"score": float(s)}) for s, c in ranked[:top_k]]


_FTE_TERMS = frozenset({"fte", "ftes", "employee", "employees", "headcount", "workforce", "staff", "personnel", "people"})
_FTE_EXPANSION = "employees workforce headcount staff personnel FTE full-time equivalents payroll temporary internal external year-end average"

_SUST_TERMS = frozenset({"sustainability", "sustainable", "climate", "emissions", "emission", "ghg", "co2", "co₂", "scope", "net zero", "target", "goal"})
_SUST_EXPANSION = "sustainability climate targets goals ambition commitment GHG CO2 CO₂ CO2e CO₂e emissions scope 1 scope 2 scope 3 net zero renewable energy waste recycling carbon intensity"

_FIN_TERMS = frozenset({"revenue", "sales", "margin", "profit", "income", "dividend", "cash", "flow", "interest", "nii", "fee", "commission"})
_FIN_EXPANSION = "financial performance revenue net sales gross margin operating income net income net interest income NII fee commission dividend cash flow R&D research and development"


def expand_query_for_retrieval(question: str) -> str:
    """Expand a user question with domain terms and return the retrieval query text."""
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


_BM25_CACHE: dict[str, Any] = {"sig": None, "records": [], "tokens": []}


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
    return records


def passes_filters(record: dict[str, Any], query: RetrievalQuery) -> bool:
    """Return whether one persisted record matches the active retrieval filters."""
    if query.company is not None and record.get("company") != query.company:
        return False
    if query.year is not None and record.get("year") != query.year:
        return False
    return True


def bm25_candidates(query: RetrievalQuery) -> list[RetrievedChunk]:
    """Score persisted chunks with BM25-style ranking and return candidate chunks."""
    all_records = iter_processed_chunk_records()
    all_tokens = _BM25_CACHE["tokens"]
    pairs = [
        (rec, toks)
        for rec, toks in zip(all_records, all_tokens, strict=True)
        if passes_filters(rec, query)
    ]
    query_terms = tokenize(query.question)
    if not pairs or not query_terms:
        return []

    records = [p[0] for p in pairs]
    doc_tokens = [p[1] for p in pairs]
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


def retrieve(query: RetrievalQuery, *, skip_rerank: bool = False) -> RetrievalResult:
    """Run the retrieval pipeline and return ranked chunks plus the effective query."""
    collection = get_collection()
    retrieval_text = expand_query_for_retrieval(query.question)
    retrieval_query = query.model_copy(update={"question": retrieval_text}) if retrieval_text != query.question else query

    # BM25 has no dependency on the embedding — run it in parallel with embed+chroma.
    with ThreadPoolExecutor(max_workers=1) as ex:
        bm25_future = ex.submit(bm25_candidates, retrieval_query)

        [embedding] = embed_texts([retrieval_text])

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
    if settings.enable_rerank and not skip_rerank:
        chunks = rerank(query.question, candidates, query.top_k)
    else:
        chunks = candidates[: query.top_k]

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
    """Decompose the query (with optional history), retrieve for each sub-question, merge results."""
    if history:
        sub_questions = rewrite_and_decompose(query.question, history)
        logger.info("decomposed into %d sub-questions", len(sub_questions))
    else:
        sub_questions = [query.question]
    chunk_lists = [
        retrieve(query.model_copy(update={"question": q}), skip_rerank=True).chunks
        for q in sub_questions
    ]
    candidates = merge_chunks(chunk_lists, settings.rerank_top_n)
    if settings.enable_rerank:
        chunks = rerank(query.question, candidates, query.top_k)
    else:
        chunks = candidates[: query.top_k]
    return RetrievalResult(query=query, chunks=chunks)


__all__ = ["retrieve", "retrieve_decomposed", "rewrite_and_decompose", "settings"]
