from __future__ import annotations

from typing import Any

from backend.app.config import settings
from backend.app.ingestion import embed_texts, get_collection
from backend.app.schemas import RetrievalQuery, RetrievalResult, RetrievedChunk


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


def retrieve(query: RetrievalQuery) -> RetrievalResult:
    collection = get_collection()
    [embedding] = embed_texts([query.question])

    where = _build_where(query.company, query.year)
    raw = collection.query(
        query_embeddings=[embedding],
        n_results=query.top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = (raw.get("ids") or [[]])[0]
    docs = (raw.get("documents") or [[]])[0]
    metas = (raw.get("metadatas") or [[]])[0]
    dists = (raw.get("distances") or [[]])[0]

    chunks: list[RetrievedChunk] = []
    for cid, doc, meta, dist in zip(ids, docs, metas, dists, strict=True):
        meta = meta or {}
        chunks.append(
            RetrievedChunk(
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
                score=1.0 - float(dist),
            )
        )

    return RetrievalResult(query=query, chunks=chunks)


__all__ = ["retrieve", "settings"]
