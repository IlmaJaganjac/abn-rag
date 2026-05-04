from __future__ import annotations

import logging

import chromadb

from backend.app.config import openai_client, settings
from backend.app.schemas import Chunk

logger = logging.getLogger(__name__)

CHROMA_UPSERT_BATCH_SIZE = 1000


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed input texts in batches and return one embedding vector per text."""
    if not texts:
        return []
    client = openai_client()
    out: list[list[float]] = []
    batch = settings.embedding_batch_size
    for i in range(0, len(texts), batch):
        window = texts[i : i + batch]
        resp = client.embeddings.create(model=settings.openai_embedding_model, input=window)
        out.extend(item.embedding for item in resp.data)
        logger.info("embedded batch %d-%d", i, i + len(window))
    return out


def get_collection(reset: bool = False):
    """Return the persistent Chroma collection, optionally recreating it first."""
    client = chromadb.PersistentClient(path=str(settings.get_chroma_path()))
    name = settings.chroma_collection
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
    """Convert a `Chunk` into flat metadata suitable for vector-store persistence."""
    md: dict[str, str | int] = {
        "source": chunk.source,
        "page": chunk.page,
        "token_count": chunk.token_count,
    }
    cid_idx = chunk.id.rsplit(":", 1)[-1]
    md["idx"] = int(cid_idx)
    if chunk.company is not None:
        md["company"] = chunk.company
    if chunk.year is not None:
        md["year"] = chunk.year
    if chunk.parser is not None:
        md["parser"] = chunk.parser
    if chunk.chunk_kind is not None:
        md["chunk_kind"] = chunk.chunk_kind
    if chunk.section_path is not None:
        md["section_path"] = chunk.section_path
    for field in (
        "fact_kind",
        "basis",
        "scope_type",
        "quality",
        "validation_status",
        "canonical_metric",
    ):
        value = getattr(chunk, field, None)
        if value is not None:
            md[field] = value
    return md


def _source_where(source: str, company: str | None, year: int | None) -> dict:
    """Build the metadata filter used to find existing chunks for one source."""
    clauses: list[dict[str, str | int]] = [{"source": source}]
    if company is not None:
        clauses.append({"company": company})
    if year is not None:
        clauses.append({"year": year})
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def delete_existing_source_chunks(
    collection,
    *,
    source: str,
    company: str | None,
    year: int | None,
) -> None:
    """Delete stored chunks for one source before re-indexing new versions of the same file."""
    try:
        collection.delete(where=_source_where(source, company, year))
    except Exception as exc:  # noqa: BLE001
        logger.info("could not delete existing chunks for %s: %s", source, exc)
