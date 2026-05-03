#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.config import settings
from backend.app.extracted_datapoints import NormalizedDatapointSet, datapoints_to_chunks
from backend.app.ingestion import (
    CHROMA_UPSERT_BATCH_SIZE,
    _chunk_metadata,
    deduplicate_chunks,
    embed_texts,
    get_collection,
)

_CHUNKS_ROOT = Path("backend/data/processed/chunks")


def _delete_extracted_datapoint_chunks(collection, *, source: str) -> None:
    try:
        collection.delete(where={"$and": [
            {"source": source},
            {"chunk_kind": "extracted_datapoint"},
        ]})
    except Exception as exc:
        logging.getLogger(__name__).info(
            "could not delete extracted chunks for %s: %s", source, exc
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Index pre-extracted LlamaExtract datapoints into Chroma."
    )
    parser.add_argument("pre_extracted_json", type=Path)
    parser.add_argument(
        "--reset-existing",
        action="store_true",
        help="delete existing extracted_datapoint chunks for this source before upserting",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    # Load
    ds = NormalizedDatapointSet.model_validate_json(
        args.pre_extracted_json.read_text(encoding="utf-8")
    )
    log.info("loaded %d datapoints from %s", len(ds.datapoints), args.pre_extracted_json)

    # Convert to chunks
    chunks = datapoints_to_chunks(ds)
    log.info("converted to %d chunks", len(chunks))
    chunks = deduplicate_chunks(chunks)
    log.info("kept %d chunks after exact embedding-text deduplication", len(chunks))

    by_type: Counter = Counter(dp.datapoint_type for dp in ds.datapoints)
    print(f"\nDatapoints loaded: {len(ds.datapoints)}")
    print(f"Chunks created:   {len(chunks)}")
    print("Count by type:")
    for dtype, count in sorted(by_type.items()):
        print(f"  {dtype}: {count}")

    # Embed
    embedding_texts = [c.embedding_text or c.text for c in chunks]
    log.info("embedding %d chunks", len(chunks))
    embeddings = embed_texts(embedding_texts)
    if len(embeddings) != len(chunks):
        print(f"error: embedding count {len(embeddings)} != chunk count {len(chunks)}", file=sys.stderr)
        return 1

    # Upsert to Chroma
    collection = get_collection(reset=False)
    if args.reset_existing:
        log.info("deleting existing extracted_datapoint chunks for source=%s", ds.source)
        _delete_extracted_datapoint_chunks(collection, source=ds.source)

    for i in range(0, len(chunks), CHROMA_UPSERT_BATCH_SIZE):
        batch_chunks = chunks[i: i + CHROMA_UPSERT_BATCH_SIZE]
        batch_embeddings = embeddings[i: i + CHROMA_UPSERT_BATCH_SIZE]
        collection.upsert(
            ids=[c.id for c in batch_chunks],
            documents=[c.text for c in batch_chunks],
            embeddings=batch_embeddings,
            metadatas=[_chunk_metadata(c) for c in batch_chunks],
        )
        log.info("upserted batch %d-%d", i, i + len(batch_chunks))

    print(f"\nChroma collection: '{settings.chroma_collection}'")

    # Write extracted JSONL sidecar for BM25
    stem = Path(ds.source).stem
    out_jsonl = _CHUNKS_ROOT / f"{stem}.extracted.jsonl"
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "id": chunk.id,
                "source": chunk.source,
                "company": chunk.company,
                "year": chunk.year,
                "page": chunk.page,
                "chunk_kind": chunk.chunk_kind,
                "section_path": chunk.section_path,
                "token_count": chunk.token_count,
                "text": chunk.text,
                "embedding_text": chunk.embedding_text,
                "fact_kind": chunk.fact_kind,
                "basis": chunk.basis,
                "scope_type": chunk.scope_type,
                "quality": chunk.quality,
                "validation_status": chunk.validation_status,
                "canonical_metric": chunk.canonical_metric,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"BM25 sidecar:      {out_jsonl}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
