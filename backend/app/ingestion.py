from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from backend.app.config import settings
from backend.app.extract.categorize import extract_categorized_datapoints
from backend.app.ingest.chunks import (
    build_chunks,
    build_datapoint_chunks,
    deduplicate_chunks,
)
from backend.app.ingest.embedding import (
    CHROMA_UPSERT_BATCH_SIZE,
    chunk_metadata,
    delete_existing_source_chunks,
    embed_texts,
    get_collection,
)
from backend.app.ingest.persistence import (
    load_parsed_pages,
    persist_chunks,
    persist_datapoints,
    persist_parsed_pages,
)
from backend.app.ingest.vision import enhance_pages_with_vision
from backend.app.ingest.parsers import as_page_tuples, parse_pdf_pages

logger = logging.getLogger(__name__)


def ingest_pdf(
    pdf_path: Path,
    *,
    company: str | None,
    year: int | None,
    reset: bool = False,
    source_name: str | None = None,
    validate_datapoints: bool = True,
    skip_vision: bool = False,
    skip_parse: bool = False,
    status_callback: Callable[[str], None] | None = None,
) -> int:
    """Ingest one PDF and return the total number of chunks prepared for indexing."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    source = source_name or pdf_path.name
    processed_dir = settings.get_processed_path()

    if skip_parse:
        pages = load_parsed_pages(source, processed_dir)
        parser_name = "llamaparse"
        logger.info("loaded %d persisted pages for %s (skipped parse)", len(pages), source)
    else:
        logger.info("parsing %s with llamaparse", source)
        parsed = parse_pdf_pages(
            pdf_path,
            processed_dir=processed_dir,
            llama_cloud_api_key=settings.llama_cloud_api_key.get_secret_value(),
        )
        pages = list(as_page_tuples(parsed.pages))
        parser_name = parsed.parser
        logger.info("parsed %d non-empty pages with %s", len(pages), parser_name)
        pages_path = persist_parsed_pages(
            pages,
            source=source,
            company=company,
            year=year,
            parser=parser_name,
            processed_dir=processed_dir,
        )
        logger.info("wrote parsed pages to %s", pages_path)
    if not skip_vision:
        pages = enhance_pages_with_vision(
            pdf_path=pdf_path,
            pages=pages,
            source=source,
            company=company,
            year=year,
            parser=parser_name,
            processed_dir=processed_dir,
        )
        logger.info("applied automatic vision enhancement to difficult table pages")
    if status_callback:
        try:
            status_callback("embedding")
        except Exception:
            pass
    chunks = build_chunks(
        iter(pages),
        source=source,
        company=company,
        year=year,
        max_tokens=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
        parser=parser_name,
    )
    logger.info("built %d semantic chunks", len(chunks))
    chunks = deduplicate_chunks(chunks)
    logger.info("kept %d semantic chunks after exact deduplication", len(chunks))

    page_sections: dict[int, set[str]] = {}
    for chunk in chunks:
        if chunk.section_path:
            page_sections.setdefault(chunk.page, set()).add(chunk.section_path)
    datapoints = extract_categorized_datapoints(
        pages,
        source=source,
        company=company,
        year=year,
        validate=validate_datapoints,
        page_sections=page_sections,
    )
    datapoints_path = persist_datapoints(
        datapoints,
        source=source,
        processed_dir=processed_dir,
    )
    logger.info(
        "wrote %d categorized LLM-extracted datapoints to %s",
        len(datapoints),
        datapoints_path,
    )
    raw_datapoints = [dp if isinstance(dp, dict) else dp.model_dump() for dp in datapoints]

    if raw_datapoints:
        dp_chunks = build_datapoint_chunks(
            raw_datapoints,
            source=source,
            company=company,
            year=year,
            parser=parser_name,
            existing_chunks=chunks,
        )
        chunks.extend(dp_chunks)
        logger.info("added %d datapoint chunks", len(dp_chunks))

    chunks = deduplicate_chunks(chunks)
    logger.info("kept %d chunks after exact deduplication", len(chunks))
    chunks_path = persist_chunks(chunks, source=source, processed_dir=processed_dir)
    logger.info("wrote debug chunks to %s", chunks_path)

    embeddings = embed_texts([c.embedding_text or c.text for c in chunks])
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"embedding count {len(embeddings)} != chunk count {len(chunks)}"
        )

    collection = get_collection(reset=reset)
    if not reset:
        delete_existing_source_chunks(
            collection,
            source=source,
            company=company,
            year=year,
        )
    for i in range(0, len(chunks), CHROMA_UPSERT_BATCH_SIZE):
        batch_chunks = chunks[i : i + CHROMA_UPSERT_BATCH_SIZE]
        batch_embeddings = embeddings[i : i + CHROMA_UPSERT_BATCH_SIZE]
        collection.upsert(
            ids=[c.id for c in batch_chunks],
            documents=[c.text for c in batch_chunks],
            embeddings=batch_embeddings,
            metadatas=[chunk_metadata(c) for c in batch_chunks],
        )
        logger.info(
            "upserted batch %d-%d into '%s'",
            i,
            i + len(batch_chunks),
            settings.chroma_collection,
        )
    logger.info(
        "upserted %d chunks into '%s' at %s",
        len(chunks),
        settings.chroma_collection,
        settings.chroma_persist_dir,
    )
    return len(chunks)


def cli(argv: list[str] | None = None) -> int:
    """Parse CLI flags for ingestion and return the process exit code."""
    parser = argparse.ArgumentParser(description="Ingest an annual report PDF into ChromaDB.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--company", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--no-validate-datapoints", action="store_false", dest="validate_datapoints")
    parser.add_argument("--skip-vision", action="store_true", dest="skip_vision")
    parser.add_argument("--skip-parse", action="store_true", dest="skip_parse")
    parser.set_defaults(validate_datapoints=True, skip_vision=False, skip_parse=False)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        n = ingest_pdf(
            args.pdf,
            company=args.company,
            year=args.year,
            reset=args.reset,
            source_name=args.source_name,
            validate_datapoints=args.validate_datapoints,
            skip_vision=args.skip_vision,
            skip_parse=args.skip_parse,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ingested {n} chunks from {args.pdf.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
