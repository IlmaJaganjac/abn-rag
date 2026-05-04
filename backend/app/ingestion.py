from __future__ import annotations

import argparse
import logging
import sys
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
    apply_enhanced_text,
    load_datapoints_json,
    load_pages_jsonl,
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
    parser: str | None = None,
    source_name: str | None = None,
    enhanced_jsonl: Path | None = None,
    pages_jsonl: Path | None = None,
    extract_datapoints: bool = True,
    enhance_vision: bool = True,
    datapoints_json: Path | None = None,
    skip_embed: bool = False,
    validate_datapoints: bool = True,
) -> int:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    source = source_name or pdf_path.name
    requested_parser = parser or settings.pdf_parser
    processed_dir = settings.get_processed_path()

    if pages_jsonl is not None:
        logger.info("loading parsed pages from %s", pages_jsonl)
        pages, parsed_parser = load_pages_jsonl(pages_jsonl)
        parser_name = parsed_parser
        logger.info("loaded %d non-empty pages from %s", len(pages), pages_jsonl)
    else:
        logger.info("parsing %s with %s", source, requested_parser)
        parsed = parse_pdf_pages(
            pdf_path,
            parser=requested_parser,
            processed_dir=processed_dir,
            llama_cloud_api_key=settings.llama_cloud_api_key.get_secret_value(),
        )
        pages = list(as_page_tuples(parsed.pages))
        parser_name = parsed.parser
        logger.info("parsed %d non-empty pages with %s", len(pages), parser_name)

    if enhanced_jsonl is not None and pages_jsonl is None:
        pages = apply_enhanced_text(pages, enhanced_jsonl)

    pages_path = persist_parsed_pages(
        pages,
        source=source,
        company=company,
        year=year,
        parser=parser_name,
        processed_dir=processed_dir,
    )
    logger.info("wrote parsed pages to %s", pages_path)
    if pages_jsonl is None and enhanced_jsonl is None and enhance_vision:
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

    raw_datapoints: list[dict] = []
    if extract_datapoints:
        datapoints = extract_categorized_datapoints(
            pages,
            source=source,
            company=company,
            year=year,
            validate=validate_datapoints,
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
    elif datapoints_json is not None:
        raw_datapoints = load_datapoints_json(datapoints_json)
        logger.info("loaded %d datapoints from %s", len(raw_datapoints), datapoints_json)

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

    if skip_embed:
        logger.info("skip_embed=True: skipping embedding and ChromaDB upsert")
        return len(chunks)

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


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest an annual report PDF into ChromaDB.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--company", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--parser", choices=["llamaparse"], default=settings.pdf_parser)
    parser.add_argument("--enhanced-jsonl", type=Path, default=None)
    parser.add_argument("--pages-jsonl", type=Path, default=None)
    parser.add_argument("--no-extract-datapoints", action="store_false", dest="extract_datapoints")
    parser.add_argument("--no-vision-enhancement", action="store_false", dest="enhance_vision")
    parser.add_argument("--datapoints-json", type=Path, default=None)
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--no-validate-datapoints", action="store_false", dest="validate_datapoints")
    parser.set_defaults(extract_datapoints=True, enhance_vision=True, skip_embed=False, validate_datapoints=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        n = ingest_pdf(
            args.pdf,
            company=args.company,
            year=args.year,
            reset=args.reset,
            parser=args.parser,
            source_name=args.source_name,
            enhanced_jsonl=args.enhanced_jsonl,
            pages_jsonl=args.pages_jsonl,
            extract_datapoints=args.extract_datapoints,
            enhance_vision=args.enhance_vision,
            datapoints_json=args.datapoints_json,
            skip_embed=args.skip_embed,
            validate_datapoints=args.validate_datapoints,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ingested {n} chunks from {args.pdf.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
