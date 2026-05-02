from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import chromadb
import tiktoken

from backend.app._openai import openai_client
from backend.app.chunking import build_semantic_chunks
from backend.app.config import settings
from backend.app.parsers import as_page_tuples, parse_pdf_pages
from backend.app.schemas import Chunk

logger = logging.getLogger(__name__)

EMBEDDING_MAX_TOKENS = 8191
CHROMA_UPSERT_BATCH_SIZE = 1000
_ENCODING = tiktoken.get_encoding("cl100k_base")
_FTE_PATTERNS = [
    re.compile(r"\bFTEs?\b", re.IGNORECASE),
    re.compile(r"\bfull[-\s]time equivalents?\b", re.IGNORECASE),
]
_SUSTAINABILITY_PATTERNS = [
    re.compile(r"\b(target|goal|aim|ambition|commit(?:ment|ted)?|plan)\b", re.IGNORECASE),
    re.compile(r"\b(net[-\s]?zero|greenhouse gas|ghg|co2e|co₂e|scope [123])\b", re.IGNORECASE),
    re.compile(r"\b20\d{2}\b", re.IGNORECASE),
]


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text, disallowed_special=()))


def parse_pdf(path: Path) -> Iterator[tuple[int, str]]:
    parsed = parse_pdf_pages(
        path,
        parser=settings.pdf_parser,
        processed_dir=settings.get_processed_path(),
        llama_cloud_api_key=settings.llama_cloud_api_key.get_secret_value(),
    )
    return as_page_tuples(parsed.pages)


def _processed_pages_path(source: str, processed_dir: Path | None = None) -> Path:
    root = processed_dir or settings.get_processed_path()
    return root / "pages" / f"{Path(source).stem}.jsonl"


def persist_parsed_pages(
    pages: list[tuple[int, str]],
    *,
    source: str,
    company: str | None,
    year: int | None,
    parser: str,
    processed_dir: Path | None = None,
) -> Path:
    out_path = _processed_pages_path(source, processed_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for page, text in pages:
            record = {
                "id": f"{source}:{page}",
                "source": source,
                "company": company,
                "year": year,
                "page": page,
                "parser": parser,
                "text": text,
                "char_count": len(text),
                "token_count": _count_tokens(text),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def _processed_datapoints_path(source: str, processed_dir: Path | None = None) -> Path:
    root = processed_dir or settings.get_processed_path()
    return root / "datapoints" / f"{Path(source).stem}.json"


def _processed_chunks_path(source: str, processed_dir: Path | None = None) -> Path:
    root = processed_dir or settings.get_processed_path()
    return root / "chunks" / f"{Path(source).stem}.jsonl"


def _fte_verbatim_text(text: str) -> str | None:
    lines = text.splitlines()
    matching_indices = [
        i for i, line in enumerate(lines) if any(pattern.search(line) for pattern in _FTE_PATTERNS)
    ]
    if not matching_indices:
        return None

    selected: list[str] = []
    seen: set[int] = set()
    for idx in matching_indices:
        for context_idx in (idx - 1, idx, idx + 1):
            if context_idx < 0 or context_idx >= len(lines) or context_idx in seen:
                continue
            line = lines[context_idx].strip()
            if line:
                selected.append(line)
                seen.add(context_idx)
    return "\n".join(selected) if selected else None


def extract_fte_candidates(
    pages: list[tuple[int, str]],
    *,
    source: str,
    company: str | None,
    year: int | None,
    parser: str,
) -> list[dict[str, str | int | None]]:
    candidates: list[dict[str, str | int | None]] = []
    for page, text in pages:
        verbatim_text = _fte_verbatim_text(text)
        if verbatim_text is None:
            continue
        candidates.append(
            {
                "source": source,
                "company": company,
                "year": year,
                "datapoint_type": "fte_candidate",
                "page": page,
                "verbatim_text": verbatim_text,
                "parser": parser,
            }
        )
    return candidates


def _is_sustainability_goal_text(text: str) -> bool:
    return all(pattern.search(text) for pattern in _SUSTAINABILITY_PATTERNS)


def extract_datapoint_candidates(
    chunks: list[Chunk],
    *,
    parser: str,
) -> list[dict[str, str | int | None]]:
    candidates: list[dict[str, str | int | None]] = []
    seen: set[tuple[str, int, str, str]] = set()
    for chunk in chunks:
        if chunk.chunk_kind == "table":
            continue
        datapoint_type: str | None = None
        if any(pattern.search(chunk.text) for pattern in _FTE_PATTERNS):
            datapoint_type = "fte_candidate"
        elif _is_sustainability_goal_text(chunk.text):
            datapoint_type = "sustainability_goal_candidate"

        if datapoint_type is None:
            continue

        verbatim_text = chunk.text.strip()
        key = (datapoint_type, chunk.page, chunk.source, verbatim_text)
        if not verbatim_text or key in seen:
            continue
        seen.add(key)
        candidates.append(
            {
                "source": chunk.source,
                "company": chunk.company,
                "year": chunk.year,
                "datapoint_type": datapoint_type,
                "page": chunk.page,
                "section_path": chunk.section_path,
                "verbatim_text": verbatim_text,
                "parser": parser,
            }
        )
    return candidates


def persist_datapoints(
    datapoints: list[dict[str, str | int | None]],
    *,
    source: str,
    processed_dir: Path | None = None,
) -> Path:
    out_path = _processed_datapoints_path(source, processed_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(datapoints, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def persist_chunks(
    chunks: list[Chunk],
    *,
    source: str,
    processed_dir: Path | None = None,
) -> Path:
    out_path = _processed_chunks_path(source, processed_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
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
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def _split_oversize(text: str, max_tokens: int, overlap: int) -> list[str]:
    tokens = _ENCODING.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return [text]
    if overlap >= max_tokens:
        overlap = max_tokens // 4
    step = max_tokens - overlap
    pieces: list[str] = []
    start = 0
    while start < len(tokens):
        window = tokens[start : start + max_tokens]
        pieces.append(_ENCODING.decode(window))
        if start + max_tokens >= len(tokens):
            break
        start += step
    return pieces


def build_chunks(
    pages: Iterator[tuple[int, str]],
    *,
    source: str,
    company: str | None,
    year: int | None,
    max_tokens: int,
    overlap: int,
    parser: str | None = None,
) -> list[Chunk]:
    cap = min(max_tokens, EMBEDDING_MAX_TOKENS)
    return build_semantic_chunks(
        pages,
        source=source,
        company=company,
        year=year,
        parser=parser,
        max_tokens=cap,
        overlap=overlap,
        token_counter=_count_tokens,
        split_oversize=_split_oversize,
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
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
    client = chromadb.PersistentClient(path=str(settings.get_chroma_path()))
    name = settings.chroma_collection
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _chunk_metadata(chunk: Chunk) -> dict[str, str | int]:
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
    return md


def _source_where(source: str, company: str | None, year: int | None) -> dict:
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
    try:
        collection.delete(where=_source_where(source, company, year))
    except Exception as exc:  # noqa: BLE001
        logger.info("could not delete existing chunks for %s: %s", source, exc)


def build_datapoint_chunks(
    datapoints: list[dict[str, str | int | None]],
    *,
    source: str,
    company: str | None,
    year: int | None,
    parser: str | None,
    existing_chunks: list[Chunk],
) -> list[Chunk]:
    next_idx_by_page: dict[int, int] = {}
    for chunk in existing_chunks:
        idx = int(chunk.id.rsplit(":", 1)[-1])
        next_idx_by_page[chunk.page] = max(next_idx_by_page.get(chunk.page, 0), idx + 1)

    chunks: list[Chunk] = []
    for datapoint in datapoints:
        page = int(datapoint["page"] or 1)
        idx = next_idx_by_page.get(page, 0)
        next_idx_by_page[page] = idx + 1
        datapoint_type = str(datapoint["datapoint_type"])
        text = str(datapoint["verbatim_text"]).strip()
        section_path = datapoint.get("section_path")
        embedding_parts = [
            company,
            str(year) if year is not None else None,
            parser,
            "datapoint",
            datapoint_type,
            str(section_path) if section_path else None,
            text,
        ]
        chunks.append(
            Chunk(
                id=f"{source}:{page}:{idx}",
                source=source,
                company=company,
                year=year,
                page=page,
                text=text,
                token_count=_count_tokens(text),
                parser=parser,
                chunk_kind="datapoint",
                section_path=str(section_path) if section_path else datapoint_type,
                embedding_text="\n".join(part for part in embedding_parts if part),
            )
        )
    return chunks


def ingest_pdf(
    pdf_path: Path,
    *,
    company: str | None,
    year: int | None,
    reset: bool = False,
    parser: str | None = None,
    source_name: str | None = None,
) -> int:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    source = source_name or pdf_path.name
    requested_parser = parser or settings.pdf_parser
    logger.info("parsing %s with %s", source, requested_parser)
    processed_dir = settings.get_processed_path()
    parsed = parse_pdf_pages(
        pdf_path,
        parser=requested_parser,
        processed_dir=processed_dir,
        llama_cloud_api_key=settings.llama_cloud_api_key.get_secret_value(),
    )
    pages = list(as_page_tuples(parsed.pages))
    logger.info("parsed %d non-empty pages with %s", len(pages), parsed.parser)
    pages_path = persist_parsed_pages(
        pages,
        source=source,
        company=company,
        year=year,
        parser=parsed.parser,
        processed_dir=processed_dir,
    )
    logger.info("wrote parsed pages to %s", pages_path)
    chunks = build_chunks(
        iter(pages),
        source=source,
        company=company,
        year=year,
        max_tokens=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
        parser=parsed.parser,
    )
    logger.info("built %d semantic chunks", len(chunks))

    datapoints = extract_datapoint_candidates(chunks, parser=parsed.parser)
    datapoints_path = persist_datapoints(
        datapoints,
        source=source,
        processed_dir=processed_dir,
    )
    logger.info(
        "wrote %d pre-extracted candidate datapoints to %s",
        len(datapoints),
        datapoints_path,
    )

    datapoint_chunks = build_datapoint_chunks(
        datapoints,
        source=source,
        company=company,
        year=year,
        parser=parsed.parser,
        existing_chunks=chunks,
    )
    chunks.extend(datapoint_chunks)
    logger.info("built %d total chunks including datapoints", len(chunks))
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
            metadatas=[_chunk_metadata(c) for c in batch_chunks],
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
    parser.add_argument(
        "--source-name",
        default=None,
        help="stable source name to store in chunks/citations (defaults to PDF filename)",
    )
    parser.add_argument("--reset", action="store_true", help="drop and recreate the collection")
    parser.add_argument(
        "--parser",
        choices=["llamaparse"],
        default=settings.pdf_parser,
        help="PDF parser to use before chunking",
    )
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
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ingested {n} chunks from {args.pdf.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
