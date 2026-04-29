from __future__ import annotations

import argparse
import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
import tiktoken

from backend.app._openai import openai_client
from backend.app.config import settings
from backend.app.schemas import Chunk

logger = logging.getLogger(__name__)

EMBEDDING_MAX_TOKENS = 8191
_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text, disallowed_special=()))


def parse_pdf(path: Path) -> Iterator[tuple[int, str]]:
    doc = fitz.open(path)
    try:
        for i in range(doc.page_count):
            text = doc.load_page(i).get_text("text").strip()
            if text:
                yield i + 1, text
    finally:
        doc.close()


# Matches standalone metric values: 52.8%, €8.5bn, > 44,000, 5,100, 535
# Requires a unit/symbol OR ≥3 digits to exclude bare page numbers (1, 5, 14…)
_KPI_VALUE_RE = re.compile(
    r"^(?:"
    r"(?:>|≥|~)\s*(?:€|\$)?\s*[\d,]+(?:\.\d+)?\s*(?:%|bn|m|kt)?"  # > or ≥ prefix
    r"|(?:€|\$)\s*[\d,]+(?:\.\d+)?\s*(?:bn|m|kt)?"                  # currency prefix
    r"|[\d,]+(?:\.\d+)?\s*(?:%|bn|m|kt)"                             # number + unit
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"                                  # comma-thousands: 5,100 / 44,000
    r"|\d{3,}(?:\.\d+)?"                                              # plain int ≥3 digits: 535
    r")\s*$",
    re.IGNORECASE,
)
# Lines that are clearly numeric-only (table values, not labels)
_NUMERIC_LINE_RE = re.compile(r"^\s*-?[\d,]+(?:\.\d+)?\s*%?\s*$")


def extract_kpi_facts(
    pages: list[tuple[int, str]],
    *,
    source: str,
    company: str | None,
    year: int | None,
) -> list[Chunk]:
    """Detect value\\nlabel KPI pairs and emit one clean chunk per fact."""
    chunks: list[Chunk] = []
    for page, text in pages:
        lines = [ln.strip() for ln in text.splitlines()]
        i = 0
        kpi_idx = 0
        while i < len(lines):
            line = lines[i]
            if not _KPI_VALUE_RE.match(line) or not line:
                i += 1
                continue
            value = line.strip()
            # collect label: next non-empty line(s) that aren't numeric
            label_parts: list[str] = []
            j = i + 1
            while j < len(lines):
                candidate = lines[j].strip()
                if not candidate:
                    j += 1
                    continue
                if _NUMERIC_LINE_RE.match(candidate) or _KPI_VALUE_RE.match(candidate):
                    break
                if len(candidate) > 80:
                    break
                label_parts.append(candidate)
                j += 1
                # allow one continuation line (e.g. "(headcount)")
                if len(label_parts) == 2:
                    break
            if not label_parts:
                i += 1
                continue
            label = " ".join(label_parts)
            fact_text = f"{label}: {value}"
            chunks.append(
                Chunk(
                    id=f"{source}:{page}:kpi:{kpi_idx}",
                    source=source,
                    company=company,
                    year=year,
                    page=page,
                    text=fact_text,
                    token_count=_count_tokens(fact_text),
                )
            )
            kpi_idx += 1
            i += 1
    return chunks


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
) -> list[Chunk]:
    cap = min(max_tokens, EMBEDDING_MAX_TOKENS)
    chunks: list[Chunk] = []
    for page, text in pages:
        parts = _split_oversize(text, cap, overlap)
        for idx, part in enumerate(parts):
            chunks.append(
                Chunk(
                    id=f"{source}:{page}:{idx}",
                    source=source,
                    company=company,
                    year=year,
                    page=page,
                    text=part,
                    token_count=_count_tokens(part),
                )
            )
    return chunks


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
    return md


def ingest_pdf(
    pdf_path: Path,
    *,
    company: str | None,
    year: int | None,
    reset: bool = False,
) -> int:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    source = pdf_path.name
    logger.info("parsing %s", source)
    pages = list(parse_pdf(pdf_path))
    logger.info("parsed %d non-empty pages", len(pages))

    chunks = build_chunks(
        iter(pages),
        source=source,
        company=company,
        year=year,
        max_tokens=settings.chunk_size_tokens,
        overlap=settings.chunk_overlap_tokens,
    )
    kpi_chunks = extract_kpi_facts(pages, source=source, company=company, year=year)
    chunks.extend(kpi_chunks)
    logger.info("built %d chunks (%d kpi facts)", len(chunks), len(kpi_chunks))

    embeddings = embed_texts([c.text for c in chunks])
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"embedding count {len(embeddings)} != chunk count {len(chunks)}"
        )

    collection = get_collection(reset=reset)
    collection.upsert(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[_chunk_metadata(c) for c in chunks],
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
    parser.add_argument("--reset", action="store_true", help="drop and recreate the collection")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        n = ingest_pdf(args.pdf, company=args.company, year=args.year, reset=args.reset)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"ingested {n} chunks from {args.pdf.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
