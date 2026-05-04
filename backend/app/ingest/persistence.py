from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from backend.app.config import settings
from backend.app.ingest.tokens import count_tokens
from backend.app.schemas import Chunk

logger = logging.getLogger(__name__)


def _root(processed_dir: Path | None) -> Path:
    """Return the effective processed-data root directory for persistence helpers."""
    return processed_dir or settings.get_processed_path()


def processed_pages_path(source: str, processed_dir: Path | None = None) -> Path:
    """Return the JSONL path used to store parsed pages for one source."""
    return _root(processed_dir) / "pages" / f"{Path(source).stem}.jsonl"


def processed_pages_enhanced_path(source: str, processed_dir: Path | None = None) -> Path:
    """Return the JSONL path used to store vision-enhanced pages for one source."""
    return _root(processed_dir) / "pages_enhanced" / f"{Path(source).stem}.jsonl"


def processed_datapoints_path(source: str, processed_dir: Path | None = None) -> Path:
    """Return the JSON path used to store extracted datapoints for one source."""
    return _root(processed_dir) / "datapoints" / f"{Path(source).stem}.json"


def processed_chunks_path(source: str, processed_dir: Path | None = None) -> Path:
    """Return the JSONL path used to store retrieval chunks for one source."""
    return _root(processed_dir) / "chunks" / f"{Path(source).stem}.jsonl"


def persist_parsed_pages(
    pages: list[tuple[int, str]],
    *,
    source: str,
    company: str | None,
    year: int | None,
    parser: str,
    processed_dir: Path | None = None,
) -> Path:
    """Write parsed page text to JSONL and return the output path."""
    out_path = processed_pages_path(source, processed_dir)
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
                "token_count": count_tokens(text),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def persist_enhanced_pages(
    records: list[dict[str, Any]],
    *,
    source: str,
    processed_dir: Path | None = None,
) -> Path:
    """Write vision-enhanced page records to JSONL and return the output path."""
    out_path = processed_pages_enhanced_path(source, processed_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def persist_datapoints(
    datapoints: list[object],
    *,
    source: str,
    processed_dir: Path | None = None,
) -> Path:
    """Write extracted datapoints to JSON and return the output path."""
    out_path = processed_datapoints_path(source, processed_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable: list[object] = []
    for datapoint in datapoints:
        model_dump = getattr(datapoint, "model_dump", None)
        if callable(model_dump):
            serializable.append(model_dump())
        else:
            serializable.append(datapoint)
    out_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def persist_chunks(
    chunks: list[Chunk],
    *,
    source: str,
    processed_dir: Path | None = None,
) -> Path:
    """Write retrieval chunks to JSONL and return the output path."""
    out_path = processed_chunks_path(source, processed_dir)
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
                "fact_kind": chunk.fact_kind,
                "basis": chunk.basis,
                "scope_type": chunk.scope_type,
                "quality": chunk.quality,
                "validation_status": chunk.validation_status,
                "canonical_metric": chunk.canonical_metric,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def apply_enhanced_text(
    pages: list[tuple[int, str]],
    enhanced_jsonl: Path,
) -> list[tuple[int, str]]:
    """Overlay enhanced page text where available and return updated page tuples."""
    overrides: dict[int, str] = {}
    with enhanced_jsonl.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            enhanced = record.get("enhanced_text")
            if enhanced:
                overrides[int(record["page"])] = enhanced
    if overrides:
        logger.info("applying enhanced_text overrides for %d pages", len(overrides))
    return [(page, overrides.get(page, text)) for page, text in pages]


def load_pages_jsonl(path: Path) -> tuple[list[tuple[int, str]], str]:
    """Load persisted pages and return page tuples together with the parser name."""
    pages: list[tuple[int, str]] = []
    parser: str | None = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("enhanced_text") or record.get("text")
            if not text:
                continue
            pages.append((int(record["page"]), str(text)))
            if parser is None and record.get("parser"):
                parser = str(record["parser"])
    return pages, parser or "pages_jsonl"


def load_datapoints_json(path: Path) -> list[dict[str, Any]]:
    """Load persisted datapoints and return them as plain dictionaries."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("datapoints", [])
    if not isinstance(payload, list):
        raise RuntimeError(f"datapoints JSON must be a list or contain a 'datapoints' list: {path}")
    return [item for item in payload if isinstance(item, dict)]
