from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from backend.app.config import settings
from backend.app.ingest.persistence import (
    persist_enhanced_pages,
    processed_pages_enhanced_path,
)
from backend.app.ingest.tokens import count_tokens
from backend.app.ingest.table_vision import (
    PageTableExtraction,
    classify_table_complexity,
    enhance_page_tables,
    tables_to_text,
)

logger = logging.getLogger(__name__)

_VISION_MAX_PAGES = 50
_VISION_MAX_ATTEMPTS = 2
_VISION_DETAIL = "high"
_VISION_DPI = 180

_VISION_WORKERS = 6


def render_page(pdf_path: Path, page_1based: int, dpi: int) -> bytes:
    """Render one PDF page to PNG bytes and return the image payload."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PyMuPDF not installed: pip install pymupdf") from exc
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_1based - 1]
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        return pix.tobytes("png")
    finally:
        doc.close()


def strip_markdown_tables(text: str) -> str:
    """Remove markdown table rows from text and return the remaining narrative."""
    out_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
            continue
        out_lines.append(line)
    cleaned: list[str] = []
    blank = False
    for line in out_lines:
        if line.strip():
            cleaned.append(line)
            blank = False
        elif not blank:
            cleaned.append(line)
            blank = True
    return "\n".join(cleaned).strip()


def apply_vision_enhancement(
    record: dict[str, Any], extraction: PageTableExtraction, model: str
) -> dict[str, Any]:
    """Apply successful table extraction output to one persisted page record."""
    tables_dicts = [table.model_dump() for table in extraction.tables]
    out = dict(record)
    out["tables"] = tables_dicts
    out["table_enhanced"] = extraction.has_tables and bool(tables_dicts)
    out["table_enhancement_model"] = model if out["table_enhanced"] else None
    out["table_enhancement_error"] = None
    if out["table_enhanced"] and tables_dicts:
        summary = tables_to_text(tables_dicts).strip()
        if summary:
            narrative = strip_markdown_tables(record.get("text") or "")
            out["enhanced_text"] = (narrative + "\n\n" + summary).strip() if narrative else summary
    return out


def apply_empty_enhancement(record: dict[str, Any]) -> dict[str, Any]:
    """Return a page record marked as processed but without usable table enhancement."""
    out = dict(record)
    out["tables"] = []
    out["table_enhanced"] = False
    out["table_enhancement_model"] = None
    out["table_enhancement_error"] = None
    return out


def apply_enhancement_error(record: dict[str, Any], error: str) -> dict[str, Any]:
    """Return a page record annotated with a table-enhancement error message."""
    out = apply_empty_enhancement(record)
    out["table_enhancement_error"] = error
    return out


def enhance_page_record_with_retry(
    *,
    pdf_path: Path,
    record: dict[str, Any],
    company: str | None,
    year: int | None,
    source_name: str | None,
    model: str,
    detail: str,
    dpi: int,
    max_attempts: int,
) -> PageTableExtraction:
    """Try table vision for one page record with retries and return the extraction result."""
    page_num = int(record["page"])
    last_exc: Exception | None = None
    token_limit_phrases = ("length limit was reached", "max_tokens", "finish_reason")
    for attempt in range(max_attempts):
        try:
            image_bytes = render_page(pdf_path, page_num, dpi)
            return enhance_page_tables(
                image_bytes=image_bytes,
                page=page_num,
                company=company or record.get("company"),
                year=year or record.get("year"),
                source=source_name or record.get("source"),
                page_text=record.get("text"),
                model=model,
                detail=detail,
            )
        except Exception as exc:
            last_exc = exc
            if any(phrase in str(exc) for phrase in token_limit_phrases):
                raise
            if attempt < max_attempts - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def enhance_pages_with_vision(
    *,
    pdf_path: Path,
    pages: list[tuple[int, str]],
    source: str,
    company: str | None,
    year: int | None,
    parser: str | None = None,
    processed_dir: Path | None = None,
    model: str | None = None,
) -> list[tuple[int, str]]:
    """Enhance selected pages with table vision and return updated `(page, text)` tuples."""
    cached_path = processed_pages_enhanced_path(source, processed_dir)
    if cached_path.exists():
        import json as _json
        cached_pages: list[tuple[int, str]] = []
        with cached_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = _json.loads(line)
                cached_pages.append((int(rec["page"]), str(rec.get("enhanced_text") or rec.get("text") or "")))
        logger.info("loaded %d cached vision-enhanced pages from %s", len(cached_pages), cached_path)
        return cached_pages

    vision_model = model or settings.openai_table_vision_model
    records = [
        {
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
        for page, text in pages
    ]
    scored_candidates: list[tuple[int, str, float]] = []
    page_map = {int(record["page"]): record for record in records}
    for record in records:
        kind, score = classify_table_complexity(record.get("text", ""))
        if kind != "skip":
            scored_candidates.append((int(record["page"]), kind, score))
    scored_candidates.sort(key=lambda item: item[2], reverse=True)
    selected = [page for page, _, _ in scored_candidates[:_VISION_MAX_PAGES]]
    selected_set = set(selected)

    results: dict[int, dict[str, Any]] = {}
    max_workers = max(1, min(_VISION_WORKERS, len(selected))) if selected else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                enhance_page_record_with_retry,
                pdf_path=pdf_path,
                record=page_map[page_num],
                company=company,
                year=year,
                source_name=source,
                model=vision_model,
                detail=_VISION_DETAIL,
                dpi=_VISION_DPI,
                max_attempts=_VISION_MAX_ATTEMPTS,
            ): page_num
            for page_num in selected
        }
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                extraction = future.result()
                results[page_num] = apply_vision_enhancement(page_map[page_num], extraction, vision_model)
            except Exception as exc:
                results[page_num] = apply_enhancement_error(page_map[page_num], str(exc))

    enhanced_records: list[dict[str, Any]] = []
    enhanced_pages: list[tuple[int, str]] = []
    for record in records:
        page_num = int(record["page"])
        if page_num in selected_set:
            row = results.get(page_num) or apply_enhancement_error(record, "no result returned")
        else:
            row = apply_empty_enhancement(record)
        enhanced_records.append(row)
        enhanced_pages.append((page_num, str(row.get("enhanced_text") or row.get("text") or "")))

    persist_enhanced_pages(
        enhanced_records,
        source=source,
        processed_dir=processed_dir,
    )
    return enhanced_pages
