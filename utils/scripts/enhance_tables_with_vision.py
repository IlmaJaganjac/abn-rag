#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.ingest.table_vision import (
    PageTableExtraction,
    enhance_page_tables,
    tables_to_text,
)
from backend.app.ingest.vision_selection import classify_table_complexity

log = logging.getLogger(__name__)


def _parse_page_range(s: str) -> list[int]:
    pages: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


def _load_pages(path: Path) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pages.append(json.loads(line))
    return pages


def _render_page(pdf_path: Path, page_1based: int, dpi: int) -> bytes:
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


def _enhance_with_retry(
    *,
    pdf_path: Path,
    record: dict[str, Any],
    company: str | None,
    year: int | None,
    source_name: str | None,
    model: str,
    detail: str,
    dpi: int,
    max_attempts: int = 3,
) -> PageTableExtraction:
    page_num = int(record["page"])
    eff_company = company or record.get("company")
    eff_year = year or record.get("year")
    eff_source = source_name or record.get("source")

    _TOKEN_LIMIT_PHRASES = ("length limit was reached", "max_tokens", "finish_reason")

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            image_bytes = _render_page(pdf_path, page_num, dpi)
            return enhance_page_tables(
                image_bytes=image_bytes,
                page=page_num,
                company=eff_company,
                year=eff_year,
                source=eff_source,
                page_text=record.get("text"),
                model=model,
                detail=detail,
            )
        except Exception as exc:
            last_exc = exc
            exc_str = str(exc)
            if any(phrase in exc_str for phrase in _TOKEN_LIMIT_PHRASES):
                log.warning("page %d token limit hit, skipping retries", page_num)
                raise
            if attempt < max_attempts - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                log.warning("page %d attempt %d/%d failed (%s), retrying in %.1fs",
                            page_num, attempt + 1, max_attempts, exc, delay)
                time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _strip_markdown_tables(text: str) -> str:
    """Remove contiguous markdown-table line blocks from text.

    Lines starting and ending with '|' (with >=2 pipes) are dropped, along with
    their separator rows. Surrounding narrative is preserved.
    """
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


def _apply_enhancement(record: dict[str, Any], extraction: PageTableExtraction, model: str) -> dict[str, Any]:
    tables_dicts = [t.model_dump() for t in extraction.tables]
    out = dict(record)
    out["tables"] = tables_dicts
    out["table_enhanced"] = extraction.has_tables and bool(tables_dicts)
    out["table_enhancement_model"] = model if out["table_enhanced"] else None
    out["table_enhancement_error"] = None
    if out["table_enhanced"] and tables_dicts:
        summary = tables_to_text(tables_dicts).strip()
        if summary:
            narrative = _strip_markdown_tables(record.get("text") or "")
            out["enhanced_text"] = (narrative + "\n\n" + summary).strip() if narrative else summary
    return out


def _apply_empty(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out["tables"] = []
    out["table_enhanced"] = False
    out["table_enhancement_model"] = None
    out["table_enhancement_error"] = None
    return out


def _apply_error(record: dict[str, Any], error: str) -> dict[str, Any]:
    out = _apply_empty(record)
    out["table_enhancement_error"] = error
    return out


def run(
    pdf_path: Path,
    input_jsonl: Path,
    output_jsonl: Path,
    *,
    company: str | None,
    year: int | None,
    source_name: str | None,
    model: str,
    detail: str,
    max_pages: int,
    pages_override: list[int] | None,
    dpi: int,
    max_workers: int,
    max_attempts: int,
    dry_run: bool,
    force: bool,
) -> int:
    if output_jsonl.exists() and not force:
        log.error("output %s already exists; pass --force to overwrite", output_jsonl)
        return 1

    pages = _load_pages(input_jsonl)
    log.info("loaded %d pages", len(pages))

    page_map: dict[int, dict[str, Any]] = {int(p["page"]): p for p in pages}

    # scored_candidates: list of (page_num, kind, score)
    scored_candidates: list[tuple[int, str, float]] = []

    if pages_override is not None:
        for page_num in pages_override:
            if page_num not in page_map:
                continue
            kind, score = classify_table_complexity(page_map[page_num].get("text", ""))
            scored_candidates.append((page_num, kind or "override", score))
        log.info("using --pages override: %d pages", len(scored_candidates))
    else:
        for p in pages:
            kind, score = classify_table_complexity(p.get("text", ""))
            if kind != "skip":
                scored_candidates.append((int(p["page"]), kind, score))
        log.info("detected %d complex-table candidate pages (vision needed)", len(scored_candidates))

    # rank by score descending
    scored_candidates.sort(key=lambda x: x[2], reverse=True)

    if max_pages > 0 and len(scored_candidates) > max_pages:
        scored_candidates = scored_candidates[:max_pages]
        log.info("limited to top %d pages by score (--max-pages)", max_pages)

    selected = [c[0] for c in scored_candidates]
    selected_set = set(selected)

    print(f"\nSelected {len(selected)} pages needing vision (complex table types):")
    for page_num, kind, score in scored_candidates[:50]:
        print(f"  page {page_num:4d}  [{kind:<20}]  score={score:.1f}")
    if len(scored_candidates) > 50:
        print(f"  ... and {len(scored_candidates) - 50} more")
    print()

    if dry_run:
        log.info("dry-run complete — no images rendered, no vision calls made, no output written")
        return 0

    # Cache dir: one JSON file per page for incremental saves + resume support
    cache_dir = output_jsonl.with_suffix(".cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(page_num: int) -> Path:
        return cache_dir / f"page_{page_num}.json"

    def _save_to_cache(page_num: int, record: dict[str, Any]) -> None:
        _cache_path(page_num).write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    def _load_from_cache(page_num: int) -> dict[str, Any] | None:
        p = _cache_path(page_num)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
        return None

    # Skip pages already cached from a previous run
    to_process = []
    results: dict[int, dict[str, Any]] = {}
    for page_num in selected:
        cached = _load_from_cache(page_num)
        if cached is not None:
            results[page_num] = cached
            log.info("page %d loaded from cache", page_num)
        else:
            to_process.append(page_num)

    if results:
        log.info("resumed %d pages from cache, %d remaining", len(results), len(to_process))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _enhance_with_retry,
                pdf_path=pdf_path,
                record=page_map[page_num],
                company=company,
                year=year,
                source_name=source_name,
                model=model,
                detail=detail,
                dpi=dpi,
                max_attempts=max_attempts,
            ): page_num
            for page_num in to_process
        }
        for future in as_completed(futures):
            page_num = futures[future]
            try:
                extraction = future.result()
                out_record = _apply_enhancement(page_map[page_num], extraction, model)
                n_tables = len(extraction.tables)
                log.info("page %d enhanced: %d tables", page_num, n_tables)
            except Exception as exc:
                log.warning("page %d failed after retries: %s", page_num, exc)
                out_record = _apply_error(page_map[page_num], str(exc))
            results[page_num] = out_record
            _save_to_cache(page_num, out_record)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with output_jsonl.open("w", encoding="utf-8") as f:
        for p in pages:
            page_num = int(p["page"])
            if page_num in selected_set:
                row = results.get(page_num) or _apply_error(p, "no result returned")
            else:
                row = _apply_empty(p)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    enhanced_count = sum(1 for r in results.values() if r.get("table_enhanced"))
    log.info("output saved → %s (%d/%d pages successfully enhanced)",
             output_jsonl, enhanced_count, len(selected))
    print(f"\nOutput saved → {output_jsonl}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enhance table pages in a pages JSONL with OpenAI vision.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    parser.add_argument("--company", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="high")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="max number of detected pages to process (0 = no limit)")
    parser.add_argument("--pages", default=None,
                        help="explicit page selection, e.g. '75,87,89,98,360' or '75-100,200'")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="overwrite output file if it already exists")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from backend.app.config import settings as _settings
    model = args.model or _settings.openai_table_vision_model
    pages_override = _parse_page_range(args.pages) if args.pages else None

    return run(
        pdf_path=args.pdf,
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        company=args.company,
        year=args.year,
        source_name=args.source_name,
        model=model,
        detail=args.detail,
        max_pages=args.max_pages,
        pages_override=pages_override,
        dpi=args.dpi,
        max_workers=args.max_workers,
        max_attempts=args.max_attempts,
        dry_run=args.dry_run,
        force=args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
