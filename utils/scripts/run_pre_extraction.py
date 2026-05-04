#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.extract.page_scanner import load_pages_jsonl, scan_pages
from backend.app.extract.datapoints import (
    deduplicate_datapoints,
    normalize_llamaextract_result,
    save_datapoint_set,
)
from backend.app.extract.schemas import (
    AnnualReportDatapoints,
    extract_annual_report_datapoints,
)
from backend.app.extract.openai_extract import extract_annual_report_datapoints_openai
from backend.app.extract.openai_validate import validate_datapoints_openai

_PRE_EXTRACTED_ROOT = Path("backend/data/processed/pre_extracted")
_RAW_AUDIT_ROOT = Path("backend/data/processed/pre_extracted_raw")
_PAGE_PLANS_ROOT = Path("backend/data/processed/extraction_page_plans")


def _build_raw_audit_payload(
    source: str,
    company: str | None,
    year: int | None,
    extractor: str,
    categories: list[str],
    raw_by_cat: dict[str, list[dict]],
) -> dict:
    return {
        "source": source,
        "company": company,
        "year": year,
        "extractor": extractor,
        "categories": categories,
        "raw": raw_by_cat,
    }


def _save_raw_audit(
    stem: str,
    source: str,
    company: str | None,
    year: int | None,
    extractor: str,
    categories: list[str],
    raw_by_cat: dict[str, list[dict]],
) -> Path:
    path = _RAW_AUDIT_ROOT / f"{stem}.openai.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_raw_audit_payload(source, company, year, extractor, categories, raw_by_cat)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _preview(text: str | None, n: int = 70) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:n] + ("…" if len(text) > n else "")


def _filter_category(cat: str, result: AnnualReportDatapoints) -> AnnualReportDatapoints:
    if cat == "fte":
        return AnnualReportDatapoints(
            company=result.company, year=result.year,
            fte_datapoints=result.fte_datapoints,
        )
    if cat == "sustainability":
        return AnnualReportDatapoints(
            company=result.company, year=result.year,
            sustainability_goals=result.sustainability_goals,
        )
    if cat == "esg":
        return AnnualReportDatapoints(
            company=result.company, year=result.year,
            esg_datapoints=result.esg_datapoints,
        )
    if cat == "financial_highlight":
        return AnnualReportDatapoints(
            company=result.company, year=result.year,
            financial_highlights=result.financial_highlights,
        )
    if cat == "business_performance":
        return AnnualReportDatapoints(
            company=result.company, year=result.year,
            business_performance=result.business_performance,
        )
    if cat == "shareholder_return":
        return AnnualReportDatapoints(
            company=result.company, year=result.year,
            shareholder_returns=result.shareholder_returns,
        )
    return result


def parse_page_range_to_pages(page_range: str) -> list[int]:
    pages: list[int] = []
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            pages.extend(range(int(start), int(end) + 1))
        else:
            pages.append(int(part))
    return sorted(set(pages))


def pages_to_range(pages: list[int]) -> str:
    if not pages:
        return ""
    pages = sorted(set(pages))
    parts: list[str] = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            parts.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = p
    parts.append(str(start) if start == prev else f"{start}-{prev}")
    return ",".join(parts)


def chunk_pages(pages: list[int], size: int) -> list[list[int]]:
    return [pages[i : i + size] for i in range(0, len(pages), size)]


def _run_category(
    cat: str,
    pdf: Path,
    pages: list[dict],
    company: str | None,
    year: int | None,
    source: str,
    page_range: str,
    extractor: str,
    batch_pages: int = 0,
) -> tuple[str, list, list[dict]]:
    log = logging.getLogger(__name__)
    log.info("running category=%s pages=%s", cat, page_range)

    if extractor == "openai" and batch_pages > 0:
        all_page_nums = parse_page_range_to_pages(page_range)
        batches = chunk_pages(all_page_nums, batch_pages)
        normalized: list = []
        raw_batches: list[dict] = []
        for i, batch in enumerate(batches, 1):
            batch_range = pages_to_range(batch)
            log.info("category=%s batch=%d/%d pages=%s", cat, i, len(batches), batch_range)
            result = extract_annual_report_datapoints_openai(
                pages=pages,
                company=company,
                year=year,
                page_range=batch_range,
                category=cat,
            )
            raw_batches.append(result.model_dump(mode="json"))
            filtered = _filter_category(cat, result)
            batch_normalized = normalize_llamaextract_result(
                filtered,
                source=source,
                company=company,
                year=year,
                extractor=extractor,
            )
            normalized.extend(batch_normalized)
        log.info("category=%s: %d datapoints extracted across %d batches", cat, len(normalized), len(batches))
        return cat, normalized, raw_batches

    if extractor == "openai":
        result = extract_annual_report_datapoints_openai(
            pages=pages,
            company=company,
            year=year,
            page_range=page_range,
            category=cat,
        )
        raw_batches = [result.model_dump(mode="json")]
    else:
        result = extract_annual_report_datapoints(
            pdf,
            company=company,
            year=year,
            page_range=page_range,
            category=cat,
        )
        raw_batches = []
    filtered = _filter_category(cat, result)
    normalized = normalize_llamaextract_result(
        filtered,
        source=source,
        company=company,
        year=year,
        extractor=extractor,
    )
    log.info("category=%s: %d datapoints extracted", cat, len(normalized))
    return cat, normalized, raw_batches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run LlamaExtract pre-extraction pipeline.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("pages_jsonl", type=Path)
    parser.add_argument("--company", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--context-window", type=int, default=1)
    parser.add_argument("--max-pages-per-category", type=int, default=20)
    parser.add_argument(
        "--categories",
        default="fte,sustainability,financial_highlight,business_performance,shareholder_return",
        help="comma-separated list of categories to extract",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="max parallel extraction jobs (default: one worker per category)",
    )
    parser.add_argument(
        "--extractor",
        choices=["llamaextract", "openai"],
        default="llamaextract",
        help="structured extractor backend to use",
    )
    parser.add_argument(
        "--batch-pages",
        type=int,
        default=1,
        help="when --extractor openai: split page range into batches of N pages (default: 1 page per batch)",
    )
    parser.add_argument(
        "--validate-extracted",
        action="store_true",
        default=False,
        help="run an OpenAI validation/deduplication pass on extracted datapoints per category",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    source = args.source_name or args.pdf.name
    stem = Path(source).stem
    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    max_workers = args.max_workers if args.max_workers is not None else max(1, len(categories))

    # 1. Load parsed pages
    log.info("loading pages from %s", args.pages_jsonl)
    pages = load_pages_jsonl(args.pages_jsonl)
    log.info("loaded %d pages", len(pages))

    # 2. Scan and score
    scan_result, page_ranges = scan_pages(
        pages,
        context_window=args.context_window,
        max_pages_per_category=args.max_pages_per_category,
    )

    # Save page plan
    plan_path = _PAGE_PLANS_ROOT / f"{stem}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(page_ranges, indent=2), encoding="utf-8")
    print(f"\nPage plan saved → {plan_path}")
    print("\n=== Page ranges per category ===")
    for cat, rng in page_ranges.items():
        if cat in categories:
            print(f"  {cat}: {rng or '(none)'}")

    # Determine which categories have page ranges
    active: list[tuple[str, str]] = []
    for cat in categories:
        rng = page_ranges.get(cat, "")
        if not rng:
            log.warning("no candidate pages for category %s, skipping", cat)
        else:
            active.append((cat, rng))

    # 3. Extract categories in parallel
    results_by_cat: dict[str, list] = {}
    raw_by_cat: dict[str, list[dict]] = {}
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _run_category,
                cat,
                args.pdf,
                pages,
                args.company,
                args.year,
                source,
                rng,
                args.extractor,
                args.batch_pages,
            ): cat
            for cat, rng in active
        }
        for future in as_completed(futures):
            cat = futures[future]
            try:
                cat_name, dps, raw_batches = future.result()
                results_by_cat[cat_name] = dps
                if raw_batches:
                    raw_by_cat[cat_name] = raw_batches
            except Exception as exc:
                errors.append(f"category={cat}: {exc}")

    if errors:
        for e in errors:
            log.error("extraction failed — %s", e)
        return 1

    log.info("all categories complete")

    # 3b. Save raw audit for OpenAI extractor
    if args.extractor == "openai" and raw_by_cat:
        raw_path = _save_raw_audit(
            stem=stem,
            source=source,
            company=args.company,
            year=args.year,
            extractor=args.extractor,
            categories=categories,
            raw_by_cat=raw_by_cat,
        )
        print(f"\nRaw audit saved → {raw_path}")

    # 3c. Optionally validate/deduplicate per category with OpenAI
    if args.validate_extracted:
        for cat in categories:
            dps = results_by_cat.get(cat)
            if not dps:
                continue
            try:
                items = validate_datapoints_openai(
                    category=cat,
                    datapoints=dps,
                    company=args.company,
                    year=args.year,
                )
                if items:
                    keep_indices = {
                        it.index
                        for it in items
                        if it.is_valid and it.duplicate_of_index is None
                    }
                    kept = [dp for i, dp in enumerate(dps) if i in keep_indices]
                    log.info("category=%s validation: kept %d/%d datapoints", cat, len(kept), len(dps))
                    results_by_cat[cat] = kept
                else:
                    log.warning("category=%s validation returned no items, keeping original", cat)
            except Exception as exc:
                log.warning("category=%s validation failed (%s), keeping original datapoints", cat, exc)

    # 4. Combine in deterministic category order
    all_datapoints = []
    for cat in categories:
        all_datapoints.extend(results_by_cat.get(cat, []))

    # 5. Dedup and prioritize
    deduped = deduplicate_datapoints(all_datapoints)

    # 6. Save
    out_path = args.out or (_PRE_EXTRACTED_ROOT / f"{stem}.json")
    save_datapoint_set(
        deduped,
        source=source,
        company=args.company,
        year=args.year,
        out_path=out_path,
    )
    print(f"\nOutput saved → {out_path}")

    # 7. Summary
    by_type: dict[str, list] = {}
    for dp in deduped:
        by_type.setdefault(dp.datapoint_type, []).append(dp)

    print("\n=== Count by datapoint type ===")
    for dtype, items in sorted(by_type.items()):
        print(f"  {dtype}: {len(items)}")

    print("\n=== Top datapoints by type and priority ===")
    for dtype, items in sorted(by_type.items()):
        ranked = sorted(items, key=lambda d: d.priority, reverse=True)
        print(f"\n  -- {dtype} --")
        for dp in ranked[:10]:
            parts = [
                f"priority={dp.priority}",
                f"value={dp.value or '-'}",
                f"metric={dp.metric}",
            ]
            if dp.unit:
                parts.append(f"unit={dp.unit}")
            if dp.period:
                parts.append(f"period={dp.period}")
            if dp.page:
                parts.append(f"page={dp.page}")
            print("    " + " | ".join(parts))
            if dp.quote:
                print(f"      quote: {_preview(dp.quote)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
