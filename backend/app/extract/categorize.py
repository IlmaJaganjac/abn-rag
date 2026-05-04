from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from backend.app.extract.datapoints import deduplicate_datapoints, normalize_llamaextract_result
from backend.app.extract.openai import (
    extract_annual_report_datapoints_openai,
    validate_datapoints_openai,
)
from backend.app.ingest.categories import (
    CATEGORY_MAX_PAGES,
    CATEGORY_PATTERNS,
    DATAPOINT_CATEGORIES,
    category_page_score,
)

logger = logging.getLogger(__name__)


def extract_categorized_datapoints(
    pages: list[tuple[int, str]],
    *,
    source: str,
    company: str | None,
    year: int | None,
    validate: bool = False,
) -> list[object]:
    """Extract category-specific datapoints and return one merged normalized list."""
    page_records = [
        {
            "source": source,
            "company": company,
            "year": year,
            "page": page,
            "text": text,
        }
        for page, text in pages
        if text.strip()
    ]

    def _pages_label(records: list[dict[str, Any]]) -> str:
        """Render a compact page label string for logging selected page ranges."""
        pages_str = [str(record["page"]) for record in records]
        label = ", ".join(pages_str[:12])
        if len(pages_str) > 12:
            label += f", ... (+{len(pages_str) - 12})"
        return label

    def _extract_category(category: str) -> list[object]:
        """Extract and optionally validate datapoints for one category."""
        max_pages = CATEGORY_MAX_PAGES.get(category, 20)
        if category in CATEGORY_PATTERNS:
            scored = [
                (p, category_page_score(category, p["text"]))
                for p in page_records
            ]
            scored = [(p, s) for p, s in scored if s > 0]
            scored.sort(key=lambda x: x[1], reverse=True)
            candidate_pages = [p for p, _ in scored[:max_pages]] or page_records[:max_pages]
        else:
            candidate_pages = page_records[:max_pages]
        logger.info(
            "category %s: %d/%d pages selected by regex: %s",
            category,
            len(candidate_pages),
            len(page_records),
            _pages_label(candidate_pages),
        )

        def _extract_page(page_record: dict) -> list[object]:
            """Extract normalized datapoints from one page record."""
            result = extract_annual_report_datapoints_openai(
                pages=[page_record],
                company=company,
                year=year,
                category=category,
            )
            return normalize_llamaextract_result(
                result, source=source, company=company, year=year, extractor="openai"
            )

        category_results: list[object] = []
        with ThreadPoolExecutor(max_workers=3) as page_executor:
            page_futures = [page_executor.submit(_extract_page, p) for p in candidate_pages]
            for f in as_completed(page_futures):
                category_results.extend(f.result())
        logger.info("category %s: %d structured datapoints extracted", category, len(category_results))
        if validate and category_results:
            try:
                validation_items = validate_datapoints_openai(
                    category=category,
                    datapoints=category_results,
                    company=company,
                    year=year,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("category %s validation failed (%s), keeping extracted datapoints", category, exc)
                return category_results
            if not validation_items:
                logger.warning("category %s validation returned no items, keeping extracted datapoints", category)
                return category_results
            keep_indices = {
                item.index
                for item in validation_items
                if item.is_valid and item.duplicate_of_index is None
            }
            validated = [
                dp.model_copy(update={"validation_status": "valid"})
                if hasattr(dp, "model_copy")
                else dp
                for i, dp in enumerate(category_results)
                if i in keep_indices
            ]
            logger.info(
                "category %s validation: kept %d/%d datapoints",
                category,
                len(validated),
                len(category_results),
            )
            return validated
        return category_results

    extracted = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(_extract_category, cat): cat for cat in DATAPOINT_CATEGORIES}
        for future in as_completed(futures):
            extracted.extend(future.result())
    deduped = deduplicate_datapoints(extracted)
    counts = Counter(getattr(dp, "datapoint_type", None) for dp in deduped)
    logger.info("structured datapoints after deduplication: %s", dict(sorted(counts.items())))
    return deduped
