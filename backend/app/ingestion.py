from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from collections import Counter
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chromadb
import tiktoken

from backend.app._openai import openai_client
from backend.app.chunking import build_semantic_chunks
from backend.app.config import settings
from backend.app.openai_table_vision import PageTableExtraction, enhance_page_tables, tables_to_text
from backend.app.extracted_datapoints import deduplicate_datapoints, normalize_llamaextract_result
from backend.app.openai_extract_datapoints import extract_annual_report_datapoints_openai
from backend.app.openai_validate_datapoints import validate_datapoints_openai
from backend.app.parsers import as_page_tuples, parse_pdf_pages
from backend.app.schemas import Chunk
from backend.app.vision_page_selection import classify_table_complexity

logger = logging.getLogger(__name__)

EMBEDDING_MAX_TOKENS = 8191
CHROMA_UPSERT_BATCH_SIZE = 1000
_ENCODING = tiktoken.get_encoding("cl100k_base")
_FTE_PATTERNS = [
    re.compile(
        r"\bFTEs?\b|\bfull[-\s]time equivalents?\b|"
        r"total\s+employees?|number\s+of\s+employees|headcount|workforce|"
        r"payroll\s+employees?|internal\s+employees?|external\s+employees?|"
        r"permanent\s+employees?|temporary\s+employees?|part[-\s]time\s+employees?|"
        r"employee\s+turnover|attrition\s+rate",
        re.IGNORECASE,
    ),
]
_SUSTAINABILITY_PATTERNS = [
    re.compile(
        r"\b(target|goal|aim|ambition|commit(?:ment|ted)?|plan|intend|achieve|reduce|reduction)\b|"
        r"net[-\s]?zero|science\s+based\s+targets?|\bsbti\b|transition\s+plan",
        re.IGNORECASE,
    ),
    re.compile(
        r"greenhouse\s+gas|\bghg\b|co2e?|co₂e?|\bscope\s+[123]\b|emissions?|"
        r"climate|carbon|methane|flaring|renewable|energy|waste|recycl|circular|"
        r"water|biodiversity|supplier|diversity|inclusion|safety|human\s+rights",
        re.IGNORECASE,
    ),
    re.compile(r"\b20[3-9]\d\b", re.IGNORECASE),
]
_ESG_PATTERNS = [
    re.compile(
        r"\b(ESG|environmental|social|governance)\b|"
        r"greenhouse\s+gas|\bghg\b|co2e?|co₂e?|\bscope\s+[123]\b|emissions?|"
        r"renewable|energy|water|waste|recycl|circular|biodiversity|supplier|"
        r"diversity|inclusion|safety|ethics",
        re.IGNORECASE,
    ),
]
_FINANCIAL_HIGHLIGHT_PATTERNS = [
    re.compile(
        r"financial\s+highlights?|financial\s+performance|at\s+a\s+glance|"
        r"\brevenue\b|net\s+sales|total\s+income|net\s+income|net\s+profit|"
        r"operating\s+(?:income|profit)|\bebit(?:da)?\b|gross\s+margin|gross\s+profit|"
        r"earnings\s+per\s+share|\beps\b|free\s+cash\s+flow|operating\s+cash\s+flow|"
        r"cash\s+flow\s+from\s+operat|r&d|research\s+and\s+development|"
        r"return\s+on\s+equity|\broe\b|\bcet1\b|capital\s+ratio|"
        r"liquidity\s+coverage|net\s+interest\s+margin|\bnim\b",
        re.IGNORECASE,
    ),
]
_BUSINESS_PERFORMANCE_PATTERNS = [
    re.compile(
        r"business\s+performance|operational\s+highlights?|segment\s+performance|"
        r"systems?\s+sold|lithography\s+systems?|installed\s+base|order\s+intake|"
        r"order\s+book|backlog|bookings|customers?|clients?|suppliers?|"
        r"customer\s+satisfaction|market\s+share|production\s+volume|deliveries|"
        r"\bloans?\b|\bdeposits?\b|\bmortgages?\b|\blng\b|barrels?\s+per\s+day|"
        r"refining\s+throughput|assets\s+under\s+management|\baum\b",
        re.IGNORECASE,
    ),
]
_SHAREHOLDER_RETURN_PATTERNS = [
    re.compile(
        r"returned?\s+to\s+shareholders?|shareholder\s+(?:returns?|distributions?)|"
        r"capital\s+return|dividends?|dividend\s+per\s+share|payout\s+ratio|"
        r"share\s+buybacks?|share\s+repurchases?|repurchased|treasury\s+shares?|"
        r"shares?\s+cancelled",
        re.IGNORECASE,
    ),
]
_CATEGORY_MAX_PAGES: dict[str, int] = {
    "sustainability": 30,
    "fte": 20,
    "esg": 20,
    "financial_highlight": 20,
    "business_performance": 20,
    "shareholder_return": 20,
}
_CATEGORY_PATTERNS: dict[str, list[re.Pattern]] = {
    "fte": _FTE_PATTERNS,
    "sustainability": _SUSTAINABILITY_PATTERNS,
    "esg": _ESG_PATTERNS,
    "financial_highlight": _FINANCIAL_HIGHLIGHT_PATTERNS,
    "business_performance": _BUSINESS_PERFORMANCE_PATTERNS,
    "shareholder_return": _SHAREHOLDER_RETURN_PATTERNS,
}
_DATAPOINT_CATEGORIES = (
    "fte",
    "sustainability",
    "esg",
    "financial_highlight",
    "business_performance",
    "shareholder_return",
)
_VISION_MAX_PAGES = 30
_VISION_MAX_ATTEMPTS = 2
_VISION_DETAIL = "high"
_VISION_DPI = 180
_VISION_WORKERS = 6


def _category_page_score(category: str, text: str) -> int:
    return sum(len(pattern.findall(text)) for pattern in _CATEGORY_PATTERNS.get(category, []))


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


def _processed_pages_enhanced_path(source: str, processed_dir: Path | None = None) -> Path:
    root = processed_dir or settings.get_processed_path()
    return root / "pages_enhanced" / f"{Path(source).stem}.jsonl"


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


def persist_enhanced_pages(
    records: list[dict[str, Any]],
    *,
    source: str,
    processed_dir: Path | None = None,
) -> Path:
    out_path = _processed_pages_enhanced_path(source, processed_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
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
    datapoints: list[object],
    *,
    source: str,
    processed_dir: Path | None = None,
) -> Path:
    out_path = _processed_datapoints_path(source, processed_dir)
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


def extract_categorized_datapoints(
    pages: list[tuple[int, str]],
    *,
    source: str,
    company: str | None,
    year: int | None,
    validate: bool = False,
) -> list[object]:
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
        pages = [str(record["page"]) for record in records]
        label = ", ".join(pages[:12])
        if len(pages) > 12:
            label += f", ... (+{len(pages) - 12})"
        return label

    def _extract_category(category: str) -> list[object]:
        max_pages = _CATEGORY_MAX_PAGES.get(category, 20)
        if category in _CATEGORY_PATTERNS:
            scored = [
                (p, _category_page_score(category, p["text"]))
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
            result = extract_annual_report_datapoints_openai(
                pages=[page_record],
                company=company,
                year=year,
                category=category,
            )
            return normalize_llamaextract_result(result, source=source, company=company, year=year, extractor="openai")

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
        futures = {executor.submit(_extract_category, cat): cat for cat in _DATAPOINT_CATEGORIES}
        for future in as_completed(futures):
            extracted.extend(future.result())
    deduped = deduplicate_datapoints(extracted)
    counts = Counter(getattr(dp, "datapoint_type", None) for dp in deduped)
    logger.info("structured datapoints after deduplication: %s", dict(sorted(counts.items())))
    return deduped


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


def _strip_markdown_tables(text: str) -> str:
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


def _apply_vision_enhancement(record: dict[str, Any], extraction: PageTableExtraction, model: str) -> dict[str, Any]:
    tables_dicts = [table.model_dump() for table in extraction.tables]
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


def _apply_empty_enhancement(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    out["tables"] = []
    out["table_enhanced"] = False
    out["table_enhancement_model"] = None
    out["table_enhancement_error"] = None
    return out


def _apply_enhancement_error(record: dict[str, Any], error: str) -> dict[str, Any]:
    out = _apply_empty_enhancement(record)
    out["table_enhancement_error"] = error
    return out


def _enhance_page_record_with_retry(
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
    page_num = int(record["page"])
    last_exc: Exception | None = None
    token_limit_phrases = ("length limit was reached", "max_tokens", "finish_reason")
    for attempt in range(max_attempts):
        try:
            image_bytes = _render_page(pdf_path, page_num, dpi)
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
            "token_count": _count_tokens(text),
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
                _enhance_page_record_with_retry,
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
                results[page_num] = _apply_vision_enhancement(page_map[page_num], extraction, vision_model)
            except Exception as exc:
                results[page_num] = _apply_enhancement_error(page_map[page_num], str(exc))

    enhanced_records: list[dict[str, Any]] = []
    enhanced_pages: list[tuple[int, str]] = []
    for record in records:
        page_num = int(record["page"])
        if page_num in selected_set:
            row = results.get(page_num) or _apply_enhancement_error(record, "no result returned")
        else:
            row = _apply_empty_enhancement(record)
        enhanced_records.append(row)
        enhanced_pages.append((page_num, str(row.get("enhanced_text") or row.get("text") or "")))

    persist_enhanced_pages(
        enhanced_records,
        source=source,
        processed_dir=processed_dir,
    )
    return enhanced_pages


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
                "fact_kind": chunk.fact_kind,
                "basis": chunk.basis,
                "scope_type": chunk.scope_type,
                "quality": chunk.quality,
                "validation_status": chunk.validation_status,
                "canonical_metric": chunk.canonical_metric,
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


def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[tuple[str, str]] = set()
    deduped: list[Chunk] = []
    for chunk in chunks:
        key = (chunk.source, chunk.embedding_text or chunk.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def _chunks_for_embedding(chunks: list[Chunk]) -> list[Chunk]:
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
    if chunk.parser is not None:
        md["parser"] = chunk.parser
    if chunk.chunk_kind is not None:
        md["chunk_kind"] = chunk.chunk_kind
    if chunk.section_path is not None:
        md["section_path"] = chunk.section_path
    for field in (
        "fact_kind",
        "basis",
        "scope_type",
        "quality",
        "validation_status",
        "canonical_metric",
    ):
        value = getattr(chunk, field, None)
        if value is not None:
            md[field] = value
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


def _load_datapoints_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("datapoints", [])
    if not isinstance(payload, list):
        raise RuntimeError(f"datapoints JSON must be a list or contain a 'datapoints' list: {path}")
    return [item for item in payload if isinstance(item, dict)]


def _format_datapoint_chunk_text(datapoint: dict[str, Any]) -> str:
    lines: list[str] = []
    metric = str(datapoint.get("metric") or datapoint.get("label") or "").strip()
    if metric:
        lines.append(f"Metric: {metric}")
    datapoint_type = str(datapoint.get("datapoint_type") or "").strip()
    if datapoint_type:
        lines.append(f"Type: {datapoint_type}")
    value = str(datapoint.get("value") or datapoint.get("value_or_target") or "").strip()
    unit = str(datapoint.get("unit") or "").strip()
    if value and unit:
        lines.append(f"Value: {value} {unit}")
    elif value:
        lines.append(f"Value: {value}")
    for label, key in (
        ("Period", "period"),
        ("Basis", "basis"),
        ("Scope", "scope"),
        ("Scope type", "scope_type"),
        ("Target year", "target_year"),
        ("Fact kind", "fact_kind"),
        ("Canonical metric", "canonical_metric"),
    ):
        value = str(datapoint.get(key) or "").strip()
        if value:
            lines.append(f"{label}: {value}")
    quote = str(datapoint.get("verbatim_text") or datapoint.get("quote") or "").strip()
    if quote:
        lines.append(f"Quote: {quote}")
    return "\n".join(lines).strip()


def build_datapoint_chunks(
    datapoints: list[dict[str, Any]],
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
        page = int(datapoint.get("page") or 1)
        idx = next_idx_by_page.get(page, 0)
        next_idx_by_page[page] = idx + 1
        datapoint_type = str(datapoint.get("datapoint_type") or "datapoint")
        text = _format_datapoint_chunk_text(datapoint)
        if not text:
            continue
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
                chunk_kind="extracted_datapoint",
                section_path=str(section_path) if section_path else datapoint_type,
                embedding_text="\n".join(part for part in embedding_parts if part),
                fact_kind=datapoint.get("fact_kind"),
                basis=datapoint.get("basis"),
                scope_type=datapoint.get("scope_type"),
                quality=datapoint.get("quality"),
                validation_status=datapoint.get("validation_status"),
                canonical_metric=datapoint.get("canonical_metric"),
            )
        )
    return chunks


def _apply_enhanced_text(
    pages: list[tuple[int, str]],
    enhanced_jsonl: Path,
) -> list[tuple[int, str]]:
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


def _load_pages_jsonl(path: Path) -> tuple[list[tuple[int, str]], str]:
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
        pages, parsed_parser = _load_pages_jsonl(pages_jsonl)
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
        pages = _apply_enhanced_text(pages, enhanced_jsonl)

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
        raw_datapoints = _load_datapoints_json(datapoints_json)
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

    embed_chunks = _chunks_for_embedding(chunks)
    embeddings = embed_texts([c.embedding_text or c.text for c in embed_chunks])
    if len(embeddings) != len(embed_chunks):
        raise RuntimeError(
            f"embedding count {len(embeddings)} != embeddable chunk count {len(embed_chunks)}"
        )

    collection = get_collection(reset=reset)
    if not reset:
        delete_existing_source_chunks(
            collection,
            source=source,
            company=company,
            year=year,
        )
    for i in range(0, len(embed_chunks), CHROMA_UPSERT_BATCH_SIZE):
        batch_chunks = embed_chunks[i : i + CHROMA_UPSERT_BATCH_SIZE]
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
        len(embed_chunks),
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
    parser.add_argument(
        "--enhanced-jsonl",
        type=Path,
        default=None,
        help="path to enhanced pages JSONL; pages with enhanced_text override parser output",
    )
    parser.add_argument(
        "--pages-jsonl",
        type=Path,
        default=None,
        help="path to parsed pages JSONL to ingest directly without running the PDF parser",
    )
    parser.add_argument(
        "--no-extract-datapoints",
        action="store_false",
        dest="extract_datapoints",
        help="skip categorized LLM datapoint extraction",
    )
    parser.add_argument(
        "--no-vision-enhancement",
        action="store_false",
        dest="enhance_vision",
        help="skip automatic vision enhancement of difficult table pages after parsing",
    )
    parser.add_argument(
        "--datapoints-json",
        type=Path,
        default=None,
        help="path to existing datapoints JSON to embed without re-extracting",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="skip embedding and ChromaDB upsert (parse, vision and datapoint extraction only)",
    )
    parser.add_argument(
        "--no-validate-datapoints",
        action="store_false",
        dest="validate_datapoints",
        help="skip OpenAI validation/deduplication of extracted datapoints",
    )
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
