from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from backend.app.ingest.chunking import build_semantic_chunks
from backend.app.ingest.tokens import EMBEDDING_MAX_TOKENS, count_tokens, split_oversize
from backend.app.schemas import Chunk


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
        token_counter=count_tokens,
        split_oversize=split_oversize,
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
                token_count=count_tokens(text),
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
