from __future__ import annotations

import base64
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from backend.app.config import openai_client

log = logging.getLogger(__name__)

_SEP_RE = re.compile(r"^\|[\s\-:|]+\|$")
_ESRS_RE = re.compile(r"ESRS|disclosure requirement", re.IGNORECASE)

_SYSTEM_PROMPT = """\
You convert annual report page images into clean, faithful markdown tables.

Rules:
- Render every table on the page as a GitHub-flavored markdown table.
- Preserve table structure, row labels, column headers, units, periods, target years,
  baselines, and footnote markers exactly as visible on the page.
- Do not infer or invent values. If a cell is blank, leave it blank.
- If a header spans multiple columns, repeat the group label across the cells it covers
  so each column has a self-contained header.
- Include a short caption per table when one is visible on the page.
- If the page contains no reliable table, return has_tables=false and no tables.
- Do not extract narrative prose. Only tables.
"""


class ExtractedTable(BaseModel):
    """One extracted markdown table with source metadata."""
    table_id: str
    caption: str | None = None
    markdown: str
    page: int
    source: str | None = None


class PageTableExtraction(BaseModel):
    """All table-extraction output for a single page image."""
    page: int
    has_tables: bool
    tables: list[ExtractedTable] = []
    notes: str | None = None


def classify_table_complexity(text: str) -> tuple[str, float]:
    """Classify markdown-table complexity and return (kind, score)."""
    lines = text.splitlines()
    data_lines = [
        line.strip()
        for line in lines
        if line.strip().startswith("|")
        and line.strip().endswith("|")
        and not _SEP_RE.match(line.strip())
    ]
    if len(data_lines) < 2:
        return "skip", 0.0

    if _ESRS_RE.search(text[:400]):
        return "skip", 0.0

    pipe_counts = [line.count("|") for line in data_lines]
    max_cols = max(pipe_counts) - 1
    col_variance = max(pipe_counts) - min(pipe_counts)

    rows_cells = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in data_lines
    ]
    total_cells = sum(len(row) for row in rows_cells)
    empty_cells = sum(1 for row in rows_cells for cell in row if not cell)
    empty_ratio = (empty_cells / total_cells) if total_cells else 0.0

    unit_col_only = False
    if col_variance == 0 and rows_cells:
        n_cols = len(rows_cells[0])
        for col_idx in range(n_cols):
            if all(not row[col_idx] for row in rows_cells if col_idx < len(row)):
                unit_col_only = True
                break
    if unit_col_only:
        return "skip", 0.0

    score = col_variance * 10.0 + max_cols * 2.0 + empty_ratio * 20.0

    if max_cols >= 5 and empty_ratio > 0.20:
        return "visual_infographic", score
    if max_cols >= 5 and col_variance >= 1:
        return "wide_irregular", score
    if max_cols >= 4 and empty_ratio > 0.30:
        return "wide_irregular", score

    return "skip", 0.0


def _make_table_id(source: str | None, page: int, n: int) -> str:
    """Build a deterministic table id from source name, page number, and ordinal."""
    stem = Path(source).stem if source else "doc"
    return f"{stem}_p{page}_t{n}"


def _assign_table_ids(extraction: PageTableExtraction, source: str | None) -> None:
    """Fill missing table ids and copy page/source metadata onto extracted tables."""
    for n, table in enumerate(extraction.tables, 1):
        if not table.table_id:
            table.table_id = _make_table_id(source, extraction.page, n)
        table.page = extraction.page
        if source:
            table.source = source


def tables_to_text(tables: list[dict[str, Any]]) -> str:
    """Render extracted tables into one retrieval-friendly text block."""
    parts: list[str] = []
    for t in tables:
        header = f"TABLE {t.get('table_id', '')}"
        if t.get("caption"):
            header += f": {t['caption']}"
        md = (t.get("markdown") or "").strip()
        if md:
            parts.append(f"{header}\n{md}")
        else:
            parts.append(header)
    return "\n\n".join(parts)


def enhance_page_tables(
    *,
    image_bytes: bytes,
    page: int,
    company: str | None,
    year: int | None,
    source: str | None,
    page_text: str | None,
    model: str,
    detail: str = "high",
) -> PageTableExtraction:
    """Extract page tables with vision and return a structured page-level result."""
    client = openai_client(timeout=120)
    b64 = base64.b64encode(image_bytes).decode("ascii")

    meta: list[str] = []
    if company:
        meta.append(f"Company: {company}")
    if year:
        meta.append(f"Year: {year}")
    if source:
        meta.append(f"Source: {source}")
    meta.append(f"Page: {page}")

    text_ctx = ""
    if page_text:
        truncated = page_text[:800].strip()
        text_ctx = f"\n\nLlamaParse text (may be garbled for complex tables):\n{truncated}"

    user_text = "\n".join(meta) + text_ctx + "\n\nExtract all tables from this annual report page."

    is_gpt5 = "gpt-5" in model
    extra: dict[str, Any] = {}
    if is_gpt5:
        extra["max_completion_tokens"] = 16000
    else:
        extra["max_tokens"] = 16000
        extra["temperature"] = 0

    completion = client.beta.chat.completions.parse(
        model=model,
        **extra,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": detail,
                }},
            ]},
        ],
        response_format=PageTableExtraction,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return PageTableExtraction(page=page, has_tables=False)

    parsed.page = page
    _assign_table_ids(parsed, source)
    return parsed
