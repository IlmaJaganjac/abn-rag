from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from backend.app._openai import openai_client

log = logging.getLogger(__name__)

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
    table_id: str
    caption: str | None = None
    markdown: str
    page: int
    source: str | None = None


class PageTableExtraction(BaseModel):
    page: int
    has_tables: bool
    tables: list[ExtractedTable] = []
    notes: str | None = None


def _make_table_id(source: str | None, page: int, n: int) -> str:
    stem = Path(source).stem if source else "doc"
    return f"{stem}_p{page}_t{n}"


def _assign_table_ids(extraction: PageTableExtraction, source: str | None) -> None:
    for n, table in enumerate(extraction.tables, 1):
        if not table.table_id:
            table.table_id = _make_table_id(source, extraction.page, n)
        table.page = extraction.page
        if source:
            table.source = source


def tables_to_text(tables: list[dict[str, Any]]) -> str:
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
