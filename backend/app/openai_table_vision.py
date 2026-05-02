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
You extract tables from annual report page images and produce structured metrics.

Table extraction rules:
- Preserve table structure exactly as visible on the page.
- Extract only factual values visible on the page. Do not infer or invent values.
- If a cell is blank, output null or empty string.
- Keep row labels and column labels exactly as printed.
- Preserve units, periods, target years, baselines, and footnote markers when visible.
- If the page is not table-heavy or no reliable table is visible, return has_tables=false and no tables.
- If table structure is uncertain, include notes and set lower confidence values.

Metric extraction rules:
- Derive one metric record per (row, column) cell where the value is concrete and unambiguous.
- The metric name must identify the full subject: combine the row group header + sub-row label.
  Example: row group "Halve Scope 1 and 2 emissions" + sub-row "Actual" → metric = "Halve Scope 1 and 2 emissions — Actual"
  Never use a generic column header like "% reduction" or "Target" as the metric name.
- row_label = the full row label as printed (include group header if present).
- column_label = the column header for that cell (e.g. "2025", "2030", "Baseline").
- period = year if the column is a reporting year (e.g. "2025", "2024").
- target_year = year if the column is a target/ambition year (e.g. "2030", "2050").
- Do not create metrics from narrative text outside table cells.
- The evidence field must quote the visible row + column context exactly.
- Never hallucinate values not visible in the image.

Classify each metric into exactly one datapoint_type:
- fte: workforce size, headcount, FTE, employees, payroll
- sustainability_goal: forward-looking targets, ambitions, commitments, reductions by a future year
- esg: actual reported ESG performance values (emissions actuals, energy use, water, safety)
- financial_highlight: revenue, profit, EPS, cash flow, capex, ROE, net income, EBITDA
- business_performance: operational KPIs — systems sold, order intake, production volume, loans, AUM
- shareholder_return: dividends, buybacks, payout ratio, capital return
- other: anything that does not fit the above\
"""

_DATAPOINT_TYPES = frozenset({
    "fte",
    "sustainability_goal",
    "esg",
    "financial_highlight",
    "business_performance",
    "shareholder_return",
    "other",
})


class TableMetric(BaseModel):
    datapoint_type: str | None = None
    metric: str
    value: str
    unit: str | None = None
    period: str | None = None
    target_year: str | None = None
    baseline: str | None = None
    scope: str | None = None
    row_label: str | None = None
    column_label: str | None = None
    evidence: str
    confidence: float | None = None


class ExtractedTable(BaseModel):
    table_id: str
    caption: str | None = None
    markdown: str
    page: int
    source: str | None = None
    metrics: list[TableMetric] = []


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
        lines = [header]
        md = (t.get("markdown") or "").strip()
        if md:
            lines.append(md)
        metrics = t.get("metrics") or []
        if metrics:
            lines.append(f"Metrics ({len(metrics)}):")
            for m in metrics[:15]:
                seg: list[str] = []
                if m.get("metric"):
                    seg.append(m["metric"])
                if m.get("period"):
                    seg.append(m["period"])
                if m.get("value"):
                    seg.append(f"= {m['value']}")
                if m.get("unit"):
                    seg.append(m["unit"])
                if m.get("scope"):
                    seg.append(f"scope={m['scope']}")
                lines.append("  " + " | ".join(s for s in seg if s))
        parts.append("\n".join(lines))
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

    supports_temp_zero = "gpt-5-mini" not in model

    completion = client.beta.chat.completions.parse(
        model=model,
        **({"temperature": 0} if supports_temp_zero else {}),
        max_tokens=32000,
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
