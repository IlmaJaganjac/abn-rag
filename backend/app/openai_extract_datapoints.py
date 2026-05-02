from __future__ import annotations

import os
from typing import Any

from backend.app._openai import openai_client
from backend.app.llama_extract_datapoints import AnnualReportDatapoints, category_prompt


def _parse_page_range(page_range: str | None) -> set[int]:
    if not page_range:
        return set()
    out: set[int] = set()
    for part in page_range.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            out.update(range(start, end + 1))
        else:
            out.add(int(part))
    return out


def _format_pages_for_prompt(pages: list[dict[str, Any]], selected_pages: set[int]) -> str:
    rendered: list[str] = []
    for record in pages:
        page = int(record.get("page", 0))
        if selected_pages and page not in selected_pages:
            continue
        text = str(record.get("enhanced_text") or record.get("text", "")).strip()
        if not text:
            continue
        rendered.append(f"=== Page {page} ===\n{text}")
    return "\n\n".join(rendered)


def extract_annual_report_datapoints_openai(
    *,
    pages: list[dict[str, Any]],
    company: str | None,
    year: int | None,
    category: str | None,
    page_range: str | None = None,
) -> AnnualReportDatapoints:
    selected_pages = _parse_page_range(page_range)
    context = _format_pages_for_prompt(pages, selected_pages)
    if not context:
        return AnnualReportDatapoints(company=company, year=year)

    client = openai_client()
    model = os.environ.get("PRE_EXTRACT_MODEL", "gpt-4o-mini")
    supports_temp_zero = "gpt-5-mini" not in model
    prompt = category_prompt(category)
    user_msg = (
        f"Company: {company or ''}\n"
        f"Year: {year or ''}\n"
        f"Selected pages: {page_range or 'all'}\n\n"
        f"Extract structured datapoints from these annual-report pages.\n\n{context}"
    )
    completion = client.beta.chat.completions.parse(
        model=model,
        **({"temperature": 0} if supports_temp_zero else {}),
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_msg},
        ],
        response_format=AnnualReportDatapoints,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return AnnualReportDatapoints(company=company, year=year)
    return AnnualReportDatapoints(
        company=company or parsed.company,
        year=year or parsed.year,
        fte_datapoints=parsed.fte_datapoints,
        sustainability_goals=parsed.sustainability_goals,
        esg_datapoints=parsed.esg_datapoints,
        financial_highlights=parsed.financial_highlights,
        business_performance=parsed.business_performance,
        shareholder_returns=parsed.shareholder_returns,
    )
