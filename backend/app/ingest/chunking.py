from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from backend.app.ingest.chunking_heuristics import (
    HEADING_RE,
    TABLE_SEPARATOR_RE,
    YEAR_RE,
    clean_block,
    clean_heading,
    find_boilerplate_lines,
    is_noise_line,
    is_table_line,
    looks_like_heading,
    looks_like_value,
    normalize_line,
    parse_table_cells,
    remove_boilerplate,
    starts_with_value,
    year_period,
)
from backend.app.schemas import Chunk


@dataclass(frozen=True)
class ChunkDraft:
    """Intermediate chunk representation before ids and final metadata are assigned."""
    page: int
    text: str
    chunk_kind: str
    section_path: str | None
    embedding_text: str


def build_semantic_chunks(
    pages: Iterable[tuple[int, str]],
    *,
    source: str,
    company: str | None,
    year: int | None,
    parser: str | None,
    max_tokens: int,
    overlap: int,
    token_counter: Callable[[str], int],
    split_oversize: Callable[[str, int, int], list[str]],
) -> list[Chunk]:
    """Convert parsed pages into final `Chunk` objects ready for persistence and retrieval."""
    page_list = list(pages)
    boilerplate = find_boilerplate_lines(text for _, text in page_list)
    drafts: list[ChunkDraft] = []
    heading_stack: list[tuple[int, str]] = []
    for page, text in page_list:
        drafts.extend(
            _page_drafts(
                page=page,
                text=text,
                heading_stack=heading_stack,
                company=company,
                year=year,
                boilerplate=boilerplate,
                max_tokens=max_tokens,
                overlap=overlap,
                token_counter=token_counter,
                split_oversize=split_oversize,
            )
        )

    chunks: list[Chunk] = []
    idx_by_page: defaultdict[int, int] = defaultdict(int)
    for draft in drafts:
        idx = idx_by_page[draft.page]
        idx_by_page[draft.page] += 1
        chunks.append(
            Chunk(
                id=f"{source}:{draft.page}:{idx}",
                source=source,
                company=company,
                year=year,
                page=draft.page,
                text=draft.text,
                token_count=token_counter(draft.text),
                parser=parser,
                chunk_kind=draft.chunk_kind,
                section_path=draft.section_path,
                embedding_text=draft.embedding_text,
            )
        )
    return chunks


def _page_drafts(
    *,
    page: int,
    text: str,
    heading_stack: list[tuple[int, str]],
    company: str | None,
    year: int | None,
    boilerplate: set[str],
    max_tokens: int,
    overlap: int,
    token_counter: Callable[[str], int],
    split_oversize: Callable[[str, int, int], list[str]],
) -> list[ChunkDraft]:
    """Build draft chunks for one page by separating headings, narrative, and tables."""
    drafts: list[ChunkDraft] = []
    narrative: list[str] = []
    table: list[str] = []

    def section_path() -> str | None:
        """Return the active heading path for the current page position."""
        headings = [
            clean
            for _, clean in heading_stack
            if normalize_line(clean) not in boilerplate
        ]
        return " > ".join(headings) if headings else None

    def flush_narrative() -> None:
        """Turn buffered narrative lines into one or more section chunk drafts."""
        nonlocal narrative
        body = remove_boilerplate("\n".join(narrative), boilerplate)
        narrative = []
        if not body:
            return
        for part in split_oversize(body, max_tokens, overlap):
            drafts.append(
                _draft(
                    page=page,
                    text=part,
                    chunk_kind="section",
                    section_path=section_path(),
                    company=company,
                    year=year,

                    boilerplate=boilerplate,
                )
            )

    def flush_table() -> None:
        """Turn buffered table rows into table chunk drafts and clear the buffer."""
        nonlocal table
        rows = table
        table = []
        if not rows:
            return
        drafts.extend(
            _table_drafts(
                page=page,
                rows=rows,
                section_path=section_path(),
                company=company,
                year=year,

                boilerplate=boilerplate,
                max_tokens=max_tokens,
                token_counter=token_counter,
            )
        )

    for line in text.splitlines():
        stripped = line.strip()
        heading = HEADING_RE.match(stripped)
        if heading:
            level = len(heading.group(1))
            title = clean_heading(heading.group(2))
            if normalize_line(title) in boilerplate:
                continue
            if not looks_like_heading(title):
                if table:
                    flush_table()
                narrative.append(title)
                continue
            flush_table()
            flush_narrative()
            heading_stack[:] = [(lvl, val) for lvl, val in heading_stack if lvl < level]
            heading_stack.append((level, title))
            continue

        if normalize_line(stripped) in boilerplate or is_noise_line(stripped):
            continue

        if is_table_line(stripped):
            flush_narrative()
            table.append(stripped)
            continue

        if table:
            flush_table()
        narrative.append(line)

    flush_table()
    flush_narrative()

    return drafts


def _table_drafts(
    *,
    page: int,
    rows: list[str],
    section_path: str | None,
    company: str | None,
    year: int | None,
    boilerplate: set[str],
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> list[ChunkDraft]:
    """Build draft chunks for one table, including row- and metric-level variants."""
    drafts: list[ChunkDraft] = []
    table_kind = _classify_table(rows)
    full_table = clean_block("\n".join(rows))
    if full_table:
        table_context = _table_context(section_path, rows, table_kind)
        for part in _split_table_on_rows(
            rows=rows,
            max_tokens=max_tokens,
            token_counter=token_counter,
        ):
            drafts.append(
                _draft(
                    page=page,
                    text=part,
                    chunk_kind="table",
                    section_path=section_path,
                    company=company,
                    year=year,

                    boilerplate=boilerplate,
                    extra_embedding_context=table_context,
                )
            )

    data_rows = [row for row in rows if not TABLE_SEPARATOR_RE.match(row)]
    headers, body_rows = _table_headers_and_body(data_rows)

    if table_kind == "header_table":
        for row in body_rows:
            cells = parse_table_cells(row)
            row_text = _format_header_aware_row(headers, cells)
            if not row_text:
                continue
            drafts.append(
                _draft(
                    page=page,
                    text=row_text,
                    chunk_kind="table_row",
                    section_path=section_path,
                    company=company,
                    year=year,

                    boilerplate=boilerplate,
                    extra_embedding_context=_table_context(section_path, rows, table_kind),
                )
            )
    return drafts


def _split_table_on_rows(
    *,
    rows: list[str],
    max_tokens: int,
    token_counter: Callable[[str], int],
) -> list[str]:
    """Split a large markdown table into smaller row-based table text blocks."""
    full_table = clean_block("\n".join(rows))
    if not full_table or token_counter(full_table) <= max_tokens:
        return [full_table] if full_table else []

    header_rows: list[str] = []
    body_start = 0
    if rows:
        header_rows.append(rows[0])
        body_start = 1
    if len(rows) > 1 and TABLE_SEPARATOR_RE.match(rows[1]):
        header_rows.append(rows[1])
        body_start = 2

    body_rows = rows[body_start:]
    if not body_rows:
        return [full_table]

    parts: list[str] = []
    current: list[str] = []

    def current_text() -> str:
        """Return the current table part with header rows reattached."""
        return "\n".join(header_rows + current)

    for row in body_rows:
        candidate_rows = current + [row]
        candidate = "\n".join(header_rows + candidate_rows)
        if current and token_counter(candidate) > max_tokens:
            parts.append(clean_block(current_text()))
            current = [row]
        else:
            current = candidate_rows

    if current:
        parts.append(clean_block(current_text()))

    return parts or [full_table]


def _draft(
    *,
    page: int,
    text: str,
    chunk_kind: str,
    section_path: str | None,
    company: str | None,
    year: int | None,
    boilerplate: set[str],
    extra_embedding_context: str | None = None,
) -> ChunkDraft:
    """Create one `ChunkDraft` with cleaned body text and embedding context."""
    body = remove_boilerplate(text, boilerplate).strip()
    context = _embedding_context(
        company=company,
        year=year,
        section_path=section_path,
        extra=extra_embedding_context,
    )
    embedding_text = "\n".join(part for part in (context, body) if part).strip()
    return ChunkDraft(
        page=page,
        text=body,
        chunk_kind=chunk_kind,
        section_path=section_path,
        embedding_text=embedding_text,
    )


def _embedding_context(
    *,
    company: str | None,
    year: int | None,
    section_path: str | None,
    extra: str | None = None,
) -> str:
    """Build the contextual prefix that is included in a chunk's embedding text."""
    parts = [
        company,
        str(year) if year is not None else None,
        f"Section: {section_path}" if section_path else None,
        extra,
    ]
    return "\n".join(part for part in parts if part)


def _table_context(section_path: str | None, rows: list[str], table_kind: str) -> str | None:
    """Summarize table metadata for embeddings and return it as extra context text."""
    data_rows = [row for row in rows if not TABLE_SEPARATOR_RE.match(row)]
    headers, _ = _table_headers_and_body(data_rows)
    parts: list[str] = [f"Table type: {table_kind}"]
    if section_path:
        parts.append(f"Table caption: {section_path}")
    if headers:
        columns = [header for header in headers if header.strip()]
        if columns:
            parts.append("Columns: " + " | ".join(columns))
    return "\n".join(parts) if parts else None


def _classify_table(rows: list[str]) -> str:
    """Classify a table into a coarse kind used for downstream chunk shaping."""
    data_rows = [row for row in rows if not TABLE_SEPARATOR_RE.match(row)]
    headers, body_rows = _table_headers_and_body(data_rows)
    if headers is not None and body_rows and _looks_like_header_row(headers):
        return "header_table"
    if data_rows and _mostly_metric_pairs(parse_table_cells(data_rows[0])):
        return "kpi_pairs"
    return "generic_table"


def _looks_like_header_row(cells: list[str]) -> bool:
    """Return whether parsed table cells look like a header row."""
    normalized = {normalize_line(cell) for cell in cells if cell.strip()}
    if any(year_period(cell) is not None for cell in cells):
        return True
    return bool(normalized & {"notes", "note", "description", "metric", "topic"})


def _mostly_metric_pairs(cells: list[str]) -> bool:
    """Return whether a row mostly consists of alternating value-label metric pairs."""
    non_empty = [cell for cell in cells if cell.strip()]
    if len(non_empty) < 2:
        return False
    return len(_metric_pairs(non_empty)) * 2 >= len(non_empty)


def _metric_pairs(cells: list[str]) -> list[tuple[str, str]]:
    """Extract `(value, label)` metric pairs from table cells when they are detectable."""
    pairs: list[tuple[str, str]] = []
    i = 0
    while i + 1 < len(cells):
        value = cells[i].strip()
        label = cells[i + 1].strip()
        if value and label and looks_like_value(value) and not looks_like_value(label):
            pairs.append((value, label))
            i += 2
        elif (
            value
            and label
            and YEAR_RE.search(value)
            and starts_with_value(label)
            and not looks_like_value(label)
        ):
            pairs.append((label, value))
            i += 2
        else:
            i += 1
    return pairs



def _table_headers_and_body(data_rows: list[str]) -> tuple[list[str] | None, list[str]]:
    """Split table rows into optional headers and remaining body rows."""
    if len(data_rows) < 2:
        return None, data_rows
    headers = parse_table_cells(data_rows[0])
    if not headers or not any(header.strip() for header in headers):
        return None, data_rows
    return headers, data_rows[1:]


def _format_header_aware_row(headers: list[str] | None, cells: list[str]) -> str | None:
    """Format one table row as `Header: value` lines and return the rendered text."""
    if headers is None:
        return None
    if len(cells) < len(headers):
        cells = cells + [""] * (len(headers) - len(cells))
    elif len(cells) > len(headers):
        cells = cells[: len(headers)]
    lines = [
        f"{header.strip()}: {cell.strip()}"
        for header, cell in zip(headers, cells, strict=True)
        if header.strip() and cell.strip()
    ]
    return "\n".join(lines) if lines else None
