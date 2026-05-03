from __future__ import annotations

import re

_SEP_RE = re.compile(r"^\|[\s\-:|]+\|$")
_ESRS_RE = re.compile(r"ESRS|disclosure requirement", re.IGNORECASE)


def classify_table_complexity(text: str) -> tuple[str, float]:
    """Classify markdown-table complexity and return (kind, score)."""
    lines = text.splitlines()
    data_lines = [
        line.strip()
        for line in lines
        if line.strip().startswith("|") and line.strip().endswith("|")
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
