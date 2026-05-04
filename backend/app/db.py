from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from backend.app.config import settings

DB_COLUMNS = (
    "source", "company", "year", "datapoint_type", "metric", "value", "unit",
    "period", "page", "quote", "basis", "scope", "target_year", "extractor",
    "priority", "confidence", "fact_kind", "scope_type", "quality",
    "validation_status", "canonical_metric",
)

_CREATE = f"""
CREATE TABLE IF NOT EXISTS datapoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {', '.join(f'{c} TEXT' for c in DB_COLUMNS)}
);
CREATE INDEX IF NOT EXISTS idx_dp_company ON datapoints(company);
CREATE INDEX IF NOT EXISTS idx_dp_type ON datapoints(datapoint_type);
CREATE INDEX IF NOT EXISTS idx_dp_source ON datapoints(source);
"""


def _db_path() -> Path:
    path = settings.get_processed_path() / "datapoints.db"
    return path


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create schema and migrate existing JSON datapoints files if DB is empty."""
    with get_conn() as conn:
        conn.executescript(_CREATE)
        row = conn.execute("SELECT COUNT(*) FROM datapoints").fetchone()
        if row[0] > 0:
            return

    dp_dir = settings.get_processed_path() / "datapoints"
    records: list[dict] = []
    for path in sorted(dp_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records.extend(data)
        except Exception:
            pass

    if records:
        upsert_datapoints(records)


def upsert_datapoints(records: list[dict[str, Any]]) -> None:
    """Insert or replace datapoints, keyed on (source, metric, period)."""
    if not records:
        return

    with get_conn() as conn:
        conn.execute("""
            CREATE TEMP TABLE IF NOT EXISTS _dp_stage (
                source TEXT, metric TEXT, period TEXT
            )
        """)
        rows = [
            tuple(str(r.get(c, "") or "") for c in DB_COLUMNS)
            for r in records
        ]
        placeholders = ", ".join("?" * len(DB_COLUMNS))
        conn.executemany(
            f"INSERT OR REPLACE INTO datapoints ({', '.join(DB_COLUMNS)}) VALUES ({placeholders})",
            rows,
        )


def query_datapoints(
    company: str | None = None,
    datapoint_type: str | None = None,
) -> list[dict[str, Any]]:
    clauses: list[str] = []
    params: list[str] = []
    if company:
        clauses.append("company = ?")
        params.append(company)
    if datapoint_type:
        clauses.append("datapoint_type = ?")
        params.append(datapoint_type)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with get_conn() as conn:
        rows = conn.execute(
            f"SELECT * FROM datapoints {where} ORDER BY company, datapoint_type, metric",
            params,
        ).fetchall()
    return [dict(r) for r in rows]
