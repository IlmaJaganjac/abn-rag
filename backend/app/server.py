from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from backend.app.config import settings
from backend.app.ingestion import ingest_pdf
from backend.app.pipeline import answer_with_context
from backend.app.schemas import VerbatimAnswer

logger = logging.getLogger(__name__)

REPORTS_DIR = settings.reports_dir
PROCESSED_DIR = settings.processed_dir

DocStatus = Literal["queued", "parsing", "embedding", "ready", "error"]
ParserKind = Literal["docling", "llamaparse", "pymupdf"]
DatapointType = Literal["fte", "esg", "financial", "other"]

app = FastAPI(title="Verbatim RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_INGEST_POOL = ThreadPoolExecutor(max_workers=2)
_INGEST_STATE: dict[str, dict[str, Any]] = {}


# --------------------------- response models ---------------------------

class DocumentOut(BaseModel):
    id: str
    source: str
    company: str | None
    year: int | None
    title: str
    pages: int
    size_mb: float
    ingested_at: str
    chunks: int
    parser: ParserKind
    status: DocStatus
    fte: str | None
    net_zero_year: str | None


class DatapointOut(BaseModel):
    company: str
    source: str
    page: int
    type: DatapointType
    metric: str
    value: str
    period: str | int
    section: str
    verbatim: str


class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None
    top_k: int = Field(default=8, ge=1, le=32)


# --------------------------- helpers ---------------------------

def _pages_path(stem: str) -> Path:
    return PROCESSED_DIR / "pages" / f"{stem}.jsonl"


def _chunks_path(stem: str) -> Path:
    return PROCESSED_DIR / "chunks" / f"{stem}.jsonl"


def _datapoints_path(stem: str) -> Path:
    return PROCESSED_DIR / "datapoints" / f"{stem}.json"


def _file_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for _ in f)


def _read_first_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
    return None


def _read_all_datapoints(stem: str) -> list[dict[str, Any]]:
    p = _datapoints_path(stem)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _format_title(source: str, company: str | None, year: int | None) -> str:
    if company and year:
        return f"{company} Annual Report {year}"
    if company:
        return f"{company} Annual Report"
    return Path(source).stem.replace("-", " ").replace("_", " ").title()


def _ingested_at(stem: str) -> str:
    p = _pages_path(stem)
    if p.exists():
        ts = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    else:
        ts = datetime.now(tz=timezone.utc)
    return ts.isoformat().replace("+00:00", "Z")


_FTE_VALUE_RE = re.compile(
    r"(\b(?:approximately|around|over|more than|>)?\s*~?\s*\d{1,3}(?:[,\.]\d{3})+\b|\b>\s*\d{1,3}(?:,\d{3})+\b|\b~?\d{2,3}[,\.]?\d{3}\b)"
)
_NET_ZERO_RE = re.compile(r"net[-\s]?zero[^.\n]*?(20\d{2})", re.IGNORECASE)
_GHG_NEUTRAL_RE = re.compile(
    r"(?:greenhouse gas|GHG|ghg|climate)[-\s]neutral[^.\n]*?(20\d{2})",
    re.IGNORECASE,
)


def _best_fte(datapoints: list[dict[str, Any]]) -> str | None:
    fte_pts = [d for d in datapoints if d.get("datapoint_type") == "fte_candidate"]
    fte_pts.sort(key=lambda d: int(d.get("page") or 9999))
    for dp in fte_pts:
        text = str(dp.get("verbatim_text") or "")
        # prefer values associated with "Total"/"end of period"/"December 31"
        if not re.search(r"total|end of period|december 31|year-end", text, re.IGNORECASE):
            continue
        m = _FTE_VALUE_RE.search(text)
        if m:
            return m.group(0).strip()
    for dp in fte_pts:
        text = str(dp.get("verbatim_text") or "")
        m = _FTE_VALUE_RE.search(text)
        if m:
            return m.group(0).strip()
    return None


def _best_net_zero(datapoints: list[dict[str, Any]]) -> str | None:
    sg = [d for d in datapoints if d.get("datapoint_type") == "sustainability_goal_candidate"]
    for dp in sg:
        text = str(dp.get("verbatim_text") or "")
        m = _NET_ZERO_RE.search(text) or _GHG_NEUTRAL_RE.search(text)
        if m:
            return m.group(1)
    return None


def _classify_datapoint(text: str, dtype: str) -> DatapointType:
    if dtype == "fte_candidate":
        return "fte"
    lower = text.lower()
    if any(k in lower for k in ("scope ", "co2", "co₂", "ghg", "net-zero", "net zero", "emissions", "climate", "energy")):
        return "esg"
    if any(k in lower for k in ("revenue", "profit", "income", "ebit", "eps", "dividend", "capital expenditure", "cet1", "€", "$", "million")):
        return "financial"
    return "other"


def _short_metric(text: str) -> str:
    # First try "Metric: X" style block
    m = re.search(r"Metric:\s*([^\n]+)", text)
    if m:
        return m.group(1).strip()
    first = text.strip().split("\n", 1)[0].strip()
    return (first[:140] + "…") if len(first) > 140 else first


def _short_value(text: str) -> str:
    m = re.search(r"Value:\s*([^\n]+)", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"\b(?:by|target)\s+(20\d{2})\b", text, re.IGNORECASE)
    if m:
        return f"by {m.group(1)}"
    m = _FTE_VALUE_RE.search(text)
    if m:
        return m.group(0).strip()
    return "—"


def _short_period(text: str, year: int | None) -> str | int:
    m = re.search(r"Period:\s*([^\n]+)", text)
    if m:
        v = m.group(1).strip()
        if v.isdigit():
            return int(v)
        return v
    if year is not None:
        return year
    m = re.search(r"\b(20\d{2})\b", text)
    return int(m.group(1)) if m else "—"


def _short_verbatim(text: str) -> str:
    text = text.strip()
    return (text[:300] + "…") if len(text) > 300 else text


def _list_known_sources() -> list[dict[str, Any]]:
    """One row per ingested doc, derived from processed/pages/*.jsonl."""
    pages_root = PROCESSED_DIR / "pages"
    if not pages_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(pages_root.glob("*.jsonl")):
        first = _read_first_jsonl(path)
        if not first:
            continue
        stem = path.stem
        source = str(first.get("source") or f"{stem}.pdf")
        company = first.get("company")
        year = first.get("year")
        parser = str(first.get("parser") or "llamaparse")

        first_chunk = _read_first_jsonl(_chunks_path(stem)) or {}
        chunk_parser = str(first_chunk.get("parser") or parser)

        pdf_path = REPORTS_DIR / source
        size_mb = round(pdf_path.stat().st_size / (1024 * 1024), 1) if pdf_path.exists() else 0.0

        datapoints = _read_all_datapoints(stem)

        rows.append(
            {
                "id": stem,
                "source": source,
                "company": company,
                "year": year,
                "title": _format_title(source, company, year),
                "pages": _file_lines(path),
                "size_mb": size_mb,
                "ingested_at": _ingested_at(stem),
                "chunks": _file_lines(_chunks_path(stem)),
                "parser": (chunk_parser if chunk_parser in ("docling", "llamaparse", "pymupdf") else "llamaparse"),
                "status": "ready",
                "fte": _best_fte(datapoints),
                "net_zero_year": _best_net_zero(datapoints),
            }
        )
    return rows


# --------------------------- endpoints ---------------------------

@app.get("/api/documents", response_model=list[DocumentOut])
def list_documents() -> list[DocumentOut]:
    rows = [DocumentOut(**r) for r in _list_known_sources()]
    # merge in-flight uploads (queued/parsing/embedding/error)
    for state in _INGEST_STATE.values():
        if state["status"] == "ready":
            continue
        rows.append(DocumentOut(**state["doc"]))
    return rows


@app.post("/api/documents", response_model=DocumentOut)
async def upload_document(
    file: UploadFile = File(...),
    company: str | None = Form(default=None),
    year: int | None = Form(default=None),
) -> DocumentOut:
    if not file.filename:
        raise HTTPException(400, "filename required")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target = REPORTS_DIR / file.filename
    contents = await file.read()
    target.write_bytes(contents)

    job_id = str(uuid.uuid4())
    stem = Path(file.filename).stem
    doc = {
        "id": stem,
        "source": file.filename,
        "company": company,
        "year": year,
        "title": _format_title(file.filename, company, year),
        "pages": 0,
        "size_mb": round(len(contents) / (1024 * 1024), 1),
        "ingested_at": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
        "chunks": 0,
        "parser": settings.pdf_parser if settings.pdf_parser in ("docling", "llamaparse", "pymupdf") else "llamaparse",
        "status": "queued",
        "fte": None,
        "net_zero_year": None,
    }
    _INGEST_STATE[job_id] = {"status": "queued", "doc": doc}

    def _worker() -> None:
        st = _INGEST_STATE[job_id]
        try:
            st["status"] = "parsing"
            st["doc"]["status"] = "parsing"
            ingest_pdf(target, company=company, year=year)
            st["status"] = "ready"
            st["doc"]["status"] = "ready"
        except Exception as exc:  # noqa: BLE001
            logger.exception("ingest failed: %s", exc)
            st["status"] = "error"
            st["doc"]["status"] = "error"

    _INGEST_POOL.submit(_worker)
    return DocumentOut(**doc)


@app.delete("/api/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str) -> Response:
    # Idempotent: best-effort cleanup of processed artefacts and chroma rows.
    stem = doc_id
    candidates = [
        _pages_path(stem),
        _chunks_path(stem),
        _datapoints_path(stem),
    ]
    for p in candidates:
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    # Drop from chroma
    try:
        from backend.app.ingestion import get_collection
        first = _read_first_jsonl(_pages_path(stem))
        source = (first or {}).get("source") or f"{stem}.pdf"
        get_collection().delete(where={"source": source})
    except Exception:
        pass

    # Remove pdf if present
    for ext in (".pdf",):
        candidate = REPORTS_DIR / f"{stem}{ext}"
        if candidate.exists():
            try:
                candidate.unlink()
            except OSError:
                pass

    return Response(status_code=204)


@app.get("/api/datapoints", response_model=list[DatapointOut])
def list_datapoints(
    company: str | None = Query(default=None),
    type: str | None = Query(default=None),
) -> list[DatapointOut]:
    rows: list[DatapointOut] = []
    dp_dir = PROCESSED_DIR / "datapoints"
    if not dp_dir.exists():
        return rows

    for path in sorted(dp_dir.glob("*.json")):
        for dp in _read_all_datapoints(path.stem):
            text = str(dp.get("verbatim_text") or "")
            dtype = _classify_datapoint(text, str(dp.get("datapoint_type") or ""))
            row_company = dp.get("company") or ""
            if company and row_company and company.lower() != str(row_company).lower():
                continue
            if type and type != dtype:
                continue
            rows.append(
                DatapointOut(
                    company=str(row_company or "—"),
                    source=str(dp.get("source") or f"{path.stem}.pdf"),
                    page=int(dp.get("page") or 1),
                    type=dtype,
                    metric=_short_metric(text),
                    value=_short_value(text),
                    period=_short_period(text, dp.get("year")),
                    section=str(dp.get("section_path") or "—"),
                    verbatim=_short_verbatim(text),
                )
            )
    return rows


@app.get("/api/documents/{source}/pdf")
def get_pdf(source: str) -> FileResponse:
    safe = Path(source).name
    path = REPORTS_DIR / safe
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="application/pdf", filename=safe)


# --------------------------- chat (SSE) ---------------------------

def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _chat_stream(req: ChatRequest, request: Request) -> AsyncIterator[str]:
    yield _sse("phase", {"phase": "queued", "detail": ""})

    yield _sse("phase", {"phase": "embedding", "detail": settings.openai_embedding_model})

    yield _sse(
        "phase",
        {
            "phase": "searching",
            "detail": f"top_k={req.top_k}"
            + (f" · {req.company}" if req.company else "")
            + (f" · {req.year}" if req.year else ""),
        },
    )

    loop = asyncio.get_running_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: answer_with_context(
                req.question,
                top_k=req.top_k,
                company=req.company,
                year=req.year,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("answer failed")
        refused = VerbatimAnswer(
            question=req.question,
            answer="",
            verbatim=None,
            citations=[],
            refused=True,
            refusal_reason=f"backend error: {exc}",
        )
        yield _sse("answer", json.loads(refused.model_dump_json()))
        return

    if await request.is_disconnected():
        return

    chunks = result.retrieved_chunks
    if chunks:
        first = chunks[0]
        yield _sse("phase", {"phase": "reading", "detail": f"{first.source} p.{first.page}"})

    yield _sse("phase", {"phase": "drafting", "detail": settings.openai_answer_model})
    yield _sse("phase", {"phase": "citing", "detail": "verifying verbatim quotes"})
    yield _sse("phase", {"phase": "done", "detail": ""})

    yield _sse("answer", json.loads(result.answer.model_dump_json()))


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    return StreamingResponse(
        _chat_stream(req, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"ok": True, "ts": time.time()})


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run("backend.app.server:app", host="0.0.0.0", port=8000, reload=True)
