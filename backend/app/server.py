from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.app.answer import answer_question
from backend.app.config import settings
from backend.app.db import init_db, query_datapoints
from backend.app.ingest.embedding import get_collection
from backend.app.ingestion import ingest_pdf
from backend.app.retrieval import retrieve_decomposed
from backend.app.schemas import RetrievalQuery, VerbatimAnswer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Annualyzer API")


@app.on_event("startup")
def _startup() -> None:
    init_db()
    if settings.enable_rerank:
        from backend.app.retrieval import _get_reranker, rerank
        from backend.app.schemas import RetrievedChunk
        _get_reranker()
        # Warm MPS kernels with one tiny forward pass so user request 1 is fast.
        rerank("warmup", [RetrievedChunk(id="w", source="w", page=1, text="warmup", token_count=1, score=0.0)], 1)
        log.info("reranker warmed")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=2)
_jobs: dict[str, dict[str, Any]] = {}


# ---------- helpers ----------

_REPORT_WORDS = re.compile(
    r"\b(annual|integrated|sustainability|financial|report|jaarverslag|"
    r"verslag|20\d{2})\b",
    re.IGNORECASE,
)


def _clean_company_name(raw: str) -> str:
    """Normalize a raw company string into a cleaner display and matching form."""
    s = re.sub(r"[_\-]+", " ", raw)
    s = _REPORT_WORDS.sub("", s)
    s = re.sub(r"\s+", " ", s).strip(" -_,.")
    return s


def _company_from_filename(filename: str) -> str | None:
    """Infer a company name from the uploaded filename when possible."""
    stem = Path(filename).stem
    cleaned = _clean_company_name(stem)
    if not cleaned or len(cleaned) < 2:
        return None
    return cleaned.title() if cleaned.islower() else cleaned


def _company_from_pdf(pdf_path: Path) -> str | None:
    """Infer a company name from PDF metadata or the first visible title lines."""
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            meta_title = (doc.metadata or {}).get("title") or ""
            meta_author = (doc.metadata or {}).get("author") or ""
            first_text = "\n".join(doc[i].get_text() for i in range(min(2, len(doc))))
        finally:
            doc.close()
    except Exception as exc:
        log.warning("pdf metadata read failed: %s", exc)
        return None

    for candidate in (meta_author, meta_title):
        c = _clean_company_name(candidate)
        if c and len(c) >= 2 and not _REPORT_WORDS.fullmatch(c):
            return c

    for line in first_text.splitlines():
        line = line.strip()
        if 2 <= len(line) <= 60 and not _REPORT_WORDS.search(line):
            words = line.split()
            if 1 <= len(words) <= 6 and any(w[:1].isupper() for w in words):
                return line
    return None


def _detect_company_year(filename: str, pdf_path: Path | None = None) -> tuple[str | None, int | None]:
    """Infer company and report year from filename and optionally PDF contents."""
    year: int | None = None
    m = re.search(r"(20\d{2})", filename)
    if m:
        year = int(m.group(1))

    company = _company_from_filename(filename)

    if pdf_path is not None:
        if company is None:
            company = _company_from_pdf(pdf_path)
        if year is None:
            try:
                import fitz

                doc = fitz.open(str(pdf_path))
                try:
                    first_text = "\n".join(doc[i].get_text() for i in range(min(3, len(doc))))
                finally:
                    doc.close()
                years = re.findall(r"\b(20\d{2})\b", first_text)
                if years:
                    year = max(int(y) for y in years)
            except Exception as exc:
                log.warning("year detect via pdf failed: %s", exc)

    return company, year


def _processed_doc_index() -> dict[str, dict[str, Any]]:
    """Build a document index keyed by source filename from persisted ingestion artifacts."""
    out: dict[str, dict[str, Any]] = {}
    chunks_dir = settings.get_processed_path() / "chunks"
    if chunks_dir.exists():
        for path in sorted(chunks_dir.glob("*.jsonl")):
            company: str | None = None
            year: int | None = None
            source: str | None = None
            pages: set[int] = set()
            chunk_count = 0
            parser: str | None = None
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    chunk_count += 1
                    source = source or rec.get("source")
                    company = company or rec.get("company")
                    if year is None and rec.get("year") is not None:
                        year = int(rec["year"])
                    parser = parser or rec.get("parser")
                    if rec.get("page") is not None:
                        pages.add(int(rec["page"]))
            if source:
                pdf = settings.reports_dir / source
                size_mb = pdf.stat().st_size / (1024 * 1024) if pdf.exists() else 0.0
                out[source] = {
                    "id": source,
                    "source": source,
                    "company": company,
                    "year": year,
                    "title": _title_from_source(source, company, year),
                    "pages": max(pages) if pages else 0,
                    "size_mb": round(size_mb, 1),
                    "ingested_at": _isoformat(path.stat().st_mtime),
                    "chunks": chunk_count,
                    "parser": parser or "llamaparse",
                    "status": "ready",
                    "fte": None,
                    "net_zero_year": None,
                }
    _enrich_with_datapoints(out)
    return out


def _title_from_source(source: str, company: str | None, year: int | None) -> str:
    """Create a user-facing document title from source, company, and year."""
    stem = Path(source).stem.replace("-", " ").replace("_", " ").title()
    if company and year:
        return f"{company} Annual Report {year}"
    if company:
        return f"{company} Annual Report"
    return stem


def _isoformat(epoch: float) -> str:
    """Convert a Unix timestamp to an ISO-8601 UTC string."""
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _enrich_with_datapoints(index: dict[str, dict[str, Any]]) -> None:
    """Augment document records in place with summary datapoints such as FTE or net-zero year."""
    dp_dir = settings.get_processed_path() / "datapoints"
    if not dp_dir.exists():
        return
    for path in dp_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for dp in data:
            src = dp.get("source")
            if not src or src not in index:
                continue
            doc = index[src]
            metric = (dp.get("metric") or "").lower()
            if doc.get("fte") is None and ("fte" in metric or "employee" in metric or "headcount" in metric):
                doc["fte"] = str(dp.get("value") or "")[:32] or None
            if doc.get("net_zero_year") is None and "net" in metric and "zero" in metric:
                ty = dp.get("target_year") or dp.get("period")
                doc["net_zero_year"] = str(ty) if ty else None


# ---------- documents ----------

@app.get("/api/documents")
def list_documents() -> list[dict[str, Any]]:
    """Return indexed documents plus any in-flight ingestion jobs for the frontend."""
    docs = list(_processed_doc_index().values())
    for j in _jobs.values():
        if j["status"] not in ("ready", "error"):
            docs.append(j["doc"])
    docs.sort(key=lambda d: d.get("ingested_at") or "", reverse=True)
    return docs


@app.post("/api/documents")
async def upload_document(
    file: UploadFile = File(...),
    company: str | None = Form(default=None),
    year: int | None = Form(default=None),
) -> dict[str, Any]:
    """Upload a PDF, start background ingestion, and return the pending document record."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only PDF files are supported")

    target = settings.reports_dir / file.filename
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    existing = _processed_doc_index().get(file.filename)
    if existing is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "already_uploaded",
                "message": f"{file.filename} is already indexed.",
                "document": existing,
            },
        )

    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    detected_company, detected_year = _detect_company_year(file.filename, target)
    eff_company = company or detected_company
    eff_year = year or detected_year

    job_id = file.filename
    pdf_size_mb = round(target.stat().st_size / (1024 * 1024), 1)
    initial_doc = {
        "id": job_id,
        "source": file.filename,
        "company": eff_company,
        "year": eff_year,
        "title": _title_from_source(file.filename, eff_company, eff_year),
        "pages": 0,
        "size_mb": pdf_size_mb,
        "ingested_at": _isoformat(target.stat().st_mtime),
        "chunks": 0,
        "parser": "llamaparse",
        "status": "parsing",
        "fte": None,
        "net_zero_year": None,
    }
    _jobs[job_id] = {"status": "parsing", "doc": initial_doc, "error": None}

    def _run() -> None:
        """Ingest the uploaded PDF in the background and update job state."""
        try:
            ingest_pdf(
                target,
                company=eff_company,
                year=eff_year,
                source_name=file.filename,
                reset=False,
            )
            updated = _processed_doc_index().get(file.filename)
            if updated:
                _jobs[job_id]["doc"] = updated
            _jobs[job_id]["status"] = "ready"
        except Exception as exc:
            log.exception("ingest failed for %s", file.filename)
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)
            _jobs[job_id]["doc"]["status"] = "error"

    _executor.submit(_run)
    return initial_doc


@app.get("/api/documents/{doc_id}/status")
def document_status(doc_id: str) -> dict[str, Any]:
    """Return the current ingestion status for one uploaded document id."""
    if doc_id in _jobs:
        j = _jobs[doc_id]
        return {"id": doc_id, "status": j["status"], "error": j.get("error"), "document": j["doc"]}
    doc = _processed_doc_index().get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="not found")
    return {"id": doc_id, "status": "ready", "error": None, "document": doc}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, str]:
    """Delete one indexed document and return a summary of removed artifacts."""
    proc = settings.get_processed_path()
    stem = Path(doc_id).stem
    removed: list[str] = []

    for sub, suffix in (("pages", ".jsonl"), ("pages_enhanced", ".jsonl"),
                       ("chunks", ".jsonl"), ("datapoints", ".json"),
                       ("llamaparse", ".json"), ("markdown", ".md")):
        p = proc / sub / f"{stem}{suffix}"
        if p.exists():
            p.unlink()
            removed.append(str(p))
    cache = proc / "pages" / f"{stem}.enhanced.cache"
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)

    pdf = settings.reports_dir / doc_id
    if pdf.exists():
        pdf.unlink()
        removed.append(str(pdf))

    try:
        collection = get_collection()
        collection.delete(where={"source": doc_id})
    except Exception as exc:
        log.warning("chroma delete failed for %s: %s", doc_id, exc)

    _jobs.pop(doc_id, None)
    return {"id": doc_id, "removed": ",".join(removed)}


# ---------- datapoints ----------

@app.get("/api/datapoints")
def list_datapoints(company: str | None = None, type: str | None = None) -> list[dict[str, Any]]:
    """Return pre-extracted datapoints, optionally filtered by company or datapoint type."""
    items = query_datapoints(company=company)
    out: list[dict[str, Any]] = []
    for dp in items:
        dp_type = _bucket_type(dp.get("datapoint_type") or "")
        if type and dp_type != type:
            continue
        out.append({
            "company": dp.get("company") or "",
            "source": dp.get("source") or "",
            "page": dp.get("page") or 1,
            "type": dp_type,
            "metric": dp.get("metric") or "",
            "value": dp.get("value") or "",
            "period": dp.get("period") or dp.get("target_year") or "",
            "section": dp.get("scope") or dp.get("section") or "",
            "verbatim": dp.get("quote") or "",
        })
    return out


def _bucket_type(t: str) -> str:
    """Normalize a datapoint type into the bucket label exposed by the API."""
    return t.lower().strip() if t else "other"


# ---------- chat (SSE) ----------

class HistoryEntry(BaseModel):
    question: str
    answer: str


class ChatRequest(BaseModel):
    """Incoming chat request with question text and optional retrieval filters."""
    question: str
    company: str | None = None
    year: int | None = None
    top_k: int = 8
    history: list[HistoryEntry] = []


_STOPWORDS = {"the", "and", "of", "group", "company", "corp", "corporation",
              "inc", "nv", "n.v.", "plc", "ag", "sa", "s.a.", "ltd", "limited",
              "holdings", "holding"}


def _company_aliases(company: str) -> list[re.Pattern[str]]:
    """Generate flexible regex patterns for a company name and its tokens."""
    name = company.strip()
    if not name:
        return []
    patterns: list[re.Pattern[str]] = []
    seen: set[str] = set()

    def add(variant: str) -> None:
        """Register one company-name variant as a regex if it is useful."""
        v = variant.strip().casefold()
        if not v or v in seen or v in _STOPWORDS or len(v) < 2:
            return
        seen.add(v)
        flex = r"[\s._\-]+".join(re.escape(p) for p in v.split())
        patterns.append(re.compile(rf"(?<![a-z0-9]){flex}(?![a-z0-9])"))

    add(name)
    add(name.replace(".", "").replace("-", " "))
    tokens = [t for t in re.split(r"[\s._\-]+", name) if t]
    if len(tokens) > 1:
        for tok in tokens:
            if tok.casefold() not in _STOPWORDS and len(tok) >= 3:
                add(tok)
    return patterns


def _detect_company_from_question(question: str) -> str | None:
    """Match question against companies that are actually indexed."""
    q = question.casefold()
    candidates: list[tuple[int, str]] = []
    for doc in _processed_doc_index().values():
        company = doc.get("company")
        if not company:
            continue
        for pat in _company_aliases(company):
            if pat.search(q):
                candidates.append((len(company), company))
                break
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _sse(event: str, data: dict[str, Any] | str) -> str:
    """Format one Server-Sent Event payload line block and return it as text."""
    payload = data if isinstance(data, str) else json.dumps(data)
    return f"event: {event}\ndata: {payload}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Stream chat progress events and the final grounded answer as Server-Sent Events."""
    async def event_stream():
        """Yield SSE progress updates followed by the final answer event."""
        loop = asyncio.get_running_loop()
        yield _sse("phase", {"phase": "embedding", "detail": "text-embedding-3-small"})
        await asyncio.sleep(0)

        eff_company = req.company or _detect_company_from_question(req.question)
        query = RetrievalQuery(
            question=req.question,
            top_k=req.top_k,
            company=eff_company,
            year=req.year,
        )
        detail = f"top_k={req.top_k}"
        if eff_company and not req.company:
            detail += f" company={eff_company}"
        yield _sse("phase", {"phase": "searching", "detail": detail})
        history = [{"question": h.question, "answer": h.answer} for h in req.history]
        result = await loop.run_in_executor(_executor, retrieve_decomposed, query, history)
        yield _sse("phase", {"phase": "reading", "detail": f"{len(result.chunks)} chunks"})

        yield _sse("phase", {"phase": "drafting"})
        answer: VerbatimAnswer = await loop.run_in_executor(
            _executor,
            answer_question,
            req.question,
            result.chunks,
            history,
        )
        yield _sse("phase", {"phase": "citing", "detail": f"{len(answer.citations)} citations"})
        yield _sse("phase", {"phase": "done"})
        yield _sse("answer", json.loads(answer.model_dump_json()))

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------- pdf serving ----------

@app.get("/api/pdf/{source}")
def serve_pdf(source: str):
    """Return the stored PDF file response for the requested source filename."""
    from fastapi.responses import FileResponse

    pdf = settings.reports_dir / source
    if not pdf.exists():
        raise HTTPException(status_code=404, detail="pdf not found")
    return FileResponse(str(pdf), media_type="application/pdf", filename=source)


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    uvicorn.run("backend.app.server:app", host="0.0.0.0", port=8000, reload=False)
