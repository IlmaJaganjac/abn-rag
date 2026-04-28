# Project: abn-rag

Local RAG over annual reports. ASML 2025 baseline; multi-company-ready.

## Working principles

- **Eval-first.** `evals/questions.yaml` is ground truth. Don't optimize
  retrieval/prompts before the runner produces a baseline score.
- **Phase order:** schemas → ingest → chunk → embed → store → retrieve →
  answer → eval → API → frontend. Don't skip ahead.
- **Multi-company-ready.** Every chunk carries `source`, `company`, `year`.
  Don't hardcode "ASML" in retrieval or answering logic.
- **Always cite.** A non-refused `VerbatimAnswer` must include ≥1 `Citation`
  (`source` + `page`). Prefer verbatim spans for financial/ESG datapoints.
- **Refuse, don't fabricate.** Out-of-corpus or unsupported questions →
  `refused=True` with a reason. Hallucination checks in the eval set test this.

## Conventions

- Page numbers are 1-indexed PDF pages (not the printed page numbers).
- Chunk IDs are deterministic: `{source}:{page}:{idx}`.
- ChromaDB persists in `backend/data/chroma/` — never commit it.
- PDFs live in `reports/` — never commit them.
- Settings come from `backend/app/config.py` (`pydantic-settings`, reads `.env`).

## Out of scope right now

Ingestion, retrieval, answering, eval runner, API, frontend — placeholders only.
