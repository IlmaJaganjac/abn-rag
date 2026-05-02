# abn-rag

Local Retrieval-Augmented Generation over annual reports.

Baseline scope: **ASML 2025**. Designed to extend to more companies/years.

## Stack

- Python 3.11+
- LlamaParse — PDF parser
- OpenAI `text-embedding-3-small` — embeddings
- ChromaDB — persistent local vector store
- Pydantic v2 — schemas and settings

## Quickstart

```bash
pip install -e .[dev]
cp .env.example .env       # add your OPENAI_API_KEY
# drop the ASML 2025 PDF into reports/

# 1. ingest the PDF (LlamaParse parses pages, chunks embed + persist to ChromaDB)
python -m backend.app.ingestion \
    reports/asml-2025-annual-report-based-on-us-gaap.pdf \
    --company ASML --year 2025

# 2. ask a question — prints a structured VerbatimAnswer (JSON)
python -m backend.app.pipeline \
    "How many total employees did ASML have in 2025?" \
    --company ASML --year 2025

# add --show-context to also print the retrieved chunks above the JSON
python -m backend.app.pipeline "..." --company ASML --year 2025 --show-context

# 3. run the eval set against the live pipeline
python -m evals.runner
python -m evals.runner --show-failures-only
python -m evals.runner --top-k 12
python -m evals.runner --no-save              # don't write a run record
```

Each run writes a JSON record to `evals/runs/` named
`<UTC-timestamp>_<passed>of<total>.json`. The record includes the run config
(model, top-k, persist dir), the per-question outcomes (full `VerbatimAnswer`
+ pass/fail + reasons), and the by-category summary, so you can diff baselines
across changes. Run files are gitignored; the directory is kept via
`.gitkeep`.

`--reset` on ingestion drops and rebuilds the collection. Parsed page text is
written to `backend/data/processed/pages/` for inspection and future API/UI use.
Parser artifacts are written under `backend/data/processed/llamaparse/` and
`backend/data/processed/markdown/`
when those parsers are used.
All commands need
`OPENAI_API_KEY` in `.env` (retrieval embeds the query, answering calls the
chat model).

Out-of-corpus questions (e.g. "How many employees does Tesla have?") return
`refused=true` with `citations=[]` instead of fabricating an answer.

## Layout

```
backend/app/   schemas, config, ingestion/retrieval/answer/pipeline (phase 2)
backend/data/  ChromaDB + processed parser artifacts (gitignored)
evals/         questions.yaml + runner (phase 2)
reports/       PDFs (gitignored)
```

## Phases

1. Skeleton + schemas + config
2. Ingestion + retrieval baseline
3. Structured `VerbatimAnswer` with citations (LLM)
4. **Eval runner against `evals/questions.yaml`** ← current
5. FastAPI + frontend chat UI

The first useful milestone is a baseline eval score, not a polished UI.
