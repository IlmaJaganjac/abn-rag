# Annual Report Rag

Local RAG over annual reports with a small chat UI, document management, datapoint extraction, and eval tooling.

You can upload annual reports, ask direct questions, ask multiple questions in sequence, and ask follow-up questions that depend on earlier answers. The backend rewrites follow-up questions when conversation history is present, then retrieves grounded context and answers with citations.

## What is in the codebase

- `backend/app/`
  FastAPI server, retrieval, answering, ingestion, parsing, chunking, embeddings, and datapoint extraction.
- `backend/evals/`
  YAML eval sets and the RAGAS evaluation runner.
- `backend/tests/`
  Focused backend tests.
- `frontend/src/`
  React frontend for chat, documents, and datapoints.
- `backend/data/`
  Local runtime data only. This is where reports, processed artifacts, and Chroma persistence live.

## Main features

- Upload annual report PDFs through the UI.
- Parse and chunk reports, then store them in Chroma.
- Extract structured datapoints such as FTE, ESG, financial highlights, and shareholder-return facts.
- Ask grounded questions with citations.
- Ask several questions in a row and continue with follow-up questions.
- Run RAGAS evaluations against YAML question sets.

## Requirements

- Docker Desktop
- A `.env` file with valid keys:

```env
OPENAI_API_KEY=...
LLAMA_CLOUD_API_KEY=...
HF_TOKEN=...
```

## Running with Docker

Start the full stack with:

```bash
docker compose up -d --build
```

If you prefer the older command spelling and your machine supports it:

```bash
docker-compose up --build
```

The app is hosted at:

```text
http://localhost
```

To watch backend logs:

```bash
docker compose logs -f backend
```

To stop everything:

```bash
docker compose down
```

## First startup

The first startup can take a bit longer.

The initial Docker build can also take a few minutes because some dependencies in this stack are fairly large, especially the ML-related libraries.

This project currently downloads the BAAI reranker at runtime on first start. You may need to wait a little before the system becomes ready. We are doing this because GitHub Actions has issues at the moment, and downloading the model outside the image is faster for us than baking it into the image during every build ;)

You may also see slower startup if the Hugging Face cache volume was removed and the reranker has to be downloaded again.

## How to use it

1. Open `http://localhost`.
2. Wait until the system is ready.
3. Upload one or more annual report PDFs.
4. Wait for indexing to finish.
5. Ask simple questions first, then continue with follow-up questions if needed.

Examples:

- `How many FTE did ABN AMRO have in 2025?`
- `What was ASML's net income in 2025?`
- `Who was the CEO?`
- `What about 2024?`

The last two examples are follow-ups. The system uses recent chat history to resolve them when history is available.

## Runtime data

Generated data is stored under:

- `backend/data/reports`
- `backend/data/processed`
- `backend/data/chroma`

This data is runtime state, not source code.

## Evaluations

RAGAS eval files live in `backend/evals/`.

Example:

```bash
.venv/bin/python -m backend.evals.ragas_eval \
  --questions backend/evals/abn-amro-2025.yaml \
  --company "ABN AMRO Bank" \
  --year 2025 \
  --metrics context \
  --runs-dir backend/evals/results
```

## Notes

- The reranker is currently configured through `.env`.
- First-run latency is expected to be higher than steady-state latency.
- If you want a fully clean local reset, remove `backend/data` and run `docker compose down -v`.
