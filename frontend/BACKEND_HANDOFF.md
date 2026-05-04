# Verbatim — Backend Handoff

This document is the contract between the **Verbatim frontend** (this repo, `frontend/`) and the **RAG backend** an agent is about to wire up. Read this first; it tells you exactly which endpoints to expose, what they must return, and what the UI does with the response.

The frontend is currently demo-mode: every list, every answer, every citation is hardcoded in `frontend/src/mock.ts`. A typed client already exists at `frontend/src/api/client.ts` and matches the contract below — turning the prototype into a live product is mostly a matter of standing up the endpoints, then swapping `MOCK_*` imports for `api.*` calls in three components.

---

## 1. Source of truth

| What | Where |
|---|---|
| API base URL | `/api` (Vite dev server proxies to `http://localhost:8000`) |
| Pydantic schemas (Python) | `backend/app/schemas.py` *(to be created — match `frontend/src/types.ts`)* |
| Frontend types (TS) | `frontend/src/types.ts` |
| API client | `frontend/src/api/client.ts` |
| Mock fixtures | `frontend/src/mock.ts` (delete or gate once API is live) |

**Rule:** if you change a field name or shape in Python, mirror it in `frontend/src/types.ts` in the same PR. The two files are a single contract expressed in two languages.

---

## 2. Endpoints to implement

All JSON unless noted. CORS not needed in dev (proxy handles it).

### `GET /api/documents` → `Document[]`

List every ingested document. Powers the **Documents** view and the sidebar's "Documents" badge count.

The `Document` shape (see `types.ts`) carries enough pre-extracted summary to render a card without a second roundtrip — `fte`, `net_zero_year`, `pages`, `chunks`, `parser`, `status`. Keep that surface; the cards rely on it.

`status` drives the "queued / parsing / embedding / ready / error" pill and progress bar in the upload list. Stream status changes by re-polling this endpoint every ~2s while any document is non-`ready`, or push via SSE if you prefer (`GET /api/documents/stream`).

### `POST /api/documents` (multipart)

Fields: `file` (PDF), optional `company`, optional `year` (int). Returns the new `Document` with `status: "queued"`. The ingest pipeline (parse → chunk → embed → index) runs async; the UI polls `GET /api/documents` to watch the row progress.

### `DELETE /api/documents/:id` → `204`

Drops the document and its chunks/embeddings. Idempotent.

### `GET /api/datapoints?company=&type=` → `Datapoint[]`

Pre-extracted structured facts (FTE, ESG targets, financials). The Datapoints view renders these in a table; every row links back to a source PDF + page + verbatim quote. This is the "trust by construction" layer — every value here MUST be backed by an exact substring in `verbatim` that exists in the source page.

`type` is one of `fte | esg | financial | other`. `period` is a year (number) or a string like `"target"` / `"2024→2028"`. Keep it permissive.

### `POST /api/chat` → SSE stream

The flagship endpoint. Request body:

```json
{
  "question": "How many total employees did ASML have at the end of 2025?",
  "company": null,
  "year": null,
  "top_k": 8
}
```

Response is a `text/event-stream` with two event types:

```
event: phase
data: {"phase":"searching","detail":"top_k=8"}

event: phase
data: {"phase":"reading","detail":"asml.pdf p.301"}

event: answer
data: {"question":"...","answer":"...","verbatim":"...","citations":[...],"refused":false,"refusal_reason":null,"raw_citations":[...],"grounding_drops":[...]}
```

**Phases** drive the live "thinking" indicator in chat. The frontend treats them as opaque labels — emit the ones in the `ThinkingPhase` union (`queued | embedding | searching | reading | drafting | citing | done`) in roughly that order. `detail` is a short free-form string shown in small caps next to the spinner; use it for what's actually happening (chunk count, page being read, etc.).

**The final `answer` event** is a `VerbatimAnswer`. Field semantics:

- `answer` — natural-language synthesis. Plain prose, no markdown citations.
- `verbatim` — the single most important quote underlying the answer, copied EXACTLY from a source page. Null only when refused.
- `citations` — final, post-grounding-check list. Each must have a `quote` that appears verbatim on `source` page `page`.
- `raw_citations` — what the LLM originally produced, before grounding checks. Useful for debugging; the UI doesn't render it but keep it in the payload.
- `grounding_drops` — citations the grounding step rejected, with a `reason` from the enum. Surfaced in a "−N dropped" debug chip later; for now just include them.
- `refused` / `refusal_reason` — set when the model declines (no evidence in corpus, off-topic, etc.). The UI renders a distinct refusal card; do NOT fall back to a half-answer when refusing.

**Determinism note:** the UI shows `verbatim` and each `citation.quote` as exact text. If your model paraphrases, the grounding step must either (a) replace the quote with a verbatim substring from the cited page, or (b) move the citation into `grounding_drops`. The frontend does not re-validate.

### `GET /api/documents/:source/pdf` → `application/pdf`

Stream the original file. `source` is the filename as it appears in `Document.source` and `Citation.source` (e.g. `asml.pdf`). The PDF viewer uses this URL with `pdf.js`; it also accepts `#page=N` deep-links.

`PdfViewer.tsx` currently HEADs this URL and falls back to a stylized placeholder if the response is not OK. Once you implement it, the placeholder code path becomes dead — leave it for now; it's a graceful degrade for offline demos.

---

## 3. Wiring the frontend

Three files import from `mock.ts`. Replace them in order:

1. **`App.tsx`** — `MOCK_DOCS.length`, `MOCK_DATAPOINTS.length` for sidebar badges. Lift these into a small store (Zustand or a React context) populated from `api.listDocuments()` / `api.listDatapoints()` on mount.
2. **`DocumentsView.tsx`** — replace `useState<Document[]>(MOCK_DOCS)` with a fetch + polling hook. Keep the local `ingest` state for in-flight uploads; merge with server list on each poll.
3. **`ChatView.tsx`** — replace the `phaseTimings` simulator (search for `// Simulated phase progression`) with `api.ask(question, { onPhase, signal })`. The `onPhase` callback already maps 1:1 to the `setPhase` / `setPhaseDetail` calls used by the simulator.
4. **`DatapointsView.tsx`** — same pattern as Documents: fetch on mount, re-fetch on filter change (or filter client-side; the dataset is small).

Do NOT delete `mock.ts` yet. Gate it behind `if (import.meta.env.VITE_USE_MOCKS)` so the offline demo / Storybook keeps working.

---

## 4. Backend pipeline — what we're assuming

The frontend does not care HOW the answer is produced, but here's the implied pipeline so you build the right thing:

1. **Ingest** (`POST /api/documents`)
   - Parse PDF with Docling or LlamaParse (the `parser` field on `Document` records which was used).
   - Chunk by section; carry `section_path`, `page`, `chunk_kind` on every chunk.
   - Embed with whichever model the team standardised on; persist chunks + vectors.
   - Run a **pre-extraction pass** to fill `Document.fte`, `Document.net_zero_year`, and seed the `Datapoint` table. This is what makes the cards and the Datapoints view feel instant.

2. **Retrieve** (`POST /api/chat` step 1)
   - Hybrid search (BM25 + vector) → top-k chunks. Filter by `company`/`year` if the request constrains them.
   - Return `RetrievedChunk[]` internally (see `types.ts`). The UI doesn't see this, but log it.

3. **Generate** (`POST /api/chat` step 2)
   - LLM produces a `VerbatimAnswer` with `raw_citations`.
   - **Grounding pass**: for each `raw_citation`, verify `quote` appears on `(source, page)`. Drop failures into `grounding_drops` with a reason; promote survivors to `citations`.
   - If 0 citations survive, set `refused: true` with a `refusal_reason` like *"No supporting passage found in the indexed corpus."*
   - Pick the strongest surviving citation's quote as the top-level `verbatim`.

4. **Stream** the phase events as you go. Don't batch — the UI's "thinking" feel depends on phases arriving live.

---

## 5. Things the prototype gets wrong on purpose

Worth knowing so you don't faithfully reproduce them:

- **Highlight rectangles in the PDF viewer** are mock coordinates — they don't actually mark the cited passage. When you wire the real backend, either (a) compute bounding boxes during chunking and add a `bbox: [x, y, w, h]` to `Citation`, or (b) drop the highlight feature and keep just the page deep-link.
- **Phase timings** in chat are hardcoded delays. Real phases will be jagged (search is fast, reading is slow). That's fine — the UI handles arbitrary timing.
- **Mock answers** include cherry-picked easy questions (ASML FTE, Shell climate spend, ABN workforce reduction). The refusal example (`"How many company cars did ASML own in 2025?"`) is the contract for a clean refusal — match its shape.
- **No auth.** Add a session cookie or bearer token before this leaves a localhost; the client has no opinions about how.

---

## 6. Smoke test once wired up

1. `GET /api/documents` returns ≥1 ready document → sidebar badge shows the count, Documents view renders cards.
2. `POST /api/documents` with a small PDF → row appears with `queued`, transitions through `parsing`/`embedding`/`ready` on poll.
3. `POST /api/chat` with `"How many total employees did ASML have at the end of 2025?"` → phase events stream, final answer has ≥1 citation, `verbatim` substring exists on the cited page.
4. Click a citation chip → PDF viewer opens, fetches `/api/documents/asml.pdf/pdf`, jumps to the cited page.
5. Ask a question with no answer in the corpus → `refused: true`, refusal card renders.

If all five pass, the integration is done. Hand back to design for polish.
