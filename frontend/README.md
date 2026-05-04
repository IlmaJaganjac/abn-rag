# Verbatim Frontend

React + TypeScript + Vite frontend for the annual-report RAG system.

## Develop

```bash
cd frontend
npm install
npm run dev
```

The dev server runs on `http://localhost:5173` and proxies `/api/*` to
`http://localhost:8000` (your FastAPI backend — adjust `vite.config.ts` if
your backend runs elsewhere).

## Backend contract

The frontend talks to FastAPI through `src/api/client.ts`. Endpoints expected
(map these to `backend/app/`):

| Method | Path                        | Returns                  |
| ------ | --------------------------- | ------------------------ |
| GET    | `/api/documents`            | `Document[]`             |
| POST   | `/api/documents` (multipart)| `Document` (ingest job)  |
| DELETE | `/api/documents/:id`        | `204`                    |
| GET    | `/api/datapoints`           | `Datapoint[]`            |
| POST   | `/api/chat`                 | `VerbatimAnswer` (SSE)   |

`VerbatimAnswer` mirrors `backend/app/schemas.py` exactly — see `src/types.ts`.

## Build

```bash
npm run build       # type-check + vite build → dist/
npm run typecheck   # tsc only
```

## Project preview (no Node)

For viewing the design without a build step, open `../Verbatim.html` at the
project root — it loads the same `.tsx` files through Babel-standalone in the
browser. This is for design preview only; production must use `npm run build`.
