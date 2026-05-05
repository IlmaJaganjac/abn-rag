/**
 * Typed client for the Verbatim FastAPI backend.
 *
 * In dev, Vite proxies /api/* to http://localhost:8000 (see vite.config.ts).
 * In prod, point this at wherever you serve the backend.
 */

import type {
  Document,
  Datapoint,
  VerbatimAnswer,
  ThinkingPhase,
  DocStatus,
  SystemStatus,
} from '../types';

const BASE = '/api';

export class AlreadyIndexedError extends Error {
  document: Document;
  constructor(message: string, document: Document) {
    super(message);
    this.name = 'AlreadyIndexedError';
    this.document = document;
  }
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}${body ? ` — ${body}` : ''}`);
  }
  return res.json() as Promise<T>;
}

export interface DocumentStatusResponse {
  id: string;
  status: DocStatus;
  error: string | null;
  document: Document | null;
}

export const api = {
  async getSystemStatus(): Promise<SystemStatus> {
    return jsonOrThrow(await fetch(`${BASE}/system/status`));
  },

  async listDocuments(): Promise<Document[]> {
    return jsonOrThrow(await fetch(`${BASE}/documents`));
  },

  async uploadDocument(
    file: File,
    meta: { company?: string; year?: number } = {},
  ): Promise<Document> {
    const fd = new FormData();
    fd.append('file', file);
    if (meta.company) fd.append('company', meta.company);
    if (meta.year != null) fd.append('year', String(meta.year));
    const res = await fetch(`${BASE}/documents`, { method: 'POST', body: fd });
    if (res.status === 409) {
      let payload: any = null;
      try { payload = await res.json(); } catch {}
      const detail = payload?.detail ?? {};
      const doc = detail.document as Document;
      const message = detail.message ?? `${file.name} is already indexed.`;
      throw new AlreadyIndexedError(message, doc);
    }
    return jsonOrThrow(res);
  },

  async getDocumentStatus(id: string): Promise<DocumentStatusResponse> {
    return jsonOrThrow(
      await fetch(`${BASE}/documents/${encodeURIComponent(id)}/status`),
    );
  },

  async deleteDocument(id: string): Promise<void> {
    const res = await fetch(`${BASE}/documents/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  },

  async listDatapoints(filter: { company?: string; type?: string } = {}): Promise<Datapoint[]> {
    const qs = new URLSearchParams();
    if (filter.company) qs.set('company', filter.company);
    if (filter.type) qs.set('type', filter.type);
    return jsonOrThrow(
      await fetch(`${BASE}/datapoints${qs.toString() ? `?${qs}` : ''}`),
    );
  },

  /**
   * Streamed chat — backend is expected to emit Server-Sent Events:
   *   event: phase\ndata: {"phase":"searching","detail":"top_k=<backend setting>"}
   *   event: answer\ndata: <VerbatimAnswer JSON>
   *
   * The handler `onPhase` fires for in-flight thinking phases; the returned
   * promise resolves with the final VerbatimAnswer.
   */
  async ask(
    question: string,
    opts: {
      company?: string | null;
      year?: number | null;
      history?: { question: string; answer: string }[];
      onPhase?: (p: ThinkingPhase, detail?: string) => void;
      signal?: AbortSignal;
    } = {},
  ): Promise<VerbatimAnswer> {
    const res = await fetch(`${BASE}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({
        question,
        company: opts.company ?? null,
        year: opts.year ?? null,
        history: opts.history ?? [],
      }),
      signal: opts.signal,
    });
    if (!res.ok || !res.body) throw new Error(`${res.status} ${res.statusText}`);

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let answer: VerbatimAnswer | null = null;

    // Minimal SSE parser
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split('\n\n');
      buffer = events.pop() ?? '';
      for (const evt of events) {
        const lines = evt.split('\n');
        const type = lines.find((l) => l.startsWith('event:'))?.slice(6).trim() ?? 'message';
        const data = lines.find((l) => l.startsWith('data:'))?.slice(5).trim() ?? '';
        if (!data) continue;
        if (type === 'phase') {
          const parsed = JSON.parse(data) as { phase: ThinkingPhase; detail?: string };
          opts.onPhase?.(parsed.phase, parsed.detail);
        } else if (type === 'answer') {
          answer = JSON.parse(data) as VerbatimAnswer;
        }
      }
    }
    if (!answer) throw new Error('No answer returned from /chat');
    return answer;
  },
};
