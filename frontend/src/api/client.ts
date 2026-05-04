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
} from '../types';

const BASE = '/api';

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}${body ? ` — ${body}` : ''}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
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
    return jsonOrThrow(await fetch(`${BASE}/documents`, { method: 'POST', body: fd }));
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
   *   event: phase\ndata: {"phase":"searching","detail":"top_k=12"}
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
      top_k?: number;
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
        top_k: opts.top_k ?? 8,
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
