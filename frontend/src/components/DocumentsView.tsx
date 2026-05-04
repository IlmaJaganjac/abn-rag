import React, { useState, useEffect, useRef } from 'react';
import type { Document, DocStatus } from '../types';
import { api, AlreadyIndexedError } from '../api/client';
import { IUpload, IDoc, ITrash, IUsers, ILeaf } from './Icons';

type Stage = 'parsing' | 'embedding' | 'done';
interface Ingest { id: string; name: string; stage: Stage; pct: number; }

type ToastKind = 'indexing' | 'already' | 'ready';
interface Toast { id: string; kind: ToastKind; filename?: string; }

function statusToStage(s: DocStatus): Stage {
  if (s === 'parsing' || s === 'queued') return 'parsing';
  if (s === 'ready') return 'done';
  return 'embedding';
}

function stageToPct(stage: Stage): number {
  if (stage === 'parsing') return 25;
  if (stage === 'embedding') return 70;
  return 100;
}

export function DocumentsView() {
  const [docs, setDocs] = useState<Document[]>([]);
  const [over, setOver] = useState(false);
  const [ingest, setIngest] = useState<Ingest[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = async () => {
    try { setDocs(await api.listDocuments()); } catch {}
  };

  useEffect(() => { refresh(); }, []);

  const dismissToast = (id: string) =>
    setToasts(arr => arr.filter(t => t.id !== id));

  const showToast = (kind: ToastKind, filename?: string) => {
    const id = `t-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    setToasts(arr => [...arr, { id, kind, filename }]);
    const ttl = kind === 'already' ? 6000 : kind === 'ready' ? 8000 : 7000;
    setTimeout(() => dismissToast(id), ttl);
  };

  const pollStatus = (jobId: string, docId: string) => {
    const tick = setInterval(async () => {
      try {
        const s = await api.getDocumentStatus(docId);
        const stage = statusToStage(s.status);
        setIngest(arr =>
          arr.map(j => (j.id === jobId ? { ...j, stage, pct: stageToPct(stage) } : j)),
        );
        if (s.status === 'ready' || s.status === 'error') {
          clearInterval(tick);
          await refresh();
          if (s.status === 'ready') showToast('ready', s.document?.source ?? docId);
          setTimeout(() => {
            setIngest(arr => arr.filter(j => j.id !== jobId));
          }, 1200);
        }
      } catch {
        // network blip — keep polling
      }
    }, 2000);
  };

  const handleFiles = async (files: File[]) => {
    for (const file of files) {
      try {
        const doc = await api.uploadDocument(file);
        showToast('indexing');
        const jobId = `ing-${doc.id}-${Date.now()}`;
        const initialStage = statusToStage(doc.status);
        setIngest(arr => [
          ...arr,
          { id: jobId, name: file.name, stage: initialStage, pct: stageToPct(initialStage) },
        ]);
        pollStatus(jobId, doc.id);
      } catch (err) {
        if (err instanceof AlreadyIndexedError) {
          showToast('already', file.name);
        } else {
          // surface other errors via the indexing toast for now
          console.error(err);
        }
      }
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setOver(false);
    handleFiles(Array.from(e.dataTransfer.files));
  };

  const onPick = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    if (files.length) handleFiles(files);
    e.target.value = '';
  };

  const onDelete = async (id: string) => {
    try {
      await api.deleteDocument(id);
      setDocs(arr => arr.filter(d => d.id !== id));
    } catch (err) {
      console.error(err);
    }
  };

  const totalChunks = docs.reduce((sum, d) => sum + (d.chunks ?? 0), 0);
  const totalMb = docs.reduce((sum, d) => sum + (d.size_mb ?? 0), 0);

  return (
    <div className="view">
      <div className="docs-wrap">
        <div className="docs-hd">
          <h2>Documents</h2>
          <div className="docs-count">
            {docs.length} indexed · {totalChunks.toLocaleString()} chunks · {totalMb.toFixed(1)} MB
          </div>
        </div>

        <div
          className={`dropzone ${over ? 'over' : ''}`}
          onDragOver={e => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={onDrop}
        >
          <div style={{ marginBottom: 8, opacity: 0.5 }}><IUpload size={22} /></div>
          <div className="dz-title">Drop annual reports here</div>
          <div className="dz-sub">PDF · up to ~500 pages · parsed with Docling, chunked, embedded into Postgres pgvector</div>
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf,.pdf"
            multiple
            style={{ display: 'none' }}
            onChange={onPick}
          />
          <button className="btn" onClick={() => fileInputRef.current?.click()}>
            <IUpload /> Choose file
          </button>
        </div>

        {ingest.length > 0 && (
          <div style={{ marginBottom: 24 }}>
            {ingest.map(j => (
              <div key={j.id} className="ingest-row">
                <div style={{ flex: '0 0 220px' }}>
                  <div className="name">{j.name}</div>
                  <div className="stages">
                    <span className={j.stage === 'parsing' ? 'active' : ''}>Parsing</span>
                    <span>·</span>
                    <span className={j.stage === 'embedding' ? 'active' : ''}>Embedding</span>
                    <span>·</span>
                    <span className={j.stage === 'done' ? 'active' : ''}>Indexed</span>
                  </div>
                </div>
                <div className="progress-bar"><div className="fill" style={{ width: `${j.pct}%` }} /></div>
                <div className="mono" style={{ fontSize: 11, color: 'var(--ink-3)', minWidth: 38, textAlign: 'right' }}>
                  {Math.round(j.pct)}%
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="docs-grid">
          {docs.map(d => (
            <div key={d.id} className="doc-card">
              <div className="doc-card-hd">
                <div className="doc-icon"><IDoc size={18} /></div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div className="doc-title">{d.title}</div>
                  <div className="doc-meta">{d.source} · {d.pages}pp · {d.size_mb.toFixed(1)} MB</div>
                  <div className={`doc-status ready`} style={{ marginTop: 6, color: 'var(--ink-3)' }}>
                    <span className="dot" /> {d.status} · {d.chunks.toLocaleString()} chunks · {d.parser}
                  </div>
                </div>
                <button
                  className="btn"
                  style={{ padding: '6px 8px' }}
                  aria-label="Remove"
                  onClick={() => onDelete(d.id)}
                >
                  <ITrash />
                </button>
              </div>
              <div className="doc-stats">
                <div>
                  <div className="doc-stat-label"><IUsers /> FTE (extracted)</div>
                  <div className="doc-stat-value">{d.fte ?? '—'}</div>
                </div>
                <div>
                  <div className="doc-stat-label"><ILeaf /> Net-zero year</div>
                  <div className="doc-stat-value">{d.net_zero_year ?? '—'}</div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Upload toasts */}
      <div className="toast-stack">
        {toasts.map(t => (
          <div key={t.id} className="toast" role="status">
            <div className="toast-icon" aria-hidden="true">
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                <path d="M17 8h1a4 4 0 0 1 0 8h-1" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                <path d="M3 8h14v9a4 4 0 0 1-4 4H7a4 4 0 0 1-4-4V8z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
                <path d="M7 1.5c0 1.5-1 1.5-1 3s1 1.5 1 3M11 1.5c0 1.5-1 1.5-1 3s1 1.5 1 3M15 1.5c0 1.5-1 1.5-1 3s1 1.5 1 3" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              </svg>
            </div>
            <div className="toast-body">
              {t.kind === 'already' ? (
                <>
                  <div className="toast-title">Already indexed</div>
                  <div className="toast-text">
                    <code>{t.filename}</code> is already in your index — no need to re-upload.
                  </div>
                </>
              ) : t.kind === 'ready' ? (
                <>
                  <div className="toast-title">Ready to query</div>
                  <div className="toast-text"><code>{t.filename}</code> is indexed and available in chat.</div>
                </>
              ) : (
                <>
                  <div className="toast-title">Hang tight — we&rsquo;re indexing your document</div>
                  <div className="toast-text">This can take a minute. Grab a quick coffee, or feel free to keep asking questions — we&rsquo;ll let you know when it&rsquo;s ready.</div>
                </>
              )}
            </div>
            <button className="toast-close" aria-label="Dismiss" onClick={() => dismissToast(t.id)}>×</button>
          </div>
        ))}
      </div>
    </div>
  );
}
