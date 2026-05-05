import React, { useState, useEffect, useRef } from 'react';
import type { Document } from '../types';
import { api, AlreadyIndexedError } from '../api/client';
import { IUpload, IDoc, ITrash } from './Icons';
import type { Ingest } from '../App';

type ToastKind = 'indexing' | 'already' | 'ready';
interface Toast { id: string; kind: ToastKind; filename?: string; }

interface Props {
  ingest: Ingest[];
  setIngest: React.Dispatch<React.SetStateAction<Ingest[]>>;
  pollStatus: (jobId: string, docId: string, onReady?: () => void) => void;
  refreshDocsRef: React.MutableRefObject<(() => void) | null>;
}

export function DocumentsView({ ingest, setIngest, pollStatus, refreshDocsRef }: Props) {
  const [docs, setDocs] = useState<Document[]>([]);
  const [over, setOver] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const refresh = async () => {
    try { setDocs(await api.listDocuments()); } catch {}
  };

  useEffect(() => {
    refreshDocsRef.current = refresh;
    refresh();
    return () => { refreshDocsRef.current = null; };
  }, []);

  const dismissToast = (id: string) =>
    setToasts(arr => arr.filter(t => t.id !== id));

  const showToast = (kind: ToastKind, filename?: string) => {
    const id = `t-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    setToasts(arr => [...arr, { id, kind, filename }]);
    const ttl = kind === 'already' ? 6000 : kind === 'indexing' ? 7000 : null;
    if (ttl !== null) {
      setTimeout(() => dismissToast(id), ttl);
    }
  };

  const handleFiles = async (files: File[]) => {
    for (const file of files) {
      try {
        const doc = await api.uploadDocument(file);
        showToast('indexing');
        const jobId = `ing-${doc.id}-${Date.now()}`;
        const stage = doc.status === 'ready' ? 'done' : doc.status === 'embedding' ? 'embedding' : 'parsing';
        const pct = stage === 'done' ? 100 : stage === 'embedding' ? 70 : 25;
        setIngest(arr => [
          ...arr,
          { id: jobId, name: file.name, stage, pct },
        ]);
        pollStatus(jobId, doc.id, () => showToast('ready', file.name));
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

  return (
    <div className="view">
      <div className="docs-wrap">
        <div className="docs-hd">
          <h2>Documents</h2>
          <div className="docs-count">{docs.length} indexed</div>
        </div>

        <div
          className={`dropzone ${over ? 'over' : ''}`}
          onDragOver={e => { e.preventDefault(); setOver(true); }}
          onDragLeave={() => setOver(false)}
          onDrop={onDrop}
        >
          <div style={{ marginBottom: 8, opacity: 0.5 }}><IUpload size={22} /></div>
          <div className="dz-title">Drop annual reports here</div>
          <div className="dz-sub">PDF only</div>
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
                  <div className="doc-meta">{d.pages}pp</div>
                  <div className={`doc-status ready`} style={{ marginTop: 6, color: 'var(--ink-3)' }}>
                    <span className="dot" /> {d.status}
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
