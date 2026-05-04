import React, { useState } from 'react';
import { MOCK_DOCS } from '../mock';
import type { Document } from '../types';
import { IUpload, IDoc, ITrash, IUsers, ILeaf } from './Icons';

interface Ingest { id: string; name: string; stage: 'parsing' | 'embedding' | 'done'; pct: number; }

interface Toast { id: string; }

export function DocumentsView() {
  const [docs] = useState<Document[]>(MOCK_DOCS);
  const [over, setOver] = useState(false);
  const [ingest, setIngest] = useState<Ingest[]>([]);
  const [toasts, setToasts] = useState<Toast[]>([]);

  const showToast = () => {
    const id = `t-${Date.now()}`;
    setToasts(arr => [...arr, { id }]);
    setTimeout(() => setToasts(arr => arr.filter(t => t.id !== id)), 7000);
  };
  const dismissToast = (id: string) => setToasts(arr => arr.filter(t => t.id !== id));

  const simulate = (name: string) => {
    showToast();
    const id = `ing-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`;
    setIngest(arr => [...arr, { id, name, stage: 'parsing', pct: 6 }]);
    let pct = 6;
    const tick = setInterval(() => {
      pct += 4 + Math.random() * 8;
      setIngest(arr => arr.map(j => {
        if (j.id !== id) return j;
        if (pct >= 100) return { ...j, pct: 100, stage: 'done' };
        if (pct > 55) return { ...j, pct, stage: 'embedding' };
        return { ...j, pct };
      }));
      if (pct >= 100) {
        clearInterval(tick);
        setTimeout(() => setIngest(arr => arr.filter(j => j.id !== id)), 1400);
      }
    }, 220);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setOver(false);
    Array.from(e.dataTransfer.files).forEach(f => simulate(f.name));
  };

  return (
    <div className="view">
      <div className="docs-wrap">
        <div className="docs-hd">
          <h2>Documents</h2>
          <div className="docs-count">{docs.length} indexed · 22,265 chunks · 100.4 MB</div>
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
          <button className="btn" onClick={() => simulate(`upload-${Date.now() % 1000}.pdf`)}>
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
                <button className="btn" style={{ padding: '6px 8px' }} aria-label="Remove">
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
              <div className="toast-title">Hang tight — we&rsquo;re indexing your document</div>
              <div className="toast-text">This can take a minute. Grab a quick coffee, or feel free to keep asking questions — we&rsquo;ll let you know when it&rsquo;s ready.</div>
            </div>
            <button className="toast-close" aria-label="Dismiss" onClick={() => dismissToast(t.id)}>×</button>
          </div>
        ))}
      </div>
    </div>
  );
}
