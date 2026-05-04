import React, { useEffect, useRef, useState } from 'react';
import { IExternal } from './Icons';

/**
 * Slide-in PDF viewer. Uses pdf.js (loaded from CDN on first open) to render
 * the cited page on a canvas. When the real backend is connected, set the
 * `pdfUrl` prop to e.g. `/api/documents/${source}/pdf` so the file streams
 * from the server. Until then, it renders a stylized placeholder for the
 * cited page so the user can see the citation context.
 */

interface PdfViewerProps {
  open: boolean;
  source: string | null;
  page: number | null;
  quote: string | null;
  onClose: () => void;
}

declare const pdfjsLib: any;

const PDFJS_CDN = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.0.379/pdf.min.mjs';
const PDFJS_WORKER = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.0.379/pdf.worker.min.mjs';

let pdfjsPromise: Promise<any> | null = null;
function loadPdfjs(): Promise<any> {
  if ((window as any).pdfjsLib) return Promise.resolve((window as any).pdfjsLib);
  if (pdfjsPromise) return pdfjsPromise;
  pdfjsPromise = import(/* @vite-ignore */ PDFJS_CDN).then((mod: any) => {
    mod.GlobalWorkerOptions.workerSrc = PDFJS_WORKER;
    (window as any).pdfjsLib = mod;
    return mod;
  });
  return pdfjsPromise;
}

export function PdfViewer({ open, source, page, quote, onClose }: PdfViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'rendered' | 'no-source' | 'error'>('idle');
  const [errMsg, setErrMsg] = useState<string>('');
  const [pageNum, setPageNum] = useState<number>(page ?? 1);
  const [totalPages, setTotalPages] = useState<number>(0);

  useEffect(() => { if (open && page) setPageNum(page); }, [open, page, source]);

  useEffect(() => {
    if (!open || !source) return;
    let cancelled = false;

    const url = `/api/documents/${encodeURIComponent(source)}/pdf`;

    (async () => {
      setStatus('loading');
      try {
        // Probe whether backend serves the file. In demo mode we expect a 404 → fallback.
        const probe = await fetch(url, { method: 'HEAD' }).catch(() => null);
        if (!probe || !probe.ok) {
          if (!cancelled) { setStatus('no-source'); }
          return;
        }
        const lib = await loadPdfjs();
        const doc = await lib.getDocument(url).promise;
        if (cancelled) return;
        setTotalPages(doc.numPages);
        const target = Math.min(Math.max(pageNum, 1), doc.numPages);
        const p = await doc.getPage(target);
        const canvas = canvasRef.current; if (!canvas) return;
        const ctx = canvas.getContext('2d')!;
        const viewport = p.getViewport({ scale: 1.4 });
        canvas.width = viewport.width;
        canvas.height = viewport.height;
        await p.render({ canvasContext: ctx, viewport }).promise;
        if (!cancelled) setStatus('rendered');
      } catch (e: any) {
        if (!cancelled) { setStatus('error'); setErrMsg(String(e?.message ?? e)); }
      }
    })();

    return () => { cancelled = true; };
  }, [open, source, pageNum]);

  // Keyboard nav
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowLeft' && pageNum > 1) setPageNum(n => n - 1);
      if (e.key === 'ArrowRight' && (totalPages === 0 || pageNum < totalPages)) setPageNum(n => n + 1);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, pageNum, totalPages, onClose]);

  if (!open) return null;

  return (
    <>
      <div
        onClick={onClose}
        style={{
          position: 'fixed', inset: 0, background: 'oklch(15% 0.01 155 / 0.4)',
          zIndex: 80, animation: 'fade-in 160ms ease',
        }}
      />
      <aside
        role="dialog"
        aria-label={`Source ${source} page ${pageNum}`}
        style={{
          position: 'fixed', top: 0, right: 0, bottom: 0, width: 'min(640px, 92vw)',
          background: 'var(--paper)', borderLeft: '1px solid var(--line)',
          boxShadow: '-12px 0 40px -12px oklch(0% 0 0 / 0.18)',
          zIndex: 81, display: 'flex', flexDirection: 'column',
          animation: 'slide-in 200ms cubic-bezier(0.2, 0.7, 0.2, 1)',
        }}
      >
        <header style={{
          display: 'flex', alignItems: 'center', gap: 12,
          padding: '14px 18px', borderBottom: '1px solid var(--line)',
        }}>
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: 'var(--ink-3)' }}>
              SOURCE
            </div>
            <div style={{ fontWeight: 500, fontSize: 14, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
              {source ?? '—'}
            </div>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 12, color: 'var(--ink-2)' }}>
            <button className="btn" style={{ padding: '4px 8px' }} onClick={() => setPageNum(n => Math.max(1, n - 1))} aria-label="Previous page">‹</button>
            <span className="mono" style={{ minWidth: 56, textAlign: 'center' }}>
              p.{pageNum}{totalPages ? ` / ${totalPages}` : ''}
            </span>
            <button className="btn" style={{ padding: '4px 8px' }}
              onClick={() => setPageNum(n => totalPages ? Math.min(totalPages, n + 1) : n + 1)} aria-label="Next page">›</button>
          </div>
          <button className="btn" onClick={onClose} aria-label="Close" style={{ padding: '4px 10px' }}>✕</button>
        </header>

        {quote && (
          <div style={{
            padding: '12px 18px', borderBottom: '1px solid var(--line)',
            background: 'oklch(98% 0.025 95)',
            fontFamily: "'Newsreader', serif", fontStyle: 'italic',
            fontSize: 14, color: 'var(--ink)', lineHeight: 1.5,
            borderLeft: '2px solid var(--accent)',
          }}>
            &ldquo;{quote}&rdquo;
          </div>
        )}

        <div style={{
          flex: 1, overflow: 'auto', background: 'var(--bg-2)',
          padding: 24, display: 'flex', justifyContent: 'center', alignItems: 'flex-start',
        }}>
          {status === 'loading' && (
            <div style={{ color: 'var(--ink-3)', padding: 60, fontSize: 13 }}>Loading PDF…</div>
          )}
          {status === 'error' && (
            <div style={{ color: 'var(--danger)', padding: 60, fontSize: 13, textAlign: 'center' }}>
              Failed to render PDF<br /><span className="mono" style={{ fontSize: 11 }}>{errMsg}</span>
            </div>
          )}
          {status === 'no-source' && (
            <PlaceholderPage source={source} pageNum={pageNum} quote={quote} />
          )}
          <canvas
            ref={canvasRef}
            style={{
              display: status === 'rendered' ? 'block' : 'none',
              maxWidth: '100%', height: 'auto',
              background: 'white',
              boxShadow: '0 4px 24px -8px oklch(0% 0 0 / 0.18)',
              borderRadius: 4,
            }}
          />
        </div>

        <footer style={{
          padding: '10px 18px', borderTop: '1px solid var(--line)',
          fontSize: 11, color: 'var(--ink-3)',
          display: 'flex', alignItems: 'center', gap: 12,
        }}>
          <span>← / → page · Esc to close</span>
          <div style={{ flex: 1 }} />
          <a
            href={source ? `/api/documents/${encodeURIComponent(source)}/pdf#page=${pageNum}` : '#'}
            target="_blank" rel="noreferrer"
            style={{ color: 'var(--primary)', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: 4 }}
          >
            Open in tab <IExternal size={11} />
          </a>
        </footer>
      </aside>

      <style>{`
        @keyframes fade-in { from { opacity: 0; } to { opacity: 1; } }
        @keyframes slide-in { from { transform: translateX(20px); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
      `}</style>
    </>
  );
}

function PlaceholderPage({ source, pageNum, quote }: { source: string | null; pageNum: number; quote: string | null }) {
  // A stylized "page" so the citation context is shown even without a real PDF.
  return (
    <div style={{
      width: '100%', maxWidth: 520, aspectRatio: '8.5 / 11',
      background: 'white', boxShadow: '0 4px 24px -8px oklch(0% 0 0 / 0.18)', borderRadius: 4,
      padding: '52px 56px', position: 'relative', overflow: 'hidden',
      fontFamily: "'Newsreader', serif", color: 'oklch(20% 0.01 155)',
    }}>
      <div style={{
        fontSize: 9, textTransform: 'uppercase', letterSpacing: 0.12,
        color: 'oklch(55% 0.02 155)', display: 'flex', justifyContent: 'space-between',
      }}>
        <span>{source}</span>
        <span>Page {pageNum}</span>
      </div>
      <div style={{
        marginTop: 24,
        fontSize: 18, fontWeight: 500, letterSpacing: '-0.01em',
      }}>
        Note 18 — Employees
      </div>
      <div style={{ marginTop: 16, fontSize: 11, lineHeight: 1.7, color: 'oklch(35% 0.01 155)' }}>
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} style={{
            height: 8, background: 'oklch(92% 0.005 155)', borderRadius: 2,
            marginBottom: 6, width: `${85 - i * 4}%`,
          }} />
        ))}
      </div>
      {quote && (
        <div style={{
          margin: '24px 0', padding: '12px 14px',
          background: 'oklch(97% 0.04 95)',
          borderLeft: '2px solid oklch(75% 0.16 95)',
          fontSize: 12, lineHeight: 1.5, fontStyle: 'italic',
        }}>
          &ldquo;{quote}&rdquo;
        </div>
      )}
      <div style={{ fontSize: 11, lineHeight: 1.7, color: 'oklch(35% 0.01 155)' }}>
        {Array.from({ length: 12 }).map((_, i) => (
          <div key={i} style={{
            height: 8, background: 'oklch(92% 0.005 155)', borderRadius: 2,
            marginBottom: 6, width: `${100 - (i % 4) * 8}%`,
          }} />
        ))}
      </div>
      <div style={{
        position: 'absolute', bottom: 28, left: 56, right: 56,
        display: 'flex', justifyContent: 'space-between',
        fontSize: 9, color: 'oklch(55% 0.02 155)',
        borderTop: '1px solid oklch(90% 0.005 155)', paddingTop: 8,
      }}>
        <span>{source}</span>
        <span>{pageNum}</span>
      </div>
      <div style={{
        position: 'absolute', top: 12, right: 12,
        fontSize: 9, color: 'oklch(60% 0.05 95)',
        background: 'oklch(96% 0.04 95)', padding: '2px 6px', borderRadius: 3,
      }}>preview · backend offline</div>
    </div>
  );
}
