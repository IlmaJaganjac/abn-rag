import React, { useState, useEffect, useRef } from 'react';
import { Sidebar, ViewKey } from './components/Sidebar';
import { ChatView } from './components/ChatView';
import { DocumentsView } from './components/DocumentsView';
import { DatapointsView } from './components/DatapointsView';
import { Tour, TourStep } from './components/Tour';
import { api } from './api/client';
import { IChat, IDoc, ITable, IUpload } from './components/Icons';
import type { Document, SystemStatus } from './types';

type Stage = 'parsing' | 'embedding' | 'done';
export interface Ingest { id: string; name: string; stage: Stage; pct: number; }

const TOUR_STEPS: TourStep[] = [
  {
    selector: '',
    placement: 'center',
    title: 'Welcome to Annualyzer',
    body: 'A 60-second tour of how to ask questions, ground answers in source PDFs, and add new annual reports. You can skip anytime — we won’t show this again.',
  },
  {
    selector: '.brand',
    view: 'chat',
    placement: 'right',
    title: 'Your workspace',
    body: 'Annualyzer is a private RAG over annual reports. Everything stays on your indexed corpus — answers cite the exact page they came from.',
  },
  {
    selector: '.nav',
    view: 'chat',
    placement: 'right',
    title: 'Three views',
    body: 'Ask questions, manage indexed documents, or browse pre-extracted datapoints (FTE, ESG targets, financials) for instant lookups without an LLM call.',
  },
  {
    selector: '.composer',
    view: 'chat',
    placement: 'top',
    title: 'Ask anything',
    body: 'Type a question and we retrieve the most relevant chunks, then generate a cited answer.',
  },
  {
    selector: '.dropzone',
    view: 'docs',
    placement: 'bottom',
    title: 'Add reports',
    body: 'Drop a PDF here. We parse it with Docling, chunk by section, embed into pgvector, and extract key datapoints automatically. A friendly toast lets you know it can take a minute.',
  },
  {
    selector: '.help-btn',
    view: 'chat',
    placement: 'bottom',
    title: 'Need this again?',
    body: 'The Help button is always there. Click it to replay this tour anytime.',
  },
];

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "primaryHue": 155,
  "primaryChroma": 0.06,
  "density": "comfortable",
  "fontPair": "newsreader-inter",
  "showVerbatim": true
}/*EDITMODE-END*/;

type Tweaks = typeof TWEAK_DEFAULTS;

const FONT_PAIRS: Record<string, { editorial: string; ui: string; gfont: string }> = {
  'newsreader-inter': { editorial: "'Newsreader', Georgia, serif", ui: "'Inter Tight', sans-serif", gfont: 'Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Inter+Tight:wght@400;500;600' },
  'fraunces-geist': { editorial: "'Fraunces', Georgia, serif", ui: "'Geist', sans-serif", gfont: 'Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&family=Geist:wght@400;500;600' },
  'instrument-ibm': { editorial: "'Instrument Serif', Georgia, serif", ui: "'IBM Plex Sans', sans-serif", gfont: 'Instrument+Serif:ital@0;1&family=IBM+Plex+Sans:wght@400;500;600' },
};

function applyTweaks(t: Tweaks) {
  const root = document.documentElement;
  root.style.setProperty('--primary', `oklch(35% ${t.primaryChroma} ${t.primaryHue})`);
  root.style.setProperty('--primary-hover', `oklch(30% ${t.primaryChroma + 0.01} ${t.primaryHue})`);
  root.style.setProperty('--primary-tint', `oklch(94% ${Math.min(t.primaryChroma * 0.4, 0.04)} ${t.primaryHue})`);
  root.style.setProperty('--accent', `oklch(85% 0.15 ${(t.primaryHue + 300) % 360})`);

  const pair = FONT_PAIRS[t.fontPair] ?? FONT_PAIRS['newsreader-inter'];
  document.body.style.fontFamily = pair.ui;
  document.querySelectorAll<HTMLElement>('.editorial, .answer-text, .verbatim-text, .doc-stat-value, .dp-row .value, .chat-empty h2, .docs-hd h2, .dp-hd h2').forEach(el => {
    el.style.fontFamily = pair.editorial;
  });

  const dens = t.density === 'compact' ? { fs: '13px' } : t.density === 'spacious' ? { fs: '15px' } : { fs: '14px' };
  document.body.style.fontSize = dens.fs;

  // Hide / show verbatim blocks
  document.querySelectorAll<HTMLElement>('.verbatim-block').forEach(el => {
    el.style.display = t.showVerbatim ? '' : 'none';
  });
}

function ensureFontLink(fontPair: string) {
  const id = 'gfont-' + fontPair;
  document.querySelectorAll('link[data-tweak-font]').forEach(n => n.remove());
  const link = document.createElement('link');
  link.rel = 'stylesheet';
  link.href = `https://fonts.googleapis.com/css2?family=${FONT_PAIRS[fontPair].gfont}&family=JetBrains+Mono:wght@400;500&display=swap`;
  link.id = id;
  link.setAttribute('data-tweak-font', '1');
  document.head.appendChild(link);
}

function PreparingOverlay({ status }: { status: SystemStatus | null }) {
  const [iconIndex, setIconIndex] = useState(0);
  const icons = [
    <IDoc size={28} key="doc" />,
    <ITable size={28} key="table" />,
    <IChat size={28} key="chat" />,
    <IUpload size={28} key="upload" />,
  ];
  const failed = status?.reranker.status === 'error';

  useEffect(() => {
    const id = setInterval(() => setIconIndex(i => (i + 1) % icons.length), 850);
    return () => clearInterval(id);
  }, [icons.length]);

  return (
    <div className="system-prep-root" role="dialog" aria-modal="true" aria-label="System preparation">
      <div className="system-prep-card">
        <div className="system-prep-icon" aria-hidden="true">
          {icons[iconIndex]}
        </div>
        <div className="system-prep-step">Startup</div>
        <div className="system-prep-title">
          {failed ? 'Voorbereiden mislukt' : 'Systeem voorbereiden'}
        </div>
        <div className="system-prep-body">
          {failed
            ? 'De reranker kon niet worden geladen. Controleer de backend logs en Hugging Face toegang.'
            : 'De reranker wordt bij deze eerste start gedownload en geladen. Annualyzer is zo klaar om geciteerde antwoorden te geven.'}
        </div>
        <div className="system-prep-meta">
          <span>{status?.reranker.model ?? 'BAAI/bge-reranker-v2-m3'}</span>
          <span>{status?.reranker.status ?? 'connecting'}</span>
        </div>
        {!failed && (
          <div className="system-prep-loader" aria-hidden="true">
            <span />
          </div>
        )}
        {failed && status?.reranker.error && (
          <div className="system-prep-error">{status.reranker.error}</div>
        )}
      </div>
    </div>
  );
}

export function App() {
  const [view, setView] = useState<ViewKey>('chat');
  const [tweaks, setTweaks] = useState<Tweaks>(TWEAK_DEFAULTS);
  const [tweaksOpen, setTweaksOpen] = useState(false);
  const [tourOpen, setTourOpen] = useState(false);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [firstReadyReport, setFirstReadyReport] = useState<Document | null>(null);
  const [chatClearKey, setChatClearKey] = useState(0);
  const [docCount, setDocCount] = useState(0);
  const [datapointCount, setDatapointCount] = useState(0);
  const [dataRefreshKey, setDataRefreshKey] = useState(0);
  const [ingest, setIngest] = useState<Ingest[]>([]);
  const refreshDocsRef = useRef<(() => void) | null>(null);

  const refreshCounts = async () => {
    const [docs, dps] = await Promise.all([
      api.listDocuments(),
      api.listDatapoints(),
    ]);
    setDocCount(docs.length);
    setDatapointCount(dps.length);
  };

  const handleDataChanged = () => {
    setDataRefreshKey(k => k + 1);
    refreshCounts().catch(() => {});
  };

  const pollStatus = (jobId: string, docId: string, onReady?: (doc: Document | null) => void) => {
    const tick = setInterval(async () => {
      try {
        const s = await api.getDocumentStatus(docId);
        const stage: Stage =
          s.status === 'parsing' || s.status === 'queued' ? 'parsing'
          : s.status === 'ready' ? 'done'
          : 'embedding';
        const pct = stage === 'parsing' ? 25 : stage === 'embedding' ? 70 : 100;
        setIngest(arr =>
          arr.map(j => (j.id === jobId ? { ...j, stage, pct } : j)),
        );
        if (s.status === 'ready' || s.status === 'error') {
          clearInterval(tick);
          if (s.status === 'ready') {
            setFirstReadyReport(prev => prev ?? s.document);
            onReady?.(s.document);
            handleDataChanged();
          }
          refreshDocsRef.current?.();
          setTimeout(() => {
            setIngest(arr => arr.filter(j => j.id !== jobId));
          }, 1200);
        }
      } catch {}
    }, 2000);
  };

  useEffect(() => {
    let cancelled = false;
    let intervalId: number | null = null;
    const refresh = async () => {
      try {
        const [status, docs, dps] = await Promise.all([
          api.getSystemStatus(),
          api.listDocuments(),
          api.listDatapoints(),
        ]);
        if (!cancelled) {
          setSystemStatus(status);
          setDocCount(docs.length);
          setDatapointCount(dps.length);
          if (status.status === 'ready' && intervalId !== null) {
            clearInterval(intervalId);
            intervalId = null;
          }
        }
      } catch {
        if (!cancelled) setSystemStatus(null);
      }
    };
    refresh();
    intervalId = window.setInterval(refresh, 1500);
    return () => {
      cancelled = true;
      if (intervalId !== null) clearInterval(intervalId);
    };
  }, []);

  // Open the tour on every load so the instructions are immediately visible.
  useEffect(() => {
    const id = setTimeout(() => setTourOpen(true), 400);
    return () => clearTimeout(id);
  }, []);

  const closeTour = () => {
    setTourOpen(false);
  };
  const openTour = () => setTourOpen(true);

  const handleClearChat = () => {
    setChatClearKey(k => k + 1);
    setView('chat');
  };

  // Tweaks protocol
  useEffect(() => {
    const onMsg = (e: MessageEvent) => {
      if (e.data?.type === '__activate_edit_mode') setTweaksOpen(true);
      if (e.data?.type === '__deactivate_edit_mode') setTweaksOpen(false);
    };
    window.addEventListener('message', onMsg);
    window.parent.postMessage({ type: '__edit_mode_available' }, '*');
    return () => window.removeEventListener('message', onMsg);
  }, []);

  useEffect(() => {
    ensureFontLink(tweaks.fontPair);
    // tiny delay so fonts apply before we re-target
    const id = setTimeout(() => applyTweaks(tweaks), 50);
    applyTweaks(tweaks);
    return () => clearTimeout(id);
  }, [tweaks, view]);

  const setTweak = <K extends keyof Tweaks>(k: K, v: Tweaks[K]) => {
    setTweaks(prev => {
      const next = { ...prev, [k]: v };
      window.parent.postMessage({ type: '__edit_mode_set_keys', edits: { [k]: v } }, '*');
      return next;
    });
  };

  const titles: Record<ViewKey, { h: string; sub: string }> = {
    chat: { h: 'Ask', sub: 'Verbatim Q&A grounded in indexed PDFs' },
    docs: { h: 'Documents', sub: 'Annual reports indexed for retrieval' },
    datapoints: { h: 'Datapoints', sub: 'Pre-extracted facts for fast answers' },
  };

  return (
    <div className="app">
      <Sidebar
        view={view}
        onChange={setView}
        docCount={docCount}
        datapointCount={datapointCount}
      />
      <main className="main">
        <header className="topbar">
          <div>
            <h1>{titles[view].h}</h1>
            <div className="crumb">{titles[view].sub}</div>
          </div>
          <div className="topbar-spacer" />
          <button className="btn help-btn" onClick={openTour} aria-label="Help / take a tour" title="Take the tour">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10" />
              <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
              <line x1="12" y1="17" x2="12.01" y2="17" />
            </svg>
            Help
          </button>
          {view === 'chat' && (
            <button className="btn" onClick={handleClearChat}>Clear chat</button>
          )}
        </header>

        {view === 'chat' && <ChatView clearKey={chatClearKey} firstReadyReport={firstReadyReport} />}
        {view === 'docs' && (
          <DocumentsView
            ingest={ingest}
            setIngest={setIngest}
            pollStatus={pollStatus}
            refreshDocsRef={refreshDocsRef}
            onDataChanged={handleDataChanged}
          />
        )}
        {view === 'datapoints' && <DatapointsView refreshKey={dataRefreshKey} />}
      </main>

      {tourOpen && (
        <Tour steps={TOUR_STEPS} onClose={closeTour} onViewChange={setView} />
      )}

      {systemStatus?.status !== 'ready' && (
        <PreparingOverlay status={systemStatus} />
      )}

      {tweaksOpen && (
        <div style={{
          position: 'fixed', right: 20, bottom: 20, width: 280,
          background: 'var(--paper)', border: '1px solid var(--line-2)', borderRadius: 12,
          boxShadow: '0 12px 40px -12px oklch(0% 0 0 / 0.18)',
          padding: 16, zIndex: 50,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: 12 }}>
            <div style={{ fontWeight: 600, fontSize: 13 }}>Tweaks</div>
            <button
              style={{ marginLeft: 'auto', color: 'var(--ink-3)', fontSize: 18, lineHeight: 1 }}
              onClick={() => {
                setTweaksOpen(false);
                window.parent.postMessage({ type: '__edit_mode_dismissed' }, '*');
              }}
            >×</button>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div>
              <label style={{ fontSize: 11, color: 'var(--ink-3)', textTransform: 'uppercase', letterSpacing: 0.6 }}>
                Primary hue · {tweaks.primaryHue}°
              </label>
              <input type="range" min={0} max={360} value={tweaks.primaryHue}
                onChange={e => setTweak('primaryHue', Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
            <div>
              <label style={{ fontSize: 11, color: 'var(--ink-3)', textTransform: 'uppercase', letterSpacing: 0.6 }}>
                Saturation · {tweaks.primaryChroma.toFixed(2)}
              </label>
              <input type="range" min={0} max={0.18} step={0.01} value={tweaks.primaryChroma}
                onChange={e => setTweak('primaryChroma', Number(e.target.value))}
                style={{ width: '100%' }}
              />
            </div>
            <div>
              <label style={{ fontSize: 11, color: 'var(--ink-3)', textTransform: 'uppercase', letterSpacing: 0.6, display: 'block', marginBottom: 4 }}>
                Font pair
              </label>
              <select value={tweaks.fontPair} onChange={e => setTweak('fontPair', e.target.value as Tweaks['fontPair'])}
                style={{ width: '100%', padding: '6px 8px', border: '1px solid var(--line)', borderRadius: 6, background: 'var(--bg-2)' }}>
                <option value="newsreader-inter">Newsreader · Inter Tight</option>
                <option value="fraunces-geist">Fraunces · Geist</option>
                <option value="instrument-ibm">Instrument Serif · IBM Plex</option>
              </select>
            </div>
            <div>
              <label style={{ fontSize: 11, color: 'var(--ink-3)', textTransform: 'uppercase', letterSpacing: 0.6, display: 'block', marginBottom: 4 }}>
                Density
              </label>
              <div style={{ display: 'flex', gap: 4 }}>
                {(['compact','comfortable','spacious'] as const).map(d => (
                  <button key={d}
                    className={`btn ${tweaks.density === d ? 'btn-primary' : ''}`}
                    style={{ flex: 1, padding: '5px 6px', fontSize: 11, textTransform: 'capitalize' }}
                    onClick={() => setTweak('density', d)}
                  >{d}</button>
                ))}
              </div>
            </div>
            <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontSize: 13 }}>
              <input type="checkbox" checked={tweaks.showVerbatim}
                onChange={e => setTweak('showVerbatim', e.target.checked)} />
              Show verbatim quote block
            </label>
          </div>
        </div>
      )}
    </div>
  );
}
