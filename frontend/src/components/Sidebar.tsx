import React, { useState } from 'react';
import { IChat, IDoc, ITable } from './Icons';

export type ViewKey = 'chat' | 'docs' | 'datapoints';

interface Props {
  view: ViewKey;
  onChange: (v: ViewKey) => void;
  docCount: number;
  datapointCount: number;
}

interface Pull {
  company: string;
  year: number;
  page: number;
  quote: string;
  topic: string;
}

const PULLS: Pull[] = [
  { company: 'ABN AMRO', year: 2025, page: 174, topic: 'Workforce',
    quote: 'A net workforce reduction of around 5,200 FTE by year-end 2028.' },
  { company: 'ASML', year: 2025, page: 145, topic: 'Climate',
    quote: 'Net Scope 1 & 2 GHG-neutral by 2025 — achieved.' },
  { company: 'Shell', year: 2025, page: 122, topic: 'People',
    quote: 'Approximately 96,000 employees at the end of 2025.' },
  { company: 'Heineken', year: 2024, page: 88, topic: 'Headcount',
    quote: '85,870 full-time equivalents across more than 70 markets.' },
  { company: 'ABN AMRO', year: 2025, page: 25, topic: 'Sustainability',
    quote: 'Our climate strategy targets a net-zero economy by 2050.' },
];

export function Sidebar({ view, onChange, docCount, datapointCount }: Props) {
  const [idx, setIdx] = useState(() => Math.floor(Math.random() * PULLS.length));
  const pull = PULLS[idx];
  const next = () => setIdx((idx + 1) % PULLS.length);
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark" aria-label="ABN AMRO">
          <img src="data:image/jpeg;base64,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" alt="ABN AMRO" />
        </div>
        <div>
          <div className="brand-name">Annualyzer <span className="brand-tag">v1</span></div>
          <div className="brand-sub">ABN AMRO · Cited answers from filings</div>
        </div>
      </div>

      <nav className="nav">
        <div className="nav-section">Workspace</div>
        <button className={`nav-item ${view === 'chat' ? 'active' : ''}`} onClick={() => onChange('chat')}>
          <IChat /> Ask
        </button>
        <button className={`nav-item ${view === 'docs' ? 'active' : ''}`} onClick={() => onChange('docs')}>
          <IDoc /> Documents <span className="badge">{docCount}</span>
        </button>
        <button className={`nav-item ${view === 'datapoints' ? 'active' : ''}`} onClick={() => onChange('datapoints')}>
          <ITable /> Datapoints <span className="badge">{datapointCount}</span>
        </button>
      </nav>

      <div className="sidebar-spacer" />

      <div className="pull-card" onClick={() => { onChange('chat'); next(); }} role="button" tabIndex={0}>
        <div className="pull-eyebrow">
          <span className="pull-kicker">From the index</span>
          <span className="pull-topic">{pull.topic}</span>
        </div>
        <div className="pull-quote">
          <span className="pull-mark" aria-hidden="true">&ldquo;</span>
          {pull.quote}
        </div>
        <div className="pull-meta">
          <span className="pull-source">{pull.company} · {pull.year}</span>
          <span className="pull-page">p.{pull.page}</span>
        </div>
        <button
          className="pull-shuffle"
          aria-label="Show another quote"
          onClick={(e) => { e.stopPropagation(); next(); }}
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="16 3 21 3 21 8" />
            <line x1="4" y1="20" x2="21" y2="3" />
            <polyline points="21 16 21 21 16 21" />
            <line x1="15" y1="15" x2="21" y2="21" />
            <line x1="4" y1="4" x2="9" y2="9" />
          </svg>
        </button>
      </div>
    </aside>
  );
}
