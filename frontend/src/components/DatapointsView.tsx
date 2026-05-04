import React, { useState, useMemo } from 'react';
import { MOCK_DATAPOINTS } from '../mock';
import type { DatapointType } from '../types';
import { IExternal } from './Icons';
import { PdfViewer } from './PdfViewer';

const TYPES: { key: DatapointType | 'all'; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'fte', label: 'FTE / Workforce' },
  { key: 'esg', label: 'ESG / Climate' },
  { key: 'financial', label: 'Financial' },
];

export function DatapointsView() {
  const [type, setType] = useState<DatapointType | 'all'>('all');
  const [company, setCompany] = useState<string>('all');
  const [viewer, setViewer] = useState<{ source: string; page: number; quote: string } | null>(null);

  const companies = useMemo(() => {
    return Array.from(new Set(MOCK_DATAPOINTS.map(d => d.company))).sort();
  }, []);

  const rows = useMemo(() => {
    return MOCK_DATAPOINTS.filter(d => {
      if (type !== 'all' && d.type !== type) return false;
      if (company !== 'all' && d.company !== company) return false;
      return true;
    });
  }, [type, company]);

  return (
    <div className="view">
      <div className="dp-wrap">
        <div className="dp-hd">
          <h2>Datapoints</h2>
          <div style={{ color: 'var(--ink-3)', fontSize: 13 }}>
            {rows.length} extracted · all grounded with verbatim source quotes
          </div>
        </div>

        <div className="dp-filters">
          {TYPES.map(t => (
            <button
              key={t.key}
              className={`filter-chip ${type === t.key ? 'active' : ''}`}
              onClick={() => setType(t.key)}
            >{t.label}</button>
          ))}
          <span style={{ width: 1, background: 'var(--line)', margin: '0 4px' }} />
          <button
            className={`filter-chip ${company === 'all' ? 'active' : ''}`}
            onClick={() => setCompany('all')}
          >All companies</button>
          {companies.map(c => (
            <button
              key={c}
              className={`filter-chip ${company === c ? 'active' : ''}`}
              onClick={() => setCompany(c)}
            >{c}</button>
          ))}
        </div>

        <div className="dp-table">
          <div className="dp-row header">
            <div>Company</div>
            <div>Metric</div>
            <div>Value</div>
            <div>Period</div>
            <div>Source</div>
          </div>
          {rows.map((d, i) => (
            <div key={i} className="dp-row" onClick={() => setViewer({ source: d.source, page: d.page, quote: d.verbatim })}
              role="button" tabIndex={0}
              style={{ cursor: 'pointer' }}
              onKeyDown={e => { if (e.key === 'Enter') setViewer({ source: d.source, page: d.page, quote: d.verbatim }); }}>
              <div>
                <div className="company">{d.company}</div>
                <span className="type-tag">{d.type}</span>
              </div>
              <div className="metric">
                {d.metric}
                <div className="verbatim">&ldquo;{d.verbatim}&rdquo;</div>
              </div>
              <div className="value">{d.value}</div>
              <div className="page-ref">{d.period}</div>
              <div className="source-ref">
                {d.source}
                <div>p.{d.page} · {d.section} <IExternal size={11} /></div>
              </div>
            </div>
          ))}
          {rows.length === 0 && (
            <div className="dp-row" style={{ gridTemplateColumns: '1fr', color: 'var(--ink-3)', justifyContent: 'center' }}>
              No datapoints match this filter.
            </div>
          )}
        </div>
      </div>
      <PdfViewer
        open={!!viewer}
        source={viewer?.source ?? null}
        page={viewer?.page ?? null}
        quote={viewer?.quote ?? null}
        onClose={() => setViewer(null)}
      />
    </div>
  );
}
