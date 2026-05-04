import React, { useState, useMemo, useEffect } from 'react';
import type { Datapoint, DatapointType } from '../types';
import { api } from '../api/client';

function typeLabel(t: string): string {
  return t.replace(/_/g, ' ').toUpperCase();
}

export function DatapointsView() {
  const [type, setType] = useState<DatapointType | 'all'>('all');
  const [company, setCompany] = useState<string>('all');
  const [allDatapoints, setAllDatapoints] = useState<Datapoint[]>([]);
  const [rows, setRows] = useState<Datapoint[]>([]);

  useEffect(() => {
    (async () => {
      try { setAllDatapoints(await api.listDatapoints()); } catch {}
    })();
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const filter: { company?: string; type?: string } = {};
        if (company !== 'all') filter.company = company;
        if (type !== 'all') filter.type = type;
        setRows(await api.listDatapoints(filter));
      } catch {}
    })();
  }, [type, company]);

  const companies = useMemo(() =>
    Array.from(new Set(allDatapoints.map(d => d.company))).sort(),
    [allDatapoints]);

  const types = useMemo(() =>
    Array.from(new Set(allDatapoints.map(d => d.type))).sort(),
    [allDatapoints]);

  const PAGE_SIZE = 15;
  const [page, setPage] = useState(0);
  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const pageRows = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  useEffect(() => { setPage(0); }, [type, company]);

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
          <button
            className={`filter-chip ${type === 'all' ? 'active' : ''}`}
            onClick={() => setType('all')}
          >All</button>
          {types.map(t => (
            <button
              key={t}
              className={`filter-chip ${type === t ? 'active' : ''}`}
              onClick={() => setType(t)}
            >{typeLabel(t)}</button>
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
            <div>Datapoint</div>
            <div>Value</div>
            <div>Year</div>
            <div>Source</div>
          </div>
          {pageRows.map((d, i) => (
            <div key={i} className="dp-row">
              <div>
                <div className="company">{d.company}</div>
              </div>
              <div className="metric">
                <span className="type-tag" style={{ marginBottom: 4, display: 'inline-block' }}>{typeLabel(d.type)}</span>
                <div>{d.metric}</div>
              </div>
              <div className="value">{d.value}</div>
              <div className="page-ref">{d.period}</div>
              <div className="source-ref">
                {d.source}
                <div>p.{d.page} · {d.section}</div>
              </div>
            </div>
          ))}
          {rows.length === 0 && (
            <div className="dp-row" style={{ gridTemplateColumns: '1fr', color: 'var(--ink-3)', justifyContent: 'center' }}>
              No datapoints match this filter.
            </div>
          )}
        </div>
        {totalPages > 1 && (
          <div className="dp-pager">
            <button className="pager-btn" onClick={() => setPage(p => Math.max(0, p - 1))} disabled={page === 0}>←</button>
            <span className="pager-count">{page + 1} / {totalPages}</span>
            <button className="pager-btn" onClick={() => setPage(p => Math.min(totalPages - 1, p + 1))} disabled={page === totalPages - 1}>→</button>
          </div>
        )}
      </div>
    </div>
  );
}
