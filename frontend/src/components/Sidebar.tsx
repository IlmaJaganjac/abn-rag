import { IChat, IDoc, ITable } from './Icons';

export type ViewKey = 'chat' | 'docs' | 'datapoints';

interface Props {
  view: ViewKey;
  onChange: (v: ViewKey) => void;
  docCount: number;
  datapointCount: number;
}

export function Sidebar({ view, onChange, docCount, datapointCount }: Props) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brand-mark" aria-label="Annual reports">
          <IDoc />
        </div>
        <div>
          <div className="brand-name">Annualyzer <span className="brand-tag">v1</span></div>
          <div className="brand-sub">Cited answers from filings</div>
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
      <div className="sidebar-footer">Annualyzer 2026 V1</div>
    </aside>
  );
}
