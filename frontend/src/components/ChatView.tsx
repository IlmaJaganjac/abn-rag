import React, { useState, useRef, useEffect, useMemo } from 'react';
import type { ChatMessage, ThinkingPhase, VerbatimAnswer, Document } from '../types';
import { api } from '../api/client';
import { ISend } from './Icons';

const SUGGESTED_QUESTIONS = [
  'How many total employees did ASML have at the end of 2025?',
  'What were ASML’s total net sales in 2025?',
  'What was ABN AMRO’s CET1 ratio in 2025?',
  'Who was ABN AMRO’s CEO in 2025?',
];

const PHASE_LABELS: Record<ThinkingPhase, string> = {
  queued: 'Queued',
  embedding: 'Embedding question',
  searching: 'Searching index',
  reading: 'Reading source pages',
  drafting: 'Drafting answer',
  citing: 'Verifying verbatim quotes',
  done: 'Complete',
};

const PHASE_ORDER: ThinkingPhase[] = ['embedding', 'searching', 'reading', 'drafting', 'citing'];


function ThinkingPanel({ phase, detail, retrievedCount }: { phase: ThinkingPhase; detail?: string; retrievedCount: number }) {
  const idx = PHASE_ORDER.indexOf(phase);
  return (
    <div className="thinking">
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <div className="thinking-bars"><span /><span /><span /><span /></div>
        <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: 12, color: 'var(--ink-2)' }}>
          thinking · {PHASE_LABELS[phase].toLowerCase()}{detail ? ` · ${detail}` : ''}
        </span>
      </div>
      {PHASE_ORDER.map((p, i) => {
        const state = i < idx ? 'done' : i === idx ? 'active' : 'pending';
        const detailText =
          p === 'embedding' && state !== 'pending' ? 'text-embedding-3-small · 1536d' :
          p === 'searching' && state !== 'pending' ? `${retrievedCount} chunks · k=12` :
          p === 'reading' && state !== 'pending' ? `${Math.min(retrievedCount, 8)} pages` :
          p === 'drafting' && state !== 'pending' ? 'gpt-4o-mini' :
          p === 'citing' && state !== 'pending' ? 'verbatim match' : '';
        return (
          <div key={p} className={`thinking-row ${state}`}>
            <span className="pulse" />
            <span>{PHASE_LABELS[p]}</span>
            {detailText && <span className="detail">{detailText}</span>}
          </div>
        );
      })}
    </div>
  );
}

function AnswerBlock({ ans, caveat, sourceLabel }: { ans: VerbatimAnswer; caveat?: string; sourceLabel: (src: string) => string }) {
  if (ans.refused) {
    return (
      <div className="msg-assistant">
        <div className="refusal">
          <div className="refusal-tag">Cannot answer from indexed documents</div>
          <div>{ans.refusal_reason}</div>
        </div>
      </div>
    );
  }
  return (
    <div className="msg-assistant">
      <div className="answer-text">{ans.answer}</div>
      {ans.verbatim && (
        <div className="verbatim-block">
          <div className="verbatim-label">Verbatim from source</div>
          <div className="verbatim-text">&ldquo;{ans.verbatim}&rdquo;</div>
        </div>
      )}
      {caveat && (
        <div className="caveat">
          <span style={{ flexShrink: 0, marginTop: 1 }}>⚠</span>
          <span>{caveat}</span>
        </div>
      )}
      {ans.citations.length > 0 && (
        <div className="citations">
          <div className="citations-label">Sources · {ans.citations.length}</div>
          {ans.citations.map((c, i) => (
            <div key={i} className="cite-card">
              <div className="cite-num">[{i + 1}]</div>
              <div className="cite-body">
                <div className="cite-source">
                  <span>{sourceLabel(c.source)}</span>
                  <span className="cite-page-tag">p.{c.page}</span>
                </div>
                <div className="cite-quote">&ldquo;{c.quote}&rdquo;</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SuggestedQuestions({ questions, onPick }: { questions: string[]; onPick: (q: string) => void }) {
  const PAGE_SIZE = 4;
  const totalPages = Math.max(1, Math.ceil(questions.length / PAGE_SIZE));
  const [page, setPage] = useState(0);
  const start = page * PAGE_SIZE;
  const slice = questions.slice(start, start + PAGE_SIZE);
  // Pad to keep grid height stable across pages
  const pad = PAGE_SIZE - slice.length;

  const next = () => setPage(p => (p + 1) % totalPages);
  const prev = () => setPage(p => (p - 1 + totalPages) % totalPages);

  return (
    <div className="suggested">
      <div className="suggested-head">
        <span className="suggested-label">Try one of these</span>
        {totalPages > 1 && (
          <div className="suggested-pager">
            <button className="pager-btn" onClick={prev} aria-label="Previous suggestions">←</button>
            <span className="pager-count">{page + 1} / {totalPages}</span>
            <button className="pager-btn" onClick={next} aria-label="Next suggestions">→</button>
          </div>
        )}
      </div>
      <div className="suggested-grid">
        {slice.map(q => (
          <button key={q} className="suggested-btn" onClick={() => onPick(q)}>
            {q}<span className="arrow">→</span>
          </button>
        ))}
        {Array.from({ length: pad }).map((_, i) => (
          <div key={`pad-${i}`} className="suggested-btn suggested-btn-empty" aria-hidden="true" />
        ))}
      </div>
    </div>
  );
}

export function ChatView() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [scope, setScope] = useState<string>('all');
  const [busy, setBusy] = useState(false);
  const [phase, setPhase] = useState<ThinkingPhase>('queued');
  const [phaseDetail, setPhaseDetail] = useState<string>('');
  const taRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const [docs, setDocs] = useState<Document[]>([]);

  useEffect(() => {
    (async () => {
      try { setDocs(await api.listDocuments()); } catch {}
    })();
  }, []);

  const reports = useMemo(() =>
    docs
      .filter(d => d.company)
      .map(d => ({
        source: d.source,
        company: d.company!,
        year: d.year,
        label: [d.company, d.year ? String(d.year) : null].filter(Boolean).join(' '),
      }))
      .sort((a, b) => a.label.localeCompare(b.label)),
    [docs]);

  const sourceLabel = useMemo(() => {
    const m = new Map(docs.map(d => [
      d.source,
      [d.company, d.year ? String(d.year) : null].filter(Boolean).join(' ') || d.source,
    ]));
    return (src: string) => m.get(src) ?? src;
  }, [docs]);

  useEffect(() => { endRef.current?.scrollIntoView({ block: 'end' }); }, [messages, busy, phase]);

  const send = async (q: string) => {
    if (!q.trim() || busy) return;
    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: 'user', content: q, ts: Date.now() };
    setMessages(m => [...m, userMsg]);
    setInput('');
    setBusy(true);
    setPhase('queued');
    setPhaseDetail('');

    try {
      const selected = reports.find(r => r.source === scope);
      const ans = await api.ask(q, {
        company: selected?.company ?? null,
        year: selected?.year ?? null,
        onPhase: (p, detail) => {
          setPhase(p);
          setPhaseDetail(detail ?? '');
        },
      });
      const aMsg: ChatMessage = {
        id: `a-${Date.now()}`,
        role: 'assistant',
        content: ans.answer,
        answer: ans,
        ts: Date.now(),
      };
      setMessages(m => [...m, aMsg]);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      const aMsg: ChatMessage = {
        id: `a-${Date.now()}`,
        role: 'assistant',
        content: '',
        answer: {
          question: q,
          answer: '',
          verbatim: null,
          citations: [],
          refused: true,
          refusal_reason: `Request failed: ${msg}`,
          raw_citations: [],
          grounding_drops: [],
        },
        ts: Date.now(),
      };
      setMessages(m => [...m, aMsg]);
    } finally {
      setBusy(false);
      setPhase('queued');
    }
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };

  // autosize
  useEffect(() => {
    const ta = taRef.current; if (!ta) return;
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 200) + 'px';
  }, [input]);

  const lastAssistantQuestion = messages.findLast?.(m => m.role === 'user')?.content;

  return (
    <>
      <div className="view">
        <div className="chat-wrap">
          {messages.length === 0 ? (
            <div className="chat-empty">
              <h2>Ask anything from indexed reports.</h2>
              <p>Every answer is grounded in a verbatim quote with page-level citations. Off-topic questions and hallucinations are refused.</p>
              <SuggestedQuestions questions={SUGGESTED_QUESTIONS} onPick={send} />
            </div>
          ) : (
            messages.map(m => (
              <div key={m.id} className="msg">
                {m.role === 'user' ? (
                  <div className="msg-user">
                    <div className="msg-meta">You asked</div>
                    {m.content}
                  </div>
                ) : (
                  m.answer && <AnswerBlock ans={m.answer} caveat={m.caveat} sourceLabel={sourceLabel} />
                )}
              </div>
            ))
          )}

          {busy && (
            <div className="msg">
              <ThinkingPanel phase={phase} detail={phaseDetail} retrievedCount={12} />
            </div>
          )}

          <div ref={endRef} />
        </div>
      </div>

      <div className="composer-dock">
        <div className="composer-dock-inner">
          <div className="composer">
            <textarea
              ref={taRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={onKey}
              placeholder={lastAssistantQuestion ? 'Ask a follow-up…' : 'Ask about ASML, Shell, ABN AMRO, Heineken…'}
              rows={1}
            />
            <div className="composer-row">
              <span className="scope-pill">
                <span style={{ color: 'var(--ink-3)' }}>Scope</span>
                <select value={scope} onChange={e => setScope(e.target.value)}>
                  <option value="all">All reports</option>
                  {reports.map(r => <option key={r.source} value={r.source}>{r.label}</option>)}
                </select>
              </span>
              <span style={{ fontSize: 11, color: 'var(--ink-3)' }}>
                {busy ? '· streaming' : 'Enter to send · Shift+Enter newline'}
              </span>
              <button
                className="send-btn"
                disabled={!input.trim() || busy}
                onClick={() => send(input)}
                aria-label="Send"
              ><ISend /></button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
