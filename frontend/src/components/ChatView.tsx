import React, { useState, useRef, useEffect, useMemo } from 'react';
import type { ChatMessage, ThinkingPhase, VerbatimAnswer, Citation } from '../types';
import { SAMPLE_ANSWERS, SUGGESTED_QUESTIONS, MOCK_DOCS } from '../mock';
import { ISend, IExternal } from './Icons';
import { PdfViewer } from './PdfViewer';

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

function pickMockAnswer(q: string): VerbatimAnswer & { caveat?: string } {
  const lower = q.toLowerCase();
  if (lower.includes('asml') && (lower.includes('employee') || lower.includes('fte'))) return SAMPLE_ANSWERS.asml_fte;
  if (lower.includes('shell') && (lower.includes('climate') || lower.includes('adapt'))) return SAMPLE_ANSWERS.shell_climate;
  if (lower.includes('abn') || lower.includes('workforce')) return SAMPLE_ANSWERS.abn_workforce;
  if (lower.includes('car') || lower.includes('vehicle')) return SAMPLE_ANSWERS.refusal;
  // default: a synthesized answer
  return {
    question: q,
    answer: 'I searched across 5 indexed reports but couldn\u2019t find a verbatim figure that directly answers this. The closest matches discuss adjacent topics. Try narrowing to a specific company or year, or rephrase using language likely to appear in the report.',
    verbatim: null,
    citations: [
      { source: 'asml.pdf', page: 178, quote: 'We aim to be greenhouse gas neutral across our value chain by 2040.' },
    ],
    refused: false, refusal_reason: null, raw_citations: [], grounding_drops: [],
  };
}

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
          p === 'searching' && state !== 'pending' ? `${retrievedCount} chunks · k=8` :
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

function AnswerBlock({ ans, caveat, onCite }: { ans: VerbatimAnswer; caveat?: string; onCite: (c: Citation) => void }) {
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
            <div key={i} className="cite-card" onClick={() => onCite(c)} role="button" tabIndex={0}
              onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onCite(c); } }}>
              <div className="cite-num">[{i + 1}]</div>
              <div className="cite-body">
                <div className="cite-source">
                  <span>{c.source}</span>
                  <span className="cite-page-tag">p.{c.page}</span>
                </div>
                <div className="cite-quote">&ldquo;{c.quote}&rdquo;</div>
              </div>
              <div className="cite-action"><IExternal /></div>
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
  const [viewer, setViewer] = useState<{ source: string; page: number; quote: string } | null>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  const companies = useMemo(() => {
    const s = new Set(MOCK_DOCS.map(d => d.company).filter((c): c is string => !!c));
    return Array.from(s).sort();
  }, []);

  useEffect(() => { endRef.current?.scrollIntoView({ block: 'end' }); }, [messages, busy, phase]);

  const send = async (q: string) => {
    if (!q.trim() || busy) return;
    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: 'user', content: q, ts: Date.now() };
    setMessages(m => [...m, userMsg]);
    setInput('');
    setBusy(true);

    // Simulated phase progression. Real call would use api.ask() with onPhase.
    const phaseTimings: [ThinkingPhase, string, number][] = [
      ['embedding', '', 350],
      ['searching', 'k=8 · cosine', 600],
      ['reading', '', 700],
      ['drafting', '', 1100],
      ['citing', '', 500],
    ];
    for (const [p, d, ms] of phaseTimings) {
      setPhase(p); setPhaseDetail(d);
      await new Promise(r => setTimeout(r, ms));
    }

    const ans = pickMockAnswer(q);
    const aMsg: ChatMessage = {
      id: `a-${Date.now()}`,
      role: 'assistant',
      content: ans.answer,
      answer: ans,
      caveat: (ans as VerbatimAnswer & { caveat?: string }).caveat,
      ts: Date.now(),
    };
    setMessages(m => [...m, aMsg]);
    setBusy(false); setPhase('queued');
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
                  m.answer && <AnswerBlock ans={m.answer} caveat={m.caveat} onCite={c => setViewer({ source: c.source, page: c.page, quote: c.quote })} />
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
                  <option value="all">All documents</option>
                  {companies.map(c => <option key={c} value={c}>{c}</option>)}
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
      <PdfViewer
        open={!!viewer}
        source={viewer?.source ?? null}
        page={viewer?.page ?? null}
        quote={viewer?.quote ?? null}
        onClose={() => setViewer(null)}
      />
    </>
  );
}
