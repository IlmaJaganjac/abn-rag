import React, { useEffect, useState, useRef, useReducer } from 'react';

export interface TourStep {
  selector: string;     // CSS selector for the element to highlight
  view?: 'chat' | 'docs' | 'datapoints'; // switch to this view before showing
  title: string;
  body: string;
  placement?: 'right' | 'bottom' | 'left' | 'top' | 'center';
}

interface Props {
  steps: TourStep[];
  onClose: () => void;
  onViewChange: (v: 'chat' | 'docs' | 'datapoints') => void;
}

const PAD = 8;
const GAP = 14;
const CARD_W = 340;

export function Tour({ steps, onClose, onViewChange }: Props) {
  const [i, setI] = useState(0);
  const [rect, setRect] = useState<DOMRect | null>(null);
  const [, forceUpdate] = useReducer((n: number) => n + 1, 0);
  const cardRef = useRef<HTMLDivElement>(null);

  const step = steps[i];

  // Switch view if step needs it
  useEffect(() => {
    if (step?.view) onViewChange(step.view);
  }, [i]);

  // Measure target rect — retries briefly because view changes might not be
  // painted yet on the very first frame.
  useEffect(() => {
    if (!step) return;
    let cancelled = false;
    let tries = 0;
    const measure = () => {
      if (cancelled) return;
      if (step.placement === 'center' || !step.selector) {
        setRect(null);
        return;
      }
      const el = document.querySelector(step.selector) as HTMLElement | null;
      if (el) {
        const r = el.getBoundingClientRect();
        setRect(r);
        // Make sure it's visible
        if (r.top < 0 || r.bottom > window.innerHeight) {
          el.scrollIntoView({ block: 'center', behavior: 'smooth' });
          // re-measure after scroll settles
          setTimeout(() => {
            if (!cancelled) setRect(el.getBoundingClientRect());
          }, 320);
        }
      } else if (tries++ < 20) {
        setTimeout(measure, 60);
      } else {
        setRect(null);
      }
    };
    measure();

    const onResize = () => {
      const el = step.selector ? (document.querySelector(step.selector) as HTMLElement | null) : null;
      if (el) setRect(el.getBoundingClientRect());
    };
    window.addEventListener('resize', onResize);
    window.addEventListener('scroll', onResize, true);
    return () => {
      cancelled = true;
      window.removeEventListener('resize', onResize);
      window.removeEventListener('scroll', onResize, true);
    };
  }, [i, step]);

  // After step changes, force one re-layout so we use the card's real height
  // for clamping (cardRef is null on the first render). Guard so we don't loop.
  const lastMeasuredKey = useRef<string>('');
  useEffect(() => {
    const key = `${i}:${rect?.x ?? 'x'}:${rect?.y ?? 'y'}:${rect?.width ?? 'w'}:${rect?.height ?? 'h'}`;
    if (cardRef.current && lastMeasuredKey.current !== key) {
      lastMeasuredKey.current = key;
      forceUpdate();
    }
  });

  // ESC closes
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
      if (e.key === 'ArrowRight' || e.key === 'Enter') next();
      if (e.key === 'ArrowLeft') prev();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [i]);

  const next = () => {
    if (i < steps.length - 1) setI(i + 1);
    else onClose();
  };
  const prev = () => { if (i > 0) setI(i - 1); };

  // Compute spotlight + card position
  const spot = rect
    ? { x: rect.left - PAD, y: rect.top - PAD, w: rect.width + PAD * 2, h: rect.height + PAD * 2 }
    : null;

  let cardStyle: React.CSSProperties = {};
  if (!spot || step.placement === 'center') {
    cardStyle = {
      left: '50%', top: '50%',
      transform: 'translate(-50%, -50%)',
      width: CARD_W,
    };
  } else {
    const place = step.placement || 'right';
    const vw = window.innerWidth, vh = window.innerHeight;
    // Estimate card height (roughly: title + body + actions + dots) — clamp uses this.
    // Actual height varies, so we use a conservative estimate. The card auto-grows
    // and we clamp top so the bottom stays inside the viewport.
    const cardH = (cardRef.current?.offsetHeight ?? 220);
    let left = 0, top = 0;
    if (place === 'right') {
      left = spot.x + spot.w + GAP;
      top = spot.y + spot.h / 2 - cardH / 2;  // already accounts for centering
    } else if (place === 'left') {
      left = spot.x - GAP - CARD_W;
      top = spot.y + spot.h / 2 - cardH / 2;
    } else if (place === 'bottom') {
      left = spot.x + spot.w / 2 - CARD_W / 2;
      top = spot.y + spot.h + GAP;
    } else {
      left = spot.x + spot.w / 2 - CARD_W / 2;
      top = spot.y - GAP - cardH;
    }
    // clamp to viewport
    left = Math.max(16, Math.min(left, vw - CARD_W - 16));
    top = Math.max(16, Math.min(top, vh - cardH - 16));
    cardStyle = { left, top, width: CARD_W };
  }

  if (!step) return null;

  return (
    <div className="tour-root" role="dialog" aria-modal="true" aria-label="Product tour">
      {/* SVG mask for spotlight */}
      <svg className="tour-mask" width="100%" height="100%">
        <defs>
          <mask id="tour-cut">
            <rect width="100%" height="100%" fill="white" />
            {spot && (
              <rect x={spot.x} y={spot.y} width={spot.w} height={spot.h} rx={10} ry={10} fill="black" />
            )}
          </mask>
        </defs>
        <rect width="100%" height="100%" fill="rgba(15, 23, 30, 0.55)" mask="url(#tour-cut)" />
        {spot && (
          <rect
            x={spot.x} y={spot.y} width={spot.w} height={spot.h}
            rx={10} ry={10} fill="none"
            stroke="var(--primary)" strokeWidth={2}
            className="tour-ring"
          />
        )}
      </svg>

      <div className="tour-card" style={cardStyle} ref={cardRef}>
        <div className="tour-step">Step {i + 1} of {steps.length}</div>
        <div className="tour-title">{step.title}</div>
        <div className="tour-body">{step.body}</div>
        <div className="tour-actions">
          <button className="tour-skip" onClick={onClose}>Skip tour</button>
          <div style={{ flex: 1 }} />
          {i > 0 && <button className="btn tour-back" onClick={prev}>Back</button>}
          <button className="btn btn-primary tour-next" onClick={next}>
            {i === steps.length - 1 ? 'Got it' : 'Next'}
          </button>
        </div>
        <div className="tour-progress" aria-hidden="true">
          {steps.map((_, k) => (
            <span key={k} className={`tour-dot ${k === i ? 'on' : ''} ${k < i ? 'done' : ''}`} />
          ))}
        </div>
      </div>
    </div>
  );
}
