/**
 * Types mirroring backend/app/schemas.py (Pydantic v2).
 * Keep in sync with the Python source of truth.
 */

export type Category =
  | 'pre_extracted_fact'
  | 'verbatim_financial'
  | 'operational_datapoint'
  | 'esg_datapoint'
  | 'named_program'
  | 'sustainability_target'
  | 'multi_document'
  | 'hallucination_check';

export type Difficulty = 'easy' | 'medium' | 'hard';

export interface Citation {
  source: string;
  page: number;
  quote: string;
}

export interface Chunk {
  id: string;
  source: string;
  company: string | null;
  year: number | null;
  page: number;
  text: string;
  token_count: number;
}

export interface RetrievedChunk extends Chunk {
  score: number;
}

export interface VerbatimAnswer {
  question: string;
  answer: string;
  verbatim: string | null;
  citations: Citation[];
  refused: boolean;
  refusal_reason: string | null;
}

export interface RetrievalQuery {
  question: string;
  company?: string | null;
  year?: number | null;
}

export interface RetrievalResult {
  query: RetrievalQuery;
  chunks: RetrievedChunk[];
}

/* --- UI-side types (frontend only) --- */

export type DocStatus = 'queued' | 'parsing' | 'embedding' | 'ready' | 'error';

export interface Document {
  id: string;
  source: string;
  company: string | null;
  year: number | null;
  title: string;
  pages: number;
  size_mb: number;
  ingested_at: string; // ISO
  chunks: number;
  parser: 'docling' | 'llamaparse' | 'pymupdf';
  status: DocStatus;
  /* Pre-extracted summary (cheap to show on a card) */
  fte: string | null;
  net_zero_year: string | null;
}

export type DatapointType = string;

export interface Datapoint {
  company: string;
  source: string;
  page: number;
  type: DatapointType;
  metric: string;
  value: string;
  period: string | number;
  section: string;
  verbatim: string;
}

export interface SystemStatus {
  status: 'preparing' | 'ready';
  reranker: {
    status: 'idle' | 'loading' | 'ready' | 'error';
    model: string | null;
    error: string | null;
  };
}

export type ThinkingPhase =
  | 'queued'
  | 'embedding'
  | 'searching'
  | 'reading'
  | 'drafting'
  | 'citing'
  | 'done';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  /* assistant-only */
  answer?: VerbatimAnswer;
  phase?: ThinkingPhase;
  ts: number;
}
