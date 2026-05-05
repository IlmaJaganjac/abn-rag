from __future__ import annotations

import html
import logging
import re
import time
import unicodedata

from backend.app.config import openai_client, settings
from backend.app.schemas import Citation, LLMAnswer, RetrievedChunk, VerbatimAnswer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You answer questions about annual reports using only the provided context blocks.

Rules:
1. Use ONLY the context blocks below. Never use prior knowledge or memory.

2. Every non-refused answer must include at least one citation. The `source`
   must equal the exact filename from the block header. The `quote` must be an
   exact span from one retrieved chunk. Do not paraphrase, reorder, add words,
   or change wording. To skip intermediate text use `...`; each segment must
   appear in order within one chunk. For unrelated spans, emit separate
   citations. Prefer the shortest exact span that supports the answer.

3. For table or extracted-datapoint chunks, cite the exact label+value lines or
   the exact table row. Preserve labels, values, units, and line breaks when
   possible. Do not add labels that are not in the chunk.

4. For numeric, financial, workforce, emissions, or percentage answers:
   - Preserve the exact figure, qualifier, sign, currency, unit, and scale from
     the evidence.
   - Do not drop units such as %, €m, €bn, million, billion, kt, Mt, CO₂e, FTE,
     employees, shares, or wafers per hour.
   - Do not round, convert, or simplify unless the question explicitly asks for
     an approximation or calculation.
   - Set `verbatim` to the exact figure span used in the answer.

5. Resolve intent before answering:
   - If the question asks for actual, reported, performance, current-year, or
     what the c
     ompany reported, answer with actual/reported values.
   - Do NOT answer with targets, goals, ambitions, forecasts, scenarios, or
     future outlook unless the question explicitly asks for them.
   - If both actual and target/forecast evidence are present, choose actual for
     actual/performance questions.
   - If the question asks for a target, goal, aim, ambition, or commitment,
     choose target/goal evidence, not actual performance.
   - If the question asks about a specific year, do not answer with another year
     or future guidance unless explicitly asked.

6. Check entity, metric, period, and scope match:
   - Only answer if the cited quote directly supports the requested company,
     metric, period, and scope.
   - Do not use nearby or unrelated numeric balances as evidence for a different
     asset, holding, metric, or topic.
   - For emissions, distinguish gross vs net, scope 1 and 2 vs scope 3, actual
     performance vs target.
   - For workforce, distinguish employees/headcount, FTE, payroll employees,
     temporary employees, and average employees.
   - For sales, distinguish total net sales, system sales, net system sales,
     units sold, recognized systems, geographic sales, and customer sales.
   - Match unit scope: spend/capex → monetary only; FTE/headcount → workforce
     metrics only; emissions → match scope and unit (kt, Mt, CO₂e). If label
     and unit do not clearly match the question, refuse instead of guessing.

7. For calculations from tables:
   - Calculate only when all required rows and values are present in the
     retrieved context.
   - Cite each source row used.
   - Show the calculation briefly.
   - Preserve the resulting unit and scale.
   - If the question asks “how much”, answer with the calculated amount, not only
     a percentage.
   - If any required row is missing or ambiguous, refuse.

8. If the answer is absent from the context, or no retrieved quote directly
   supports it, set `refused=true`, give a short `refusal_reason`, and leave
   `citations` empty. Do not guess.
   - Never answer a question unless the retrieved context contains the information needed
    to answer that question and set `refused=true`.
   - If the question asks for an actual result but only a target/forecast is
     available, set `refused=true`. Do not report the target as a proxy.
   - If the question asks about company A but retrieved chunks are from company
     B, set `refused=true`. Do not summarise what the other company says.
   - Any answer that begins "not reported", "not available", or "not in the
     context" must have `refused=true` — never give an explanatory non-refusal.

9. Keep `answer` to one or two sentences. Do not invent, round, drop units, or
   remove qualifiers.

"""


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks into the context block string sent to the answer model."""
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = f"[Rank {i}] source={c.source} page={c.page}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


_WS_RE = re.compile(r"\s+")
_MD_RE = re.compile(r"\*+|~~|`+|<br\s*/?>|#{1,6}\s*")
_ELLIPSIS_RE = re.compile(r"\.{3,}")
_DASH_TRANS = str.maketrans({"\u2013": "-", "\u2014": "-", "\u2212": "-", "\u2012": "-"})
_QUOTE_TRANS = str.maketrans({
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
})
_MIN_FRAGMENT_LEN = 3

# Bracketed footnote/reference markers common in PDF-parsed annual reports: [A], [B], [1], [A1]
_FOOTNOTE_RE = re.compile(r"\[[A-Za-z0-9]{1,3}\]")
# PDF bullet/list characters that appear as layout artifacts after parsing
_PDF_BULLET_RE = re.compile(r"[\uffee\u2022\u2023\u25cf\u2043\uff65\u2219\u25e6]")

# Table grounding fallback
_NUM_RE = re.compile(r"\d[\d,\.]*")
_TABLE_KINDS = frozenset({"table", "table_row"})
_METRIC_STOP = frozenset({
    # generic English function words
    "that", "this", "were", "have", "been", "with", "which", "from", "into",
    "when", "than", "after", "before", "during", "between", "about", "under",
    "within", "through", "their", "they", "more", "less", "some", "also",
    "both", "each", "such",
    # question words (what/how/was/did are len<4 so already excluded by the regex)
    "what", "many", "much", "does",
    # units / scale
    "million", "billion", "percent",
    # generic report / table noise
    "total", "amount", "value", "year", "period", "report", "reported", "company",
})


def source_tokens(source: str) -> frozenset[str]:
    """Return source-name tokens used to ignore company words when grounding table quotes."""
    return frozenset(re.findall(r"[a-z]{4,}", source.lower()))


def normalize_for_grounding(s: str) -> str:
    """Normalize text for strict grounding comparisons and verbatim matching."""
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_DASH_TRANS).translate(_QUOTE_TRANS)
    s = s.replace("\u2026", "...")
    s = _MD_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


def normalize_for_grounding_layout(s: str) -> str:
    """Normalize text for fallback grounding while stripping PDF layout artifacts."""
    s = _FOOTNOTE_RE.sub(" ", s)
    s = _PDF_BULLET_RE.sub(" ", s)
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_DASH_TRANS).translate(_QUOTE_TRANS)
    s = s.replace("\u2026", "...")
    s = _MD_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


def split_on_ellipsis(needle: str) -> list[str]:
    """Split a grounded quote on ellipsis markers and return the non-empty fragments."""
    parts = [p.strip() for p in _ELLIPSIS_RE.split(needle)]
    return [p for p in parts if p]


def fragments_in_order(fragments: list[str], haystack: str) -> bool:
    """Return whether all quote fragments appear in order inside the candidate text."""
    cursor = 0
    for frag in fragments:
        idx = haystack.find(frag, cursor)
        if idx == -1:
            return False
        cursor = idx + len(frag)
    return True


def table_grounding_fallback(
    cite: Citation,
    indexed: list[tuple[RetrievedChunk, str]],
    verbatim: str | None,
    question: str | None = None,
) -> RetrievedChunk | None:
    """Fallback for table/table_row chunks when exact quote grounding fails.

    Accepts only if:
    - chunk_kind is table or table_row
    - an exact numeric token from verbatim or citation.quote appears in the chunk
      (unit conversions are rejected — "32.7" does not match "32,667")
    - AND either ≥2 distinct metric words from question/quote appear in the chunk,
      OR a metric bigram (adjacent pair of metric words) appears in the chunk
    """
    quote_norm = normalize_for_grounding(html.unescape(cite.quote))
    verbatim_norm = normalize_for_grounding(verbatim) if verbatim else ""

    value_nums = frozenset(
        m
        for src in (quote_norm, verbatim_norm)
        for m in _NUM_RE.findall(src)
        if len(m) >= 3
    )
    if not value_nums:
        return None

    question_norm = normalize_for_grounding(question) if question else ""
    source_stop = source_tokens(cite.source)
    combined = quote_norm + " " + question_norm
    words_in_order = re.findall(r"[a-z]{4,}", combined)
    metric_words = [w for w in words_in_order if w not in _METRIC_STOP and w not in source_stop]

    if not metric_words:
        return None

    metric_word_set = set(metric_words)
    bigrams = [
        f"{words_in_order[i]} {words_in_order[i + 1]}"
        for i in range(len(words_in_order) - 1)
        if words_in_order[i] in metric_word_set and words_in_order[i + 1] in metric_word_set
    ]

    matches: list[RetrievedChunk] = []
    for chunk, chunk_norm in indexed:
        if chunk.chunk_kind not in _TABLE_KINDS:
            continue
        if not any(v in chunk_norm for v in value_nums):
            continue
        matching_words = {w for w in metric_word_set if w in chunk_norm}
        if len(matching_words) < 2 and not any(b in chunk_norm for b in bigrams):
            continue
        matches.append(chunk)

    if not matches:
        return None

    return (
        next((c for c in matches if c.page == cite.page and c.source == cite.source), None)
        or next((c for c in matches if c.page == cite.page), None)
        or matches[0]
    )


def clean_citation_quote(raw: str) -> str:
    """Strip serialized datapoint metadata; collapse table rows to a hint."""
    if not raw:
        return raw
    text = raw.strip()
    # Datapoint chunks are formatted "Metric: ...\n...\nQuote: <real>".
    if "Quote:" in text:
        text = text.split("Quote:", 1)[1].strip()
    # If pipe-separated table content remains, replace with table hint.
    if text.count("|") >= 2:
        return "Refer to the table on this page."
    return text


def citation_contains_verbatim_number(cite: Citation, verbatim: str | None) -> bool:
    """Return whether a citation quote contains the key numeric token from the verbatim answer."""
    if not verbatim:
        return True
    verbatim_nums = {
        n for n in _NUM_RE.findall(normalize_for_grounding(verbatim))
        if len(n) >= 3
    }
    if not verbatim_nums:
        return True
    quote_nums = set(_NUM_RE.findall(normalize_for_grounding(cite.quote)))
    return bool(verbatim_nums & quote_nums)


def ground_citations(
    citations: list[Citation],
    chunks: list[RetrievedChunk],
    question: str | None = None,
    verbatim: str | None = None,
) -> tuple[list[Citation], str | None]:
    """Return (grounded_citations, failure_reason)."""
    if not citations:
        return [], "no citations returned"

    indexed_strict: list[tuple[RetrievedChunk, str]] = [
        (c, normalize_for_grounding(html.unescape(c.text))) for c in chunks
    ]
    indexed_layout: list[tuple[RetrievedChunk, str]] | None = None

    grounded: list[Citation] = []
    for cite in citations:
        needle = normalize_for_grounding(html.unescape(cite.quote))
        if not needle:
            continue

        fragments = split_on_ellipsis(needle)
        if not fragments or any(len(f) < _MIN_FRAGMENT_LEN for f in fragments):
            continue

        matches = [
            chunk for chunk, hay in indexed_strict
            if fragments_in_order(fragments, hay)
        ]

        if not matches:
            if indexed_layout is None:
                indexed_layout = [
                    (c, normalize_for_grounding_layout(html.unescape(c.text)))
                    for c in chunks
                ]
            needle_layout = normalize_for_grounding_layout(html.unescape(cite.quote))
            frags_layout = split_on_ellipsis(needle_layout)
            if frags_layout and all(len(f) >= _MIN_FRAGMENT_LEN for f in frags_layout):
                matches = [
                    chunk for chunk, hay in indexed_layout
                    if fragments_in_order(frags_layout, hay)
                ]

        if not matches:
            table_chunk = table_grounding_fallback(cite, indexed_strict, verbatim, question=question)
            if table_chunk is not None:
                matches = [table_chunk]

        if not matches:
            continue

        best = next(
            (m for m in matches if m.page == cite.page and m.source == cite.source),
            None,
        ) or next(
            (m for m in matches if m.page == cite.page), None,
        ) or matches[0]

        if len(citations) == 1 and not citation_contains_verbatim_number(cite, verbatim):
            continue

        grounded.append(Citation(
            source=best.source,
            page=best.page,
            quote=clean_citation_quote(cite.quote),
        ))

    if not grounded:
        return [], "no citations could be grounded in the retrieved chunks"
    return grounded, None


def refuse(question: str, reason: str) -> VerbatimAnswer:
    """Build a refusal answer object with a consistent fallback message."""
    return VerbatimAnswer(
        question=question,
        answer="The answer is not available in the provided reports.",
        verbatim=None,
        citations=[],
        refused=True,
        refusal_reason=reason,
    )


def answer_question(question: str, chunks: list[RetrievedChunk], history: list[dict] | None = None) -> VerbatimAnswer:
    """Answer one question from retrieved chunks and return a grounded `VerbatimAnswer`."""
    if not chunks:
        return refuse(question, "no retrieved context")

    client = openai_client()
    context = format_context(chunks)
    user_msg = f"Question: {question}\n\nContext:\n{context}"
    model = settings.openai_answer_model

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (history or [])[-3:]:
        messages.append({"role": "user", "content": h["question"]})
        messages.append({"role": "assistant", "content": h["answer"]})
    messages.append({"role": "user", "content": user_msg})

    t0 = time.perf_counter()
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=LLMAnswer,
    )
    logger.info("TIMING answer_llm(%s): %.2fs", model, time.perf_counter() - t0)
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return refuse(question, "LLM returned no parsed output")

    if parsed.refused:
        return VerbatimAnswer(
            question=question,
            answer=parsed.answer or "The answer is not available in the provided reports.",
            verbatim=None,
            citations=[],
            refused=True,
            refusal_reason=parsed.refusal_reason or "model declined to answer",
        )

    grounded, failure = ground_citations(parsed.citations, chunks, question=question, verbatim=parsed.verbatim)
    if failure is not None:
        return VerbatimAnswer(
            question=question,
            answer="The answer is not available in the provided reports.",
            verbatim=None,
            citations=[],
            refused=True,
            refusal_reason=f"ungrounded citation: {failure}",
        )

    return VerbatimAnswer(
        question=question,
        answer=parsed.answer,
        verbatim=parsed.verbatim,
        citations=grounded,
        refused=False,
        refusal_reason=None,
    )
