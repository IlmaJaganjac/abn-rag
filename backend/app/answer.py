from __future__ import annotations

import html
import os
import re
import unicodedata
from collections import defaultdict

from backend.app._openai import openai_client
from backend.app.config import settings
from backend.app.schemas import Citation, GroundingDrop, LLMAnswer, RetrievedChunk, VerbatimAnswer

SYSTEM_PROMPT = """\
You answer questions about annual reports using only the provided context blocks.

Rules:
1. Use ONLY the context blocks below. Never use prior knowledge or memory.

2. Every non-refused answer must include at least one citation. The `source`
   must equal the exact filename from the block header (e.g. `shell-2025.pdf`).
   The `quote` must be copied verbatim — character by character, preserving
   exact wording, capitalisation, tense, spacing, and punctuation:
   - Do NOT paraphrase, reorder, add words, or change any word.
   - Do NOT change capitalisation or verb tense.
   - Do NOT insert context words (year, company, period) unless they appear at
     that exact position in the chunk.
   - To skip intermediate text use `...` (three ASCII dots) explicitly; each
     segment must appear in order within one chunk. For unrelated spans, emit
     separate citations.
   - Prefer the SHORTEST exact span that supports the answer. For numeric or
     table facts, prefer the single shortest sentence or row with the figure.

3. For extracted-datapoint or table chunks, cite the exact label+value lines:
   e.g. "Metric: ...\nValue: ...\nUnit: ..." or the exact table row. Preserve
   labels and line breaks exactly. Do not join lines unless the joined text
   appears verbatim in the chunk. Do not add labels that aren't in the chunk.

4. For numeric, financial, or percentage answers, set `verbatim` to the exact
   figure span (e.g. "15.4%", "€32.7bn", "> 44,000"). Preserve all qualifiers
   and units exactly — do not round, convert, or simplify. Match unit scope:
   spend/capex → monetary only; FTE/headcount → workforce metrics only;
   emissions → match scope and unit (kt, Mt, CO₂e). If label and unit do not
   clearly match the question, refuse instead of guessing.

5. If you cannot copy an exact verbatim quote from a single retrieved chunk, or
   the answer is absent from the context, set `refused=true`, give a short
   `refusal_reason`, and leave `citations` empty. Do not guess.

6. Keep `answer` to one or two sentences. Do not invent, round, or remove
   qualifiers.
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
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


def _source_tokens(source: str) -> frozenset[str]:
    """Tokens of len≥4 extracted from a source filename, used to filter company names."""
    return frozenset(re.findall(r"[a-z]{4,}", source.lower()))


def _normalize_for_grounding(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_DASH_TRANS).translate(_QUOTE_TRANS)
    s = s.replace("\u2026", "...")
    s = _MD_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


def _normalize_for_grounding_layout(s: str) -> str:
    """Fallback normalization that additionally strips PDF layout artifacts:
    bracketed footnote markers like [A][B][C] and PDF bullet characters.
    Used only when the strict normalization fails to find a match.
    Footnote/bullet stripping runs before NFKC so original codepoints are matched.
    """
    s = _FOOTNOTE_RE.sub(" ", s)
    s = _PDF_BULLET_RE.sub(" ", s)
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_DASH_TRANS).translate(_QUOTE_TRANS)
    s = s.replace("\u2026", "...")
    s = _MD_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip().lower()
    return s


def _split_on_ellipsis(needle: str) -> list[str]:
    parts = [p.strip() for p in _ELLIPSIS_RE.split(needle)]
    return [p for p in parts if p]


def _fragments_in_order(fragments: list[str], haystack: str) -> bool:
    cursor = 0
    for frag in fragments:
        idx = haystack.find(frag, cursor)
        if idx == -1:
            return False
        cursor = idx + len(frag)
    return True


def _table_grounding_fallback(
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
    quote_norm = _normalize_for_grounding(html.unescape(cite.quote))
    verbatim_norm = _normalize_for_grounding(verbatim) if verbatim else ""

    value_nums = frozenset(
        m
        for src in (quote_norm, verbatim_norm)
        for m in _NUM_RE.findall(src)
        if len(m) >= 3
    )
    if not value_nums:
        return None

    question_norm = _normalize_for_grounding(question) if question else ""
    source_stop = _source_tokens(cite.source)
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


def _ground_citations(
    citations: list[Citation],
    chunks: list[RetrievedChunk],
    question: str | None = None,
    verbatim: str | None = None,
) -> tuple[list[Citation], list[GroundingDrop], str | None]:
    """Return (grounded_citations, drops, failure_reason).

    Grounding rules:
    - The quote must appear verbatim (modulo Unicode/whitespace normalization)
      in a retrieved chunk. The quote may be split on `...`; each fragment
      must appear in order within a single chunk's text, with fragments
      shorter than _MIN_FRAGMENT_LEN rejected to prevent trivial matches.
    - The LLM sometimes mislabels the citation's `source` (using the report
      title from the page footer) or `page` (off-by-a-few when the same name
      appears on adjacent pages). When the quote matches a retrieved chunk,
      we rewrite the citation's `source` and `page` to that chunk's metadata.
      The grounding guarantee ("the cited evidence is in the retrieved set")
      still holds; we just trust the chunk over the LLM's labelling.
    - Preference order when multiple chunks contain the quote: same page+
      source as cited, then same page, then any retrieved chunk.
    """
    if not citations:
        return [], [], "no citations returned"

    indexed_strict: list[tuple[RetrievedChunk, str]] = [
        (c, _normalize_for_grounding(html.unescape(c.text))) for c in chunks
    ]
    # Computed lazily only when the strict pass misses
    indexed_layout: list[tuple[RetrievedChunk, str]] | None = None

    grounded: list[Citation] = []
    drops: list[GroundingDrop] = []
    for cite in citations:
        needle = _normalize_for_grounding(html.unescape(cite.quote))
        if not needle:
            drops.append(GroundingDrop(
                source=cite.source, page=cite.page, quote=cite.quote,
                reason="empty_quote",
            ))
            continue

        fragments = _split_on_ellipsis(needle)
        if not fragments or any(len(f) < _MIN_FRAGMENT_LEN for f in fragments):
            drops.append(GroundingDrop(
                source=cite.source, page=cite.page, quote=cite.quote,
                reason="quote_not_found_verbatim",
            ))
            continue

        matches = [
            chunk for chunk, hay in indexed_strict
            if _fragments_in_order(fragments, hay)
        ]

        if not matches:
            # Fallback: strip PDF layout artifacts (footnote markers, bullet chars)
            if indexed_layout is None:
                indexed_layout = [
                    (c, _normalize_for_grounding_layout(html.unescape(c.text)))
                    for c in chunks
                ]
            needle_layout = _normalize_for_grounding_layout(html.unescape(cite.quote))
            frags_layout = _split_on_ellipsis(needle_layout)
            if frags_layout and all(len(f) >= _MIN_FRAGMENT_LEN for f in frags_layout):
                matches = [
                    chunk for chunk, hay in indexed_layout
                    if _fragments_in_order(frags_layout, hay)
                ]

        if not matches:
            table_chunk = _table_grounding_fallback(cite, indexed_strict, verbatim, question=question)
            if table_chunk is not None:
                matches = [table_chunk]

        if not matches:
            page_in_set = any(chunk.page == cite.page for chunk, _ in indexed_strict)
            drops.append(GroundingDrop(
                source=cite.source, page=cite.page, quote=cite.quote,
                reason=(
                    "quote_not_found_verbatim" if page_in_set
                    else "source_page_not_in_retrieved_set"
                ),
            ))
            continue

        best = next(
            (m for m in matches if m.page == cite.page and m.source == cite.source),
            None,
        ) or next(
            (m for m in matches if m.page == cite.page), None,
        ) or matches[0]

        grounded.append(Citation(
            source=best.source,
            page=best.page,
            quote=cite.quote,
        ))

    if not grounded:
        return [], drops, "no citations could be grounded in the retrieved chunks"
    return grounded, drops, None


def _refuse(question: str, reason: str) -> VerbatimAnswer:
    return VerbatimAnswer(
        question=question,
        answer="The answer is not available in the provided reports.",
        verbatim=None,
        citations=[],
        refused=True,
        refusal_reason=reason,
    )


def answer_question(
    question: str, chunks: list[RetrievedChunk]
) -> VerbatimAnswer:
    if not chunks:
        return _refuse(question, "no retrieved context")

    client = openai_client()
    context = _format_context(chunks)
    user_msg = f"Question: {question}\n\nContext:\n{context}"
    model = os.environ.get("ANSWER_MODEL") or settings.openai_answer_model
    # gpt-5-mini only supports the default temperature (1); omit parameter for those models
    supports_temp_zero = "gpt-5-mini" not in model

    completion = client.beta.chat.completions.parse(
        model=model,
        **( {"temperature": 0} if supports_temp_zero else {}),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=LLMAnswer,
    )
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        return _refuse(question, "LLM returned no parsed output")

    if parsed.refused:
        return VerbatimAnswer(
            question=question,
            answer=parsed.answer or "The answer is not available in the provided reports.",
            verbatim=None,
            citations=[],
            refused=True,
            refusal_reason=parsed.refusal_reason or "model declined to answer",
            raw_citations=parsed.citations,
        )

    grounded, drops, failure = _ground_citations(parsed.citations, chunks, question=question, verbatim=parsed.verbatim)
    if failure is not None:
        return VerbatimAnswer(
            question=question,
            answer="The answer is not available in the provided reports.",
            verbatim=None,
            citations=[],
            refused=True,
            refusal_reason=f"ungrounded citation: {failure}",
            raw_citations=parsed.citations,
            grounding_drops=drops,
        )

    return VerbatimAnswer(
        question=question,
        answer=parsed.answer,
        verbatim=parsed.verbatim,
        citations=grounded,
        refused=False,
        refusal_reason=None,
        raw_citations=parsed.citations,
        grounding_drops=drops,
    )
