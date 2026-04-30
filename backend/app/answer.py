from __future__ import annotations

import html
import re
import unicodedata
from collections import defaultdict

from backend.app._openai import openai_client
from backend.app.config import settings
from backend.app.schemas import Citation, GroundingDrop, LLMAnswer, RetrievedChunk, VerbatimAnswer

SYSTEM_PROMPT = """\
You answer questions about annual reports using only the provided context blocks.

Rules:
1. Use ONLY the context blocks below. Never answer from prior knowledge or memory.
2. Every non-refused answer MUST include at least one citation. Each citation's
   `source` and `page` must come from a context block header, and `quote` must be
   a verbatim span copied from that block's text (no paraphrasing, no edits).
3. The `source` field MUST equal the literal `source=` value in the context
   block header (e.g. `asml-2024.pdf`). Do NOT use the report's title, footer,
   or any other label. Copy the filename exactly.
4. The `quote` must be copied verbatim from the chunk's text. Do not
   paraphrase, reorder, or restate. You MAY use `...` (three ASCII dots) to
   skip intermediate cells in a multi-year table (e.g. to skip a prior-year
   column), but each segment around the `...` must appear word-for-word in
   the chunk and the segments must appear in the original order. Do NOT use
   `...` to skip across unrelated text; if the answer needs two unrelated
   spans, emit two separate citations.
5. For numerical, financial, percentage, employee-count, or date answers:
   - Set `verbatim` to the exact span from the context that contains the figure.
   - When citing a metric chunk, quote the contiguous metric block verbatim,
     preserving labels and line breaks:
     Metric: ...
     Period: ...
     Value: ...
     Unit: ...
   - Preserve qualifiers exactly, including symbols/words such as ">", "<",
     "approximately", "around", "about", "more than", "less than", and "over".
   - Preserve units exactly, including "FTEs", "employees", "€ million",
     "€ billion", "%", "kt", "Mt", and "CO₂e".
   - Do not round, normalize, convert, or simplify figures unless the context
     itself gives that converted form.
   - If the evidence says "> 44,000", the answer must say "more than 44,000"
     or "> 44,000", not "44,000".

6. If the answer is not present in the context, set `refused=true`, give a
   short `refusal_reason`, and leave `citations` empty. Do not guess.

7. Keep `answer` concise — one or two sentences. Do not invent, round, simplify,
   or remove qualifiers from figures.
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


def _normalize_for_grounding(s: str) -> str:
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


def _ground_citations(
    citations: list[Citation], chunks: list[RetrievedChunk]
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

    indexed: list[tuple[RetrievedChunk, str]] = [
        (c, _normalize_for_grounding(html.unescape(c.text))) for c in chunks
    ]

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
            chunk for chunk, hay in indexed
            if _fragments_in_order(fragments, hay)
        ]

        if not matches:
            page_in_set = any(chunk.page == cite.page for chunk, _ in indexed)
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

        grounded.append(Citation(source=best.source, page=best.page, quote=cite.quote))

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

    completion = client.beta.chat.completions.parse(
        model=settings.openai_answer_model,
        temperature=0,
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

    grounded, drops, failure = _ground_citations(parsed.citations, chunks)
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
