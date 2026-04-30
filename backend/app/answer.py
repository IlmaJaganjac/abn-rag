from __future__ import annotations

import html
import logging
import re
import unicodedata
from collections import defaultdict

from backend.app._openai import openai_client
from backend.app.config import settings
from backend.app.schemas import Citation, GroundingDrop, LLMAnswer, RetrievedChunk, VerbatimAnswer

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You answer questions about annual reports using only the provided context blocks.

Rules:
1. Use ONLY the context blocks below. Never answer from prior knowledge or memory.

2. Cross-company guard: if the question explicitly names a company by
   proper noun (e.g. 'Intel', 'Tesla', 'ASM International', 'Samsung') AND
   that named company does NOT appear in any context block's source filename
   or company metadata, set refused=true with refusal_reason 'question is
   about a different company than the retrieved reports'. Do NOT refuse
   when the question contains no explicit company name, or when the named
   company matches the source filename. When in doubt, do not refuse on
   these grounds — fall through to the other rules.

3. Every non-refused answer MUST include at least one citation. Each citation's
   `source` and `page` must come from a context block header, and `quote` must be
   a verbatim span copied from that block's text (no paraphrasing, no edits).

4. The `source` field MUST equal the literal `source=` value in the context
   block header (e.g. `asml-2024.pdf`). Do NOT use the report's title, footer,
   or any other label. Copy the filename exactly.

5. The `quote` must be copied verbatim from the chunk's text. Do not paraphrase,
   reorder, restate, remove labels, or join separate lines unless the joined
   text appears exactly in the chunk. You MAY use `...` (three ASCII dots) to
   skip intermediate cells in a multi-year table (e.g. to skip a prior-year
   column), but each segment around the `...` must appear word-for-word in the
   chunk and the segments must appear in the original order. Do NOT use `...`
   to skip across unrelated text; if the answer needs two unrelated spans, emit
   two separate citations.

6. Metric chunks may be formatted like this:
   Metric: ...
   Period: ...
   Value: ...
   Unit: ...

   When citing a metric chunk, prefer quoting the full contiguous metric block
   exactly as it appears in the context. Preserve labels and line breaks exactly.
   Do not remove labels such as "Metric:", "Period:", "Value:", or "Unit:".
   Do not join separate lines unless the joined text appears exactly in the
   chunk.

7. For numerical, financial, percentage, employee-count, or date answers:
   - Set `verbatim` to the exact figure span from the context whenever possible,
     such as "15.4%", "23,126", "32,667.3", "€32.7bn", or "> 44,000".
   - Do not set `verbatim` to the whole metric block unless the figure cannot be
     isolated.
   - Preserve qualifiers exactly, including symbols/words such as ">", "<",
     "approximately", "around", "about", "more than", "less than", and "over".
   - Preserve units exactly, including "FTEs", "employees", "€ million",
     "€ billion", "%", "kt", "Mt", and "CO₂e".
   - Do not round, normalize, convert, or simplify figures unless the context
     itself gives that converted form.
   - If the evidence says "> 44,000", the answer must say "more than 44,000"
     or "> 44,000", not "44,000".

8. If multiple retrieved chunks give different valid definitions for the same
   question, prefer the definition whose metric name and unit most closely match
   the question. If the question is ambiguous and multiple definitions are
   present in the context, mention the definition used briefly, e.g. "on an FTE
   basis" or "on a headcount basis".

9. Unit and scope matching:
   - For spend, cost, expense, R&D spend, or capex questions, answer only from
     monetary units such as €, $, EUR, USD, million, or billion. Never answer
     with FTE, headcount, employees, units, kt, Mt, tonnes, or percentages.
   - For FTE, headcount, or employee questions, use workforce metrics only and
     preserve average vs year-end, payroll vs temporary, and FTE vs headcount.
   - For margin, rate, ratio, or percentage questions, prefer explicitly
     labelled margin/rate/ratio/percentage metrics with %. Do not calculate
     unless the user asks to calculate.
   - For emissions, scope, CO2, CO₂, or GHG questions, prefer matching
     scope/category and emissions units such as kt, Mt, tCO2e, or CO2e.
   - If label and unit do not clearly match the question, refuse instead of
     guessing.

10. If the answer is not present in the context, set `refused=true`, give a short
   `refusal_reason`, and leave `citations` empty. Do not guess.

11. Keep `answer` concise — one or two sentences. Do not invent, round, simplify,
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
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_FRAGMENT_LEN = 3
_MIN_REPAIR_TOKENS = 8
_MAX_REPAIR_TOKEN_WINDOW = 50
_PROTECTED_REPAIR_TOKENS = {
    "about",
    "above",
    "approximately",
    "around",
    "at",
    "below",
    "excluding",
    "fewer",
    "greater",
    "least",
    "less",
    "more",
    "not",
    "over",
    "than",
    "under",
}
_PROTECTED_REPAIR_SYMBOLS = {"<", ">", "%"}


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


def _protected_symbols(s: str) -> set[str]:
    return {symbol for symbol in _PROTECTED_REPAIR_SYMBOLS if symbol in s}


def _safe_token_subsequence_match(needle: str, haystack: str) -> bool:
    needle_tokens = _TOKEN_RE.findall(needle)
    haystack_matches = list(_TOKEN_RE.finditer(haystack))
    haystack_tokens = [match.group(0) for match in haystack_matches]
    if len(needle_tokens) < _MIN_REPAIR_TOKENS:
        return False

    needle_protected = set(needle_tokens) & _PROTECTED_REPAIR_TOKENS
    needle_symbols = _protected_symbols(needle)
    first_token = needle_tokens[0]

    for start in [i for i, token in enumerate(haystack_tokens) if token == first_token]:
        positions = [start]
        cursor = start + 1
        for token in needle_tokens[1:]:
            try:
                idx = haystack_tokens.index(token, cursor)
            except ValueError:
                break
            positions.append(idx)
            cursor = idx + 1
        if len(positions) != len(needle_tokens):
            continue

        token_window = positions[-1] - positions[0] + 1
        if token_window > _MAX_REPAIR_TOKEN_WINDOW:
            continue

        char_start = haystack_matches[positions[0]].start()
        char_end = haystack_matches[positions[-1]].end()
        window = haystack[char_start:char_end]
        window_tokens = set(_TOKEN_RE.findall(window))
        window_protected = window_tokens & _PROTECTED_REPAIR_TOKENS
        if not window_protected.issubset(needle_protected):
            continue
        if not _protected_symbols(window).issubset(needle_symbols):
            continue

        return True
    return False


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
        repair_quote = False

        if not matches:
            matches = [
                chunk for chunk, hay in indexed
                if _safe_token_subsequence_match(needle, hay)
            ]
            if matches:
                repair_quote = True

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

        grounded.append(Citation(
            source=best.source,
            page=best.page,
            quote=best.text if repair_quote else cite.quote,
        ))
        if repair_quote:
            logger.info(
                "repaired citation via token subsequence: source=%s page=%s",
                best.source,
                best.page,
            )

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
    user_msg = (
        f"Active company in context: infer from the source filenames and "
        f"company metadata in the context blocks below.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )

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
