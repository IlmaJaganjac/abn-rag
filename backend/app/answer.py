from __future__ import annotations

import html
import re
import unicodedata
from collections import defaultdict

from backend.app._openai import openai_client
from backend.app.config import settings
from backend.app.schemas import (
    Citation,
    EvidenceItem,
    GroundingDrop,
    LLMAnswer,
    RetrievedChunk,
    VerbatimAnswer,
)

SYSTEM_PROMPT = """\
You answer questions about annual reports using only the provided context blocks.

Rules:
1. Use ONLY the context blocks below. Never answer from prior knowledge.
2. If the answer is not present in the context, set refused=true and leave evidence empty.
3. Every non-refused answer MUST include at least one evidence item.
4. Keep answer concise: one or two sentences.
5. For numerical, financial, date, FTE, ESG, or sustainability-target answers,
   set verbatim to the exact value copied from the context.

Evidence rules:
A. Use evidence_type="exact_quote" for normal prose.
   - quote MUST be copied verbatim from one context block.
   - source and page MUST come from that same context block header.

B. Use evidence_type="table_value" for tables, KPI cards, or values split across lines.
   - Do NOT invent a prose quote from a table.
   - Fill table_title, metric, period, and value using exact strings from the context when available.
   - value MUST be copied verbatim.
   - source and page MUST come from the context block containing the table/KPI value.

C. Use evidence_type="datapoint" when the context already contains a pre-extracted datapoint.
   - value MUST be copied verbatim.
   - metric/datapoint_type should match the context.
   - source and page MUST come from the context block.

Output requirements:
- Put evidence items in the `evidence` field.
- Leave `citations` empty; it is kept only for backward compatibility.
- If evidence_type="exact_quote", include quote.
- If evidence_type="table_value", include value and at least one of metric, period, or table_title.
- If evidence_type="datapoint", include value and metric or datapoint_type.
- Do not paraphrase evidence quotes.
- Do not use exact_quote for table-derived answers unless the exact sentence exists in the context.
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
_TRAILING_ARTIFACT_RE = re.compile(r"[.!?,;}\]\s]+$")


def _clean_needle(needle: str) -> str:
    """Strip trailing punctuation and JSON artifacts the LLM appends to verbatim quotes."""
    return _TRAILING_ARTIFACT_RE.sub("", needle)


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
        needle = _clean_needle(_normalize_for_grounding(html.unescape(cite.quote)))
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


def _evidence_text(ev: EvidenceItem) -> str:
    return ev.quote if ev.evidence_type == "exact_quote" else (ev.value or "")


def _evidence_to_citation(ev: EvidenceItem, chunk: RetrievedChunk) -> Citation:
    return Citation(source=chunk.source, page=chunk.page, quote=_evidence_text(ev))


def _evidence_with_chunk_metadata(ev: EvidenceItem, chunk: RetrievedChunk) -> EvidenceItem:
    return ev.model_copy(update={"source": chunk.source, "page": chunk.page})


def _ground_evidence(
    evidence: list[EvidenceItem], chunks: list[RetrievedChunk]
) -> tuple[list[EvidenceItem], list[Citation], list[GroundingDrop], str | None]:
    """Ground model evidence and convert it to legacy citations for evals.

    exact_quote evidence is grounded like old citations. table_value/datapoint
    evidence is grounded by requiring the copied value to appear in a retrieved
    chunk, preferring the model's cited source/page when possible.
    """
    if not evidence:
        return [], [], [], "no evidence returned"

    indexed: list[tuple[RetrievedChunk, str]] = [
        (c, _normalize_for_grounding(html.unescape(c.text))) for c in chunks
    ]

    grounded_evidence: list[EvidenceItem] = []
    citations: list[Citation] = []
    drops: list[GroundingDrop] = []
    for ev in evidence:
        raw_text = _evidence_text(ev)
        needle = _clean_needle(_normalize_for_grounding(html.unescape(raw_text)))
        if not needle:
            drops.append(GroundingDrop(
                source=ev.source, page=ev.page, quote=raw_text,
                reason="empty_quote",
            ))
            continue

        if ev.evidence_type == "exact_quote":
            matches = [chunk for chunk, hay in indexed if needle in hay]
        else:
            matches = [chunk for chunk, hay in indexed if needle in hay]

        if not matches:
            page_in_set = any(chunk.page == ev.page for chunk, _ in indexed)
            drops.append(GroundingDrop(
                source=ev.source,
                page=ev.page,
                quote=raw_text,
                reason=(
                    "quote_not_found_verbatim" if page_in_set
                    else "source_page_not_in_retrieved_set"
                ),
            ))
            continue

        best = next(
            (m for m in matches if m.page == ev.page and m.source == ev.source),
            None,
        ) or next(
            (m for m in matches if m.page == ev.page), None,
        ) or matches[0]

        fixed = _evidence_with_chunk_metadata(ev, best)
        grounded_evidence.append(fixed)
        citations.append(_evidence_to_citation(fixed, best))

    if not grounded_evidence:
        return [], [], drops, "no evidence could be grounded in the retrieved chunks"
    return grounded_evidence, citations, drops, None


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
            evidence=[],
            citations=[],
            refused=True,
            refusal_reason=parsed.refusal_reason or "model declined to answer",
            raw_evidence=parsed.evidence,
            raw_citations=parsed.citations,
        )

    if parsed.evidence:
        grounded_evidence, grounded, drops, failure = _ground_evidence(
            parsed.evidence, chunks
        )
        failure_label = "evidence"
    else:
        grounded_evidence = []
        grounded, drops, failure = _ground_citations(parsed.citations, chunks)
        failure_label = "citation"

    if failure is not None:
        return VerbatimAnswer(
            question=question,
            answer="The answer is not available in the provided reports.",
            verbatim=None,
            evidence=[],
            citations=[],
            refused=True,
            refusal_reason=f"ungrounded {failure_label}: {failure}",
            raw_evidence=parsed.evidence,
            raw_citations=parsed.citations,
            grounding_drops=drops,
        )

    return VerbatimAnswer(
        question=question,
        answer=parsed.answer,
        verbatim=parsed.verbatim,
        evidence=grounded_evidence,
        citations=grounded,
        refused=False,
        refusal_reason=None,
        raw_evidence=parsed.evidence,
        raw_citations=parsed.citations,
        grounding_drops=drops,
    )
