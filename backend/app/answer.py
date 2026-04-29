from __future__ import annotations

import re

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
3. For numerical, financial, or date answers, set `verbatim` to the exact span
   from the context that contains the figure (e.g. "€32.7 billion", "44,000",
   "9,609,432"). Keep the surrounding currency/units exactly as written.
4. If the answer is not present in the context, set `refused=true`, give a
   short `refusal_reason`, and leave `citations` empty. Do not guess.
5. Keep `answer` concise — one or two sentences. Do not invent figures.
"""


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = f"[Rank {i}] source={c.source} page={c.page}"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


_WS_RE = re.compile(r"\s+")
_MD_RE = re.compile(r"\*+|~~|`+|<br\s*/?>|#{1,6}\s*")


def _normalize_for_grounding(s: str) -> str:
    s = _MD_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def _ground_citations(
    citations: list[Citation], chunks: list[RetrievedChunk]
) -> tuple[list[Citation], list[GroundingDrop], str | None]:
    """Return (grounded_citations, drops, failure_reason).

    Strips ungrounded citations rather than refusing the whole answer.
    Returns a failure only when no citations survive grounding.
    """
    if not citations:
        return [], [], "no citations returned"
    by_key: dict[tuple[str, int], str] = {
        (c.source, c.page): _normalize_for_grounding(c.text) for c in chunks
    }
    grounded: list[Citation] = []
    drops: list[GroundingDrop] = []
    for cite in citations:
        haystack = by_key.get((cite.source, cite.page))
        if haystack is None:
            drops.append(GroundingDrop(
                source=cite.source, page=cite.page, quote=cite.quote,
                reason="source_page_not_in_retrieved_set",
            ))
            continue
        needle = _normalize_for_grounding(cite.quote)
        if not needle:
            drops.append(GroundingDrop(
                source=cite.source, page=cite.page, quote=cite.quote,
                reason="empty_quote",
            ))
            continue
        if needle not in haystack:
            drops.append(GroundingDrop(
                source=cite.source, page=cite.page, quote=cite.quote,
                reason="quote_not_found_verbatim",
            ))
            continue
        grounded.append(cite)
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
