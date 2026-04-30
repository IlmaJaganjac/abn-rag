from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

from backend.app.answer import answer_question
from backend.app.config import settings
from backend.app.retrieval import retrieve
from backend.app.schemas import RetrievalQuery, RetrievedChunk, VerbatimAnswer

PREVIEW_CHARS = 300
KNOWN_COMPANY_ALIASES = {
    "ASML": ("asml",),
    "ABN AMRO": ("abn amro", "abn-amro"),
    "SHELL": ("shell",),
    "CM": ("cm.com", "cm com"),
    "HEINEKEN": ("heineken",),
    "TESLA": ("tesla",),
    "APPLE": ("apple",),
    "MICROSOFT": ("microsoft",),
    "GOOGLE": ("google", "alphabet"),
}


@dataclass
class AnswerResult:
    answer: VerbatimAnswer
    retrieved_chunks: list[RetrievedChunk]


def _alias_pattern(alias: str) -> re.Pattern[str]:
    flexible = r"[\s-]+".join(re.escape(part) for part in alias.casefold().split())
    return re.compile(rf"(?<![a-z0-9]){flexible}(?![a-z0-9])")


def _canonical_company(company: str | None) -> str | None:
    if company is None:
        return None
    company_key = company.casefold().strip()
    for canonical, aliases in KNOWN_COMPANY_ALIASES.items():
        if company_key == canonical.casefold() or company_key in aliases:
            return canonical
    return company.strip()


def _mentioned_company(question: str) -> str | None:
    for canonical, aliases in sorted(
        KNOWN_COMPANY_ALIASES.items(),
        key=lambda item: max(len(alias) for alias in item[1]),
        reverse=True,
    ):
        if any(_alias_pattern(alias).search(question.casefold()) for alias in aliases):
            return canonical
    return None


def _company_mismatch_refusal(
    question: str,
    company: str | None,
) -> VerbatimAnswer | None:
    active_company = _canonical_company(company)
    mentioned_company = _mentioned_company(question)
    if active_company is None or mentioned_company is None:
        return None
    if mentioned_company == active_company:
        return None
    return VerbatimAnswer(
        question=question,
        answer="",
        citations=[],
        refused=True,
        refusal_reason=(
            f"Question mentions {mentioned_company}, "
            f"but the active company filter is {active_company}."
        ),
    )


def _print_context(chunks) -> None:
    for i, c in enumerate(chunks, start=1):
        preview = c.text.replace("\n", " ").strip()
        if len(preview) > PREVIEW_CHARS:
            preview = preview[:PREVIEW_CHARS] + "…"
        print(f"[#{i}  score={c.score:.3f}  page={c.page}  id={c.id}]")
        print(f"    {preview}\n")
    print(
        f"-- {len(chunks)} hits  "
        f"collection='{settings.chroma_collection}'  "
        f"persist='{settings.chroma_persist_dir}'\n"
    )


def answer_with_context(
    question: str,
    *,
    top_k: int,
    company: str | None,
    year: int | None,
) -> AnswerResult:
    refusal = _company_mismatch_refusal(question, company)
    if refusal is not None:
        return AnswerResult(answer=refusal, retrieved_chunks=[])

    result = retrieve(
        RetrievalQuery(question=question, top_k=top_k, company=company, year=year)
    )
    return AnswerResult(
        answer=answer_question(question, result.chunks),
        retrieved_chunks=result.chunks,
    )


def answer(
    question: str,
    *,
    top_k: int,
    company: str | None,
    year: int | None,
) -> VerbatimAnswer:
    return answer_with_context(
        question,
        top_k=top_k,
        company=company,
        year=year,
    ).answer


def run(
    question: str,
    *,
    top_k: int,
    company: str | None,
    year: int | None,
    show_context: bool,
) -> VerbatimAnswer:
    result = answer_with_context(
        question,
        top_k=top_k,
        company=company,
        year=year,
    )
    if show_context:
        _print_context(result.retrieved_chunks)
    ans = result.answer
    print(ans.model_dump_json(indent=2))
    return ans


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Retrieve top-k chunks and produce a structured VerbatimAnswer."
    )
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--company", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="print retrieved chunks before the JSON answer",
    )
    args = parser.parse_args(argv)

    try:
        run(
            args.question,
            top_k=args.top_k,
            company=args.company,
            year=args.year,
            show_context=args.show_context,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
