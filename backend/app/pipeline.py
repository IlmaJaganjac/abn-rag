from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

from backend.app.answer import answer_question
from backend.app.config import settings
from backend.app.retrieval import retrieve_decomposed as retrieve
from backend.app.schemas import RetrievalQuery, RetrievedChunk, VerbatimAnswer

PREVIEW_CHARS = 300


@dataclass
class AnswerResult:
    """Answer payload paired with the chunks that were retrieved to produce it."""
    answer: VerbatimAnswer
    retrieved_chunks: list[RetrievedChunk]


def _print_context(chunks) -> None:
    """Print a readable preview of retrieved chunks for local debugging."""
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
    """Run retrieval plus answering and return both the answer and retrieved chunks."""
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
    """Return only the final answer object for a question and optional filters."""
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
    """Execute the local pipeline, optionally print context, and return the final answer."""
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
    """Parse CLI arguments for one question-answering run and return an exit code."""
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
