from __future__ import annotations

import argparse
import sys

from backend.app.answer import answer_question
from backend.app.config import settings
from backend.app.retrieval import retrieve
from backend.app.schemas import RetrievalQuery, VerbatimAnswer

PREVIEW_CHARS = 300


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


def answer(
    question: str,
    *,
    top_k: int,
    company: str | None,
    year: int | None,
) -> VerbatimAnswer:
    result = retrieve(
        RetrievalQuery(question=question, top_k=top_k, company=company, year=year)
    )
    return answer_question(question, result.chunks)


def run(
    question: str,
    *,
    top_k: int,
    company: str | None,
    year: int | None,
    show_context: bool,
) -> VerbatimAnswer:
    result = retrieve(
        RetrievalQuery(question=question, top_k=top_k, company=company, year=year)
    )
    if show_context:
        _print_context(result.chunks)
    ans = answer_question(question, result.chunks)
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
