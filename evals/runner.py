from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from backend.app.config import settings
from backend.app.pipeline import answer_with_context as pipeline_answer_with_context
from backend.app.schemas import EvalQuestion, EvalSet, RetrievedChunk, VerbatimAnswer

PREVIEW_CHARS = 80
RETRIEVAL_PREVIEW_CHARS = 160
DEFAULT_QUESTIONS = Path("evals/questions.yaml")
DEFAULT_RUNS_DIR = Path("evals/runs")


@dataclass
class Outcome:
    question: EvalQuestion
    answer: VerbatimAnswer | None
    retrieved_pages: list[int]
    retrieved_chunks: list[RetrievedChunk]
    passed: bool
    reasons: list[str]


def load_eval_set(path: Path) -> EvalSet:
    with path.open() as f:
        data = yaml.safe_load(f)
    return EvalSet(**data)


def _haystack(ans: VerbatimAnswer) -> str:
    return f"{ans.answer} {ans.verbatim or ''}".lower()


def _refusal_failure_reason(ans: VerbatimAnswer) -> str:
    reason = ans.refusal_reason or "unknown refusal reason"
    return f"no final answer produced: {reason}"


def score_one(q: EvalQuestion, ans: VerbatimAnswer) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if q.expected_behavior == "refuse":
        if not (ans.refused or not ans.citations):
            reasons.append("expected refusal but answer was given")
        return (not reasons, reasons)

    if ans.refused:
        reasons.append(_refusal_failure_reason(ans))
        return (False, reasons)

    hay = _haystack(ans)
    if q.expected_answer_contains_any:
        if not any(needle.lower() in hay for needle in q.expected_answer_contains_any):
            reasons.append(
                f"none of {q.expected_answer_contains_any!r} found in answer"
            )
    if q.expected_answer_contains_all:
        missing = [n for n in q.expected_answer_contains_all if n.lower() not in hay]
        if missing:
            reasons.append(f"missing required substrings: {missing!r}")

    if q.expected_source:
        if not any(c.source == q.expected_source for c in ans.citations):
            reasons.append(f"no citation with source={q.expected_source!r}")

    if q.expected_page is not None:
        if not any(abs(c.page - q.expected_page) <= 1 for c in ans.citations):
            cited = [c.page for c in ans.citations]
            reasons.append(f"no citation page within ±1 of {q.expected_page} (got {cited})")

    return (not reasons, reasons)


def _preview(ans: VerbatimAnswer | None) -> str:
    if ans is None:
        return "<no answer>"
    if ans.refused:
        text = f"refused: {ans.refusal_reason or ''}"
    else:
        text = ans.answer
    text = text.replace("\n", " ").strip()
    if len(text) > PREVIEW_CHARS:
        text = text[:PREVIEW_CHARS] + "…"
    return text


def _chunk_preview(chunk: RetrievedChunk) -> str:
    text = chunk.text.replace("\n", " ").strip()
    if len(text) > RETRIEVAL_PREVIEW_CHARS:
        text = text[:RETRIEVAL_PREVIEW_CHARS] + "…"
    return text


def format_result_line(outcome: Outcome) -> str:
    status = "PASS" if outcome.passed else "FAIL"
    pages = (
        [c.page for c in outcome.answer.citations] if outcome.answer else []
    )
    line = (
        f"{status}  {outcome.question.id:<38} "
        f"{outcome.question.category:<22} "
        f"pages={pages!s:<10} "
        f'"{_preview(outcome.answer)}"'
    )
    if not outcome.passed:
        line += (
            f"\n        retrieved={outcome.retrieved_pages} "
            f"expected={outcome.question.expected_page}"
        )
        for i, chunk in enumerate(outcome.retrieved_chunks, start=1):
            line += (
                f"\n        [#{i} page={chunk.page} score={chunk.score:.3f} id={chunk.id}] "
                f"{_chunk_preview(chunk)}"
            )
    return line


def _category_breakdown(outcomes: list[Outcome]) -> dict[str, dict[str, int | float]]:
    by_cat: dict[str, list[Outcome]] = defaultdict(list)
    for o in outcomes:
        by_cat[o.question.category].append(o)
    out: dict[str, dict[str, int | float]] = {}
    for cat in sorted(by_cat):
        items = by_cat[cat]
        c_pass = sum(1 for o in items if o.passed)
        out[cat] = {
            "passed": c_pass,
            "total": len(items),
            "pct": round(c_pass / len(items) * 100, 1),
        }
    return out


def format_summary(outcomes: list[Outcome]) -> str:
    total = len(outcomes)
    passed = sum(1 for o in outcomes if o.passed)
    pct = (passed / total * 100) if total else 0.0
    lines = [
        "",
        f"Overall: {passed}/{total}  {pct:.1f}%",
        "By category:",
    ]
    for cat, stats in _category_breakdown(outcomes).items():
        lines.append(
            f"  {cat:<24} {stats['passed']}/{stats['total']}  {stats['pct']:.0f}%"
        )
    return "\n".join(lines)


def _run_filename(timestamp: datetime, passed: int, total: int) -> str:
    ts = timestamp.strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{ts}_{passed:02d}of{total:02d}.json"


def _build_run_record(
    *,
    timestamp: datetime,
    questions_path: Path,
    eval_set: EvalSet,
    company: str,
    year: int,
    top_k: int,
    outcomes: list[Outcome],
) -> dict:
    total = len(outcomes)
    passed = sum(1 for o in outcomes if o.passed)
    pct = round(passed / total * 100, 1) if total else 0.0

    return {
        "timestamp": timestamp.isoformat(),
        "questions_path": str(questions_path),
        "eval_set": {
            "source": eval_set.source,
            "company": eval_set.company,
            "year": eval_set.year,
            "n_questions": len(eval_set.questions),
        },
        "config": {
            "company": company,
            "year": year,
            "top_k": top_k,
            "answer_model": settings.openai_answer_model,
            "embedding_model": settings.openai_embedding_model,
            "chroma_collection": settings.chroma_collection,
            "chroma_persist_dir": str(settings.chroma_persist_dir),
        },
        "summary": {
            "total": total,
            "passed": passed,
            "pct": pct,
            "by_category": _category_breakdown(outcomes),
        },
        "outcomes": [
            {
                "id": o.question.id,
                "question": o.question.question,
                "category": o.question.category,
                "difficulty": o.question.difficulty,
                "expected": {
                    "answer_contains_any": o.question.expected_answer_contains_any,
                    "answer_contains_all": o.question.expected_answer_contains_all,
                    "page": o.question.expected_page,
                    "source": o.question.expected_source,
                    "behavior": o.question.expected_behavior,
                },
                "passed": o.passed,
                "reasons": o.reasons,
                "retrieval": {
                    "pages": o.retrieved_pages,
                    "chunks": [
                        {
                            "id": chunk.id,
                            "source": chunk.source,
                            "company": chunk.company,
                            "year": chunk.year,
                            "page": chunk.page,
                            "score": chunk.score,
                            "text": chunk.text,
                        }
                        for chunk in o.retrieved_chunks
                    ],
                },
                "answer": (
                    json.loads(o.answer.model_dump_json()) if o.answer is not None else None
                ),
            }
            for o in outcomes
        ],
    }


def _load_run_record(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _format_comparison(previous_path: Path, outcomes: list[Outcome]) -> str:
    previous = _load_run_record(previous_path)
    previous_passed_by_id = {
        item["id"]: item["passed"] for item in previous.get("outcomes", [])
    }

    flips: list[str] = []
    for outcome in outcomes:
        previous_passed = previous_passed_by_id.get(outcome.question.id)
        if previous_passed is None or previous_passed == outcome.passed:
            continue
        marker = "✅→❌" if previous_passed and not outcome.passed else "❌→✅"
        flips.append(f"  {marker} {outcome.question.id}")

    lines = ["", f"Changes since {previous_path}:"]
    if flips:
        lines.extend(flips)
    else:
        lines.append("  (no pass/fail changes)")
    return "\n".join(lines)


def save_run(
    record: dict,
    runs_dir: Path,
) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.fromisoformat(record["timestamp"])
    fname = _run_filename(ts, record["summary"]["passed"], record["summary"]["total"])
    path = runs_dir / fname
    with path.open("w") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


def run_eval(
    path: Path,
    *,
    company: str | None,
    year: int | None,
    top_k: int,
    show_failures_only: bool,
    compare_to: Path | None,
    runs_dir: Path | None,
) -> int:
    started = datetime.now(timezone.utc).replace(microsecond=0)
    eval_set = load_eval_set(path)
    eff_company = company if company is not None else eval_set.company
    eff_year = year if year is not None else eval_set.year

    outcomes: list[Outcome] = []
    for q in eval_set.questions:
        try:
            result = pipeline_answer_with_context(
                q.question, top_k=top_k, company=eff_company, year=eff_year
            )
            ans = result.answer
            retrieved_chunks = result.retrieved_chunks
            retrieved_pages = [chunk.page for chunk in retrieved_chunks]
            passed, reasons = score_one(q, ans)
        except Exception as exc:  # noqa: BLE001
            ans = None
            retrieved_chunks = []
            retrieved_pages = []
            passed = False
            reasons = [f"exception: {exc!s}"]

        outcome = Outcome(
            question=q,
            answer=ans,
            retrieved_pages=retrieved_pages,
            retrieved_chunks=retrieved_chunks,
            passed=passed,
            reasons=reasons,
        )
        outcomes.append(outcome)

        if show_failures_only and passed:
            continue
        print(format_result_line(outcome))
        if not passed:
            for r in reasons:
                print(f"        - {r}")

    print(format_summary(outcomes))
    if compare_to is not None:
        print(_format_comparison(compare_to, outcomes))

    if runs_dir is not None:
        record = _build_run_record(
            timestamp=started,
            questions_path=path,
            eval_set=eval_set,
            company=eff_company,
            year=eff_year,
            top_k=top_k,
            outcomes=outcomes,
        )
        out_path = save_run(record, runs_dir)
        print(f"\nrun saved → {out_path}")

    return 0


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run eval set against the live RAG pipeline.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--company", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=settings.top_k)
    parser.add_argument("--show-failures-only", action="store_true")
    parser.add_argument(
        "--compare-to",
        type=Path,
        default=None,
        help="path to a previous run JSON to compare pass/fail flips against",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="directory to save per-run JSON records (default: evals/runs)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="do not write a run record to --runs-dir",
    )
    args = parser.parse_args(argv)

    try:
        return run_eval(
            args.questions,
            company=args.company,
            year=args.year,
            top_k=args.top_k,
            show_failures_only=args.show_failures_only,
            compare_to=args.compare_to,
            runs_dir=None if args.no_save else args.runs_dir,
        )
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
