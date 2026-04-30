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
    retrieval_hit: bool | None
    answer_correct: bool | None
    citation_grounded: bool | None
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


def _citation_chunk_contains_answer(
    ans: VerbatimAnswer,
    chunks: list[RetrievedChunk],
    needles: list[str],
) -> bool:
    cited_locations = {(cite.source, cite.page) for cite in ans.citations}
    for cite in ans.citations:
        for chunk in chunks:
            if (chunk.source, chunk.page) not in cited_locations:
                continue
            if chunk.source != cite.source or chunk.page != cite.page:
                continue
            text = chunk.text.lower()
            if any(n.lower() in text for n in needles):
                return True
    return False


def _expected_pages(q: EvalQuestion) -> list[int]:
    accepted_pages = getattr(q, "accepted_pages", None)
    if accepted_pages:
        return [int(page) for page in accepted_pages]
    if q.expected_page is not None:
        return [q.expected_page]
    return []


def _page_matches_expected(page: int, expected_pages: list[int]) -> bool:
    return any(abs(page - expected_page) <= 1 for expected_page in expected_pages)


def _format_expected_pages(expected_pages: list[int]) -> str:
    if len(expected_pages) == 1:
        return str(expected_pages[0])
    return repr(expected_pages)


def _retrieval_hit(q: EvalQuestion, retrieved_chunks: list[RetrievedChunk]) -> bool | None:
    expected_pages = _expected_pages(q)
    if not expected_pages:
        return None
    return any(
        _page_matches_expected(chunk.page, expected_pages)
        and (q.expected_source is None or chunk.source == q.expected_source)
        for chunk in retrieved_chunks
    )


def _citation_grounded(q: EvalQuestion, ans: VerbatimAnswer) -> bool | None:
    expected_pages = _expected_pages(q)
    if not expected_pages:
        return None
    return any(
        _page_matches_expected(cite.page, expected_pages)
        and (q.expected_source is None or cite.source == q.expected_source)
        for cite in ans.citations
    )


def score_one(
    q: EvalQuestion,
    ans: VerbatimAnswer,
    retrieved_chunks: list[RetrievedChunk] | None = None,
) -> tuple[bool, list[str], bool | None, bool | None, bool | None]:
    reasons: list[str] = []
    chunks = retrieved_chunks or []
    retrieval_hit = _retrieval_hit(q, chunks)
    answer_correct: bool | None = None
    citation_grounded: bool | None = _citation_grounded(q, ans)

    if q.expected_behavior == "refuse":
        if ans.refused is not True:
            reasons.append("expected refusal but answer was given")
        return (not reasons, reasons, retrieval_hit, answer_correct, citation_grounded)

    if ans.refused:
        reasons.append(_refusal_failure_reason(ans))
        answer_correct = False
        return (False, reasons, retrieval_hit, answer_correct, citation_grounded)

    if retrieval_hit is False:
        source_msg = (
            f" and source={q.expected_source!r}" if q.expected_source is not None else ""
        )
        expected_pages = _expected_pages(q)
        reasons.append(
            "no retrieved chunk page within ±1 of "
            f"{_format_expected_pages(expected_pages)}{source_msg}"
        )

    hay = _haystack(ans)
    answer_correct = True
    if q.expected_answer_contains_any:
        if not any(needle.lower() in hay for needle in q.expected_answer_contains_any):
            reasons.append(
                f"none of {q.expected_answer_contains_any!r} found in answer"
            )
            answer_correct = False
    if q.expected_answer_contains_all:
        missing = [n for n in q.expected_answer_contains_all if n.lower() not in hay]
        if missing:
            reasons.append(f"missing required substrings: {missing!r}")
            answer_correct = False

    if q.expected_source and q.category != "multi_document":
        if not any(c.source == q.expected_source for c in ans.citations):
            reasons.append(f"no citation with source={q.expected_source!r}")

    expected_pages = _expected_pages(q)
    if expected_pages:
        fallback_ok = False
        if not citation_grounded and answer_correct and retrieved_chunks:
            needles = q.expected_answer_contains_any or q.expected_answer_contains_all or []
            fallback_ok = bool(needles) and _citation_chunk_contains_answer(
                ans, retrieved_chunks, needles
            )
            if fallback_ok:
                citation_grounded = True
        if not citation_grounded:
            cited = [c.page for c in ans.citations]
            reasons.append(
                "no citation page within ±1 of "
                f"{_format_expected_pages(expected_pages)} (got {cited})"
            )

    applicable_flags = [
        flag for flag in (retrieval_hit, answer_correct, citation_grounded) if flag is not None
    ]
    passed = not reasons and all(applicable_flags)
    return (passed, reasons, retrieval_hit, answer_correct, citation_grounded)


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


def _format_expected(outcome: Outcome) -> str:
    expected_parts: list[str] = []
    q = outcome.question
    if q.expected_behavior is not None:
        expected_parts.append(f"behavior={q.expected_behavior}")
    if q.expected_source is not None:
        expected_parts.append(f"source={q.expected_source}")
    expected_pages = _expected_pages(q)
    if expected_pages:
        expected_parts.append(f"page≈{_format_expected_pages(expected_pages)}")
    if q.expected_answer_contains_any:
        expected_parts.append(f"contains_any={q.expected_answer_contains_any!r}")
    if q.expected_answer_contains_all:
        expected_parts.append(f"contains_all={q.expected_answer_contains_all!r}")
    return ", ".join(expected_parts) if expected_parts else "<none>"


def _format_citations(ans: VerbatimAnswer | None) -> str:
    if ans is None or not ans.citations:
        return "<none>"
    return ", ".join(f"{c.source} p.{c.page}" for c in ans.citations)


def format_result_line(outcome: Outcome) -> str:
    status = "PASS" if outcome.passed else "FAIL"
    cited_pages = [c.page for c in outcome.answer.citations] if outcome.answer else []

    if outcome.passed:
        return (
            f"{status}  {outcome.question.id:<38} "
            f"{outcome.question.category:<22} "
            f"pages={cited_pages!s:<10} "
            f'"{_preview(outcome.answer)}"'
        )

    lines = [
        f"{status}  {outcome.question.id}  [{outcome.question.category}]",
        f"        question: {outcome.question.question}",
        f"        expected: {_format_expected(outcome)}",
        f'        answer: "{_preview(outcome.answer)}"',
        f"        citations: {_format_citations(outcome.answer)}",
        (
            f"        retrieval: pages={outcome.retrieved_pages} "
            f"(expected page {_format_expected_pages(_expected_pages(outcome.question))})"
        ),
        "        reasons:",
    ]
    lines.extend(f"        - {reason}" for reason in outcome.reasons)

    if outcome.retrieved_chunks:
        lines.append("        retrieved chunks:")
        for i, chunk in enumerate(outcome.retrieved_chunks, start=1):
            lines.append(
                f"        [#{i} page={chunk.page} score={chunk.score:.3f} id={chunk.id}] "
                f"{_chunk_preview(chunk)}"
            )

    return "\n".join(lines)


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


def _flag_rate(outcomes: list[Outcome], attr: str) -> tuple[int, int, float]:
    values = [getattr(o, attr) for o in outcomes if getattr(o, attr) is not None]
    passed = sum(1 for value in values if value)
    total = len(values)
    pct = (passed / total * 100) if total else 0.0
    return passed, total, pct


def format_summary(outcomes: list[Outcome]) -> str:
    total = len(outcomes)
    passed = sum(1 for o in outcomes if o.passed)
    pct = (passed / total * 100) if total else 0.0
    retrieval_passed, retrieval_total, retrieval_pct = _flag_rate(outcomes, "retrieval_hit")
    answer_passed, answer_total, answer_pct = _flag_rate(outcomes, "answer_correct")
    citation_passed, citation_total, citation_pct = _flag_rate(outcomes, "citation_grounded")
    lines = [
        "",
        f"Overall:           {passed}/{total}  {pct:.1f}%",
        f"Retrieval hit:     {retrieval_passed}/{retrieval_total}  {retrieval_pct:.1f}%",
        (
            f"Answer correct:    {answer_passed}/{answer_total}  {answer_pct:.1f}%"
            "   (excludes refusal questions)"
        ),
        f"Citation grounded: {citation_passed}/{citation_total}  {citation_pct:.1f}%",
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
                    "accepted_pages": o.question.accepted_pages,
                    "source": o.question.expected_source,
                    "behavior": o.question.expected_behavior,
                },
                "passed": o.passed,
                "retrieval_hit": o.retrieval_hit,
                "answer_correct": o.answer_correct,
                "citation_grounded": o.citation_grounded,
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
            eff_year_for_q = None if q.category == "multi_document" else eff_year
            result = pipeline_answer_with_context(
                q.question, top_k=top_k, company=eff_company, year=eff_year_for_q
            )
            ans = result.answer
            retrieved_chunks = result.retrieved_chunks
            retrieved_pages = [chunk.page for chunk in retrieved_chunks]
            (
                passed,
                reasons,
                retrieval_hit,
                answer_correct,
                citation_grounded,
            ) = score_one(q, ans, retrieved_chunks)
        except Exception as exc:  # noqa: BLE001
            ans = None
            retrieved_chunks = []
            retrieved_pages = []
            retrieval_hit = _retrieval_hit(q, retrieved_chunks)
            answer_correct = None if q.expected_behavior == "refuse" else False
            citation_grounded = None if not _expected_pages(q) else False
            passed = False
            reasons = [f"exception: {exc!s}"]

        outcome = Outcome(
            question=q,
            answer=ans,
            retrieved_pages=retrieved_pages,
            retrieved_chunks=retrieved_chunks,
            retrieval_hit=retrieval_hit,
            answer_correct=answer_correct,
            citation_grounded=citation_grounded,
            passed=passed,
            reasons=reasons,
        )
        outcomes.append(outcome)

        if show_failures_only and passed:
            continue
        print(format_result_line(outcome))

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
