"""RAGAS evaluation for the annual-report RAG pipeline."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from backend.app.answer import answer_question
from backend.app.config import settings
from backend.app.retrieval import retrieve_decomposed
from backend.app.schemas import RetrievalQuery

DEFAULT_QUESTIONS = Path("backend/evals/questions-2025-multi-company.yaml")
DEFAULT_RUNS_DIR = Path("backend/evals/ragas_runs")

METRIC_SETS = ("faithfulness", "context")


def _question_text(q: dict[str, Any]) -> str:
    return q.get("user_input") or q.get("question") or ""


def _ground_truth(q: dict[str, Any]) -> str:
    if q.get("reference"):
        return str(q["reference"])
    any_list = q.get("expected_answer_contains_any") or []
    all_list = q.get("expected_answer_contains_all") or []
    if any_list:
        return str(any_list[0])
    if all_list:
        return " ".join(str(x) for x in all_list)
    return ""


def _build_metrics(llm: Any, embeddings: Any, metric_set: str) -> list[Any]:
    if metric_set == "context":
        from ragas.metrics import ContextPrecision, ContextRecall

        return [ContextRecall(llm=llm), ContextPrecision(llm=llm)]

    from ragas.metrics import AnswerRelevancy, Faithfulness

    return [Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings)]


def _build_llm_and_embeddings() -> tuple[Any, Any]:
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    api_key = settings.openai_api_key.get_secret_value()
    llm = LangchainLLMWrapper(ChatOpenAI(model=settings.openai_answer_model, api_key=api_key))
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=settings.openai_embedding_model, api_key=api_key)
    )
    return llm, embeddings


def run_ragas_eval(
    path: Path,
    *,
    company: str | None,
    year: int | None,
    limit: int | None,
    runs_dir: Path | None,
    no_save: bool,
    metric_set: str = "faithfulness",
) -> None:
    raw = yaml.safe_load(path.read_text())
    questions: list[dict[str, Any]] = raw.get("questions", [])

    filtered = [
        q
        for q in questions
        if q.get("expected_behavior") != "refuse"
        and q.get("eval_type") != "refusal_rule"
        and (company is None or q.get("company") == company)
        and (year is None or q.get("year") == year)
        and _question_text(q)
    ]
    if limit:
        filtered = filtered[:limit]

    if not filtered:
        print("No questions matched filters.")
        return

    print(f"Running {len(filtered)} questions through pipeline… (metrics: {metric_set})")

    records: list[dict[str, Any]] = []
    skipped = 0

    for i, q in enumerate(filtered, 1):
        print(f"  [{i}/{len(filtered)}] {q['id']}", end="", flush=True)
        try:
            retrieval = retrieve_decomposed(
                RetrievalQuery(
                    question=_question_text(q),
                    top_k=settings.top_k,
                    company=q.get("company"),
                    year=q.get("year"),
                )
            )
            answer = answer_question(_question_text(q), retrieval.chunks)
            if answer.refused:
                print(" (refused, skipped)")
                skipped += 1
                continue
            records.append(
                {
                    "question": _question_text(q),
                    "answer": answer.answer,
                    "contexts": [c.text for c in retrieval.chunks],
                    "ground_truth": _ground_truth(q),
                }
            )
            print(" ok")
        except Exception as exc:
            print(f" ERROR: {exc}")
            skipped += 1

    if not records:
        print("No valid records to evaluate.")
        return

    print(f"\nEvaluating {len(records)} answers with RAGAS ({metric_set})…")

    from ragas import EvaluationDataset, SingleTurnSample, evaluate

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["answer"],
            retrieved_contexts=r["contexts"],
            reference=r["ground_truth"] or None,
        )
        for r in records
    ]
    dataset = EvaluationDataset(samples=samples)

    llm, embeddings = _build_llm_and_embeddings()
    metrics = _build_metrics(llm, embeddings, metric_set)

    result = evaluate(dataset, metrics=metrics)

    print(f"\n=== RAGAS Results ({metric_set}) ===")
    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in ("user_input", "response", "retrieved_contexts", "reference")]
    scores: dict[str, float] = {}
    for col in metric_cols:
        mean = df[col].dropna().mean()
        scores[col] = float(mean)
        print(f"  {col:35s}: {mean:.3f}")

    if runs_dir and not no_save:
        runs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "config": {
                "metric_set": metric_set,
                "questions_file": str(path),
                "company": company,
                "year": year,
                "n_evaluated": len(records),
                "n_skipped": skipped,
                "answer_model": settings.openai_answer_model,
                "embedding_model": settings.openai_embedding_model,
            },
            "scores": scores,
            "rows": records,
        }
        out = runs_dir / f"{ts}_{metric_set}.json"
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nrun saved → {out}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="RAGAS evaluation against the live RAG pipeline.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--company", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--metrics",
        choices=METRIC_SETS,
        default="faithfulness",
        help="faithfulness = Faithfulness+AnswerRelevancy; context = ContextRecall+ContextPrecision",
    )
    args = parser.parse_args()
    run_ragas_eval(
        args.questions,
        company=args.company,
        year=args.year,
        limit=args.limit,
        runs_dir=args.runs_dir,
        no_save=args.no_save,
        metric_set=args.metrics,
    )


if __name__ == "__main__":
    _cli()
