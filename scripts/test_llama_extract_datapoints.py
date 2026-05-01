#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.llama_extract_datapoints import extract_annual_report_datapoints

_OUT_ROOT = Path("backend/data/processed/llamaextract")


def _preview(text: str | None, n: int = 80) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text[:n] + ("…" if len(text) > n else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Test LlamaExtract on an annual report PDF.")
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--company", default=None)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--target-pages",
        default=None,
        help="page range string passed to LlamaExtract (e.g. '5,145-180')",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    result = extract_annual_report_datapoints(
        args.pdf,
        company=args.company,
        year=args.year,
        page_range=args.target_pages,
    )

    out_path = args.out or (_OUT_ROOT / f"{args.pdf.stem}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"saved → {out_path}")

    print(f"\n=== FTE datapoints ({len(result.fte_datapoints)}) ===")
    for dp in result.fte_datapoints:
        parts = [
            f"value={dp.value}",
            f"label={dp.label}",
            f"basis={dp.basis or '-'}",
            f"period={dp.period or '-'}",
            f"page={dp.page or '-'}",
        ]
        print("  " + " | ".join(parts))
        if dp.quote:
            print(f"    quote: {_preview(dp.quote)}")

    print(f"\n=== Sustainability goals ({len(result.sustainability_goals)}) ===")
    for g in result.sustainability_goals:
        parts = [
            f"year={g.target_year or '-'}",
            f"scope={g.scope or '-'}",
            f"metric={g.metric or '-'}",
            f"value={g.value_or_target or '-'}",
            f"page={g.page or '-'}",
        ]
        print("  " + " | ".join(parts))
        print(f"    goal: {_preview(g.goal)}")
        if g.quote:
            print(f"    quote: {_preview(g.quote)}")

    print(f"\n=== KPI highlights ({len(result.kpi_highlights)}) ===")
    for k in result.kpi_highlights:
        parts = [
            f"value={k.value}",
            f"metric={k.metric}",
            f"unit={k.unit or '-'}",
            f"period={k.period or '-'}",
            f"page={k.page or '-'}",
        ]
        print("  " + " | ".join(parts))
        if k.quote:
            print(f"    quote: {_preview(k.quote)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
