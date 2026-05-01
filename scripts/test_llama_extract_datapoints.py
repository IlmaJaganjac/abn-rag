#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
    parser.add_argument("--category", default=None, help="category-specific prompt (e.g. fte, sustainability, financial_highlight)")
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
        category=args.category,
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

    print(f"\n=== Financial highlights ({len(result.financial_highlights)}) ===")
    for fh in result.financial_highlights:
        parts = [
            f"value={fh.value}",
            f"metric={fh.metric}",
            f"unit={fh.unit or '-'}",
            f"period={fh.period or '-'}",
            f"page={fh.page or '-'}",
        ]
        print("  " + " | ".join(parts))
        if fh.quote:
            print(f"    quote: {_preview(fh.quote)}")

    print(f"\n=== Business performance ({len(result.business_performance)}) ===")
    for bp in result.business_performance:
        parts = [
            f"value={bp.value}",
            f"metric={bp.metric}",
            f"unit={bp.unit or '-'}",
            f"period={bp.period or '-'}",
            f"page={bp.page or '-'}",
        ]
        print("  " + " | ".join(parts))
        if bp.quote:
            print(f"    quote: {_preview(bp.quote)}")

    print(f"\n=== Shareholder returns ({len(result.shareholder_returns)}) ===")
    for sr in result.shareholder_returns:
        parts = [
            f"value={sr.value}",
            f"metric={sr.metric}",
            f"unit={sr.unit or '-'}",
            f"period={sr.period or '-'}",
            f"page={sr.page or '-'}",
        ]
        print("  " + " | ".join(parts))
        if sr.quote:
            print(f"    quote: {_preview(sr.quote)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
