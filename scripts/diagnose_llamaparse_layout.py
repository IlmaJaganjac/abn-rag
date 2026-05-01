from __future__ import annotations

import argparse
import json
import sys
import textwrap
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.config import settings


DEFAULT_PAGES = [5, 51, 55, 62, 100]
OUTPUT_DIR = Path("backend/data/processed/llamaparse_layout_diagnostics")


def _parse_pages(value: str | None) -> list[int]:
    if not value:
        return DEFAULT_PAGES
    pages: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        pages.append(int(part))
    return pages


def _markdown_text(page: dict[str, Any]) -> str:
    for key in ("md", "markdown", "text", "content"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _short(value: Any, width: int = 220) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = repr(value)
    text = " ".join(text.split())
    return textwrap.shorten(text, width=width, placeholder="...")


def _page_number(page: dict[str, Any], fallback: int) -> int:
    value = page.get("page") or page.get("page_number") or page.get("pageNumber") or fallback
    return int(value)


def _page_items(page: dict[str, Any]) -> list[Any]:
    for key in ("items", "structuredContent", "structured_content"):
        value = page.get(key)
        if isinstance(value, list):
            return value
    return []


def _page_images(page: dict[str, Any]) -> list[Any]:
    value = page.get("images")
    return value if isinstance(value, list) else []


def _page_charts(page: dict[str, Any]) -> list[Any]:
    value = page.get("charts")
    return value if isinstance(value, list) else []


def _layout_elements(page: dict[str, Any]) -> list[Any]:
    layout = page.get("layout")
    if isinstance(layout, list):
        return layout
    if isinstance(layout, dict):
        for key in ("elements", "items", "blocks"):
            value = layout.get(key)
            if isinstance(value, list):
                return value
    return []


def _type_name(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("type", "element_type", "category", "label"):
            item = value.get(key)
            if item:
                return str(item)
    return type(value).__name__


def _find_pages(results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    pages_by_number: dict[int, dict[str, Any]] = {}
    fallback = 1
    for result in results:
        result_pages = result.get("pages", [])
        if not isinstance(result_pages, list):
            continue
        for page in result_pages:
            if not isinstance(page, dict):
                fallback += 1
                continue
            page_no = _page_number(page, fallback)
            pages_by_number[page_no] = page
            fallback = page_no + 1
    return pages_by_number


def _run_llamaparse(pdf_path: Path, api_key: str) -> list[dict[str, Any]]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from llama_parse import LlamaParse
    except ModuleNotFoundError:
        print("ERROR: LlamaParse is not installed. Run `pip install -e .[dev]`.", file=sys.stderr)
        raise SystemExit(1)

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        split_by_page=True,
        extract_layout=True,
        verbose=False,
        show_progress=False,
        ignore_errors=False,
    )
    results = parser.get_json_result(str(pdf_path))
    if not isinstance(results, list):
        raise RuntimeError(f"Expected LlamaParse JSON result list, got {type(results).__name__}")
    return results


def _print_page_report(page_no: int, page: dict[str, Any] | None) -> None:
    print(f"\n=== Page {page_no} ===")
    if page is None:
        print("page not found")
        return

    items = _page_items(page)
    images = _page_images(page)
    charts = _page_charts(page)
    layout_elements = _layout_elements(page)
    has_layout = "layout" in page and bool(layout_elements)
    type_counts = Counter(_type_name(element) for element in layout_elements)

    print(f"keys present: {', '.join(sorted(page.keys()))}")
    print(f"confidence: {page.get('confidence')}")
    print(f"parsingMode: {page.get('parsingMode')}")
    print(f"triggeredAutoMode: {page.get('triggeredAutoMode')}")
    print(f"noStructuredContent: {page.get('noStructuredContent')}")
    print(f"noTextContent: {page.get('noTextContent')}")
    print(f"number of items: {len(items)}")
    print(f"number of images: {len(images)}")
    print(f"number of charts: {len(charts)}")
    print(f"layout exists: {has_layout}")
    if has_layout:
        print(f"layout elements: {len(layout_elements)}")
        print(f"layout type counts: {dict(sorted(type_counts.items()))}")
        print("first 5 layout elements:")
        for index, element in enumerate(layout_elements[:5], start=1):
            print(f"  {index}. {_short(element)}")
    else:
        print("layout elements: 0")

    print("first 5 items:")
    for index, item in enumerate(items[:5], start=1):
        print(f"  {index}. {_short(item)}")

    markdown = _markdown_text(page)
    print("first 1000 characters of markdown:")
    print(markdown[:1000])


def main() -> int:
    arg_parser = argparse.ArgumentParser(description="Diagnose LlamaParse layout metadata.")
    arg_parser.add_argument("pdf", type=Path)
    arg_parser.add_argument("--company", default=None)
    arg_parser.add_argument("--year", type=int, default=None)
    arg_parser.add_argument("--pages", default=None, help="Comma-separated 1-indexed PDF pages.")
    args = arg_parser.parse_args()

    api_key = settings.llama_cloud_api_key.get_secret_value()
    if not api_key:
        print("ERROR: LLAMA_CLOUD_API_KEY is missing. Add it to .env.", file=sys.stderr)
        return 1

    if not args.pdf.exists():
        print(f"ERROR: PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    selected_pages = _parse_pages(args.pages)
    results = _run_llamaparse(args.pdf, api_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.pdf.stem}.json"
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"pdf: {args.pdf}")
    print(f"company: {args.company}")
    print(f"year: {args.year}")
    print(f"selected pages: {selected_pages}")
    print(f"raw json: {output_path}")

    pages_by_number = _find_pages(results)
    for page_no in selected_pages:
        _print_page_report(page_no, pages_by_number.get(page_no))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
