from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedPage:
    """One parsed PDF page with its 1-based page number and normalized text."""
    page: int
    text: str


@dataclass(frozen=True)
class ParseResult:
    """Parsed document pages plus the parser name that produced them."""
    pages: list[ParsedPage]
    parser: str


class ParserUnavailableError(RuntimeError):
    """Raised when the configured parser cannot be used in the current environment."""
    pass


def llamaparse_page_text(page: dict[str, Any]) -> str:
    """Extract the best available text field from one raw LlamaParse page record."""
    for key in ("md", "markdown", "text", "content"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def llamaparse_json_to_pages(results: list[dict[str, Any]]) -> list[ParsedPage]:
    """Convert raw LlamaParse JSON output into normalized `ParsedPage` records."""
    pages: list[ParsedPage] = []
    fallback_page = 1
    for result in results:
        for page in result.get("pages", []):
            text = llamaparse_page_text(page)
            if not text:
                fallback_page += 1
                continue
            page_no = page.get("page") or page.get("page_number") or fallback_page
            pages.append(ParsedPage(page=int(page_no), text=text))
            fallback_page = int(page_no) + 1
    return pages



def strip_boilerplate(text: str) -> str:
    """Return parsed page text after lightweight boilerplate stripping."""
    return text.strip()



def persist_llamaparse_artifacts(
    results: list[dict[str, Any]],
    *,
    source_path: Path,
    processed_dir: Path,
) -> tuple[Path, Path]:
    """Persist raw parser JSON and concatenated markdown, then return both output paths."""
    json_path = processed_dir / "llamaparse" / f"{source_path.stem}.json"
    markdown_path = processed_dir / "markdown" / f"{source_path.stem}.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    pages = llamaparse_json_to_pages(results)
    markdown = "\n\n---\n\n".join(page.text for page in pages)
    markdown_path.write_text(markdown, encoding="utf-8")
    return json_path, markdown_path


def parse_pdf_llamaparse(
    path: Path,
    *,
    api_key: str | None,
    processed_dir: Path | None = None,
) -> list[ParsedPage]:
    """Parse a PDF with LlamaParse and return non-empty normalized pages."""
    if not api_key:
        raise ParserUnavailableError(
            "LLAMA_CLOUD_API_KEY is not set. Add it to .env."
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from llama_parse import LlamaParse
    except ModuleNotFoundError as exc:
        raise ParserUnavailableError(
            "LlamaParse is not installed. Run `pip install -e .[dev]`."
        ) from exc

    parser = LlamaParse(
        api_key=api_key,
        result_type="markdown",
        split_by_page=True,
        disable_image_extraction=True,
        verbose=False,
        show_progress=False,
        ignore_errors=False,
    )
    results = parser.get_json_result(str(path))

    if processed_dir is not None:
        persist_llamaparse_artifacts(
            results,
            source_path=path,
            processed_dir=processed_dir,
        )

    pages = llamaparse_json_to_pages(results)
    return [ParsedPage(page=p.page, text=strip_boilerplate(p.text)) for p in pages if strip_boilerplate(p.text)]


def parse_pdf_pages(
    path: Path,
    *,
    processed_dir: Path | None = None,
    llama_cloud_api_key: str | None = None,
) -> ParseResult:
    """Parse a PDF with LlamaParse and return a unified `ParseResult`."""
    llama_pages = parse_pdf_llamaparse(
        path,
        api_key=llama_cloud_api_key,
        processed_dir=processed_dir,
    )
    return ParseResult(pages=llama_pages, parser="llamaparse")


def as_page_tuples(pages: list[ParsedPage]) -> Iterator[tuple[int, str]]:
    """Yield `(page, text)` tuples from parsed page objects."""
    for page in pages:
        yield page.page, page.text
