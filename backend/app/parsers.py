from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParsedPage:
    page: int
    text: str


@dataclass(frozen=True)
class ParseResult:
    pages: list[ParsedPage]
    parser: str


class ParserUnavailableError(RuntimeError):
    pass


def _llamaparse_page_text(page: dict[str, Any]) -> str:
    for key in ("md", "markdown", "text", "content"):
        value = page.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def llamaparse_json_to_pages(results: list[dict[str, Any]]) -> list[ParsedPage]:
    pages: list[ParsedPage] = []
    fallback_page = 1
    for result in results:
        for page in result.get("pages", []):
            text = _llamaparse_page_text(page)
            if not text:
                fallback_page += 1
                continue
            page_no = page.get("page") or page.get("page_number") or fallback_page
            pages.append(ParsedPage(page=int(page_no), text=text))
            fallback_page = int(page_no) + 1
    return pages


def _normalized_text(text: str) -> str:
    return " ".join(text.split()).casefold()


def combine_with_pdf_text_layer(
    parsed_pages: list[ParsedPage],
    pdf_text_pages: list[ParsedPage],
) -> list[ParsedPage]:
    """Prefer the PDF text layer for verbatim reading, keep parser text too.

    LlamaParse markdown is useful for structure, but annual-report highlight
    pages sometimes lose visual reading order in dense KPI layouts. PyMuPDF's
    native text layer often preserves those verbatim datapoints.
    """
    pdf_text_by_page = {page.page: page.text for page in pdf_text_pages}
    combined: list[ParsedPage] = []
    for page in parsed_pages:
        pdf_text = pdf_text_by_page.get(page.page, "").strip()
        parsed_text = page.text.strip()
        if not pdf_text:
            combined.append(page)
            continue

        normalized_pdf = _normalized_text(pdf_text)
        normalized_parsed = _normalized_text(parsed_text)
        if normalized_pdf == normalized_parsed or normalized_pdf in normalized_parsed:
            combined.append(page)
        elif normalized_parsed in normalized_pdf:
            combined.append(ParsedPage(page=page.page, text=pdf_text))
        else:
            text = f"{pdf_text}\n\n--- Parsed markdown ---\n\n{parsed_text}"
            combined.append(ParsedPage(page=page.page, text=text))
    return combined


def _add_pdf_text_layer(path: Path, pages: list[ParsedPage]) -> list[ParsedPage]:
    try:
        pdf_text_pages = parse_pdf_pymupdf(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not add PDF text layer for %s: %s", path, exc)
        return pages
    return combine_with_pdf_text_layer(pages, pdf_text_pages)


def persist_llamaparse_artifacts(
    results: list[dict[str, Any]],
    *,
    source_path: Path,
    processed_dir: Path,
) -> tuple[Path, Path]:
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

    return _add_pdf_text_layer(path, llamaparse_json_to_pages(results))


def parse_pdf_pymupdf(path: Path) -> list[ParsedPage]:
    doc = fitz.open(path)
    try:
        pages: list[ParsedPage] = []
        for i in range(doc.page_count):
            text = doc.load_page(i).get_text("text").strip()
            if text:
                pages.append(ParsedPage(page=i + 1, text=text))
        return pages
    finally:
        doc.close()


def parse_pdf_pages(
    path: Path,
    *,
    processed_dir: Path | None = None,
    llama_cloud_api_key: str | None = None,
) -> ParseResult:
    pages = parse_pdf_llamaparse(
        path,
        api_key=llama_cloud_api_key,
        processed_dir=processed_dir,
    )
    return ParseResult(pages=pages, parser="llamaparse")


def as_page_tuples(pages: list[ParsedPage]) -> Iterator[tuple[int, str]]:
    for page in pages:
        yield page.page, page.text
