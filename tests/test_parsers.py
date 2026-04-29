from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.parsers import (
    ParsedPage,
    ParserUnavailableError,
    combine_with_pdf_text_layer,
    llamaparse_json_to_pages,
    parse_pdf_pages,
    persist_docling_artifacts,
    persist_llamaparse_artifacts,
)


class FakeDoclingDocument:
    def save_as_json(self, path: Path) -> None:
        path.write_text('{"document": "ok"}', encoding="utf-8")

    def export_to_markdown(self) -> str:
        return "# Annual report\n"


def test_docling_parser_falls_back_to_pymupdf(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_docling(
        _path: Path,
        *,
        processed_dir: Path | None = None,
    ) -> list[ParsedPage]:
        raise ParserUnavailableError("missing")

    def fallback_pymupdf(_path: Path) -> list[ParsedPage]:
        return [ParsedPage(page=1, text="fallback page text")]

    monkeypatch.setattr("backend.app.parsers.parse_pdf_docling", missing_docling)
    monkeypatch.setattr("backend.app.parsers.parse_pdf_pymupdf", fallback_pymupdf)

    result = parse_pdf_pages(Path("report.pdf"), parser="docling", allow_fallback=True)

    assert result.parser == "pymupdf"
    assert result.pages == [ParsedPage(page=1, text="fallback page text")]


def test_docling_parser_strict_mode_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_docling(
        _path: Path,
        *,
        processed_dir: Path | None = None,
    ) -> list[ParsedPage]:
        raise ParserUnavailableError("missing")

    monkeypatch.setattr("backend.app.parsers.parse_pdf_docling", missing_docling)

    with pytest.raises(ParserUnavailableError):
        parse_pdf_pages(Path("report.pdf"), parser="docling", allow_fallback=False)


def test_docling_runtime_error_falls_back_to_pymupdf(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_docling(
        _path: Path,
        *,
        processed_dir: Path | None = None,
    ) -> list[ParsedPage]:
        raise RuntimeError("model cache unavailable")

    def fallback_pymupdf(_path: Path) -> list[ParsedPage]:
        return [ParsedPage(page=2, text="pymupdf fallback")]

    monkeypatch.setattr("backend.app.parsers.parse_pdf_docling", broken_docling)
    monkeypatch.setattr("backend.app.parsers.parse_pdf_pymupdf", fallback_pymupdf)

    result = parse_pdf_pages(Path("report.pdf"), parser="docling", allow_fallback=True)

    assert result.parser == "pymupdf"
    assert result.pages == [ParsedPage(page=2, text="pymupdf fallback")]


def test_llamaparse_parser_dispatches_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_llamaparse(
        _path: Path,
        *,
        api_key: str | None,
        processed_dir: Path | None = None,
    ) -> list[ParsedPage]:
        assert api_key == "llx-key"
        return [ParsedPage(page=3, text="llamaparse markdown")]

    monkeypatch.setattr("backend.app.parsers.parse_pdf_llamaparse", fake_llamaparse)

    result = parse_pdf_pages(
        Path("report.pdf"),
        parser="llamaparse",
        allow_fallback=False,
        llama_cloud_api_key="llx-key",
    )

    assert result.parser == "llamaparse"
    assert result.pages == [ParsedPage(page=3, text="llamaparse markdown")]


def test_llamaparse_parser_strict_mode_requires_api_key() -> None:
    with pytest.raises(ParserUnavailableError, match="LLAMA_CLOUD_API_KEY"):
        parse_pdf_pages(Path("report.pdf"), parser="llamaparse", allow_fallback=False)


def test_llamaparse_missing_key_falls_back_to_pymupdf(monkeypatch: pytest.MonkeyPatch) -> None:
    def fallback_pymupdf(_path: Path) -> list[ParsedPage]:
        return [ParsedPage(page=1, text="fallback without llama key")]

    monkeypatch.setattr("backend.app.parsers.parse_pdf_pymupdf", fallback_pymupdf)

    result = parse_pdf_pages(Path("report.pdf"), parser="llamaparse", allow_fallback=True)

    assert result.parser == "pymupdf"
    assert result.pages == [ParsedPage(page=1, text="fallback without llama key")]


def test_persist_docling_artifacts_writes_json_and_markdown(tmp_path) -> None:
    json_path, markdown_path = persist_docling_artifacts(
        FakeDoclingDocument(),
        source_path=Path("reports/asml-2025.pdf"),
        processed_dir=tmp_path,
    )

    assert json_path == tmp_path / "docling" / "asml-2025.json"
    assert markdown_path == tmp_path / "markdown" / "asml-2025.md"
    assert json_path.read_text(encoding="utf-8") == '{"document": "ok"}'
    assert markdown_path.read_text(encoding="utf-8") == "# Annual report\n"


def test_persist_llamaparse_artifacts_writes_markdown(tmp_path) -> None:
    json_path, markdown_path = persist_llamaparse_artifacts(
        [
            {
                "job_id": "job-1",
                "pages": [
                    {"page": 1, "md": "# Page one"},
                    {"page": 2, "md": "## Page two"},
                ],
            }
        ],
        source_path=Path("reports/shell-2025.pdf"),
        processed_dir=tmp_path,
    )

    assert json_path == tmp_path / "llamaparse" / "shell-2025.json"
    assert markdown_path == tmp_path / "markdown" / "shell-2025.md"
    assert '"job_id": "job-1"' in json_path.read_text(encoding="utf-8")
    assert markdown_path.read_text(encoding="utf-8") == "# Page one\n\n---\n\n## Page two"


def test_llamaparse_json_to_pages_prefers_markdown_text() -> None:
    pages = llamaparse_json_to_pages(
        [
            {
                "pages": [
                    {"page": 7, "md": "# Markdown", "text": "plain text"},
                    {"page": 8, "text": "plain fallback"},
                ]
            }
        ]
    )

    assert pages == [
        ParsedPage(page=7, text="# Markdown"),
        ParsedPage(page=8, text="plain fallback"),
    ]


def test_combine_with_pdf_text_layer_keeps_pdf_text_first_for_datapoints() -> None:
    pages = combine_with_pdf_text_layer(
        [
            ParsedPage(
                page=5,
                text="| 535 | Nationalities | 143 | System sales in units |",
            )
        ],
        [
            ParsedPage(
                page=5,
                text="535\nSystem sales in units\n143\nNationalities",
            )
        ],
    )

    assert pages[0].text.startswith("535\nSystem sales in units")
    assert "--- Parsed markdown ---" in pages[0].text
    assert "| 535 | Nationalities |" in pages[0].text


def test_unknown_parser_raises() -> None:
    with pytest.raises(ValueError, match="unsupported PDF parser"):
        parse_pdf_pages(Path("report.pdf"), parser="unknown")
