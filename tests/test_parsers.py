from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.parsers import (
    ParsedPage,
    ParserUnavailableError,
    combine_with_pdf_text_layer,
    llamaparse_json_to_pages,
    parse_pdf_pages,
    persist_llamaparse_artifacts,
)


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
        llama_cloud_api_key="llx-key",
    )

    assert result.parser == "llamaparse"
    assert result.pages == [ParsedPage(page=3, text="llamaparse markdown")]


def test_llamaparse_parser_strict_mode_requires_api_key() -> None:
    with pytest.raises(ParserUnavailableError, match="LLAMA_CLOUD_API_KEY"):
        parse_pdf_pages(Path("report.pdf"), parser="llamaparse")


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
