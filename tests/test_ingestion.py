from __future__ import annotations

import json

from backend.app.ingestion import (
    build_chunks,
    build_datapoint_chunks,
    extract_datapoint_candidates,
    extract_fte_candidates,
    ingest_pdf,
    persist_chunks,
    persist_datapoints,
    persist_parsed_pages,
)
from backend.app.parsers import ParsedPage, ParseResult


def test_build_chunks_keeps_page_metadata() -> None:
    chunks = build_chunks(
        iter([(5, "Total employees and sustainability text.")]),
        source="asml.pdf",
        company="ASML",
        year=2025,
        max_tokens=800,
        overlap=120,
    )

    assert len(chunks) == 1
    assert chunks[0].id == "asml.pdf:5:0"
    assert chunks[0].source == "asml.pdf"
    assert chunks[0].company == "ASML"
    assert chunks[0].year == 2025
    assert chunks[0].page == 5
    assert chunks[0].chunk_kind == "section"


def test_build_chunks_splits_kpi_table_into_metric_chunks() -> None:
    chunks = build_chunks(
        iter(
            [
                (
                    5,
                    "# At a glance\n\n"
                    "| €32.7bn | Total net sales | 88% | Customer satisfaction survey score "
                    "| > 44,000 | Total employees (FTEs) |\n"
                    "| ------- | --------------- | --- | ---------------------------------- "
                    "| -------- | ---------------------- |",
                )
            ]
        ),
        source="asml.pdf",
        company="ASML",
        year=2025,
        max_tokens=800,
        overlap=120,
        parser="llamaparse",
    )

    assert [chunk.chunk_kind for chunk in chunks] == ["table", "metric", "metric", "metric"]
    assert chunks[0].text.startswith("| €32.7bn | Total net sales |")
    assert "Table type: kpi_pairs" in str(chunks[0].embedding_text)
    assert chunks[1].text == "| €32.7bn | Total net sales |"
    assert chunks[2].text == "| 88% | Customer satisfaction survey score |"
    assert chunks[3].text == "| > 44,000 | Total employees (FTEs) |"
    assert not [chunk for chunk in chunks if chunk.chunk_kind == "table_row"]
    assert chunks[3].section_path == "At a glance"
    assert "ASML" in str(chunks[3].embedding_text)
    assert "llamaparse" not in str(chunks[3].embedding_text)
    assert chunks[3].parser == "llamaparse"


def test_build_chunks_carries_section_path_across_pages() -> None:
    chunks = build_chunks(
        iter(
            [
                (10, "# Sustainability statements\n\nOpening text."),
                (11, "More climate action text without a heading."),
            ]
        ),
        source="asml.pdf",
        company="ASML",
        year=2025,
        max_tokens=800,
        overlap=120,
        parser="llamaparse",
    )

    assert chunks[0].section_path == "Sustainability statements"
    assert chunks[1].section_path == "Sustainability statements"


def test_build_chunks_formats_header_aware_table_rows() -> None:
    chunks = build_chunks(
        iter(
            [
                (
                    20,
                    "# Workforce\n\n"
                    "| Topic | Status | Notes |\n"
                    "| ----- | ------ | ----- |\n"
                    "| Training | Complete | Global rollout |",
                )
            ]
        ),
        source="asml.pdf",
        company="ASML",
        year=2025,
        max_tokens=800,
        overlap=120,
        parser="llamaparse",
    )

    row_chunks = [chunk for chunk in chunks if chunk.chunk_kind == "table_row"]
    metric_chunks = [chunk for chunk in chunks if chunk.chunk_kind == "metric"]

    assert len(row_chunks) == 1
    assert row_chunks[0].text == (
        "Topic: Training\nStatus: Complete\nNotes: Global rollout"
    )
    assert "Table type: header_table" in str(row_chunks[0].embedding_text)
    assert metric_chunks == []


def test_build_chunks_keeps_header_table_rows_without_metric_extraction() -> None:
    chunks = build_chunks(
        iter(
            [
                (
                    21,
                    "# Workforce\n\n"
                    "| Metric | 2025 | 2024 |\n"
                    "| ------ | ---- | ---- |\n"
                    "| FTE | 82,000 | 80,000 |",
                )
            ]
        ),
        source="shell.pdf",
        company="SHELL",
        year=2025,
        max_tokens=800,
        overlap=120,
        parser="llamaparse",
    )

    metric_chunks = [chunk for chunk in chunks if chunk.chunk_kind == "metric"]
    row_chunks = [chunk for chunk in chunks if chunk.chunk_kind == "table_row"]

    assert [chunk.text for chunk in metric_chunks] == [
        "Metric: FTE\nPeriod: 2025\nValue: 82,000",
        "Metric: FTE\nPeriod: 2024\nValue: 80,000",
    ]
    assert len(row_chunks) == 1
    assert row_chunks[0].text == "Metric: FTE\n2025: 82,000\n2024: 80,000"
    assert "Table type: header_table" in str(row_chunks[0].embedding_text)


def test_build_chunks_extracts_financial_metrics_from_header_table_year_columns() -> None:
    chunks = build_chunks(
        iter(
            [
                (
                    54,
                    "# Operating results\n\n"
                    "| Year ended December 31 (€, in millions) | 2024 | 2025 |\n"
                    "| --------------------------------------- | ---- | ---- |\n"
                    "| Total net sales | 28,262.9 | 32,667.3 |",
                )
            ]
        ),
        source="asml.pdf",
        company="ASML",
        year=2025,
        max_tokens=800,
        overlap=120,
        parser="llamaparse",
    )

    table_chunks = [chunk for chunk in chunks if chunk.chunk_kind == "table"]
    row_chunks = [chunk for chunk in chunks if chunk.chunk_kind == "table_row"]
    metric_chunks = [chunk for chunk in chunks if chunk.chunk_kind == "metric"]

    assert len(table_chunks) == 1
    assert len(row_chunks) == 1
    assert "Metric: Total net sales\nPeriod: 2025\nValue: 32,667.3\nUnit: €, in millions" in [
        chunk.text for chunk in metric_chunks
    ]


def test_build_chunks_keeps_generic_table_without_extra_chunks() -> None:
    chunks = build_chunks(
        iter(
            [
                (
                    22,
                    "# Notes\n\n"
                    "| Free text | More text |\n",
                )
            ]
        ),
        source="shell.pdf",
        company="SHELL",
        year=2025,
        max_tokens=800,
        overlap=120,
        parser="llamaparse",
    )

    assert [chunk.chunk_kind for chunk in chunks] == ["table"]
    assert "Table type: generic_table" in str(chunks[0].embedding_text)


def test_persist_parsed_pages_writes_inspectable_jsonl(tmp_path) -> None:
    out_path = persist_parsed_pages(
        [(5, "Total employees (FTEs): > 44,000")],
        source="asml.pdf",
        company="ASML",
        year=2025,
        parser="docling",
        processed_dir=tmp_path,
    )

    assert out_path == tmp_path / "pages" / "asml.jsonl"
    [record] = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert record["id"] == "asml.pdf:5"
    assert record["source"] == "asml.pdf"
    assert record["company"] == "ASML"
    assert record["year"] == 2025
    assert record["page"] == 5
    assert record["parser"] == "docling"
    assert record["text"] == "Total employees (FTEs): > 44,000"
    assert record["char_count"] > 0
    assert record["token_count"] > 0


def test_ingest_pdf_can_store_stable_source_name(
    tmp_path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "asml-2025-long-name.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr("backend.app.ingestion.settings.processed_dir", processed_dir)
    monkeypatch.setattr(
        "backend.app.ingestion.parse_pdf_pages",
        lambda *args, **kwargs: ParseResult(
            pages=[ParsedPage(page=5, text="Total employees (FTEs): > 44,000")],
            parser="pymupdf",
        ),
    )
    monkeypatch.setattr(
        "backend.app.ingestion.embed_texts",
        lambda texts: embedded_texts.extend(texts) or [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr("backend.app.ingestion.CHROMA_UPSERT_BATCH_SIZE", 1)

    class FakeCollection:
        def __init__(self) -> None:
            self.upsert_calls = []

        def delete(self, **kwargs) -> None:
            self.delete_kwargs = kwargs

        def upsert(self, **kwargs) -> None:
            self.upsert_calls.append(kwargs)

    embedded_texts: list[str] = []
    collection = FakeCollection()
    monkeypatch.setattr("backend.app.ingestion.get_collection", lambda reset=False: collection)

    count = ingest_pdf(
        pdf_path,
        company="ASML",
        year=2025,
        source_name="asml.pdf",
    )

    assert count == 2
    assert collection.delete_kwargs["where"] == {
        "$and": [{"source": "asml.pdf"}, {"company": "ASML"}, {"year": 2025}]
    }
    assert len(collection.upsert_calls) == 2
    assert collection.upsert_calls[0]["ids"] == ["asml.pdf:5:0"]
    assert collection.upsert_calls[0]["documents"] == ["Total employees (FTEs): > 44,000"]
    assert collection.upsert_calls[0]["embeddings"] == [[0.1, 0.2]]
    assert collection.upsert_calls[0]["metadatas"][0]["source"] == "asml.pdf"
    assert collection.upsert_calls[0]["metadatas"][0]["parser"] == "pymupdf"
    assert collection.upsert_calls[0]["metadatas"][0]["chunk_kind"] == "section"
    assert collection.upsert_calls[1]["ids"] == ["asml.pdf:5:1"]
    assert collection.upsert_calls[1]["metadatas"][0]["chunk_kind"] == "datapoint"
    assert any("fte_candidate" in text for text in embedded_texts)
    assert (processed_dir / "pages" / "asml.jsonl").exists()
    assert (processed_dir / "chunks" / "asml.jsonl").exists()


def test_extract_fte_candidates_detects_fte_keyword() -> None:
    candidates = extract_fte_candidates(
        [(5, "At a glance\n> 44,000\nTotal employees (FTEs)")],
        source="asml.pdf",
        company="ASML",
        year=2025,
        parser="llamaparse",
    )

    assert len(candidates) == 1
    assert candidates[0]["datapoint_type"] == "fte_candidate"
    assert candidates[0]["page"] == 5
    assert "Total employees (FTEs)" in str(candidates[0]["verbatim_text"])


def test_extract_fte_candidates_detects_full_time_equivalent() -> None:
    candidates = extract_fte_candidates(
        [(131, "The average number of payroll employees in full-time equivalent was 43,267.")],
        source="asml.pdf",
        company="ASML",
        year=2025,
        parser="docling",
    )

    assert len(candidates) == 1
    assert "full-time equivalent" in str(candidates[0]["verbatim_text"])


def test_persist_datapoints_writes_json(tmp_path) -> None:
    datapoints = [
        {
            "source": "asml.pdf",
            "company": "ASML",
            "year": 2025,
            "datapoint_type": "fte_candidate",
            "page": 5,
            "verbatim_text": "Total employees (FTEs): > 44,000",
            "parser": "pymupdf",
        }
    ]

    out_path = persist_datapoints(datapoints, source="asml.pdf", processed_dir=tmp_path)

    assert out_path == tmp_path / "datapoints" / "asml.json"
    assert json.loads(out_path.read_text(encoding="utf-8")) == datapoints


def test_persist_chunks_writes_debug_jsonl(tmp_path) -> None:
    [chunk] = build_chunks(
        iter([(5, "Total employees (FTEs): > 44,000")]),
        source="asml.pdf",
        company="ASML",
        year=2025,
        max_tokens=800,
        overlap=120,
        parser="llamaparse",
    )

    out_path = persist_chunks([chunk], source="asml.pdf", processed_dir=tmp_path)

    assert out_path == tmp_path / "chunks" / "asml.jsonl"
    [record] = [
        json.loads(line)
        for line in out_path.read_text(encoding="utf-8").splitlines()
    ]
    assert record["id"] == "asml.pdf:5:0"
    assert record["chunk_kind"] == "section"
    assert record["text"] == "Total employees (FTEs): > 44,000"
    assert record["embedding_text"] == chunk.embedding_text


def test_extract_datapoint_candidates_finds_sustainability_goals() -> None:
    chunks = build_chunks(
        iter(
            [
                (
                    145,
                    "# ESG sustainability at a glance\n\n"
                    "| 2026 target | 75% commitment from top-80% suppliers "
                    "to reducing CO2e footprint |\n"
                    "| ----------- | -------------------------------------"
                    "------------------------------- |",
                )
            ]
        ),
        source="asml.pdf",
        company="ASML",
        year=2025,
        max_tokens=800,
        overlap=120,
        parser="llamaparse",
    )

    datapoints = extract_datapoint_candidates(chunks, parser="llamaparse")

    assert len(datapoints) == 1
    assert datapoints[0]["datapoint_type"] == "sustainability_goal_candidate"
    assert datapoints[0]["page"] == 145
    assert "75% commitment" in str(datapoints[0]["verbatim_text"])


def test_build_datapoint_chunks_continue_page_indexes() -> None:
    existing = build_chunks(
        iter([(5, "Total employees (FTEs): > 44,000")]),
        source="asml.pdf",
        company="ASML",
        year=2025,
        max_tokens=800,
        overlap=120,
        parser="llamaparse",
    )
    datapoints = extract_datapoint_candidates(existing, parser="llamaparse")

    chunks = build_datapoint_chunks(
        datapoints,
        source="asml.pdf",
        company="ASML",
        year=2025,
        parser="llamaparse",
        existing_chunks=existing,
    )

    assert chunks[0].id == "asml.pdf:5:1"
    assert chunks[0].chunk_kind == "datapoint"
    assert "fte_candidate" in str(chunks[0].embedding_text)


def test_extract_fte_candidates_returns_empty_list_when_no_candidate_exists() -> None:
    candidates = extract_fte_candidates(
        [(1, "Revenue increased and sustainability targets were discussed.")],
        source="asml.pdf",
        company="ASML",
        year=2025,
        parser="docling",
    )

    assert candidates == []
