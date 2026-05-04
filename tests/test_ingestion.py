from __future__ import annotations

import json

from backend.app.extracted_datapoints import NormalizedDatapoint
from backend.app.ingestion import (
    build_datapoint_chunks,
    build_chunks,
    extract_categorized_datapoints,
    extract_datapoint_candidates,
    extract_fte_candidates,
    ingest_pdf,
    persist_chunks,
    persist_datapoints,
    persist_parsed_pages,
)
from backend.app.datapoint_schemas import AnnualReportDatapoints
from backend.app.openai_validate_datapoints import ValidationItem
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


def _skip_test_build_chunks_splits_kpi_table_into_metric_chunks() -> None:
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
    assert chunks[1].text == (
        "Metric: Total net sales\n"
        "Period: 2025\n"
        "Value: €32.7bn\n"
        "Unit: EUR billion\n"
        "Presentation: highlight"
    )
    assert chunks[2].text == (
        "Metric: Customer satisfaction survey score\n"
        "Period: 2025\n"
        "Value: 88%\n"
        "Unit: %\n"
        "Presentation: highlight"
    )
    assert chunks[3].text == (
        "Metric: Total employees (FTEs)\n"
        "Period: 2025\n"
        "Value: > 44,000\n"
        "Unit: FTEs\n"
        "Presentation: highlight"
    )
    assert not [chunk for chunk in chunks if chunk.chunk_kind == "table_row"]
    assert chunks[3].section_path == "At a glance"
    embedding = str(chunks[3].embedding_text)
    assert "ASML" in embedding
    assert "Type: KPI highlight" in embedding
    assert "Metric: Total employees (FTEs)" in embedding
    assert "Retrieval hints:" in embedding
    assert "headcount" in embedding
    assert "llamaparse" not in embedding
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


def test_build_chunks_treats_sentence_like_hash_lines_as_narrative() -> None:
    chunks = build_chunks(
        iter(
            [
                (
                    30,
                    "# ESG update\n\n"
                    "# We aim to be greenhouse gas neutral across our value chain by 2040.\n\n"
                    "Progress is tracked annually.",
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

    assert [chunk.chunk_kind for chunk in chunks] == ["section"]
    assert chunks[0].section_path == "ESG update"
    assert chunks[0].text == (
        "We aim to be greenhouse gas neutral across our value chain by 2040.\n\n"
        "Progress is tracked annually."
    )


def test_build_chunks_filters_read_more_navigation_lines() -> None:
    chunks = build_chunks(
        iter(
            [
                (
                    31,
                    "# Climate Transition Plan\n\n"
                    "Our Climate Transition Plan is our strategic roadmap.\n\n"
                    "Read more on page 155 >",
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

    assert len(chunks) == 1
    assert chunks[0].text == "Our Climate Transition Plan is our strategic roadmap."


def test_build_chunks_keeps_two_column_metric_table_as_generic_table() -> None:
    chunks = build_chunks(
        iter(
            [
                (
                    32,
                    "# Empowered colleagues\n\n"
                    "| Net scope 1 and 2 CO₂e emissions | 11.5 Mt |\n"
                    "| -------------------------------- | ------- |\n"
                    "| Net scope 3 CO₂e emissions       | 2       |",
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

    assert [chunk.chunk_kind for chunk in chunks] == ["table"]
    assert "Table type: generic_table" in str(chunks[0].embedding_text)


def test_build_chunks_keeps_layout_table_without_table_row_explosion() -> None:
    chunks = build_chunks(
        iter(
            [
                (
                    33,
                    "# Our commitment to sustainability\n\n"
                    "| Global scale | Asia | EMEA | North America |\n"
                    "| ------------ | ---- | ---- | ------------- |\n"
                    "| China | Japan | Belgium | Arizona |\n"
                    "| Malaysia | Germany | California | E |\n",
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

    assert [chunk.chunk_kind for chunk in chunks] == ["table"]
    assert "Table type: generic_table" in str(chunks[0].embedding_text)


def _skip_test_build_chunks_keeps_header_table_rows_without_metric_extraction() -> None:
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


def _skip_test_build_chunks_extracts_financial_metrics_from_header_table_year_columns() -> None:
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
        parser="llamaparse",
        processed_dir=tmp_path,
    )

    assert out_path == tmp_path / "pages" / "asml.jsonl"
    [record] = [json.loads(line) for line in out_path.read_text(encoding="utf-8").splitlines()]
    assert record["id"] == "asml.pdf:5"
    assert record["source"] == "asml.pdf"
    assert record["company"] == "ASML"
    assert record["year"] == 2025
    assert record["page"] == 5
    assert record["parser"] == "llamaparse"
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
            parser="llamaparse",
        ),
    )
    monkeypatch.setattr(
        "backend.app.ingestion.enhance_pages_with_vision",
        lambda **kwargs: [(5, "Enhanced page text")],
    )
    monkeypatch.setattr(
        "backend.app.ingestion.embed_texts",
        lambda texts: embedded_texts.extend(texts) or [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        "backend.app.ingestion.extract_categorized_datapoints",
        lambda pages, source, company, year, validate=False: [
            NormalizedDatapoint(
                source=source,
                company=company,
                year=year,
                datapoint_type="fte",
                metric="Total employees (FTEs)",
                value="> 44,000",
                page=5,
                quote="Total employees (FTEs): > 44,000",
                extractor="openai",
                priority=100,
            )
        ],
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
    assert collection.upsert_calls[0]["documents"] == ["Enhanced page text"]
    assert collection.upsert_calls[0]["embeddings"] == [[0.1, 0.2]]
    assert collection.upsert_calls[1]["ids"] == ["asml.pdf:5:1"]
    assert collection.upsert_calls[1]["documents"] == [
        "Metric: Total employees (FTEs)\n"
        "Type: fte\n"
        "Value: > 44,000\n"
        "Quote: Total employees (FTEs): > 44,000"
    ]
    assert embedded_texts == [
        "ASML\n2025\nEnhanced page text",
        "ASML\n2025\nllamaparse\ndatapoint\nfte\n"
        "Metric: Total employees (FTEs)\n"
        "Type: fte\n"
        "Value: > 44,000\n"
        "Quote: Total employees (FTEs): > 44,000",
    ]
    assert collection.upsert_calls[0]["metadatas"][0]["source"] == "asml.pdf"
    assert collection.upsert_calls[0]["metadatas"][0]["parser"] == "llamaparse"
    assert collection.upsert_calls[0]["metadatas"][0]["chunk_kind"] == "section"
    assert collection.upsert_calls[1]["metadatas"][0]["chunk_kind"] == "extracted_datapoint"
    assert (processed_dir / "pages" / "asml.jsonl").exists()
    assert (processed_dir / "datapoints" / "asml.json").exists()
    assert (processed_dir / "chunks" / "asml.jsonl").exists()
    records = [
        json.loads(line)
        for line in (processed_dir / "chunks" / "asml.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [record["chunk_kind"] for record in records] == ["section", "extracted_datapoint"]
    assert "Metric: Total employees (FTEs)" in records[1]["text"]
    assert "Value: > 44,000" in records[1]["text"]
    assert "Quote: Total employees (FTEs): > 44,000" in records[1]["text"]


def test_ingest_pdf_extracts_datapoints_only_when_flag_enabled(
    tmp_path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "asml-2025.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr("backend.app.ingestion.settings.processed_dir", processed_dir)
    monkeypatch.setattr(
        "backend.app.ingestion.parse_pdf_pages",
        lambda *args, **kwargs: ParseResult(
            pages=[ParsedPage(page=5, text="Total employees (FTEs): > 44,000")],
            parser="llamaparse",
        ),
    )
    monkeypatch.setattr(
        "backend.app.ingestion.enhance_pages_with_vision",
        lambda **kwargs: [(5, "Enhanced page text")],
    )
    monkeypatch.setattr(
        "backend.app.ingestion.embed_texts",
        lambda texts: [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        "backend.app.ingestion.extract_categorized_datapoints",
        lambda pages, source, company, year, validate=False: [
            NormalizedDatapoint(
                source=source,
                company=company,
                year=year,
                datapoint_type="fte",
                metric="Total employees (FTEs)",
                value="> 44,000",
                page=5,
                quote="Total employees (FTEs): > 44,000",
                extractor="openai",
                priority=100,
            )
        ],
    )

    class FakeCollection:
        def delete(self, **kwargs) -> None:
            self.delete_kwargs = kwargs

        def upsert(self, **kwargs) -> None:
            self.upsert_kwargs = kwargs

    monkeypatch.setattr("backend.app.ingestion.get_collection", lambda reset=False: FakeCollection())

    ingest_pdf(
        pdf_path,
        company="ASML",
        year=2025,
        extract_datapoints=False,
    )
    assert not (processed_dir / "datapoints" / "asml-2025.json").exists()

    ingest_pdf(
        pdf_path,
        company="ASML",
        year=2025,
        extract_datapoints=True,
    )
    assert (processed_dir / "datapoints" / "asml-2025.json").exists()


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
        parser="llamaparse",
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
            "parser": "llamaparse",
        }
    ]

    out_path = persist_datapoints(datapoints, source="asml.pdf", processed_dir=tmp_path)

    assert out_path == tmp_path / "datapoints" / "asml.json"
    assert json.loads(out_path.read_text(encoding="utf-8")) == datapoints


def test_persist_datapoints_writes_model_dump_json(tmp_path) -> None:
    datapoints = [
        NormalizedDatapoint(
            source="asml.pdf",
            company="ASML",
            year=2025,
            datapoint_type="fte",
            metric="Total employees (FTEs)",
            value="> 44,000",
            page=5,
            quote="Total employees (FTEs): > 44,000",
            extractor="openai",
            priority=100,
        )
    ]

    out_path = persist_datapoints(datapoints, source="asml.pdf", processed_dir=tmp_path)

    [record] = json.loads(out_path.read_text(encoding="utf-8"))
    assert record["datapoint_type"] == "fte"
    assert record["metric"] == "Total employees (FTEs)"
    assert record["extractor"] == "openai"


def test_extract_categorized_datapoints_uses_all_categories(monkeypatch) -> None:
    calls: list[str] = []

    def fake_extract(*, pages, company, year, category):
        calls.append(category)
        payload = {
            "company": company,
            "year": year,
            "fte_datapoints": [],
            "sustainability_goals": [],
            "esg_datapoints": [],
            "financial_highlights": [],
            "business_performance": [],
            "shareholder_returns": [],
        }
        if category == "fte":
            payload["fte_datapoints"] = [
                {
                    "label": "Total employees (FTEs)",
                    "value": "> 44,000",
                    "quote": "Total employees (FTEs): > 44,000",
                    "fact_kind": "actual",
                    "scope_type": "company_wide",
                    "page": 5,
                }
            ]
        return AnnualReportDatapoints(**payload)

    monkeypatch.setattr(
        "backend.app.ingestion.extract_annual_report_datapoints_openai",
        fake_extract,
    )

    datapoints = extract_categorized_datapoints(
        [(5, "Total employees (FTEs): > 44,000")],
        source="asml.pdf",
        company="ASML",
        year=2025,
    )

    assert sorted(calls) == sorted([
        "fte",
        "sustainability",
        "esg",
        "financial_highlight",
        "business_performance",
        "shareholder_return",
    ])
    assert len(datapoints) == 1
    assert datapoints[0].datapoint_type == "fte"
    assert datapoints[0].extractor == "openai"


def test_extract_categorized_datapoints_validates_and_deduplicates(monkeypatch) -> None:
    def fake_extract(*, pages, company, year, category):
        payload = {
            "company": company,
            "year": year,
            "fte_datapoints": [],
            "sustainability_goals": [],
            "esg_datapoints": [],
            "financial_highlights": [],
            "business_performance": [],
            "shareholder_returns": [],
        }
        if category == "fte":
            payload["fte_datapoints"] = [
                {
                    "label": "Total employees (FTEs)",
                    "value": "20,455",
                    "unit": "FTEs",
                    "quote": "Internal employees | 20,455",
                    "fact_kind": "actual",
                    "scope_type": "company_wide",
                    "page": 14,
                },
                {
                    "label": "Total employees (FTEs)",
                    "value": "20,455",
                    "unit": "FTEs",
                    "quote": "Internal employees | 20,455",
                    "fact_kind": "actual",
                    "scope_type": "company_wide",
                    "page": 14,
                },
            ]
        return AnnualReportDatapoints(**payload)

    monkeypatch.setattr(
        "backend.app.ingestion.extract_annual_report_datapoints_openai",
        fake_extract,
    )
    monkeypatch.setattr(
        "backend.app.ingestion.validate_datapoints_openai",
        lambda **kwargs: [
            ValidationItem(index=0, is_valid=True, reason="valid"),
            ValidationItem(index=1, is_valid=False, reason="duplicate", duplicate_of_index=0),
        ],
    )

    datapoints = extract_categorized_datapoints(
        [(14, "Internal employees | 20,455 FTEs")],
        source="abn-amro.pdf",
        company="ABN AMRO",
        year=2025,
        validate=True,
    )

    assert len(datapoints) == 1
    assert datapoints[0].validation_status == "valid"


def test_ingest_pdf_can_skip_vision_enhancement(
    tmp_path,
    monkeypatch,
) -> None:
    pdf_path = tmp_path / "asml.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    processed_dir = tmp_path / "processed"

    monkeypatch.setattr("backend.app.ingestion.settings.processed_dir", processed_dir)
    monkeypatch.setattr(
        "backend.app.ingestion.parse_pdf_pages",
        lambda *args, **kwargs: ParseResult(
            pages=[ParsedPage(page=5, text="Original page text")],
            parser="llamaparse",
        ),
    )
    called = {"vision": False}
    def fake_enhance(**kwargs):
        called["vision"] = True
        return [(5, "Enhanced page text")]
    monkeypatch.setattr("backend.app.ingestion.enhance_pages_with_vision", fake_enhance)
    monkeypatch.setattr("backend.app.ingestion.extract_categorized_datapoints", lambda *args, **kwargs: [])
    monkeypatch.setattr("backend.app.ingestion.embed_texts", lambda texts: [[0.1, 0.2] for _ in texts])

    class FakeCollection:
        def delete(self, **kwargs) -> None:
            self.delete_kwargs = kwargs
        def upsert(self, **kwargs) -> None:
            self.upsert_kwargs = kwargs

    collection = FakeCollection()
    monkeypatch.setattr("backend.app.ingestion.get_collection", lambda reset=False: collection)

    ingest_pdf(
        pdf_path,
        company="ASML",
        year=2025,
        enhance_vision=False,
    )

    assert called["vision"] is False
    assert collection.upsert_kwargs["documents"] == ["Original page text"]


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


def test_build_datapoint_chunks_include_structured_fact_text() -> None:
    existing = build_chunks(
        iter([(5, "Annual report page text")]),
        source="asml.pdf",
        company="ASML",
        year=2025,
        max_tokens=800,
        overlap=120,
        parser="llamaparse",
    )

    [chunk] = build_datapoint_chunks(
        [
            {
                "source": "asml.pdf",
                "company": "ASML",
                "year": 2025,
                "datapoint_type": "fte",
                "metric": "Total employees",
                "value": "> 44,000",
                "unit": "FTEs",
                "period": "2025",
                "page": 5,
                "quote": "Total employees (FTEs): > 44,000",
                "basis": "FTE",
                "fact_kind": "actual",
                "scope_type": "company_wide",
                "canonical_metric": "total_employees",
            }
        ],
        source="asml.pdf",
        company="ASML",
        year=2025,
        parser="llamaparse",
        existing_chunks=existing,
    )

    assert chunk.id == "asml.pdf:5:1"
    assert chunk.chunk_kind == "extracted_datapoint"
    assert chunk.section_path == "fte"
    assert "Metric: Total employees" in chunk.text
    assert "Value: > 44,000 FTEs" in chunk.text
    assert "Quote: Total employees (FTEs): > 44,000" in chunk.text
    assert chunk.fact_kind == "actual"
    assert chunk.scope_type == "company_wide"
    assert chunk.canonical_metric == "total_employees"


def _skip_test_extract_datapoint_candidates_finds_sustainability_goals() -> None:  # disabled: table-based candidate detection removed
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


def _skip_test_build_datapoint_chunks_continue_page_indexes() -> None:  # disabled: build_datapoint_chunks removed from pipeline
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


def _disabled_kpi_metric_chunks(rows: str, *, year: int = 2025):  # disabled: kpi metric chunks removed
    return [
        chunk
        for chunk in build_chunks(
            iter([(7, "# At a glance\n\n" + rows)]),
            source="report.pdf",
            company="ASML",
            year=year,
            max_tokens=800,
            overlap=120,
            parser="llamaparse",
        )
        if chunk.chunk_kind == "metric"
    ]


def _skip_kpi_pair_rd_investment_normalized() -> None:
    [chunk] = _kpi_metric_chunks(
        "| €4.7bn | R&D investment |\n| ------ | -------------- |"
    )
    assert chunk.text == (
        "Metric: R&D investment\n"
        "Period: 2025\n"
        "Value: €4.7bn\n"
        "Unit: EUR billion\n"
        "Presentation: highlight"
    )
    embedding = str(chunk.embedding_text)
    assert "Type: KPI highlight" in embedding
    assert "research and development spend" in embedding


def _skip_kpi_pair_gross_margin_normalized() -> None:
    [chunk] = _kpi_metric_chunks(
        "| 52.8% | Gross margin |\n| ----- | ------------ |"
    )
    assert "Metric: Gross margin" in chunk.text
    assert "Value: 52.8%" in chunk.text
    assert "Unit: %" in chunk.text
    embedding = str(chunk.embedding_text)
    assert "margin" in embedding
    assert "percentage" in embedding


def _skip_kpi_pair_returned_to_shareholders_has_hints() -> None:
    [chunk] = _kpi_metric_chunks(
        "| €8.5bn | Returned to shareholders |\n| ------ | ----------------------- |"
    )
    assert "Metric: Returned to shareholders" in chunk.text
    assert "Value: €8.5bn" in chunk.text
    assert "Unit: EUR billion" in chunk.text
    embedding = str(chunk.embedding_text)
    assert "shareholder distributions" in embedding
    assert "dividends" in embedding


def _skip_kpi_pair_total_net_sales_has_revenue_hints() -> None:
    [chunk] = _kpi_metric_chunks(
        "| €32.7bn | Total net sales |\n| ------- | --------------- |"
    )
    embedding = str(chunk.embedding_text)
    assert "net sales" in embedding
    assert "revenue" in embedding


def _skip_kpi_pair_system_sales_units() -> None:
    [chunk] = _kpi_metric_chunks(
        "| 535 | System sales in units |\n| --- | -------------------- |"
    )
    assert "Metric: System sales in units" in chunk.text
    assert "Value: 535" in chunk.text
    embedding = str(chunk.embedding_text)
    assert "net sales" not in embedding
    assert "revenue" not in embedding


def _skip_kpi_pair_scope3_emissions_unit_mt() -> None:
    [chunk] = _kpi_metric_chunks(
        "| 11.5 Mt | Scope 3 emissions |\n| ------- | ----------------- |"
    )
    assert "Unit: Mt" in chunk.text
    assert "Metric: Scope 3 emissions" in chunk.text
    embedding = str(chunk.embedding_text)
    assert "greenhouse gas" in embedding
    assert "GHG" in embedding


def _skip_kpi_pair_scope1_2_emissions_unit_kt() -> None:
    [chunk] = _kpi_metric_chunks(
        "| 26 kt | Scope 1 and 2 emissions |\n| ----- | ----------------------- |"
    )
    assert "Unit: kt" in chunk.text
    assert "Metric: Scope 1 and 2 emissions" in chunk.text


def _skip_kpi_pair_two_values_does_not_create_metric() -> None:
    metrics = _kpi_metric_chunks(
        "| 12.3% | 4.5% |\n| ----- | ---- |"
    )
    assert metrics == []


def _skip_kpi_pair_two_labels_does_not_create_metric() -> None:
    metrics = _kpi_metric_chunks(
        "| Strategic priorities | Stakeholder engagement |\n"
        "| -------------------- | ---------------------- |"
    )
    assert metrics == []


def _skip_kpi_pair_vague_label_does_not_create_metric() -> None:
    metrics = _kpi_metric_chunks(
        "| €1.2bn | Read more |\n| ------ | --------- |\n"
        "| 99% | continued |"
    )
    assert metrics == []


def _skip_kpi_pair_skips_hints_when_no_category_matches() -> None:
    [chunk] = _kpi_metric_chunks(
        "| 17 | Patents granted |\n| -- | --------------- |"
    )
    assert "Metric: Patents granted" in chunk.text
    embedding = str(chunk.embedding_text)
    assert "Retrieval hints:" not in embedding


def test_extract_fte_candidates_returns_empty_list_when_no_candidate_exists() -> None:
    candidates = extract_fte_candidates(
        [(1, "Revenue increased and sustainability targets were discussed.")],
        source="asml.pdf",
        company="ASML",
        year=2025,
        parser="llamaparse",
    )

    assert candidates == []
