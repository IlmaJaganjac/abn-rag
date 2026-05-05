from __future__ import annotations

import json

from backend.app.extract.datapoints import NormalizedDatapoint
from backend.app.ingestion import (
    build_datapoint_chunks,
    build_chunks,
    extract_categorized_datapoints,
    ingest_pdf,
    persist_chunks,
    persist_datapoints,
    persist_parsed_pages,
)
from backend.app.extract.schemas import AnnualReportDatapoints
from backend.app.extract.openai import ValidationItem
from backend.app.ingest.parsers import ParsedPage, ParseResult


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
        "backend.app.ingestion.embed_texts",
        lambda texts: embedded_texts.extend(texts) or [[0.1, 0.2] for _ in texts],
    )
    monkeypatch.setattr(
        "backend.app.ingestion.extract_categorized_datapoints",
        lambda pages, source, company, year, validate=False, page_sections=None: [
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
    monkeypatch.setattr("backend.app.db.upsert_datapoints", lambda records: None)

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
    assert collection.upsert_calls[1]["ids"] == ["asml.pdf:5:1"]
    assert collection.upsert_calls[1]["documents"] == [
        "Metric: Total employees (FTEs)\n"
        "Type: fte\n"
        "Value: > 44,000\n"
        "Quote: Total employees (FTEs): > 44,000"
    ]
    assert embedded_texts == [
        "ASML\n2025\nTotal employees (FTEs): > 44,000",
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


def test_persist_datapoints_writes_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("backend.app.db.upsert_datapoints", lambda records: None)
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


def test_persist_datapoints_writes_model_dump_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("backend.app.db.upsert_datapoints", lambda records: None)
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
        "backend.app.extract.categorize.extract_annual_report_datapoints_openai",
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
        "backend.app.extract.categorize.extract_annual_report_datapoints_openai",
        fake_extract,
    )
    monkeypatch.setattr(
        "backend.app.extract.categorize.validate_datapoints_openai",
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
