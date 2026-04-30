from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

Category = Literal[
    "pre_extracted_fact",
    "verbatim_financial",
    "operational_datapoint",
    "esg_datapoint",
    "named_program",
    "sustainability_target",
    "multi_document",
    "hallucination_check",
]
Difficulty = Literal["easy", "medium", "hard"]


class Citation(BaseModel):
    source: str
    page: int = Field(ge=1)
    quote: str


EvidenceType = Literal["exact_quote", "table_value", "datapoint"]


class EvidenceItem(BaseModel):
    evidence_type: EvidenceType
    source: str
    page: int = Field(ge=1)
    quote: str | None = None
    table_title: str | None = None
    metric: str | None = None
    period: str | None = None
    value: str | None = None
    datapoint_type: str | None = None

    @model_validator(mode="after")
    def _check_required_fields(self) -> EvidenceItem:
        if self.evidence_type == "exact_quote":
            if not self.quote:
                raise ValueError("exact_quote evidence requires quote")
            return self
        if self.evidence_type == "table_value":
            if not self.value:
                raise ValueError("table_value evidence requires value")
            if not (self.metric or self.period or self.table_title):
                raise ValueError(
                    "table_value evidence requires metric, period, or table_title"
                )
            return self
        if self.evidence_type == "datapoint":
            if not self.value:
                raise ValueError("datapoint evidence requires value")
            if not (self.metric or self.datapoint_type):
                raise ValueError("datapoint evidence requires metric or datapoint_type")
            return self
        return self


class Chunk(BaseModel):
    id: str
    source: str
    company: str | None = None
    year: int | None = None
    page: int = Field(ge=1)
    text: str
    token_count: int = Field(ge=0)
    parser: str | None = None
    chunk_kind: str | None = None
    section_path: str | None = None
    embedding_text: str | None = None


class RetrievedChunk(Chunk):
    score: float


class GroundingDrop(BaseModel):
    source: str
    page: int
    quote: str
    reason: Literal[
        "source_page_not_in_retrieved_set",
        "empty_quote",
        "quote_not_found_verbatim",
    ]


class VerbatimAnswer(BaseModel):
    question: str
    answer: str
    verbatim: str | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    raw_evidence: list[EvidenceItem] = Field(default_factory=list)
    raw_citations: list[Citation] = Field(default_factory=list)
    grounding_drops: list[GroundingDrop] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_citations(self) -> VerbatimAnswer:
        if not self.refused and not self.citations:
            raise ValueError("non-refused answers must include at least one citation")
        if self.refused and self.refusal_reason is None:
            raise ValueError("refused answers must include refusal_reason")
        return self


class LLMAnswer(BaseModel):
    answer: str
    verbatim: str | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None


class RetrievalQuery(BaseModel):
    question: str
    top_k: int = Field(ge=1)
    company: str | None = None
    year: int | None = None


class RetrievalResult(BaseModel):
    query: RetrievalQuery
    chunks: list[RetrievedChunk]


class EvalQuestion(BaseModel):
    id: str
    question: str
    category: Category
    difficulty: Difficulty
    expected_answer_contains_any: list[str] | None = None
    expected_answer_contains_all: list[str] | None = None
    expected_page: int | None = None
    expected_source: str | None = None
    expected_behavior: Literal["refuse"] | None = None
    notes: str | None = None


class EvalSet(BaseModel):
    source: str
    company: str
    year: int
    questions: list[EvalQuestion]
