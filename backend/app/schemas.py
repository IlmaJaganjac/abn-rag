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
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
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
    accepted_pages: list[int] | None = None
    expected_source: str | None = None
    expected_behavior: Literal["refuse"] | None = None
    notes: str | None = None


class EvalSet(BaseModel):
    source: str
    company: str
    year: int
    questions: list[EvalQuestion]
