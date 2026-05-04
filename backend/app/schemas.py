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
    """A verbatim evidence span tied to one source file and one PDF page."""
    source: str
    page: int = Field(ge=1)
    quote: str


class Chunk(BaseModel):
    """Stored retrieval unit created from report text or pre-extracted datapoints."""
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
    fact_kind: str | None = None
    basis: str | None = None
    scope_type: str | None = None
    quality: str | None = None
    validation_status: str | None = None
    canonical_metric: str | None = None


class RetrievedChunk(Chunk):
    """A chunk returned by retrieval with an attached ranking score."""
    score: float


class VerbatimAnswer(BaseModel):
    """Final answer returned to the UI or CLI, including citations or refusal metadata."""
    question: str
    answer: str
    verbatim: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None

    @model_validator(mode="after")
    def _check_citations(self) -> VerbatimAnswer:
        """Enforce that grounded answers cite sources and refusals explain themselves."""
        if not self.refused and not self.citations:
            raise ValueError("non-refused answers must include at least one citation")
        if self.refused and self.refusal_reason is None:
            raise ValueError("refused answers must include refusal_reason")
        return self


class LLMAnswer(BaseModel):
    """Raw model output before citation grounding is enforced."""
    answer: str
    verbatim: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None


class RetrievalQuery(BaseModel):
    """Input filters and question text for one retrieval request."""
    question: str
    top_k: int = Field(ge=1)
    company: str | None = None
    year: int | None = None


class RetrievalResult(BaseModel):
    """Ranked retrieval output containing the effective query and selected chunks."""
    query: RetrievalQuery
    chunks: list[RetrievedChunk]


class EvalQuestion(BaseModel):
    """One eval-set question with expected behavior and optional answer constraints."""
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
    """Top-level eval configuration containing multiple `EvalQuestion` entries."""
    source: str
    company: str
    year: int
    questions: list[EvalQuestion]
