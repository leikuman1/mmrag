from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


SourceType = Literal["doc", "issue", "pr"]


@dataclass(slots=True)
class SourceDocument:
    id: str
    repo: str
    source_type: SourceType
    title: str
    url: str
    text: str
    updated_at: str
    author: str | None = None
    labels: list[str] = field(default_factory=list)
    number: int | None = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentChunk:
    id: str
    repo: str
    source_id: str
    source_type: SourceType
    chunk_index: int
    title: str
    url: str
    text: str
    snippet: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SearchHit:
    chunk: DocumentChunk
    score: float


@dataclass(slots=True)
class Citation:
    source_type: SourceType
    title: str
    url: str
    snippet: str


@dataclass(slots=True)
class AgentStepTrace:
    step: str
    agent: str
    input_summary: str
    output_summary: str
    latency_ms: float
    token_usage: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AnswerResponse:
    answer: str
    citations: list[Citation]
    confidence: float
    trace_id: str
    follow_up_question: str | None = None
    grounded: bool = True
    needs_more_context: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [asdict(citation) for citation in self.citations],
            "confidence": self.confidence,
            "trace_id": self.trace_id,
            "follow_up_question": self.follow_up_question,
            "grounded": self.grounded,
            "needs_more_context": self.needs_more_context,
        }


@dataclass(slots=True)
class IngestionResult:
    repo: str
    documents_indexed: int
    chunks_indexed: int
    skipped_documents: int
    warnings: list[str]
    started_at: str
    completed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalCase:
    case_id: str
    question: str
    expected_keywords: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvalCaseResult:
    case_id: str
    question: str
    passed: bool
    citation_valid: bool
    helpful: bool
    grounded: bool
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvalResult:
    suite_name: str
    total_cases: int
    helpfulness: float
    citation_validity: float
    grounded_pass_rate: float
    cases: list[EvalCaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "total_cases": self.total_cases,
            "helpfulness": self.helpfulness,
            "citation_validity": self.citation_validity,
            "grounded_pass_rate": self.grounded_pass_rate,
            "cases": [case.to_dict() for case in self.cases],
        }
