from __future__ import annotations

from pydantic import BaseModel, Field


class IngestionRequest(BaseModel):
    repo: str
    include: list[str] = Field(default_factory=lambda: ["docs", "issues", "prs"])


class ChatRequest(BaseModel):
    repo: str
    question: str
    session_id: str | None = None


class EvalRequest(BaseModel):
    repo: str
    suite_name: str = "demo"

