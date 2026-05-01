from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence


@dataclass(slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(slots=True)
class ChatGeneration:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ChatModelProvider(Protocol):
    def generate(
        self,
        messages: Sequence[ChatMessage],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> ChatGeneration:
        ...


class EmbeddingProvider(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...

