from __future__ import annotations

import json
from typing import Sequence
from urllib import error, request

from mmrag.config import ConfigurationError, EndpointConfig
from mmrag.model_providers.base import ChatGeneration, ChatMessage


class OpenAICompatibleClient:
    def __init__(self, endpoint: EndpointConfig, timeout: float = 30.0) -> None:
        if not endpoint.is_configured:
            raise ConfigurationError("Endpoint configuration is incomplete.")
        self.endpoint = endpoint
        self.timeout = timeout

    def _build_url(self, suffix: str) -> str:
        return f"{self.endpoint.base_url.rstrip('/')}/{suffix.lstrip('/')}"

    def _request_json(self, suffix: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self._build_url(suffix),
            method="POST",
            data=body,
            headers={
                "Authorization": f"Bearer {self.endpoint.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ConfigurationError(
                f"OpenAI-compatible request failed with status {exc.code}: {detail}"
            ) from exc
        except error.URLError as exc:
            raise ConfigurationError(
                f"Could not reach model endpoint {self.endpoint.base_url}: {exc.reason}"
            ) from exc


class OpenAICompatibleChatProvider(OpenAICompatibleClient):
    def generate(
        self,
        messages: Sequence[ChatMessage],
        tools: list[dict] | None = None,
        stream: bool = False,
    ) -> ChatGeneration:
        payload: dict = {
            "model": self.endpoint.model,
            "messages": [{"role": message.role, "content": message.content} for message in messages],
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        data = self._request_json("chat/completions", payload)
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        usage = data.get("usage") or {}
        return ChatGeneration(
            text=message.get("content", "").strip(),
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            raw=data,
        )


class OpenAICompatibleEmbeddingProvider(OpenAICompatibleClient):
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        payload = {
            "model": self.endpoint.model,
            "input": list(texts),
        }
        data = self._request_json("embeddings", payload)
        rows = data.get("data") or []
        ordered = sorted(rows, key=lambda row: int(row.get("index", 0)))
        return [list(row.get("embedding") or []) for row in ordered]

