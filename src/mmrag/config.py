from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigurationError(RuntimeError):
    """Raised when required user configuration is missing or invalid."""


@dataclass(slots=True)
class EndpointConfig:
    base_url: str
    api_key: str
    model: str

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)


@dataclass(slots=True)
class AppConfig:
    github_token: str | None
    llm: EndpointConfig | None
    embedding: EndpointConfig | None
    qdrant_url: str
    qdrant_collection: str
    sqlite_path: Path
    host: str
    port: int
    default_demo_repo: str
    request_timeout: float

    @classmethod
    def from_env(cls) -> "AppConfig":
        sqlite_path = Path(os.getenv("SQLITE_PATH", "data/mmrag.sqlite3"))

        def endpoint(prefix: str) -> EndpointConfig | None:
            base_url = os.getenv(f"{prefix}_BASE_URL", "").strip()
            api_key = os.getenv(f"{prefix}_API_KEY", "").strip()
            model = os.getenv(f"{prefix}_MODEL", "").strip()
            if not any([base_url, api_key, model]):
                return None
            return EndpointConfig(base_url=base_url, api_key=api_key, model=model)

        return cls(
            github_token=os.getenv("GITHUB_TOKEN") or None,
            llm=endpoint("LLM"),
            embedding=endpoint("EMBEDDING"),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333").rstrip("/"),
            qdrant_collection=os.getenv("QDRANT_COLLECTION", "mmrag_chunks"),
            sqlite_path=sqlite_path,
            host=os.getenv("MMRAG_HOST", "0.0.0.0"),
            port=int(os.getenv("MMRAG_PORT", "8000")),
            default_demo_repo=os.getenv("MMRAG_DEFAULT_DEMO_REPO", "langchain-ai/langgraph"),
            request_timeout=float(os.getenv("MMRAG_REQUEST_TIMEOUT", "30")),
        )

    def ensure_storage_paths(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

