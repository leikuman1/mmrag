from __future__ import annotations

import re

from mmrag.models import SearchHit
from mmrag.storage.catalog import CatalogStore
from mmrag.storage.vector_store import InMemoryVectorStore, QdrantRestVectorStore


def infer_source_types(question: str) -> list[str] | None:
    lowered = question.lower()
    selected: list[str] = []
    if any(token in lowered for token in ("readme", "docs", "documentation", "install", "quickstart")):
        selected.append("doc")
    if any(token in lowered for token in ("issue", "bug", "error", "incident")):
        selected.append("issue")
    if any(token in lowered for token in ("pr", "pull request", "review", "merged")):
        selected.append("pr")
    return selected or None


def extract_reference_numbers(question: str) -> list[int]:
    matches = re.findall(r"(?:#|issue\s+#?|pr\s+#?|pull request\s+#?)(\d+)", question.lower())
    seen: list[int] = []
    for value in matches:
        number = int(value)
        if number not in seen:
            seen.append(number)
    return seen


class RetrievalService:
    def __init__(
        self,
        catalog: CatalogStore,
        vector_store: QdrantRestVectorStore | InMemoryVectorStore,
        embedder,
    ) -> None:
        self.catalog = catalog
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        repo: str,
        query: str,
        top_k: int = 6,
        source_types: list[str] | None = None,
    ) -> list[SearchHit]:
        inferred_types = source_types or infer_source_types(query)
        numbers = extract_reference_numbers(query)
        combined: dict[str, SearchHit] = {}
        if self.embedder is not None:
            vectors = self.embedder.embed([query])
            if vectors and vectors[0]:
                for hit in self.vector_store.search(repo, vectors[0], top_k, inferred_types, numbers):
                    combined[hit.chunk.id] = hit
        keyword_hits = self.catalog.keyword_search(repo, query, top_k, inferred_types, numbers)
        for hit in keyword_hits:
            existing = combined.get(hit.chunk.id)
            if existing is None or hit.score > existing.score:
                combined[hit.chunk.id] = hit
        ranked = sorted(combined.values(), key=lambda hit: hit.score, reverse=True)
        return ranked[:top_k]

