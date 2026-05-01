from __future__ import annotations

import json
import math
from typing import Any
from urllib import error, request

from mmrag.config import ConfigurationError
from mmrag.models import DocumentChunk, SearchHit


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._chunks: dict[str, DocumentChunk] = {}

    def ensure_collection(self, vector_size: int) -> None:
        _ = vector_size

    def delete_source_chunks(self, source_id: str) -> None:
        for chunk_id in [chunk_id for chunk_id, chunk in self._chunks.items() if chunk.source_id == source_id]:
            self._chunks.pop(chunk_id, None)

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        for chunk in chunks:
            self._chunks[chunk.id] = chunk

    def search(
        self,
        repo: str,
        query_vector: list[float],
        top_k: int,
        source_types: list[str] | None = None,
        numbers: list[int] | None = None,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for chunk in self._chunks.values():
            if chunk.repo != repo or chunk.embedding is None:
                continue
            if source_types and chunk.source_type not in source_types:
                continue
            if numbers and chunk.metadata.get("number") not in numbers:
                continue
            hits.append(SearchHit(chunk=chunk, score=_cosine_similarity(query_vector, chunk.embedding)))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]


class QdrantRestVectorStore:
    def __init__(self, base_url: str, collection: str, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.collection = collection
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = request.Request(
            url,
            method=method,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if exc.code == 404:
                raise FileNotFoundError(detail) from exc
            raise ConfigurationError(f"Qdrant request failed with status {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise ConfigurationError(f"Could not reach Qdrant at {self.base_url}: {exc.reason}") from exc

    def ensure_collection(self, vector_size: int) -> None:
        try:
            self._request("GET", f"/collections/{self.collection}")
        except FileNotFoundError:
            self._request(
                "PUT",
                f"/collections/{self.collection}",
                {"vectors": {"size": vector_size, "distance": "Cosine"}},
            )

    def delete_source_chunks(self, source_id: str) -> None:
        try:
            self._request(
                "POST",
                f"/collections/{self.collection}/points/delete?wait=true",
                {"filter": {"must": [{"key": "source_id", "match": {"value": source_id}}]}},
            )
        except FileNotFoundError:
            return

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        vector_size = len(chunks[0].embedding or [])
        if vector_size == 0:
            raise ConfigurationError("Cannot upsert chunks without embeddings.")
        self.ensure_collection(vector_size)
        points = [
            {
                "id": chunk.id,
                "vector": chunk.embedding,
                "payload": {
                    "chunk_id": chunk.id,
                    "source_id": chunk.source_id,
                    "repo": chunk.repo,
                    "source_type": chunk.source_type,
                    "chunk_index": chunk.chunk_index,
                    "title": chunk.title,
                    "url": chunk.url,
                    "text": chunk.text,
                    "snippet": chunk.snippet,
                    **chunk.metadata,
                },
            }
            for chunk in chunks
        ]
        self._request(
            "PUT",
            f"/collections/{self.collection}/points?wait=true",
            {"points": points},
        )

    def search(
        self,
        repo: str,
        query_vector: list[float],
        top_k: int,
        source_types: list[str] | None = None,
        numbers: list[int] | None = None,
    ) -> list[SearchHit]:
        conditions: list[dict[str, Any]] = [{"key": "repo", "match": {"value": repo}}]
        if source_types:
            conditions.append({"key": "source_type", "match": {"any": source_types}})
        if numbers:
            conditions.append({"key": "number", "match": {"any": numbers}})
        payload = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": True,
            "filter": {"must": conditions},
        }
        try:
            data = self._request(
                "POST",
                f"/collections/{self.collection}/points/search",
                payload,
            )
        except FileNotFoundError:
            return []
        results = data.get("result") or []
        hits: list[SearchHit] = []
        for result in results:
            payload = result.get("payload") or {}
            chunk = DocumentChunk(
                id=payload.get("chunk_id", str(result.get("id"))),
                source_id=payload.get("source_id", ""),
                repo=payload.get("repo", repo),
                source_type=payload.get("source_type", "doc"),
                chunk_index=int(payload.get("chunk_index", 0) or 0),
                title=payload.get("title", ""),
                url=payload.get("url", ""),
                text=payload.get("text", ""),
                snippet=payload.get("snippet", ""),
                metadata={
                    "path": payload.get("path"),
                    "number": payload.get("number"),
                    "author": payload.get("author"),
                    "labels": payload.get("labels", []),
                },
            )
            hits.append(SearchHit(chunk=chunk, score=float(result.get("score", 0.0) or 0.0)))
        return hits
