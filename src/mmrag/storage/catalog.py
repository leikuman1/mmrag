from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from mmrag.models import AgentStepTrace, AnswerResponse, DocumentChunk, SearchHit, SourceDocument
from mmrag.utils import json_dumps, tokenize_for_match, utc_now_iso


class CatalogStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _managed_connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._managed_connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS source_documents (
                    source_id TEXT PRIMARY KEY,
                    repo TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    author TEXT,
                    number_value INTEGER,
                    path TEXT,
                    metadata_json TEXT NOT NULL,
                    last_indexed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    repo TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    text TEXT NOT NULL,
                    snippet TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    repo TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    answer_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trace_steps (
                    trace_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    step TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    input_summary TEXT NOT NULL,
                    output_summary TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    token_usage INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ingestions (
                    repo TEXT PRIMARY KEY,
                    last_ingested_at TEXT NOT NULL,
                    document_count INTEGER NOT NULL,
                    chunk_count INTEGER NOT NULL
                );
                """
            )

    def document_needs_reindex(self, document: SourceDocument, content_hash: str) -> bool:
        with self._managed_connection() as connection:
            row = connection.execute(
                "SELECT updated_at, content_hash FROM source_documents WHERE source_id = ?",
                (document.id,),
            ).fetchone()
        if row is None:
            return True
        return row["updated_at"] != document.updated_at or row["content_hash"] != content_hash

    def upsert_source(self, document: SourceDocument, content_hash: str) -> None:
        with self._managed_connection() as connection:
            connection.execute(
                """
                INSERT INTO source_documents (
                    source_id, repo, source_type, title, url, updated_at, content_hash,
                    author, number_value, path, metadata_json, last_indexed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source_id) DO UPDATE SET
                    repo = excluded.repo,
                    source_type = excluded.source_type,
                    title = excluded.title,
                    url = excluded.url,
                    updated_at = excluded.updated_at,
                    content_hash = excluded.content_hash,
                    author = excluded.author,
                    number_value = excluded.number_value,
                    path = excluded.path,
                    metadata_json = excluded.metadata_json,
                    last_indexed_at = excluded.last_indexed_at
                """,
                (
                    document.id,
                    document.repo,
                    document.source_type,
                    document.title,
                    document.url,
                    document.updated_at,
                    content_hash,
                    document.author,
                    document.number,
                    document.path,
                    json_dumps(document.metadata),
                    utc_now_iso(),
                ),
            )

    def replace_chunks(self, source_id: str, chunks: Iterable[DocumentChunk]) -> None:
        chunk_list = list(chunks)
        with self._managed_connection() as connection:
            connection.execute("DELETE FROM chunks WHERE source_id = ?", (source_id,))
            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, source_id, repo, source_type, chunk_index, title,
                    url, text, snippet, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.id,
                        chunk.source_id,
                        chunk.repo,
                        chunk.source_type,
                        chunk.chunk_index,
                        chunk.title,
                        chunk.url,
                        chunk.text,
                        chunk.snippet,
                        json_dumps(chunk.metadata),
                    )
                    for chunk in chunk_list
                ],
            )

    def keyword_search(
        self,
        repo: str,
        query: str,
        top_k: int,
        source_types: list[str] | None = None,
        numbers: list[int] | None = None,
    ) -> list[SearchHit]:
        sql = "SELECT * FROM chunks WHERE repo = ?"
        params: list[Any] = [repo]
        if source_types:
            placeholders = ",".join(["?"] * len(source_types))
            sql += f" AND source_type IN ({placeholders})"
            params.extend(source_types)
        rows: list[sqlite3.Row]
        with self._managed_connection() as connection:
            rows = connection.execute(sql, params).fetchall()
        query_tokens = set(tokenize_for_match(query))
        hits: list[SearchHit] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if numbers and metadata.get("number") not in numbers:
                continue
            chunk_tokens = set(tokenize_for_match(row["text"]))
            overlap = len(query_tokens & chunk_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(query_tokens), 1)
            chunk = DocumentChunk(
                id=row["chunk_id"],
                source_id=row["source_id"],
                repo=row["repo"],
                source_type=row["source_type"],
                chunk_index=row["chunk_index"],
                title=row["title"],
                url=row["url"],
                text=row["text"],
                snippet=row["snippet"],
                metadata=metadata,
            )
            hits.append(SearchHit(chunk=chunk, score=score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def save_trace(
        self,
        trace_id: str,
        session_id: str | None,
        repo: str,
        answer: AnswerResponse,
        steps: list[AgentStepTrace],
    ) -> None:
        with self._managed_connection() as connection:
            connection.execute(
                """
                INSERT INTO traces (trace_id, session_id, repo, created_at, answer_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (trace_id, session_id, repo, utc_now_iso(), json_dumps(answer.to_dict())),
            )
            connection.execute("DELETE FROM trace_steps WHERE trace_id = ?", (trace_id,))
            connection.executemany(
                """
                INSERT INTO trace_steps (
                    trace_id, seq, step, agent, input_summary, output_summary, latency_ms, token_usage
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        trace_id,
                        index,
                        step.step,
                        step.agent,
                        step.input_summary,
                        step.output_summary,
                        step.latency_ms,
                        step.token_usage,
                    )
                    for index, step in enumerate(steps, start=1)
                ],
            )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self._managed_connection() as connection:
            trace_row = connection.execute(
                "SELECT * FROM traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
            if trace_row is None:
                return None
            steps = connection.execute(
                "SELECT * FROM trace_steps WHERE trace_id = ? ORDER BY seq ASC",
                (trace_id,),
            ).fetchall()
        return {
            "trace_id": trace_id,
            "session_id": trace_row["session_id"],
            "repo": trace_row["repo"],
            "created_at": trace_row["created_at"],
            "answer": json.loads(trace_row["answer_json"]),
            "steps": [
                {
                    "seq": step["seq"],
                    "step": step["step"],
                    "agent": step["agent"],
                    "input_summary": step["input_summary"],
                    "output_summary": step["output_summary"],
                    "latency_ms": step["latency_ms"],
                    "token_usage": step["token_usage"],
                }
                for step in steps
            ],
        }

    def record_ingestion_state(self, repo: str, document_count: int, chunk_count: int) -> None:
        with self._managed_connection() as connection:
            connection.execute(
                """
                INSERT INTO ingestions (repo, last_ingested_at, document_count, chunk_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(repo) DO UPDATE SET
                    last_ingested_at = excluded.last_ingested_at,
                    document_count = excluded.document_count,
                    chunk_count = excluded.chunk_count
                """,
                (repo, utc_now_iso(), document_count, chunk_count),
            )

    def repo_counts(self, repo: str) -> tuple[int, int]:
        with self._managed_connection() as connection:
            documents = connection.execute(
                "SELECT COUNT(*) AS value FROM source_documents WHERE repo = ?",
                (repo,),
            ).fetchone()
            chunks = connection.execute(
                "SELECT COUNT(*) AS value FROM chunks WHERE repo = ?",
                (repo,),
            ).fetchone()
        return int(documents["value"]), int(chunks["value"])

    def list_sources(self, repo: str) -> list[dict[str, Any]]:
        with self._managed_connection() as connection:
            rows = connection.execute(
                """
                SELECT source_id, source_type, title, url, updated_at, number_value, path
                FROM source_documents
                WHERE repo = ?
                ORDER BY source_type ASC, updated_at DESC
                """,
                (repo,),
            ).fetchall()
        return [
            {
                "source_id": row["source_id"],
                "source_type": row["source_type"],
                "title": row["title"],
                "url": row["url"],
                "updated_at": row["updated_at"],
                "number": row["number_value"],
                "path": row["path"],
            }
            for row in rows
        ]

    def get_ingestion_state(self, repo: str) -> dict[str, Any]:
        with self._managed_connection() as connection:
            row = connection.execute(
                "SELECT * FROM ingestions WHERE repo = ?",
                (repo,),
            ).fetchone()
        if row is None:
            return {"repo": repo, "indexed": False, "sources": [], "last_ingested_at": None}
        return {
            "repo": repo,
            "indexed": True,
            "last_ingested_at": row["last_ingested_at"],
            "document_count": row["document_count"],
            "chunk_count": row["chunk_count"],
            "sources": self.list_sources(repo),
        }
