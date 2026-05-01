from __future__ import annotations

from mmrag.config import ConfigurationError
from mmrag.connectors.github import GitHubConnector
from mmrag.indexing.chunking import TextChunker
from mmrag.models import IngestionResult
from mmrag.storage.catalog import CatalogStore
from mmrag.storage.vector_store import InMemoryVectorStore, QdrantRestVectorStore
from mmrag.utils import stable_hash, utc_now_iso


class GitHubIngestionService:
    def __init__(
        self,
        connector: GitHubConnector,
        catalog: CatalogStore,
        vector_store: QdrantRestVectorStore | InMemoryVectorStore,
        embedder,
        chunker: TextChunker | None = None,
    ) -> None:
        self.connector = connector
        self.catalog = catalog
        self.vector_store = vector_store
        self.embedder = embedder
        self.chunker = chunker or TextChunker()

    def ingest_github(self, repo: str, include: set[str]) -> IngestionResult:
        if self.embedder is None:
            raise ConfigurationError(
                "Embedding provider is not configured. Set EMBEDDING_BASE_URL, EMBEDDING_API_KEY, and EMBEDDING_MODEL."
            )
        started_at = utc_now_iso()
        documents = self.connector.fetch_sources(repo, include)
        indexed_docs = 0
        indexed_chunks = 0
        skipped = 0
        warnings: list[str] = []
        for document in documents:
            content_hash = stable_hash(document.text)
            if not self.catalog.document_needs_reindex(document, content_hash):
                skipped += 1
                continue
            chunks = self.chunker.split_document(document)
            if not chunks:
                warnings.append(f"Skipped empty source {document.id}")
                continue
            embeddings = self.embedder.embed([chunk.text for chunk in chunks])
            if len(embeddings) != len(chunks):
                raise ConfigurationError(
                    f"Embedding provider returned {len(embeddings)} embeddings for {len(chunks)} chunks."
                )
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                chunk.embedding = embedding
            self.vector_store.delete_source_chunks(document.id)
            self.vector_store.upsert(chunks)
            self.catalog.upsert_source(document, content_hash)
            self.catalog.replace_chunks(document.id, chunks)
            indexed_docs += 1
            indexed_chunks += len(chunks)
        total_documents, total_chunks = self.catalog.repo_counts(repo)
        self.catalog.record_ingestion_state(repo, total_documents, total_chunks)
        return IngestionResult(
            repo=repo,
            documents_indexed=indexed_docs,
            chunks_indexed=indexed_chunks,
            skipped_documents=skipped,
            warnings=warnings,
            started_at=started_at,
            completed_at=utc_now_iso(),
        )
