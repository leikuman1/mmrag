import unittest
from pathlib import Path

from mmrag.models import DocumentChunk, SourceDocument
from mmrag.storage.catalog import CatalogStore


def _workspace_db_path(name: str) -> Path:
    base = Path("data/test-artifacts")
    base.mkdir(parents=True, exist_ok=True)
    path = base / name
    if path.exists():
        path.unlink()
    return path


class CatalogTests(unittest.TestCase):
    def test_catalog_keyword_search_and_counts(self) -> None:
        path = _workspace_db_path("catalog.sqlite3")
        catalog = CatalogStore(path)
        document = SourceDocument(
            id="owner/repo:issue:12",
            repo="owner/repo",
            source_type="issue",
            title="Issue #12: Streaming bug",
            url="https://example.com/issues/12",
            text="Streaming crashes when the state graph retries a step.",
            updated_at="2026-05-02T00:00:00Z",
            number=12,
        )
        catalog.upsert_source(document, "hash123")
        chunks = [
            DocumentChunk(
                id="chunk-1",
                repo="owner/repo",
                source_id=document.id,
                source_type="issue",
                chunk_index=1,
                title=document.title,
                url=document.url,
                text=document.text,
                snippet="Streaming crashes when the state graph retries a step.",
                metadata={"number": 12},
            )
        ]
        catalog.replace_chunks(document.id, chunks)
        hits = catalog.keyword_search("owner/repo", "streaming retry bug", top_k=5, numbers=[12])
        doc_count, chunk_count = catalog.repo_counts("owner/repo")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].chunk.id, "chunk-1")
        self.assertEqual(doc_count, 1)
        self.assertEqual(chunk_count, 1)
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
