import unittest
from pathlib import Path

from mmrag.agents.workflow import AgentWorkflow
from mmrag.models import DocumentChunk, SourceDocument
from mmrag.retrieval.service import RetrievalService
from mmrag.storage.catalog import CatalogStore
from mmrag.storage.vector_store import InMemoryVectorStore


def _workspace_db_path(name: str) -> Path:
    base = Path("data/test-artifacts")
    base.mkdir(parents=True, exist_ok=True)
    path = base / name
    if path.exists():
        path.unlink()
    return path


class FakeEmbedder:
    def embed(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("agent" in lowered),
                    float("workflow" in lowered),
                    float("issue" in lowered),
                    float("pr" in lowered),
                ]
            )
        return vectors


class WorkflowTests(unittest.TestCase):
    def test_workflow_returns_citations_and_trace(self) -> None:
        path = _workspace_db_path("workflow.sqlite3")
        catalog = CatalogStore(path)
        vector_store = InMemoryVectorStore()
        embedder = FakeEmbedder()
        document = SourceDocument(
            id="owner/repo:doc:README.md",
            repo="owner/repo",
            source_type="doc",
            title="README.md",
            url="https://example.com/README.md",
            text="This project explains an agent workflow with routing and synthesis.",
            updated_at="sha",
        )
        catalog.upsert_source(document, "hash")
        chunk = DocumentChunk(
            id="chunk-doc-1",
            repo="owner/repo",
            source_id=document.id,
            source_type="doc",
            chunk_index=1,
            title=document.title,
            url=document.url,
            text=document.text,
            snippet="This project explains an agent workflow with routing and synthesis.",
            metadata={},
            embedding=embedder.embed([document.text])[0],
        )
        catalog.replace_chunks(document.id, [chunk])
        vector_store.upsert([chunk])
        retrieval = RetrievalService(catalog, vector_store, embedder)
        workflow = AgentWorkflow(retrieval, catalog, chat_model=None)
        workflow.graph = None

        response = workflow.answer("owner/repo", "这个项目的 agent workflow 是什么？")
        trace = catalog.get_trace(response.trace_id)

        self.assertTrue(response.citations)
        self.assertTrue(response.grounded)
        self.assertIsNotNone(trace)
        self.assertGreaterEqual(len(trace["steps"]), 4)
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
