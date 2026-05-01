import unittest

from mmrag.indexing.chunking import TextChunker
from mmrag.models import SourceDocument


class ChunkingTests(unittest.TestCase):
    def test_chunker_splits_large_documents(self) -> None:
        document = SourceDocument(
            id="repo:doc:README.md",
            repo="owner/repo",
            source_type="doc",
            title="README.md",
            url="https://example.com",
            text=("paragraph " * 300) + "\n\n" + ("tail " * 120),
            updated_at="sha",
        )
        chunker = TextChunker(chunk_size=400, overlap=50)
        chunks = chunker.split_document(document)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].source_id, document.id)
        self.assertEqual(chunks[0].repo, document.repo)


if __name__ == "__main__":
    unittest.main()
