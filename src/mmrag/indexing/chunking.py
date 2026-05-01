from __future__ import annotations

from mmrag.models import DocumentChunk, SourceDocument
from mmrag.utils import compact_text, slugify


class TextChunker:
    def __init__(self, chunk_size: int = 1200, overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split_document(self, document: SourceDocument) -> list[DocumentChunk]:
        normalized = document.text.replace("\r\n", "\n").strip()
        if not normalized:
            return []
        paragraphs = [paragraph.strip() for paragraph in normalized.split("\n\n") if paragraph.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            candidate = paragraph if not current else f"{current}\n\n{paragraph}"
            if len(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            if len(paragraph) <= self.chunk_size:
                current = paragraph
                continue
            start = 0
            while start < len(paragraph):
                end = min(start + self.chunk_size, len(paragraph))
                chunks.append(paragraph[start:end])
                if end >= len(paragraph):
                    start = end
                else:
                    start = max(0, end - self.overlap)
            current = ""
        if current:
            chunks.append(current)
        return [
            DocumentChunk(
                id=f"{slugify(document.id)}-{index}",
                repo=document.repo,
                source_id=document.id,
                source_type=document.source_type,
                chunk_index=index,
                title=document.title,
                url=document.url,
                text=chunk_text,
                snippet=compact_text(chunk_text, limit=180),
                metadata={
                    "path": document.path,
                    "number": document.number,
                    "author": document.author,
                    "labels": document.labels,
                },
            )
            for index, chunk_text in enumerate(chunks, start=1)
        ]

