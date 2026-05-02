import hashlib
from datetime import datetime

from langchain_text_splitters import TokenTextSplitter

from airagpedia.ingestion.loaders.base import RawDocument
from airagpedia.preprocessing.chunking.base import (
    BaseChunker,
    ChunkConfig,
    ChunkedDocument,
)


class FixedChunkConfig(ChunkConfig):
    chunk_size: int = 128
    chunk_overlap: int = 24


class FixedChunker(BaseChunker):
    def __init__(self, config: FixedChunkConfig):
        super().__init__(config)
        self.splitter = TokenTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def hash_file(self, document_text: str):
        return hashlib.md5(document_text.encode()).hexdigest()

    def chunk(self, doc: RawDocument) -> ChunkedDocument:
        chunks = self.splitter.split_text(doc.text)
        doc_id = self.hash_file(doc.text)

        return [
            ChunkedDocument(
                doc_id=doc_id,
                chunk_id=f"{doc_id}_{idx}",
                text=chunk,
                chunk_index=idx,
                # Approximation, can change into tiktoken
                token_count=len(chunk.split()),
                metadata={
                    **doc.metadata,
                    "title": doc.title,
                    "source_type": doc.source_type.value,
                    "total_chunks": len(chunks),
                    "text": chunk,
                    "created_at": datetime.today(),
                },
            )
            for idx, chunk in enumerate(chunks)
        ]
