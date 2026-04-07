import uuid

from langchain_text_splitters import TokenTextSplitter

from ingestion.loaders.base import RawDocument
from preprocessing.chunking.base import BaseChunker, ChunkConfig, ChunkedDocument


class FixedChunkConfig(ChunkConfig):
    chunk_size: int = 512
    chunk_overlap: int = 50


class FixedChunker(BaseChunker):
    def __init__(self, config: FixedChunkConfig):
        super().__init__(config)
        self.splitter = TokenTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )

    def chunk(self, doc: RawDocument) -> ChunkedDocument:
        chunks = self.splitter.split_text(doc.text)
        return [
            ChunkedDocument(
                chunk_id=str(uuid.uuid4()),
                text=chunk,
                # Approximation, can change into tiktoken
                token_count=len(chunk.split()),
                metadata={
                    **doc.metadata,
                    "title": doc.title,
                    "source_type": doc.source_type.value,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                },
            )
            for idx, chunk in enumerate(chunks)
        ]
