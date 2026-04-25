from abc import ABC, abstractmethod

from pydantic import BaseModel

from airagpedia.ingestion.loaders.base import RawDocument


class ChunkConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50


class ChunkedDocument(BaseModel):
    doc_id: str
    chunk_id: str
    text: str
    chunk_index: int
    token_count: int
    metadata: dict

    model_config = {"frozen": True}

    def to_vector_metadata(self) -> dict:
        return {
            # base（url, pageid, category...）
            **self.metadata,
            "text": self.text,
            "doc_id": self.doc_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
        }


class BaseChunker(ABC):
    def __init__(self, config: ChunkConfig):
        self.config = config

    @abstractmethod
    def chunk(self, doc: RawDocument) -> list[ChunkedDocument]: ...
