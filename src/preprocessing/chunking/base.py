from abc import ABC, abstractmethod

from pydantic import BaseModel

from ingestion.loaders.base import RawDocument


class ChunkConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50


class ChunkedDocument(BaseModel):
    chunk_id: str
    text: str
    token_count: int
    metadata: dict

    model_config = {"frozen": True}


class BaseChunker(ABC):
    def __init__(self, config: ChunkConfig):
        self.config = config

    @abstractmethod
    def chunk(self, doc: RawDocument) -> list[ChunkedDocument]: ...
