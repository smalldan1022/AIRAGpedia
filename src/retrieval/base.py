from abc import ABC, abstractmethod

from pydantic import BaseModel

from vector_store.base import BaseVectorStore


class BaseRetrievalConfig(BaseModel):
    top_k: int = 5
    model_config = {"frozen": True}


class BaseRetriever(ABC):
    def __init__(self, config: BaseRetrievalConfig, vector_store: BaseVectorStore):
        super().__init__()
        self.config = config
        self.vector_store = vector_store

    @abstractmethod
    def retrieve(self, query_vector: list[float], top_k: int = 5) -> list[dict]: ...
