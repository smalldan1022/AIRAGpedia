from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseVectorStoreConfig(BaseModel):
    dimension: int = 768
    in_memory: bool = True

    model_config = {"frozen": True}


class BaseVectorStore(ABC):
    def __init__(self, config: BaseVectorStoreConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def add(self, vectors: list[list[float]], metadata: list[dict]) -> None: ...

    @abstractmethod
    def search(self, query: list[float], top_k: int) -> list[dict]: ...

    @abstractmethod
    def save(self) -> None: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def update(
        self, ids: list[str], vectors: list[list[float]], metadata: list[dict]
    ) -> None: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...
