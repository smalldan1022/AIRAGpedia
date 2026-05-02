from abc import ABC, abstractmethod

from pydantic import BaseModel


class BaseVectorStoreConfig(BaseModel):
    dimension: int = 768
    in_memory: bool = True

    model_config = {"frozen": True}


class BaseVectorStore(ABC):
    """
    Abstract base class for vector stores.

    Design principles:
    1. CRUD operations are core abstractions
    2. Lifecycle management (connect/disconnect) is separated
    3. Batch operations for efficiency
    """

    def __init__(self, config: BaseVectorStoreConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def add(self, vectors: list[list[float]], metadata: list[dict]) -> None:
        """
        Add vectors to the store (upsert semantics).

        This operation is idempotent:
        - New IDs will be inserted
        - Existing IDs will be updated with new vectors/metadata

        For strict insert/update semantics, use insert() or update() methods.
        """
        ...

    @abstractmethod
    def search(self, query: list[float], top_k: int) -> list[dict]: ...

    @abstractmethod
    def delete(self, ids: list[str]) -> None: ...
