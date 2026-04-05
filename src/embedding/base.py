from abc import ABC, abstractmethod

from pydantic import BaseModel


class EmbeddingConfig(BaseModel):
    output_dimensionality: int = 512

    model_config = {"frozen": True}


class BaseEmbedder(ABC):
    def __init__(self, config: EmbeddingConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def embed(self, text: str) -> list[float]: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
