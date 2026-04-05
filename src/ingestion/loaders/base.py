from abc import ABC, abstractmethod
from enum import Enum

from pydantic import BaseModel


class SourceType(Enum):
    WIKIPEDIA = "wikipedia"
    PDF = "pdf"
    NOTION = "notion"


class RawDocument(BaseModel):
    title: str
    text: str
    category: str
    source_type: SourceType
    metadata: dict
    # immutable
    model_config = {"frozen": True}


class LoaderConfig(BaseModel):
    """Loader settings, use pydantic for validation"""

    source: str
    document_type: str
    max_pages: int = 50


class BaseLoader(ABC):
    def __init__(self, config: LoaderConfig):
        self.config = config

    @abstractmethod
    def load(self) -> list[RawDocument]:
        """Load documents from a source and return a list of RawDocument."""
        ...
