from abc import ABC, abstractmethod

from pydantic import BaseModel, Field


class BaseGeneratorConfig(BaseModel):
    max_tokens: int = 2500
    output_type: str = "stream"
    temperature: float = 0.4
    model: str = ""

    model_config = {"frozen": True}


class GenerationResult(BaseModel):
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    output_type: str = "json"
    relevant_chunk_ids: list[int] = Field(default_factory=list)
    relevant_chunk_texts: list[str] = Field(default_factory=list)


class BaseGenerator(ABC):
    def __init__(self, config: BaseGeneratorConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def generate(self, prompt: str = "") -> GenerationResult: ...
