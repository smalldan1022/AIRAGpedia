from typing import Literal

from pydantic import BaseModel, Field


class BasePrompt(BaseModel):
    system_prompt: str
    instructions: str


class RAGAnswerSchema(BaseModel):
    """Structured output schema for RAG answers"""

    steps_by_steps_analysis: str = Field(
        description="At least 3 reasoning steps explaining how the answer was derived from context"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in the answer"
    )
    relevant_chunks: list[int] = Field(
        description="List of chunks indices used to derive the answer"
    )

    final_answer: str | Literal["N/A"] = Field(
        description="Final answer. Return 'N/A' if information is not available in context"
    )
