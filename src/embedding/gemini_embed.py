import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from embedding.base import BaseEmbedder, EmbeddingConfig


class GeminiEmbedder(BaseEmbedder):
    MODEL = "models/gemini-embedding-001"

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        load_dotenv()
        self.setup_config()

    def embed(self, text: str) -> list[float]:
        result = self.client.models.embed_content(
            model=self.MODEL,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self.config.output_dimensionality
            ),
        )
        return result.embeddings[0].values

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        results = self.client.models.embed_content(
            model=self.MODEL,
            contents=texts,
            config=types.EmbedContentConfig(
                output_dimensionality=self.config.output_dimensionality
            ),
        )
        return [emb.values for emb in results.embeddings]

    def setup_config(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        self.client = genai.Client(api_key=api_key)


if __name__ == "__main__":
    embedding_config = EmbeddingConfig()
    embedding = GeminiEmbedder(config=embedding_config)
    print(embedding.embed("Hi I am Dan and I am 30 years old."))
