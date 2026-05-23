from sentence_transformers import CrossEncoder

from airagpedia.retrieval.base import BaseRetrievalConfig, BaseRetriever
from airagpedia.vector_store.base import BaseVectorStore


class RerankerConfig(BaseRetrievalConfig):
    top_k: int = 5
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker(BaseRetriever):
    def __init__(
        self, config: RerankerConfig, vector_store: BaseVectorStore = None
    ) -> None:
        super().__init__(config, vector_store)
        self.model = CrossEncoder(self.config.model_name)

    def retrieve(self, query: str, chunks: list[dict], top_k=5) -> list[dict]:
        top_k = top_k or self.config.top_k
        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self.model.predict(pairs)
        top_k_order = sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)
        top_k_chunks = [chunks[i] for i in top_k_order[:top_k]]
        return top_k_chunks


if __name__ == "__main__":
    from sentence_transformers import CrossEncoder

    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    query = "What is GPT?"
    chunks = ["GPT is a language model", "Apple is a fruit", "GPT was made by OpenAI"]

    pairs = [(query, chunk) for chunk in chunks]
    scores = model.predict(pairs)

    print(scores)
