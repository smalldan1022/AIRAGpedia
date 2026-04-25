from airagpedia.retrieval.base import BaseRetrievalConfig, BaseRetriever
from airagpedia.vector_store.pinecone_store import PineconeVectorStore


class PineconeRetrieverConfig(BaseRetrievalConfig):
    top_k: int = 5


class PineconeRetriever(BaseRetriever):
    def __init__(self, config: BaseRetrievalConfig, vector_store: PineconeVectorStore):
        super().__init__(config, vector_store)

    def retrieve(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        top_k = top_k or self.config.top_k
        return self.vector_store.search(query_vector, top_k)
