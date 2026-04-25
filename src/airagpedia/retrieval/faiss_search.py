from retrieval.base import BaseRetrievalConfig, BaseRetriever
from vector_store.faiss_store import FAISSVectorStore


class FAISSRetrieverConfig(BaseRetrievalConfig):
    top_k: int = 5


class FAISSRetriever(BaseRetriever):
    def __init__(self, config: BaseRetrievalConfig, vector_store: FAISSVectorStore):
        super().__init__(config, vector_store)

    def retrieve(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        top_k = top_k or self.config.top_k
        return self.vector_store.search(query_vector, top_k)


if __name__ == "__main__":
    from retrieval.base import BaseRetrievalConfig
    from vector_store.faiss_store import FAISSConfig, FAISSVectorStore

    # 建立 vector store
    store = FAISSVectorStore(FAISSConfig(dimension=768))
    vectors = [[0.1] * 768, [0.2] * 768, [0.3] * 768]
    metadata = [
        {"chunk_id": "1", "text": "Machine learning basics"},
        {"chunk_id": "2", "text": "Deep learning intro"},
        {"chunk_id": "3", "text": "NLP overview"},
    ]
    store.add(vectors, metadata)

    # 搜尋
    retriever = FAISSRetriever(BaseRetrievalConfig(top_k=2), store)
    results = retriever.retrieve([0.15] * 768)
    for r in results:
        print(r)
