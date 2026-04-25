import os
import time

from pinecone import Pinecone, ServerlessSpec

from airagpedia.vector_store.base import BaseVectorStore, BaseVectorStoreConfig


class PineconeConfig(BaseVectorStoreConfig):
    index_name: str = ""
    metric: str = "cosine"
    cloud: str = "aws"
    region: str = "us-east-1"


class PineconeVectorStore(BaseVectorStore):
    def __init__(self, config: PineconeConfig):
        super().__init__(config)
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.load()

    def _create_index(self) -> None:
        if self.config.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.config.index_name,
                dimension=self.config.dimension,
                metric=self.config.metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    def load(self) -> None:
        self._create_index()
        self.index = self.pc.Index(self.config.index_name)

    def add(self, vectors, metadata, batch_size=100):
        assert len(vectors) == len(metadata)

        dim = len(vectors[0])
        assert all(len(vector) == dim for vector in vectors)

        # (id_, emb, metadata)
        vectors_to_db = list(
            zip([meta["chunk_id"] for meta in metadata], vectors, metadata)
        )

        for i in range(0, len(vectors_to_db), batch_size):
            self.index.upsert(vectors=vectors_to_db[i : i + batch_size])

    def save(self):
        pass

    def search(self, query: list[float], top_k: int = 3) -> list[dict]:
        results = self.index.query(vector=query, top_k=top_k, include_metadata=True)
        return [
            {**match.metadata, "chunk_id": match.id, "score": match.score}
            for match in results.matches
        ]

    def delete(self):
        pass

    def update(self):
        pass
