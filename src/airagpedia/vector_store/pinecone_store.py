import os
import time
from typing import Any

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
        self._connect = False

    def _create_index(self) -> None:
        if self.config.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.config.index_name,
                dimension=self.config.dimension,
                metric=self.config.metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    def connect(self) -> None:
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self._create_index()
        self.index = self.pc.Index(self.config.index_name)
        self.index_stats = self.index.describe_index_stats()
        self._connect = True

    def disconnect(self) -> None:
        self.pc = None
        self.index = None
        self.index_stats = None
        self._connect = False

    def add(
        self,
        ids: list[str],
        vectors: list[float],
        metadata: list[dict],
        batch_size: int = 100,
    ):
        if not self._connect:
            raise RuntimeError("Not connected. Call connect() first.")
        assert len(vectors) == len(ids) == len(metadata), (
            "The data lengths are not consistent."
        )

        dim = len(vectors[0])
        assert all(len(vector) == dim for vector in vectors), (
            "The dimensions are not consistent."
        )

        # (id_, emb, metadata)
        vectors_to_db = list(zip(ids, vectors, metadata))

        for i in range(0, len(vectors_to_db), batch_size):
            self.index.upsert(vectors=vectors_to_db[i : i + batch_size])

    def search(self, query: list[float], top_k: int = 3) -> list[dict]:
        results = self.index.query(vector=query, top_k=top_k, include_metadata=True)
        return [
            {**match.metadata, "chunk_id": match.id, "score": match.score}
            for match in results.matches
        ]

    @property
    def index_count(self) -> int:
        return self.index_stats.total_vector_count

    def delete(
        self,
        ids: list[str] | None = None,
        filter: dict[str, Any] | None = None,
        delete_all: bool = False,
    ) -> int:
        if delete_all:
            self.index.delete(delete_all=True)
            return self.index_count  # Return previous count

        elif ids:
            self.index.delete(ids=ids)
            return len(ids)
        elif filter:
            self.index.delete(filter=filter)
            return -1

    def upsert(self, ids: list[str], vectors: list[list[float]], metadata: list[dict]):
        pass
