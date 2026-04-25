import json
from pathlib import Path

import faiss
import numpy as np

from vector_store.base import BaseVectorStore, BaseVectorStoreConfig


class FAISSConfig(BaseVectorStoreConfig):
    index_path: str = "data/faiss.index"
    metadata_path: str = "data/metadata.json"


class FAISSVectorStore(BaseVectorStore):
    DEFAULT_TOP_K = 5

    def __init__(self, config: FAISSConfig):
        super().__init__(config)
        self.index = faiss.IndexFlatL2(config.dimension)
        self.metadata = []

    def add(self, vectors: list[list[float]], metadata: list[dict]) -> None:
        vectors_np = np.array(vectors)
        self.index.add(vectors_np)
        self.metadata.extend(metadata)

    def search(self, query: list[float], top_k: int = DEFAULT_TOP_K) -> list[dict]:
        query = np.array([query])
        distances, indices = self.index.search(query, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            result = self.metadata[idx].copy()
            result["score"] = float(dist)
            result["index"] = idx
            results.append(result)

        return results

    def save(self) -> None:
        Path(self.config.index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.config.metadata_path).parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, self.config.index_path)
        # faiss.write_index(self.index, self.config.index_path)
        with open(self.config.metadata_path, "w") as file:
            json.dump(self.metadata, file, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if not Path(self.config.index_path).exists():
            raise FileNotFoundError(f"Index not found: {self.config.index_path}")
        if not Path(self.config.metadata_path).exists():
            raise FileNotFoundError(f"Metadata not found: {self.config.metadata_path}")

        self.index = faiss.read_index(self.config.index_path)

        with open(self.config.metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def update(
        self, ids: list[str], vectors: list[list[float]], metadata: list[dict]
    ) -> None:
        self.delete(ids)
        self.add(vectors, metadata)

    def delete(self, ids: list[str]) -> None:
        # We need to rebuild index
        surviving_metadata = [m for m in self.metadata if m.get("chunk_id") not in ids]
        surviving_vectors = [
            self.index.reconstruct(i)
            for i, m in enumerate(self.metadata)
            if m.get("chunk_id") not in ids
        ]

        self.index = faiss.IndexFlatL2(self.config.dimension)
        self.metadata = []

        if surviving_vectors:
            vectors_np = np.array(surviving_vectors, dtype=np.float32)
            self.index.add(vectors_np)
            self.metadata.extend(surviving_metadata)


if __name__ == "__main__":
    pass
