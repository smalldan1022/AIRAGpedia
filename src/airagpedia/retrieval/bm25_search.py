import json
import os
import pickle
import re
from datetime import datetime

import jieba
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from airagpedia.retrieval.base import BaseRetrievalConfig, BaseRetriever
from airagpedia.utils.logger import LoggerFactory


class BM25RetrieverConfig(BaseRetrievalConfig):
    top_k: int = 10
    k1: float = 1.5
    b: float = 0.75
    cache_path: str = "./cache/bm_25/"
    cache_file: str = "index.txt"


class BM25Retriever(BaseRetriever):
    def __init__(
        self, config: BM25RetrieverConfig, pinecone_index: Pinecone.Index = None
    ):
        super().__init__(config, vector_store=None)
        self.config = config
        self.logger = LoggerFactory(level="INFO").get_logger(name=__name__)
        self.all_ids = None

        if self.valid_cache():
            self._load_cache()

        elif pinecone_index:
            self.pinecone_index = pinecone_index
            self._init_from_pinecone()
        else:
            raise ValueError(
                f"Make sure we got data in cache or pinecone. Current cache path is {self.config.cache_path}"
            )

        self.bm25 = BM25Okapi(self.corpus, k1=self.config.k1, b=self.config.b)

    def valid_cache(self) -> bool:
        meta_path = self.config.cache_path + "cache_meta.json"
        if not os.path.exists(meta_path):
            return False

        # load the small json file just for validation
        with open(meta_path, "r") as f:
            meta = json.load(f)

        return self.all_ids and len(self.all_ids) == meta["total_tokens"]

    def _get_all_chunk_ids(self) -> list:
        all_ids = [
            id
            for batch in self.pinecone_index.list(namespace="__default__")
            for id in batch
        ]
        return all_ids

    def refresh(self, pinecone_index: Pinecone.Index = None) -> None:
        if not pinecone_index:
            raise ValueError("Pinecone index not exists, cannot refresh!")
        else:
            self.pinecone_index = pinecone_index

        self.logger.info("Refreshing BM25 index...")
        self.logger.info("Refreshing BM25 index...")
        self._init_from_pinecone()
        self.bm25 = BM25Okapi(self.corpus, k1=self.config.k1, b=self.config.b)

    def _init_from_pinecone(self) -> None:
        BATCH_SIZE = 100

        if not self.pinecone_index:
            raise ValueError("Pinecone index not exists.")

        self.logger.info("Reading Pinecone Index...")

        # 1. Get all chunk IDs
        self.all_ids = self._get_all_chunk_ids()

        if not self.all_ids:
            raise ValueError("Pinecone no docs!!")

        self.logger.info(f"Found {len(self.all_ids)} chunks")

        corpus = []
        doc_ids = []
        metadata_map = {}

        for i in tqdm(range(0, len(self.all_ids), BATCH_SIZE), desc="Reading Pinecone"):
            batch = self.all_ids[i : i + BATCH_SIZE]

            data = self.pinecone_index.fetch(ids=batch)
            vectors = data.vectors

            for chunk_id, vector_data in vectors.items():
                metadata = vector_data.metadata

                text = metadata["text"]
                corpus.append(list(jieba.cut(text)))
                doc_ids.append(chunk_id)
                metadata_map[chunk_id] = metadata

        if not corpus:
            raise ValueError(
                "Not reading any docs from Pinecone!"
                "Please make sure there is text in metadata"
            )

        self.corpus = corpus
        self.doc_ids = doc_ids
        self.metadata_map = metadata_map

        # save the cache
        self._save_cache()

        self.logger.info(f"✅ Get chunks from Pinecone: {len(corpus)} chunks")

    def _load_cache(self) -> None:
        with open(self.config.cache_path + self.config.cache_file, "rb") as f:
            data = pickle.load(f)

        self.corpus = data["corpus"]
        self.doc_ids = data["doc_ids"]
        self.metadata_map = data.get("metadata_map", {})
        self.logger.info(f"✅ Loaded cache: {len(self.corpus)} chunks")

    def _save_cache(self) -> None:
        data = {
            "corpus": self.corpus,
            "doc_ids": self.doc_ids,
            "metadata_map": self.metadata_map,
            "timestamp": datetime.now().isoformat(),
            "config": {"k1": self.config.k1, "b": self.config.b},
        }
        meta = {"total_tokens": len(self.corpus)}

        os.makedirs(self.config.cache_path, exist_ok=True)
        with open(self.config.cache_path + self.config.cache_file, "wb") as f:
            pickle.dump(data, f)

        with open(self.config.cache_path + "cache_meta.json", "w") as f:
            json.dump(meta, f)

        self.logger.info(f"✅ Saved cache into {self.config.cache_path}")

    def _tokenize(self, doc: str) -> list[str]:
        return list(jieba.cut(doc))

    @staticmethod
    def extract_exact_term(query: str) -> list:
        exact_terms = []
        # 1. Extract the date (2020-2030)
        years = re.findall(r"202[0-9]", query)
        exact_terms.extend(years)

        # 2. Version (GPT 3.0)
        versions = re.findall(r"(?:\d+\.)+\d+", query)
        exact_terms.extend(versions)

        # 3. AI Model names, like GPT-4, GPT 3.5, Claude-3, Gemini 1.5
        models = re.findall(r"[A-Za-z]+[- ]\d+(?:\.\d+)?", query)
        exact_terms.extend(models)
        return exact_terms

    @staticmethod
    def should_use_bm25(query: str) -> bool:
        QUESTION_WORDS = ["how", "why", "what"]
        exact_terms = BM25Retriever.extract_exact_term(query)

        if exact_terms:
            return True

        # look for short query, no more than 3 sentences
        words = query.split()
        if len(words) < 3:
            has_question = any(w in QUESTION_WORDS for w in words)
            if has_question:
                return True

        return False

    def retrieve(self, queries: list[str], top_k: int = 5) -> list[dict]:
        top_k = top_k or self.config.top_k
        query_tokens = [self._tokenize(query) for query in queries]
        return self.bm25.get_top_n(query_tokens, self.corpus, n=top_k)


####################################


# import jieba

# # 準備文檔
# docs = ["機器學習很有趣", "深度學習是機器學習的分支", "機器學習叫做 machine learning"]
# tokenized_corpus = [list(jieba.cut(d)) for d in docs]

# # 建立索引並搜尋
# bm25 = BM25Okapi(tokenized_corpus)
# query = list(jieba.cut("機器學習"))
# print(query)
# scores = bm25.get_scores(query)
# print(scores)


if __name__ == "__main__":
    import os

    from pinecone import Pinecone

    PINECONE_API = os.environ.get("PINECONE_API_KEY")

    pc = Pinecone(api_key=PINECONE_API)
    index = pc.Index("airagpedia")

    # Iterates through all IDs in a namespace
    IDS = [id for batch in index.list(namespace="__default__") for id in batch]
    data = index.fetch(ids=IDS[0:1])

    for key, val in data.vectors.items():
        print(key)
        print("\n\n\n")
        print(val)
        print("\n\n\n")
    # Example output: ['doc1#chunk1', 'doc1#chunk2', 'doc1#chunk3']
