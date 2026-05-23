from airagpedia.embedding.gemini_embed import EmbeddingConfig, GeminiEmbedder
from airagpedia.ingestion.loaders.wikipedia_loader import (
    WikipediaConfig,
    WikipediaLoader,
)
from airagpedia.preprocessing.chunking.fixed import FixedChunkConfig, FixedChunker
from airagpedia.preprocessing.cleaning import TextCleaner
from airagpedia.retrieval.bm25_search import BM25Retriever, BM25RetrieverConfig
from airagpedia.utils.logger import LoggerFactory
from airagpedia.vector_store.pinecone_store import PineconeConfig, PineconeVectorStore


def main():
    logger = LoggerFactory(level="INFO").get_logger(name=__name__)

    # Set up the configs
    wikipedia_config = WikipediaConfig(max_pages=20, document_type="html")
    chunker_config = FixedChunkConfig(chunk_size=128, chunk_overlap=24)
    embedding_config = EmbeddingConfig(output_dimensionality=768)
    pinecone_vec_store_config = PineconeConfig(dimension=768, index_name="airagpedia")

    # 1. Ingest the data
    logger.info("Ingesting the data...")
    wiki_loader = WikipediaLoader(wikipedia_config)
    raw_docs = wiki_loader.load_batch(
        categories=[
            "Large language models",
            "Generative pre-trained transformers",
            "Natural language processing",
        ]
    )

    # 2. Preprocess the data
    logger.info("Preprocessing the data...")
    cleaner = TextCleaner()
    processed_docs = [cleaner.clean(raw_doc) for raw_doc in raw_docs]

    # 3. Chunk the data
    logger.info("Chunking the documents...")
    fixed_chunker = FixedChunker(chunker_config)
    chunk_docs = []
    for processed_doc in processed_docs:
        chunk_docs.extend(fixed_chunker.chunk(processed_doc))

    # 4. Embedding the data
    logger.info("Embedding the chunks...")
    gemini_embedder = GeminiEmbedder(embedding_config)
    embedded_vectors = gemini_embedder.embed_batch(
        [chunk_doc.text for chunk_doc in chunk_docs]
    )

    # 5. Save the content into vector database
    logger.info("Connecting to vector database...")
    vec_store = PineconeVectorStore(pinecone_vec_store_config)
    vec_store.connect()
    metadata = [chunk_doc.to_vector_metadata() for chunk_doc in chunk_docs]
    logger.info("Saving the embeddings into vector database...")
    vec_store.add(
        ids=[meta["chunk_id"] for meta in metadata],
        vectors=embedded_vectors,
        metadata=metadata,
    )
    bm25_config = BM25RetrieverConfig()
    bm25_retriever = BM25Retriever(bm25_config, vec_store.index)
