from airagpedia.embedding.gemini_embed import EmbeddingConfig, GeminiEmbedder
from airagpedia.generation.gemini_gen import GeminiConfig, GeminiGenerator
from airagpedia.generation.prompt_builder import WikipediaPromptBuilder
from airagpedia.generation.templates.wikipedia_prompt import WikipediaPrompt
from airagpedia.retrieval.bm25_search import BM25Retriever, BM25RetrieverConfig
from airagpedia.retrieval.pinecone_search import (
    PineconeRetriever,
    PineconeRetrieverConfig,
)
from airagpedia.retrieval.reranker import Reranker, RerankerConfig
from airagpedia.utils.logger import LoggerFactory
from airagpedia.vector_store.pinecone_store import PineconeConfig, PineconeVectorStore


def main():
    USER_QUERY = "What is the latest GPT model?"
    logger = LoggerFactory(level="INFO").get_logger(name=__name__)

    # Set up the configs
    embedding_config = EmbeddingConfig(output_dimensionality=768)
    pinecone_vec_store_config = PineconeConfig(dimension=768, index_name="airagpedia")
    pinecone_retriever_config = PineconeRetrieverConfig(tok_k=3)
    bm25_retriever_config = BM25RetrieverConfig()
    wiki_prompt = WikipediaPrompt()
    gemini_generator_config = GeminiConfig(
        max_tokens=3000, output_type="stream", temperature=0.4, model="gemini-2.5-flash"
    )
    reranker_config = RerankerConfig()

    # 4. Embedding the data
    gemini_embedder = GeminiEmbedder(embedding_config)
    logger.info("Connecting to the Vector Store...")
    vec_store = PineconeVectorStore(pinecone_vec_store_config)
    vec_store.connect()

    # 5. Retrieval: save into vector database / search from the vector database
    logger.info("Retrieveing...")

    bm25_retriever = BM25Retriever(bm25_retriever_config, vec_store.index)
    if bm25_retriever.should_use_bm25(query=USER_QUERY):
        logger.info("Using Best Match 25...")
        bm25_retriever.retrieve(queries=[USER_QUERY])
    else:
        logger.info("Using Pinecone Vector search...")
        retriever = PineconeRetriever(
            config=pinecone_retriever_config, vector_store=vec_store
        )
        query_vector = gemini_embedder.embed(USER_QUERY)
        top_k_content = retriever.retrieve(query_vector=query_vector, top_k=10)

    reranker = Reranker(config=reranker_config)
    top_k_content = reranker.retrieve(query=USER_QUERY, chunks=top_k_content, top_k=3)

    # 6. Augmentation: Build the specific prompt to meet the question type
    _prompt_builder = WikipediaPromptBuilder(wiki_prompt)
    prompt = _prompt_builder.build(
        content="\n\n".join([doc["text"] for doc in top_k_content]), query=USER_QUERY
    )

    # 7. Generation: use the LLM to generate the answer
    logger.info("Generating...")
    generator = GeminiGenerator(gemini_generator_config)
    result = generator.generate(prompt=prompt)

    logger.info(f"Final Results:\n{result.text}")
