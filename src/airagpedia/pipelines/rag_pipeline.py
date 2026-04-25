import json

from airagpedia.embedding.gemini_embed import EmbeddingConfig, GeminiEmbedder
from airagpedia.generation.gemini_gen import GeminiConfig, GeminiGenerator
from airagpedia.generation.prompt_builder import WikipediaPromptBuilder
from airagpedia.generation.templates.wikipedia_prompt import WikipediaPrompt
from airagpedia.ingestion.loaders.wikipedia_loader import (
    WikipediaConfig,
    WikipediaLoader,
)
from airagpedia.preprocessing.chunking.fixed import FixedChunkConfig, FixedChunker
from airagpedia.preprocessing.cleaning import TextCleaner

# from airagpedia.retrieval.faiss_search import FAISSRetriever, FAISSRetrieverConfig
from airagpedia.retrieval.pinecone_search import (
    PineconeRetriever,
    PineconeRetrieverConfig,
)
from airagpedia.utils.logger import LoggerFactory
from airagpedia.vector_store.pinecone_store import PineconeConfig, PineconeVectorStore

# from airagpedia.vector_store.faiss_store import FAISSConfig, FAISSVectorStore


def main():
    USER_QUERY = "What is the latest GPT model?"
    logger = LoggerFactory(level="INFO").get_logger(name=__name__)

    # Set up the configs
    wikipedia_config = WikipediaConfig(max_pages=20, document_type="html")
    chunker_config = FixedChunkConfig(chunk_size=128, chunk_overlap=24)
    embedding_config = EmbeddingConfig(output_dimensionality=768)
    pinecone_vec_store_config = PineconeConfig(dimension=768, index_name="airagpedia")
    pinecone_retriever_config = PineconeRetrieverConfig(tok_k=3)
    wiki_prompt = WikipediaPrompt()
    gemini_generator_config = GeminiConfig(
        max_tokens=3000, output_type="stream", temperature=0.4, model="gemini-2.5-flash"
    )

    # 1. Ingest the data
    wiki_loader = WikipediaLoader(wikipedia_config)
    raw_docs = wiki_loader.load_batch(
        categories=[
            "Large language models",
            "Generative pre-trained transformers",
            "Natural language processing",
        ]
    )

    # 2. Preprocess the data
    cleaner = TextCleaner()
    processed_docs = [cleaner.clean(raw_doc) for raw_doc in raw_docs]

    # 3. Chunk the data
    fixed_chunker = FixedChunker(chunker_config)
    chunk_docs = []
    for processed_doc in processed_docs:
        chunk_docs.extend(fixed_chunker.chunk(processed_doc))

    # 4. Embedding the data
    gemini_embedder = GeminiEmbedder(embedding_config)
    vec_store = PineconeVectorStore(pinecone_vec_store_config)

    embedded_vectors = gemini_embedder.embed_batch(
        [chunk_doc.text for chunk_doc in chunk_docs]
    )
    vec_store.add(
        vectors=embedded_vectors,
        metadata=[chunk_doc.to_vector_metadata() for chunk_doc in chunk_docs],
    )

    # 5. Retrieval: save into vector database / search from the vector database
    retriever = PineconeRetriever(
        config=pinecone_retriever_config, vector_store=vec_store
    )
    query_vector = gemini_embedder.embed(USER_QUERY)
    top_k_content = retriever.retrieve(query_vector=query_vector, top_k=3)

    # 6. Augmentation: Build the specific prompt to meet the question type
    _prompt_builder = WikipediaPromptBuilder(wiki_prompt)
    prompt = _prompt_builder.build(
        content="\n\n".join([doc["text"] for doc in top_k_content]), query=USER_QUERY
    )

    # 7. Generation: use the LLM to generate the answer
    generator = GeminiGenerator(gemini_generator_config)
    result = generator.generate(prompt=prompt)
    # parsed = json.loads(result.text)
    # relevant_chunk_texts = [
    #     top_k_content[idx]["text"] for idx in parsed["relevant_chunk_ids"]
    # ]
    # result.relevant_chunk_texts = relevant_chunk_texts
    logger.info(f"Final Results:\n{result.text}")
    # for i in range(len(relevant_chunk_texts)):
    #     logger.info(f"Revelant chunks - [{i}]:\n{result.relevant_chunk_texts[i]}\n\n")
