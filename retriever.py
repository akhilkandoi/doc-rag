# Loads the index from Qdrant and builds a retriever with:
#   1. Hybrid search — dense (meaning) + BM25 (keywords), fused with RRF
#   2. Reranking     — cross-encoder scores each (query, chunk) pair together

import qdrant_client
from llama_index.core import Settings
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex
 
from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, EMBED_MODEL,
    QDRANT_URL, COLLECTION_NAME,
    RETRIEVAL_TOP_K, RERANK_TOP_N, RERANKER_MODEL
)
 
def build_retriever():

    Settings.llm=Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0
    )

    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    client = qdrant_client.QdrantClient(url=QDRANT_URL)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25"
    )

    index = VectorStoreIndex.from_vector_store(vector_store)

    reranker = SentenceTransformerRerank(
        model=RERANKER_MODEL,
        top_n=RERANK_TOP_N
    )

    retriever = index.as_retriever(
        similarity_top_k=RETRIEVAL_TOP_K,
        sparse_top_k=RETRIEVAL_TOP_K,
        vector_store_query_mode="hybrid",   # dense + BM25 fused with RRF
        node_postprocessors=[reranker]
    )
 
    return retriever


def retrieve(query: str, retriever) -> list[dict]:
    """
    Run a query through the retriever.
    Returns list of dicts with text, source, score.
    """
    nodes = retriever.retrieve(query)
 
    return [
        {
            "text": node.node.text,
            "source": node.node.metadata.get("source", "unknown"),
            "score": node.score
        }
        for node in nodes
    ]
 