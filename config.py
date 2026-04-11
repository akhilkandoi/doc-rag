import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
QDRANT_URL      = os.getenv("QDRANT_URL",      "http://localhost:6333")

# Ollama
OLLAMA_MODEL    = "qwen2.5:1.5b"           # generation
EMBED_MODEL     = "nomic-embed-text" # embeddings 
EMBED_DIM       = 768

# Qdrant
COLLECTION_NAME  = "rag_docs"

# Chunking
CHUNK_BREAKPOINT = 90 

# Retrieval
RETRIEVAL_TOP_K = 15  # candidates from hybrid search before reranking
RERANK_TOP_N    = 3   # final chunks sent to LLM after reranking

# Reranker model (runs locally, no API needed)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Paths
DATA_RAW    = "data/raw"
DATA_PARSED = "data/parsed"

MAX_HISTORY_TURNS = 10