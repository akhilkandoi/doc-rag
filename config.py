# Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL    = "llama3.2:3b"           # generation
EMBED_MODEL     = "nomic-embed-text" # embeddings 
EMBED_DIM       = 768

# Qdrant
QDRANT_URL       = "http://localhost:6333"
COLLECTION_NAME  = "rag_docs"

# Chunking
CHUNK_BREAKPOINT = 85 

# Retrieval
RETRIEVAL_TOP_K = 15  # candidates from hybrid search before reranking
RERANK_TOP_N    = 3   # final chunks sent to LLM after reranking

# Reranker model (runs locally, no API needed)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Paths
DATA_RAW    = "data/raw"
DATA_PARSED = "data/parsed"

MAX_HISTORY_TURNS = 10