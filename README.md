# Production RAG Pipeline

A fully local Retrieval-Augmented Generation (RAG) system for querying PDF documents using hybrid search, cross-encoder reranking, and a local LLM.


## Tech Stack

| Component | Technology |
|---|---|
| LLM + Embeddings | Ollama (Llama 3.2 3B + nomic-embed-text) |
| Vector Store | Qdrant (hybrid dense + BM25) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| RAG Framework | LlamaIndex |
| PDF Parsing | Docling |
| API | FastAPI |
| Evaluation | RAGAS (Faithfulness, Relevancy, Precision, Recall) |

---

## Architecture

```
PDF → Docling Parser → Semantic Chunker → Ollama Embeddings → Qdrant
                                                                  ↓
User Query → Query Rewriter → Hybrid Search (Dense + BM25) → Cross-Encoder Reranker → LLM → Answer
```

**Key design decisions:**

- **Hybrid search** fuses dense vector similarity and BM25 keyword matching via Reciprocal Rank Fusion (RRF), giving better recall than either alone
- **Cross-encoder reranking** scores each (query, chunk) pair jointly, significantly improving precision over bi-encoder retrieval alone
- **Query rewriting** rewrites follow-up questions into standalone queries before retrieval, enabling accurate multi-turn conversation
- **Semantic chunking** uses embedding similarity to find natural breakpoints rather than fixed token counts, preserving context

---

## Prerequisites

- [Ollama](https://ollama.com/download) installed and running
- [Docker](https://www.docker.com/products/docker-desktop/) (for Qdrant only)
- Python 3.11+

---

## Setup

**1. Start Qdrant:**
```bash
docker run -d -p 6333:6333 qdrant/qdrant
```

**2. Start Ollama and pull models:**
```bash
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Pre-download the reranker model (optional but recommended):**
```bash
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

---

## Usage

**Add your PDFs:**
```
data/raw/your-document.pdf
```

**Run ingestion (parse → chunk → embed → index):**
```bash
python run_ingestion.py
```

**Quick test:**
```bash
python test.py
```

**Start the API:**
```bash
uvicorn main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/query` | Ask a question (stateless or with session) |
| POST | `/session/new` | Create a conversation session |
| DELETE | `/session/{id}` | Clear session history |
| GET | `/health` | Check Qdrant + Ollama status |

**Single query:**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is multi-head attention?"}'
```

**Multi-turn conversation:**
```bash
# Create session
curl -X POST http://localhost:8000/session/new

# Query with session
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Transformer?", "session_id": "your-session-id"}'

# Follow-up (query rewriting handles pronoun resolution automatically)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How many layers does it use?", "session_id": "your-session-id"}'
```

---

## Evaluation

RAGAS evaluation measures four metrics across a test suite of 25 question/answer pairs derived from the Attention Is All You Need paper:

| Metric | What it measures |
|---|---|
| Faithfulness | Is the answer grounded in retrieved context? Low = hallucination |
| Answer Relevancy | Does the answer address the question? Low = off-topic |
| Context Precision | Are retrieved chunks actually relevant? Low = noisy retrieval |
| Context Recall | Did retrieval find everything needed? Low = missing context |

**Run evaluation:**
```bash
python evaluate.py
```

Results are printed to terminal and saved to `evaluation/results.csv`.

**Tuning guide:**

| Low Score | Fix |
|---|---|
| Faithfulness | Tighten the grounded prompt in `generate.py` |
| Answer Relevancy | Try a larger model in `config.py` |
| Context Precision | Lower `RERANK_TOP_N` in `config.py` |
| Context Recall | Lower `CHUNK_BREAKPOINT` or raise `RETRIEVAL_TOP_K` in `config.py` |

---

## Configuration

All parameters are in `config.py`:

```python
OLLAMA_MODEL     = "llama3.2:3b"      # generation model
EMBED_MODEL      = "nomic-embed-text" # embedding model
CHUNK_BREAKPOINT = 85                 # semantic chunking sensitivity (lower = more chunks)
RETRIEVAL_TOP_K  = 15                 # candidates from hybrid search before reranking
RERANK_TOP_N     = 3                  # final chunks sent to LLM after reranking
MAX_HISTORY_TURNS = 10                # conversation turns to retain per session
```

---

## Project Structure

```
├── config.py          # Central configuration
├── parse.py           # PDF → clean markdown text via Docling
├── index.py           # Semantic chunking + embedding + Qdrant indexing
├── retriever.py       # Hybrid search + cross-encoder reranking
├── generate.py        # Query rewriting + grounded answer generation
├── run_ingestion.py   # Runs parse → index pipeline
├── main.py            # FastAPI server
├── evaluate.py        # RAGAS evaluation suite
├── test.py            # Quick pipeline smoke test
├── requirements.txt
└── data/
    ├── raw/           # Place PDFs here
    └── parsed/        # Auto-generated parsed JSON
```