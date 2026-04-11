# Local RAG Pipeline

A fully local Retrieval-Augmented Generation system for querying PDF documents. No API keys, no data leaves your machine. Built with hybrid search, cross-encoder reranking, and multi-turn conversation memory.

---

## Architecture

```
PDF → Docling Parser → Semantic Chunker → Ollama Embeddings → Qdrant
                                                                  ↓
User Query → Query Rewriter → Hybrid Search (Dense + BM25) → Cross-Encoder Reranker → LLM → Answer
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM + Embeddings | Ollama (`qwen2.5:1.5b` + `nomic-embed-text`) |
| Vector Store | Qdrant (hybrid dense + BM25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| RAG Framework | LlamaIndex |
| PDF Parsing | Docling |
| API | FastAPI |
| Evaluation | RAGAS |

---

## Key Design Decisions

**Hybrid search** fuses dense vector similarity and BM25 keyword matching via Reciprocal Rank Fusion (RRF). This gives better recall than either approach alone — dense search handles semantic meaning, BM25 handles exact terminology.

**Cross-encoder reranking** scores each (query, chunk) pair jointly rather than independently. This is significantly more precise than bi-encoder retrieval alone, at the cost of being slower — which is why it only runs on the top `RETRIEVAL_TOP_K` candidates, not the full index.

**Query rewriting** rewrites follow-up questions into standalone queries before retrieval. Without this, a question like "Who made it?" after asking about BERT would retrieve nothing useful. The LLM rewrites it to "Who created the BERT language model?" before hybrid search runs.

**Semantic chunking** splits documents on embedding similarity rather than fixed token counts. This preserves natural context boundaries and avoids splitting mid-argument.

---

## Project Structure

```
├── config.py          # Central configuration
├── parse.py           # PDF → clean text via Docling
├── index.py           # Semantic chunking + embedding + Qdrant indexing
├── retriever.py       # Hybrid search + cross-encoder reranking
├── generate.py        # Query rewriting + grounded answer generation
├── run_ingestion.py   # Orchestrates parse → index pipeline
├── main.py            # FastAPI server (with session memory)
├── evaluate.py        # RAGAS evaluation suite
├── test.py            # End-to-end smoke test
└── data/
    ├── raw/           # Place PDFs here
    └── parsed/        # Auto-generated JSON after ingestion
```

---

## API

Four endpoints served via FastAPI:

- `POST /query` — ask a question, optionally with a `session_id` for multi-turn memory
- `POST /session/new` — create a conversation session
- `DELETE /session/{id}` — clear session history
- `GET /health` — check Qdrant + Ollama connectivity

Sessions are held in memory. Up to 10 conversation turns are retained per session.

---

## Evaluation

RAGAS measures four metrics against a test suite of Q&A pairs:

| Metric | What it measures |
|---|---|
| Faithfulness | Is the answer grounded in retrieved context? Low = hallucination |
| Answer Relevancy | Does the answer address the question? Low = off-topic response |
| Context Precision | Are retrieved chunks actually relevant? Low = noisy retrieval |
| Context Recall | Did retrieval find everything needed? Low = missing context |

**Tuning guide:**

| Low score | Fix |
|---|---|
| Faithfulness | Tighten the grounded prompt in `generate.py` |
| Answer Relevancy | Try a larger model in `config.py` |
| Context Precision | Lower `RERANK_TOP_N` |
| Context Recall | Raise `RETRIEVAL_TOP_K` or lower `CHUNK_BREAKPOINT` |

---

## Configuration

All parameters in `config.py`:

| Parameter | Default | Effect |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:1.5b` | Generation LLM |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_BREAKPOINT` | `90` | Semantic chunking sensitivity — lower = more, smaller chunks |
| `RETRIEVAL_TOP_K` | `15` | Hybrid search candidates before reranking |
| `RERANK_TOP_N` | `3` | Final chunks passed to the LLM |
| `MAX_HISTORY_TURNS` | `10` | Conversation turns retained per session |