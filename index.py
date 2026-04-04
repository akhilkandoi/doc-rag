# Chunks + embeds + indexes into Qdrant
# Each doc's chunks are stored into Qdrant immediately, then memory is freed

import json
import pathlib

import qdrant_client
from llama_index.core import Document, Settings, StorageContext
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL, EMBED_MODEL,
    QDRANT_URL, COLLECTION_NAME, DATA_PARSED, CHUNK_BREAKPOINT
)


def build_index():
    Settings.llm = Ollama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120.0
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL
    )

    parsed_files = list(pathlib.Path(DATA_PARSED).glob("*.json"))
    if not parsed_files:
        print(f"No parsed files in {DATA_PARSED}/ — run ingestion/parse.py first.")
        return None

    client = qdrant_client.QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        enable_hybrid=True,
        fastembed_sparse_model="Qdrant/bm25"
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Chunker reused across all docs
    splitter = SemanticSplitterNodeParser(
        embed_model=Settings.embed_model,
        breakpoint_percentile_threshold=CHUNK_BREAKPOINT
    )

    total_chunks = 0
    
    # load 1 doc → chunk it → embed + store → move to next
    # Each document's chunks fit comfortably in the embedding model's window.
    for i, json_path in enumerate(parsed_files):
        data = json.loads(json_path.read_text())
        print(f"[{i+1}/{len(parsed_files)}] {data['source']} ...", end=" ", flush=True)

        doc = Document(
            text=data["text"],
            metadata={"source": data["source"]}
        )

        # Chunk this one document
        nodes = splitter.get_nodes_from_documents([doc], show_progress=False)

        if not nodes:
            print("no content, skipping")
            continue

        # Embed and store immediately — storage_context adds to the existing
        # Qdrant collection without wiping it, so docs accumulate correctly
        VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            show_progress=False
        )

        total_chunks += len(nodes)
        avg = sum(len(n.text.split()) for n in nodes) // len(nodes)
        print(f"{len(nodes)} chunks (avg {avg} words)")

    print(f"\nDone — {total_chunks} chunks across {len(parsed_files)} docs in Qdrant.")
    print("Verify: http://localhost:6333/dashboard")
    return True


if __name__ == "__main__":
    build_index()