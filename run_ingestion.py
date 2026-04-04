import sys
from parse import parse_all
from index import build_index

print("=" * 50)
print("Step 1/2 — Parsing PDFs with Docling")
print("=" * 50)
parsed = parse_all()

if not parsed:
    print("\nNo documents parsed. Add PDFs to data/raw/ and try again.")
    sys.exit(1)

print("\n" + "=" * 50)
print("Step 2/2 — Chunking + Embedding + Indexing into Qdrant")
print("=" * 50)
index = build_index()

if index is None:
    print("\nIndexing failed. Check errors above.")
    sys.exit(1)

print("\n" + "=" * 50)
print("Ingestion complete!")
print("=" * 50)