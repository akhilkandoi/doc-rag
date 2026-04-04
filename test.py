from retriever import build_retriever, retrieve
from generate import generate

QUERY = "What is Attention?"  # <-- change this

print("Building retriever...")
retriever = build_retriever()

print(f"\nQuery: {QUERY}\n")
print("Retrieving...")
chunks = retrieve(QUERY, retriever)

print(f"Top {len(chunks)} chunks after reranking:\n")
for i, c in enumerate(chunks):
    print(f"  [{i+1}] {c['source']} (score: {c['score']:.3f})")
    print(f"       {c['text'][:150].strip()}...")
    print()

print("Generating answer...")
result = generate(QUERY, chunks)

print("\n" + "=" * 50)
print("ANSWER")
print("=" * 50)
print(result["answer"])
print(f"\nSources: {result['sources']}")