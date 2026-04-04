# evaluation/evaluate.py
# Measures your pipeline quality with 4 RAGAS metrics:
#
#   Faithfulness      — is the answer grounded in the retrieved context?
#                       low score = LLM is hallucinating
#
#   Answer Relevancy  — does the answer address the question?
#                       low score = answer is off-topic or evasive
#
#   Context Precision — are the retrieved chunks actually relevant?
#                       low score = retrieval is noisy, lower RERANK_TOP_N
#
#   Context Recall    — did retrieval find everything needed to answer?
#                       low score = raise RETRIEVAL_TOP_K or lower CHUNK_BREAKPOINT
#
# HOW TO USE:
#   1. Fill in TEST_CASES below with Q&A pairs from your actual documents
#   2. Run: python evaluation/evaluate.py
#   3. Scores printed to terminal + saved to evaluation/results.csv

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import build_retriever, retrieve
from generate import generate

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama as LangchainOllama
from langchain_community.embeddings import OllamaEmbeddings as LangchainOllamaEmbeddings
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, EMBED_MODEL

# ================================================================
# FILL THESE IN — use questions answerable from your actual PDFs
# Aim for 20-30 pairs. Quality matters more than quantity.
# ================================================================
TEST_CASES = [
    {
        "question": "What is the main contribution of this paper?",
        "ground_truth": "Replace this with the actual answer from your document."
    },
    {
        "question": "What dataset was used for evaluation?",
        "ground_truth": "Replace this with the actual answer from your document."
    },
    # Add more here...
]
# ================================================================


def run_evaluation():
    print("Building retriever...")
    retriever = build_retriever()

    print(f"\nRunning {len(TEST_CASES)} test cases...\n")
    eval_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for i, tc in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] {tc['question'][:70]}...")
        try:
            chunks = retrieve(tc["question"], retriever)
            result = generate(tc["question"], chunks)

            eval_data["question"].append(tc["question"])
            eval_data["answer"].append(result["answer"])
            eval_data["contexts"].append(result["contexts"])
            eval_data["ground_truth"].append(tc["ground_truth"])
        except Exception as e:
            print(f"  Error: {e} — skipping this question")

    if not eval_data["question"]:
        print("No results to evaluate. Check your pipeline.")
        return

    print("\nRunning RAGAS scoring (uses Ollama as judge — takes a few minutes)...")

    # RAGAS needs LangChain-wrapped versions of the models
    judge_llm = LangchainLLMWrapper(
        LangchainOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        LangchainOllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBED_MODEL)
    )

    scores = evaluate(
        Dataset.from_dict(eval_data),
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],
        llm=judge_llm,
        embeddings=judge_embeddings
    )

    # Print results
    print("\n" + "=" * 55)
    print("RAGAS SCORES")
    print("=" * 55)
    print(f"  Faithfulness:      {scores['faithfulness']:.3f}")
    print(f"  Answer Relevancy:  {scores['answer_relevancy']:.3f}")
    print(f"  Context Precision: {scores['context_precision']:.3f}")
    print(f"  Context Recall:    {scores['context_recall']:.3f}")
    print("=" * 55)
    print("  Target: > 0.75 solid  |  > 0.85 interview-worthy")
    print()
    print("  Low Faithfulness?      → tighten the prompt in generation/generate.py")
    print("  Low Answer Relevancy?  → LLM going off-topic, try a stronger model")
    print("  Low Context Precision? → too much noise, lower RERANK_TOP_N in config.py")
    print("  Low Context Recall?    → raise RETRIEVAL_TOP_K or lower CHUNK_BREAKPOINT")
    print("=" * 55)

    # Save per-question breakdown
    os.makedirs("evaluation", exist_ok=True)
    df = scores.to_pandas()
    df.to_csv("evaluation/results.csv", index=False)
    print("\nPer-question breakdown saved to evaluation/results.csv")


if __name__ == "__main__":
    run_evaluation()