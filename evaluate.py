# evaluation/evaluate.py
# Measures your pipeline quality with 4 DeepEval RAG metrics:
#
#   Faithfulness          — is the answer grounded in the retrieved context?
#                           low score = LLM is hallucinating
#
#   Answer Relevancy      — does the answer address the question?
#                           low score = answer is off-topic or evasive
#
#   Contextual Precision  — are the retrieved chunks actually relevant?
#                           low score = retrieval is noisy, lower RERANK_TOP_N
#
#   Contextual Recall     — did retrieval find everything needed to answer?
#                           low score = raise RETRIEVAL_TOP_K or lower CHUNK_BREAKPOINT
#
# HOW TO USE:
#   1. pip install deepeval
#   2. Fill in TEST_CASES below with Q&A pairs from your actual documents
#   3. Run: python evaluation/evaluate.py
#   4. Scores printed to terminal + saved to evaluation/results.csv

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
from retriever import build_retriever, retrieve
from generate import generate

from deepeval.models import OllamaModel
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate as deepeval_evaluate
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

# ================================================================
# FILL THESE IN — use questions answerable from your actual PDFs
# Aim for 20-30 pairs. Quality matters more than quantity.
# ================================================================
TEST_CASES = [
    {
        "question": "What is the main contribution of the Transformer paper?",
        "ground_truth": "The Transformer is a novel network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely, which achieves superior translation quality while being more parallelizable and requiring significantly less time to train."
    },
    {
        "question": "What were the two machine translation tasks used to evaluate the Transformer?",
        "ground_truth": "The model was evaluated on the WMT 2014 English-to-German translation task and the WMT 2014 English-to-French translation task."
    },
    {
        "question": "What BLEU score did the Transformer achieve on English-to-German translation?",
        "ground_truth": "The Transformer achieved 28.4 BLEU on the WMT 2014 English-to-German translation task, outperforming all previously published models including ensembles."
    },
    {
        "question": "What BLEU score did the Transformer achieve on English-to-French translation?",
        "ground_truth": "The Transformer achieved a BLEU score of 41.0 on the WMT 2014 English-to-French translation task, outperforming all previously published single models at less than 1/4 the training cost of the previous state-of-the-art model."
    },
    {
        "question": "What is multi-head attention?",
        "ground_truth": "Multi-head attention linearly projects queries, keys and values h times with different learned projections, performs attention in parallel on each projection, then concatenates and projects the results. This allows the model to jointly attend to information from different representation subspaces at different positions."
    },
    {
        "question": "What are the three ways attention is used in the Transformer model?",
        "ground_truth": "Attention is used in three ways: encoder-decoder attention where queries come from the decoder and keys and values come from the encoder output; encoder self-attention where each position attends to all positions in the encoder; and decoder self-attention where each position attends to all positions up to and including that position."
    },
    {
        "question": "Why do the authors scale the dot product in attention?",
        "ground_truth": "The authors scale by the square root of the dimension of the keys because for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients."
    },
    {
        "question": "What is the difference between dot-product attention and additive attention?",
        "ground_truth": "Dot-product attention multiplies queries and keys and scales by the square root of their dimension. Additive attention computes compatibility using a feed-forward network with a single hidden layer. While both are similar in theoretical complexity, dot-product attention is faster and more space-efficient in practice."
    },
    {
        "question": "What is positional encoding and why is it needed?",
        "ground_truth": "Positional encodings are added to input embeddings to inject information about the position of tokens in the sequence, since the Transformer contains no recurrence or convolution and would otherwise have no way to make use of the order of the sequence."
    },
    {
        "question": "What functions are used for positional encoding?",
        "ground_truth": "Sine and cosine functions of different frequencies are used: PE(pos,2i) = sin(pos/10000^(2i/dmodel)) and PE(pos,2i+1) = cos(pos/10000^(2i/dmodel)), where pos is the position and i is the dimension."
    },
    {
        "question": "How many layers does the encoder and decoder each have in the base Transformer model?",
        "ground_truth": "Both the encoder and decoder are composed of a stack of 6 identical layers."
    },
    {
        "question": "What are the two sub-layers in each encoder layer?",
        "ground_truth": "Each encoder layer has two sub-layers: a multi-head self-attention mechanism and a position-wise fully connected feed-forward network. A residual connection is applied around each sub-layer, followed by layer normalization."
    },
    {
        "question": "What is the dimensionality of the model and feed-forward layers in the base model?",
        "ground_truth": "The base model uses dmodel of 512 and the inner feed-forward layer has dimensionality dff of 2048."
    },
    {
        "question": "How many attention heads does the base Transformer use?",
        "ground_truth": "The base Transformer model uses 8 parallel attention heads."
    },
    {
        "question": "What optimizer and learning rate schedule was used to train the Transformer?",
        "ground_truth": "The Adam optimizer was used with beta1=0.9, beta2=0.98, epsilon=1e-9. The learning rate increases linearly for the first warmup_steps training steps then decreases proportionally to the inverse square root of the step number, with 4000 warmup steps."
    },
    {
        "question": "What regularization techniques were used during training?",
        "ground_truth": "Three types of regularization were used: residual dropout applied to the output of each sub-layer and the embedding sums with a rate of 0.1; label smoothing of value 0.1 was employed during training."
    },
    {
        "question": "How long did the base and big Transformer models take to train?",
        "ground_truth": "The base models were trained for 100,000 steps taking about 12 hours. The big model was trained for 300,000 steps taking 3.5 days, both on 8 P100 GPUs."
    },
    {
        "question": "How does the Transformer compare to recurrent models in terms of computational complexity?",
        "ground_truth": "Self-attention layers connect all positions with a constant number of sequential operations, whereas recurrent layers require O(n) sequential operations. For sequences shorter than the representation dimensionality, self-attention is faster than recurrent layers."
    },
    {
        "question": "What tasks beyond translation did the authors test the Transformer on?",
        "ground_truth": "The authors tested the Transformer on English constituency parsing, achieving competitive results with the Berkeley Parser when trained only on the Wall Street Journal training set of 40,000 sentences."
    },
    {
        "question": "What is label smoothing and how did it affect training?",
        "ground_truth": "Label smoothing of value 0.1 was used, which hurts perplexity as the model learns to be more unsure, but improves BLEU score and accuracy."
    },
    {
        "question": "What byte-pair encoding vocabulary size was used?",
        "ground_truth": "The English-German dataset used a shared vocabulary of about 37,000 tokens using byte-pair encoding. The English-French dataset used a larger shared vocabulary of 32,000 word-piece tokens."
    },
    {
        "question": "How does attention help with learning long-range dependencies?",
        "ground_truth": "The path length between any two positions in the sequence is constant O(1) with self-attention, compared to O(n) for recurrent networks, making it easier to learn dependencies between distant positions."
    },
    {
        "question": "Who are the authors of the Attention Is All You Need paper?",
        "ground_truth": "The authors are Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin."
    },
    {
        "question": "What is residual dropout and where is it applied?",
        "ground_truth": "Residual dropout is applied to the output of each sub-layer before it is added to the sub-layer input and normalized. It is also applied to the sums of the embeddings and positional encodings in both the encoder and decoder stacks, with a rate of Pdrop = 0.1 for the base model."
    },
    {
        "question": "What is the purpose of the decoder's masking in self-attention?",
        "ground_truth": "Masking is used in the decoder self-attention to prevent positions from attending to subsequent positions, ensuring that predictions for position i can depend only on the known outputs at positions less than i, preserving the auto-regressive property."
    },
]
# ================================================================

# Score threshold — a test case is considered "passing" at or above this value.
# DeepEval defaults to 0.5; raise to 0.7+ for stricter evaluation.
THRESHOLD = 0.5


def fmt(val):
    return f"{val:.3f}" if val is not None else "N/A"


def run_evaluation():
    # --- Judge model (Ollama, no API key needed) ---
    # Strip trailing /v1 from OLLAMA_BASE_URL if present — OllamaModel wants the bare host.
    ollama_base = OLLAMA_BASE_URL.rstrip("/").removesuffix("/v1")
    judge = OllamaModel(
        model=OLLAMA_MODEL,     # e.g. "qwen2.5:1.5b"
        base_url=ollama_base,   # e.g. "http://localhost:11434"
        temperature=0,          # deterministic scoring
        timeout=300,            # 5 min per call — prevents TimeoutError on slow hardware
    )

    # --- Metrics ---
    metrics = [
        FaithfulnessMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        AnswerRelevancyMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualPrecisionMetric(threshold=THRESHOLD, model=judge, include_reason=True),
        ContextualRecallMetric(threshold=THRESHOLD, model=judge, include_reason=True),
    ]

    print("Building retriever...")
    retriever = build_retriever()

    print(f"\nRunning {len(TEST_CASES)} test cases...\n")
    test_cases = []

    for i, tc in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] {tc['question'][:70]}...")
        try:
            chunks = retrieve(tc["question"], retriever)
            result = generate(tc["question"], chunks)

            test_cases.append(LLMTestCase(
                input=tc["question"],
                actual_output=result["answer"],
                retrieval_context=result["contexts"],  # list[str]
                expected_output=tc["ground_truth"],    # needed for Recall & Precision
            ))
        except Exception as e:
            print(f"  Error: {e} — skipping this question")

    if not test_cases:
        print("No results to evaluate. Check your pipeline.")
        return

    print(f"\nRunning DeepEval scoring with Ollama judge ({OLLAMA_MODEL})...\n")
    print("Evaluating one test case at a time to avoid overwhelming Ollama with")
    print("parallel requests (which causes the TimeoutError you saw before).\n")

    # We evaluate each test case individually rather than passing all at once.
    # deepeval's default runs metrics concurrently via asyncio.gather — Ollama
    # can only handle one request at a time, so parallel calls pile up and time out.
    # Looping one-by-one is the most version-compatible fix (no kwargs needed).
    metric_names = ["faithfulness", "answer_relevancy", "contextual_precision", "contextual_recall"]
    aggregated = {name: [] for name in metric_names}
    all_test_results = []

    for i, tc in enumerate(test_cases):
        print(f"  Scoring [{i+1}/{len(test_cases)}] {tc.input[:65]}...")
        try:
            result = deepeval_evaluate([tc], metrics)
            all_test_results.append(result.test_results[0])
            for m in result.test_results[0].metrics_data:
                key = m.name.lower().replace(" ", "_")
                if key in aggregated and m.score is not None:
                    aggregated[key].append(m.score)
        except Exception as e:
            print(f"    Scoring error: {e} — skipping")

    def avg(scores):
        return sum(scores) / len(scores) if scores else None

    faithfulness     = avg(aggregated["faithfulness"])
    answer_relevancy = avg(aggregated["answer_relevancy"])
    ctx_precision    = avg(aggregated["contextual_precision"])
    ctx_recall       = avg(aggregated["contextual_recall"])

    print("\n" + "=" * 55)
    print("DEEPEVAL SCORES")
    print("=" * 55)
    print(f"  Faithfulness:          {fmt(faithfulness)}")
    print(f"  Answer Relevancy:      {fmt(answer_relevancy)}")
    print(f"  Contextual Precision:  {fmt(ctx_precision)}")
    print(f"  Contextual Recall:     {fmt(ctx_recall)}")
    print("=" * 55)
    print("  Target: > 0.75 solid")
    print()
    print("  Low Faithfulness?         → tighten the prompt in generate.py")
    print("  Low Answer Relevancy?     → LLM going off-topic, try a stronger model")
    print("  Low Contextual Precision? → too much noise, lower RERANK_TOP_N in config.py")
    print("  Low Contextual Recall?    → raise RETRIEVAL_TOP_K or lower CHUNK_BREAKPOINT")
    print("=" * 55)

    # --- Save per-question breakdown to CSV ---
    os.makedirs("evaluation", exist_ok=True)
    csv_path = "evaluation/results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["question", "answer", "faithfulness", "answer_relevancy",
                         "contextual_precision", "contextual_recall"])
        for tc_obj, tc_result in zip(test_cases, all_test_results):
            row_scores = {m.name.lower().replace(" ", "_"): m.score
                          for m in tc_result.metrics_data}
            writer.writerow([
                tc_obj.input,
                tc_obj.actual_output,
                row_scores.get("faithfulness"),
                row_scores.get("answer_relevancy"),
                row_scores.get("contextual_precision"),
                row_scores.get("contextual_recall"),
            ])

    print(f"\nPer-question breakdown saved to {csv_path}")


if __name__ == "__main__":
    run_evaluation()