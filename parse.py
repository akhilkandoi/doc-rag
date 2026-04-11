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

import math
from retriever import build_retriever, retrieve
from generate import generate

# FIX 1: Replaced `from datasets import Dataset` + dict-based schema with
#         ragas-native EvaluationDataset + SingleTurnSample (v0.4 API).
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import llm_factory
# FIX 2: Corrected embedding_factory import path (was ragas.embeddings, now ragas.embeddings.base)
from ragas.embeddings.base import embedding_factory
# FIX 3: Import RunConfig so it can be passed as a proper object, not a plain dict
from ragas.run_config import RunConfig
from config import OLLAMA_BASE_URL, OLLAMA_MODEL, EMBED_MODEL

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

# How many RAGAS scoring jobs run in parallel.
# Lower this if you see TimeoutErrors with a local Ollama judge.
# 1 = fully sequential (slowest but most reliable for slow hardware)
RAGAS_CONCURRENCY = 1


def safe_score(scores, key):
    """
    RAGAS returns NaN (or a list containing NaN) when jobs time out or fail.
    This helper extracts a clean float, or returns None so we can print 'N/A'.
    """
    val = scores[key]
    # Newer RAGAS versions may return a list of per-row scores
    if isinstance(val, list):
        valid = [v for v in val if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return sum(valid) / len(valid) if valid else None
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    return float(val)


def fmt(val):
    return f"{val:.3f}" if val is not None else "N/A (all jobs failed)"


def run_evaluation():
    print("Building retriever...")
    retriever = build_retriever()

    print(f"\nRunning {len(TEST_CASES)} test cases...\n")

    # FIX 1: Build a list of SingleTurnSample objects instead of a plain dict.
    # Field names have changed in v0.4:
    #   question      → user_input
    #   answer        → response
    #   contexts      → retrieved_contexts
    #   ground_truth  → reference
    samples = []

    for i, tc in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] {tc['question'][:70]}...")
        try:
            chunks = retrieve(tc["question"], retriever)
            result = generate(tc["question"], chunks)

            samples.append(SingleTurnSample(
                user_input=tc["question"],
                response=result["answer"],
                retrieved_contexts=result["contexts"],
                reference=tc["ground_truth"],
            ))
        except Exception as e:
            print(f"  Error: {e} — skipping this question")

    if not samples:
        print("No results to evaluate. Check your pipeline.")
        return

    evaluation_dataset = EvaluationDataset(samples=samples)

    print(f"\nRunning RAGAS scoring (concurrency={RAGAS_CONCURRENCY}, "
          f"uses Ollama as judge — takes a few minutes)...")

    # Ollama exposes an OpenAI-compatible REST API at /v1.
    # The 'openai' package here is just a local HTTP client — no data leaves your machine.
    # Run: pip install openai   (one-time, no account or key needed)
    from openai import OpenAI
    ollama_client = OpenAI(
        base_url=f"{OLLAMA_BASE_URL.rstrip('/')}/v1",
        api_key="ollama",          # required field; Ollama ignores the value
    )
    judge_llm = llm_factory(OLLAMA_MODEL, provider="openai", client=ollama_client)
    judge_embeddings = embedding_factory(EMBED_MODEL, provider="openai", client=ollama_client)

    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]

    scores = evaluate(
        evaluation_dataset,
        metrics=metrics,
        # FIX 3: Pass RunConfig as a proper object, not a plain dict
        run_config=RunConfig(max_workers=RAGAS_CONCURRENCY),
    )

    faithfulness     = safe_score(scores, "faithfulness")
    answer_relevancy = safe_score(scores, "answer_relevancy")
    ctx_precision    = safe_score(scores, "context_precision")
    ctx_recall       = safe_score(scores, "context_recall")

    print("\n" + "=" * 55)
    print("RAGAS SCORES")
    print("=" * 55)
    print(f"  Faithfulness:      {fmt(faithfulness)}")
    print(f"  Answer Relevancy:  {fmt(answer_relevancy)}")
    print(f"  Context Precision: {fmt(ctx_precision)}")
    print(f"  Context Recall:    {fmt(ctx_recall)}")
    print("=" * 55)
    print("  Target: > 0.75 solid  |  > 0.85 interview-worthy")
    print()
    print("  Low Faithfulness?      → tighten the prompt in generation/generate.py")
    print("  Low Answer Relevancy?  → LLM going off-topic, try a stronger model")
    print("  Low Context Precision? → too much noise, lower RERANK_TOP_N in config.py")
    print("  Low Context Recall?    → raise RETRIEVAL_TOP_K or lower CHUNK_BREAKPOINT")
    print()
    if any(v is None for v in [faithfulness, answer_relevancy, ctx_precision, ctx_recall]):
        print("  ⚠  Some metrics are N/A — most scoring jobs timed out.")
        print("     Try: set RAGAS_CONCURRENCY = 1 (already default) and ensure")
        print("     Ollama is running and responding within ~60 s per call.")
    print("=" * 55)

    # Save per-question breakdown
    os.makedirs("evaluation", exist_ok=True)
    df = scores.to_pandas()
    df.to_csv("evaluation/results.csv", index=False)
    print("\nPer-question breakdown saved to evaluation/results.csv")


if __name__ == "__main__":
    run_evaluation()