# generate.py
#
# Two functions:
#
#   rewrite_query()  — uses chat history to make the current question
#                      self-contained before retrieval.
#                      Example:
#                        History:  "Tell me about BERT"
#                        Question: "Who made it?"
#                        Rewritten: "Who created the BERT language model?"
#                      Without this, retrieval would search for "Who made it?"
#                      which is meaningless without context.
#
#   generate()       — builds a grounded prompt including chat history
#                      and calls the local Ollama LLM.

from llama_index.llms.ollama import Ollama
from config import OLLAMA_BASE_URL, OLLAMA_MODEL

_llm = Ollama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    request_timeout=120.0
)


def rewrite_query(question: str, history: list[dict], format_history_fn=None) -> str:
    """
    If there's no history, returns the question unchanged.
    If there is history, asks the LLM to rewrite the question
    as a standalone query that makes sense without context.
    This is called BEFORE retrieval so hybrid search gets a good query.
    """
    if not history:
        return question

    history_text = format_history_fn(history) if format_history_fn else ""

    prompt = f"""Given this conversation history and a follow-up question,
rewrite the follow-up question to be fully self-contained.
Only return the rewritten question — nothing else, no explanation.

Conversation history:
{history_text}

Follow-up question: {question}

Rewritten question:"""

    result = _llm.complete(prompt)
    rewritten = str(result).strip()

    # Safety fallback — if LLM returns something weird, use original
    if not rewritten or len(rewritten) > 500:
        return question

    return rewritten


def generate(query: str, chunks: list[dict], history: list[dict] = None, format_history_fn=None) -> dict:
    """
    Generates a grounded answer using retrieved chunks + conversation history.
    history: list of {"role": "user"|"assistant", "content": "..."} — can be empty
    """
    if history is None:
        history = []

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(f"[{i+1}] Source: {chunk['source']}\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    # Build conversation history block for the prompt
    history_block = ""
    if history and format_history_fn:
        history_block = f"""Previous conversation:
{format_history_fn(history)}

"""

    prompt = f"""You are a precise research assistant.
Answer the question using ONLY the context provided below.
You may refer to the previous conversation to understand what the user means,
but your answer must be grounded in the context documents.
If the answer is not in the context, say "I don't have enough information in the provided documents."

{history_block}CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""

    response = _llm.complete(prompt)

    return {
        "query": query,
        "answer": str(response).strip(),
        "contexts": [c["text"] for c in chunks],
        "sources": list(set(c["source"] for c in chunks))
    }