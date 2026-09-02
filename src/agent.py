"""Agentic retrieval and generation loop."""

from typing import Callable, Optional


CONFIDENCE_THRESHOLD = 0.35
MAX_REWRITE_ATTEMPTS = 1


def _rewrite_query(question: str) -> str:
    """Rewrite a weak query using the configured language model."""
    from src.generation import rewrite_query

    return rewrite_query(question)


def _generate_answer(question: str, context: str, mode: str, grounded: bool) -> str:
    """Generate an answer lazily so importing this module stays lightweight."""
    from src.generation import generate_answer

    return generate_answer(question, context, mode, grounded=grounded)


def agentic_query(
    question: str,
    mode: str = "concise",
    retriever_instance=None,
    answer_generator: Optional[Callable] = None,
    query_rewriter: Optional[Callable] = None,
):
    """Retrieve, retry once after a weak match, then answer with a trace.

    Dependency arguments are injectable to make the routing behavior testable
    without loading FAISS or TinyLlama.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    if retriever_instance is None:
        from src.rag_core import retriever as retriever_instance

    answer_generator = answer_generator or _generate_answer
    query_rewriter = query_rewriter or _rewrite_query
    trace = []
    current_question = question.strip()

    for attempt in range(MAX_REWRITE_ATTEMPTS + 1):
        results = retriever_instance.retrieve(current_question, k=3)
        top_score = results[0]["score"] if results else 0.0
        trace.append({
            "attempt": attempt,
            "query": current_question,
            "top_score": float(top_score),
            "result_count": len(results),
        })

        if results and top_score >= CONFIDENCE_THRESHOLD:
            context = "\n".join(result["text"] for result in results)
            answer = answer_generator(current_question, context, mode, grounded=True)
            return answer, results, {"route": "grounded", "trace": trace}

        if attempt < MAX_REWRITE_ATTEMPTS:
            rewritten = query_rewriter(current_question).strip()
            if rewritten and rewritten != current_question:
                current_question = rewritten
                continue

        answer = answer_generator(current_question, "", mode, grounded=False)
        return f"[Ungrounded answer] {answer}", [], {
            "route": "ungrounded",
            "trace": trace,
        }
