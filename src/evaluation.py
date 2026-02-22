from src.rag_core import retriever


# ---- Define Labeled Evaluation Set ----
# Replace IDs with the exact ones you identified earlier
def find_ids_by_keyword(keyword):
    ids = []
    for i, doc in enumerate(retriever.documents):
        if keyword.lower() in doc["text"].lower():
            ids.append(i)
    return ids


evaluation_set = [
    {
        "query": "How does D2L define a one-hidden-layer MLP?",
        "relevant_ids": find_ids_by_keyword("one-hidden-layer MLP")
    },
    {
        "query": "How is linear regression introduced?",
        "relevant_ids": find_ids_by_keyword("linear regression")
    },
    {
        "query": "What optimization algorithms does Ruder discuss?",
        "relevant_ids": find_ids_by_keyword("Adam")
    },
    {
        "query": "How do LSTMs prevent vanishing gradients?",
        "relevant_ids": find_ids_by_keyword("conveyor belt")
    }
]


# ---- Recall@K ----
def recall_at_k(eval_set, k=5):
    correct = 0

    for item in eval_set:
        results = retriever.retrieve(item["query"], k)
        retrieved_ids = [r["id"] for r in results]

        if any(rid in item["relevant_ids"] for rid in retrieved_ids):
            correct += 1

    return correct / len(eval_set)


# ---- MRR@K ----
def mrr(eval_set, k=5):
    reciprocal_ranks = []

    for item in eval_set:
        results = retriever.retrieve(item["query"], k)

        rank_score = 0
        for i, r in enumerate(results):
            if r["id"] in item["relevant_ids"]:
                rank_score = 1 / (i + 1)
                break

        reciprocal_ranks.append(rank_score)

    return sum(reciprocal_ranks) / len(eval_set)


# ---- Diagnostic Output ----
def detailed_report(eval_set, k=5):
    for item in eval_set:
        results = retriever.retrieve(item["query"], k)
        retrieved_ids = [r["id"] for r in results]

        print("Query:", item["query"])
        print("Retrieved:", retrieved_ids)
        print("Relevant:", item["relevant_ids"])
        print("Hit:", any(rid in item["relevant_ids"] for rid in retrieved_ids))
        print("-" * 60)


if __name__ == "__main__":
    print("Recall@5:", recall_at_k(evaluation_set, k=5))
    print("MRR@5:", mrr(evaluation_set, k=5))
    print("\nDetailed Per-Query Report\n")
    detailed_report(evaluation_set, k=5)