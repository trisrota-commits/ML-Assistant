from src.rag_core import query_rag

if __name__ == "__main__":
    while True:
        question = input("Ask: ")
        answer, retrieved = query_rag(question)

        print("\nAnswer:\n", answer)
        print("\nTop Retrieved IDs:\n", [r["id"] for r in retrieved])
        print("-" * 60)