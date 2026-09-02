from src.agent import agentic_query

if __name__ == "__main__":
    while True:
        question = input("Ask: ")
        answer, retrieved, metadata = agentic_query(question)

        print("\nAnswer:\n", answer)
        print("\nTop Retrieved IDs:\n", [r["id"] for r in retrieved])
        print("\nRoute:\n", metadata["route"])
        print("\nTrace:\n", metadata["trace"])
        print("-" * 60)