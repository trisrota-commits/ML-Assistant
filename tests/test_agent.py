import unittest

from src.agent import agentic_query


class FakeRetriever:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.queries = []

    def retrieve(self, query, k):
        self.queries.append((query, k))
        return next(self.responses)


class AgenticQueryTests(unittest.TestCase):
    def test_confident_result_answers_with_context_without_rewrite(self):
        retriever = FakeRetriever([[{"id": 1, "text": "context", "score": 0.8}]])
        calls = []

        def answer(question, context, mode, grounded):
            calls.append((question, context, mode, grounded))
            return "grounded answer"

        answer_text, results, metadata = agentic_query(
            "original",
            answer_generator=answer,
            query_rewriter=lambda _: "rewritten",
            retriever_instance=retriever,
        )

        self.assertEqual(answer_text, "grounded answer")
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(metadata["route"], "grounded")
        self.assertEqual(retriever.queries, [("original", 3)])
        self.assertEqual(calls, [("original", "context", "concise", True)])

    def test_weak_result_rewrites_and_retries(self):
        retriever = FakeRetriever([
            [{"id": 1, "text": "weak", "score": 0.2}],
            [{"id": 2, "text": "strong", "score": 0.6}],
        ])
        calls = []

        def answer(question, context, mode, grounded):
            calls.append((question, context, grounded))
            return "answer"

        result = agentic_query(
            "original",
            retriever_instance=retriever,
            answer_generator=answer,
            query_rewriter=lambda _: "rewritten",
        )

        self.assertEqual(result[2]["route"], "grounded")
        self.assertEqual(retriever.queries, [("original", 3), ("rewritten", 3)])
        self.assertEqual(calls[-1], ("rewritten", "strong", True))

    def test_still_weak_falls_back_without_context(self):
        retriever = FakeRetriever([
            [{"id": 1, "text": "weak", "score": 0.2}],
            [],
        ])
        calls = []

        def answer(question, context, mode, grounded):
            calls.append((question, context, grounded))
            return "ungrounded answer"

        answer_text, results, metadata = agentic_query(
            "original",
            retriever_instance=retriever,
            answer_generator=answer,
            query_rewriter=lambda _: "rewritten",
        )

        self.assertEqual(answer_text, "[Ungrounded answer] ungrounded answer")
        self.assertEqual(results, [])
        self.assertEqual(metadata["route"], "ungrounded")
        self.assertEqual(calls, [("rewritten", "", False)])


if __name__ == "__main__":
    unittest.main()
