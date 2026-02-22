import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, documents):
        self.documents = documents
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        embeddings = self.embedder.encode(
            [doc["text"] for doc in documents],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype("float32")

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 5):
        q_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "id": idx,
                "text": self.documents[idx]["text"],
                "source": self.documents[idx]["source"],
                "score": float(score)
            })
        if results and results[0]["score"] < 0.20:
            return [] #hallucination control

        return results