ML RAG Assistant:

A production-focused Retrieval-Augmented Generation system built over technical ML blogs (D2L, Karpathy, Ruder, Colah). Demonstrates practical RAG engineering: retrieval, evaluation, hallucination control, and API deployment.
Overview
What it does: Semantic search + grounded answer generation over curated ML content
Why it matters: Shows evaluation-driven system design, not just "RAG in 10 lines"

Scrapes and chunks technical blog content with token-aware splitting
FAISS similarity search with all-MiniLM-L6-v2 embeddings
Grounded generation using TinyLlama (1.1B, CPU-friendly)
Quantitative evaluation: Recall@5 ≈ 0.75, MRR@5 ≈ 0.62
Hallucination mitigation via similarity thresholding
FastAPI + Gradio interface

Tech Stack
LayerChoiceRationaleEmbeddingall-MiniLM-L6-v2Fast, 384-dim, good for semantic similarityIndexFAISS IndexFlatIPExact search, L2-normalized for cosine simLLMTinyLlama-1.1BCPU-friendly, 2K context, decent instruction followingAPIFastAPI + GradioREST endpoints + demo UI
Evaluation
Recall@5: 0.75 | MRR@5: 0.62
Tested on chunk-level labeled queries. 75% of relevant chunks retrieved in top-5; relevant chunks typically rank ~2nd.
Key failure mode: Dense retrieval struggles with optimization-heavy content (Ruder's SGD sections) where lexical matching would help → motivates hybrid retrieval roadmap.
Key Engineering Decisions
Cosine similarity via normalized embeddings: L2 normalization → cosine = dot product (FAISS IndexFlatIP)
Context window management: TinyLlama's 2048 token limit → k=2 retrieval + aggressive truncation
Hallucination mitigation: Similarity threshold rejects low-confidence queries outside domain
Example API call:
bashcurl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "How do LSTMs prevent vanishing gradients?"}'
Returns: {answer, retrieved_chunks, confidence}
Why This Project Matters
Most RAG tutorials show "embed + retrieve + generate" in 20 lines. This project demonstrates:
✅ Evaluation methodology - Quantitative IR metrics, not just vibes
✅ Failure analysis - Documented ranking issues → actionable improvements
✅ Resource constraints - CPU inference, context limits, real tradeoffs
✅ Production patterns - API design, error handling, hallucination control
✅ Systems thinking - Modular architecture, not a monolithic notebook
Roadmap

 Cross-encoder reranking (reorder top-k with powerful model)
 Hybrid retrieval (BM25 + dense embeddings)
 Model quantization (INT8 TinyLlama for 2-3x speedup)
 Expanded evaluation set (500+ labeled queries)


Content sources: D2L · Karpathy · Ruder · Colah
