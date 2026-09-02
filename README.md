# ML RAG Assistant

A compact Retrieval-Augmented Generation (RAG) system for answering machine-learning questions from a curated collection of technical blog posts. It demonstrates document ingestion, tokenized chunking, dense-vector retrieval with FAISS, grounded generation with TinyLlama, retrieval evaluation, and an agentic retry path.

## What it does

1. Fetches paragraphs from the configured blog URLs.
2. Cleans short lines and selected acknowledgement noise.
3. Splits the text into overlapping windows and normalizes each window through the TinyLlama tokenizer.
4. Embeds chunks with `all-MiniLM-L6-v2` and indexes normalized vectors in FAISS.
5. Retrieves the most similar chunks for a question.
6. Generates a concise or detailed answer with TinyLlama.
7. Optionally routes the question through the agentic loop:
   - confident retrieval → grounded answer;
   - weak retrieval → one query rewrite and retry;
   - still weak → explicitly labeled ungrounded fallback.

The current source list in `src/rag_core.py` contains:

- Andrej Karpathy's recipe for training neural networks
- Sebastian Ruder's optimization article
- Christopher Olah's LSTM article

## Architecture

```text
blog URLs
   │
   ▼
fetch_clean_text → clean_text → chunk_text
                                      │
                                      ▼
                         MiniLM embeddings + FAISS
                                      │
question ────────────────────────────┘
   │
   ▼
agentic_query
   ├─ score ≥ 0.35 → grounded TinyLlama answer
   ├─ score < 0.35 → rewrite query → retrieve again
   └─ still weak → TinyLlama fallback marked [Ungrounded answer]
```

### Important implementation details

- `src/data_loader.py` fetches HTML with `requests` and extracts `<p>` elements with BeautifulSoup.
- `src/chunking.py` uses 400-word windows with 100-word overlap, then tokenizes and decodes each chunk using TinyLlama's tokenizer.
- `src/retrieval.py` normalizes embeddings and uses `faiss.IndexFlatIP`. With normalized vectors, inner product is equivalent to cosine similarity.
- `src/generation.py` runs `TinyLlama/TinyLlama-1.1B-Chat-v1.0` on CPU.
- `src/agent.py` uses deterministic score-based routing rather than relying on TinyLlama to emit a reliable tool-call JSON object.
- `app.py` keeps the original `/query` endpoint and adds `/agent-query`, which returns the answer, retrieved IDs, route, and decision trace.

## Setup

```bash
python -m pip install -r requirements.txt
```

The first run downloads the MiniLM and TinyLlama model files from Hugging Face. The application also fetches the configured blog pages at import/startup time, so network access is required on startup.

## Run the interfaces

### FastAPI

```bash
uvicorn app:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/
```

Original RAG endpoint:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"How does gradient descent work?"}'
```

Agentic endpoint:

```bash
curl -X POST http://127.0.0.1:8000/agent-query \
  -H "Content-Type: application/json" \
  -d '{"question":"How does gradient descent work?"}'
```

The agentic response includes `route` (`grounded` or `ungrounded`) and a `trace` showing the query, top score, and result count for each attempt.

### CLI

```bash
python main.py
```

The CLI prints the answer, retrieved IDs, selected route, and routing trace.

### Gradio

`hf_app.py` contains a Gradio interface. Install Gradio separately if it is not already present in your environment because it is not currently listed in `requirements.txt`:

```bash
python -m pip install gradio
python hf_app.py
```

## Evaluation

The repository includes `src/evaluation.py`, which computes Recall@K and MRR@K over a small keyword-derived evaluation set.

The README previously reported:

- Recall@5 ≈ 0.75
- MRR@5 ≈ 0.62

Those numbers should be treated as reported historical project results, not as a reproducible benchmark until the evaluation set and run are preserved and rerun. The current evaluation code derives relevant chunk IDs from keyword matches at startup, so it is useful as a diagnostic but not a rigorous held-out benchmark.

```bash
python src/evaluation.py
```

## Repository map

```text
app.py                 FastAPI application and HTTP endpoints
hf_app.py              Gradio interface
main.py                Interactive CLI
requirements.txt       Python dependencies
render.yaml            Render web-service configuration
src/data_loader.py     HTML fetching and text cleaning
src/chunking.py        Overlapping tokenizer-backed chunking
src/retrieval.py       MiniLM embeddings and FAISS search
src/generation.py      TinyLlama answer generation and query rewriting
src/rag_core.py        Document construction and original RAG path
src/agent.py           Score-based agentic retrieval loop
src/evaluation.py      Recall@K, MRR@K, and per-query diagnostics
tests/test_agent.py    Focused tests for agent routing

docs/INTERVIEW_PREP.md Interview learning guide and likely questions
```

## Interview preparation

See [`docs/INTERVIEW_PREP.md`](docs/INTERVIEW_PREP.md) for a repo-specific study guide covering the architecture, retrieval math, evaluation caveats, agentic routing, production trade-offs, debugging questions, and concise answers you can practice aloud.

## Known limitations and honest talking points

- The agent threshold (`0.35`) is a starting heuristic and has not been tuned against a preserved labeled validation set.
- The retriever has a separate low-score cutoff (`0.20`), so threshold behavior should be evaluated rather than assumed to be calibrated probability.
- The index and models are rebuilt in memory during startup; there is no persistent vector database or model-serving layer.
- Blog fetching happens during module import and has no explicit timeout or `raise_for_status()` check yet.
- The current evaluation set is small and keyword-derived, and the agentic path is not separately benchmarked.
- The fallback can answer from model knowledge, but the explicit label does not make that answer factually reliable.
- Source citations, document freshness tracking, authentication, rate limiting, and prompt-injection defenses are not implemented.
