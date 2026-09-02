# ML Assistant Interview Preparation Guide

This is a practical study guide for explaining this repository in an interview. It is intentionally tied to the code that exists today, including its trade-offs and limitations. Do not describe planned improvements as if they are already implemented.

## How to use this guide

Practice answering the questions aloud before looking at the suggested answer. A strong interview answer usually has this shape:

1. State what the system does.
2. Explain the design choice.
3. Name the trade-off or failure mode.
4. Say how you would measure or improve it.

For example:

> “I use normalized MiniLM embeddings with FAISS inner-product search because that gives cosine-similarity retrieval with a simple exact index. The trade-off is that `IndexFlatIP` is linear in the number of stored chunks, so for a much larger corpus I would benchmark an approximate index or a vector database.”

That answer is stronger than saying only “I used FAISS.”

---

## 1. The 60-second project explanation

> “This project is a small RAG assistant for machine-learning blog content. At startup, it fetches paragraph text from three configured blog URLs, cleans it, splits it into overlapping windows, embeds the chunks with `all-MiniLM-L6-v2`, and stores the normalized vectors in a FAISS inner-product index. For a user question, the system retrieves the most similar chunks and gives them to TinyLlama to generate a concise or detailed answer. I also added an agentic routing layer: if the top retrieval score is at least `0.35`, it produces a grounded answer; if the score is weak, it rewrites the query once and retries; if retrieval is still weak, it produces a response explicitly marked `[Ungrounded answer]`. The API is exposed through FastAPI, and the repository includes a small Recall@K/MRR@K diagnostic. The main limitations are that the threshold is heuristic, the evaluation set is small and keyword-derived, and startup currently rebuilds the corpus and models in memory.”

### Shorter version

> “It is a CPU-friendly, modular RAG application: HTML ingestion, cleaning, overlapping chunking, normalized MiniLM embeddings, exact FAISS retrieval, TinyLlama generation, and score-based retry/fallback routing behind FastAPI.”

---

## 2. Read the repository in this order

| Order | File | What to learn |
|---|---|---|
| 1 | `src/rag_core.py` | The original document-building and RAG path |
| 2 | `src/data_loader.py` | How HTML becomes raw text |
| 3 | `src/chunking.py` | Chunk size, overlap, and tokenizer use |
| 4 | `src/retrieval.py` | Embeddings, normalization, FAISS, scores |
| 5 | `src/generation.py` | TinyLlama prompts and decoding |
| 6 | `src/agent.py` | The routing, retry, trace, and fallback behavior |
| 7 | `app.py` | HTTP API behavior and response shape |
| 8 | `src/evaluation.py` | Recall/MRR implementation and its caveats |
| 9 | `tests/test_agent.py` | How the agent loop is tested without model downloads |
| 10 | `render.yaml` | The deployment command |

### Symbols worth tracing

- `build_documents()` → `fetch_clean_text()` → `clean_text()` → `chunk_text()`
- `Retriever.__init__()` → MiniLM embedding → `faiss.IndexFlatIP`
- `Retriever.retrieve()` → score threshold inside the retriever
- `agentic_query()` → retry and route selection
- `generate_answer()` → grounded versus ungrounded prompt
- `rewrite_query()` → deterministic query rewrite generation
- `query_rag()` → original non-agentic path
- `/query` versus `/agent-query` → backward compatibility and new behavior
- `recall_at_k()` and `mrr()` → what the reported metrics actually measure

---

## 3. Architecture walkthrough

### Ingestion

`src/rag_core.py` defines the source URLs and calls `fetch_clean_text()` for each one. `src/data_loader.py` uses `requests` to download the page, parses HTML with BeautifulSoup, selects paragraph tags, and joins their text.

The current code loads three sources: Karpathy, Ruder, and Colah. The older README description mentioned D2L, but the current URL list does not include a D2L URL. In an interview, describe the source list in the code, not the stale description.

### Cleaning

`clean_text()` strips whitespace, discards lines shorter than 40 characters, and removes lines containing selected acknowledgement-related keywords. This is lightweight preprocessing rather than a full HTML/content extraction system.

### Chunking

`chunk_text()` creates 400-word windows with 100 words of overlap. Each window is then passed through TinyLlama's tokenizer and decoded again. The overlap helps preserve concepts that cross a chunk boundary.

A precise description is better than calling this a perfect token-aware splitter: the window is chosen by words first, then normalized through the tokenizer. A production version could split by tokens directly and preserve paragraph or heading boundaries.

### Retrieval

`Retriever` embeds all chunks with `all-MiniLM-L6-v2`, normalizes the embeddings, and inserts them into `faiss.IndexFlatIP`. A query is embedded and searched against the index. The returned result contains an ID, text, source label, and similarity score.

Because the vectors are normalized, the inner product is equivalent to cosine similarity:

```text
cosine_similarity(a, b) = (a · b) / (||a|| ||b||)
```

After normalization, both norms are one, so the score is simply `a · b`.

### Generation

`generate_answer()` constructs a TinyLlama chat-style prompt containing the system instruction, retrieved context, and question. Grounded mode tells the model to answer only from the supplied context. Ungrounded mode tells it that no retrieved context is available and asks it to state that the answer is not grounded in the collection.

### Agentic routing

`agentic_query()` is a bounded control loop rather than an autonomous planner:

```text
retrieve original query
        │
        ├─ top score >= 0.35 → grounded generation
        │
        └─ top score < 0.35
              │
              ├─ rewrite query once
              ├─ retrieve rewritten query
              │
              ├─ strong result → grounded generation
              └─ weak/empty result → ungrounded fallback
```

The function returns the answer, the accepted retrieval results, and metadata containing the selected route and trace. Weak results are not passed to final generation as if they were trustworthy context.

---

## 4. Interview questions and strong answers

### Q1. What problem does RAG solve here?

**Answer:**

RAG gives the language model access to a controlled document collection at query time. Instead of expecting TinyLlama to remember every detail, the system retrieves relevant passages and asks the model to answer from those passages. This can improve freshness and traceability without fine-tuning the model for every document update.

**Follow-up:** What does RAG not guarantee?

**Answer:**

Retrieval quality is a separate failure point. If the relevant passage is not retrieved, the generator cannot reliably use it. Even with good context, the generator can misread or overstate the evidence. RAG reduces some hallucination risk; it does not prove factual correctness.

---

### Q2. Walk me through one request end to end.

**Answer:**

The application builds the document list and retriever when `src.rag_core` is imported. An HTTP request reaches either `/query` or `/agent-query`. The original endpoint calls `query_rag()`, which retrieves two chunks and generates an answer. The agentic endpoint calls `agentic_query()`, which retrieves three chunks, inspects the highest score, and either generates a grounded response, rewrites and retries once, or uses the labeled ungrounded fallback. The API serializes the answer, retrieved IDs, route, and trace.

**Important detail:** The models and source fetching occur during startup/import, not per request. That reduces repeated setup work but increases startup time and makes startup depend on network access.

---

### Q3. Why use embeddings instead of keyword search?

**Answer:**

Embeddings allow semantic matches even when the question and document use different words. A question about “preventing unstable recurrent training” may retrieve a passage about vanishing gradients or LSTM memory even without exact keyword overlap. The trade-off is that embedding similarity can be vague, domain-dependent, and difficult to calibrate. Keyword or hybrid retrieval would be useful for exact terminology, equations, names, and rare identifiers.

---

### Q4. Why `all-MiniLM-L6-v2`?

**Answer:**

It is a compact sentence-transformer model that is practical for a CPU-oriented demo. It gives dense semantic embeddings without requiring a large embedding service. The choice is a latency and resource trade-off, not a claim that it is optimal for all technical ML content. I would compare it against a domain-specific embedding model using a fixed validation set.

---

### Q5. Why normalize the embeddings?

**Answer:**

The code normalizes document and query embeddings so inner-product search behaves like cosine-similarity search. That makes the score easier to interpret as angular similarity and avoids vector magnitude dominating the ranking. It also makes the retrieval setup consistent between indexing and querying.

**Follow-up:** What happens if only document vectors are normalized?

**Answer:**

The comparison is no longer symmetric cosine similarity. Query normalization must match the indexing convention; otherwise the score can be affected by query-vector magnitude.

---

### Q6. Why use `IndexFlatIP`?

**Answer:**

`IndexFlatIP` is simple and performs exact inner-product search. It is a good baseline for a small corpus because it has no training step or approximate-neighbor tuning. Its limitation is scale: search cost grows with the number of stored vectors. For a much larger corpus I would benchmark an approximate FAISS index such as an IVF or HNSW configuration, or use a managed/vector database, while checking recall against the exact index.

---

### Q7. What does a retrieval score mean here?

**Answer:**

It is the inner product of the normalized query and chunk embeddings, which is cosine similarity under this setup. It is a ranking signal, not a calibrated probability that the chunk is correct. That is why the `0.35` agent threshold should be tuned using labeled validation data rather than interpreted as “35% confidence.”

There are currently two separate controls:

- `src/retrieval.py` returns no results when the top score is below `0.20`.
- `src/agent.py` treats a top score below `0.35` as weak and attempts a rewrite.

These thresholds have different roles and should be evaluated together.

---

### Q8. Why does the agent retrieve three chunks while the original RAG path retrieves two?

**Answer:**

The agentic path was designed to give the final generator up to three candidate chunks after routing, while the original path remains unchanged for backward compatibility and still uses two. This is a behavior difference worth measuring: increasing `k` can improve recall but can also add irrelevant context and increase prompt length.

---

### Q9. Why not ask TinyLlama to output a tool-call JSON decision?

**Answer:**

TinyLlama is a small 1.1B model and is not a dependable tool-calling router without additional constrained decoding or fine-tuning. The retrieval score is already a deterministic signal available from the existing retriever, so it is more reliable for this bounded routing decision. The model is used for free-form tasks where it is more appropriate: rewriting a weak query and generating answer text.

A stronger production system could use structured tool calling with a model trained for it, but I would still keep guardrails and deterministic checks around the model decision.

---

### Q10. Why only one rewrite attempt?

**Answer:**

The loop is deliberately bounded. One retry can recover from a vocabulary mismatch without creating an unbounded latency loop or repeatedly drifting the query away from the user's intent. More retries would need an explicit latency budget, loop detection, and evaluation showing that the additional retrieval recall is worth the cost.

---

### Q11. What if the query rewrite is identical to the original?

**Answer:**

The code checks whether the rewritten query is non-empty and different. If it is empty or unchanged, it does not spend another retrieval attempt and moves to the ungrounded fallback. That prevents a no-op rewrite from creating an unnecessary loop.

---

### Q12. Why return an ungrounded answer at all?

**Answer:**

The fallback keeps the assistant useful for questions outside the document collection, but it makes the evidence boundary visible with `[Ungrounded answer]`. The label is important because a model-generated answer without retrieved evidence should not look like a source-grounded answer.

The label is not a factuality guarantee. A safer product policy for high-stakes use could abstain completely, route to a human, or ask the user to broaden the source collection instead.

---

### Q13. How would you add citations?

**Answer:**

I would preserve the source metadata through retrieval, include source and chunk identifiers in the generation context, and return citations from the API. I would also ask the model to associate claims with context IDs, then validate that cited IDs came from the accepted retrieval results. I would not rely only on the model to invent or reproduce URLs accurately.

The current API returns retrieved IDs but not source URLs or text spans, so citation support is an identified next step rather than an implemented feature.

---

### Q14. What are the main hallucination controls?

**Answer:**

The current controls are similarity thresholding, grounded prompting, bounded retry behavior, and explicit labeling of the ungrounded fallback. The retriever rejects very low top scores at `0.20`, and the agent uses `0.35` to decide whether to retry. These controls reduce the chance of silently presenting irrelevant context as evidence, but they do not provide calibrated factuality or claim-level verification.

---

### Q15. What is wrong with using a fixed score threshold?

**Answer:**

Embedding score distributions vary by model, corpus, query type, and corpus size. A threshold that works for one collection may reject valid queries in another or accept semantically broad but incorrect chunks. I would tune the threshold on a labeled validation set, report precision/recall or answer-level metrics at different thresholds, and potentially use a cross-encoder reranker or a learned calibration layer.

---

### Q16. Explain Recall@5.

**Answer:**

For each evaluation query, Recall@5 in this repository counts whether at least one relevant chunk appears in the top five retrieved results, then averages that hit indicator across the evaluation set. It answers: “How often did the retriever surface at least one relevant chunk within five candidates?”

This implementation is closer to hit-rate-at-5 for the query set than to a full document-recall analysis when each query has multiple relevant items. I would define the metric carefully in a production evaluation report.

---

### Q17. Explain MRR@5.

**Answer:**

Mean Reciprocal Rank records the reciprocal of the rank of the first relevant result for each query, using zero when no relevant result appears in the top five, and averages those values. It rewards putting the first relevant chunk near the top rather than merely retrieving one somewhere in the candidate list.

The code in `src/evaluation.py` implements this first-relevant-rank behavior directly.

---

### Q18. Are the reported Recall@5 and MRR@5 production-quality metrics?

**Answer:**

Not by themselves. The README reports approximately `0.75` Recall@5 and `0.62` MRR@5, but the current evaluation set is small and constructs relevant IDs by searching for keywords in the loaded documents. That makes it a useful diagnostic, not a strong held-out benchmark. The evaluation set, labels, corpus snapshot, model versions, and run configuration should be versioned before making a performance claim.

I would also evaluate the agentic path separately because the original evaluation code calls the retriever directly and does not measure rewrite recovery, fallback rate, answer faithfulness, latency, or token cost.

---

### Q19. How would you design a better evaluation set?

**Answer:**

I would create a versioned set of representative questions with human-labeled relevant chunks or source passages. I would split questions into development and held-out test sets, avoid generating labels from the same keyword heuristic being evaluated, and include paraphrases, ambiguous questions, out-of-domain questions, and questions whose answer spans multiple chunks.

I would report retrieval Recall@K and MRR separately from answer metrics such as citation precision, answer faithfulness, answer relevance, abstention quality, latency, and token usage. I would tune the threshold only on the development set and report final test results once.

---

### Q20. What happens if the relevant answer spans two chunks?

**Answer:**

The overlap may help preserve boundary context, and retrieving multiple chunks can provide both parts. However, the current metrics and generation path do not explicitly guarantee multi-chunk coverage or ordering. I would test multi-hop and boundary-spanning questions, preserve chunk positions, and consider adjacent-chunk expansion or a reranker that scores the combined evidence.

---

### Q21. What are the biggest performance bottlenecks?

**Answer:**

Startup fetches remote pages, tokenizes the corpus, computes embeddings, loads the embedding model, and loads TinyLlama. Requests then perform embedding plus CPU generation, with generation likely dominating latency. The index is rebuilt at startup rather than persisted.

For production I would cache the cleaned corpus and embeddings, persist the vector index with model/version metadata, pre-load or separately serve the generator, measure p50/p95 latency, and add request limits and concurrency controls.

---

### Q22. What reliability issues do you see in the data loader?

**Answer:**

`fetch_clean_text()` currently calls `requests.get(url)` without an explicit timeout or `raise_for_status()`. It also assumes that useful content is in paragraph tags. A robust version would set connect/read timeouts, check HTTP status, use a user agent where appropriate, handle retries carefully, record the corpus version, and fail or skip a source explicitly rather than silently indexing an error page.

This is a good example of separating the demo's architecture from production hardening.

---

### Q23. Why does startup network access matter?

**Answer:**

Because `documents = build_documents()` and `retriever = Retriever(documents)` execute during module import, importing the API can trigger network calls, tokenization, embedding computation, and model loading. That makes cold starts slow and fragile. A production design would use an offline ingestion job, persist versioned artifacts, and load them during service startup with health/readiness reporting.

---

### Q24. What security risks would you consider?

**Answer:**

The source pages and user questions are untrusted text that enters a prompt. A malicious or accidental instruction embedded in a document could try to override the system instruction, so source text should be treated as data and tested for prompt injection. The API also needs rate limiting, request-size limits, authentication if private data is added, safe error handling, and logging that does not expose sensitive content.

The current repository does not implement those controls, so I would describe them as production gaps.

---

### Q25. How would you test this project without downloading the models?

**Answer:**

I would inject fake retriever, rewriter, and answer-generator dependencies into `agentic_query()`. The repository's `tests/test_agent.py` does this and checks the three important routes: confident grounded retrieval, weak-query rewrite followed by successful retrieval, and persistent weakness followed by an explicitly labeled ungrounded fallback. This makes routing tests fast and deterministic.

For integration tests, I would use a small local fixture corpus and mocked model interfaces, then keep one separately marked end-to-end test for actual model loading and network-backed ingestion.

---

### Q26. What does the `retrieved_ids` response tell the client?

**Answer:**

It exposes which in-memory chunk IDs were accepted for the response. The agentic endpoint additionally returns the route and trace, which helps diagnose whether an answer was grounded or came from fallback. IDs alone are not human-readable citations because they are tied to the current in-memory document ordering. Stable document/chunk IDs and source metadata would be needed for durable citations.

---

### Q27. Why keep `/query` if `/agent-query` is better?

**Answer:**

Keeping `/query` preserves backward compatibility for existing clients and gives a baseline path for comparison. The new endpoint can be evaluated independently rather than silently changing the behavior of existing consumers. Once the new path is validated, the project could deprecate the old endpoint deliberately rather than breaking it unexpectedly.

---

### Q28. What would you improve first?

**Answer:**

First I would create a versioned, human-labeled evaluation set and use it to tune the retrieval and agent thresholds. Without that, it is hard to know whether rewriting improves answer quality or only adds latency. Next I would persist the corpus/index artifacts and harden fetching with timeouts and status checks. Then I would add source citations and evaluate faithfulness, latency, fallback rate, and out-of-domain behavior.

---

## 5. Questions about design alternatives

### Fine-tuning versus RAG

| Question | RAG | Fine-tuning |
|---|---|---|
| Updating documents | Re-index documents | Usually requires another training/update process |
| Source traceability | Can return retrieved passages | Not naturally traceable |
| Domain behavior | Adds context at query time | Changes model behavior/weights |
| Cost for small changing corpus | Often lower | Can be higher |
| Best fit here | Curated technical articles | Consistent style/task behavior if enough data exists |

A combined system is possible: fine-tune behavior or formatting, then use RAG for current factual content.

### Dense retrieval versus BM25

Dense retrieval handles semantic similarity and paraphrases. BM25 is strong for exact words, names, equations, and rare terms. A hybrid retriever can combine both rankings. The right choice should be decided by an evaluation set, not by assuming neural retrieval is always better.

### Exact versus approximate nearest-neighbor search

The current exact index is easier to reason about and provides a useful recall reference. Approximate indexes improve scale and latency but can lose some recall and require tuning. A sound migration compares the approximate index to the exact baseline on the same queries.

### Model-based router versus deterministic router

A model-based router can consider richer signals, such as question type or expected answerability, but it can be inconsistent and harder to validate. The current deterministic score router is narrow, transparent, and easy to test. A hybrid design could combine both while enforcing a hard safety policy around low-quality retrieval.

---

## 6. Technical exercises to practice

### Exercise A: Draw the data flow

Without looking at the repository, draw:

```text
URLs → HTML paragraphs → cleaned text → overlapping chunks
     → embeddings → normalized FAISS index
question → query embedding → top chunks → prompt → TinyLlama answer
```

Then add the agentic branch and identify where each threshold is applied.

### Exercise B: Explain a score from first principles

Be able to explain why normalized inner-product search is cosine similarity, what a high score means, and why a score is not a probability.

### Exercise C: Trace a weak query

Use the fake components from `tests/test_agent.py` and describe:

1. The initial query.
2. The first retrieval score.
3. Why the rewrite runs.
4. The second query.
5. Why the final route is grounded or ungrounded.
6. What appears in the trace.

### Exercise D: Design a threshold experiment

Propose a validation experiment with candidate thresholds, fixed questions, relevance labels, and metrics for:

- grounded answer success;
- fallback/abstention quality;
- rewrite recovery rate;
- latency;
- prompt/token cost.

Do not tune on the final test set.

### Exercise E: Add citations on paper

Design the response shape you would want:

```json
{
  "answer": "...",
  "route": "grounded",
  "citations": [
    {"id": 7, "source": "Ruder", "score": 0.61}
  ],
  "trace": []
}
```

Then explain how you would make IDs stable across restarts rather than relying on list position.

### Exercise F: Production hardening review

Identify the changes needed before exposing this publicly:

- request timeout and source-fetch error handling;
- offline/versioned corpus ingestion;
- persisted index and model version metadata;
- source citations;
- prompt-injection handling;
- authentication and rate limiting;
- request and context length limits;
- observability for route, score, latency, and fallback rate;
- regression evaluation in CI.

---

## 7. A realistic mock interview

### Prompt 1

**Interviewer:** “Your system says it is agentic. Is it really an agent?”

**Good response:**

> “It is a bounded agentic workflow rather than a fully autonomous tool-using agent. It observes retrieval quality, chooses among grounded generation, one rewrite-and-retry action, and an ungrounded fallback, and records a trace. I intentionally use a deterministic score for routing because the selected small model is not a reliable structured tool caller. I would avoid overselling this as open-ended planning.”

### Prompt 2

**Interviewer:** “Why should I trust the `0.35` threshold?”

**Good response:**

> “I should not claim it is calibrated. It is an initial heuristic. The score is cosine similarity under normalized embeddings, but its distribution depends on the corpus and model. I would tune the threshold on a labeled development set and report the trade-off between retrieval recall, grounded answer quality, fallback rate, and latency.”

### Prompt 3

**Interviewer:** “What happens when retrieval fails?”

**Good response:**

> “The agent rewrites the query once and retries. If the retry remains weak or empty, it does not pass those weak chunks to final generation; it calls the model without retrieved context and prefixes the result with `[Ungrounded answer]`. For a high-stakes product I would likely abstain instead of using that fallback.”

### Prompt 4

**Interviewer:** “How do you know your metric is correct?”

**Good response:**

> “The code directly implements top-K hit rate and first-relevant-rank reciprocal scoring, so I can explain exactly what it computes. But correctness of the formula does not make the benchmark rigorous: the current labels are keyword-derived and the set is small. I would preserve a human-labeled, versioned set and test the metric implementation separately with hand-checked cases.”

### Prompt 5

**Interviewer:** “What is the first production issue you would fix?”

**Good response:**

> “I would move ingestion and indexing out of API import time. The service should load a versioned offline corpus and persisted index, while a separate job handles fetching, cleaning, embedding, and validation. That improves cold starts, reproducibility, and failure isolation.”

---

## 8. Red flags to avoid in an interview

Do not say:

- “The model is guaranteed not to hallucinate.”
- “A score of `0.35` means 35% confidence.”
- “The system is fully autonomous.”
- “The evaluation proves production accuracy.”
- “The code is token-aware” without explaining that windows are selected by words first.
- “The current corpus includes D2L” unless the source URL list has been updated.
- “The API has citations” when it currently returns only in-memory IDs.
- “The fallback is grounded” when it explicitly has no retrieved context.
- “FAISS is always scalable” without distinguishing exact and approximate indexes.
- “The agent is evaluated” when the current evaluation code measures direct retrieval only.

Honest limitations usually make the technical explanation stronger.

---

## 9. Suggested learning materials

Use the repository first, then compare the implementation with these primary references:

- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — the original RAG formulation.
- [FAISS documentation](https://github.com/facebookresearch/faiss) — vector indexes, exact search, and approximate search trade-offs.
- [Sentence Transformers documentation](https://www.sbert.net/) — embedding models and semantic search patterns.
- [Hugging Face Transformers text generation](https://huggingface.co/docs/transformers/main/en/main_classes/text_generation) — decoding controls such as sampling, temperature, top-p, and token limits.
- [FastAPI documentation](https://fastapi.tiangolo.com/) — request models, endpoints, validation, and deployment patterns.
- [Information Retrieval lecture notes from Stanford IR](https://nlp.stanford.edu/IR-book/) — precision, recall, ranking, and evaluation foundations.

### Study order

1. Read `src/rag_core.py` and draw the pipeline.
2. Read `src/retrieval.py` and explain normalization plus inner product.
3. Read `src/evaluation.py` and restate the exact metrics in plain English.
4. Read `src/agent.py` and explain every branch.
5. Read `tests/test_agent.py` and explain the dependency injection strategy.
6. Read the RAG paper introduction and compare its assumptions with this demo.
7. Read FAISS index documentation and explain why `IndexFlatIP` is a baseline rather than a universal production answer.
8. Prepare the five mock-interview answers above without notes.

---

## 10. Final checklist before an interview

You should be able to answer all of these without opening the code:

- What is the full data flow from URL to answer?
- Why use RAG instead of fine-tuning for this corpus?
- Why MiniLM, normalization, and inner-product search?
- What exactly does the retrieval score represent?
- What are the two thresholds and why are they not probabilities?
- What makes the new path agentic, and what does it not do?
- Why is the retry bounded to one rewrite?
- How is the ungrounded fallback made visible?
- What do Recall@5 and MRR@5 measure here?
- Why are the current metrics not a rigorous benchmark?
- What happens when startup cannot reach a source URL?
- What are the startup and request-time bottlenecks?
- How would you add citations?
- How would you defend against prompt injection?
- How would you test routing without downloading models?
- What would you improve first, and how would you measure the improvement?

If you can answer those with the trade-offs included, you understand the project rather than just recognizing its filenames.
