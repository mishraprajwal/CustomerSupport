# Customer Support Agent (Hybrid-Search RAG)

Support executives ask questions in plain language and get answers grounded in your knowledge base, with citations.

```
pip install -r requirements.txt
ollama pull llama3.2                      # local LLM
python -m support_agent.cli index data/kb # 1. ingest: chunk + embed + index
python -m support_agent.cli chat          # 2-4. retrieve, augment, generate
```

Options: `--rerank` (cross-encoder), `--model`, `--top-k`, `--no-llm` (passages only), `--embedder hashing` (offline). Put `.md`/`.txt` files in `data/kb/`.
Tests: `pytest` (no model downloads needed).

## Pipeline (modular)

| Stage | Module | Notes |
|---|---|---|
| Ingestion | `ingest.py`, `embeddings.py` | Heading-aware chunks (≤800 chars, 120 overlap), heading prefixed to each chunk, source metadata kept |
| Retrieval | `retrieval.py` | BM25 (NLTK stopwords + stemming) + FAISS dense search, fused with Reciprocal Rank Fusion; optional cross-encoder rerank |
| Augmentation | `generate.py` | Numbered context + last 3 turns; follow-ups rewritten to standalone queries |
| Generation | `generate.py` | Ollama; answers only from context, cites `[n]`, says "not found" otherwise |

## Improvements over [Rman410/hybrid-search](https://github.com/Rman410/hybrid-search)

- BM25 now runs over the **whole corpus**, not just FAISS' top-k, so exact IDs/error codes (`ERR-4012`) are never missed.
- **RRF** replaces min-max + harmonic mean (which divided by zero on a 0 score and was unstable on small candidate sets); the unused `WEIGHT_SEMANTIC` is now real `w_dense`/`w_sparse`.
- NLTK stopwords + Porter stemming ("refunds" matches "refund"), with identifiers like `ERR-4012` kept exact; falls back to a built-in stopword list if NLTK data can't be downloaded.
- Overlapping, heading-aware chunks with citations (original: char-split, no source).
- Persistent index with embedder check (original mmap cache validated only by row count).
- Guardrail: no lexical or semantic evidence → refuse without calling the LLM.
- FAISS `IndexFlatIP` kept for dense search and saved to `index/vectors.faiss`; dimension is read from the model instead of hardcoded 384.

## Advanced vs. modular RAG

Not either/or: **modular is the architecture, advanced is the set of techniques plugged into it.** Hybrid search, reranking and query rewriting are "advanced" pre/post-retrieval techniques, each a swappable module here (embedder, retriever, reranker, LLM). Full modular RAG routing/agentic loops (multi-source, tool calls to CRM/order systems) are worth adding later, not for a first KB Q&A tool.
