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

## Retrieval features

- BM25 runs over the whole corpus alongside FAISS dense search, so exact IDs/error codes (`ERR-4012`) are never missed.
- Reciprocal Rank Fusion merges the two rankings, with tunable `w_dense`/`w_sparse` weights.
- NLTK stopwords + Porter stemming ("refunds" matches "refund"), with identifiers like `ERR-4012` kept exact; falls back to a built-in stopword list if NLTK data can't be downloaded.
- Overlapping, heading-aware chunks with source citations.
- Persistent index (`index/vectors.faiss`) that records its embedder and refuses to load with a different one; vector dimension is read from the model.
- Guardrail: no lexical or semantic evidence means the LLM is not called.
