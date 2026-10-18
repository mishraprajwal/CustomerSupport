"""End-to-end RAG pipeline: ingest -> index -> retrieve -> augment -> generate."""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .embeddings import get_embedder
from .generate import NO_ANSWER, build_context, build_messages
from .ingest import Chunk, ingest
from .retrieval import HybridRetriever, load_reranker


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)
    query_used: str = ""


class SupportAgent:
    def __init__(self, index_dir: str = "index", embedder: str = "sentence-transformers/all-MiniLM-L6-v2",
                 llm=None, reranker: str | None = None, min_cosine: float = 0.25):
        self.dir = Path(index_dir)
        self.embedder = get_embedder(embedder)
        self.embedder_name = embedder
        self.llm = llm
        self.reranker = load_reranker(reranker) if reranker else None
        self.min_cosine = min_cosine
        self.chunks: list[Chunk] = []
        self.retriever: HybridRetriever | None = None
        self.history: list[tuple[str, str]] = []

    # 1. Ingestion
    def build_index(self, docs_path: str, max_chars: int = 800, overlap: int = 120) -> int:
        chunks = ingest(docs_path, max_chars, overlap)
        if not chunks:
            raise ValueError(f"No .txt/.md documents found in {docs_path}")
        emb = self.embedder.encode([c.text for c in chunks])
        self.dir.mkdir(parents=True, exist_ok=True)
        np.save(self.dir / "embeddings.npy", emb)
        (self.dir / "chunks.json").write_text(json.dumps([c.__dict__ for c in chunks]))
        (self.dir / "meta.json").write_text(json.dumps({"embedder": self.embedder_name}))
        self._activate(chunks, emb)
        return len(chunks)

    def load_index(self) -> None:
        meta = json.loads((self.dir / "meta.json").read_text())
        if meta["embedder"] != self.embedder_name:
            raise ValueError(f"Index built with {meta['embedder']!r}, not {self.embedder_name!r}; rebuild it.")
        chunks = [Chunk(**c) for c in json.loads((self.dir / "chunks.json").read_text())]
        self._activate(chunks, np.load(self.dir / "embeddings.npy"))

    def _activate(self, chunks, emb):
        self.chunks = chunks
        self.retriever = HybridRetriever([c.text for c in chunks], emb, self.embedder, self.reranker)

    # 2-4. Retrieval, augmentation, generation
    def ask(self, question: str, top_k: int = 4) -> Answer:
        if self.retriever is None:
            raise RuntimeError("Index not loaded. Call build_index() or load_index().")
        query = question
        if self.llm and self.history:
            try:
                query = self.llm.rewrite(question, self.history)
            except Exception:
                pass
        hits = self.retriever.search(query, top_k=top_k)
        # Guardrail: neither lexical match nor semantic similarity => don't let the LLM guess.
        if not hits or all(h.bm25 == 0 and h.cosine < self.min_cosine for h in hits):
            return Answer(NO_ANSWER, [], query)
        passages = [(self.chunks[h.idx].label(), self.chunks[h.idx].text) for h in hits]
        sources = [{"n": i, "source": self.chunks[h.idx].label(), "score": round(h.score, 4),
                    "snippet": self.chunks[h.idx].text[:200]} for i, h in enumerate(hits, 1)]
        if self.llm is None:
            text = "(LLM unavailable - top matching passages)\n\n" + build_context(passages)
        else:
            try:
                text = self.llm.chat(build_messages(question, build_context(passages), self.history))
            except Exception as e:
                text = f"(LLM error: {e})\n\n" + build_context(passages)
        self.history.append((question, text))
        return Answer(text, sources, query)
