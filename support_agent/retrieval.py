"""Retrieval: independent BM25 + dense search fused with Reciprocal Rank Fusion (RRF)."""
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in is it my of on or our so that the "
    "their there this to was we what when where which who will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Keeps hyphenated identifiers (ERR-4012, ORD-99) whole *and* split, so exact IDs match."""
    toks: list[str] = []
    for w in re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower()):
        parts = re.split(r"[-_.]", w)
        if len(parts) > 1:
            toks.append(w)
        toks.extend(p for p in parts if p not in STOPWORDS)
    return toks


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=np.float32)
        self.avg = float(self.len.mean()) if len(docs) else 0.0
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf), dtype=np.float32)
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t, 0)
                if f:
                    norm = 1 - self.b + self.b * self.len[i] / (self.avg or 1.0)
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * norm)
        return s


@dataclass
class Hit:
    idx: int
    score: float
    cosine: float
    bm25: float


class HybridRetriever:
    def __init__(self, texts: list[str], embeddings: np.ndarray, embedder, reranker=None,
                 rrf_k: int = 60, w_dense: float = 1.0, w_sparse: float = 1.0):
        self.texts, self.emb, self.embedder, self.reranker = texts, embeddings, embedder, reranker
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.rrf_k, self.w_dense, self.w_sparse = rrf_k, w_dense, w_sparse

    def search(self, query: str, top_k: int = 5, candidates: int = 30) -> list[Hit]:
        if not self.texts:
            return []
        cos = self.emb @ self.embedder.encode([query])[0]
        bm = self.bm25.scores(tokenize(query))
        n = min(candidates, len(self.texts))
        dense_rank = np.argsort(-cos)[:n]
        sparse_rank = [i for i in np.argsort(-bm)[:n] if bm[i] > 0]  # no match => no vote

        fused: dict[int, float] = {}
        for r, i in enumerate(dense_rank):
            fused[int(i)] = fused.get(int(i), 0.0) + self.w_dense / (self.rrf_k + r + 1)
        for r, i in enumerate(sparse_rank):
            fused[int(i)] = fused.get(int(i), 0.0) + self.w_sparse / (self.rrf_k + r + 1)

        ranked = sorted(fused, key=fused.get, reverse=True)
        if self.reranker is not None:
            pool = ranked[: max(top_k * 3, top_k)]
            ce = self.reranker.predict([(query, self.texts[i]) for i in pool])
            ranked = [i for _, i in sorted(zip(ce, pool), key=lambda x: -x[0])]
            fused = {i: float(s) for s, i in zip(sorted(ce, reverse=True), ranked)}
        return [Hit(i, fused[i], float(cos[i]), float(bm[i])) for i in ranked[:top_k]]


def load_reranker(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)
