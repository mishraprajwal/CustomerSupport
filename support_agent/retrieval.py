"""Retrieval: independent BM25 + dense search fused with Reciprocal Rank Fusion (RRF)."""
import math
import re
from collections import Counter
from dataclasses import dataclass

import faiss
import nltk
import numpy as np
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer

_FALLBACK_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in is it my of on or our so that the "
    "their there this to was we what when where which who will with you your".split()
)
_stemmer = PorterStemmer()


def _load_stopwords() -> frozenset:
    try:
        return frozenset(nltk_stopwords.words("english"))
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
            return frozenset(nltk_stopwords.words("english"))
        except Exception:
            return _FALLBACK_STOPWORDS


STOPWORDS = _load_stopwords()


def _stem(tok: str) -> str:
    # Identifiers/numbers (err-4012, 30) must match exactly, so only plain words are stemmed.
    return tok if any(c.isdigit() for c in tok) else _stemmer.stem(tok)


def tokenize(text: str) -> list[str]:
    """NLTK stopwords + Porter stemming; hyphenated identifiers are kept whole *and* split."""
    toks: list[str] = []
    for w in re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower()):
        parts = re.split(r"[-_.]", w)
        if len(parts) > 1:
            toks.append(w)
        toks.extend(_stem(p) for p in parts if p not in STOPWORDS)
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
    def __init__(self, texts: list[str], index: "faiss.Index", embedder, reranker=None,
                 rrf_k: int = 60, w_dense: float = 1.0, w_sparse: float = 1.0):
        self.texts, self.index, self.embedder, self.reranker = texts, index, embedder, reranker
        self.bm25 = BM25([tokenize(t) for t in texts])
        self.rrf_k, self.w_dense, self.w_sparse = rrf_k, w_dense, w_sparse

    def search(self, query: str, top_k: int = 5, candidates: int = 30) -> list[Hit]:
        if not self.texts:
            return []
        q = self.embedder.encode([query])
        bm = self.bm25.scores(tokenize(query))
        n = min(candidates, len(self.texts))
        sims, ids = self.index.search(q, n)  # inner product == cosine (unit vectors)
        dense_rank = [int(i) for i in ids[0] if i >= 0]
        cos = dict(zip(dense_rank, sims[0].tolist()))
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
        # BM25-only hits fall outside the dense candidates; reconstruct their cosine for the guardrail.
        def cosine(i: int) -> float:
            return cos[i] if i in cos else float(self.index.reconstruct(i) @ q[0])

        return [Hit(i, fused[i], cosine(i), float(bm[i])) for i in ranked[:top_k]]


def build_faiss_index(embeddings: np.ndarray) -> "faiss.Index":
    index = faiss.IndexFlatIP(embeddings.shape[1])  # dimension taken from the model, not hardcoded
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    return index


def load_reranker(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name)
