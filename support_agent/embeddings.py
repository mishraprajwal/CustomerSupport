"""Pluggable embedders. All return L2-normalised float32 matrices."""
import hashlib
import re

import numpy as np


def _norm(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


class SentenceTransformerEmbedder:
    def __init__(self, name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.name = name
        self.model = SentenceTransformer(name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return _norm(self.model.encode(texts, show_progress_bar=False))


class HashingEmbedder:
    """Dependency-free unigram/bigram hashing embedder (offline fallback and tests)."""

    name = "hashing-512"

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = re.findall(r"\w+", t.lower())
            for f in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
                out[i, int(hashlib.md5(f.encode()).hexdigest(), 16) % self.dim] += 1.0
        return _norm(out)


def get_embedder(name: str):
    if name == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(name)
