"""Hybrid retriever: BM25 + dense embeddings, fused with Reciprocal Rank Fusion."""
import json

import numpy as np
from rank_bm25 import BM25Okapi

from src import config, llm
from src.text import query_prefix, tokenize


class Retriever:
    def __init__(self):
        with open(config.INDEX_DIR / "chunks.json", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.vecs = np.load(config.INDEX_DIR / "embeddings.npy")
        self.bm25 = BM25Okapi([tokenize(c["title"] + " " + c["text"]) for c in self.chunks])

    def bm25_search(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        return list(np.argsort(-scores)[:k])

    def dense_search(self, query, k):
        q = np.array(llm.embed([query_prefix(query)])[0], dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12
        return list(np.argsort(-(self.vecs @ q))[:k])

    def search(self, query, k=None, mode="hybrid"):
        """mode: 'bm25', 'dense' or 'hybrid'. Returns the top-k chunks (dicts with a 'score')."""
        k = k or config.TOP_K
        n = config.CANDIDATES
        if mode == "bm25":
            ranked = [(i, 1.0 / (r + 1)) for r, i in enumerate(self.bm25_search(query, k))]
        elif mode == "dense":
            ranked = [(i, 1.0 / (r + 1)) for r, i in enumerate(self.dense_search(query, k))]
        else:
            fused = {}
            for ids in (self.bm25_search(query, n), self.dense_search(query, n)):
                for rank, i in enumerate(ids):
                    fused[i] = fused.get(i, 0.0) + 1.0 / (config.RRF_K + rank + 1)
            ranked = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [{**self.chunks[i], "score": round(float(s), 4)} for i, s in ranked]
