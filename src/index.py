"""Step 2 - Build the BM25 (keyword) index and the dense (embedding) index.

Usage:  python -m src.index
Reads   data/processed/chunks.jsonl
Writes  data/index/chunks.json, data/index/embeddings.npy
"""
import json
import time

import numpy as np

from src import config, llm
from src.text import doc_prefix


def load_chunks():
    with open(config.CHUNKS_FILE, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    chunks = load_chunks()
    print(f"Embedding {len(chunks)} chunks with {config.EMBED_MODEL} (first run takes a few minutes)...")
    t0 = time.time()
    texts = [doc_prefix(c) for c in chunks]
    vecs = np.array(llm.embed(texts), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12   # unit length -> dot product = cosine
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(config.INDEX_DIR / "embeddings.npy", vecs)
    with open(config.INDEX_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    print(f"Done in {time.time() - t0:.0f}s. Index saved to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
