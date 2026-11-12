"""Dense retrieval over clause chunks (sentence-transformers, cosine similarity, numpy).

The index is small (tens of thousands of chunks), so a plain normalised matrix and a dot product
are enough; no vector database is needed. Embeddings are cached in data/index.npz.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from . import DATA, config
from . import glossary as gl
from .specs import Chunk, build_chunks, load_chunks, save_chunks


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, encoder) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.encoder = encoder

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        q = self.encoder(query)
        scores = self.vectors @ q
        top = np.argsort(-scores, kind="stable")[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


def st_encoder(model_name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encode_many(texts: list[str]) -> np.ndarray:
        return model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    def encode_one(text: str) -> np.ndarray:
        return model.encode([text], normalize_embeddings=True)[0]

    return encode_many, encode_one


def chunk_text_for_embedding(c: Chunk) -> str:
    return f"TS {c.spec} {c.clause} {c.title}. {c.text}"


def load_index(cfg) -> Index:
    chunks = load_chunks()
    vectors = np.load(DATA / "index.npz")["vectors"]
    _, encode_one = st_encoder(cfg["retrieval"]["embedding_model"])
    return Index(chunks, vectors, encode_one)


def expanded_query(question: str, glossary: dict[str, list[str]], limit: int) -> str:
    terms = gl.find_terms(question, glossary, limit)
    if not terms:
        return question
    return question + " (" + "; ".join(gl.describe(terms, glossary, per_term=1)) + ")"


def main() -> None:
    """make index: parse the specs, build chunks and the glossary, embed all chunks."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = config.load(args.config) if args.config else config.load()
    t0 = time.time()
    chunks = build_chunks(cfg)
    save_chunks(chunks)
    by_spec: dict[str, int] = {}
    for c in chunks:
        by_spec[c.spec] = by_spec.get(c.spec, 0) + 1
    print(f"{len(chunks)} chunks {by_spec} in {time.time() - t0:.0f} s")
    glossary = gl.build(cfg)
    gl.save(glossary)
    print(f"glossary: {len(glossary)} acronyms")
    encode_many, _ = st_encoder(cfg["retrieval"]["embedding_model"])
    t0 = time.time()
    vectors = encode_many([chunk_text_for_embedding(c) for c in chunks]).astype(np.float32)
    np.savez_compressed(DATA / "index.npz", vectors=vectors)
    print(f"embedded {len(chunks)} chunks in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
