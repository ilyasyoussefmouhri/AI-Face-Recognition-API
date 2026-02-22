# benchmarks/data_gen.py
#
# Generates synthetic but realistic face embeddings.
#
# Why this matters:
# InsightFace's ArcFace model outputs L2-normalised 512-dim float32 vectors.
# Your recognition.py converts stored embeddings back with:
#     np.array(face.embedding, dtype=np.float32)
# and then calls matcher.similarity(query, stored) which is a pure dot product.
# We replicate those properties here so the benchmark reflects real behaviour.

import numpy as np
from config import RANDOM_SEED, EMBEDDING_DIM


def generate_embeddings(n: int) -> list[list[float]]:
    """
    Return n L2-normalised 512-dim float32 embeddings as plain Python lists.

    Plain lists are what your registration service stores:
        face.embedding = embedding_obj.embedding.tolist()
    and what psycopg needs for ARRAY(Float) columns.

    The fixed seed guarantees the same synthetic dataset every time
    this function is called with the same n, making runs reproducible.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    vectors = rng.standard_normal((n, EMBEDDING_DIM)).astype(np.float32)

    # L2 normalise — same step your InsightFaceEmbedder does before storage
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalised = vectors / norms

    return normalised.tolist()


def generate_query_embedding() -> list[float]:
    """
    A single query embedding to use in recognition requests.

    Uses seed+1 so it's distinct from the stored embeddings, simulating
    a real probe image that may or may not match something in the DB.
    """
    rng = np.random.default_rng(RANDOM_SEED + 1)
    vec = rng.standard_normal(EMBEDDING_DIM).astype(np.float32)
    return (vec / np.linalg.norm(vec)).tolist()


if __name__ == "__main__":
    # Quick sanity check
    embs = generate_embeddings(5)
    print(f"Generated {len(embs)} embeddings")
    print(f"Dimension : {len(embs[0])}")
    arr = np.array(embs[0])
    print(f"L2 norm   : {np.linalg.norm(arr):.6f}  (should be 1.0)")
    print(f"dtype     : {arr.dtype}")
