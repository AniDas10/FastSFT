"""Local sentence embeddings for the distillation-fidelity metric: how close the
child's answers sit to the parent's in embedding space.

Uses sentence-transformers (evaluation extra); the model runs locally, so no
embedding API or key is involved.
"""

from functools import lru_cache

from fastsft.eval.constants import DEFAULT_EMBEDDING_MODEL


@lru_cache(maxsize=2)
def _load_model(model_id: str):
    """Loads and caches a SentenceTransformer (heavy import kept local)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id)


def embed(texts: list[str], model_id: str = DEFAULT_EMBEDDING_MODEL):
    """L2-normalized embeddings for `texts` (one row each)."""
    return _load_model(model_id).encode(list(texts), normalize_embeddings=True)


def pairwise_similarities(
    a_texts: list[str], b_texts: list[str], model_id: str = DEFAULT_EMBEDDING_MODEL
) -> list[float]:
    """Cosine similarity of each aligned (a, b) pair. Embeddings are normalized,
    so cosine is a row-wise dot product."""
    if len(a_texts) != len(b_texts):
        raise ValueError(
            f"pairwise_similarities needs aligned lists, got {len(a_texts)} vs "
            f"{len(b_texts)}."
        )
    if not a_texts:
        return []
    a = embed(a_texts, model_id)
    b = embed(b_texts, model_id)
    return [float(sim) for sim in (a * b).sum(axis=1)]
