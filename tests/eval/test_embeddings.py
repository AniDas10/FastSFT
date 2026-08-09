"""Tier-2 unit tests for fastsft.eval.embeddings -- the pure logic around the
sentence-transformers edge (row-wise cosine as a dot product, alignment guard,
and the encode/model-id forwarding), with the heavy model always mocked out.

sentence-transformers is never imported: `pairwise_similarities` tests patch
`embed`, and `embed`/`_load_model` tests patch the loader or its constructor.
"""

import numpy as np
import pytest

import fastsft.eval.embeddings as embeddings
from fastsft.eval.constants import DEFAULT_EMBEDDING_MODEL
from fastsft.eval.embeddings import _load_model, embed, pairwise_similarities


@pytest.fixture(autouse=True)
def _clear_load_cache():
    """_load_model is lru_cached process-wide; clear it around every test so a
    patched constructor in one test can't leak a cached fake into the next."""
    _load_model.cache_clear()
    yield
    _load_model.cache_clear()


# --- pairwise_similarities: alignment guard ------------------------------


def test_pairwise_similarities_length_mismatch_raises(monkeypatch):
    def _fail(*args, **kwargs):  # embed must never be reached
        raise AssertionError("embed should not be called on a length mismatch")

    monkeypatch.setattr(embeddings, "embed", _fail)
    with pytest.raises(ValueError) as excinfo:
        pairwise_similarities(["a", "b", "c"], ["x"])
    message = str(excinfo.value)
    assert "3" in message and "1" in message


# --- pairwise_similarities: empty short-circuit --------------------------


def test_pairwise_similarities_empty_returns_empty_without_embedding(monkeypatch):
    calls = []

    def _record(*args, **kwargs):
        calls.append(args)
        raise AssertionError("embed should not be called for empty inputs")

    monkeypatch.setattr(embeddings, "embed", _record)
    assert pairwise_similarities([], []) == []
    assert calls == []


# --- pairwise_similarities: row-wise dot product -------------------------


def test_pairwise_similarities_is_rowwise_dot_product(monkeypatch):
    # Hand-chosen unit vectors: identical -> 1.0, orthogonal -> 0.0, and an
    # oblique pair (60 degrees apart) -> 0.5.
    a_vecs = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    )
    b_vecs = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.5, np.sqrt(3) / 2]],
    )

    def _fake_embed(texts, model_id=DEFAULT_EMBEDDING_MODEL):
        return a_vecs if list(texts) == ["a0", "a1", "a2"] else b_vecs

    monkeypatch.setattr(embeddings, "embed", _fake_embed)

    result = pairwise_similarities(["a0", "a1", "a2"], ["b0", "b1", "b2"])
    assert result == pytest.approx([1.0, 0.0, 0.5])
    assert all(isinstance(sim, float) for sim in result)


def test_pairwise_similarities_single_pair(monkeypatch):
    # Both sides embed to the same unit vector, so the pair's cosine is 1.0.
    def _fake_embed(texts, model_id=DEFAULT_EMBEDDING_MODEL):
        return np.array([[0.6, 0.8]])

    monkeypatch.setattr(embeddings, "embed", _fake_embed)
    result = pairwise_similarities(["a"], ["b"])
    assert result == pytest.approx([0.6 * 0.6 + 0.8 * 0.8])  # == 1.0
    assert isinstance(result[0], float)


def test_pairwise_similarities_threads_model_id(monkeypatch):
    seen = []

    def _fake_embed(texts, model_id=DEFAULT_EMBEDDING_MODEL):
        seen.append(model_id)
        return np.array([[1.0, 0.0]])

    monkeypatch.setattr(embeddings, "embed", _fake_embed)
    pairwise_similarities(["a"], ["b"], model_id="custom/model")
    assert seen == ["custom/model", "custom/model"]  # both a and b


# --- embed: forwards to the loaded model ---------------------------------


class _FakeEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, normalize_embeddings=None):
        self.calls.append((texts, normalize_embeddings))
        return "ENCODED"


def test_embed_forwards_texts_and_normalization(monkeypatch):
    fake = _FakeEncoder()
    seen_ids = []

    def _fake_load(model_id):
        seen_ids.append(model_id)
        return fake

    monkeypatch.setattr(embeddings, "_load_model", _fake_load)

    out = embed(("t1", "t2"))
    assert out == "ENCODED"
    assert seen_ids == [DEFAULT_EMBEDDING_MODEL]
    (texts, normalize), = fake.calls
    assert texts == ["t1", "t2"]  # coerced to a list
    assert normalize is True


def test_embed_threads_custom_model_id(monkeypatch):
    fake = _FakeEncoder()
    seen_ids = []

    def _fake_load(model_id):
        seen_ids.append(model_id)
        return fake

    monkeypatch.setattr(embeddings, "_load_model", _fake_load)
    embed(["t"], model_id="custom/embed-model")
    assert seen_ids == ["custom/embed-model"]


# --- _load_model: caches per id ------------------------------------------


def test_load_model_caches_per_id(monkeypatch):
    constructed = []

    class _FakeST:
        def __init__(self, model_id):
            constructed.append(model_id)

    # _load_model does `from sentence_transformers import SentenceTransformer`
    # lazily; inject a fake module so the real package is never imported.
    import sys
    import types

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    first = _load_model("some/model")
    second = _load_model("some/model")
    third = _load_model("other/model")

    assert first is second  # cached: same id -> same object
    assert third is not first
    assert constructed == ["some/model", "other/model"]  # built once per id
