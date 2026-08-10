"""Tier-1 unit tests for fastsft.training.heuristic -- the pure cost/feasibility
math. The Hugging Face edges (`_model_metadata`, `_max_sequence_length`) are
monkeypatched so nothing hits the network.
"""

import pytest

from fastsft.training import heuristic
from fastsft.training.constants import (
    LORA,
    MODAL_GPU_TIERS,
    QLORA,
    TARGET_EFFECTIVE_BATCH,
)
from fastsft.training.heuristic import (
    _best_batch_size,
    _estimate_memory_gb,
    recommend_configs,
)

# --- _estimate_memory_gb -----------------------------------------------------

def test_estimate_memory_gb_exact_formula():
    # base = 1e9 * 2.0 / 1e9 = 2.0; activation = 1000*10*100*1*20/1e9 = 0.02;
    # adapter = 0.05 + 16*0.002 = 0.082; total = (2.102) * 1.3.
    est = _estimate_memory_gb(int(1e9), LORA, 1000, 10, 100, 1)
    assert est == pytest.approx((2.0 + 0.02 + 0.082) * 1.3)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
def test_estimate_memory_gb_monotonic_in_batch_size(batch_size):
    smaller = _estimate_memory_gb(int(1e9), LORA, 1024, 24, 512, batch_size)
    larger = _estimate_memory_gb(int(1e9), LORA, 1024, 24, 512, batch_size + 1)
    assert larger > smaller


@pytest.mark.parametrize("seq_len", [128, 512, 2048])
def test_estimate_memory_gb_monotonic_in_seq_len(seq_len):
    smaller = _estimate_memory_gb(int(1e9), LORA, 1024, 24, seq_len, 4)
    larger = _estimate_memory_gb(int(1e9), LORA, 1024, 24, seq_len * 2, 4)
    assert larger > smaller


def test_estimate_memory_gb_qlora_cheaper_than_lora():
    args = (int(7e9), 4096, 32, 512, 1)
    lora = _estimate_memory_gb(args[0], LORA, *args[1:])
    qlora = _estimate_memory_gb(args[0], QLORA, *args[1:])
    assert qlora < lora


# --- _best_batch_size --------------------------------------------------------

def test_best_batch_size_returns_largest_that_fits():
    # vram=3.0: batch16 ~3.12 (too big), batch8 ~2.91 (fits) -> 8.
    assert _best_batch_size(3.0, int(1e9), LORA, 1000, 10, 100) == 8


def test_best_batch_size_returns_16_when_everything_fits():
    assert _best_batch_size(80.0, int(1e9), LORA, 1000, 10, 100) == 16


def test_best_batch_size_none_when_even_one_does_not_fit():
    # vram=2.0 < batch1 estimate ~2.73 -> nothing fits.
    assert _best_batch_size(2.0, int(1e9), LORA, 1000, 10, 100) is None


# --- recommend_configs -------------------------------------------------------

@pytest.fixture
def patched_metadata(monkeypatch):
    """Factory: pin (_model_metadata, _max_sequence_length) so no HF fetch."""

    def _apply(param_count, hidden_size, num_layers, seq_len):
        monkeypatch.setattr(
            heuristic, "_model_metadata",
            lambda _id: (param_count, hidden_size, num_layers),
        )
        monkeypatch.setattr(heuristic, "_max_sequence_length", lambda _id, _t: seq_len)

    return _apply


def test_recommend_configs_small_model_ranks_cheapest_first_all_lora(patched_metadata):
    patched_metadata(int(5e8), 896, 24, 512)  # fits every tier at batch 16
    result = recommend_configs("tiny/model", [], top_n=3)

    assert len(result) == 3
    prices = [c.est_usd_per_hour for c in result]
    assert prices == sorted(prices)                      # cheapest first
    assert result[0].gpu_tier == "L4"                    # the cheapest tier
    assert all(c.strategy == LORA for c in result)       # lora fits -> preferred


def test_recommend_configs_respects_top_n(patched_metadata):
    patched_metadata(int(5e8), 896, 24, 512)
    assert len(recommend_configs("tiny/model", [], top_n=2)) == 2


def test_recommend_configs_feasible_and_grad_accum_consistent(patched_metadata):
    patched_metadata(int(5e8), 896, 24, 512)
    for cfg in recommend_configs("tiny/model", [], top_n=6):
        vram = MODAL_GPU_TIERS[cfg.gpu_tier][0]
        assert cfg.est_memory_gb <= vram                 # only feasible tiers
        assert cfg.est_usd_per_hour == MODAL_GPU_TIERS[cfg.gpu_tier][1]
        expected_ga = max(1, -(-TARGET_EFFECTIVE_BATCH // cfg.loop.batch_size))
        assert cfg.loop.grad_accumulation == expected_ga


def test_recommend_configs_falls_back_to_qlora_when_lora_does_not_fit(patched_metadata):
    # 13B model: on L4 (24GB, cheapest tier) lora doesn't fit even at batch 1, but qlora does.
    patched_metadata(int(13e9), 4096, 32, 512)
    # Precondition documenting the scenario (guards against constant drift):
    assert _best_batch_size(24, int(13e9), LORA, 4096, 32, 512) is None
    assert _best_batch_size(24, int(13e9), QLORA, 4096, 32, 512) is not None

    result = recommend_configs("mid/model", [], top_n=3)
    l4 = next(c for c in result if c.gpu_tier == "L4")
    assert l4.strategy == QLORA                           # fell back
    assert any(c.strategy == LORA for c in result)        # bigger tiers still lora


def test_recommend_configs_raises_when_no_tier_fits(patched_metadata):
    patched_metadata(int(1e12), 8192, 80, 4096)          # absurdly large
    with pytest.raises(RuntimeError, match="No Modal GPU tier"):
        recommend_configs("huge/model", [], top_n=3)
