"""Tier-0 pin tests for fastsft.training.constants -- GPU tiers + training defaults.

MODAL_GPU_TIERS and BYTES_PER_PARAM feed heuristic.py's cost/feasibility math
(see tests/training/test_heuristic.py for the math itself); a silent edit to a
tier's VRAM/price or a strategy's bytes-per-param would silently mis-rank
candidates without failing loudly, so pin the catalog and key defaults here.
"""

from fastsft.training.constants import (
    ALL_LINEAR_TARGET,
    BATCH_SIZE_CANDIDATES,
    BYTES_PER_PARAM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_MASK_PROMPT_LOSS,
    DEFAULT_MODAL_TIMEOUT_SECONDS,
    DEFAULT_STRATEGY,
    DEFAULT_VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE,
    EVAL_STEPS,
    LORA,
    LORA_TARGET_MODULES,
    MAX_EPOCHS,
    MODAL_GPU_TIERS,
    QLORA,
    TARGET_EFFECTIVE_BATCH,
    TOP_N_CONFIGS,
)


def test_strategy_names():
    assert LORA == "lora"
    assert QLORA == "qlora"
    assert DEFAULT_STRATEGY == LORA


def test_modal_gpu_tier_catalog():
    # No T4: lacks FlashAttention + native bf16, trains much slower despite the lower $/hr.
    assert MODAL_GPU_TIERS == {
        "L4": (24, 0.80),
        "A10G": (24, 1.10),
        "A100-40GB": (40, 2.10),
        "A100-80GB": (80, 2.50),
        "H100": (80, 3.95),
    }


def test_modal_gpu_tiers_vram_and_price_are_monotonic_by_tier_order():
    # Not a strict business requirement, but a sanity check on the catalog:
    # tiers should get more expensive as VRAM increases.
    tiers = list(MODAL_GPU_TIERS.values())
    prices = [price for _, price in tiers]
    assert prices == sorted(prices)


def test_bytes_per_param_qlora_cheaper_than_lora():
    assert BYTES_PER_PARAM == {LORA: 2.0, QLORA: 0.5}
    assert BYTES_PER_PARAM[QLORA] < BYTES_PER_PARAM[LORA]


def test_lora_defaults():
    assert DEFAULT_LORA_RANK == 16
    assert DEFAULT_LORA_DROPOUT == 0.05
    assert LORA_TARGET_MODULES == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_all_linear_target_sentinel():
    assert ALL_LINEAR_TARGET == "all-linear"


def test_training_loop_defaults():
    assert DEFAULT_LEARNING_RATE == 2e-4
    assert DEFAULT_VALIDATION_SPLIT == 0.15
    assert MAX_EPOCHS == 10
    assert EVAL_STEPS == 20
    assert EARLY_STOPPING_PATIENCE == 3


def test_default_mask_prompt_loss_is_on():
    # Training/eval loss should reflect the answer only by default.
    assert DEFAULT_MASK_PROMPT_LOSS is True


def test_modal_timeout_default():
    assert DEFAULT_MODAL_TIMEOUT_SECONDS == 3600


def test_batch_size_search_space():
    assert TARGET_EFFECTIVE_BATCH == 16
    assert BATCH_SIZE_CANDIDATES == (16, 8, 4, 2, 1)
    # Candidates must be sorted descending -- heuristic picks the first that fits.
    assert list(BATCH_SIZE_CANDIDATES) == sorted(BATCH_SIZE_CANDIDATES, reverse=True)


def test_top_n_configs():
    assert TOP_N_CONFIGS == 3
