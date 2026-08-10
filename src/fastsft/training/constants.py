"""Constants for the training package (FineTuner's config heuristic + Modal job)."""

# Modal GPU tier catalog: name -> (vram_gb, approx_usd_per_hour).
# NOTE: these are approximate published Modal figures at time of writing --
# verify against modal.com/pricing before relying on them; both the tier
# list and pricing drift over time.
# No T4: lacks FlashAttention + native bf16, trains much slower despite the lower $/hr.
MODAL_GPU_TIERS = {
    "L4": (24, 0.80),
    "A10G": (24, 1.10),
    "A100-40GB": (40, 2.10),
    "A100-80GB": (80, 2.50),
    "H100": (80, 3.95),
}

# Canonical strategy names -- the single symbolic source for every place
# that refers to a training strategy by name.
LORA = "lora"
QLORA = "qlora"

DEFAULT_STRATEGY = LORA
DEFAULT_LORA_RANK = 16
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_BATCH_SIZE = 8
DEFAULT_GRAD_ACCUMULATION = 1
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
DEFAULT_LEARNING_RATE = 2e-4
DEFAULT_VALIDATION_SPLIT = 0.15
DEFAULT_MODAL_TIMEOUT_SECONDS = 3600
TARGET_EFFECTIVE_BATCH = 16
BATCH_SIZE_CANDIDATES = (16, 8, 4, 2, 1)
TOP_N_CONFIGS = 3

# Bytes/parameter for the frozen base, by strategy.
BYTES_PER_PARAM = {LORA: 2.0, QLORA: 0.5}

# Deliberately coarse memory-estimate constants -- good enough to RANK
# candidates by feasibility, not a precise predictor (see heuristic.py).
SAFETY_MARGIN = 1.3
ACTIVATION_BYTES_PER_TOKEN_PER_LAYER_UNIT = 20.0
ADAPTER_BASE_OVERHEAD_GB = 0.05
ADAPTER_PER_RANK_GB = 0.002
DEFAULT_FALLBACK_SEQ_LEN = 512

# MAX_EPOCHS is an upper bound; early stopping decides the actual count.
MAX_EPOCHS = 10
EVAL_STEPS = 20
EARLY_STOPPING_PATIENCE = 3

# Mask the prompt (system+user) tokens from the loss so training/eval loss
# reflects the answer only. On by default; the trainer resolves HOW to mask in
# a model-agnostic way (see training/trainer.py::_resolve_loss_masking). Set
# False to fall back to loss over the whole formatted sequence.
DEFAULT_MASK_PROMPT_LOSS = True

# Passing this single value as target_modules makes PEFT auto-target every
# linear layer, regardless of a model's attention-projection naming -- the
# model-agnostic escape hatch for architectures the default names don't fit.
ALL_LINEAR_TARGET = "all-linear"
