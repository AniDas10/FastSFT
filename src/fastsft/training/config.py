"""Fine-tuning configuration: adapter shape, training loop, and Modal job envelope."""

from dataclasses import dataclass, field

from fastsft.training.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_ACCUMULATION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_MASK_PROMPT_LOSS,
    DEFAULT_MODAL_TIMEOUT_SECONDS,
    DEFAULT_STRATEGY,
    DEFAULT_VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE,
    EVAL_STEPS,
    LORA_TARGET_MODULES,
    MAX_EPOCHS,
)


@dataclass
class AdapterConfig:
    """Shape of the LoRA adapter itself -- not the base model's quantization
    (see TrainingConfig.strategy for that)."""

    rank: int = DEFAULT_LORA_RANK
    target_modules: list[str] = field(default_factory=lambda: list(LORA_TARGET_MODULES))
    dropout: float = DEFAULT_LORA_DROPOUT


@dataclass
class TrainingLoopConfig:
    """The training loop's own optimization/evaluation knobs."""

    batch_size: int = DEFAULT_BATCH_SIZE
    grad_accumulation: int = DEFAULT_GRAD_ACCUMULATION
    learning_rate: float = DEFAULT_LEARNING_RATE
    # Upper bound on epochs -- early stopping decides the actual count, this
    # is never fixed a priori, only capped.
    max_epochs: int = MAX_EPOCHS
    eval_steps: int = EVAL_STEPS
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    validation_split: float = DEFAULT_VALIDATION_SPLIT
    # Mask prompt tokens from the loss (answer-only); False = loss over the whole sequence.
    mask_prompt_loss: bool = DEFAULT_MASK_PROMPT_LOSS


@dataclass
class TrainingConfig:
    """A single candidate fine-tuning configuration."""

    gpu_tier: str
    # Whether the frozen base model is quantized; paired with gpu_tier (heuristic searches both jointly).
    strategy: str = DEFAULT_STRATEGY  # training.constants.LORA | QLORA
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    loop: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)
    modal_timeout_seconds: int = DEFAULT_MODAL_TIMEOUT_SECONDS
    # Cost/memory estimates from the heuristic; purely informational, never read by the training job.
    est_memory_gb: float = 0.0
    est_usd_per_hour: float = 0.0
