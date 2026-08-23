"""Configuration for the evaluation run: which adapter to score, which parent/
judge models to score it against, and the eval-set/inference tuning knobs."""

from dataclasses import dataclass

from fastsft.constants import DEFAULT_JUDGE_MODEL, DEFAULT_PARENT_MODEL
from fastsft.data.constants import DEFAULT_PARENT_TEMPERATURE
from fastsft.eval.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_EVAL_PROMPTS,
    DEFAULT_SWAP_POSITIONS,
)
from fastsft.model.constants import DEFAULT_MAX_TOKENS


@dataclass
class EvalConfig:
    """A single evaluation run against one saved adapter directory."""

    adapter_dir: str
    # This eval run's own id (names its evalsets_dir()/<run_id> folder), distinct from adapter_dir's.
    run_id: str
    parent_model: str = DEFAULT_PARENT_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    num_eval_prompts: int = DEFAULT_NUM_EVAL_PROMPTS
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    inference_batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE
    swap_positions: bool = DEFAULT_SWAP_POSITIONS
    # The parent's style system prompt, inferred from the run's training metadata by default (eval/run.py).
    parent_instruction: str = ""
    # The parent's generation recipe, also inferred from training metadata; falls back to pipeline defaults.
    parent_max_tokens: int = DEFAULT_MAX_TOKENS
    parent_temperature: float = DEFAULT_PARENT_TEMPERATURE
