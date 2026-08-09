"""Configuration for DataGenerator's generation-side models and tuning."""

from dataclasses import dataclass, field

from fastsft.constants import (
    DEFAULT_GUIDE_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PARENT_MODEL,
)
from fastsft.data.constants import (
    BREADTH_EXPONENT,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_PARENT_TEMPERATURE,
)
from fastsft.model.constants import DEFAULT_MAX_TOKENS, DEFAULT_SCORE_THRESHOLD


@dataclass
class ParentGenerationConfig:
    """Tuning for the parent model's own generation calls."""

    temperature: float = DEFAULT_PARENT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


@dataclass
class DataGenerationConfig:
    guide_model: str = DEFAULT_GUIDE_MODEL
    parent_model: str = DEFAULT_PARENT_MODEL
    judge_model: str = DEFAULT_JUDGE_MODEL
    num_samples: int = DEFAULT_NUM_SAMPLES
    breadth_exponent: float = BREADTH_EXPONENT
    score_threshold: float = DEFAULT_SCORE_THRESHOLD
    parent_generation: ParentGenerationConfig = field(default_factory=ParentGenerationConfig)
