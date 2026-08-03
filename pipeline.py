"""Top-level DistillationPipeline: DataGenerator -> DataFormatter -> FineTuner."""

from collections.abc import Iterator
from typing import Any

from constants import DEFAULT_CHILD_MODEL_ID
from data.config import DataGenerationConfig
from stages.base import Stage
from stages.constants import (
    DATA_FORMATTER,
    DATA_GENERATOR,
    FINE_TUNER,
    STAGE_NAMES,
    STAGE_ORDER,
)
from stages.data_formatter import DataFormatter
from stages.data_generator import DataGenerator
from stages.fine_tuner import FineTuner
from training.config import TrainingConfig


class DistillationPipeline:
    """Runs the stages in STAGE_ORDER, starting at start_stage.

    - "data_generator" (default): run(prompt) runs everything.
    - "data_formatter": run(raw_dataset), needs a `messages` column.
    - "fine_tuner": run(formatted_dataset), needs a `text` column.

    run() yields (stage, output) as each stage completes, so a caller can
    persist incrementally.
    """

    def __init__(
        self,
        child_model_id: str = DEFAULT_CHILD_MODEL_ID,
        generation: DataGenerationConfig | None = None,
        training: TrainingConfig | None = None,
        local_training: bool = False,
        start_stage: str = STAGE_ORDER[0],
        verbose: bool = True,
    ):
        self._validate_start_stage(start_stage)
        self.start_stage = start_stage
        self._start_index = STAGE_ORDER.index(start_stage)

        self.stages: list[Stage] = self._build_stages(
            child_model_id=child_model_id,
            generation=generation or DataGenerationConfig(),
            training=training,
            local_training=local_training,
            verbose=verbose,
        )

    def _validate_start_stage(self, start_stage: str) -> None:
        if start_stage not in STAGE_NAMES:
            raise ValueError(
                f"start_stage must be one of {STAGE_ORDER}, got {start_stage!r}."
            )

    def _build_stages(
        self,
        child_model_id: str,
        generation: DataGenerationConfig,
        training: TrainingConfig | None,
        local_training: bool,
        verbose: bool,
    ) -> list[Stage]:
        """Builds only the stages from self._start_index onward."""
        factories = {
            DATA_GENERATOR: lambda: DataGenerator(
                guide_model=generation.guide_model,
                parent_model=generation.parent_model,
                judge_model=generation.judge_model,
                num_samples=generation.num_samples,
                breadth_exponent=generation.breadth_exponent,
                score_threshold=generation.score_threshold,
                parent_temperature=generation.parent_generation.temperature,
                parent_max_tokens=generation.parent_generation.max_tokens,
                verbose=verbose,
            ),
            DATA_FORMATTER: lambda: DataFormatter(
                child_model_id=child_model_id, verbose=verbose
            ),
            FINE_TUNER: lambda: FineTuner(
                child_model_id=child_model_id,
                training_config=training,
                local_training=local_training,
                verbose=verbose,
            ),
        }
        return [factories[name]() for name in STAGE_ORDER[self._start_index:]]

    def run(self, data: Any) -> Iterator[tuple[Stage, Any]]:
        """Yields (stage, output) as each stage completes. Validates the input
        eagerly, before any stage runs."""
        if data is None:
            raise ValueError(
                f"run() requires an input for start_stage={self.start_stage!r} "
                f"({'a prompt string' if self.start_stage == STAGE_ORDER[0] else 'a Distiset'})."
            )
        return self._run_stages(data)

    def _run_stages(self, data: Any) -> Iterator[tuple[Stage, Any]]:
        for stage in self.stages:
            data = stage.run(data)
            yield stage, data
