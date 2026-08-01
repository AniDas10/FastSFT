"""Top-level DistillationPipeline: DataGenerator -> DataFormatter -> FineTuner."""

from typing import Any, Iterator, List, Tuple

from constants import (
    DEFAULT_CHILD_MODEL_ID,
    DEFAULT_GUIDE_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PARENT_MODEL,
)
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
        guide_model: str = DEFAULT_GUIDE_MODEL,
        parent_model: str = DEFAULT_PARENT_MODEL,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 100,
        start_stage: str = STAGE_ORDER[0],
        verbose: bool = True,
    ):
        self._validate_start_stage(start_stage)
        self.start_stage = start_stage
        self._start_index = STAGE_ORDER.index(start_stage)

        self.stages: List[Stage] = self._build_stages(
            child_model_id=child_model_id,
            guide_model=guide_model,
            parent_model=parent_model,
            judge_model=judge_model,
            num_samples=num_samples,
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
        guide_model: str,
        parent_model: str,
        judge_model: str,
        num_samples: int,
        verbose: bool,
    ) -> List[Stage]:
        """Builds only the stages from self._start_index onward."""
        factories = {
            DATA_GENERATOR: lambda: DataGenerator(
                guide_model=guide_model,
                parent_model=parent_model,
                judge_model=judge_model,
                num_samples=num_samples,
                verbose=verbose,
            ),
            DATA_FORMATTER: lambda: DataFormatter(
                child_model_id=child_model_id, verbose=verbose
            ),
            FINE_TUNER: lambda: FineTuner(
                child_model_id=child_model_id, verbose=verbose
            ),
        }
        return [factories[name]() for name in STAGE_ORDER[self._start_index:]]

    def run(self, data: Any) -> Iterator[Tuple[Stage, Any]]:
        """Yields (stage, output) as each stage completes. Validates the input
        eagerly, before any stage runs."""
        if data is None:
            raise ValueError(
                f"run() requires an input for start_stage={self.start_stage!r} "
                f"({'a prompt string' if self.start_stage == STAGE_ORDER[0] else 'a Distiset'})."
            )
        return self._run_stages(data)

    def _run_stages(self, data: Any) -> Iterator[Tuple[Stage, Any]]:
        for stage in self.stages:
            data = stage.run(data)
            yield stage, data
