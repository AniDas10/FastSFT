"""Top-level DistillationPipeline: DataGenerator -> DataFormatter -> FineTuner."""

from typing import Optional

from distilabel.distiset import Distiset

from constants import (
    DEFAULT_CHILD_MODEL_ID,
    DEFAULT_GUIDE_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PARENT_MODEL,
)
from stages.data_formatter import DataFormatter
from stages.data_generator import DataGenerator
from stages.fine_tuner import FineTuner


class DistillationPipeline:
    """Runs the three mini-pipelines in sequence: DataGenerator ->
    DataFormatter -> FineTuner.

    Each stage can be skipped by supplying its input directly instead:
    - skip_generation=True + raw_dataset=...: bring your own raw dataset,
      still runs DataFormatter + FineTuner.
    - skip_formatting=True + formatted_dataset=...: bring your own dataset
      already formatted for the child model, runs only FineTuner (implies
      skipping generation too, since there'd be nothing left to format).

    Stages that won't run given the skip flags aren't constructed at all --
    e.g. DataFormatter.__init__ does a real network fetch (the child
    model's tokenizer), so building it when skip_formatting=True would pay
    for (and could fail on) a stage that's never actually used.

    run() validates that prompt/raw_dataset/formatted_dataset are each
    provided if and only if the current skip flags will use them -- see
    _validate_inputs.
    """

    def __init__(
        self,
        child_model_id: str = DEFAULT_CHILD_MODEL_ID,
        guide_model: str = DEFAULT_GUIDE_MODEL,
        parent_model: str = DEFAULT_PARENT_MODEL,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 100,
        skip_generation: bool = False,
        skip_formatting: bool = False,
        verbose: bool = True,
    ):
        self.skip_generation = skip_generation
        self.skip_formatting = skip_formatting

        self.data_generator: Optional[DataGenerator] = (
            None
            if skip_generation or skip_formatting
            else DataGenerator(
                guide_model=guide_model,
                parent_model=parent_model,
                judge_model=judge_model,
                num_samples=num_samples,
                verbose=verbose,
            )
        )
        self.data_formatter: Optional[DataFormatter] = (
            None
            if skip_formatting
            else DataFormatter(child_model_id=child_model_id, verbose=verbose)
        )
        self.fine_tuner = FineTuner(child_model_id=child_model_id, verbose=verbose)

        # Populated by run() -- lets callers (e.g. main.py) save each stage's
        # output even though FineTuner may raise before returning anything.
        self.raw_dataset: Optional[Distiset] = None
        self.formatted_dataset: Optional[Distiset] = None

    def run(
        self,
        prompt: Optional[str] = None,
        raw_dataset: Optional[Distiset] = None,
        formatted_dataset: Optional[Distiset] = None,
    ):
        self._validate_inputs(prompt, raw_dataset, formatted_dataset)

        if self.skip_formatting:
            self.formatted_dataset = formatted_dataset
        else:
            if self.skip_generation:
                self.raw_dataset = raw_dataset
            else:
                self.raw_dataset = self.data_generator.run(prompt)
            self.formatted_dataset = self.data_formatter.run(self.raw_dataset)

        return self.fine_tuner.run(self.formatted_dataset)

    def _validate_inputs(
        self,
        prompt: Optional[str],
        raw_dataset: Optional[Distiset],
        formatted_dataset: Optional[Distiset],
    ) -> None:
        """Every input must be provided if and only if this run will
        actually use it -- catches contradictory skip-flag/input
        combinations instead of silently ignoring the unused one."""
        if self.skip_formatting:
            if formatted_dataset is None:
                raise ValueError(
                    "skip_formatting=True requires formatted_dataset to be provided."
                )
            if prompt is not None or raw_dataset is not None:
                raise ValueError(
                    "skip_formatting=True skips both DataGenerator and "
                    "DataFormatter, so prompt/raw_dataset would be silently "
                    "ignored -- pass only formatted_dataset, or use "
                    "skip_generation=True instead if you want DataFormatter to "
                    "still run on your own raw_dataset."
                )
            return

        if formatted_dataset is not None:
            raise ValueError(
                "formatted_dataset was provided but skip_formatting=False, so "
                "DataFormatter will run and overwrite it -- set "
                "skip_formatting=True to use your own formatted_dataset directly."
            )

        if self.skip_generation:
            if raw_dataset is None:
                raise ValueError(
                    "skip_generation=True requires raw_dataset to be provided."
                )
            if prompt is not None:
                raise ValueError(
                    "skip_generation=True skips DataGenerator, so prompt would "
                    "be silently ignored -- pass only raw_dataset."
                )
        else:
            if raw_dataset is not None:
                raise ValueError(
                    "raw_dataset was provided but skip_generation=False, so "
                    "DataGenerator will run and overwrite it -- set "
                    "skip_generation=True to use your own raw_dataset directly."
                )
            if not prompt or not prompt.strip():
                raise ValueError("prompt is required unless skip_generation=True.")
