"""Refines a Distiset by regenerating samples that fail the judge's quality bar."""

from typing import Dict

from datasets import Dataset, DatasetDict, concatenate_datasets

from distilabel.distiset import Distiset

from constants import DEFAULT_SCORE_THRESHOLD, MAX_REFINE_ITERATIONS
from data.generator import SyntheticDatasetGenerator
from model.base import Model
from model.judge import Judge


class DatasetRefactor:
    """Iteratively replaces low-scoring samples in a Distiset with fresh ones.

    Scores every sample with the judge, drops the ones below threshold, and
    regenerates exactly that many replacements via SyntheticDatasetGenerator
    (keeping the total sample count constant) -- repeating up to
    MAX_REFINE_ITERATIONS times or until nothing fails anymore.
    """

    def __init__(self, parent_model: Model, judge_model: Judge):
        self.parent_model = parent_model
        self.judge_model = judge_model

    def refine(
        self, distiset: Distiset, threshold: float = DEFAULT_SCORE_THRESHOLD
    ) -> Distiset:
        train = distiset["default"]["train"]
        prompt = train[0]["instruction"]

        for _ in range(MAX_REFINE_ITERATIONS):
            scores = self._score(train)
            failed_count = self.judge_model.failed_sample_count(
                scores, threshold=threshold
            )
            if failed_count == 0:
                break

            train = self._drop_failed(train, scores, threshold)
            train = concatenate_datasets(
                [train, self._regenerate(failed_count, prompt)]
            )

        return Distiset({"default": DatasetDict({"train": train})})

    def _score(self, train: Dataset) -> Dict[str, float]:
        """Scores every row in `train`, keyed by its positional index."""
        samples = {str(i): row["generation"] for i, row in enumerate(train)}
        return self.judge_model.score_samples(samples)

    def _drop_failed(
        self, train: Dataset, scores: Dict[str, float], threshold: float
    ) -> Dataset:
        """Returns `train` with every row scoring below `threshold` removed."""
        keep_indices = [
            i for i in range(len(train)) if scores[str(i)] >= threshold
        ]
        return train.select(keep_indices)

    def _regenerate(self, count: int, prompt: str) -> Dataset:
        """Generates `count` fresh replacement rows for `prompt`."""
        replacements = SyntheticDatasetGenerator(
            model=self.parent_model, num_samples=count
        ).generate(prompt)
        return replacements["default"]["train"]
