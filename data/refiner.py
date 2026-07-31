"""Refines a Distiset by regenerating samples that fail the judge's quality bar."""

from typing import List, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets
from distilabel.distiset import Distiset

from constants import DEFAULT_SCORE_THRESHOLD, MAX_REFINE_ITERATIONS
from data.generator import SyntheticDataGenerator
from model.base import Model
from model.judge import Judge


class DataRefiner:
    """Replaces low-scoring samples with fresh ones, keeping the count
    constant, up to MAX_REFINE_ITERATIONS times or until nothing fails.
    """

    def __init__(self, parent_model: Model, judge_model: Judge):
        self.parent_model = parent_model
        self.judge_model = judge_model

    def refine(
        self, distiset: Distiset, threshold: float = DEFAULT_SCORE_THRESHOLD
    ) -> Distiset:
        train = distiset["default"]["train"]
        prompt = train[0]["instruction"]

        # Score the initial batch once; thereafter score only fresh rows.
        scores = self._score(train)
        for _ in range(MAX_REFINE_ITERATIONS):
            failed_count = self.judge_model.failed_sample_count(
                scores, threshold=threshold
            )
            if failed_count == 0:
                break

            train, scores = self._drop_failed(train, scores, threshold)
            replacements = self._regenerate(failed_count, prompt)
            train = concatenate_datasets([train, replacements])
            scores = scores + self._score(replacements)

        return Distiset({"default": DatasetDict({"train": train})})

    def _score(self, train: Dataset) -> List[float]:
        """Scores every row in `train`, returned aligned to row order."""
        samples = {str(i): row["generation"] for i, row in enumerate(train)}
        scores_by_id = self.judge_model.score_samples(samples)
        return [scores_by_id[str(i)] for i in range(len(train))]

    def _drop_failed(
        self, train: Dataset, scores: List[float], threshold: float
    ) -> Tuple[Dataset, List[float]]:
        """Returns `train` and `scores` with every row scoring below
        `threshold` removed, kept in alignment."""
        keep_indices = [i for i in range(len(train)) if scores[i] >= threshold]
        return train.select(keep_indices), [scores[i] for i in keep_indices]

    def _regenerate(self, count: int, prompt: str) -> Dataset:
        """Generates `count` fresh replacement rows for `prompt`."""
        replacements = SyntheticDataGenerator(
            model=self.parent_model, num_samples=count
        ).generate(prompt)
        return replacements["default"]["train"]
