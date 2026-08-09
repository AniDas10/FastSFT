"""Refines a Distiset by regenerating samples that fail the judge's quality bar."""


from datasets import Dataset, concatenate_datasets
from distilabel.distiset import Distiset

from fastsft.data.constants import MAX_REFINE_ITERATIONS
from fastsft.data.response_generator import ResponseGenerator
from fastsft.helper import convert_to_distiset
from fastsft.model.base import Model
from fastsft.model.constants import DEFAULT_SCORE_THRESHOLD
from fastsft.model.judge import Judge


class DataRefiner:
    """Re-answers low-scoring samples, keeping their instructions and the
    count constant, up to MAX_REFINE_ITERATIONS times or until nothing fails.
    """

    def __init__(self, parent_model: Model, judge_model: Judge):
        self._parent_model = parent_model
        self._judge_model = judge_model

    def refine(
        self, distiset: Distiset, threshold: float = DEFAULT_SCORE_THRESHOLD
    ) -> Distiset:
        train = distiset["default"]["train"]

        # Score the initial batch once; thereafter score only fresh rows.
        scores = self._score(train)
        for _ in range(MAX_REFINE_ITERATIONS):
            if self._judge_model.failed_sample_count(scores, threshold=threshold) == 0:
                break

            failed_instructions = self._failed_instructions(train, scores, threshold)
            train, scores = self._drop_failed(train, scores, threshold)
            replacements = self._regenerate(failed_instructions)
            train = concatenate_datasets([train, replacements])
            scores = scores + self._score(replacements)

        return convert_to_distiset(train)

    def _score(self, train: Dataset) -> list[float]:
        """Scores every row in `train`, returned aligned to row order."""
        samples = {str(i): row["generation"] for i, row in enumerate(train)}
        scores_by_id = self._judge_model.score_samples(samples)
        return [scores_by_id[str(i)] for i in range(len(train))]

    def _failed_instructions(
        self, train: Dataset, scores: list[float], threshold: float
    ) -> list[str]:
        """Instructions of the rows scoring below `threshold`."""
        return [
            train[i]["instruction"]
            for i in range(len(train))
            if scores[i] < threshold
        ]

    def _drop_failed(
        self, train: Dataset, scores: list[float], threshold: float
    ) -> tuple[Dataset, list[float]]:
        """Returns `train` and `scores` with every row scoring below
        `threshold` removed, kept in alignment."""
        keep_indices = [i for i in range(len(train)) if scores[i] >= threshold]
        return train.select(keep_indices), [scores[i] for i in keep_indices]

    def _regenerate(self, instructions: list[str]) -> Dataset:
        """Generates fresh answers for `instructions`."""
        replacements = ResponseGenerator(model=self._parent_model).generate(
            instructions
        )
        return replacements["default"]["train"]
