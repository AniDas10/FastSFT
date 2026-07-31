"""Synthetic dataset generation using distilabel + OpenRouter."""

from typing import Optional

from distilabel.distiset import Distiset

from model.base import Model


class SyntheticDataGenerator:
    """Generates `num_samples` independent completions of one instruction,
    one API call per sample (not the `n` param, which many providers ignore).
    """

    def __init__(self, model: Optional[Model] = None, num_samples: int = 100):
        self.model = model or Model()
        self.num_samples = num_samples

    def generate(self, prompt: str) -> Distiset:
        data = [{"instruction": prompt}] * self.num_samples
        return self.model.run_pipeline(
            data, self.model.get_instruction(), name="synthetic-dataset-generation"
        )
