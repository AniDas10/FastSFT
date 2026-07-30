"""Synthetic dataset generation using distilabel + OpenRouter."""

from typing import Optional

from distilabel.distiset import Distiset

from model.base import Model


class SyntheticDatasetGenerator:
    """Generates a synthetic dataset from a single prompt via distilabel.

    Repeats the prompt across `num_samples` input rows and asks the LLM for
    one generation per row (rather than relying on the API's `n` parameter,
    which many OpenRouter providers silently ignore for `n > 1`), so each
    output row is a distinct completion for the same instruction.
    """

    def __init__(self, model: Optional[Model] = None, num_samples: int = 100):
        self.model = model or Model()
        self.num_samples = num_samples

    def generate(self, prompt: str) -> Distiset:
        data = [{"instruction": prompt}] * self.num_samples
        return self.model.run_pipeline(
            data, self.model.get_instruction(), name="synthetic-dataset-generation"
        )
