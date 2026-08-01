"""Generates the assistant answers for a set of instructions."""

from typing import List, Optional

from distilabel.distiset import Distiset

from model.base import Model


class ResponseGenerator:
    """Generates one styled answer per instruction via the parent model,
    one API call per instruction (not the `n` param, which many providers ignore).
    """

    def __init__(self, model: Optional[Model] = None):
        self._model = model or Model()

    def generate(self, instructions: List[str]) -> Distiset:
        data = [{"instruction": instruction} for instruction in instructions]
        return self._model.run_pipeline(
            data, self._model.get_instruction(), name="response-generation"
        )
