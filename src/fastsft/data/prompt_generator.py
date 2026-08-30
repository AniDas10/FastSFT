"""Generates the dataset's user instructions directly via the parent model.

Seeds provide breadth (distinct topics); each seed is expanded into depth
(complexity-varied) instructions.
"""

import math

from pydantic import BaseModel

from fastsft.data.constants import (
    BREADTH_EXPONENT,
    MAX_PROMPT_ATTEMPTS,
    PROMPT_GENERATOR_INSTRUCTION,
)
from fastsft.model.base import Model


def seed_count(num_samples: int, breadth_exponent: float = BREADTH_EXPONENT) -> int:
    """Breadth: number of distinct seed topics for `num_samples`
    (ceil(N ** breadth_exponent), clamped to [1, num_samples])."""
    # math.pow (not `**`) so the result types as float, not Any -- typeshed types `**` as Any.
    return max(1, min(num_samples, math.ceil(math.pow(num_samples, breadth_exponent))))


class GeneratedPrompts(BaseModel):
    prompts: list[str]


class PromptGenerator:
    """Expands seed topics into exactly `num_samples` user instructions: each
    seed is turned into its allotted number of questions spanning simple to
    complex.
    """

    def __init__(self, model: Model, num_samples: int):
        self._model = model
        self._num_samples = num_samples

    def generate(self, seeds: list[str]) -> list[str]:
        """Returns exactly `num_samples` instructions spread across `seeds`."""
        if not seeds:
            raise ValueError("PromptGenerator.generate() requires at least one seed.")

        # Each row is capped at its requested count, so a pass yields at most
        # the deficit -- top up whatever a model under-delivers.
        prompts: list[str] = []
        for _ in range(MAX_PROMPT_ATTEMPTS):
            deficit = self._num_samples - len(prompts)
            if deficit == 0:
                break
            prompts.extend(self._generate(self._allocate(seeds, deficit)))
        else:
            raise RuntimeError(
                f"PromptGenerator produced {len(prompts)}/{self._num_samples} "
                f"instructions after {MAX_PROMPT_ATTEMPTS} attempts."
            )
        return prompts

    def _generate(self, allocation: list[tuple[str, int]]) -> list[str]:
        """Generates the capped prompts for one (seed, count) allocation."""
        data = [
            {"instruction": self._row_prompt(seed, count), "count": count}
            for seed, count in allocation
        ]
        distiset = self._model.run_pipeline(
            data,
            PROMPT_GENERATOR_INSTRUCTION,
            structured_output={"schema": GeneratedPrompts, "format": "json"},
            name="prompt-generation",
        )

        prompts: list[str] = []
        for row in distiset["default"]["train"]:
            generation = self._model.assert_structured_output(row["generation"])
            parsed = GeneratedPrompts.model_validate_json(generation)
            prompts.extend(parsed.prompts[: row["count"]])
        return prompts

    def _allocate(self, seeds: list[str], n: int) -> list[tuple[str, int]]:
        """Distributes `n` instructions as evenly as possible across `seeds`,
        dropping any seed that would get zero."""
        base, extra = divmod(n, len(seeds))
        counts = (
            (seed, base + (1 if i < extra else 0)) for i, seed in enumerate(seeds)
        )
        return [(seed, count) for seed, count in counts if count > 0]

    def _row_prompt(self, seed: str, count: int) -> str:
        return (
            f"Seed question: {seed}\n\n"
            f"Produce exactly {count} distinct user questions on this same topic, "
            f"ordered from the simplest to the most complex."
        )
