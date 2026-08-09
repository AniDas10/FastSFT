"""Turns a freeform user request into tailored parent/judge instructions."""


from pydantic import BaseModel

from fastsft.constants import DEFAULT_GUIDE_MODEL
from fastsft.model.base import Model
from fastsft.model.constants import DEFAULT_GUIDE_INSTRUCTION, DEFAULT_MAX_TOKENS


class GuideInstructions(BaseModel):
    parent_instruction: str
    judge_instruction: str
    sample_instructions: list[str]


class Guide(Model):
    """Generate parent/judge prompts and seed instructions from a freeform request."""

    _enforce_open_weight = False

    def __init__(
        self,
        model_id: str = DEFAULT_GUIDE_MODEL,
        api_key: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__(
            model_id=model_id, api_key=api_key, temperature=temperature, max_tokens=max_tokens
        )

    def _instruction(self) -> str:
        return DEFAULT_GUIDE_INSTRUCTION

    def generate_instructions(
        self, user_input: str, num_seeds: int
    ) -> GuideInstructions:
        """Produce parent/judge prompts and num_seeds topic instructions."""
        distiset = self.run_pipeline(
            [{"instruction": user_input}],
            self.get_instruction().format(num_seeds=num_seeds),
            structured_output={"schema": GuideInstructions, "format": "json"},
            name="guide-instructions",
        )
        row = next(iter(distiset["default"]["train"]))
        generation = self.assert_structured_output(row["generation"])
        return GuideInstructions.model_validate_json(generation)
