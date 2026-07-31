"""Turns a freeform user request into tailored parent/judge instructions."""

from typing import Optional

from pydantic import BaseModel

from constants import DEFAULT_GUIDE_INSTRUCTION, DEFAULT_GUIDE_MODEL, DEFAULT_MAX_TOKENS
from model.base import Model


class GuideInstructions(BaseModel):
    parent_instruction: str
    judge_instruction: str
    sample_instruction: str


class Guide(Model):
    """Generates tailored parent/judge system prompts, and a per-sample
    instruction, from a freeform request -- in one structured-output call.
    """

    # Output shapes instructions, not training data; exempt from open-weight.
    _enforce_open_weight = False

    def __init__(
        self,
        model_id: str = DEFAULT_GUIDE_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__(
            model_id=model_id, api_key=api_key, temperature=temperature, max_tokens=max_tokens
        )

    def _instruction(self) -> str:
        return DEFAULT_GUIDE_INSTRUCTION

    def generate_instructions(self, user_input: str) -> GuideInstructions:
        """Produces (parent_instruction, judge_instruction, sample_instruction)
        from `user_input`; sample_instruction asks for exactly one item."""
        distiset = self.run_pipeline(
            [{"instruction": user_input}],
            self.get_instruction(),
            structured_output={"schema": GuideInstructions, "format": "json"},
            name="guide-instructions",
        )
        row = next(iter(distiset["default"]["train"]))
        generation = self._assert_structured_output(row["generation"])
        return GuideInstructions.model_validate_json(generation)
