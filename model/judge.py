"""LLM-as-judge model, usable standalone at any pipeline stage."""

from typing import Dict, List, Optional

from pydantic import BaseModel

from constants import DEFAULT_JUDGE_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_SCORE_THRESHOLD
from model.base import Model


class Score(BaseModel):
    score: float


class Judge(Model):
    """Evaluates arbitrary text samples against an instruction/prompt:
    numeric scoring and failed-sample counting.
    """

    # Verdicts filter the dataset but never enter it; exempt from open-weight.
    _enforce_open_weight = False

    def __init__(
        self,
        model_id: str = DEFAULT_JUDGE_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__(
            model_id=model_id, api_key=api_key, temperature=temperature, max_tokens=max_tokens
        )

    def score_samples(
        self, samples: Dict[str, str], prompt: Optional[str] = None
    ) -> Dict[str, float]:
        """Scores each sample in `samples` (id -> text). Returns id -> score."""
        instruction = prompt if prompt is not None else self.get_instruction()
        data = [{"id": id_, "instruction": text} for id_, text in samples.items()]

        distiset = self.run_pipeline(
            data,
            instruction,
            structured_output={"schema": Score, "format": "json"},
            name="judge-scoring",
        )
        scores = {}
        for row in distiset["default"]["train"]:
            generation = self._assert_structured_output(row["generation"], row["id"])
            scores[row["id"]] = Score.model_validate_json(generation).score
        return scores

    def failed_sample_count(
        self, scores: List[float], threshold: float = DEFAULT_SCORE_THRESHOLD
    ) -> int:
        """Returns how many of `scores` fall below `threshold`."""
        return sum(1 for score in scores if score < threshold)
