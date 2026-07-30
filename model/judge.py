"""LLM-as-judge model, usable standalone at any pipeline stage."""

from typing import Dict, List, Optional

from distilabel.distiset import Distiset
from pydantic import BaseModel

from constants import DEFAULT_JUDGE_MODEL, DEFAULT_MAX_TOKENS, DEFAULT_SCORE_THRESHOLD
from model.base import Model


class Score(BaseModel):
    score: float


class Judge(Model):
    """Evaluates arbitrary text samples against an instruction/prompt.

    Inherits model_id/api_key/temperature/build_llm/instruction handling
    from Model, and adds judging-specific behavior on top: free-text
    evaluation, numeric scoring, and a helper for counting failed samples.
    """

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

    def evaluate(self, samples: List[str], prompt: Optional[str] = None) -> Distiset:
        """Runs the judging instructions against each of `samples`.

        `prompt` overrides this instance's instruction for this call only;
        if omitted, `get_instruction()` is used. Returns the resulting
        Distiset; each row's "generation" column holds the judge's verdict
        for the sample at the same index.
        """
        instruction = prompt if prompt is not None else self.get_instruction()
        data = [{"instruction": s} for s in samples]
        return self.run_pipeline(data, instruction, name="judge-evaluation")

    def score_samples(
        self, samples: Dict[str, str], prompt: Optional[str] = None
    ) -> Dict[str, float]:
        """Scores each sample in `samples` (id -> text) against the judging
        instructions.

        `prompt` overrides this instance's instruction for this call only
        (should describe a 0-10 rating scale); if omitted, `get_instruction()`
        is used. The judge is forced via structured output to return only a
        numeric score. Returns a mapping of id -> score so callers (e.g.
        data/refactor.py) can look up individual samples by id afterwards.
        """
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
        self, scores: Dict[str, float], threshold: float = DEFAULT_SCORE_THRESHOLD
    ) -> int:
        """Returns how many of `scores` fall below `threshold`.

        data/refactor.py uses this to know how many replacement samples to
        generate for the ones that failed the quality bar.
        """
        return sum(1 for score in scores.values() if score < threshold)
