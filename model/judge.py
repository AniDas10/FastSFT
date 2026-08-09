"""LLM-as-judge model, usable standalone at any pipeline stage."""

from typing import Literal

from pydantic import BaseModel

from constants import DEFAULT_JUDGE_MODEL
from model.base import Model
from model.constants import DEFAULT_MAX_TOKENS, DEFAULT_SCORE_THRESHOLD


class Score(BaseModel):
    score: float


class Verdict(BaseModel):
    winner: Literal["A", "B", "tie"]


class Judge(Model):
    """Evaluates arbitrary text samples against an instruction/prompt: absolute
    numeric scoring, failed-sample counting, and pairwise comparison.
    """

    # Verdicts filter the dataset but never enter it; exempt from open-weight.
    _enforce_open_weight = False

    def __init__(
        self,
        model_id: str = DEFAULT_JUDGE_MODEL,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        super().__init__(
            model_id=model_id, api_key=api_key, temperature=temperature, max_tokens=max_tokens
        )

    def score_samples(
        self, samples: dict[str, str], prompt: str | None = None
    ) -> dict[str, float]:
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
            generation = self.assert_structured_output(row["generation"], row["id"])
            scores[row["id"]] = Score.model_validate_json(generation).score
        return scores

    def failed_sample_count(
        self, scores: list[float], threshold: float = DEFAULT_SCORE_THRESHOLD
    ) -> int:
        """Returns how many of `scores` fall below `threshold`."""
        return sum(1 for score in scores if score < threshold)

    def compare_samples(
        self, pairs: dict[str, tuple[str, str, str]], prompt: str | None = None
    ) -> dict[str, Verdict]:
        """Pairwise-compares two candidate answers to the same question.

        `pairs` maps id -> (question, answer_a, answer_b); returns id -> Verdict
        naming the better answer ("A"/"B") or a "tie". The comparison rubric is
        the caller-supplied `prompt` (else this role's instruction) -- the same
        pattern as `score_samples`. Callers cancel position bias by comparing
        each pair a second time with A/B swapped.
        """
        data = [
            {"id": id_, "instruction": self._comparison_prompt(question, a, b)}
            for id_, (question, a, b) in pairs.items()
        ]
        return self._run_comparison(data, prompt, name="judge-comparison")

    def compare_to_reference(
        self, pairs: dict[str, tuple[str, str, str, str]], prompt: str | None = None
    ) -> dict[str, Verdict]:
        """Pairwise-compares two candidate answers by how closely each matches a
        reference answer.

        `pairs` maps id -> (question, reference, answer_a, answer_b); returns
        id -> Verdict naming the candidate ("A"/"B") more like the reference, or
        "tie". Same structured-output mechanism as `compare_samples`; the rubric
        (`prompt`) defines what "like the reference" means (e.g. matching the
        reference's style rather than its correctness). Position bias is
        cancelled the same way, by swapping A/B on a second pass.
        """
        data = [
            {"id": id_, "instruction": self._reference_prompt(question, reference, a, b)}
            for id_, (question, reference, a, b) in pairs.items()
        ]
        return self._run_comparison(data, prompt, name="judge-reference-comparison")

    def _run_comparison(
        self, data: list[dict], prompt: str | None, name: str
    ) -> dict[str, Verdict]:
        """Runs a Verdict-schema comparison over already-rendered `data` rows
        (each {"id", "instruction"}); shared by the pairwise comparators."""
        instruction = prompt if prompt is not None else self.get_instruction()
        distiset = self.run_pipeline(
            data,
            instruction,
            structured_output={"schema": Verdict, "format": "json"},
            name=name,
        )
        verdicts = {}
        for row in distiset["default"]["train"]:
            generation = self.assert_structured_output(row["generation"], row["id"])
            verdicts[row["id"]] = Verdict.model_validate_json(generation)
        return verdicts

    def _comparison_prompt(self, question: str, a: str, b: str) -> str:
        """Presents a question and two candidate answers (A, B) for judging."""
        return (
            f"User question:\n{question}\n\n"
            f"Response A:\n{a}\n\n"
            f"Response B:\n{b}"
        )

    def _reference_prompt(self, question: str, reference: str, a: str, b: str) -> str:
        """Presents a question, a reference answer, and two candidates (A, B)."""
        return (
            f"User question:\n{question}\n\n"
            f"Reference response:\n{reference}\n\n"
            f"Response A:\n{a}\n\n"
            f"Response B:\n{b}"
        )
