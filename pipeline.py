"""Orchestrates the full guide -> generate -> refine pipeline."""

from distilabel.distiset import Distiset

from constants import DEFAULT_GUIDE_MODEL, DEFAULT_JUDGE_MODEL, DEFAULT_PARENT_MODEL
from data.generator import SyntheticDatasetGenerator
from data.refactor import DatasetRefactor
from model.base import Model
from model.guide import Guide
from model.judge import Judge


class DistillationPipeline:
    """Runs guide -> generate -> refine and returns the refined Distiset.

    Extracted out of main.py so the pipeline is reusable independently of
    the CLI (a script, a notebook, ...) -- main.py itself only handles
    argument parsing and saving the result to disk.
    """

    def __init__(
        self,
        guide_model: str = DEFAULT_GUIDE_MODEL,
        parent_model: str = DEFAULT_PARENT_MODEL,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 100,
        verbose: bool = True,
    ):
        self.guide_model = guide_model
        self.parent_model = parent_model
        self.judge_model = judge_model
        self.num_samples = num_samples
        self.verbose = verbose

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def run(self, prompt: str) -> Distiset:
        self._log(
            f"[1/4] Deriving parent/judge instructions via guide model "
            f"'{self.guide_model}'..."
        )
        guide = Guide(model_id=self.guide_model)
        instructions = guide.generate_instructions(prompt)
        self._log("[1/4] Done: derived parent and judge instructions.")

        parent_model = Model(model_id=self.parent_model)
        parent_model.set_instruction(instructions.parent_instruction)

        judge_model = Judge(model_id=self.judge_model)
        judge_model.set_instruction(instructions.judge_instruction)

        self._log(
            f"[2/4] Generating {self.num_samples} raw samples via parent model "
            f"'{self.parent_model}'..."
        )
        generator = SyntheticDatasetGenerator(
            model=parent_model, num_samples=self.num_samples
        )
        distiset = generator.generate(instructions.sample_instruction)
        self._log(
            f"[2/4] Done: generated {len(distiset['default']['train'])} raw samples."
        )

        self._log(f"[3/4] Refining dataset via judge model '{self.judge_model}'...")
        refactor = DatasetRefactor(parent_model=parent_model, judge_model=judge_model)
        refined_distiset = refactor.refine(distiset)
        self._log(
            f"[3/4] Done: refined dataset has "
            f"{len(refined_distiset['default']['train'])} samples."
        )

        return refined_distiset
