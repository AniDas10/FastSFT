"""DataGenerator mini-pipeline: guide -> raw generation -> quality refinement."""

from datasets import DatasetDict
from distilabel.distiset import Distiset

from constants import DEFAULT_GUIDE_MODEL, DEFAULT_JUDGE_MODEL, DEFAULT_PARENT_MODEL
from data.generator import SyntheticDataGenerator
from data.refiner import DataRefiner
from model.base import Model
from model.guide import Guide
from model.judge import Judge
from stages.base import Stage
from stages.constants import DATA_GENERATOR


class DataGenerator(Stage):
    """Prompt in, quality-filtered Distiset out with a `messages` column.

    Runs guide -> generate -> refine, then converts to `messages`.
    """

    name = DATA_GENERATOR

    def __init__(
        self,
        guide_model: str = DEFAULT_GUIDE_MODEL,
        parent_model: str = DEFAULT_PARENT_MODEL,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = 100,
        verbose: bool = True,
    ):
        super().__init__(verbose=verbose)
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        self._guide_model = guide_model
        self._parent_model = parent_model
        self._judge_model = judge_model
        self._num_samples = num_samples

    def _validate_input(self, prompt: str) -> None:
        if not prompt or not prompt.strip():
            raise ValueError("DataGenerator.run() requires a non-empty prompt.")

    def _run(self, prompt: str) -> Distiset:
        self._log(
            f"[1/3] Deriving parent/judge instructions via guide model "
            f"'{self._guide_model}'..."
        )
        guide = Guide(model_id=self._guide_model)
        instructions = guide.generate_instructions(prompt)
        self._log("[1/3] Done: derived parent and judge instructions.")

        parent_model = Model(model_id=self._parent_model)
        parent_model.set_instruction(instructions.parent_instruction)

        judge_model = Judge(model_id=self._judge_model)
        judge_model.set_instruction(instructions.judge_instruction)

        self._log(
            f"[2/3] Generating {self._num_samples} raw samples via parent model "
            f"'{self._parent_model}'..."
        )
        generator = SyntheticDataGenerator(
            model=parent_model, num_samples=self._num_samples
        )
        distiset = generator.generate(instructions.sample_instruction)
        self._log(
            f"[2/3] Done: generated {len(distiset['default']['train'])} raw samples."
        )

        self._log(f"[3/3] Refining dataset via judge model '{self._judge_model}'...")
        refiner = DataRefiner(parent_model=parent_model, judge_model=judge_model)
        refined_distiset = refiner.refine(distiset)
        self._log(
            f"[3/3] Done: refined dataset has "
            f"{len(refined_distiset['default']['train'])} samples."
        )

        return self._to_messages(refined_distiset)

    def _to_messages(self, distiset: Distiset) -> Distiset:
        """Converts (instruction, generation) rows to the `messages` schema."""
        train = distiset["default"]["train"]

        def convert(row):
            return {
                "messages": [
                    {"role": "user", "content": row["instruction"]},
                    {"role": "assistant", "content": row["generation"]},
                ]
            }

        train = train.map(convert, remove_columns=["instruction", "generation"])
        return Distiset({"default": DatasetDict({"train": train})})
