"""DataGenerator mini-pipeline: guide -> generate instructions -> answer -> refine."""

from distilabel.distiset import Distiset

from fastsft.constants import (
    DEFAULT_GUIDE_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PARENT_MODEL,
    RAW_OUTPUT_SUBDIR,
)
from fastsft.data.constants import (
    BREADTH_EXPONENT,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_PARENT_TEMPERATURE,
    GUIDE_TOKENS_PER_SEED,
)
from fastsft.data.prompt_generator import PromptGenerator, seed_count
from fastsft.data.refiner import DataRefiner
from fastsft.data.response_generator import ResponseGenerator
from fastsft.helper import convert_to_distiset, save_distiset, save_training_metadata
from fastsft.model.base import Model
from fastsft.model.constants import DEFAULT_MAX_TOKENS, DEFAULT_SCORE_THRESHOLD
from fastsft.model.guide import Guide, GuideInstructions
from fastsft.model.judge import Judge
from fastsft.stages.base import Stage
from fastsft.stages.constants import DATA_GENERATOR


class DataGenerator(Stage):
    """Prompt in, quality-filtered Distiset out with a `messages` column.

    Runs guide -> generate instructions -> answer -> refine, then converts to
    `messages`.
    """

    name = DATA_GENERATOR

    def __init__(
        self,
        guide_model: str = DEFAULT_GUIDE_MODEL,
        parent_model: str = DEFAULT_PARENT_MODEL,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        breadth_exponent: float = BREADTH_EXPONENT,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        parent_temperature: float = DEFAULT_PARENT_TEMPERATURE,
        parent_max_tokens: int = DEFAULT_MAX_TOKENS,
        verbose: bool = True,
    ):
        super().__init__(verbose=verbose)
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        self._guide_model = guide_model
        self._parent_model = parent_model
        self._judge_model = judge_model
        self._num_samples = num_samples
        self._breadth_exponent = breadth_exponent
        self._score_threshold = score_threshold
        self._parent_temperature = parent_temperature
        self._parent_max_tokens = parent_max_tokens
        # The Guide-derived style prompt, captured in _run so save_output can
        # persist it as this run's teacher provenance (for evaluation to reuse).
        self._parent_instruction: str | None = None

    def _validate_input(self, prompt: str) -> None:
        if not prompt or not prompt.strip():
            raise ValueError("DataGenerator.run() requires a non-empty prompt.")

    def _run(self, prompt: str) -> Distiset:
        instructions = self._setup(prompt)
        self._parent_instruction = instructions.parent_instruction
        self._log(
            f"[1/4] Derived instructions and "
            f"{len(instructions.sample_instructions)} seed topics via guide "
            f"model '{self._guide_model}'."
        )

        parent_model = Model(
            model_id=self._parent_model,
            temperature=self._parent_temperature,
            max_tokens=self._parent_max_tokens,
        )
        parent_model.set_instruction(instructions.parent_instruction)

        judge_model = Judge(model_id=self._judge_model)
        judge_model.set_instruction(instructions.judge_instruction)

        self._log(
            f"[2/4] Generating {self._num_samples} instructions across "
            f"{len(instructions.sample_instructions)} topics via parent model "
            f"'{self._parent_model}'..."
        )
        prompt_generator = PromptGenerator(
            model=parent_model, num_samples=self._num_samples
        )
        generated_instructions = prompt_generator.generate(
            instructions.sample_instructions
        )
        self._log(
            f"[2/4] Done: generated {len(generated_instructions)} instructions."
        )

        self._log(
            f"[3/4] Generating answers via parent model '{self._parent_model}'..."
        )
        response_generator = ResponseGenerator(model=parent_model)
        distiset = response_generator.generate(generated_instructions)
        self._log(
            f"[3/4] Done: generated {len(distiset['default']['train'])} raw samples."
        )

        self._log(f"[4/4] Refining dataset via judge model '{self._judge_model}'...")
        refiner = DataRefiner(parent_model=parent_model, judge_model=judge_model)
        refined_distiset = refiner.refine(distiset, threshold=self._score_threshold)
        self._log(
            f"[4/4] Done: refined dataset has "
            f"{len(refined_distiset['default']['train'])} samples."
        )

        return self._to_messages(refined_distiset)

    def save_output(self, output: Distiset, run_id: str) -> str:
        path = save_distiset(output, RAW_OUTPUT_SUBDIR, run_id)
        # Record this run's teacher -- identity, style prompt, and generation
        # recipe -- alongside the dataset, so evaluation reconstructs the true
        # parent reference (answering like the actual teacher) instead of guessing.
        save_training_metadata(
            path,
            parent_model=self._parent_model,
            parent_instruction=self._parent_instruction or "",
            parent_max_tokens=self._parent_max_tokens,
            parent_temperature=self._parent_temperature,
        )
        return path

    def _setup(self, prompt: str) -> GuideInstructions:
        """Builds the guide (output budget scaled to the seed count) and
        derives its instructions."""
        num_seeds = seed_count(self._num_samples, breadth_exponent=self._breadth_exponent)
        guide = Guide(
            model_id=self._guide_model,
            max_tokens=DEFAULT_MAX_TOKENS + num_seeds * GUIDE_TOKENS_PER_SEED,
        )
        return guide.generate_instructions(prompt, num_seeds=num_seeds)

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
        return convert_to_distiset(train)
