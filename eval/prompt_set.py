"""The held-out eval prompt set: generate a fresh one, persist it, load it back.

Generated once and reused across every adapter you evaluate, so scores stay
comparable over time. Fresh prompts are grown from the training run's own user
questions (used as topic seeds, so the eval set stays in-domain) and then
deduped against those questions to prevent train/eval leakage.
"""

import os
import random

from datasets import Dataset

from constants import DEFAULT_OUTPUT_DIR, EVAL_PROMPTS_SUBDIR, RAW_OUTPUT_SUBDIR
from data.constants import BREADTH_EXPONENT
from data.prompt_generator import PromptGenerator, seed_count
from helper import (
    convert_to_distiset,
    latest_run_path,
    load_data,
    matched_raw_run,
    save_distiset,
)
from model.base import Model

# Column name the eval prompts are persisted under.
PROMPT_COLUMN = "prompt"


def _normalize(prompt: str) -> str:
    """Whitespace/case-insensitive key for exact-duplicate detection."""
    return " ".join(prompt.lower().split())


def load_training_prompts(adapter_dir: str) -> list[str]:
    """The user instructions from the training run behind `adapter_dir`.

    Matched by run id when that raw dataset is still on disk (FineTuner and
    DataGenerator share the pipeline's run id), else falls back to the latest
    raw run so eval still works against an adapter trained elsewhere.
    """
    raw_root = os.path.join(DEFAULT_OUTPUT_DIR, RAW_OUTPUT_SUBDIR)
    path = matched_raw_run(adapter_dir) or latest_run_path(raw_root)

    distiset = load_data(path)
    prompts: list[str] = []
    for row in distiset["default"]["train"]:
        for message in row.get("messages", []):
            if message.get("role") == "user":
                prompts.append(message["content"])
                break
    return prompts


class EvalPromptSet:
    """A held-out set of eval prompts."""

    def __init__(self, prompts: list[str]):
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    @classmethod
    def generate(
        cls, adapter_dir: str, model: Model, num_prompts: int
    ) -> "EvalPromptSet":
        """Grows a fresh prompt set from the training questions via `model`,
        then drops any that collide with a training question."""
        training = load_training_prompts(adapter_dir)
        if not training:
            raise ValueError(
                f"No training prompts found for '{adapter_dir}'; cannot seed an "
                "in-domain eval set. Was the raw dataset kept on disk?"
            )
        seeds = cls._select_seeds(training, num_prompts)
        generated = PromptGenerator(model=model, num_samples=num_prompts).generate(
            seeds
        )
        return cls(cls._dedup(generated, training))

    @staticmethod
    def _select_seeds(training: list[str], num_prompts: int) -> list[str]:
        """Compresses the training questions to a breadth-appropriate seed set
        (same seed_count/BREADTH_EXPONENT split as DataGenerator), so each seed
        drives several expansions rather than one near-paraphrase of itself that
        exact-text _dedup couldn't catch. Fixed seed for reproducibility."""
        num_seeds = min(len(training), seed_count(num_prompts, breadth_exponent=BREADTH_EXPONENT))
        return random.Random(42).sample(training, num_seeds)

    @staticmethod
    def _dedup(generated: list[str], training: list[str]) -> list[str]:
        """Removes prompts matching a training question or an earlier pick."""
        seen = {_normalize(p) for p in training}
        unique: list[str] = []
        for prompt in generated:
            key = _normalize(prompt)
            if key in seen:
                continue
            seen.add(key)
            unique.append(prompt)
        return unique

    def save(self, run_id: str) -> str:
        """Persists the set under datasets/eval_prompts/<run_id>; returns the path."""
        dataset = Dataset.from_dict({PROMPT_COLUMN: self.prompts})
        return save_distiset(convert_to_distiset(dataset), EVAL_PROMPTS_SUBDIR, run_id)

    @classmethod
    def load(cls, path: str | None = None) -> "EvalPromptSet":
        """Loads a saved set (default: the latest under datasets/eval_prompts/)."""
        if path is None:
            path = latest_run_path(os.path.join(DEFAULT_OUTPUT_DIR, EVAL_PROMPTS_SUBDIR))
        distiset = load_data(path)
        return cls(list(distiset["default"]["train"][PROMPT_COLUMN]))
