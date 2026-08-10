"""CLI entry point for the evaluation module:

    uv run python -m fastsft.eval.run [modelsets/<run_id>]

Resolves the eval prompt set (reuse the latest, or generate + persist a fresh
one), runs the Evaluator against the adapter, and writes eval_results.json next
to the adapter. View the result with `python -m fastsft.eval.results_viewer`.
"""

import fastsft.warnings_filter  # noqa: F401

import argparse
import os
from argparse import Namespace

from fastsft.constants import (
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PARENT_MODEL,
    OUTPUT_DIR_ENV_VAR,
)
from fastsft.data.constants import DEFAULT_PARENT_TEMPERATURE
from fastsft.eval.config import EvalConfig
from fastsft.eval.constants import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_EVAL_PROMPTS,
)
from fastsft.eval.evaluator import Evaluator
from fastsft.eval.prompt_set import EvalPromptSet
from fastsft.eval.results import save_results
from fastsft.helper import (
    current_timestamp,
    latest_run_path,
    load_training_metadata,
    modelsets_dir,
)
from fastsft.model.base import Model
from fastsft.model.constants import DEFAULT_MAX_TOKENS
from fastsft.progress import log
from fastsft.validation_checks import validate_eval_flags


def _resolve_prompt_set(args: Namespace, config: EvalConfig) -> EvalPromptSet:
    """Loads an existing eval prompt set, or generates + persists a fresh one.

    Priority: an explicit --eval-prompts-path; else --regenerate-prompts forces
    a new set; else reuse the latest saved set, generating one only if none
    exists yet (so scores stay comparable across adapters by default).
    """
    if args.eval_prompts_path:
        prompt_set = EvalPromptSet.load(args.eval_prompts_path)
        log(f"Loaded {len(prompt_set)} eval prompts from '{args.eval_prompts_path}'.")
        return prompt_set

    if not args.regenerate_prompts:
        try:
            prompt_set = EvalPromptSet.load()
            log(f"Reusing latest saved eval prompt set ({len(prompt_set)} prompts).")
            return prompt_set
        except FileNotFoundError:
            log("No saved eval prompt set found -- generating a fresh one.")

    model = Model(model_id=config.parent_model)
    prompt_set = EvalPromptSet.generate(
        config.adapter_dir, model=model, num_prompts=config.num_eval_prompts
    )
    path = prompt_set.save(current_timestamp())
    log(f"Generated and saved {len(prompt_set)} eval prompts to '{path}'.")
    return prompt_set


def _resolve_parent(args: Namespace, adapter_dir: str) -> tuple[str, str, int, float]:
    """Resolves the parent teacher for the eval reference from training metadata:
    identity + style prompt (explicit flags win) and the generation recipe
    (max tokens + temperature, inferred so the reference answers like the actual
    teacher). Falls back to defaults -- with a warning -- when there's no
    metadata, since a guessed reference may not match the real teacher."""
    metadata = load_training_metadata(adapter_dir) or {}

    if args.parent_model is not None:
        parent_model = args.parent_model
    elif metadata.get("parent_model"):
        parent_model = metadata["parent_model"]
    else:
        parent_model = DEFAULT_PARENT_MODEL

    if args.parent_instruction is not None:
        parent_instruction = args.parent_instruction
    elif "parent_instruction" in metadata:
        parent_instruction = metadata["parent_instruction"]
    else:
        parent_instruction = ""

    # Generation recipe: inferred-only (no flags) -- it's a property of the
    # teacher, not an eval choice. Defaults to the pipeline's training defaults.
    parent_max_tokens = metadata.get("parent_max_tokens", DEFAULT_MAX_TOKENS)
    parent_temperature = metadata.get("parent_temperature", DEFAULT_PARENT_TEMPERATURE)

    styled = "styled" if parent_instruction else "no style prompt"
    source = "training metadata" if (metadata and args.parent_model is None) else "flags/defaults"
    log(f"Parent reference: {parent_model} ({styled}) [from {source}].")
    if not metadata and (args.parent_model is None or args.parent_instruction is None):
        log(
            "  No training metadata for this adapter (e.g. a bring-your-own "
            "dataset) -- pass --parent-model / --parent-instruction if the real "
            "teacher differs, or parent-relative metrics won't reflect it."
        )
    return parent_model, parent_instruction, parent_max_tokens, parent_temperature


def _input_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Registers CLI arguments on `parser` and returns the parsed args."""
    parser.add_argument(
        "adapter_dir",
        nargs="?",
        default=None,
        help="Adapter directory to evaluate (default: latest run under modelsets/).",
    )
    parser.add_argument(
        "--num-eval-prompts",
        type=int,
        default=DEFAULT_NUM_EVAL_PROMPTS,
        help="How many eval prompts to generate when creating a fresh set.",
    )
    parser.add_argument(
        "--regenerate-prompts",
        action="store_true",
        help="Force a fresh eval prompt set instead of reusing the latest saved one.",
    )
    parser.add_argument(
        "--eval-prompts-path",
        default=None,
        help="Load the eval prompt set from this specific path instead of the latest.",
    )
    parser.add_argument(
        "--parent-model",
        default=None,
        help="Parent teacher model id (default: inferred from the run's training "
        "metadata, else the pipeline default).",
    )
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--parent-instruction",
        default=None,
        help="Parent teacher's style system prompt (default: inferred from the "
        "run's training metadata, else none).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--no-swap",
        action="store_true",
        help="Judge each pair in one A/B order only (skip position-bias debiasing).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base directory holding datasets/ and modelsets/ (default: the "
        "current directory). Point this at the same location used for training "
        f"so the adapter is found. Sets {OUTPUT_DIR_ENV_VAR} for this run.",
    )
    return parser.parse_args()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned child adapter against its untuned base "
        "and its parent teacher (LLM-judge win rates + parent similarity)."
    )
    args = _input_args(parser)
    validate_eval_flags(args, parser)
    if args.output_dir:
        os.environ[OUTPUT_DIR_ENV_VAR] = args.output_dir

    adapter_dir = args.adapter_dir or latest_run_path(modelsets_dir())
    log(f"Evaluating adapter '{adapter_dir}'.")

    parent_model, parent_instruction, parent_max_tokens, parent_temperature = (
        _resolve_parent(args, adapter_dir)
    )
    config = EvalConfig(
        adapter_dir=adapter_dir,
        parent_model=parent_model,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        num_eval_prompts=args.num_eval_prompts,
        max_new_tokens=args.max_new_tokens,
        swap_positions=not args.no_swap,
        parent_instruction=parent_instruction,
        parent_max_tokens=parent_max_tokens,
        parent_temperature=parent_temperature,
    )

    prompt_set = _resolve_prompt_set(args, config)
    results = Evaluator(config).run(prompt_set.prompts)
    path = save_results(results, adapter_dir)

    tvu = results["comparisons"]["tuned_vs_untuned"]["win_rate"]
    log(
        f"\nDone. Tuned vs untuned win rate: {tvu:.0%}. Results saved to '{path}'.\n"
        "View with: uv run python -m fastsft.eval.results_viewer"
    )


if __name__ == "__main__":
    main()
