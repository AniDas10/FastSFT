"""CLI-level argument validation for the entry points (main.py, eval/run.py).

Each check takes the parsed args and the parser, and calls parser.error() on a
bad combination -- kept out of helper.py so that stays data/IO helpers only.
"""

import argparse

from fastsft.hf_helper import has_token, repo_id_error
from fastsft.stages.constants import DATA_FORMATTER, STAGE_ORDER


def validate_start_stage(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate prompt/input-path rules for each --start-stage."""
    first_stage = STAGE_ORDER[0]
    if args.start_stage == first_stage:
        if not (args.prompt and args.prompt.strip()):
            parser.error(f"prompt is required when --start-stage={first_stage} (the default).")
        if args.input_path:
            parser.error(f"--input-path is only used when --start-stage is not {first_stage}.")
    else:
        if not args.input_path:
            parser.error(f"--start-stage={args.start_stage} requires --input-path.")
        if args.prompt:
            parser.error(
                f"prompt is ignored when --start-stage={args.start_stage} -- "
                "pass --input-path instead."
            )


def validate_training_flags(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate --gpu-tier/--local mutual exclusivity and override-flag dependencies."""
    if args.gpu_tier is not None and args.local:
        parser.error("--gpu-tier and --local are mutually exclusive.")

    if args.modal_timeout is not None and args.gpu_tier is None:
        parser.error("--modal-timeout requires --gpu-tier to be set.")

    if args.gpu_tier is not None or args.local:
        return
    other_flags = {
        "--strategy": args.strategy,
        "--lora-rank": args.lora_rank,
        "--target-modules": args.target_modules,
        "--lora-dropout": args.lora_dropout,
        "--batch-size": args.batch_size,
        "--grad-accumulation": args.grad_accumulation,
        "--learning-rate": args.learning_rate,
        "--max-epochs": args.max_epochs,
        "--eval-steps": args.eval_steps,
        "--early-stopping-patience": args.early_stopping_patience,
        "--validation-split": args.validation_split,
        "--modal-timeout": args.modal_timeout,
    }
    given = [name for name, value in other_flags.items() if value is not None]
    if given:
        parser.error(f"{', '.join(given)} require --gpu-tier or --local to be set.")


def validate_eval_flags(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate that --num-eval-prompts is positive."""
    if args.num_eval_prompts <= 0:
        parser.error(f"--num-eval-prompts must be positive, got {args.num_eval_prompts}.")


def validate_hf_flags(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Validate --dataset-repo-id/--model-repo-id before any paid stage runs:
    the pushing stage actually runs, repo ids are well-formed, and a Hub token
    is available -- so a typo or missing login fails fast instead of surfacing
    only after a billed Modal GPU run completes."""
    if args.dataset_repo_id and STAGE_ORDER.index(args.start_stage) > STAGE_ORDER.index(
        DATA_FORMATTER
    ):
        parser.error(
            f"--dataset-repo-id has no effect when --start-stage={args.start_stage} -- "
            f"only {DATA_FORMATTER} pushes the dataset, and it's skipped at that start "
            "stage. Drop --dataset-repo-id, or start from an earlier stage."
        )

    for flag, repo_id in (
        ("--dataset-repo-id", args.dataset_repo_id),
        ("--model-repo-id", args.model_repo_id),
    ):
        if repo_id is None:
            continue
        error = repo_id_error(repo_id)
        if error:
            parser.error(f"{flag} '{repo_id}' is not a valid Hugging Face repo id: {error}")

    if (args.dataset_repo_id or args.model_repo_id) and not has_token():
        parser.error(
            "--dataset-repo-id/--model-repo-id require a Hugging Face token -- "
            "set HF_TOKEN (in .env or the environment), or run `huggingface-cli login`."
        )
