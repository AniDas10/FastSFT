"""CLI-level argument validation/loading/timestamp helpers shared by main.py."""

import argparse
from datetime import datetime

from distilabel.distiset import Distiset

from constants import RUN_TIMESTAMP_FORMAT
from stages.constants import STAGE_ORDER


def current_timestamp() -> str:
    """Current time formatted as RUN_TIMESTAMP_FORMAT."""
    # Local (naive) time on purpose: these timestamps name run folders for a
    # human to read, not to compare across timezones.
    return datetime.now().strftime(RUN_TIMESTAMP_FORMAT)  # noqa: DTZ005


def load_data(path: str | None) -> Distiset | None:
    """Loads a saved Distiset from `path`, or None if no path was given."""
    return Distiset.load_from_disk(path) if path else None


def validate_start_stage(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """--start-stage=STAGE_ORDER[0] (default) requires a prompt and no
    --input-path; any other --start-stage requires --input-path and no prompt.
    """
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
    """--gpu-tier (dispatch to Modal) and --local (train on this machine) are
    mutually exclusive training destinations. The adapter/loop override flags
    are only meaningful alongside one of them -- using one without either is
    likely a mistake, not a silent no-op. --modal-timeout is Modal-specific,
    so it requires --gpu-tier specifically (not satisfied by --local alone).
    """
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
