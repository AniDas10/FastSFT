"""CLI-level argument validation/loading/timestamp helpers shared by main.py."""

import argparse
from datetime import datetime
from typing import Optional

from distilabel.distiset import Distiset

from constants import RUN_TIMESTAMP_FORMAT
from stages.constants import STAGE_ORDER


def current_timestamp() -> str:
    """Current time formatted as RUN_TIMESTAMP_FORMAT."""
    return datetime.now().strftime(RUN_TIMESTAMP_FORMAT)


def load_data(path: Optional[str]) -> Optional[Distiset]:
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
