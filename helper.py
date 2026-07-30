"""CLI-level argument validation/loading/saving helpers shared by main.py."""

import argparse
import os
from datetime import datetime
from typing import Optional

from distilabel.distiset import Distiset

from constants import DEFAULT_OUTPUT_DIR, RUN_TIMESTAMP_FORMAT


def current_timestamp() -> str:
    """Returns the current time formatted as RUN_TIMESTAMP_FORMAT -- used to
    group a run's raw/formatted saved datasets under matching subfolders."""
    return datetime.now().strftime(RUN_TIMESTAMP_FORMAT)


def load_data(path: Optional[str]) -> Optional[Distiset]:
    """Loads a saved Distiset from `path`, or returns None if no path was given."""
    return Distiset.load_from_disk(path) if path else None


def save_data(dataset: Optional[Distiset], subdir: str, label: str) -> None:
    """Saves `dataset` to DEFAULT_OUTPUT_DIR/subdir/<current timestamp>, or
    does nothing if `dataset` is None (e.g. a stage that didn't run)."""
    if dataset is None:
        return
    output_dir = os.path.join(DEFAULT_OUTPUT_DIR, subdir, current_timestamp())
    dataset.save_to_disk(output_dir)
    print(f"Saved {label} dataset to '{output_dir}'")


def validate_skip_flags(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Ensures the prompt/skip-flag/dataset-path CLI args are a coherent
    combination before any pipeline object is constructed.

    Calls parser.error() (prints usage + exits) rather than raising, since
    this checks CLI arg presence -- DistillationPipeline._validate_inputs
    separately checks the resulting prompt/raw_dataset/formatted_dataset
    values once they're loaded into Python objects.
    """
    if (
        not args.skip_generation
        and not args.skip_formatting
        and not (args.prompt and args.prompt.strip())
    ):
        parser.error("prompt is required unless --skip-generation or --skip-formatting is set.")
    if args.skip_generation and not args.raw_dataset_path:
        parser.error("--skip-generation requires --raw-dataset-path.")
    if args.skip_formatting and not args.formatted_dataset_path:
        parser.error("--skip-formatting requires --formatted-dataset-path.")
