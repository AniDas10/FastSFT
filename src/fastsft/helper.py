"""Shared helpers: Distiset load/shape and run-folder timestamps.

CLI argument validation lives in validation_checks.py.
"""

import json
import os
from datetime import datetime

from datasets import Dataset, DatasetDict
from distilabel.distiset import Distiset

from fastsft.constants import (
    DEFAULT_OUTPUT_DIR,
    MODELSETS_OUTPUT_DIR,
    OUTPUT_DIR_ENV_VAR,
    RAW_OUTPUT_SUBDIR,
    RUN_TIMESTAMP_FORMAT,
    TRAINING_METADATA_FILENAME,
)


def _output_root() -> str:
    """Base directory for datasets/ and modelsets/ (empty = CWD)."""
    return os.environ.get(OUTPUT_DIR_ENV_VAR, "")


def datasets_dir() -> str:
    """Directory holding raw / formatted / eval-prompt run folders."""
    return os.path.join(_output_root(), DEFAULT_OUTPUT_DIR)


def modelsets_dir() -> str:
    """Directory holding trained adapter run folders."""
    return os.path.join(_output_root(), MODELSETS_OUTPUT_DIR)


def current_timestamp() -> str:
    """Current time formatted as RUN_TIMESTAMP_FORMAT (naive/local on purpose)."""
    return datetime.now().strftime(RUN_TIMESTAMP_FORMAT)  # noqa: DTZ005


def load_data(path: str | None) -> Distiset | None:
    """Loads a saved Distiset from `path`, or None if no path was given."""
    return Distiset.load_from_disk(path) if path else None


def save_distiset(dataset: Distiset, subdir: str, run_id: str) -> str:
    """Saves dataset to datasets_dir()/subdir/run_id; returns the path."""
    path = os.path.join(datasets_dir(), subdir, run_id)
    dataset.save_to_disk(path)
    return path


def convert_to_distiset(train: Dataset) -> Distiset:
    """Wrap a single Dataset into the Distiset shape stages expect."""
    return Distiset({"default": DatasetDict({"train": train})})


def latest_run_path(base_dir: str) -> str:
    """Return the most recent timestamped run folder under base_dir."""
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"No '{base_dir}' directory found. Run the pipeline first, or pass an explicit path."
        )
    runs = sorted(
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    )
    if not runs:
        raise FileNotFoundError(f"No runs found under '{base_dir}'.")
    return os.path.join(base_dir, runs[-1])


def matched_raw_run(adapter_dir: str) -> str | None:
    """The raw run matching adapter_dir's id (both share the same run_id), or None if missing."""
    run_id = os.path.basename(os.path.normpath(adapter_dir))
    path = os.path.join(datasets_dir(), RAW_OUTPUT_SUBDIR, run_id)
    return path if os.path.isdir(path) else None


def _training_metadata_path(run_dir: str) -> str:
    """Path for the training metadata sidecar (sibling file, not inside run_dir)."""
    return os.path.normpath(run_dir) + "." + TRAINING_METADATA_FILENAME


def save_training_metadata(run_dir: str, **fields) -> str:
    """Write training metadata (parent model/instruction) as JSON sidecar; return path."""
    path = _training_metadata_path(run_dir)
    with open(path, "w") as f:
        json.dump(fields, f, indent=2)
    return path


def load_training_metadata(adapter_dir: str) -> dict | None:
    """Load training metadata sidecar for an adapter run (by run_id match), or None if missing."""
    raw = matched_raw_run(adapter_dir)
    if raw is None:
        return None
    path = _training_metadata_path(raw)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
