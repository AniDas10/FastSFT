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
    RAW_OUTPUT_SUBDIR,
    RUN_TIMESTAMP_FORMAT,
    TRAINING_METADATA_FILENAME,
)


def current_timestamp() -> str:
    """Current time formatted as RUN_TIMESTAMP_FORMAT."""
    # Local (naive) time on purpose: these timestamps name run folders for a
    # human to read, not to compare across timezones.
    return datetime.now().strftime(RUN_TIMESTAMP_FORMAT)  # noqa: DTZ005


def load_data(path: str | None) -> Distiset | None:
    """Loads a saved Distiset from `path`, or None if no path was given."""
    return Distiset.load_from_disk(path) if path else None


def save_distiset(dataset: Distiset, subdir: str, run_id: str) -> str:
    """Saves `dataset` under DEFAULT_OUTPUT_DIR/subdir/run_id; returns the path.
    Counterpart to load_data (via latest_run_path); used by the pipeline stages
    and the evaluation module to persist their run artifacts."""
    path = os.path.join(DEFAULT_OUTPUT_DIR, subdir, run_id)
    dataset.save_to_disk(path)
    return path


def convert_to_distiset(train: Dataset) -> Distiset:
    """Wraps a single `train` split into the Distiset({"default": {"train": ...}})
    shape the stages pass between one another."""
    return Distiset({"default": DatasetDict({"train": train})})


def latest_run_path(base_dir: str) -> str:
    """Returns the most recent timestamped run folder under `base_dir`
    (shared by the dataset viewer and the training-stats viewer)."""
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
    """The raw dataset run whose id matches `adapter_dir`'s (FineTuner and
    DataGenerator share the pipeline run id), or None if it isn't on disk --
    e.g. a bring-your-own dataset that skipped DataGenerator."""
    run_id = os.path.basename(os.path.normpath(adapter_dir))
    path = os.path.join(DEFAULT_OUTPUT_DIR, RAW_OUTPUT_SUBDIR, run_id)
    return path if os.path.isdir(path) else None


def save_training_metadata(run_dir: str, **fields) -> str:
    """Writes `fields` (training provenance, e.g. parent model/instruction) as a
    JSON sidecar in `run_dir`; returns the path."""
    path = os.path.join(run_dir, TRAINING_METADATA_FILENAME)
    with open(path, "w") as f:
        json.dump(fields, f, indent=2)
    return path


def load_training_metadata(adapter_dir: str) -> dict | None:
    """Loads the training provenance persisted for `adapter_dir`, matched by run
    id (exact only -- a wrong teacher is worse than none). None when there's no
    matching raw run or no sidecar (an older, or bring-your-own, run)."""
    raw = matched_raw_run(adapter_dir)
    if raw is None:
        return None
    path = os.path.join(raw, TRAINING_METADATA_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)
