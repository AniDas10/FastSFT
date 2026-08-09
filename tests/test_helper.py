"""Tier-1 unit tests for fastsft.helper (Distiset shape, run-folder resolution,
training-metadata sidecar round-trip). Pure logic + local filesystem only."""

import re

import pytest
from datasets import Dataset

from fastsft.constants import (
    DEFAULT_OUTPUT_DIR,
    RAW_OUTPUT_SUBDIR,
    TRAINING_METADATA_FILENAME,
)
from fastsft.helper import (
    convert_to_distiset,
    current_timestamp,
    latest_run_path,
    load_data,
    load_training_metadata,
    matched_raw_run,
    save_training_metadata,
)


def test_current_timestamp_matches_format():
    # RUN_TIMESTAMP_FORMAT is "%Y%m%d_%H%M%S" -> YYYYmmdd_HHMMSS.
    assert re.fullmatch(r"\d{8}_\d{6}", current_timestamp())


def test_load_data_none_returns_none():
    assert load_data(None) is None


def test_convert_to_distiset_shape():
    train = Dataset.from_dict({"text": ["a", "b"]})
    distiset = convert_to_distiset(train)
    inner = distiset["default"]["train"]
    assert inner.column_names == ["text"]
    assert inner["text"] == ["a", "b"]


class TestLatestRunPath:
    def test_returns_most_recent_run(self, tmp_path):
        base = tmp_path / "raw"
        base.mkdir()
        for run_id in ("20260101_000000", "20260301_120000", "20260201_000000"):
            (base / run_id).mkdir()
        # Timestamped names sort lexicographically == chronologically.
        assert latest_run_path(str(base)) == str(base / "20260301_120000")

    def test_ignores_non_directories(self, tmp_path):
        base = tmp_path / "raw"
        base.mkdir()
        (base / "20260101_000000").mkdir()
        (base / "stray_file.json").write_text("{}")
        assert latest_run_path(str(base)) == str(base / "20260101_000000")

    def test_missing_base_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            latest_run_path(str(tmp_path / "does_not_exist"))

    def test_empty_base_raises(self, tmp_path):
        base = tmp_path / "empty"
        base.mkdir()
        with pytest.raises(FileNotFoundError):
            latest_run_path(str(base))


def _make_raw_run(tmp_path, run_id):
    """Creates datasets/raw/<run_id>/ under a tmp cwd; returns its path."""
    raw_run = tmp_path / DEFAULT_OUTPUT_DIR / RAW_OUTPUT_SUBDIR / run_id
    raw_run.mkdir(parents=True)
    return raw_run


class TestMatchedRawRun:
    def test_exact_run_id_match(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_raw_run(tmp_path, "20260101_000000")
        # adapter_dir shares the run id (modelsets/<run_id>).
        matched = matched_raw_run("modelsets/20260101_000000")
        assert matched is not None
        assert matched.endswith(f"{RAW_OUTPUT_SUBDIR}/20260101_000000")

    def test_trailing_slash_normalized(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_raw_run(tmp_path, "20260101_000000")
        assert matched_raw_run("modelsets/20260101_000000/") is not None

    def test_no_match_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _make_raw_run(tmp_path, "20260101_000000")
        assert matched_raw_run("modelsets/29990101_000000") is None


class TestTrainingMetadataRoundTrip:
    def test_save_then_load(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_id = "20260101_000000"
        raw_run = _make_raw_run(tmp_path, run_id)

        returned = save_training_metadata(
            str(raw_run),
            parent_model="meta-llama/llama-3.3-70b-instruct",
            parent_instruction="Answer like a pirate.",
            parent_max_tokens=1024,
            parent_temperature=0.9,
        )
        assert returned.endswith(TRAINING_METADATA_FILENAME)

        loaded = load_training_metadata(f"modelsets/{run_id}")
        assert loaded == {
            "parent_model": "meta-llama/llama-3.3-70b-instruct",
            "parent_instruction": "Answer like a pirate.",
            "parent_max_tokens": 1024,
            "parent_temperature": 0.9,
        }

    def test_load_none_when_no_matching_raw_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # No datasets/raw/<id> at all.
        assert load_training_metadata("modelsets/20260101_000000") is None

    def test_load_none_when_sidecar_absent(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        run_id = "20260101_000000"
        _make_raw_run(tmp_path, run_id)  # raw run exists, but no sidecar written
        assert load_training_metadata(f"modelsets/{run_id}") is None
