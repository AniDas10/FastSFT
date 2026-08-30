"""Tier-1 unit tests for fastsft.validation_checks (CLI argument cross-checks).

Each validator calls parser.error(...) on a bad combination. A real argparse
parser turns that into SystemExit; we use a tiny fake that records the message
instead, so tests can assert on both the error path and its content."""

import argparse

import pytest

from fastsft.stages.constants import STAGE_ORDER
from fastsft.validation_checks import (
    validate_eval_flags,
    validate_hf_flags,
    validate_start_stage,
    validate_training_flags,
)

FIRST_STAGE = STAGE_ORDER[0]
LATER_STAGE = STAGE_ORDER[1]
TERMINAL_STAGE = STAGE_ORDER[-1]  # fine_tuner -- skips data_formatter


class _RecordingParser:
    """Stand-in for argparse.ArgumentParser whose .error records and raises,
    mirroring the real parser (which raises SystemExit)."""

    def __init__(self):
        self.message = None

    def error(self, message):
        self.message = message
        raise SystemExit(2)


def _ns(**attrs):
    return argparse.Namespace(**attrs)


# --- validate_start_stage ---------------------------------------------------

def test_start_stage_default_happy_path():
    parser = _RecordingParser()
    validate_start_stage(_ns(start_stage=FIRST_STAGE, prompt="a pirate bot", input_path=None), parser)
    assert parser.message is None


@pytest.mark.parametrize("prompt", [None, "", "   "])
def test_start_stage_default_requires_prompt(prompt):
    parser = _RecordingParser()
    with pytest.raises(SystemExit):
        validate_start_stage(_ns(start_stage=FIRST_STAGE, prompt=prompt, input_path=None), parser)
    assert "prompt is required" in parser.message


def test_start_stage_default_rejects_input_path():
    parser = _RecordingParser()
    with pytest.raises(SystemExit):
        validate_start_stage(
            _ns(start_stage=FIRST_STAGE, prompt="x", input_path="datasets/raw/run"), parser
        )
    assert "--input-path is only used" in parser.message


def test_later_stage_happy_path():
    parser = _RecordingParser()
    validate_start_stage(
        _ns(start_stage=LATER_STAGE, prompt=None, input_path="datasets/raw/run"), parser
    )
    assert parser.message is None


def test_later_stage_requires_input_path():
    parser = _RecordingParser()
    with pytest.raises(SystemExit):
        validate_start_stage(_ns(start_stage=LATER_STAGE, prompt=None, input_path=None), parser)
    assert "requires --input-path" in parser.message


def test_later_stage_rejects_prompt():
    parser = _RecordingParser()
    with pytest.raises(SystemExit):
        validate_start_stage(
            _ns(start_stage=LATER_STAGE, prompt="stray", input_path="datasets/raw/run"), parser
        )
    assert "prompt is ignored" in parser.message


# --- validate_training_flags ------------------------------------------------

def _training_ns(**overrides):
    base = dict(
        gpu_tier=None, local=False, modal_timeout=None, strategy=None, lora_rank=None,
        target_modules=None, lora_dropout=None, batch_size=None, grad_accumulation=None,
        learning_rate=None, max_epochs=None, eval_steps=None, early_stopping_patience=None,
        validation_split=None,
    )
    base.update(overrides)
    return _ns(**base)


def test_training_no_flags_is_ok():
    parser = _RecordingParser()
    validate_training_flags(_training_ns(), parser)
    assert parser.message is None


@pytest.mark.parametrize("dest", [{"gpu_tier": "A100-80GB"}, {"local": True}])
def test_training_destination_alone_is_ok(dest):
    parser = _RecordingParser()
    validate_training_flags(_training_ns(**dest), parser)
    assert parser.message is None


def test_training_gpu_tier_and_local_mutually_exclusive():
    parser = _RecordingParser()
    with pytest.raises(SystemExit):
        validate_training_flags(_training_ns(gpu_tier="A100-80GB", local=True), parser)
    assert "mutually exclusive" in parser.message


def test_training_modal_timeout_requires_gpu_tier():
    parser = _RecordingParser()
    with pytest.raises(SystemExit):
        validate_training_flags(_training_ns(modal_timeout=1800), parser)
    assert "--modal-timeout requires --gpu-tier" in parser.message


def test_training_modal_timeout_with_local_still_errors():
    # --local satisfies neither: --modal-timeout needs --gpu-tier specifically.
    parser = _RecordingParser()
    with pytest.raises(SystemExit):
        validate_training_flags(_training_ns(local=True, modal_timeout=1800), parser)
    assert "--modal-timeout requires --gpu-tier" in parser.message


def test_training_override_without_destination_errors():
    parser = _RecordingParser()
    with pytest.raises(SystemExit):
        validate_training_flags(_training_ns(lora_rank=32, batch_size=8), parser)
    assert "require --gpu-tier or --local" in parser.message
    assert "--lora-rank" in parser.message
    assert "--batch-size" in parser.message


def test_training_override_with_local_is_ok():
    parser = _RecordingParser()
    validate_training_flags(_training_ns(local=True, lora_rank=32), parser)
    assert parser.message is None


def test_training_zero_override_still_counts():
    # 0.0 is a meaningful value (not None), so it counts as "given".
    parser = _RecordingParser()
    with pytest.raises(SystemExit):
        validate_training_flags(_training_ns(lora_dropout=0.0), parser)
    assert "--lora-dropout" in parser.message


# --- validate_eval_flags ----------------------------------------------------

@pytest.mark.parametrize("n", [1, 10, 50])
def test_eval_positive_prompts_ok(n):
    parser = _RecordingParser()
    validate_eval_flags(_ns(num_eval_prompts=n), parser)
    assert parser.message is None


@pytest.mark.parametrize("n", [0, -1, -50])
def test_eval_non_positive_prompts_error(n):
    parser = _RecordingParser()
    with pytest.raises(SystemExit):
        validate_eval_flags(_ns(num_eval_prompts=n), parser)
    assert "must be positive" in parser.message


# --- validate_hf_flags -------------------------------------------------------

def _hf_ns(**overrides):
    base = dict(dataset_repo_id=None, model_repo_id=None, start_stage=FIRST_STAGE)
    base.update(overrides)
    return _ns(**base)


def test_hf_no_flags_is_ok(monkeypatch):
    monkeypatch.setattr("fastsft.validation_checks.has_token", lambda: False)
    parser = _RecordingParser()
    validate_hf_flags(_hf_ns(), parser)
    assert parser.message is None


def test_hf_dataset_repo_id_with_terminal_start_stage_errors(monkeypatch):
    # fine_tuner skips data_formatter entirely, so --dataset-repo-id would be a no-op.
    monkeypatch.setattr("fastsft.validation_checks.has_token", lambda: True)
    parser = _RecordingParser()
    with pytest.raises(SystemExit):
        validate_hf_flags(
            _hf_ns(dataset_repo_id="org/data", start_stage=TERMINAL_STAGE), parser
        )
    assert "has no effect" in parser.message


def test_hf_model_repo_id_with_terminal_start_stage_is_ok(monkeypatch):
    # fine_tuner always runs, so --model-repo-id is never a no-op regardless of start stage.
    monkeypatch.setattr("fastsft.validation_checks.has_token", lambda: True)
    parser = _RecordingParser()
    validate_hf_flags(_hf_ns(model_repo_id="org/model", start_stage=TERMINAL_STAGE), parser)
    assert parser.message is None


def test_hf_malformed_repo_id_errors(monkeypatch):
    monkeypatch.setattr("fastsft.validation_checks.has_token", lambda: True)
    parser = _RecordingParser()
    with pytest.raises(SystemExit):
        validate_hf_flags(_hf_ns(dataset_repo_id="too/many/slashes"), parser)
    assert "--dataset-repo-id" in parser.message


def test_hf_missing_token_errors(monkeypatch):
    monkeypatch.setattr("fastsft.validation_checks.has_token", lambda: False)
    parser = _RecordingParser()
    with pytest.raises(SystemExit):
        validate_hf_flags(_hf_ns(model_repo_id="org/model"), parser)
    assert "Hugging Face token" in parser.message


def test_hf_valid_repo_id_with_token_is_ok(monkeypatch):
    monkeypatch.setattr("fastsft.validation_checks.has_token", lambda: True)
    parser = _RecordingParser()
    validate_hf_flags(
        _hf_ns(dataset_repo_id="org/data", model_repo_id="org/model"), parser
    )
    assert parser.message is None
