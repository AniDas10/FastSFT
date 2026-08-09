"""Tier-1 unit tests for fastsft.training.stats -- telemetry loading, series
extraction, and the RunInterpreter diagnostic checks. Pure logic + tmp files.
"""

import json
import math

import pytest

from fastsft.training.stats import (
    RunInterpreter,
    _checkpoint_number,
    load_stats,
    series,
    summarize,
)


def _stats(eval_points, *, loss=None, epoch=None, max_epochs=None, masking=None):
    """Builds a stats dict whose log_history carries the given (step, eval_loss)
    points (and optional (step, loss) train points)."""
    log = [{"step": s, "eval_loss": y} for s, y in eval_points]
    if loss:
        log += [{"step": s, "loss": y} for s, y in loss]
    stats = {"log_history": log}
    if epoch is not None:
        stats["epoch"] = epoch
    if max_epochs is not None:
        stats["num_train_epochs"] = max_epochs
    if masking is not None:
        stats["loss_masking"] = masking
    return stats


def _find(interpreter, method_name):
    return getattr(interpreter, method_name)()


# --- series / _checkpoint_number --------------------------------------------

def test_series_extracts_only_entries_with_step_and_key():
    log = [
        {"step": 1, "eval_loss": 0.9},
        {"step": 2, "loss": 0.5},          # missing eval_loss
        {"eval_loss": 0.4},                # missing step
        {"step": 3, "eval_loss": 0.3},
    ]
    assert series(log, "eval_loss") == [(1, 0.9), (3, 0.3)]


@pytest.mark.parametrize(
    "path,expected",
    [
        ("out/checkpoint-500", 500),
        ("checkpoint-0", 0),
        ("checkpoint-final", -1),
        ("checkpoint-", -1),
    ],
)
def test_checkpoint_number(path, expected):
    assert _checkpoint_number(path) == expected


# --- _check_epoch_density ----------------------------------------------------

def test_epoch_density_no_evals_warns():
    finding = _find(RunInterpreter(_stats([])), "_check_epoch_density")
    assert finding.status == "warn"
    assert "No usable validation-loss" in finding.message


def test_epoch_density_single_eval_warns():
    finding = _find(RunInterpreter(_stats([(1, 0.5)])), "_check_epoch_density")
    assert finding.status == "warn"
    assert "Only one evaluation" in finding.message


def test_epoch_density_enough_evals_returns_none():
    assert _find(RunInterpreter(_stats([(1, 1.0), (2, 0.5)])), "_check_epoch_density") is None


# --- _check_improvement ------------------------------------------------------

def test_improvement_flags_meaningful_drop():
    finding = _find(RunInterpreter(_stats([(1, 1.0), (2, 0.5)])), "_check_improvement")
    assert finding.status == "good"
    assert "learned" in finding.message


def test_improvement_none_when_barely_moved():
    assert _find(RunInterpreter(_stats([(1, 1.0), (2, 0.999)])), "_check_improvement") is None


# --- _check_overfitting ------------------------------------------------------

def test_overfitting_flags_bottom_then_rise():
    finding = _find(
        RunInterpreter(_stats([(1, 1.0), (2, 0.5), (3, 0.7)])), "_check_overfitting"
    )
    assert finding.status == "warn"
    assert "overfitting" in finding.message


def test_overfitting_none_when_still_descending():
    assert _find(
        RunInterpreter(_stats([(1, 1.0), (2, 0.6), (3, 0.5)])), "_check_overfitting"
    ) is None


def test_overfitting_none_with_too_few_points():
    assert _find(RunInterpreter(_stats([(1, 1.0), (2, 0.5)])), "_check_overfitting") is None


# --- _check_underfitting -----------------------------------------------------

def test_underfitting_flags_barely_moved():
    finding = _find(RunInterpreter(_stats([(1, 1.0), (2, 0.999)])), "_check_underfitting")
    assert finding.status == "warn"
    assert "underfit" in finding.message


def test_underfitting_flags_still_falling_at_ceiling():
    # Strictly descending to the end, epoch == max_epochs (not early stopped).
    stats = _stats([(1, 1.0), (2, 0.8), (3, 0.7)], epoch=3.0, max_epochs=3)
    finding = _find(RunInterpreter(stats), "_check_underfitting")
    assert finding.status == "warn"
    assert "still falling" in finding.message


def test_underfitting_none_when_improved_and_early_stopped():
    stats = _stats([(1, 1.0), (2, 0.6), (3, 0.5)], epoch=3.0, max_epochs=10)
    assert _find(RunInterpreter(stats), "_check_underfitting") is None


# --- _check_early_stopping ---------------------------------------------------

def test_early_stopping_flags_stop_before_ceiling():
    stats = _stats([(1, 1.0), (2, 0.5)], epoch=2.0, max_epochs=5)
    finding = _find(RunInterpreter(stats), "_check_early_stopping")
    assert finding.status == "good"
    assert "Stopped early" in finding.message


def test_early_stopping_none_when_ran_full_epochs():
    stats = _stats([(1, 1.0), (2, 0.5)], epoch=5.0, max_epochs=5)
    assert _find(RunInterpreter(stats), "_check_early_stopping") is None


# --- _loss_scope_caveat ------------------------------------------------------

@pytest.mark.parametrize("masking", ["completion", "assistant"])
def test_loss_scope_caveat_masked_wording(masking):
    finding = RunInterpreter(_stats([(1, 0.5)], masking=masking))._loss_scope_caveat()
    assert finding.status == "info"
    assert "answer tokens only" in finding.message


@pytest.mark.parametrize("masking", ["full", None])
def test_loss_scope_caveat_full_wording(masking):
    finding = RunInterpreter(_stats([(1, 0.5)], masking=masking))._loss_scope_caveat()
    assert "whole formatted example" in finding.message


# --- summarize ---------------------------------------------------------------

def test_summarize_drops_nonfinite_and_is_json_safe():
    stats = _stats(
        [(1, 1.0), (2, float("nan")), (3, float("inf")), (4, 0.5)],
        loss=[(1, 2.0), (2, float("inf"))],
        epoch=4.0,
        max_epochs=4,
    )
    result = summarize(stats, "modelsets/run")

    assert result["series"]["eval_loss"] == [[1, 1.0], [4, 0.5]]
    assert result["series"]["train_loss"] == [[1, 2.0]]
    assert result["best_eval_loss"] == 0.5
    # Must serialize without raising / emitting NaN or Infinity tokens.
    dumped = json.dumps(result, allow_nan=False)
    assert "NaN" not in dumped and "Infinity" not in dumped
    assert all("status" in f and "message" in f for f in result["findings"])


def test_summarize_handles_empty_history():
    result = summarize({}, "modelsets/run")
    assert result["best_eval_loss"] is None
    assert result["series"]["eval_loss"] == []
    json.dumps(result, allow_nan=False)


def test_runinterpreter_ignores_nonfinite_eval_points():
    interp = RunInterpreter(_stats([(1, 1.0), (2, float("nan")), (3, 0.5)]))
    assert all(math.isfinite(y) for _, y in interp.eval)
    assert interp.best == 0.5


# --- load_stats --------------------------------------------------------------

def test_load_stats_prefers_training_stats_json(tmp_path):
    (tmp_path / "training_stats.json").write_text(json.dumps({"epoch": 3, "source": "direct"}))
    ckpt = tmp_path / "checkpoint-100"
    ckpt.mkdir()
    (ckpt / "trainer_state.json").write_text(json.dumps({"source": "checkpoint"}))

    assert load_stats(str(tmp_path))["source"] == "direct"


def test_load_stats_falls_back_to_newest_checkpoint(tmp_path):
    for n in (50, 200, 100):
        ckpt = tmp_path / f"checkpoint-{n}"
        ckpt.mkdir()
        (ckpt / "trainer_state.json").write_text(json.dumps({"global_step": n}))

    assert load_stats(str(tmp_path))["global_step"] == 200  # newest


def test_load_stats_raises_when_nothing_present(tmp_path):
    with pytest.raises(FileNotFoundError, match="No training telemetry"):
        load_stats(str(tmp_path))
