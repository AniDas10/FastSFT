"""Core training-telemetry logic: load a finished run's stats, structure them,
and diagnose them. Pure stdlib -- no `rich`, no heavy project deps -- so the
data/logic is reusable and testable on its own (the `--json` output, a future
evaluation module, or any direct library consumer).

Reads `training_stats.json` (written by training/trainer.py::run_sft) if
present, otherwise falls back to the latest checkpoint's `trainer_state.json`,
so it also works on adapters trained before stats were persisted.

Terminal rendering AND the `python -m` CLI live in training/stats_viewer.py
(`uv run python -m training.stats_viewer`); this module only powers them.
"""

import glob
import json
import math
import os

from findings import Finding

# --- Diagnostic thresholds (deliberately lenient -- these classify a trend,
# they aren't precise measurements). ---
MIN_EVALS_FOR_TREND = 2  # fewer than this and there's no trend to read at all
MIN_EVALS_FOR_TURN = 3   # need at least this many to see loss turn back up (overfit)
IMPROVEMENT_MARGIN = 0.01  # relative drop from the first eval to count as "learned"
OVERFIT_MARGIN = 0.02      # relative rise above the best eval to flag overfitting
UNDERFIT_SLOPE_MARGIN = 0.01  # relative drop over the last eval step = "still improving"
_EPS = 1e-6


def load_stats(adapter_dir: str) -> dict:
    """Loads training telemetry for `adapter_dir`: the persisted
    `training_stats.json` if present, else the newest checkpoint's
    `trainer_state.json` (same schema for the fields we read)."""
    direct = os.path.join(adapter_dir, "training_stats.json")
    if os.path.exists(direct):
        with open(direct) as f:
            return json.load(f)

    checkpoints = glob.glob(os.path.join(adapter_dir, "checkpoint-*"))
    checkpoints.sort(key=lambda p: _checkpoint_number(p))
    for ckpt in reversed(checkpoints):
        state_path = os.path.join(ckpt, "trainer_state.json")
        if os.path.exists(state_path):
            with open(state_path) as f:
                return json.load(f)

    raise FileNotFoundError(
        f"No training telemetry in '{adapter_dir}' (looked for training_stats.json "
        "and checkpoint-*/trainer_state.json). Was it produced by FineTuner?"
    )


def _checkpoint_number(path: str) -> int:
    tail = path.rsplit("-", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def series(log_history: list, key: str) -> list[tuple[int, float]]:
    """Extracts (step, value) points for `key` from the trainer's log history."""
    return [(e["step"], e[key]) for e in log_history if key in e and "step" in e]


class RunInterpreter:
    """Runs a battery of named diagnostic checks over one run's telemetry.

    Each `_check_*` method inspects the parsed metrics and returns a `Finding`
    when its condition holds, or `None` when it doesn't apply. `run()` executes
    them in reading order and collects the findings. To add a diagnostic, write
    a `_check_*` method and list it in `run()` -- nothing else changes.
    """

    def __init__(self, stats: dict):
        log = stats.get("log_history", [])
        # Keep only finite points -- a too-short/degenerate run can log NaN/inf,
        # which would poison the comparisons below.
        self.eval = [(s, y) for s, y in series(log, "eval_loss") if math.isfinite(y)]
        self.epoch = stats.get("epoch")
        self.max_epochs = stats.get("num_train_epochs")
        self.loss_masking = stats.get("loss_masking")  # "completion"|"assistant"|"full"|None

        losses = [y for _, y in self.eval]
        self.first = losses[0] if losses else None
        self.last = losses[-1] if losses else None
        self.best = min(losses) if losses else None
        self.best_step = min(self.eval, key=lambda p: p[1])[0] if self.eval else None
        self.early_stopped = (
            self.epoch is not None
            and bool(self.max_epochs)
            and self.epoch < self.max_epochs - _EPS
        )

    def run(self) -> list[Finding]:
        checks = (
            self._check_epoch_density,
            self._check_improvement,
            self._check_overfitting,
            self._check_underfitting,
            self._check_early_stopping,
        )
        findings = [f for f in (check() for check in checks) if f is not None]
        findings.append(self._loss_scope_caveat())
        return findings

    def _check_epoch_density(self) -> Finding | None:
        """Too few evaluations to read any trend at all (thin telemetry)."""
        if len(self.eval) >= MIN_EVALS_FOR_TREND:
            return None
        if not self.eval:
            return Finding(
                "warn",
                "No usable validation-loss points were recorded -- the run was likely too "
                "short. Train for more steps before reading the curve.",
            )
        return Finding(
            "warn",
            "Only one evaluation was recorded -- not enough to see a trend. Raise "
            "--max-epochs or lower --eval-steps for a real curve.",
        )

    def _check_improvement(self) -> Finding | None:
        """Validation loss dropped meaningfully from where it started (it learned)."""
        if len(self.eval) < MIN_EVALS_FOR_TREND:
            return None
        drop = (self.first - self.best) / self.first if self.first else 0
        if drop <= IMPROVEMENT_MARGIN:
            return None
        return Finding(
            "good",
            f"Validation loss fell {self.first:.3f} → {self.best:.3f} "
            f"(-{drop * 100:.0f}%) -- the model learned from the data.",
        )

    def _check_overfitting(self) -> Finding | None:
        """Validation loss bottomed out and then climbed back up while training
        continued -- the classic overfitting signature."""
        if len(self.eval) < MIN_EVALS_FOR_TURN or self.best <= _EPS:
            return None
        turned_up = self.best_step != self.eval[-1][0]
        rose = (self.last - self.best) / self.best > OVERFIT_MARGIN
        if not (turned_up and rose):
            return None
        return Finding(
            "warn",
            f"Validation loss bottomed at {self.best:.3f} (step {self.best_step}) then rose "
            f"to {self.last:.3f} -- a sign of overfitting. The best checkpoint was kept; "
            "consider fewer epochs, more data, or higher dropout.",
        )

    def _check_underfitting(self) -> Finding | None:
        """The model looks like it had more to learn: either it barely improved,
        or it was still improving when it ran out of epochs."""
        if len(self.eval) < MIN_EVALS_FOR_TREND:
            return None
        drop = (self.first - self.best) / self.first if self.first else 0
        if drop <= IMPROVEMENT_MARGIN:
            return Finding(
                "warn",
                f"Validation loss barely moved ({self.first:.3f} → {self.best:.3f}) -- likely "
                "underfit. Try more data, more epochs, or a higher learning rate.",
            )
        prev = self.eval[-2][1]
        still_descending = self.last <= self.best + _EPS and (prev - self.last) / prev > (
            UNDERFIT_SLOPE_MARGIN
        )
        hit_ceiling = self.epoch is not None and not self.early_stopped
        if still_descending and hit_ceiling:
            return Finding(
                "warn",
                "Validation loss was still falling at the last step and training hit the "
                "max epochs -- likely underfit. Raise --max-epochs to let it keep learning.",
            )
        return None

    def _check_early_stopping(self) -> Finding | None:
        """Training ended before the epoch ceiling -- early stopping engaged."""
        if not self.early_stopped:
            return None
        return Finding(
            "good",
            f"Stopped early at {self.epoch:.2f}/{int(self.max_epochs)} epochs -- early "
            "stopping saw no further improvement and saved compute.",
        )

    def _loss_scope_caveat(self) -> Finding:
        """Always-on reminder of what this loss does and doesn't measure -- worded
        for how the loss was actually masked (recorded at training time)."""
        if self.loss_masking in ("completion", "assistant"):
            scope = (
                "Note: loss is measured over the answer tokens only (the prompt is masked), "
                "so it tracks answer quality fairly directly."
            )
        else:
            scope = (
                "Note: loss is measured over the whole formatted example (system + user + "
                "answer), not the answer alone -- read it as a relative trend."
            )
        return Finding(
            "info",
            f"{scope} For a semantic quality score, use the evaluation module (judge-scored).",
        )


def summarize(stats: dict, adapter_dir: str) -> dict:
    """Structures a run's telemetry into a single dict -- key derived values, the
    finite metric series, and the diagnostic findings. This is the programmatic
    entry point for library consumers who want everything in one object;
    `stats_as_json` is just this, serialized. Non-finite points (NaN/inf from a
    degenerate run) are dropped so the result is JSON-safe."""
    log = stats.get("log_history", [])

    def finite_series(key: str) -> list[list]:
        return [[step, value] for step, value in series(log, key) if math.isfinite(value)]

    interpreter = RunInterpreter(stats)
    return {
        "adapter_dir": adapter_dir,
        "loss_masking": stats.get("loss_masking"),
        "epochs_run": stats.get("epoch"),
        "max_epochs": stats.get("num_train_epochs"),
        "global_step": stats.get("global_step"),
        "best_eval_loss": interpreter.best,
        "series": {
            "train_loss": finite_series("loss"),
            "eval_loss": finite_series("eval_loss"),
            "token_accuracy": finite_series("eval_mean_token_accuracy"),
        },
        "findings": [{"status": f.status, "message": f.message} for f in interpreter.run()],
    }


def stats_as_json(stats: dict, adapter_dir: str) -> str:
    """`summarize` serialized to a JSON string (used by the CLI's --json)."""
    return json.dumps(summarize(stats, adapter_dir), indent=2)
