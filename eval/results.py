"""Core evaluation-results logic: persist a finished run's results next to its
adapter, load them back, and turn them into plain-English takeaways. Pure stdlib
-- no `rich`, no heavy deps -- so the data/logic stays reusable and testable
(the `--json` output or any direct library consumer).

Terminal rendering AND the `python -m` CLI live in eval/results_viewer.py
(`uv run python -m eval.results_viewer`); this module only powers them.
"""

import json
import math
import os

from eval.constants import EVAL_RESULTS_FILENAME
from findings import Finding

# Standard errors from 50% a win rate must clear to count as a real edge, not
# noise. Conservative on purpose: LLM-judge verdicts add noise on top of prompt
# sampling, and a wrong verdict here would misdirect a training decision.
WIN_MARGIN_SIGMAS = 1.5
# Minimum mean-cosine change to call a shift toward/away from the parent real.
SIM_MARGIN = 0.01


def _win_margin(num_prompts: int) -> float:
    """Standard-error-based margin for `num_prompts` eval prompts: how far a
    win rate must sit from 50% to be distinguishable from noise at this
    sample size (binomial SE at p=0.5 is sqrt(0.25/n)), widened by
    WIN_MARGIN_SIGMAS. Smaller eval sets get a wider, more cautious margin."""
    return WIN_MARGIN_SIGMAS * math.sqrt(0.25 / num_prompts)


def save_results(results: dict, adapter_dir: str) -> str:
    """Writes `results` to adapter_dir/eval_results.json; returns the path."""
    path = os.path.join(adapter_dir, EVAL_RESULTS_FILENAME)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    return path


def load_results(adapter_dir: str) -> dict:
    """Loads a finished run's eval_results.json from `adapter_dir`."""
    path = os.path.join(adapter_dir, EVAL_RESULTS_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No {EVAL_RESULTS_FILENAME} in '{adapter_dir}'. Run "
            "`python -m eval.run [adapter_dir]` first."
        )
    with open(path) as f:
        return json.load(f)


def interpret(results: dict) -> list[Finding]:
    """Plain-English takeaways from a results dict, in reading order."""
    comparisons = results.get("comparisons", {})
    num_prompts = results.get("num_prompts") or 0
    findings = [
        _tuned_vs_untuned(comparisons.get("tuned_vs_untuned"), num_prompts),
        _parent_likeness(comparisons.get("parent_likeness"), num_prompts),
        _tuned_vs_parent(comparisons.get("tuned_vs_parent"), num_prompts),
        _similarity(results.get("similarity_to_parent")),
        _caveat(),
    ]
    return [f for f in findings if f is not None]


def _tuned_vs_untuned(comparison: dict | None, num_prompts: int) -> Finding | None:
    """The primary signal: did fine-tuning beat the untuned baseline?"""
    if not comparison or not num_prompts:
        return None
    rate = comparison["win_rate"]
    margin = _win_margin(num_prompts)
    if rate > 0.5 + margin:
        return Finding(
            "good",
            f"Fine-tuning improved quality -- the tuned child beat the untuned "
            f"baseline {rate:.0%} of the time (position-debiased, and outside "
            f"the ±{margin:.0%} noise floor for {num_prompts} eval prompts).",
        )
    if rate < 0.5 - margin:
        return Finding(
            "warn",
            f"The tuned child lost to the untuned baseline ({rate:.0%}, outside "
            f"the ±{margin:.0%} noise floor for {num_prompts} eval prompts) -- "
            "tuning may have hurt. Revisit data quality, epochs, or learning rate.",
        )
    return Finding(
        "warn",
        f"Tuned and untuned were statistically indistinguishable ({rate:.0%}, "
        f"within the ±{margin:.0%} noise floor for {num_prompts} eval prompts) "
        "-- no reliable quality signal yet. Evaluate on more prompts for a "
        "clearer read.",
    )


def _parent_likeness(comparison: dict | None, num_prompts: int) -> Finding | None:
    """The distillation objective: did tuning make the child answer more like
    the parent's style than the untuned baseline?"""
    if not comparison or not num_prompts:
        return None
    rate = comparison["win_rate"]
    margin = _win_margin(num_prompts)
    if rate > 0.5 + margin:
        return Finding(
            "good",
            f"Distillation is transferring style -- the tuned child matched the "
            f"parent's style more than the untuned baseline {rate:.0%} of the "
            f"time (outside the ±{margin:.0%} noise floor for {num_prompts} prompts).",
        )
    if rate < 0.5 - margin:
        return Finding(
            "warn",
            f"The tuned child matches the parent's style LESS than the untuned "
            f"baseline ({rate:.0%}) -- fine-tuning isn't transferring the style. "
            "Check the training data, and that --parent-instruction gives the "
            "judge the true styled teacher as reference.",
        )
    return Finding(
        "warn",
        f"No measurable shift toward the parent's style ({rate:.0%}, within the "
        f"±{margin:.0%} noise floor for {num_prompts} prompts) -- distillation "
        "may not be transferring style yet.",
    )


def _tuned_vs_parent(comparison: dict | None, num_prompts: int) -> Finding | None:
    """The gap to the teacher -- a small model rarely matches the parent."""
    if not comparison or not num_prompts:
        return None
    rate = comparison["win_rate"]
    margin = _win_margin(num_prompts)
    if rate >= 0.5 - margin:
        return Finding(
            "good",
            f"The tuned child is competitive with the parent teacher "
            f"({rate:.0%} win rate, within the ±{margin:.0%} noise floor of "
            "parity) -- the gap to the teacher is small.",
        )
    return Finding(
        "info",
        f"The tuned child wins {rate:.0%} against the parent teacher -- a gap "
        "remains, expected for a much smaller model.",
    )


def _similarity(similarity: dict | None) -> Finding | None:
    """Did tuning move the child's answers toward the parent in embedding space?"""
    if not similarity:
        return None
    tuned = similarity.get("tuned_vs_parent")
    untuned = similarity.get("untuned_vs_parent")
    if tuned is None or untuned is None:
        return None
    if tuned > untuned + SIM_MARGIN:
        return Finding(
            "good",
            f"Tuning moved the child's answers closer to the parent in embedding "
            f"space ({untuned:.2f} -> {tuned:.2f} mean cosine).",
        )
    if tuned < untuned - SIM_MARGIN:
        return Finding(
            "warn",
            f"Tuning moved the child's answers away from the parent "
            f"({untuned:.2f} -> {tuned:.2f} mean cosine) -- unexpected for distillation.",
        )
    return Finding(
        "info",
        f"Child-to-parent similarity barely changed ({untuned:.2f} -> {tuned:.2f} "
        "mean cosine).",
    )


def _caveat() -> Finding:
    """Always-on reminder of what these Phase-1 signals do and don't measure."""
    return Finding(
        "info",
        "Phase-1 signals: a subjective LLM judge (pairwise, position-debiased) "
        "plus embedding similarity. Read them together, not as absolute scores.",
    )


def results_as_json(results: dict) -> str:
    """The results dict plus its findings, serialized (used by the CLI's --json)."""
    enriched = dict(results)
    enriched["findings"] = [
        {"status": f.status, "message": f.message} for f in interpret(results)
    ]
    return json.dumps(enriched, indent=2)
