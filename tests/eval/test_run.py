"""Tier-3 integration tests for fastsft.eval.run (the `fastsft-eval` CLI).

Covers the three resolution helpers and main()'s wiring without any inference,
judging, or embedding: _resolve_parent's flag-vs-metadata precedence,
_resolve_prompt_set's load/regenerate/reuse priority, and main() threading the
resolved EvalConfig into the Evaluator and persisting its results.
"""

import argparse
import sys

import pytest

import fastsft.eval.run as run_mod
from fastsft.constants import DEFAULT_PARENT_MODEL
from fastsft.data.constants import DEFAULT_PARENT_TEMPERATURE
from fastsft.model.constants import DEFAULT_MAX_TOKENS


def _parent_args(parent_model=None, parent_instruction=None):
    return argparse.Namespace(
        parent_model=parent_model, parent_instruction=parent_instruction
    )


# --- _resolve_parent: flags win, then metadata, then defaults ---------------

def test_resolve_parent_defaults_when_no_metadata_no_flags(monkeypatch):
    monkeypatch.setattr(run_mod, "load_training_metadata", lambda _dir: None)
    model, instruction, max_tokens, temperature = run_mod._resolve_parent(
        _parent_args(), "modelsets/run"
    )
    assert model == DEFAULT_PARENT_MODEL
    assert instruction == ""
    assert max_tokens == DEFAULT_MAX_TOKENS
    assert temperature == DEFAULT_PARENT_TEMPERATURE


def test_resolve_parent_uses_metadata_when_flags_absent(monkeypatch):
    metadata = {
        "parent_model": "meta/teacher",
        "parent_instruction": "talk like a pirate",
        "parent_max_tokens": 512,
        "parent_temperature": 0.3,
    }
    monkeypatch.setattr(run_mod, "load_training_metadata", lambda _dir: metadata)
    model, instruction, max_tokens, temperature = run_mod._resolve_parent(
        _parent_args(), "modelsets/run"
    )
    assert model == "meta/teacher"
    assert instruction == "talk like a pirate"
    assert max_tokens == 512
    assert temperature == 0.3


def test_resolve_parent_flags_override_metadata(monkeypatch):
    metadata = {"parent_model": "meta/teacher", "parent_instruction": "from-meta"}
    monkeypatch.setattr(run_mod, "load_training_metadata", lambda _dir: metadata)
    model, instruction, _mt, _temp = run_mod._resolve_parent(
        _parent_args(parent_model="flag/teacher", parent_instruction="from-flag"),
        "modelsets/run",
    )
    assert model == "flag/teacher"
    assert instruction == "from-flag"


def test_resolve_parent_recipe_is_inferred_only(monkeypatch):
    # max_tokens/temperature come from metadata even when identity is flag-set.
    metadata = {"parent_max_tokens": 256, "parent_temperature": 0.1}
    monkeypatch.setattr(run_mod, "load_training_metadata", lambda _dir: metadata)
    _m, _i, max_tokens, temperature = run_mod._resolve_parent(
        _parent_args(parent_model="flag/teacher"), "modelsets/run"
    )
    assert max_tokens == 256
    assert temperature == 0.1


# --- _resolve_prompt_set: explicit path / regenerate / reuse ----------------

class _FakePromptSet:
    """Records which classmethod produced it; stands in for EvalPromptSet."""

    calls = []

    def __init__(self, prompts):
        self._prompts = list(prompts)

    @property
    def prompts(self):
        return self._prompts

    def __len__(self):
        return len(self._prompts)

    def save(self, timestamp):
        _FakePromptSet.calls.append(("save", timestamp))
        return "datasets/eval/set"

    @classmethod
    def load(cls, path=None):
        _FakePromptSet.calls.append(("load", path))
        return cls(["loaded-1", "loaded-2"])

    @classmethod
    def generate(cls, adapter_dir, model=None, num_prompts=0):
        _FakePromptSet.calls.append(("generate", adapter_dir, num_prompts))
        return cls(["gen-1", "gen-2"])


@pytest.fixture
def fake_prompt_set(monkeypatch):
    _FakePromptSet.calls = []
    monkeypatch.setattr(run_mod, "EvalPromptSet", _FakePromptSet)
    # generate() constructs a Model(); keep it inert (no API key needed).
    monkeypatch.setattr(run_mod, "Model", lambda **kwargs: object())
    monkeypatch.setattr(run_mod, "current_timestamp", lambda: "TS")
    return _FakePromptSet


def _prompt_set_args(eval_prompts_path=None, regenerate_prompts=False):
    return argparse.Namespace(
        eval_prompts_path=eval_prompts_path, regenerate_prompts=regenerate_prompts
    )


def test_resolve_prompt_set_explicit_path(fake_prompt_set, make_eval_config):
    run_mod._resolve_prompt_set(
        _prompt_set_args(eval_prompts_path="datasets/eval/mine"), make_eval_config()
    )
    assert fake_prompt_set.calls == [("load", "datasets/eval/mine")]


def test_resolve_prompt_set_reuses_latest_by_default(fake_prompt_set, make_eval_config):
    run_mod._resolve_prompt_set(_prompt_set_args(), make_eval_config())
    assert fake_prompt_set.calls == [("load", None)]


def test_resolve_prompt_set_regenerate_forces_fresh(fake_prompt_set, make_eval_config):
    run_mod._resolve_prompt_set(
        _prompt_set_args(regenerate_prompts=True),
        make_eval_config(adapter_dir="modelsets/run", num_eval_prompts=5),
    )
    kinds = [c[0] for c in fake_prompt_set.calls]
    assert "load" not in kinds
    assert ("generate", "modelsets/run", 5) in fake_prompt_set.calls
    assert ("save", "TS") in fake_prompt_set.calls


def test_resolve_prompt_set_generates_when_none_saved(
    fake_prompt_set, make_eval_config, monkeypatch
):
    def _missing(path=None):
        _FakePromptSet.calls.append(("load", path))
        raise FileNotFoundError

    monkeypatch.setattr(_FakePromptSet, "load", classmethod(lambda cls, path=None: _missing(path)))
    run_mod._resolve_prompt_set(_prompt_set_args(), make_eval_config())
    kinds = [c[0] for c in _FakePromptSet.calls]
    # Tried the latest, missed, then generated + saved a fresh set.
    assert kinds == ["load", "generate", "save"]


# --- main(): resolved config -> Evaluator -> save_results -------------------

class _RecordingEvaluator:
    def __init__(self, config, verbose=True):
        self.config = config
        _RecordingEvaluator.last = self

    def run(self, prompts):
        self.run_prompts = prompts
        return {"comparisons": {"tuned_vs_untuned": {"win_rate": 0.75}}}


@pytest.fixture
def run_eval_main(monkeypatch):
    """Runs main() with the resolution helpers + Evaluator + IO patched out.
    Returns (evaluator, save_calls)."""

    def _run(argv):
        save_calls = []
        _RecordingEvaluator.last = None
        monkeypatch.setattr(run_mod, "load_training_metadata", lambda _dir: {})
        monkeypatch.setattr(run_mod, "latest_run_path", lambda _dir: "modelsets/latest")
        monkeypatch.setattr(
            run_mod, "_resolve_prompt_set",
            lambda args, config: _FakePromptSet(["a", "b", "c"]),
        )
        monkeypatch.setattr(run_mod, "Evaluator", _RecordingEvaluator)
        monkeypatch.setattr(
            run_mod, "save_results",
            lambda results, adapter_dir: save_calls.append((results, adapter_dir))
            or "modelsets/latest/eval_results.json",
        )
        monkeypatch.setattr(sys, "argv", ["fastsft-eval", *argv])
        run_mod.main()
        return _RecordingEvaluator.last, save_calls

    return _run


def test_main_defaults_adapter_to_latest_run(run_eval_main):
    evaluator, save_calls = run_eval_main([])
    assert evaluator.config.adapter_dir == "modelsets/latest"
    # Evaluator got the resolved prompt set's prompts...
    assert evaluator.run_prompts == ["a", "b", "c"]
    # ...and results were persisted against the same adapter dir.
    (results, adapter_dir), = save_calls
    assert adapter_dir == "modelsets/latest"
    assert results["comparisons"]["tuned_vs_untuned"]["win_rate"] == 0.75


def test_main_uses_positional_adapter_dir(run_eval_main):
    evaluator, save_calls = run_eval_main(["modelsets/run-7"])
    assert evaluator.config.adapter_dir == "modelsets/run-7"
    assert save_calls[0][1] == "modelsets/run-7"


def test_main_swap_positions_default_true(run_eval_main):
    evaluator, _ = run_eval_main([])
    assert evaluator.config.swap_positions is True


def test_main_no_swap_disables_swap(run_eval_main):
    evaluator, _ = run_eval_main(["--no-swap"])
    assert evaluator.config.swap_positions is False


def test_main_threads_model_flags_into_config(run_eval_main):
    evaluator, _ = run_eval_main(["--judge-model", "org/judge", "--max-new-tokens", "64"])
    assert evaluator.config.judge_model == "org/judge"
    assert evaluator.config.max_new_tokens == 64
