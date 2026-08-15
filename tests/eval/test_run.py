"""Tier-3 integration tests for fastsft.eval.run (the `fastsft-eval` CLI).

Covers the three resolution helpers and main()'s wiring without any inference,
judging, or embedding: _resolve_parent's flag-vs-metadata precedence,
_resolve_prompt_set's load/regenerate/reuse priority, and main() threading the
resolved EvalConfig into the Evaluator and persisting its results.
"""

import argparse
import json
import os
import sys

import pytest

import fastsft.eval.run as run_mod
from fastsft.constants import DEFAULT_PARENT_MODEL
from fastsft.data.constants import DEFAULT_PARENT_TEMPERATURE
from fastsft.helper import evalsets_dir
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
        _prompt_set_args(eval_prompts_path="datasets/eval/mine"),
        make_eval_config(run_id="TS"),
    )
    # Always saved into this run's own folder too, even when loaded from elsewhere.
    assert fake_prompt_set.calls == [("load", "datasets/eval/mine"), ("save", "TS")]


def test_resolve_prompt_set_reuses_latest_by_default(fake_prompt_set, make_eval_config):
    run_mod._resolve_prompt_set(_prompt_set_args(), make_eval_config(run_id="TS"))
    assert fake_prompt_set.calls == [("load", None), ("save", "TS")]


def test_resolve_prompt_set_regenerate_forces_fresh(fake_prompt_set, make_eval_config):
    run_mod._resolve_prompt_set(
        _prompt_set_args(regenerate_prompts=True),
        make_eval_config(adapter_dir="modelsets/run", num_eval_prompts=5, run_id="TS"),
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


# --- _resolve_reused_answers: opt-in only, no implicit "latest" guessing ----


def _write_answers(run_dir, records):
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "eval_answers.json"), "w") as f:
        json.dump(records, f)


def test_resolve_reused_answers_none_when_flag_absent():
    # No --reuse-answers-from -- returns immediately, no filesystem access.
    assert run_mod._resolve_reused_answers(None, _FakePromptSet(["p0"])) is None


def test_resolve_reused_answers_bare_run_id_resolves_under_evalsets(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records = [{"prompt": "p0", "parent": "par0", "tuned": "t0", "untuned": "u0"}]
    _write_answers(os.path.join("evalsets", "run-123"), records)

    result = run_mod._resolve_reused_answers("run-123", _FakePromptSet(["p0"]))

    assert result == records


def test_resolve_reused_answers_full_path_used_directly(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records = [{"prompt": "p0", "parent": "par0", "tuned": "t0", "untuned": "u0"}]
    run_dir = tmp_path / "elsewhere" / "run-abc"
    _write_answers(str(run_dir), records)

    result = run_mod._resolve_reused_answers(str(run_dir), _FakePromptSet(["p0"]))

    assert result == records


def test_resolve_reused_answers_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs(os.path.join("evalsets", "empty-run"))

    with pytest.raises(FileNotFoundError, match="eval_answers.json"):
        run_mod._resolve_reused_answers("empty-run", _FakePromptSet(["p0"]))


def test_resolve_reused_answers_missing_prompts_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records = [{"prompt": "p0", "parent": "par0", "tuned": "t0", "untuned": "u0"}]
    _write_answers(os.path.join("evalsets", "partial-run"), records)

    with pytest.raises(ValueError, match=r"missing answers for 1/2"):
        run_mod._resolve_reused_answers("partial-run", _FakePromptSet(["p0", "p1"]))


def test_resolve_reused_answers_full_coverage_returns_answers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    records = [
        {"prompt": "p0", "parent": "par0", "tuned": "t0", "untuned": "u0"},
        {"prompt": "p1", "parent": "par1", "tuned": "t1", "untuned": "u1"},
    ]
    _write_answers(os.path.join("evalsets", "full-run"), records)

    result = run_mod._resolve_reused_answers("full-run", _FakePromptSet(["p0", "p1"]))

    assert result == records


# --- main(): resolved config -> Evaluator -> save_results -------------------

class _RecordingEvaluator:
    def __init__(self, config, verbose=True):
        self.config = config
        _RecordingEvaluator.last = self

    def run(self, prompts, reused_answers=None):
        self.run_prompts = prompts
        self.reused_answers = reused_answers
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
            lambda results, run_dir: save_calls.append((results, run_dir))
            or "evalsets/latest/eval_results.json",
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
    # ...and results were persisted under this eval run's own evalsets/<run_id> dir.
    (results, run_dir), = save_calls
    assert run_dir == os.path.join(evalsets_dir(), evaluator.config.run_id)
    assert results["comparisons"]["tuned_vs_untuned"]["win_rate"] == 0.75


def test_main_uses_positional_adapter_dir(run_eval_main):
    evaluator, save_calls = run_eval_main(["modelsets/run-7"])
    assert evaluator.config.adapter_dir == "modelsets/run-7"
    assert save_calls[0][1] == os.path.join(evalsets_dir(), evaluator.config.run_id)


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
