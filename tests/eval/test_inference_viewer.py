"""Tier-2 unit tests for fastsft.eval.inference_viewer's main() wiring: adapter_dir
resolution (Hub repo id vs. local path vs. default-to-latest-run) and threading
into ChildInferenceEngine/render, with ChildInferenceEngine and rich rendering
faked out.
"""

import sys

import pytest

import fastsft.eval.inference_viewer as viewer_mod


class _FakeEngine:
    """Records what it was constructed with and asked to generate."""

    last = None

    def __init__(self, adapter_dir, max_new_tokens=None):
        self.adapter_dir = adapter_dir
        self.max_new_tokens = max_new_tokens
        _FakeEngine.last = self

    def generate_tuned(self, prompts):
        return ["tuned answer"]

    def generate_untuned(self, prompts):
        return ["untuned answer"]


@pytest.fixture
def run_viewer_main(monkeypatch):
    """Runs main() with ChildInferenceEngine/render/latest_run_path/resolve_input
    patched out. Returns (engine, render_calls)."""

    def _run(argv, resolve_input_fn=None):
        render_calls = []
        _FakeEngine.last = None
        monkeypatch.setattr(viewer_mod, "latest_run_path", lambda _dir: "modelsets/latest")
        monkeypatch.setattr(
            viewer_mod, "resolve_input", resolve_input_fn or (lambda path, repo_type: path)
        )
        monkeypatch.setattr(viewer_mod, "ChildInferenceEngine", _FakeEngine)
        monkeypatch.setattr(
            viewer_mod, "render",
            lambda *args, **kwargs: render_calls.append((args, kwargs)),
        )
        monkeypatch.setattr(sys, "argv", ["inference_viewer", *argv])
        viewer_mod.main()
        return _FakeEngine.last, render_calls

    return _run


def test_defaults_adapter_to_latest_run_without_resolving(run_viewer_main):
    captured = {}
    engine, _ = run_viewer_main(
        ["hello"], resolve_input_fn=lambda path, repo_type: captured.setdefault("called", True)
    )
    assert engine.adapter_dir == "modelsets/latest"
    assert "called" not in captured


def test_positional_adapter_dir_resolved_via_hf_helper(run_viewer_main):
    captured = {}

    def fake_resolve(path, repo_type):
        captured["args"] = (path, repo_type)
        return "RESOLVED_LOCAL_ADAPTER"

    engine, _ = run_viewer_main(["hello", "org/child-adapter"], resolve_input_fn=fake_resolve)
    assert captured["args"] == ("org/child-adapter", "model")
    assert engine.adapter_dir == "RESOLVED_LOCAL_ADAPTER"


def test_local_path_passes_through_resolve_input_unchanged(run_viewer_main):
    engine, _ = run_viewer_main(["hello", "./modelsets/run-7"])
    assert engine.adapter_dir == "./modelsets/run-7"


def test_render_receives_resolved_adapter_dir_and_answers(run_viewer_main):
    _, render_calls = run_viewer_main(["hello", "org/child-adapter"])
    (args, _), = render_calls
    adapter_dir, prompt, tuned, untuned = args
    assert adapter_dir == "org/child-adapter"
    assert prompt == "hello"
    assert tuned == "tuned answer"
    assert untuned == "untuned answer"


def test_tuned_only_skips_untuned_generation(run_viewer_main):
    _, render_calls = run_viewer_main(["hello", "./modelsets/run-7", "--tuned-only"])
    (args, _), = render_calls
    assert args[3] is None
