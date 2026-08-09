"""Unit tests for fastsft.stages.base (the Stage validate-then-run template)."""

import pytest

from fastsft.stages.base import Stage


class _StubStage(Stage):
    """Minimal Stage with a name, recording the order of validate/run calls."""

    name = "stub"

    def __init__(self, verbose=False, fail_validation=False):
        super().__init__(verbose=verbose)
        self.calls = []
        self._fail_validation = fail_validation

    def _validate_input(self, data):
        self.calls.append("validate")
        if self._fail_validation:
            raise ValueError("bad input")

    def _run(self, data):
        self.calls.append("run")
        return f"ran:{data}"


class _NamelessStage(Stage):
    """A Stage subclass that forgot to set `name`."""


def test_missing_name_raises():
    with pytest.raises(NotImplementedError, match="non-empty class attribute"):
        _NamelessStage()


def test_named_stage_constructs():
    stage = _StubStage()
    assert stage.name == "stub"


def test_run_validates_before_running_and_returns_run_result():
    stage = _StubStage()
    result = stage.run("payload")
    assert result == "ran:payload"
    assert stage.calls == ["validate", "run"]


def test_run_aborts_when_validation_fails():
    stage = _StubStage(fail_validation=True)
    with pytest.raises(ValueError, match="bad input"):
        stage.run("payload")
    # _run must never be reached once validation raises.
    assert stage.calls == ["validate"]


def test_base_validate_and_run_are_abstract():
    stage = _StubStage()
    with pytest.raises(NotImplementedError, match="_validate_input"):
        Stage._validate_input(stage, None)
    with pytest.raises(NotImplementedError, match="_run"):
        Stage._run(stage, None)


def test_save_output_defaults_to_none():
    stage = _StubStage()
    assert stage.save_output("anything", "run-1") is None
