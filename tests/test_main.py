"""Tier-3 integration tests for fastsft.main (the `fastsft` CLI entry point).

Drives main() with crafted argv and asserts the argv -> config-construction
path: which DataGenerationConfig / TrainingConfig it builds, the prompt-vs-
--input-path branch, and that bad flag combos surface as SystemExit (via the
real validation_checks). The pipeline itself is replaced by a recorder, so no
stage ever runs -- this exercises wiring, not training.
"""

import sys

import pytest

import fastsft.main as main_mod
from fastsft.constants import DEFAULT_CHILD_MODEL_ID
from fastsft.data.config import DataGenerationConfig
from fastsft.stages.constants import STAGE_ORDER
from fastsft.training.config import AdapterConfig, TrainingConfig

FIRST_STAGE = STAGE_ORDER[0]
LATER_STAGE = STAGE_ORDER[-1]  # fine_tuner


class _RecordingPipeline:
    """Captures the DistillationPipeline ctor kwargs and the run() input; run()
    yields nothing so main()'s save loop is a no-op."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.run_input = _UNSET
        _RecordingPipeline.last = self

    def run(self, data):
        self.run_input = data
        return iter(())


_UNSET = object()


@pytest.fixture
def run_main(monkeypatch):
    """Runs main() with the given argv, the pipeline + IO tail patched out.
    Returns the _RecordingPipeline instance that main() constructed."""

    def _run(argv, input_marker="LOADED_DATASET"):
        _RecordingPipeline.last = None
        monkeypatch.setattr(main_mod, "DistillationPipeline", _RecordingPipeline)
        monkeypatch.setattr(main_mod, "load_data", lambda path: input_marker)
        monkeypatch.setattr(main_mod, "current_timestamp", lambda: "RUNID")
        monkeypatch.setattr(sys, "argv", ["fastsft", *argv])
        main_mod.main()
        return _RecordingPipeline.last

    return _run


# --- default (data_generator) branch ---------------------------------------

def test_default_invocation_builds_generation_config(run_main):
    pipe = run_main(["make pirate data", "--num-samples", "7"])

    assert pipe.kwargs["child_model_id"] == DEFAULT_CHILD_MODEL_ID
    assert pipe.kwargs["start_stage"] == FIRST_STAGE
    assert pipe.kwargs["local_training"] is False
    assert pipe.kwargs["training"] is None

    generation = pipe.kwargs["generation"]
    assert isinstance(generation, DataGenerationConfig)
    assert generation.num_samples == 7
    # The prompt argument flows straight through as the pipeline input.
    assert pipe.run_input == "make pirate data"


def test_child_model_id_override(run_main):
    pipe = run_main(["some prompt", "--child-model-id", "org/tiny-model"])
    assert pipe.kwargs["child_model_id"] == "org/tiny-model"


# --- start-stage / input-path branch ---------------------------------------

def test_later_stage_uses_load_data_and_no_generation(run_main):
    pipe = run_main(
        ["--start-stage", LATER_STAGE, "--input-path", "datasets/formatted/run"],
        input_marker="FORMATTED_DS",
    )
    assert pipe.kwargs["start_stage"] == LATER_STAGE
    assert pipe.kwargs["generation"] is None
    # Non-default start stage feeds the loaded dataset, not the prompt.
    assert pipe.run_input == "FORMATTED_DS"


# --- training-config construction ------------------------------------------

def test_gpu_tier_builds_training_config_with_only_given_overrides(run_main):
    pipe = run_main(
        ["p", "--gpu-tier", "A100-80GB", "--lora-rank", "8"]
    )
    training = pipe.kwargs["training"]
    assert isinstance(training, TrainingConfig)
    assert training.gpu_tier == "A100-80GB"
    # Given override lands...
    assert training.adapter.rank == 8
    # ...unpassed adapter/loop fields keep their dataclass defaults.
    assert training.adapter.dropout == AdapterConfig().dropout
    assert pipe.kwargs["local_training"] is False


def test_zero_valued_override_is_not_dropped_as_falsy(run_main):
    # --lora-dropout 0.0 is meaningful (is-not-None, not truthy) and must land.
    pipe = run_main(["p", "--gpu-tier", "A100-80GB", "--lora-dropout", "0.0"])
    assert pipe.kwargs["training"].adapter.dropout == 0.0


def test_local_alone_leaves_training_none(run_main):
    pipe = run_main(["p", "--local"])
    assert pipe.kwargs["local_training"] is True
    # No explicit overrides -> let FineTuner apply local defaults.
    assert pipe.kwargs["training"] is None


def test_local_with_override_builds_local_training_config(run_main):
    pipe = run_main(["p", "--local", "--batch-size", "2"])
    training = pipe.kwargs["training"]
    assert isinstance(training, TrainingConfig)
    assert training.gpu_tier == "local"
    assert training.loop.batch_size == 2
    assert pipe.kwargs["local_training"] is True


# --- validation errors surface as SystemExit -------------------------------

def test_later_stage_without_input_path_exits(run_main):
    with pytest.raises(SystemExit):
        run_main(["--start-stage", LATER_STAGE])


def test_gpu_tier_and_local_mutually_exclusive_exits(run_main):
    with pytest.raises(SystemExit):
        run_main(["p", "--gpu-tier", "A100-80GB", "--local"])


def test_override_without_destination_exits(run_main):
    with pytest.raises(SystemExit):
        run_main(["p", "--lora-rank", "16"])


def test_default_stage_without_prompt_exits(run_main):
    with pytest.raises(SystemExit):
        run_main([])
