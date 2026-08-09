"""Integration tests for fastsft.pipeline (the DistillationPipeline orchestrator).

Two layers: (1) real _build_stages / start_stage selection with the actual
stage classes (their __init__ is network-free), and (2) the real run/_run_stages
loop driving lightweight recording stub stages so no network/GPU is touched.
"""

import pytest

from fastsft.data.config import DataGenerationConfig
from fastsft.pipeline import DistillationPipeline
from fastsft.stages.base import Stage
from fastsft.stages.constants import (
    DATA_FORMATTER,
    DATA_GENERATOR,
    FINE_TUNER,
    STAGE_ORDER,
)
from fastsft.stages.data_formatter import DataFormatter
from fastsft.stages.data_generator import DataGenerator
from fastsft.stages.fine_tuner import FineTuner
from fastsft.training.config import TrainingConfig

# --- Layer 1: construction / stage selection (real _build_stages) ------------


@pytest.mark.parametrize(
    "start_stage, expected_types",
    [
        (DATA_GENERATOR, [DataGenerator, DataFormatter, FineTuner]),
        (DATA_FORMATTER, [DataFormatter, FineTuner]),
        (FINE_TUNER, [FineTuner]),
    ],
)
def test_build_stages_is_the_suffix_from_start_stage(start_stage, expected_types):
    pipeline = DistillationPipeline(start_stage=start_stage)
    assert [type(s) for s in pipeline.stages] == expected_types
    assert pipeline._start_index == STAGE_ORDER.index(start_stage)
    assert pipeline.start_stage == start_stage


def test_default_start_stage_builds_the_full_pipeline():
    pipeline = DistillationPipeline()
    assert pipeline.start_stage == STAGE_ORDER[0]
    assert [type(s) for s in pipeline.stages] == [
        DataGenerator,
        DataFormatter,
        FineTuner,
    ]


def test_invalid_start_stage_raises_naming_stage_order():
    with pytest.raises(ValueError, match="start_stage must be one of"):
        DistillationPipeline(start_stage="not_a_stage")


def test_generation_config_threads_into_data_generator():
    generation = DataGenerationConfig(num_samples=7, guide_model="guide/x")
    pipeline = DistillationPipeline(generation=generation)
    data_gen = pipeline.stages[0]
    assert isinstance(data_gen, DataGenerator)
    assert data_gen._num_samples == 7
    assert data_gen._guide_model == "guide/x"


def test_child_model_id_and_training_thread_into_fine_tuner():
    training = TrainingConfig(gpu_tier="A100")
    pipeline = DistillationPipeline(
        child_model_id="org/child-1b",
        training=training,
        start_stage=FINE_TUNER,
    )
    fine_tuner = pipeline.stages[0]
    assert isinstance(fine_tuner, FineTuner)
    assert fine_tuner._child_model_id == "org/child-1b"
    assert fine_tuner._training_config is training


def test_local_training_flag_threads_into_fine_tuner():
    pipeline = DistillationPipeline(local_training=True, start_stage=FINE_TUNER)
    assert pipeline.stages[0]._local_training is True


# --- Layer 2: run / _run_stages orchestration (real loop, stub stages) -------


class _StubStage(Stage):
    """Records the input it received and returns a transformed marker
    (input + this stage's tag), so tests can assert data is threaded stage
    to stage. `name` must be a real stage name for the Stage base guard."""

    def __init__(self, name, tag, ran, fail=False):
        self.name = name
        super().__init__(verbose=False)
        self._tag = tag
        self._ran = ran  # shared list recording run order across stubs
        self._fail = fail
        self.received = None

    def run(self, data):
        self.received = data
        self._ran.append(self._tag)
        if self._fail:
            raise RuntimeError(f"{self._tag} boom")
        return f"{data}->{self._tag}"


def _pipeline_with_stubs(stubs, start_stage=DATA_GENERATOR):
    pipeline = DistillationPipeline(start_stage=start_stage)
    pipeline.stages = stubs
    return pipeline


def test_run_none_raises_prompt_message_for_data_generator():
    pipeline = DistillationPipeline(start_stage=DATA_GENERATOR)
    with pytest.raises(ValueError, match="prompt string"):
        # run() guards eagerly, before returning the iterator.
        pipeline.run(None)


def test_run_none_raises_distiset_message_for_later_stage():
    pipeline = DistillationPipeline(start_stage=FINE_TUNER)
    with pytest.raises(ValueError, match="a Distiset"):
        pipeline.run(None)


def test_run_yields_each_stage_in_order_threading_output():
    ran = []
    stubs = [
        _StubStage(DATA_GENERATOR, "gen", ran),
        _StubStage(DATA_FORMATTER, "fmt", ran),
        _StubStage(FINE_TUNER, "tune", ran),
    ]
    pipeline = _pipeline_with_stubs(stubs)

    results = list(pipeline.run("seed"))

    # One (stage, output) per stage, in order.
    assert [stage for stage, _ in results] == stubs
    assert ran == ["gen", "fmt", "tune"]
    # Each stage received the prior stage's output, not the original input.
    assert stubs[0].received == "seed"
    assert stubs[1].received == "seed->gen"
    assert stubs[2].received == "seed->gen->fmt"
    # Yielded outputs are the running transformation; final is the last stage's.
    assert [output for _, output in results] == [
        "seed->gen",
        "seed->gen->fmt",
        "seed->gen->fmt->tune",
    ]


def test_run_is_lazy_nothing_executes_until_iterated():
    ran = []
    stubs = [_StubStage(DATA_GENERATOR, "gen", ran)]
    pipeline = _pipeline_with_stubs(stubs)

    iterator = pipeline.run("seed")
    # Building the iterator must not run any stage.
    assert ran == []

    next(iter(iterator))
    assert ran == ["gen"]


def test_earlier_yields_survive_a_later_stage_failure():
    ran = []
    stubs = [
        _StubStage(DATA_GENERATOR, "gen", ran),
        _StubStage(DATA_FORMATTER, "fmt", ran, fail=True),
        _StubStage(FINE_TUNER, "tune", ran),
    ]
    pipeline = _pipeline_with_stubs(stubs)

    gen = pipeline.run("seed")
    # First stage yields fine...
    first_stage, first_output = next(gen)
    assert first_stage is stubs[0]
    assert first_output == "seed->gen"

    # ...the second stage raises when pulled, and the third never runs.
    with pytest.raises(RuntimeError, match="fmt boom"):
        next(gen)
    assert ran == ["gen", "fmt"]
