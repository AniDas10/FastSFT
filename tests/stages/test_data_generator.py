"""Unit tests for fastsft.stages.data_generator (DataGenerator stage).

The network-facing _run is not exercised here; the pure contract surface is:
constructor guards, the input contract, the messages conversion, and the
provenance sidecar written by save_output.
"""

import pytest
from datasets import Dataset

from fastsft.helper import convert_to_distiset
from fastsft.stages.data_generator import DataGenerator


def _generator(**overrides):
    params = {"verbose": False}
    params.update(overrides)
    return DataGenerator(**params)


class _RecordingGuide:
    """Captures the kwargs Guide was constructed with, and the
    generate_instructions call, without touching OpenRouter."""

    last = None

    def __init__(self, **kwargs):
        _RecordingGuide.last = {"init": kwargs}

    def generate_instructions(self, prompt, num_seeds):
        _RecordingGuide.last["call"] = {"prompt": prompt, "num_seeds": num_seeds}
        return "INSTRUCTIONS"


def test_setup_scales_guide_token_budget_to_seed_count(monkeypatch):
    from fastsft.data.constants import GUIDE_TOKENS_PER_SEED
    from fastsft.data.prompt_generator import seed_count
    from fastsft.model.constants import DEFAULT_MAX_TOKENS

    monkeypatch.setattr(
        "fastsft.stages.data_generator.Guide", _RecordingGuide
    )

    gen = _generator(guide_model="guide/x", num_samples=27)
    result = gen._setup("write like a pirate")

    num_seeds = seed_count(27, breadth_exponent=gen._breadth_exponent)
    init = _RecordingGuide.last["init"]
    assert init["model_id"] == "guide/x"
    # Budget grows one GUIDE_TOKENS_PER_SEED allotment per seed topic.
    assert init["max_tokens"] == DEFAULT_MAX_TOKENS + num_seeds * GUIDE_TOKENS_PER_SEED

    call = _RecordingGuide.last["call"]
    assert call == {"prompt": "write like a pirate", "num_seeds": num_seeds}
    assert result == "INSTRUCTIONS"


@pytest.mark.parametrize("bad", [0, -1, -50])
def test_init_rejects_nonpositive_num_samples(bad):
    with pytest.raises(ValueError, match="num_samples must be positive"):
        _generator(num_samples=bad)


def test_init_accepts_positive_num_samples():
    gen = _generator(num_samples=3)
    assert gen._num_samples == 3


@pytest.mark.parametrize("bad", ["", "   ", "\n\t", None])
def test_validate_input_rejects_empty_prompt(bad):
    with pytest.raises(ValueError, match="non-empty prompt"):
        _generator()._validate_input(bad)


def test_validate_input_accepts_nonempty_prompt():
    # Returns None (no raise) for a real prompt.
    assert _generator()._validate_input("write like a pirate") is None


def test_to_messages_builds_user_assistant_pairs_and_drops_source_columns():
    raw = convert_to_distiset(
        Dataset.from_dict(
            {
                "instruction": ["what is a knot?", "hoist the sail"],
                "generation": ["Arr, a nautical mile per hour.", "Aye, hoisting."],
            }
        )
    )
    out = _generator()._to_messages(raw)
    train = out["default"]["train"]

    assert "messages" in train.column_names
    assert "instruction" not in train.column_names
    assert "generation" not in train.column_names

    first = train[0]["messages"]
    assert first == [
        {"role": "user", "content": "what is a knot?"},
        {"role": "assistant", "content": "Arr, a nautical mile per hour."},
    ]


def test_save_output_writes_dataset_and_provenance_sidecar(monkeypatch):
    recorded = {}

    def fake_save_distiset(output, subdir, run_id):
        recorded["save"] = (subdir, run_id)
        return f"datasets/{subdir}/{run_id}"

    def fake_save_training_metadata(path, **kwargs):
        recorded["path"] = path
        recorded["metadata"] = kwargs

    monkeypatch.setattr(
        "fastsft.stages.data_generator.save_distiset", fake_save_distiset
    )
    monkeypatch.setattr(
        "fastsft.stages.data_generator.save_training_metadata",
        fake_save_training_metadata,
    )

    gen = _generator(
        parent_model="parent/x", parent_max_tokens=777, parent_temperature=0.5
    )
    gen._parent_instruction = "talk like a pirate"

    path = gen.save_output(output=object(), run_id="run-42")

    subdir, run_id = recorded["save"]
    assert run_id == "run-42"
    assert path == f"datasets/{subdir}/run-42"
    assert recorded["path"] == path
    assert recorded["metadata"] == {
        "parent_model": "parent/x",
        "parent_instruction": "talk like a pirate",
        "parent_max_tokens": 777,
        "parent_temperature": 0.5,
    }


def test_save_output_defaults_missing_instruction_to_empty_string(monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        "fastsft.stages.data_generator.save_distiset",
        lambda output, subdir, run_id: "p",
    )
    monkeypatch.setattr(
        "fastsft.stages.data_generator.save_training_metadata",
        lambda path, **kwargs: recorded.update(kwargs),
    )

    gen = _generator()  # _parent_instruction stays None
    gen.save_output(output=object(), run_id="run-1")

    assert recorded["parent_instruction"] == ""
