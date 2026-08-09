"""Unit tests for fastsft.stages.fine_tuner (FineTuner stage).

Only the pure/decision surface is exercised: the input contract, the
train/validation split, and the three config-resolution branches. The actual
training paths (_run_modal / _run_local / _download_adapter) hit Modal, tar,
and the GPU and are out of scope for this tier.
"""

import pytest
from datasets import Dataset

from fastsft.helper import convert_to_distiset
from fastsft.stages.fine_tuner import FineTuner
from fastsft.training.config import AdapterConfig, TrainingConfig, TrainingLoopConfig


def _tuner(**overrides):
    params = {"child_model_id": "child/model", "verbose": False}
    params.update(overrides)
    return FineTuner(**params)


def _formatted(n):
    return convert_to_distiset(
        Dataset.from_dict({"text": [f"row-{i}" for i in range(n)]})
    )


def test_validate_input_rejects_missing_text_column():
    bad = convert_to_distiset(Dataset.from_dict({"messages": [[]]}))
    with pytest.raises(ValueError, match="requires a 'text' column"):
        _tuner()._validate_input(bad)


def test_validate_input_accepts_text_column(formatted_distiset):
    assert _tuner()._validate_input(formatted_distiset) is None


def test_resolve_uses_caller_supplied_config():
    supplied = TrainingConfig(gpu_tier="A100", strategy="qlora")
    tuner = _tuner(training_config=supplied)
    chosen = tuner._resolve_training_config(_formatted(4))
    assert chosen is supplied


def test_resolve_local_training_uses_detected_device(monkeypatch):
    monkeypatch.setattr("fastsft.device.detect_device", lambda: "mps")
    tuner = _tuner(local_training=True)

    chosen = tuner._resolve_training_config(_formatted(4))

    assert "mps" in chosen.gpu_tier
    assert chosen.gpu_tier.startswith("local")
    # Defaults for the adapter/loop when trained locally.
    assert chosen.adapter == AdapterConfig()
    assert chosen.loop == TrainingLoopConfig()


def test_resolve_heuristic_picks_cheapest_feasible(monkeypatch):
    cheapest = TrainingConfig(
        gpu_tier="T4", strategy="qlora", est_usd_per_hour=0.5, est_memory_gb=15.0
    )
    pricier = TrainingConfig(
        gpu_tier="A100", strategy="lora", est_usd_per_hour=3.0, est_memory_gb=40.0
    )
    captured = {}

    def fake_recommend_configs(child_model_id, sample_texts, top_n):
        captured["child"] = child_model_id
        captured["texts"] = list(sample_texts)
        return [cheapest, pricier]

    monkeypatch.setattr(
        "fastsft.stages.fine_tuner.recommend_configs", fake_recommend_configs
    )

    tuner = _tuner()
    chosen = tuner._resolve_training_config(_formatted(3))

    assert chosen is cheapest  # shortlist[0] == cheapest feasible
    assert captured["child"] == "child/model"
    assert captured["texts"] == ["row-0", "row-1", "row-2"]


def test_split_validation_partitions_all_rows():
    tuner = _tuner()
    train, eval_ = tuner._split_validation(_formatted(10), validation_split=0.2)
    assert len(train) + len(eval_) == 10
    assert len(eval_) == 2
    assert len(train) == 8


def test_split_validation_is_deterministic():
    tuner = _tuner()
    ds = _formatted(10)
    train_a, eval_a = tuner._split_validation(ds, validation_split=0.3)
    train_b, eval_b = tuner._split_validation(ds, validation_split=0.3)
    # seed=42 -> identical partition across calls.
    assert train_a["text"] == train_b["text"]
    assert eval_a["text"] == eval_b["text"]
