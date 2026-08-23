"""Unit tests for fastsft.stages.data_formatter (DataFormatter stage).

The Hugging Face tokenizer is mocked; no model is downloaded.
"""

import pytest
from datasets import Dataset

from fastsft.helper import convert_to_distiset
from fastsft.stages.data_formatter import DataFormatter


def _formatter(**overrides):
    params = {"child_model_id": "child/model", "verbose": False}
    params.update(overrides)
    return DataFormatter(**params)


def _distiset(rows):
    return convert_to_distiset(Dataset.from_dict({"messages": rows}))


def test_validate_input_rejects_missing_messages_column():
    bad = convert_to_distiset(Dataset.from_dict({"text": ["hi"]}))
    with pytest.raises(ValueError, match="requires a 'messages' column"):
        _formatter()._validate_input(bad)


def test_validate_input_rejects_malformed_messages():
    # messages present but not a list-of-{role,content} dicts.
    bad = _distiset([["not-a-dict"]])
    with pytest.raises(ValueError, match="non-empty list"):
        _formatter()._validate_input(bad)


def test_validate_input_accepts_valid_messages(sample_distiset):
    assert _formatter()._validate_input(sample_distiset) is None


def test_validate_input_passes_empty_train_set():
    # An empty dataset skips the row-shape check (len == 0 branch).
    empty = convert_to_distiset(
        Dataset.from_dict({"messages": []}, features=None)
    )
    assert _formatter()._validate_input(empty) is None


def test_load_tokenizer_rejects_model_without_chat_template(monkeypatch):
    class FakeTokenizer:
        chat_template = None

    monkeypatch.setattr(
        "fastsft.stages.data_formatter.AutoTokenizer.from_pretrained",
        lambda model_id: FakeTokenizer(),
    )
    with pytest.raises(ValueError, match="no chat_template"):
        _formatter()._load_tokenizer()


def test_load_tokenizer_caches_after_first_load(monkeypatch):
    calls = []

    class FakeTokenizer:
        chat_template = "{{ messages }}"

    def fake_from_pretrained(model_id):
        calls.append(model_id)
        return FakeTokenizer()

    monkeypatch.setattr(
        "fastsft.stages.data_formatter.AutoTokenizer.from_pretrained",
        fake_from_pretrained,
    )

    fmt = _formatter()
    first = fmt._load_tokenizer()
    second = fmt._load_tokenizer()

    assert first is second
    assert calls == ["child/model"]  # loaded exactly once


def test_run_renders_text_column(monkeypatch, sample_distiset):
    class FakeTokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, tokenize=False):
            return f"RENDERED:{messages[0]['content']}"

    monkeypatch.setattr(
        "fastsft.stages.data_formatter.AutoTokenizer.from_pretrained",
        lambda model_id: FakeTokenizer(),
    )

    out = _formatter()._run(sample_distiset)
    train = out["default"]["train"]
    assert "text" in train.column_names
    assert train[0]["text"].startswith("RENDERED:")


def test_save_output_persists_under_formatted_subdir(monkeypatch):
    from fastsft.constants import FORMATTED_OUTPUT_SUBDIR

    recorded = {}

    def fake_save_distiset(output, subdir, run_id):
        recorded["args"] = (output, subdir, run_id)
        return f"datasets/{subdir}/{run_id}"

    monkeypatch.setattr(
        "fastsft.stages.data_formatter.save_distiset", fake_save_distiset
    )

    sentinel = object()
    path = _formatter().save_output(sentinel, run_id="run-7")

    output, subdir, run_id = recorded["args"]
    assert output is sentinel
    assert subdir == FORMATTED_OUTPUT_SUBDIR
    assert run_id == "run-7"
    assert path == f"datasets/{FORMATTED_OUTPUT_SUBDIR}/run-7"


def test_save_output_pushes_to_hub_when_repo_id_set(monkeypatch):
    monkeypatch.setattr(
        "fastsft.stages.data_formatter.save_distiset",
        lambda output, subdir, run_id: "datasets/formatted/run-7",
    )
    calls = []
    monkeypatch.setattr(
        "fastsft.stages.data_formatter.push_to_hub",
        lambda local_dir, repo_id, repo_type, run_id: calls.append(
            (local_dir, repo_id, repo_type, run_id)
        )
        or "https://huggingface.co/datasets/org/name/tree/main/run-7",
    )

    formatter = _formatter(dataset_repo_id="org/name")
    path = formatter.save_output(object(), run_id="run-7")

    assert path == "datasets/formatted/run-7"
    assert calls == [("datasets/formatted/run-7", "org/name", "dataset", "run-7")]


def test_save_output_skips_hub_push_when_repo_id_absent(monkeypatch):
    monkeypatch.setattr(
        "fastsft.stages.data_formatter.save_distiset",
        lambda output, subdir, run_id: "datasets/formatted/run-7",
    )
    calls = []
    monkeypatch.setattr(
        "fastsft.stages.data_formatter.push_to_hub",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    _formatter().save_output(object(), run_id="run-7")

    assert calls == []
