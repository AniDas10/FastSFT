"""Tier-1 unit tests for fastsft.hf_helper (repo-id detection, push/pull).

huggingface_hub's create_repo/upload_folder/snapshot_download are mocked; no
network calls are made.
"""

import pytest

from fastsft.hf_helper import (
    has_token,
    looks_like_repo_id,
    push_to_hub,
    repo_id_error,
    resolve_input,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("org/dataset-name", True),
        ("gpt2", True),
        ("./local/path", False),
        ("/abs/path", False),
        ("~/data", False),
        ("datasets/raw/nested/run", False),
        ("", False),
        (".", False),
        ("..", False),
    ],
)
def test_looks_like_repo_id_by_shape(value, expected):
    assert looks_like_repo_id(value) is expected


def test_looks_like_repo_id_false_for_existing_local_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "modelsets" / "run-1"
    existing.mkdir(parents=True)
    # Same shape as an "org/name" repo id, but it exists locally -> not a repo id.
    assert looks_like_repo_id("modelsets/run-1") is False


def test_resolve_input_downloads_for_repo_id(monkeypatch):
    captured = {}

    def fake_snapshot_download(*, repo_id, repo_type):
        captured["args"] = (repo_id, repo_type)
        return "/cache/downloaded"

    monkeypatch.setattr("fastsft.hf_helper.snapshot_download", fake_snapshot_download)

    result = resolve_input("org/dataset-name", "dataset")

    assert result == "/cache/downloaded"
    assert captured["args"] == ("org/dataset-name", "dataset")


def test_resolve_input_returns_local_path_unchanged(monkeypatch):
    called = []
    monkeypatch.setattr(
        "fastsft.hf_helper.snapshot_download",
        lambda **kwargs: called.append(kwargs),
    )

    result = resolve_input("./datasets/formatted/run", "dataset")

    assert result == "./datasets/formatted/run"
    assert called == []


def test_push_to_hub_creates_then_uploads(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "fastsft.hf_helper.create_repo",
        lambda repo_id, repo_type, exist_ok: calls.append(("create", repo_id, repo_type, exist_ok)),
    )
    monkeypatch.setattr(
        "fastsft.hf_helper.upload_folder",
        lambda repo_id, folder_path, repo_type: calls.append(
            ("upload", repo_id, folder_path, repo_type)
        ),
    )

    url = push_to_hub("/local/dir", "org/dataset-name", "dataset")

    assert calls == [
        ("create", "org/dataset-name", "dataset", True),
        ("upload", "org/dataset-name", "/local/dir", "dataset"),
    ]
    assert url == "https://huggingface.co/datasets/org/dataset-name"


def test_push_to_hub_model_url_has_no_dataset_prefix(monkeypatch):
    monkeypatch.setattr("fastsft.hf_helper.create_repo", lambda repo_id, **_: None)
    monkeypatch.setattr("fastsft.hf_helper.upload_folder", lambda **_: None)

    url = push_to_hub("/local/adapter", "org/child-model", "model")

    assert url == "https://huggingface.co/org/child-model"


@pytest.mark.parametrize(
    "repo_id", ["org/dataset-name", "gpt2", "Foo-BAR_foo.bar123"]
)
def test_repo_id_error_none_for_valid_ids(repo_id):
    assert repo_id_error(repo_id) is None


@pytest.mark.parametrize(
    "repo_id", ["too/many/slashes", "other..repo..id", "-leading-dash", "repo.git"]
)
def test_repo_id_error_message_for_invalid_ids(repo_id):
    assert repo_id_error(repo_id) is not None


def test_has_token_true_when_hf_token_env_set(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    assert has_token() is True


def test_has_token_false_when_no_token_available(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr("fastsft.hf_helper.get_token", lambda: None)
    assert has_token() is False
