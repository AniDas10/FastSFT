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


def test_resolve_input_downloads_latest_run_for_repo_id(monkeypatch):
    monkeypatch.setattr(
        "fastsft.hf_helper.list_repo_files",
        lambda repo_id, repo_type: [
            "20260101_000000/adapter_config.json",
            "20260301_120000/adapter_config.json",
            "20260201_000000/adapter_config.json",
            ".gitattributes",
        ],
    )
    captured = {}

    def fake_snapshot_download(*, repo_id, repo_type, allow_patterns):
        captured["args"] = (repo_id, repo_type, allow_patterns)
        return "/cache/downloaded"

    monkeypatch.setattr("fastsft.hf_helper.snapshot_download", fake_snapshot_download)

    result = resolve_input("org/dataset-name", "dataset")

    # Timestamped run ids sort lexicographically == chronologically -> latest wins.
    assert captured["args"] == ("org/dataset-name", "dataset", ["20260301_120000/*"])
    assert result == "/cache/downloaded/20260301_120000"


def test_resolve_input_raises_when_repo_has_no_runs(monkeypatch):
    monkeypatch.setattr(
        "fastsft.hf_helper.list_repo_files",
        lambda repo_id, repo_type: [".gitattributes", "README.md"],
    )
    with pytest.raises(FileNotFoundError, match="No runs found"):
        resolve_input("org/dataset-name", "dataset")


def test_resolve_input_returns_local_path_unchanged(monkeypatch):
    called = []
    monkeypatch.setattr(
        "fastsft.hf_helper.snapshot_download",
        lambda **kwargs: called.append(kwargs),
    )

    result = resolve_input("./datasets/formatted/run", "dataset")

    assert result == "./datasets/formatted/run"
    assert called == []


class _FakeRepoUrl(str):
    """Stands in for huggingface_hub's RepoUrl: a str URL with a resolved .repo_id."""

    def __new__(cls, url, repo_id):
        obj = super().__new__(cls, url)
        obj.repo_id = repo_id
        return obj


def test_push_to_hub_uploads_under_run_id_path_with_already_namespaced_id(monkeypatch):
    calls = []
    resolved = _FakeRepoUrl(
        "https://huggingface.co/datasets/org/dataset-name", "org/dataset-name"
    )
    monkeypatch.setattr(
        "fastsft.hf_helper.create_repo",
        lambda repo_id, repo_type, exist_ok: calls.append(("create", repo_id, repo_type, exist_ok))
        or resolved,
    )
    monkeypatch.setattr(
        "fastsft.hf_helper.upload_folder",
        lambda repo_id, folder_path, path_in_repo, repo_type, ignore_patterns: calls.append(
            ("upload", repo_id, folder_path, path_in_repo, repo_type, ignore_patterns)
        ),
    )

    url = push_to_hub("/local/dir", "org/dataset-name", "dataset", "20260822_163827")

    assert calls == [
        ("create", "org/dataset-name", "dataset", True),
        ("upload", "org/dataset-name", "/local/dir", "20260822_163827", "dataset", None),
    ]
    # Never overwrites a prior run -- each push lands under its own run_id path.
    assert url == "https://huggingface.co/datasets/org/dataset-name/tree/main/20260822_163827"


def test_push_to_hub_uploads_using_namespace_resolved_by_create_repo(monkeypatch):
    """Regression: create_repo auto-namespaces a bare "name" to the token
    owner's "namespace/name" -- upload_folder must use that resolved id, not
    the original bare one, or it 404s against a repo that doesn't exist."""
    calls = []
    resolved = _FakeRepoUrl(
        "https://huggingface.co/datasets/alice/fastsft-test-dataset",
        "alice/fastsft-test-dataset",
    )
    monkeypatch.setattr(
        "fastsft.hf_helper.create_repo",
        lambda repo_id, repo_type, exist_ok: calls.append(("create", repo_id, repo_type, exist_ok))
        or resolved,
    )
    monkeypatch.setattr(
        "fastsft.hf_helper.upload_folder",
        lambda repo_id, folder_path, path_in_repo, repo_type, ignore_patterns: calls.append(
            ("upload", repo_id, folder_path, path_in_repo, repo_type, ignore_patterns)
        ),
    )

    url = push_to_hub("/local/dir", "fastsft-test-dataset", "dataset", "20260822_163827")

    assert calls == [
        ("create", "fastsft-test-dataset", "dataset", True),
        ("upload", "alice/fastsft-test-dataset", "/local/dir", "20260822_163827", "dataset", None),
    ]
    assert (
        url
        == "https://huggingface.co/datasets/alice/fastsft-test-dataset/tree/main/20260822_163827"
    )


def test_push_to_hub_forwards_ignore_patterns(monkeypatch):
    calls = []
    resolved = _FakeRepoUrl("https://huggingface.co/org/child-model", "org/child-model")
    monkeypatch.setattr(
        "fastsft.hf_helper.create_repo", lambda repo_id, repo_type, exist_ok: resolved
    )
    monkeypatch.setattr(
        "fastsft.hf_helper.upload_folder",
        lambda repo_id, folder_path, path_in_repo, repo_type, ignore_patterns: calls.append(
            ignore_patterns
        ),
    )

    push_to_hub(
        "/local/adapter", "org/child-model", "model", "run-7", ignore_patterns=["checkpoint-*"]
    )

    assert calls == [["checkpoint-*"]]


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
