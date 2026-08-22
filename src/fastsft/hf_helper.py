"""Shared helpers: Hugging Face Hub push/pull for stage-output folders (dataset
or model repos). CLI argument validation lives in validation_checks.py.
"""

import os

from dotenv import load_dotenv
from huggingface_hub import create_repo, get_token, snapshot_download, upload_folder
from huggingface_hub.utils import HFValidationError, validate_repo_id

load_dotenv()


def looks_like_repo_id(value: str) -> bool:
    """True if `value` looks like a Hub repo id ("name" or "org/name") rather
    than a local filesystem path."""
    if not value or value in (".", "..") or os.path.exists(value):
        return False
    if os.sep in value or (os.altsep and os.altsep in value):
        # A single "org/name" segment is still repo-id-shaped; deeper or rooted paths aren't.
        return value.count("/") == 1 and not value.startswith((".", "/", "~"))
    return True


def push_to_hub(local_dir: str, repo_id: str, repo_type: str) -> str:
    """Create `repo_id` if needed and upload `local_dir`'s contents; returns the
    Hub URL. Called in addition to, never instead of, the local save."""
    create_repo(repo_id, repo_type=repo_type, exist_ok=True)
    upload_folder(repo_id=repo_id, folder_path=local_dir, repo_type=repo_type)
    prefix = "datasets/" if repo_type == "dataset" else ""
    return f"https://huggingface.co/{prefix}{repo_id}"


def resolve_input(path_or_repo_id: str, repo_type: str) -> str:
    """If `path_or_repo_id` looks like a repo id, download a local snapshot and
    return its path; otherwise return it unchanged as a local path."""
    if looks_like_repo_id(path_or_repo_id):
        return snapshot_download(repo_id=path_or_repo_id, repo_type=repo_type)
    return path_or_repo_id


def repo_id_error(repo_id: str) -> str | None:
    """None if `repo_id` is a well-formed Hub repo id, else a human-readable reason."""
    try:
        validate_repo_id(repo_id)
    except HFValidationError as e:
        return str(e)
    return None


def has_token() -> bool:
    """True if huggingface_hub can find a token (HF_TOKEN env var, or a prior
    `huggingface-cli login`)."""
    return get_token() is not None
