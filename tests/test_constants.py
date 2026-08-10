"""Tier-0 pin tests for fastsft.constants -- entry-point/output-layout constants.

These are load-bearing: run folder paths, env var names, and model defaults
that other modules and on-disk artifacts depend on by exact value. A silent
change here (e.g. a typo'd env var name or a swapped model id) wouldn't fail
loudly elsewhere, so pin them directly.
"""

from fastsft.constants import (
    DEFAULT_CHILD_MODEL_ID,
    DEFAULT_GUIDE_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARENT_MODEL,
    EVAL_PROMPTS_SUBDIR,
    FORMATTED_OUTPUT_SUBDIR,
    MODELSETS_OUTPUT_DIR,
    OUTPUT_DIR_ENV_VAR,
    RAW_OUTPUT_SUBDIR,
    RUN_TIMESTAMP_FORMAT,
    TRAINING_METADATA_FILENAME,
)


def test_output_dir_env_var_name():
    assert OUTPUT_DIR_ENV_VAR == "FASTSFT_OUTPUT_DIR"


def test_output_layout_subdirs():
    assert DEFAULT_OUTPUT_DIR == "datasets"
    assert RAW_OUTPUT_SUBDIR == "raw"
    assert FORMATTED_OUTPUT_SUBDIR == "formatted"
    assert EVAL_PROMPTS_SUBDIR == "eval_prompts"
    assert MODELSETS_OUTPUT_DIR == "modelsets"


def test_run_timestamp_format_is_sortable_and_filesystem_safe():
    # %Y%m%d_%H%M%S: no separators that break paths, and lexical sort ==
    # chronological sort (latest_run_path relies on this).
    assert RUN_TIMESTAMP_FORMAT == "%Y%m%d_%H%M%S"
    from datetime import UTC, datetime

    stamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC).strftime(RUN_TIMESTAMP_FORMAT)
    assert stamp == "20260102_030405"


def test_training_metadata_filename():
    assert TRAINING_METADATA_FILENAME == "training_metadata.json"


def test_default_parent_model_is_open_weight_llama():
    assert DEFAULT_PARENT_MODEL == "meta-llama/llama-3.3-70b-instruct"


def test_default_judge_model_is_different_family_from_parent():
    # Different model family than the parent avoids self-preference bias.
    assert DEFAULT_JUDGE_MODEL == "deepseek/deepseek-chat"
    assert "llama" not in DEFAULT_JUDGE_MODEL.lower()


def test_default_guide_model():
    assert DEFAULT_GUIDE_MODEL == "qwen/qwen-2.5-7b-instruct"


def test_default_child_model_id_is_hugging_face_instruct_repo():
    assert DEFAULT_CHILD_MODEL_ID == "Qwen/Qwen2.5-0.5B-Instruct"
    assert "instruct" in DEFAULT_CHILD_MODEL_ID.lower()
