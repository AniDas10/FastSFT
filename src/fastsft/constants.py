"""Constants used directly by the pipeline entry points (main.py, pipeline.py).

Model-mechanics constants live in model/constants.py and data-generation
tuning in data/constants.py.
"""

# Base directory holding `datasets/` and `modelsets/`. Overridable via the
# OUTPUT_DIR_ENV_VAR env var or the --output-dir CLI flag, so an installed
# fastsft can write outside the current directory (resolved by helper.py's
# datasets_dir()/modelsets_dir()). Empty/unset means the current directory, so
# running from the repo is unchanged.
OUTPUT_DIR_ENV_VAR = "FASTSFT_OUTPUT_DIR"
DEFAULT_OUTPUT_DIR = "datasets"
RAW_OUTPUT_SUBDIR = "raw"
FORMATTED_OUTPUT_SUBDIR = "formatted"
MODELSETS_OUTPUT_DIR = "modelsets"
# One evalsets/<run_id>/ folder per eval run: eval_prompts, eval_answers.json, eval_results.json.
EVALSETS_OUTPUT_DIR = "evalsets"
# Subfolder name for the eval prompt set within an evalsets run dir (see eval/prompt_set.py).
EVAL_PROMPTS_SUBDIR = "eval_prompts"
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# Sidecar (in a raw dataset run dir) recording the teacher that produced the
# training data, so evaluation can reconstruct the true parent reference.
TRAINING_METADATA_FILENAME = "training_metadata.json"

DEFAULT_PARENT_MODEL = "meta-llama/llama-3.3-70b-instruct"

# Different family from the parent, to avoid self-preference bias.
DEFAULT_JUDGE_MODEL = "deepseek/deepseek-chat"

# Must support tool calls (structured output).
DEFAULT_GUIDE_MODEL = "qwen/qwen-2.5-7b-instruct"

# Hugging Face repo id, not an OpenRouter model id.
DEFAULT_CHILD_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
