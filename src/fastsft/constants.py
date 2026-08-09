"""Constants used directly by the pipeline entry points (main.py, pipeline.py).

Model-mechanics constants live in model/constants.py and data-generation
tuning in data/constants.py.
"""

DEFAULT_OUTPUT_DIR = "datasets"
RAW_OUTPUT_SUBDIR = "raw"
FORMATTED_OUTPUT_SUBDIR = "formatted"
# Held-out eval prompts (evaluation module). Persisted once and reused across
# every adapter evaluated, for apples-to-apples comparison over time.
EVAL_PROMPTS_SUBDIR = "eval_prompts"
MODELSETS_OUTPUT_DIR = "modelsets"
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
