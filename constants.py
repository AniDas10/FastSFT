"""Project-wide constants."""

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"

DEFAULT_OUTPUT_DIR = "datasets"
RAW_OUTPUT_SUBDIR = "raw"
FORMATTED_OUTPUT_SUBDIR = "formatted"
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# distilabel's default (128) truncates structured-output JSON.
DEFAULT_MAX_TOKENS = 1024

DEFAULT_PARENT_MODEL = "meta-llama/llama-3.3-70b-instruct"

# Different family from the parent, to avoid self-preference bias.
DEFAULT_JUDGE_MODEL = "deepseek/deepseek-chat"
DEFAULT_SCORE_THRESHOLD = 5.0
MAX_REFINE_ITERATIONS = 5

# Must support tool calls (structured output).
DEFAULT_GUIDE_MODEL = "qwen/qwen-2.5-7b-instruct"
DEFAULT_GUIDE_INSTRUCTION = (
    "Given a user's description of a synthetic dataset (style, domain, tone), "
    "produce: `parent_instruction`, a system prompt for generating one sample "
    "in that style; `judge_instruction`, a system prompt for scoring a sample "
    "0-10 against that style; and `sample_instruction`, a user instruction "
    "for a single item (never a batch or 'a dataset of' multiple items)."
)

# Hugging Face repo id, not an OpenRouter model id.
DEFAULT_CHILD_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
