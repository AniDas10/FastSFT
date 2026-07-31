"""Project-wide constants."""

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"

# Parent directory every run's timestamped output folder is created under.
# Raw (DataGenerator) and formatted (DataFormatter) outputs are kept in
# separate subdirs so a saved run's stage is unambiguous from its path alone.
DEFAULT_OUTPUT_DIR = "datasets"
RAW_OUTPUT_SUBDIR = "raw"
FORMATTED_OUTPUT_SUBDIR = "formatted"
RUN_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# distilabel/OpenAILLM defaults max_new_tokens to 128, which is enough for a
# single score but truncates anything that has to produce a few sentences
# (e.g. Guide's two instruction strings) -- 128 was silently cutting off
# structured-output JSON mid-object, causing the whole batch to fail.
DEFAULT_MAX_TOKENS = 1024

# Primarily responsible for data generation at every stage.
DEFAULT_PARENT_MODEL = "meta-llama/llama-3.3-70b-instruct"

# Deliberately a different model family than the parent (DeepSeek vs Llama) --
# using the same model to both generate and judge its own outputs risks
# self-preference bias. Higher score ensure higher passing criterias leading to better dataset quality.
# Max refine iterations acts as a fail safe to cap iterations during refactoring of dataset quality.
DEFAULT_JUDGE_MODEL = "deepseek/deepseek-chat"
DEFAULT_SCORE_THRESHOLD = 5.0
MAX_REFINE_ITERATIONS = 5

# Guide only needs to produce short instruction text, not the dataset
# content itself, so a small/cheap model is enough -- but it must reliably
# support tool calls, since that's what structured_output (GuideInstructions)
# relies on under the hood. mistralai/mistral-nemo was tried first (has
# tool-capable OpenRouter endpoints on paper) but failed unpredictably in
# practice -- verified qwen/qwen-2.5-7b-instruct reliable across 8/8 live runs.
DEFAULT_GUIDE_MODEL = "qwen/qwen-2.5-7b-instruct"
DEFAULT_GUIDE_INSTRUCTION = (
    "Given a user's description of a synthetic dataset (style, domain, tone), "
    "produce: `parent_instruction`, a system prompt for generating one sample "
    "in that style; `judge_instruction`, a system prompt for scoring a sample "
    "0-10 against that style; and `sample_instruction`, a user instruction "
    "for a single item (never a batch or 'a dataset of' multiple items)."
)

# A Hugging Face repo id (NOT an OpenRouter model id -- different id space),
# used by DataFormatter/FineTuner for AutoTokenizer/model loading. Small and
# open, with a known chat_template, so it works out of the box as a default.
DEFAULT_CHILD_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
