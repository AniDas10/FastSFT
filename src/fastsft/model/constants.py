"""Constants for the model package (OpenRouter access, LLM roles)."""

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"

# distilabel's default (128) truncates structured-output JSON.
DEFAULT_MAX_TOKENS = 1024

DEFAULT_SCORE_THRESHOLD = 5.0

DEFAULT_GUIDE_INSTRUCTION = (
    "Given a user's description of a synthetic dataset (style, domain, tone), "
    "produce three fields:\n"
    "- `parent_instruction`: a system prompt telling the model how to answer. "
    "All of the style, persona, tone, and formatting live here -- this is where "
    "the voice is defined.\n"
    "- `judge_instruction`: a system prompt for scoring one sample 0-10 on how "
    "well it fits that style.\n"
    "- `sample_instructions`: a list of exactly {num_seeds} distinct user "
    "questions or requests, each on a DIFFERENT topic within the dataset's "
    "domain -- make the topics as varied as the domain allows. Each must be a "
    "single, concrete question about one topic, phrased exactly as a real user "
    "would type it. None may mention the style, persona, tone, or format, and "
    "none may be a meta-instruction such as 'write a response in the style of...' "
    "or 'provide a sample response...'."
)
