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
    "questions or requests.\n"
    "\n"
    "IMPORTANT - Domain Diversity:\n"
    "If the user's description mentions applying the style/persona to 'everything', "
    "'any topic', 'any question', or 'all subjects' (not a single narrow domain), "
    "generate seed topics from DIVERSE DOMAINS. Examples:\n"
    "  - If persona='software engineer': include seeds about software, cooking, "
    "home repair, finance, relationships, philosophy, travel.\n"
    "  - If persona='pirate': include seeds about sailing, trading, adventure, "
    "history, humor, life lessons, problem-solving.\n"
    "  - If persona='poet': include seeds about nature, love, human nature, "
    "technology, social issues, personal growth, loss.\n"
    "Try to have at least 50% of seeds outside the primary domain.\n"
    "\n"
    "If the user describes a SPECIFIC domain (e.g., 'respond like a doctor to "
    "medical questions'), stay within that domain.\n"
    "\n"
    "Each seed must be a single, concrete question phrased exactly as a real "
    "user would type it. Topics should span different subjects. None may mention "
    "the style, persona, tone, or format, and none may be meta-instructions "
    "such as 'write a response in the style of...' or 'provide a sample response...'."
)
