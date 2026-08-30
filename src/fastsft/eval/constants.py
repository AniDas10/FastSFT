"""Constants for the eval package (extrinsic judge-based evaluation)."""

# Every prompt costs three generations (parent, tuned, untuned) plus judging.
DEFAULT_NUM_EVAL_PROMPTS = 50

# CPU-friendly; runs via sentence-transformers (evaluation extra).
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_INFERENCE_BATCH_SIZE = 8

# Compare each pair twice with A/B swapped to cancel the judge's position bias.
DEFAULT_SWAP_POSITIONS = True

EVAL_RESULTS_FILENAME = "eval_results.json"

# Written right after generation so a judging failure doesn't lose them.
EVAL_ANSWERS_FILENAME = "eval_answers.json"

# Deliberately generic -- drives both tuned-vs-untuned and tuned-vs-parent comparisons.
COMPARISON_JUDGE_INSTRUCTION = (
    "You are comparing two AI assistant responses, A and B, to the same user "
    "question. Decide which response is higher quality overall -- more helpful, "
    "accurate, relevant, and clearly written. Judge only on quality: ignore the "
    "order in which the responses are presented, and do not prefer a response "
    "merely for being longer. Respond with a single verdict: \"A\" if A is "
    "better, \"B\" if B is better, or \"tie\" if they are of genuinely equal "
    "quality."
)

# Excludes correctness, to stay orthogonal to the quality rubric above.
STYLE_JUDGE_INSTRUCTION = (
    "You are given a user question, a REFERENCE response, and two candidate "
    "responses, A and B. Decide which candidate more closely matches the "
    "REFERENCE's STYLE -- its tone, voice, structure, formatting, verbosity, "
    "and overall approach -- regardless of which candidate is more correct, "
    "helpful, or better written. You are judging stylistic resemblance to the "
    "reference, not quality. Ignore the order in which the candidates are "
    "presented. Respond with a single verdict: \"A\" if A is more like the "
    "reference, \"B\" if B is more like the reference, or \"tie\" if they "
    "resemble it equally."
)
