"""Constants for the eval package (extrinsic judge-based evaluation)."""

# Default size of the held-out eval prompt set. Smaller than a training run --
# every prompt costs three generations (parent, tuned, untuned) plus judging.
DEFAULT_NUM_EVAL_PROMPTS = 50

# Local Hugging Face sentence-embedding model for parent-similarity scoring.
# Small and CPU-friendly; runs via sentence-transformers (evaluation extra).
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Child-inference generation budget (new tokens per answer) and how many
# prompts to feed the model at once.
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_INFERENCE_BATCH_SIZE = 8

# Compare each pair twice with A/B swapped to cancel the judge's position bias.
DEFAULT_SWAP_POSITIONS = True

# Eval results are colocated with the adapter they describe (same place the
# trainer writes training_stats.json), not in a dataset run folder.
EVAL_RESULTS_FILENAME = "eval_results.json"

# Rubric (judge system prompt) for the pairwise quality comparison. Deliberately
# generic -- it drives both tuned-vs-untuned and tuned-vs-parent comparisons.
COMPARISON_JUDGE_INSTRUCTION = (
    "You are comparing two AI assistant responses, A and B, to the same user "
    "question. Decide which response is higher quality overall -- more helpful, "
    "accurate, relevant, and clearly written. Judge only on quality: ignore the "
    "order in which the responses are presented, and do not prefer a response "
    "merely for being longer. Respond with a single verdict: \"A\" if A is "
    "better, \"B\" if B is better, or \"tie\" if they are of genuinely equal "
    "quality."
)

# Rubric for the parent-likeness comparison: which candidate answer is more like
# the reference (parent) in STYLE -- tone, voice, structure, formatting,
# verbosity -- regardless of which is more correct. This is the metric aligned
# with the phase-0 distillation objective (voice/tone transfer) that the
# generic-quality rubric above doesn't capture; it excludes correctness (the
# quality metric's job) so the two stay orthogonal.
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
