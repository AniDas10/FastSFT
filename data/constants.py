"""Constants for the data package (synthetic-data generation and refinement)."""

# Default dataset size when the caller doesn't specify.
DEFAULT_NUM_SAMPLES = 100

# Default sampling temperature for the parent model's generations -- high, to
# keep the synthetic dataset varied. (The Model base class's own default is
# separate; the model layer can't depend on data/ without inverting the layers.)
DEFAULT_PARENT_TEMPERATURE = 0.9

# Breadth (distinct seed topics) = ceil(N ** BREADTH_EXPONENT); >0.5 favors
# breadth over depth (breadth:depth = N^exp : N^(1-exp)).
BREADTH_EXPONENT = 2 / 3

# Extra guide-output budget per seed, so many-seed requests don't truncate.
GUIDE_TOKENS_PER_SEED = 64

# Max passes to top up under-delivered prompt generations to num_samples.
MAX_PROMPT_ATTEMPTS = 5

MAX_REFINE_ITERATIONS = 5

# System prompt for the parent model when generating user instructions.
PROMPT_GENERATOR_INSTRUCTION = (
    "You generate user questions for a synthetic dataset. Given a seed question "
    "and a requested count, produce that many DISTINCT user questions on the same "
    "topic, spanning a range of complexity -- the simplest a plain question, the "
    "most complex adding multiple constraints, sub-parts, or multi-step reasoning. "
    "Each must be a natural, concrete question a real user would type. None may "
    "mention answer style, persona, tone, or format."
)
