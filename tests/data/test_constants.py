"""Tier-0 pin tests for fastsft.data.constants -- data-generation tuning knobs.

BREADTH_EXPONENT and PROMPT_GENERATOR_INSTRUCTION directly shape training-data
quality/diversity; a silent change wouldn't fail loudly (generation would just
run with different topic breadth or a laxer prompt-generation rubric), so pin
the values and the invariants they're supposed to satisfy.
"""

from fastsft.data.constants import (
    BREADTH_EXPONENT,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_PARENT_TEMPERATURE,
    GUIDE_TOKENS_PER_SEED,
    MAX_PROMPT_ATTEMPTS,
    MAX_REFINE_ITERATIONS,
    PROMPT_GENERATOR_INSTRUCTION,
)


def test_default_num_samples():
    assert DEFAULT_NUM_SAMPLES == 100


def test_default_parent_temperature_is_high_for_variety():
    assert DEFAULT_PARENT_TEMPERATURE == 0.9


def test_breadth_exponent_favors_breadth_over_depth():
    # >0.5 means breadth:depth = N^exp : N^(1-exp) skews toward more distinct
    # seed topics rather than more variants per topic. Set high (0.85) to
    # actively favor topic diversity over per-topic complexity depth.
    assert BREADTH_EXPONENT == 0.85
    assert BREADTH_EXPONENT > 0.5


def test_guide_tokens_per_seed():
    assert GUIDE_TOKENS_PER_SEED == 64


def test_max_prompt_attempts():
    assert MAX_PROMPT_ATTEMPTS == 5


def test_max_refine_iterations():
    assert MAX_REFINE_ITERATIONS == 5


def test_prompt_generator_instruction_full_text():
    # Full-string pin, not a keyword search: a partial edit (e.g. softening
    # "None may mention answer style" into something weaker) should fail this
    # test even if it keeps every keyword a substring check would look for.
    assert PROMPT_GENERATOR_INSTRUCTION == (
        "You generate user questions for a synthetic dataset. Given a seed question "
        "and a requested count, produce that many DISTINCT user questions on the same "
        "topic, spanning a range of complexity -- the simplest a plain question, the "
        "most complex adding multiple constraints, sub-parts, or multi-step reasoning. "
        "Each must be a natural, concrete question a real user would type. None may "
        "mention answer style, persona, tone, or format."
    )
