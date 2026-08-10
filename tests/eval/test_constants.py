"""Tier-0 pin tests for fastsft.eval.constants -- eval defaults and judge rubrics.

COMPARISON_JUDGE_INSTRUCTION and STYLE_JUDGE_INSTRUCTION drive every win-rate
metric eval reports; a silent wording change (e.g. STYLE_JUDGE_INSTRUCTION
drifting into judging correctness) would corrupt results without any test
elsewhere catching it, so pin their defining properties directly.
"""

from fastsft.eval.constants import (
    COMPARISON_JUDGE_INSTRUCTION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_INFERENCE_BATCH_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NUM_EVAL_PROMPTS,
    DEFAULT_SWAP_POSITIONS,
    EVAL_RESULTS_FILENAME,
    STYLE_JUDGE_INSTRUCTION,
)


def test_default_num_eval_prompts():
    assert DEFAULT_NUM_EVAL_PROMPTS == 50


def test_default_embedding_model():
    assert DEFAULT_EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_default_inference_budgets():
    assert DEFAULT_MAX_NEW_TOKENS == 512
    assert DEFAULT_INFERENCE_BATCH_SIZE == 8


def test_default_swap_positions_is_on():
    # A/B position-swapping cancels judge position bias; must default on.
    assert DEFAULT_SWAP_POSITIONS is True


def test_eval_results_filename():
    assert EVAL_RESULTS_FILENAME == "eval_results.json"


def test_comparison_judge_instruction_full_text():
    # Full-string pin, not a keyword search: catches wording drift (e.g. a
    # weakened anti-length-bias clause) that would still contain every keyword
    # a substring check might look for.
    assert COMPARISON_JUDGE_INSTRUCTION == (
        "You are comparing two AI assistant responses, A and B, to the same user "
        "question. Decide which response is higher quality overall -- more helpful, "
        "accurate, relevant, and clearly written. Judge only on quality: ignore the "
        "order in which the responses are presented, and do not prefer a response "
        "merely for being longer. Respond with a single verdict: \"A\" if A is "
        "better, \"B\" if B is better, or \"tie\" if they are of genuinely equal "
        "quality."
    )


def test_style_judge_instruction_full_text():
    # This rubric is what distinguishes distillation fidelity (voice/tone) from
    # generic quality -- it must exclude correctness. Pinned as a full string
    # rather than a keyword search so any rewording is caught, not just the
    # removal of specific words.
    assert STYLE_JUDGE_INSTRUCTION == (
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
