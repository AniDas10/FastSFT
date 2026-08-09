"""Tier-1 unit tests for fastsft.data.prompt_generator.

Pure logic (`seed_count`, `_allocate`, `_row_prompt`) plus the `generate`
top-up loop driven by a fake model whose `run_pipeline` returns canned
structured-output rows -- no network.
"""

import json
import math

import pytest

from fastsft.data.constants import MAX_PROMPT_ATTEMPTS
from fastsft.data.prompt_generator import (
    GeneratedPrompts,
    PromptGenerator,
    seed_count,
)

# --- seed_count -------------------------------------------------------------

@pytest.mark.parametrize(
    "num_samples,exponent,expected",
    [
        (1, 2 / 3, 1),        # edge: single sample -> single seed
        (8, 2 / 3, 4),        # ceil(8 ** 2/3) = ceil(4.0) = 4
        (100, 2 / 3, 22),     # ceil(100 ** 2/3) = ceil(21.54) = 22
        (10, 1.0, 10),        # exponent 1 -> N seeds, clamped to num_samples
        (10, 0.0, 1),         # exponent 0 -> ceil(1) = 1
    ],
)
def test_seed_count_matches_clamped_breadth_formula(num_samples, exponent, expected):
    assert seed_count(num_samples, breadth_exponent=exponent) == expected


@pytest.mark.parametrize("num_samples", [1, 2, 7, 8, 50, 100, 999])
def test_seed_count_never_exceeds_num_samples_and_at_least_one(num_samples):
    result = seed_count(num_samples)
    assert 1 <= result <= num_samples


def test_seed_count_equals_raw_formula_within_bounds():
    # For a mid-range N the clamp is inactive, so it equals the bare formula.
    n = 27
    assert seed_count(n) == math.ceil(n ** (2 / 3))


# --- _allocate --------------------------------------------------------------

def _pg(num_samples=10):
    # _allocate / _row_prompt don't touch the model, so None is fine here.
    return PromptGenerator(model=None, num_samples=num_samples)


@pytest.mark.parametrize(
    "n,seeds,expected",
    [
        # Even split.
        (6, ["a", "b", "c"], [("a", 2), ("b", 2), ("c", 2)]),
        # Remainder goes to the earliest seeds.
        (7, ["a", "b", "c"], [("a", 3), ("b", 2), ("c", 2)]),
        # Fewer instructions than seeds -> zero-count seeds dropped.
        (2, ["a", "b", "c"], [("a", 1), ("b", 1)]),
        # Single seed absorbs everything.
        (5, ["only"], [("only", 5)]),
    ],
)
def test_allocate_distributes_evenly_and_drops_zero_counts(n, seeds, expected):
    assert _pg()._allocate(seeds, n) == expected


@pytest.mark.parametrize(
    "n,num_seeds",
    [(6, 3), (7, 3), (2, 3), (100, 7), (1, 1), (13, 5)],
)
def test_allocate_counts_sum_to_n(n, num_seeds):
    seeds = [f"s{i}" for i in range(num_seeds)]
    allocation = _pg()._allocate(seeds, n)
    assert sum(count for _, count in allocation) == n
    assert all(count > 0 for _, count in allocation)


# --- _row_prompt ------------------------------------------------------------

def test_row_prompt_names_the_seed_and_exact_count():
    prompt = _pg()._row_prompt("What is a knot?", 3)
    assert "What is a knot?" in prompt
    assert "exactly 3" in prompt


# --- generate (top-up loop, fake model) -------------------------------------

class _FakeModel:
    """Returns, per pass, exactly `per_pass` prompts for each allocated row.

    Mimics Model.run_pipeline's Distiset shape and assert_structured_output.
    With `per_pass` < the requested count, each pass under-delivers, exercising
    the top-up loop / MAX_PROMPT_ATTEMPTS ceiling.
    """

    def __init__(self, per_pass=None):
        self._per_pass = per_pass
        self.calls = 0

    def assert_structured_output(self, generation, sample_id=None):
        return generation

    def run_pipeline(self, data, system_prompt, structured_output=None, name=""):
        self.calls += 1
        rows = []
        for entry in data:
            count = entry["count"]
            n = count if self._per_pass is None else min(self._per_pass, count)
            prompts = [f"q{self.calls}-{i}" for i in range(n)]
            rows.append(
                {"generation": GeneratedPrompts(prompts=prompts).model_dump_json(),
                 "count": count}
            )
        return {"default": {"train": rows}}


def test_generate_returns_exactly_num_samples_in_one_pass():
    model = _FakeModel()  # delivers the full requested count each row
    gen = PromptGenerator(model=model, num_samples=6)

    prompts = gen.generate(["a", "b", "c"])

    assert len(prompts) == 6
    assert model.calls == 1


def test_generate_tops_up_across_passes_when_underdelivering():
    # Each row yields at most 1 prompt/pass, so 4 samples need several passes.
    model = _FakeModel(per_pass=1)
    gen = PromptGenerator(model=model, num_samples=4)

    prompts = gen.generate(["a", "b"])

    assert len(prompts) == 4
    assert 1 < model.calls <= MAX_PROMPT_ATTEMPTS


def test_generate_caps_prompts_at_requested_count_per_row():
    # Model returns MORE than asked; generate must slice to the row's count.
    class OverDelivering(_FakeModel):
        def run_pipeline(self, data, system_prompt, structured_output=None, name=""):
            self.calls += 1
            rows = []
            for entry in data:
                extra = [f"x{i}" for i in range(entry["count"] + 5)]
                rows.append(
                    {"generation": GeneratedPrompts(prompts=extra).model_dump_json(),
                     "count": entry["count"]}
                )
            return {"default": {"train": rows}}

    gen = PromptGenerator(model=OverDelivering(), num_samples=6)
    assert len(gen.generate(["a", "b", "c"])) == 6


def test_generate_raises_after_max_attempts_when_never_enough():
    # Zero prompts per pass -> deficit never closes -> RuntimeError.
    model = _FakeModel(per_pass=0)
    gen = PromptGenerator(model=model, num_samples=3)

    with pytest.raises(RuntimeError, match=r"0/3 instructions after 5 attempts"):
        gen.generate(["a", "b"])
    assert model.calls == MAX_PROMPT_ATTEMPTS


def test_generate_rejects_empty_seeds():
    gen = PromptGenerator(model=_FakeModel(), num_samples=3)
    with pytest.raises(ValueError, match="at least one seed"):
        gen.generate([])


def test_generate_passes_structured_output_json_through_parser():
    # Sanity: the canned generation really is GeneratedPrompts-shaped JSON.
    payload = GeneratedPrompts(prompts=["a", "b"]).model_dump_json()
    assert json.loads(payload) == {"prompts": ["a", "b"]}
