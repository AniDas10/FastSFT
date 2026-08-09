"""Tier-1 unit tests for fastsft.eval.prompt_set -- normalization, leakage-safe
dedup, deterministic seed selection, and save/load round-trip."""

import pytest

from fastsft.data.constants import BREADTH_EXPONENT
from fastsft.data.prompt_generator import seed_count
from fastsft.eval.prompt_set import EvalPromptSet, _normalize

# --- _normalize ----------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Hello World", "hello world"),
        ("  Hello   World  ", "hello world"),
        ("HELLO\tWorld\n", "hello world"),
        ("already normal", "already normal"),
    ],
)
def test_normalize(raw, expected):
    assert _normalize(raw) == expected


def test_normalize_collapses_case_and_whitespace_equally():
    assert _normalize("What  IS   a Knot?") == _normalize("what is a knot?")


# --- _dedup --------------------------------------------------------------


def test_dedup_drops_training_collisions_case_and_space_insensitive():
    training = ["What is a knot?"]
    generated = ["what   IS a knot?", "How do sails work?"]
    assert EvalPromptSet._dedup(generated, training) == ["How do sails work?"]


def test_dedup_drops_intra_list_duplicates_keeping_first():
    generated = ["New question", "NEW   question", "Other"]
    assert EvalPromptSet._dedup(generated, []) == ["New question", "Other"]


def test_dedup_preserves_order_and_original_casing():
    generated = ["Bravo", "Alpha", "Charlie"]
    assert EvalPromptSet._dedup(generated, []) == ["Bravo", "Alpha", "Charlie"]


def test_dedup_all_collide_returns_empty():
    training = ["a", "b"]
    assert EvalPromptSet._dedup(["A", "  b "], training) == []


# --- _select_seeds -------------------------------------------------------

TRAINING = [f"question number {i}" for i in range(50)]


@pytest.mark.parametrize("num_prompts", [10, 25, 50])
def test_select_seeds_count_matches_breadth_split(num_prompts):
    seeds = EvalPromptSet._select_seeds(TRAINING, num_prompts)
    expected = min(len(TRAINING), seed_count(num_prompts, breadth_exponent=BREADTH_EXPONENT))
    assert len(seeds) == expected


def test_select_seeds_is_deterministic():
    a = EvalPromptSet._select_seeds(TRAINING, 25)
    b = EvalPromptSet._select_seeds(TRAINING, 25)
    assert a == b


def test_select_seeds_returns_subset_without_duplicates():
    seeds = EvalPromptSet._select_seeds(TRAINING, 25)
    assert set(seeds) <= set(TRAINING)
    assert len(set(seeds)) == len(seeds)


def test_select_seeds_caps_at_training_size():
    small = ["only", "three", "here"]
    seeds = EvalPromptSet._select_seeds(small, 100)
    assert sorted(seeds) == sorted(small)  # can't exceed the training pool


# --- EvalPromptSet basics + save/load round-trip -------------------------


def test_len():
    assert len(EvalPromptSet(["a", "b", "c"])) == 3


def test_save_load_roundtrip(tmp_path, monkeypatch):
    # save_distiset writes under the relative datasets/ dir; chdir to tmp so the
    # round-trip stays isolated from the repo.
    monkeypatch.chdir(tmp_path)
    prompts = ["what is a knot?", "how do sails work?", "define port and starboard"]
    path = EvalPromptSet(prompts).save("20260809_000000")
    loaded = EvalPromptSet.load(path)
    assert loaded.prompts == prompts
    assert len(loaded) == len(prompts)
