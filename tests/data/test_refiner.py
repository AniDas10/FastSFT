"""Tier-1 unit tests for fastsft.data.refiner.

The load-bearing invariant: `_score`, `_failed_instructions`, and `_drop_failed`
keep every row aligned with its score. A misalignment would silently drop or
regenerate the wrong samples, corrupting the training set -- so alignment is the
focus. The full `refine()` loop is exercised with a fake judge + fake
regeneration (no network).
"""

import pytest
from datasets import Dataset

from fastsft.data.refiner import DataRefiner


def _train(rows):
    """A Dataset of {instruction, generation} rows in order."""
    return Dataset.from_list(rows)


def _rows(n):
    return [{"instruction": f"q{i}", "generation": f"a{i}"} for i in range(n)]


def _fake_response_generator(answer_template):
    """A stand-in ResponseGenerator class whose generate() re-answers each failed
    instruction with `answer_template.format(ins=...)` -- lets a test control the
    regenerated answers (e.g. "fixed::{ins}" to pass, "still-bad" to keep
    failing) without redeclaring the fake each time."""

    class FakeResponseGenerator:
        def __init__(self, model=None):
            pass

        def generate(self, instructions):
            from fastsft.helper import convert_to_distiset
            rows = [{"instruction": ins, "generation": answer_template.format(ins=ins)}
                    for ins in instructions]
            return convert_to_distiset(Dataset.from_list(rows))

    return FakeResponseGenerator


@pytest.fixture
def refiner(fake_judge):
    # score_samples keys by str(index); scripted below per-test via _scores.
    judge = fake_judge()
    return DataRefiner(parent_model=object(), judge_model=judge)


# --- _score keeps row order -------------------------------------------------

def test_score_returns_scores_in_row_order(fake_judge):
    train = _train(_rows(4))
    # Judge maps id -> score; refiner must return them positionally.
    judge = fake_judge(scores={"0": 9.0, "1": 2.0, "2": 7.0, "3": 1.0})
    refiner = DataRefiner(parent_model=object(), judge_model=judge)

    assert refiner._score(train) == [9.0, 2.0, 7.0, 1.0]


# --- _failed_instructions ---------------------------------------------------

def test_failed_instructions_selects_below_threshold_in_order():
    train = _train(_rows(5))
    scores = [9.0, 2.0, 7.0, 1.0, 4.9]
    refiner = DataRefiner(parent_model=object(), judge_model=object())

    failed = refiner._failed_instructions(train, scores, threshold=5.0)

    assert failed == ["q1", "q3", "q4"]  # exactly the <5.0 rows, in order


# --- _drop_failed: the alignment invariant ----------------------------------

def test_drop_failed_removes_below_threshold_and_keeps_alignment():
    train = _train(_rows(5))
    scores = [9.0, 2.0, 7.0, 1.0, 4.9]
    refiner = DataRefiner(parent_model=object(), judge_model=object())

    kept, kept_scores = refiner._drop_failed(train, scores, threshold=5.0)

    assert kept["instruction"] == ["q0", "q2"]  # only >=5.0 survive
    assert kept_scores == [9.0, 7.0]
    # Every surviving row still sits opposite its own score.
    for row, score in zip(kept, kept_scores, strict=True):
        idx = int(row["instruction"][1:])
        assert score == scores[idx]


@pytest.mark.parametrize(
    "scores,threshold,kept_idx",
    [
        ([9.0, 8.0, 7.0], 5.0, [0, 1, 2]),   # nothing fails
        ([1.0, 2.0, 3.0], 5.0, []),          # everything fails
        ([5.0, 4.999, 5.001], 5.0, [0, 2]),  # boundary: >= keeps, < drops
    ],
)
def test_drop_failed_boundary_and_extremes(scores, threshold, kept_idx):
    train = _train(_rows(len(scores)))
    refiner = DataRefiner(parent_model=object(), judge_model=object())

    kept, kept_scores = refiner._drop_failed(train, scores, threshold)

    assert kept["instruction"] == [f"q{i}" for i in kept_idx]
    assert kept_scores == [scores[i] for i in kept_idx]


# --- refine() end to end (fake judge + fake regeneration) -------------------

def test_refine_returns_original_distiset_when_nothing_fails(
    sample_refine_distiset, fake_judge
):
    distiset, _ = sample_refine_distiset(3)
    judge = fake_judge(scores={"0": 9.0, "1": 8.0, "2": 7.0})
    refiner = DataRefiner(parent_model=object(), judge_model=judge)

    out = refiner.refine(distiset, threshold=5.0)

    assert out["default"]["train"]["instruction"] == ["q0", "q1", "q2"]


def test_refine_drops_and_regenerates_failing_rows(
    monkeypatch, sample_refine_distiset, fake_judge
):
    """One row fails on the first scoring pass, then its regenerated
    replacement passes -- refine must terminate with all rows kept and the
    failed instruction preserved (re-answered), not lost."""
    import fastsft.data.refiner as refiner_mod

    distiset, _ = sample_refine_distiset(3)  # q0,q1,q2

    monkeypatch.setattr(
        refiner_mod, "ResponseGenerator", _fake_response_generator("fixed::{ins}")
    )

    # First pass: q1 fails (2.0). After drop+regen, the replacement (indexed 0
    # among fresh rows) scores 8.0 and passes, ending the loop.
    call = {"n": 0}

    def scripted(samples):
        call["n"] += 1
        if call["n"] == 1:  # initial batch of 3: q1 fails
            return {"0": 9.0, "1": 2.0, "2": 7.0}
        return dict.fromkeys(samples, 8.0)  # regenerated rows pass

    refiner = DataRefiner(parent_model=object(), judge_model=fake_judge(scores=scripted))
    out = refiner.refine(distiset, threshold=5.0)

    train = out["default"]["train"]
    instructions = list(train["instruction"])
    # q0 and q2 kept as-is; q1 dropped then re-answered (still present).
    assert sorted(instructions) == ["q0", "q1", "q2"]
    fixed = [r for r in train if r["generation"].startswith("fixed::")]
    assert [r["instruction"] for r in fixed] == ["q1"]


def test_refine_stops_after_max_iterations_if_always_failing(
    monkeypatch, sample_refine_distiset, fake_judge
):
    import fastsft.data.refiner as refiner_mod

    distiset, _ = sample_refine_distiset(2)

    monkeypatch.setattr(
        refiner_mod, "ResponseGenerator", _fake_response_generator("still-bad")
    )

    always_fail = fake_judge(scores=lambda samples: dict.fromkeys(samples, 0.0))
    refiner = DataRefiner(parent_model=object(), judge_model=always_fail)
    # Must terminate (not loop forever) even when nothing ever passes.
    out = refiner.refine(distiset, threshold=5.0)
    assert out["default"]["train"].num_rows >= 0


@pytest.fixture
def sample_refine_distiset():
    """Factory: a Distiset with n {instruction, generation} rows (q0..q{n-1})."""
    def _make(n):
        from fastsft.helper import convert_to_distiset
        train = Dataset.from_list(_rows(n))
        return convert_to_distiset(train), train
    return _make
