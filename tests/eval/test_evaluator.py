"""Tier-1 unit tests for the win-rate math in fastsft.eval.evaluator.

This is the headline metric, so it's the first thing pinned down. Also serves as
the first consumer of the conftest fakes (FakeJudge + make_eval_config).
"""

import pytest

from fastsft.eval.evaluator import Evaluator


@pytest.mark.parametrize(
    "winner,a_label,expected",
    [
        ("A", "A", 1.0),   # a held slot A and won
        ("B", "A", 0.0),   # a held slot A and lost
        ("A", "B", 0.0),   # a held slot B, A won -> a lost
        ("B", "B", 1.0),   # a held slot B and won
        ("tie", "A", 0.5),
        ("tie", "B", 0.5),
    ],
)
def test_credit(winner, a_label, expected):
    assert Evaluator._credit(winner, a_label) == expected


def test_win_rate_no_swap_counts_wins_ties_losses(fake_judge, make_eval_config):
    prompts = ["p0", "p1", "p2"]
    a, b = ["a0", "a1", "a2"], ["b0", "b1", "b2"]
    # id 0 -> a wins, id 1 -> b wins, id 2 -> tie.
    judge = fake_judge(compare=lambda pairs: {"0": "A", "1": "B", "2": "tie"})
    evaluator = Evaluator(make_eval_config(swap_positions=False), verbose=False)

    result = evaluator._win_rate(judge, prompts, a, b)

    assert result == {
        "wins": 1, "ties": 1, "losses": 1,
        "win_rate": pytest.approx(0.5),  # (1.0 + 0.0 + 0.5) / 3
        "orders_judged": 1,
    }


def test_win_rate_swap_gives_full_credit_when_answer_wins_both_orders(
    fake_judge, make_eval_config
):
    prompts, a, b = ["p0", "p1"], ["a0", "a1"], ["b0", "b1"]
    a_set = set(a)
    # Content-based, position-independent judge: whichever slot holds an
    # a-answer wins. `pairs[i]` is (question, first, second) -> first is [1].
    judge = fake_judge(
        compare=lambda pairs: {i: ("A" if pairs[i][1] in a_set else "B") for i in pairs}
    )
    evaluator = Evaluator(make_eval_config(swap_positions=True), verbose=False)

    result = evaluator._win_rate(judge, prompts, a, b)

    assert result["win_rate"] == pytest.approx(1.0)
    assert (result["wins"], result["ties"], result["losses"]) == (2, 0, 0)
    assert result["orders_judged"] == 2


def test_win_rate_swap_cancels_a_position_biased_judge(fake_judge, make_eval_config):
    # A judge that always picks slot "A" regardless of content should net to 0.5
    # (pure tie) once positions are swapped -- the whole point of debiasing.
    prompts, a, b = ["p0", "p1"], ["a0", "a1"], ["b0", "b1"]
    judge = fake_judge(compare=lambda pairs: dict.fromkeys(pairs, "A"))
    evaluator = Evaluator(make_eval_config(swap_positions=True), verbose=False)

    result = evaluator._win_rate(judge, prompts, a, b)

    assert result["win_rate"] == pytest.approx(0.5)
    assert result["ties"] == len(prompts)
