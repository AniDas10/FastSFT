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


# --- Tier 3: Evaluator.run() integration ---------------------------------
# Drives the REAL run()/_run/_parent_answers/_similarity/_samples wiring with
# every collaborator faked at its module boundary (no OpenRouter, torch, or
# sentence-transformers). _win_rate math is already unit-tested above, so these
# assert the assembled results dict and the orchestration around it.


def _wire(monkeypatch, *, prompts, parent, tuned, untuned, sims, compare=None):
    """Patch every Evaluator collaborator at its import boundary. Returns the
    list of parent Model instances built during the run (for set_instruction
    assertions)."""
    created_models = []
    parent_by_prompt = dict(zip(prompts, parent, strict=True))

    def verdicts(pairs):
        from fastsft.model.judge import Verdict

        winners = compare(pairs) if compare else dict.fromkeys(pairs, "tie")
        return {i: Verdict(winner=w) for i, w in winners.items()}

    class _Model:
        def __init__(self, model_id=None, temperature=None, max_tokens=None):
            self.model_id = model_id
            self.set_instruction_calls = []
            created_models.append(self)

        def set_instruction(self, instruction):
            self.set_instruction_calls.append(instruction)

        def assert_generation(self, generation, sample_id=None):
            if not generation:
                raise RuntimeError(f"empty generation for {sample_id!r}")
            return generation

    class _ResponseGenerator:
        def __init__(self, model=None):
            self._model = model

        def generate(self, prompts):
            # Rows deliberately in a different order than `prompts` to exercise
            # _parent_answers' re-key-by-instruction (order-robustness).
            rows = [
                {"instruction": p, "generation": parent_by_prompt[p]}
                for p in reversed(prompts)
            ]
            return {"default": {"train": rows}}

    class _Engine:
        def __init__(self, adapter_dir, max_new_tokens=None, batch_size=None):
            self.adapter_dir = adapter_dir

        def generate_tuned(self, prompts):
            return list(tuned)

        def generate_untuned(self, prompts):
            return list(untuned)

    class _Judge:
        def __init__(self, model_id=None):
            self.model_id = model_id

        def compare_samples(self, pairs, prompt=None):
            return verdicts(pairs)

        def compare_to_reference(self, pairs, prompt=None):
            return verdicts(pairs)

    monkeypatch.setattr("fastsft.eval.evaluator.Model", _Model)
    monkeypatch.setattr(
        "fastsft.data.response_generator.ResponseGenerator", _ResponseGenerator
    )
    monkeypatch.setattr("fastsft.eval.evaluator.ChildInferenceEngine", _Engine)
    monkeypatch.setattr("fastsft.eval.evaluator.Judge", _Judge)
    monkeypatch.setattr(
        "fastsft.eval.evaluator.pairwise_similarities",
        lambda a, b, model_id: list(sims),
    )
    return created_models


def test_run_requires_non_empty_prompt_set(make_eval_config):
    with pytest.raises(ValueError, match="non-empty prompt set"):
        Evaluator(make_eval_config(), verbose=False).run([])


def test_run_assembles_full_results_dict(monkeypatch, make_eval_config):
    prompts = ["p0", "p1"]
    _wire(
        monkeypatch,
        prompts=prompts,
        parent=["par0", "par1"],
        tuned=["t0", "t1"],
        untuned=["u0", "u1"],
        sims=[0.8, 0.6],
    )
    config = make_eval_config(
        adapter_dir="modelsets/run-x",
        parent_model="parent/model",
        judge_model="judge/model",
        embedding_model="embed/model",
        swap_positions=True,
    )

    result = Evaluator(config, verbose=False).run(prompts)

    assert result["adapter_dir"] == "modelsets/run-x"
    assert result["parent_model"] == "parent/model"
    assert result["judge_model"] == "judge/model"
    assert result["embedding_model"] == "embed/model"
    assert result["num_prompts"] == 2
    assert result["swap_positions"] is True
    assert set(result["comparisons"]) == {
        "tuned_vs_untuned",
        "parent_likeness",
        "tuned_vs_parent",
    }
    for block in result["comparisons"].values():
        assert set(block) == {"wins", "ties", "losses", "win_rate", "orders_judged"}
    # Both similarity calls return `sims`; mean([0.8, 0.6]) == 0.7.
    assert result["similarity_to_parent"] == {
        "tuned_vs_parent": pytest.approx(0.7),
        "untuned_vs_parent": pytest.approx(0.7),
    }
    assert result["samples"] == [
        {"prompt": "p0", "parent": "par0", "tuned": "t0", "untuned": "u0"},
        {"prompt": "p1", "parent": "par1", "tuned": "t1", "untuned": "u1"},
    ]


def test_run_tuned_beats_untuned_gives_full_win_rate(monkeypatch, make_eval_config):
    prompts = ["p0", "p1", "p2"]
    _wire(
        monkeypatch,
        prompts=prompts,
        parent=["r0", "r1", "r2"],
        tuned=["t0", "t1", "t2"],
        untuned=["u0", "u1", "u2"],
        sims=[1.0, 1.0, 1.0],
        # tuned always in slot "A" (swap off) and always wins.
        compare=lambda pairs: dict.fromkeys(pairs, "A"),
    )
    config = make_eval_config(swap_positions=False)

    result = Evaluator(config, verbose=False).run(prompts)

    assert result["comparisons"]["tuned_vs_untuned"]["win_rate"] == pytest.approx(1.0)


def test_run_all_ties_gives_half_win_rate(monkeypatch, make_eval_config):
    prompts = ["p0", "p1"]
    _wire(
        monkeypatch,
        prompts=prompts,
        parent=["r0", "r1"],
        tuned=["t0", "t1"],
        untuned=["u0", "u1"],
        sims=[0.5, 0.5],
        compare=lambda pairs: dict.fromkeys(pairs, "tie"),
    )

    result = Evaluator(make_eval_config(swap_positions=False), verbose=False).run(prompts)

    assert result["comparisons"]["tuned_vs_untuned"]["win_rate"] == pytest.approx(0.5)


def test_run_caps_samples_at_three(monkeypatch, make_eval_config):
    prompts = [f"p{i}" for i in range(5)]
    _wire(
        monkeypatch,
        prompts=prompts,
        parent=[f"par{i}" for i in range(5)],
        tuned=[f"t{i}" for i in range(5)],
        untuned=[f"u{i}" for i in range(5)],
        sims=[0.5] * 5,
    )

    result = Evaluator(make_eval_config(), verbose=False).run(prompts)

    assert len(result["samples"]) == 3
    assert [s["prompt"] for s in result["samples"]] == ["p0", "p1", "p2"]
    for sample in result["samples"]:
        assert set(sample) == {"prompt", "parent", "tuned", "untuned"}


def test_run_sets_parent_instruction_when_configured(monkeypatch, make_eval_config):
    prompts = ["p0"]
    models = _wire(
        monkeypatch,
        prompts=prompts,
        parent=["par0"],
        tuned=["t0"],
        untuned=["u0"],
        sims=[0.5],
    )
    config = make_eval_config(parent_instruction="Answer like a pirate.")

    Evaluator(config, verbose=False).run(prompts)

    assert len(models) == 1
    assert models[0].set_instruction_calls == ["Answer like a pirate."]


def test_run_skips_parent_instruction_when_absent(monkeypatch, make_eval_config):
    prompts = ["p0"]
    models = _wire(
        monkeypatch,
        prompts=prompts,
        parent=["par0"],
        tuned=["t0"],
        untuned=["u0"],
        sims=[0.5],
    )

    Evaluator(make_eval_config(parent_instruction=""), verbose=False).run(prompts)

    assert models[0].set_instruction_calls == []
