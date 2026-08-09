"""Tier-2 unit tests for fastsft.model.judge (LLM-as-judge role).

run_pipeline (the OpenRouter edge) is stubbed to return canned Score/Verdict
rows, so what's under test is Judge's own logic: id->score / id->Verdict
mapping, the strict-below-threshold failure count, the comparison-prompt
rendering, and the open-weight exemption.
"""

import pytest

from fastsft.model.judge import Judge, Score, Verdict


def test_judge_is_open_weight_exempt():
    # Verdicts filter the dataset but never enter it.
    assert Judge._enforce_open_weight is False


def test_score_samples_maps_ids_to_scores(monkeypatch, fake_distiset):
    captured = {}

    def fake_run_pipeline(data, instruction, structured_output=None, name="p"):
        captured["structured_output"] = structured_output
        captured["instruction"] = instruction
        return fake_distiset(
            [
                {"id": "a", "generation": Score(score=7.5).model_dump_json()},
                {"id": "b", "generation": Score(score=2.0).model_dump_json()},
            ]
        )

    judge = Judge(api_key="k")
    monkeypatch.setattr(judge, "run_pipeline", fake_run_pipeline)

    scores = judge.score_samples({"a": "answer a", "b": "answer b"})

    assert scores == {"a": 7.5, "b": 2.0}
    assert captured["structured_output"]["schema"] is Score
    assert captured["structured_output"]["format"] == "json"


def test_score_samples_uses_prompt_override(monkeypatch, fake_distiset):
    captured = {}

    def fake_run_pipeline(data, instruction, structured_output=None, name="p"):
        captured["instruction"] = instruction
        return fake_distiset([{"id": "a", "generation": Score(score=1.0).model_dump_json()}])

    judge = Judge(api_key="k")
    monkeypatch.setattr(judge, "run_pipeline", fake_run_pipeline)
    judge.score_samples({"a": "x"}, prompt="custom rubric")
    assert captured["instruction"] == "custom rubric"


def test_score_samples_raises_on_empty_generation(monkeypatch, fake_distiset):
    judge = Judge(api_key="k")
    monkeypatch.setattr(
        judge,
        "run_pipeline",
        lambda *a, **k: fake_distiset([{"id": "a", "generation": ""}]),
    )
    with pytest.raises(RuntimeError, match="'a'"):
        judge.score_samples({"a": "x"})


@pytest.mark.parametrize(
    "scores, threshold, expected",
    [
        ([1.0, 2.0, 9.0], 5.0, 2),
        ([5.0, 5.0], 5.0, 0),  # strict <: exactly at threshold is not a failure
        ([4.999, 5.0, 5.001], 5.0, 1),
        ([], 5.0, 0),
        ([1.0, 2.0, 3.0], 2.5, 2),  # custom threshold
    ],
)
def test_failed_sample_count_below_threshold(scores, threshold, expected):
    assert Judge(api_key="k").failed_sample_count(scores, threshold=threshold) == expected


def test_failed_sample_count_default_threshold_is_five():
    assert Judge(api_key="k").failed_sample_count([4.0, 6.0]) == 1


def _verdict_run(fake_distiset, mapping):
    """A run_pipeline stub that echoes a scripted id->winner mapping."""

    def fake_run_pipeline(data, instruction, structured_output=None, name="p"):
        return fake_distiset(
            [
                {"id": row["id"], "generation": Verdict(winner=mapping[row["id"]]).model_dump_json()}
                for row in data
            ]
        )

    return fake_run_pipeline


def test_compare_samples_maps_ids_to_verdicts(monkeypatch, fake_distiset):
    judge = Judge(api_key="k")
    monkeypatch.setattr(
        judge, "run_pipeline", _verdict_run(fake_distiset, {"1": "A", "2": "tie"})
    )
    result = judge.compare_samples(
        {"1": ("q1", "ans a", "ans b"), "2": ("q2", "ans a", "ans b")}
    )
    assert result == {"1": Verdict(winner="A"), "2": Verdict(winner="tie")}


def test_compare_to_reference_maps_ids_to_verdicts(monkeypatch, fake_distiset):
    judge = Judge(api_key="k")
    monkeypatch.setattr(
        judge, "run_pipeline", _verdict_run(fake_distiset, {"1": "B"})
    )
    result = judge.compare_to_reference({"1": ("q", "ref", "ans a", "ans b")})
    assert result == {"1": Verdict(winner="B")}


def test_comparison_prompt_renders_question_and_both_answers():
    prompt = Judge(api_key="k")._comparison_prompt("What is a knot?", "AAA", "BBB")
    assert "What is a knot?" in prompt
    assert "AAA" in prompt
    assert "BBB" in prompt
    assert "Response A" in prompt
    assert "Response B" in prompt


def test_reference_prompt_renders_reference_and_both_answers():
    prompt = Judge(api_key="k")._reference_prompt(
        "What is a knot?", "REF", "AAA", "BBB"
    )
    assert "What is a knot?" in prompt
    assert "REF" in prompt
    assert "AAA" in prompt
    assert "BBB" in prompt
    assert "Reference response" in prompt
    assert "Response A" in prompt
    assert "Response B" in prompt
