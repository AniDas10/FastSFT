"""Tier-2 unit tests for fastsft.model.guide (Guide role).

run_pipeline (the distilabel/OpenRouter edge) is stubbed to return a canned
structured-output row, so what's under test is Guide's own logic: parsing the
GuideInstructions JSON, the num_seeds-formatted system prompt it sends, and the
open-weight exemption for an instruction-producing role.
"""

import pytest

from fastsft.model.guide import Guide, GuideInstructions


def test_guide_is_open_weight_exempt():
    # Guide output shapes instructions, not training data.
    assert Guide._enforce_open_weight is False


def test_generate_instructions_parses_structured_output(monkeypatch, fake_distiset):
    payload = GuideInstructions(
        parent_instruction="answer like a pirate",
        judge_instruction="score pirate-ness 0-10",
        sample_instructions=["what is a knot?", "how to tie a bowline?"],
    )
    captured = {}

    def fake_run_pipeline(data, system_prompt, structured_output=None, name="pipeline"):
        captured["data"] = data
        captured["system_prompt"] = system_prompt
        captured["structured_output"] = structured_output
        captured["name"] = name
        return fake_distiset([{"generation": payload.model_dump_json()}])

    guide = Guide(api_key="k")
    monkeypatch.setattr(guide, "run_pipeline", fake_run_pipeline)

    result = guide.generate_instructions("make pirate answers", num_seeds=2)

    assert isinstance(result, GuideInstructions)
    assert result.parent_instruction == "answer like a pirate"
    assert result.judge_instruction == "score pirate-ness 0-10"
    assert result.sample_instructions == ["what is a knot?", "how to tie a bowline?"]


def test_generate_instructions_formats_num_seeds_into_prompt(monkeypatch, fake_distiset):
    payload = GuideInstructions(
        parent_instruction="p", judge_instruction="j", sample_instructions=["a"] * 5
    )
    captured = {}

    def fake_run_pipeline(data, system_prompt, structured_output=None, name="pipeline"):
        captured["system_prompt"] = system_prompt
        captured["structured_output"] = structured_output
        captured["data"] = data
        return fake_distiset([{"generation": payload.model_dump_json()}])

    guide = Guide(api_key="k")
    monkeypatch.setattr(guide, "run_pipeline", fake_run_pipeline)

    guide.generate_instructions("user request", num_seeds=5)

    # The default guide instruction has a `{num_seeds}` placeholder that must be
    # rendered to the requested count before it's sent as the system prompt.
    assert "5" in captured["system_prompt"]
    assert "{num_seeds}" not in captured["system_prompt"]
    assert captured["data"] == [{"instruction": "user request"}]
    assert captured["structured_output"]["schema"] is GuideInstructions
    assert captured["structured_output"]["format"] == "json"


def test_generate_instructions_raises_on_empty_generation(monkeypatch, fake_distiset):
    guide = Guide(api_key="k")
    monkeypatch.setattr(
        guide,
        "run_pipeline",
        lambda *a, **k: fake_distiset([{"generation": None}]),
    )
    with pytest.raises(RuntimeError, match="no structured output"):
        guide.generate_instructions("req", num_seeds=1)
