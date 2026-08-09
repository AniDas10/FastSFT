"""Tier-2 unit tests for fastsft.model.base (the OpenRouter-backed Model).

The network edge -- OpenRouter's catalog fetch and distilabel's LLM/pipeline --
is patched at the module boundary, so nothing here touches the network or loads
a real LLM. What's under test is Model's own logic: api-key resolution, the
once-only open-weight gate, the empty-generation asserts, instruction
precedence, and how build_llm / run_pipeline wire their collaborators.
"""

import pytest

from fastsft.model import base as base_mod
from fastsft.model.base import Model
from fastsft.model.constants import OPENROUTER_BASE_URL


@pytest.fixture(autouse=True)
def _clear_openrouter_cache():
    """_fetch_openrouter_models is lru_cached; tests patch the name, but clear
    the real cache too so nothing leaks between tests."""
    base_mod._fetch_openrouter_models.cache_clear()
    yield
    base_mod._fetch_openrouter_models.cache_clear()


class TestInit:
    def test_uses_explicit_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        model = Model(api_key="explicit-key")
        assert model._api_key == "explicit-key"

    def test_falls_back_to_env_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        model = Model()
        assert model._api_key == "env-key"

    def test_explicit_key_wins_over_env(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        model = Model(api_key="explicit-key")
        assert model._api_key == "explicit-key"

    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No OpenRouter API key"):
            Model()

    def test_stores_config(self):
        model = Model(model_id="some/model", api_key="k", temperature=0.5, max_tokens=256)
        assert model.model_id == "some/model"
        assert model._temperature == 0.5
        assert model._max_tokens == 256


class TestAssertOpenWeight:
    def _catalog(self, monkeypatch, catalog):
        monkeypatch.setattr(base_mod, "_fetch_openrouter_models", lambda: catalog)

    def test_accepts_open_weight_model(self, monkeypatch):
        self._catalog(monkeypatch, {"open/model": {"hugging_face_id": "org/open-model"}})
        model = Model(model_id="open/model", api_key="k")
        # Should not raise.
        model._assert_open_weight()

    def test_rejects_unknown_model(self, monkeypatch):
        self._catalog(monkeypatch, {"open/model": {"hugging_face_id": "org/x"}})
        model = Model(model_id="missing/model", api_key="k")
        with pytest.raises(ValueError, match="not found on OpenRouter"):
            model._assert_open_weight()

    @pytest.mark.parametrize("info", [{}, {"hugging_face_id": ""}, {"hugging_face_id": None}])
    def test_rejects_closed_weight_model(self, monkeypatch, info):
        self._catalog(monkeypatch, {"closed/model": info})
        model = Model(model_id="closed/model", api_key="k")
        with pytest.raises(ValueError, match="no hugging_face_id"):
            model._assert_open_weight()


class TestEnsureOpenWeight:
    def test_checks_once_then_latches(self, monkeypatch):
        model = Model(api_key="k")
        calls = []
        monkeypatch.setattr(
            model, "_assert_open_weight", lambda: calls.append(1)
        )
        model._ensure_open_weight()
        model._ensure_open_weight()
        assert calls == [1]
        assert model._open_weight_verified is True

    def test_skipped_when_not_enforced(self, monkeypatch):
        model = Model(api_key="k")
        model._enforce_open_weight = False
        calls = []
        monkeypatch.setattr(model, "_assert_open_weight", lambda: calls.append(1))
        model._ensure_open_weight()
        assert calls == []
        assert model._open_weight_verified is False


class TestAssertGenerationHelpers:
    @pytest.mark.parametrize(
        "method", ["assert_structured_output", "assert_generation"]
    )
    def test_passes_through_nonempty(self, method):
        model = Model(api_key="k")
        assert getattr(model, method)("hello") == "hello"

    @pytest.mark.parametrize(
        "method", ["assert_structured_output", "assert_generation"]
    )
    @pytest.mark.parametrize("empty", [None, ""])
    def test_raises_on_empty(self, method, empty):
        model = Model(api_key="k")
        with pytest.raises(RuntimeError):
            getattr(model, method)(empty)

    @pytest.mark.parametrize(
        "method", ["assert_structured_output", "assert_generation"]
    )
    def test_sample_id_in_message(self, method):
        model = Model(api_key="k")
        with pytest.raises(RuntimeError, match="row-42"):
            getattr(model, method)(None, sample_id="row-42")

    @pytest.mark.parametrize(
        "method", ["assert_structured_output", "assert_generation"]
    )
    def test_no_sample_id_context_when_absent(self, method):
        model = Model(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match="returned no"):
            getattr(model, method)(None)

    @pytest.mark.parametrize(
        "method, phrase",
        [
            ("assert_structured_output", "returned no structured output"),
            ("assert_generation", "returned no generation"),
        ],
    )
    def test_message_is_specific_to_the_method(self, method, phrase):
        # The two helpers deliberately emit distinct wording (structured-output
        # vs plain-text failure modes); pin each so swapping them would fail.
        model = Model(model_id="m", api_key="k")
        with pytest.raises(RuntimeError, match=phrase):
            getattr(model, method)(None)


class TestInstructionPrecedence:
    def test_base_default_is_empty(self):
        assert Model(api_key="k").get_instruction() == ""

    def test_custom_overrides_default(self):
        model = Model(api_key="k")
        model.set_instruction("be terse")
        assert model.get_instruction() == "be terse"

    def test_subclass_default_used_until_overridden(self):
        class Styled(Model):
            def _instruction(self):
                return "styled default"

        model = Styled(api_key="k")
        assert model.get_instruction() == "styled default"
        model.set_instruction("override")
        assert model.get_instruction() == "override"


class _RecordingLLM:
    """Records the kwargs OpenAILLM was constructed with."""

    last_kwargs = None

    def __init__(self, **kwargs):
        _RecordingLLM.last_kwargs = kwargs


class TestBuildLLM:
    def test_constructs_openai_llm_with_expected_kwargs(self, monkeypatch):
        monkeypatch.setattr(base_mod, "OpenAILLM", _RecordingLLM)
        model = Model(model_id="open/model", api_key="k", temperature=0.3, max_tokens=512)
        ensured = []
        monkeypatch.setattr(model, "_ensure_open_weight", lambda: ensured.append(1))

        schema = {"schema": object, "format": "json"}
        llm = model.build_llm(structured_output=schema)

        assert isinstance(llm, _RecordingLLM)
        assert ensured == [1]  # open-weight gate ran before constructing the LLM
        kwargs = _RecordingLLM.last_kwargs
        assert kwargs["model"] == "open/model"
        assert kwargs["base_url"] == OPENROUTER_BASE_URL
        assert kwargs["api_key"] == "k"
        assert kwargs["generation_kwargs"] == {"temperature": 0.3, "max_new_tokens": 512}
        assert kwargs["structured_output"] is schema

    def test_default_structured_output_is_none(self, monkeypatch):
        monkeypatch.setattr(base_mod, "OpenAILLM", _RecordingLLM)
        model = Model(api_key="k")
        monkeypatch.setattr(model, "_ensure_open_weight", lambda: None)
        model.build_llm()
        assert _RecordingLLM.last_kwargs["structured_output"] is None


class TestRunPipeline:
    def test_wires_and_runs_pipeline(self, monkeypatch):
        events = []

        class FakeStep:
            def __init__(self, *a, **k):
                pass

            def __rshift__(self, other):  # load_data >> task
                events.append("wired")
                return other

        class FakePipeline:
            def __init__(self, name="pipeline"):
                events.append(("pipeline", name))

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run(self, use_cache):
                events.append(("run", use_cache))
                return "DISTISET"

        monkeypatch.setattr(base_mod, "Pipeline", FakePipeline)
        monkeypatch.setattr(base_mod, "LoadDataFromDicts", FakeStep)
        monkeypatch.setattr(base_mod, "TextGeneration", FakeStep)
        monkeypatch.setattr(base_mod, "OpenAILLM", _RecordingLLM)
        detached = []
        monkeypatch.setattr(
            base_mod, "detach_stale_queue_handlers", lambda: detached.append(1)
        )

        model = Model(model_id="open/model", api_key="k")
        monkeypatch.setattr(model, "_ensure_open_weight", lambda: None)

        result = model.run_pipeline(
            [{"instruction": "hi"}], system_prompt="sys", name="my-pipe"
        )

        assert result == "DISTISET"
        assert ("pipeline", "my-pipe") in events
        assert "wired" in events
        assert ("run", False) in events  # use_cache=False
        assert detached == [1]  # stale-handler cleanup ran once
