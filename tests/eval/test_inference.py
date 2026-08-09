"""Tier-2 unit tests for fastsft.eval.inference.ChildInferenceEngine -- the
batching, chat-templating, left-padding setup, and adapter-toggle logic around
child generation, with torch/peft/transformers always faked out (this module's
whole point is to keep those heavy imports deferred, so tests keep them
deferred too -- none of torch/peft/transformers needs to be installed).

Two groups:
- generate_tuned/generate_untuned/_generate/_generate_batch: _load() is
  bypassed by injecting fake _model/_tokenizer directly (exactly what _load()
  would leave in place), so these exercise the real batching/toggle/slicing
  logic against scripted fakes.
- _load(): stubs peft/transformers/torch via sys.modules to exercise the
  pad-token/padding-side setup and idempotency without a real ML import graph.
"""

import sys
import types
from contextlib import contextmanager

import pytest

from fastsft.eval.inference import ChildInferenceEngine

# --- shared fakes ------------------------------------------------------------


class _FakeTensor:
    """Minimal 2D-tensor stand-in: just enough shape/slicing to exercise
    _generate_batch's prompt-length slicing without real torch."""

    def __init__(self, rows):
        self.rows = [list(r) for r in rows]

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def __getitem__(self, key):
        row_sel, col_sel = key
        assert row_sel == slice(None)
        return _FakeTensor([row[col_sel] for row in self.rows])


class _FakeBatch(dict):
    """Stand-in for the BatchEncoding apply_chat_template returns: dict-like
    (so `**inputs` works in model.generate) plus a `.to(device)` that records
    the device it was moved to."""

    def __init__(self, data, to_calls):
        super().__init__(data)
        self._to_calls = to_calls

    def to(self, device):
        self._to_calls.append(device)
        return self


class _FakeTokenizer:
    def __init__(self, pad_token=None, eos_token="<eos>", prompt_len=4):
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.padding_side = None
        self.pad_token_id = 0
        self.prompt_len = prompt_len
        self.chat_template_calls = []
        self.to_calls = []
        self.batch_decode_calls = []

    def apply_chat_template(self, conversations, **kwargs):
        self.chat_template_calls.append((conversations, kwargs))
        rows = [[1] * self.prompt_len for _ in conversations]
        return _FakeBatch({"input_ids": _FakeTensor(rows)}, self.to_calls)

    def batch_decode(self, tokens, skip_special_tokens=None):
        self.batch_decode_calls.append((tokens, skip_special_tokens))
        return [f" answer-{i} " for i in range(tokens.shape[0])]


class _FakeModel:
    def __init__(self):
        self.device = "fake-device"
        self.eval_calls = 0
        self.disable_adapter_calls = 0
        self.generate_calls = []
        self.new_tokens = 3

    def eval(self):
        self.eval_calls += 1

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        n, prompt_len = kwargs["input_ids"].shape
        rows = [[9] * (prompt_len + self.new_tokens) for _ in range(n)]
        return _FakeTensor(rows)

    @contextmanager
    def disable_adapter(self):
        self.disable_adapter_calls += 1
        yield


def _fake_torch_module():
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bf16"
    torch.float32 = "fp32"

    @contextmanager
    def no_grad():
        yield

    torch.no_grad = no_grad
    return torch


@pytest.fixture(autouse=True)
def _fake_torch(monkeypatch):
    """_generate_batch does `import torch` for torch.no_grad(); torch isn't
    installed by default (evaluation extra), so every test here injects a
    fake rather than requiring it."""
    monkeypatch.setitem(sys.modules, "torch", _fake_torch_module())


@pytest.fixture(autouse=True)
def _stub_detect_device(monkeypatch):
    # __init__ calls detect_device(), which itself imports torch; stub it so
    # constructing an engine doesn't depend on real torch/GPU probing.
    monkeypatch.setattr("fastsft.eval.inference.detect_device", lambda: "cpu")


def _engine(model, tokenizer, **overrides):
    """A ChildInferenceEngine with _load() bypassed: _model/_tokenizer set
    directly, exactly as _load() would leave them, so generate_tuned/
    generate_untuned exercise the real batching/toggle logic against fakes."""
    params = {"adapter_dir": "fake/adapter"}
    params.update(overrides)
    engine = ChildInferenceEngine(**params)
    engine._model = model
    engine._tokenizer = tokenizer
    return engine


# --- batching ------------------------------------------------------------


def test_generate_batches_prompts_by_configured_batch_size():
    model, tokenizer = _FakeModel(), _FakeTokenizer()
    engine = _engine(model, tokenizer, batch_size=2)

    answers = engine.generate_tuned(["p0", "p1", "p2", "p3", "p4"])

    # 5 prompts at batch_size=2 -> batches of [2, 2, 1].
    batch_sizes = [len(conversations) for conversations, _ in tokenizer.chat_template_calls]
    assert batch_sizes == [2, 2, 1]
    assert len(model.generate_calls) == 3
    assert len(answers) == 5


def test_generate_single_batch_when_under_batch_size():
    model, tokenizer = _FakeModel(), _FakeTokenizer()
    engine = _engine(model, tokenizer, batch_size=8)

    engine.generate_tuned(["p0", "p1"])

    assert len(tokenizer.chat_template_calls) == 1
    assert len(model.generate_calls) == 1


# --- adapter toggling ------------------------------------------------------


def test_generate_untuned_disables_the_adapter():
    model, tokenizer = _FakeModel(), _FakeTokenizer()
    engine = _engine(model, tokenizer)

    engine.generate_untuned(["p0"])

    assert model.disable_adapter_calls == 1


def test_generate_tuned_leaves_the_adapter_applied():
    model, tokenizer = _FakeModel(), _FakeTokenizer()
    engine = _engine(model, tokenizer)

    engine.generate_tuned(["p0"])

    assert model.disable_adapter_calls == 0


# --- chat templating + device placement -------------------------------------


def test_conversations_are_single_user_turn_with_no_system_prompt():
    model, tokenizer = _FakeModel(), _FakeTokenizer()
    engine = _engine(model, tokenizer)

    engine.generate_tuned(["hello"])

    conversations, _ = tokenizer.chat_template_calls[0]
    assert conversations == [[{"role": "user", "content": "hello"}]]


def test_chat_template_requests_padding_and_generation_prompt():
    model, tokenizer = _FakeModel(), _FakeTokenizer()
    engine = _engine(model, tokenizer)

    engine.generate_tuned(["p0"])

    _, kwargs = tokenizer.chat_template_calls[0]
    assert kwargs["padding"] is True
    assert kwargs["add_generation_prompt"] is True
    assert kwargs["return_dict"] is True


def test_batch_inputs_are_moved_to_the_model_device():
    model, tokenizer = _FakeModel(), _FakeTokenizer()
    model.device = "cuda:0"
    engine = _engine(model, tokenizer, batch_size=2)

    engine.generate_tuned(["p0", "p1", "p2"])

    # One batch of 2 + one batch of 1 -> .to() called once per batch.
    assert tokenizer.to_calls == ["cuda:0", "cuda:0"]


# --- generation call + decode slicing -------------------------------------


def test_generate_uses_greedy_decoding_and_configured_max_new_tokens():
    model, tokenizer = _FakeModel(), _FakeTokenizer()
    engine = _engine(model, tokenizer, max_new_tokens=77)

    engine.generate_tuned(["p0"])

    call = model.generate_calls[0]
    assert call["do_sample"] is False
    assert call["max_new_tokens"] == 77
    assert call["pad_token_id"] == tokenizer.pad_token_id


def test_decode_only_sees_tokens_generated_past_the_prompt():
    model, tokenizer = _FakeModel(), _FakeTokenizer(prompt_len=4)
    model.new_tokens = 3
    engine = _engine(model, tokenizer)

    engine.generate_tuned(["p0", "p1"])

    decoded_tokens, skip_special = tokenizer.batch_decode_calls[0]
    assert decoded_tokens.shape == (2, 3)  # prompt tokens sliced off
    assert skip_special is True


def test_answers_are_stripped_of_decode_whitespace():
    model, tokenizer = _FakeModel(), _FakeTokenizer()
    engine = _engine(model, tokenizer)

    answers = engine.generate_tuned(["p0", "p1"])

    assert answers == ["answer-0", "answer-1"]  # no leading/trailing spaces


# --- _load(): pad-token/padding-side setup + idempotency ---------------------


def _fake_peft_module(base_model_id, made_models):
    peft = types.ModuleType("peft")

    class _FakePeftConfig:
        def __init__(self, base_model_name_or_path):
            self.base_model_name_or_path = base_model_name_or_path

    class PeftConfig:
        @staticmethod
        def from_pretrained(adapter_dir):
            return _FakePeftConfig(base_model_id)

    class PeftModel:
        @staticmethod
        def from_pretrained(base, adapter_dir):
            model = _FakeModel()
            made_models.append((base, adapter_dir, model))
            return model

    peft.PeftConfig = PeftConfig
    peft.PeftModel = PeftModel
    return peft


def _fake_transformers_module(tokenizer_factory, made_bases):
    transformers = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(adapter_dir):
            return tokenizer_factory()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(base_id, dtype=None, device_map=None):
            base = object()
            made_bases.append((base_id, dtype, device_map, base))
            return base

    transformers.AutoTokenizer = AutoTokenizer
    transformers.AutoModelForCausalLM = AutoModelForCausalLM
    return transformers


def _wire_load(monkeypatch, tokenizer_factory, base_model_id="base/model-id"):
    made_models, made_bases = [], []
    monkeypatch.setitem(sys.modules, "peft", _fake_peft_module(base_model_id, made_models))
    monkeypatch.setitem(
        sys.modules, "transformers", _fake_transformers_module(tokenizer_factory, made_bases)
    )
    monkeypatch.setattr("fastsft.eval.inference.dtype_for_device", lambda device: "fp32")
    return made_models, made_bases


def test_load_fills_missing_pad_token_from_eos(monkeypatch):
    tokenizer = _FakeTokenizer(pad_token=None, eos_token="<eos>")
    _wire_load(monkeypatch, lambda: tokenizer)

    engine = ChildInferenceEngine(adapter_dir="adapter/dir")
    engine._load()

    assert engine._tokenizer.pad_token == "<eos>"
    assert engine._tokenizer.padding_side == "left"


def test_load_preserves_an_already_set_pad_token(monkeypatch):
    tokenizer = _FakeTokenizer(pad_token="<pad>", eos_token="<eos>")
    _wire_load(monkeypatch, lambda: tokenizer)

    engine = ChildInferenceEngine(adapter_dir="adapter/dir")
    engine._load()

    assert engine._tokenizer.pad_token == "<pad>"
    assert engine._tokenizer.padding_side == "left"


def test_load_recovers_base_model_id_from_the_adapter_dir(monkeypatch):
    _, made_bases = _wire_load(monkeypatch, _FakeTokenizer, base_model_id="parent/base-model")

    engine = ChildInferenceEngine(adapter_dir="adapter/dir")
    engine._load()

    assert made_bases[0][0] == "parent/base-model"


def test_load_is_idempotent_across_repeated_calls(monkeypatch):
    call_count = {"tokenizer": 0}

    def tokenizer_factory():
        call_count["tokenizer"] += 1
        return _FakeTokenizer()

    made_models, made_bases = _wire_load(monkeypatch, tokenizer_factory)

    engine = ChildInferenceEngine(adapter_dir="adapter/dir")
    engine._load()
    engine._load()

    assert call_count["tokenizer"] == 1
    assert len(made_bases) == 1
    assert len(made_models) == 1
    assert made_models[0][2].eval_calls == 1


def test_load_puts_the_model_in_eval_mode(monkeypatch):
    made_models, _ = _wire_load(monkeypatch, _FakeTokenizer)

    engine = ChildInferenceEngine(adapter_dir="adapter/dir")
    engine._load()

    assert made_models[0][2].eval_calls == 1
