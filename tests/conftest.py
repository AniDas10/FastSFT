"""Shared test fixtures and fakes for the FastSFT suite.

Because the pipeline injects its collaborators (Model / Judge / inference
engine), tests prefer hand-written FAKES over patching the network or GPU --
they read clearly, enforce the real interface, and survive refactors. The fakes
live here once; tests request them by fixture name.

Heavy imports (datasets, distilabel via the model layer, pydantic Verdict) are
kept out of module top-level and pulled in lazily inside each fixture/method,
so the pure-logic test tiers run without loading the ML import graph.
"""

import pytest

# --- Live-test gating ---------------------------------------------------------
# Tiers 0-3 fake every external edge, so the default run is fully hermetic.
# Tier 4 tests that make REAL OpenRouter calls / train locally are marked
# `@pytest.mark.live` and skipped unless `--run-live` is passed, so `pytest`
# (and CI) never bills a provider or spins a GPU by accident. Run them with:
#   pytest --run-live


def pytest_addoption(parser):
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run @pytest.mark.live tests (real OpenRouter calls / local "
        "training). Off by default.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-live"):
        return
    skip_live = pytest.mark.skip(reason="live test; pass --run-live to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


class FakeJudge:
    """Stand-in for fastsft.model.judge.Judge with scripted results -- no
    OpenRouter calls. Implements only the surface the pipeline uses.

    - `scores`: either a {sample_id: float} mapping returned by score_samples,
      or a callable(samples) -> {sample_id: float} for per-call behavior (e.g. a
      judge that fails a row on the first pass then passes its regeneration, or
      one that fails everything). Receiving `samples` lets a test key its reply
      to the ids it was handed.
    - `compare`: callable(pairs) -> {id: "A"|"B"|"tie"} driving the pairwise
      comparators; omitted means every pair is a "tie". Receiving `pairs` lets a
      test script order-dependent verdicts (to exercise A/B position-swapping).
    """

    def __init__(self, scores=None, compare=None):
        self._scores = scores if scores is not None else {}
        self._compare = compare

    def score_samples(self, samples, prompt=None):
        if callable(self._scores):
            return self._scores(samples)
        return {sid: self._scores[sid] for sid in samples}

    def failed_sample_count(self, scores, threshold=5.0):
        return sum(1 for score in scores if score < threshold)

    def compare_samples(self, pairs, prompt=None):
        return self._verdicts(pairs)

    def compare_to_reference(self, pairs, prompt=None):
        return self._verdicts(pairs)

    def _verdicts(self, pairs):
        from fastsft.model.judge import Verdict

        winners = self._compare(pairs) if self._compare else dict.fromkeys(pairs, "tie")
        return {i: Verdict(winner=w) for i, w in winners.items()}


class FakeInferenceEngine:
    """Stand-in for fastsft.eval.inference.ChildInferenceEngine: scripted
    tuned/untuned answers, no torch/peft load. `tuned`/`untuned` are the answer
    lists returned in prompt order (their length should match the prompt set)."""

    def __init__(self, tuned, untuned):
        self._tuned, self._untuned = list(tuned), list(untuned)

    def generate_tuned(self, prompts):
        return list(self._tuned)

    def generate_untuned(self, prompts):
        return list(self._untuned)


@pytest.fixture
def fake_judge():
    """The FakeJudge class; call it with scripted `scores`/`compare`."""
    return FakeJudge


@pytest.fixture
def fake_inference_engine():
    """The FakeInferenceEngine class; call it with `tuned`/`untuned` answers."""
    return FakeInferenceEngine


@pytest.fixture
def make_eval_config():
    """Factory for an EvalConfig with sensible test defaults and overrides,
    e.g. make_eval_config(swap_positions=False)."""

    def _make(**overrides):
        from fastsft.eval.config import EvalConfig

        params = {"adapter_dir": "modelsets/test-run"}
        params.update(overrides)
        return EvalConfig(**params)

    return _make


@pytest.fixture
def sample_messages():
    """A couple of chat conversations in the `messages` schema (pure Python)."""
    return [
        [{"role": "user", "content": "what is a knot?"},
         {"role": "assistant", "content": "Arr, a knot be a nautical mile per hour."}],
        [{"role": "user", "content": "hoist the sail"},
         {"role": "assistant", "content": "Aye, hoisting now, matey."}],
    ]


@pytest.fixture
def sample_distiset(sample_messages):
    """A Distiset with a `messages` column -- the shape stages hand each other."""
    from datasets import Dataset

    from fastsft.helper import convert_to_distiset

    return convert_to_distiset(Dataset.from_dict({"messages": sample_messages}))


@pytest.fixture
def formatted_distiset():
    """A Distiset with a rendered `text` column -- FineTuner's expected input."""
    from datasets import Dataset

    from fastsft.helper import convert_to_distiset

    texts = ["<|user|>what is a knot?<|assistant|>Arr, a knot be a nautical mile per hour."]
    return convert_to_distiset(Dataset.from_dict({"text": texts}))
