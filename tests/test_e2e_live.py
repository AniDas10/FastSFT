"""Tier 4 -- live end-to-end SMOKE test (real OpenRouter + local train + eval).

This is a WIRING proof, not a quality benchmark. It runs the whole pipeline --
guide/generate/refine over OpenRouter, local LoRA training on the child, then
evaluation (local tuned/untuned inference + OpenRouter judge + local embeddings)
-- on a deliberately TINY dataset for ONE epoch, so it finishes in a few minutes
and costs cents. It asserts the machine runs end to end and every artifact/shape
is valid. It does NOT assert the child convincingly adopts the mathematician
voice or that win-rates move: at ~8 training samples and ~6 eval prompts those
numbers are pure noise (far below the sample-size floor). Proving quality is a
separate, deliberately larger MANUAL run -- see the command at the bottom.

The whole module is `@pytest.mark.live` (via `pytestmark`), so it is skipped by
default. To run it:

    OPENROUTER_API_KEY=... uv run --extra local-training --extra evaluation \\
        pytest --run-live tests/test_e2e_live.py -s

Needs OPENROUTER_API_KEY and the local-training + evaluation extras; it skips
cleanly (not fails) if either is missing, even under --run-live.
"""

import os

import pytest

pytestmark = pytest.mark.live

# The voice/tone we distill into the child (phase-0 target: style, not reasoning
# or format). Short styled answers stay well under the token caps. The Guide
# turns this into the parent's system prompt, the judge's rubric, and the seed
# topics; eval reuses it as the styled-teacher reference for parent-likeness.
MATHEMATICIAN_PROMPT = (
    "Respond as a distinguished mathematician scholar: precise and rigorous, "
    "defining terms before using them, reasoning in clear logical steps, and "
    "occasionally noting historical or philosophical context -- always in formal, "
    "elegant prose."
)

# Smoke-scale knobs: small enough to be cheap/fast, big enough that the train/eval
# split and the judge comparisons have something to chew on.
NUM_SAMPLES = 8
NUM_EVAL_PROMPTS = 6


def _require_live_env():
    """Skip (not fail) unless this machine can actually run the live path."""
    # Mirror the app: the key may live in .env rather than the exported env, and
    # our guard runs before any fastsft import loads it.
    from dotenv import load_dotenv

    load_dotenv()
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("live e2e needs OPENROUTER_API_KEY (export it or put it in .env)")
    pytest.importorskip("torch", reason="live e2e needs the local-training/evaluation extras")
    pytest.importorskip("peft")
    pytest.importorskip("trl")
    pytest.importorskip("sentence_transformers")


@pytest.fixture(scope="module")
def distilled(tmp_path_factory):
    """Runs DataGenerator -> DataFormatter -> FineTuner once, exactly as the CLI
    does (each stage's output saved under a shared run id in a throwaway CWD), and
    returns the stage outputs + the on-disk adapter path. Module-scoped: this is
    the expensive step, shared by every assertion below."""
    _require_live_env()

    from fastsft.data.config import DataGenerationConfig
    from fastsft.helper import current_timestamp
    from fastsft.pipeline import DistillationPipeline
    from fastsft.stages.constants import DATA_FORMATTER, FINE_TUNER
    from fastsft.training.config import (
        AdapterConfig,
        TrainingConfig,
        TrainingLoopConfig,
    )

    generation = DataGenerationConfig(num_samples=NUM_SAMPLES)
    # gpu_tier "local" + local_training=True trains on this machine; one epoch,
    # tiny batch, a big-enough validation slice for early stopping to have >=1 row.
    training = TrainingConfig(
        gpu_tier="local",
        adapter=AdapterConfig(),
        loop=TrainingLoopConfig(
            max_epochs=1,
            batch_size=2,
            eval_steps=5,
            early_stopping_patience=1,
            validation_split=0.25,
        ),
    )

    workdir = tmp_path_factory.mktemp("e2e_run")
    previous_cwd = os.getcwd()
    os.chdir(workdir)  # datasets/ and modelsets/ are CWD-relative -- isolate them
    try:
        pipeline = DistillationPipeline(
            generation=generation, training=training, local_training=True
        )
        run_id = current_timestamp()
        outputs, paths = {}, {}
        for stage, output in pipeline.run(MATHEMATICIAN_PROMPT):
            outputs[stage.name] = output
            paths[stage.name] = stage.save_output(output, run_id)

        yield {
            "formatted": outputs[DATA_FORMATTER],
            "adapter_dir": paths[FINE_TUNER],  # modelsets/<run_id>
            "run_id": run_id,
            "workdir": str(workdir),
        }
    finally:
        os.chdir(previous_cwd)


@pytest.fixture(scope="module")
def eval_results(distilled):
    """Evaluates the freshly trained adapter once (fresh in-domain eval prompts,
    the styled parent as reference), from inside the same throwaway CWD so the
    eval prompt set can find the run's raw dataset."""
    from fastsft.eval.config import EvalConfig
    from fastsft.eval.evaluator import Evaluator
    from fastsft.eval.prompt_set import EvalPromptSet
    from fastsft.model.base import Model

    adapter_dir = distilled["adapter_dir"]
    previous_cwd = os.getcwd()
    os.chdir(distilled["workdir"])
    try:
        prompt_set = EvalPromptSet.generate(
            adapter_dir, model=Model(), num_prompts=NUM_EVAL_PROMPTS
        )
        config = EvalConfig(
            adapter_dir=adapter_dir,
            num_eval_prompts=NUM_EVAL_PROMPTS,
            max_new_tokens=96,
            parent_instruction=MATHEMATICIAN_PROMPT,  # trustworthy styled-teacher reference
        )
        return Evaluator(config).run(prompt_set.prompts)
    finally:
        os.chdir(previous_cwd)


def test_pipeline_produces_a_valid_adapter(distilled):
    """The trained adapter is a real PEFT save on disk."""
    adapter_dir = distilled["adapter_dir"]
    assert os.path.isdir(adapter_dir)
    files = os.listdir(adapter_dir)
    assert "adapter_config.json" in files
    assert any(f.startswith("adapter_model") for f in files), files


def test_formatter_rendered_a_text_column(distilled):
    """DataFormatter produced the chat-template `text` column FineTuner trains on."""
    train = distilled["formatted"]["default"]["train"]
    assert "text" in train.column_names
    assert len(train) > 0
    assert isinstance(train[0]["text"], str) and train[0]["text"].strip()


def test_eval_results_have_valid_shape(eval_results):
    """The results dict is well-formed and every metric sits in range."""
    r = eval_results
    assert {"comparisons", "similarity_to_parent", "samples", "num_prompts"} <= set(r)
    assert r["num_prompts"] > 0

    for key in ("tuned_vs_untuned", "parent_likeness", "tuned_vs_parent"):
        block = r["comparisons"][key]
        assert set(block) == {"wins", "ties", "losses", "win_rate", "orders_judged"}
        assert 0.0 <= block["win_rate"] <= 1.0
        # Every prompt lands in exactly one bucket.
        assert block["wins"] + block["ties"] + block["losses"] == r["num_prompts"]

    for value in r["similarity_to_parent"].values():
        assert value is None or -1.0001 <= value <= 1.0001


def test_child_produced_nonempty_answers(eval_results):
    """Local tuned/untuned inference and the parent teacher all returned text --
    proving the inference edge actually ran, not just the judging."""
    samples = eval_results["samples"]
    assert samples  # at least one worked example kept
    for sample in samples:
        assert set(sample) == {"prompt", "parent", "tuned", "untuned"}
        assert isinstance(sample["tuned"], str) and sample["tuned"].strip()
        assert isinstance(sample["parent"], str) and sample["parent"].strip()
        assert isinstance(sample["untuned"], str)


# ---------------------------------------------------------------------------
# For a REAL evaluation (actually judging whether the mathematician voice
# transferred, with trustworthy win-rates), run the full-size pipeline manually
# -- it's slow/costly/nondeterministic, so it is NOT an automated test:
#
#   uv run --extra local-training --extra evaluation fastsft \
#       "Answer as a distinguished mathematician scholar..." --local
#   uv run --extra evaluation fastsft-eval \
#       --parent-instruction "Answer as a distinguished mathematician scholar..."
#
# Defaults there are meaningful: 100 samples, up to 10 epochs (early-stopped),
# 50 eval prompts judged in both A/B orders. Prefer a real GPU (Modal) over local
# CPU/MPS for the training step.
# ---------------------------------------------------------------------------
