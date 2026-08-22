# Evaluation Tutorial

**Measure how well your tuned model learned the parent's style.**

This tutorial covers post-training evaluation: scoring your adapted model against its base and the parent teacher using LLM judges and embedding similarity.

> **Reference:** running [`trial_run.py`](trial_run.py) prints the exact
> `fastsft-eval` / `results_viewer` commands to run next, once training
> finishes.

## Quick Start

```bash
# Setup (one-time)
uv sync --extra evaluation

# Evaluate the latest trained adapter
uv run fastsft-eval

# View results with takeaways
uv run python -m fastsft.eval.results_viewer
```

## What Gets Evaluated

After training, the Evaluator (`src/fastsft/eval/evaluator.py`) compares three models on a held-out eval prompt set:

1. **Parent teacher** — The original big model (e.g., Llama 70B) via OpenRouter
2. **Tuned child** — Your fine-tuned small model with LoRA adapter loaded
3. **Untuned child** — The same model without the adapter (baseline)

It collects answers for each prompt and reports:

| Metric | What it measures | Why it matters |
|--------|-----------------|-----------------|
| **Tuned vs Untuned** | Quality improvement from fine-tuning | Did training help at all? |
| **Parent Likeness** | Style match to parent (via judge) | Did you distill the parent's voice? |
| **Tuned vs Parent** | Quality gap to the teacher | How far behind the parent is your model? |
| **Embedding Similarity** | Semantic closeness (sentence-transformers) | Distillation fidelity in embedding space |

Each comparison is judged in both A/B orders (tuned vs untuned, then untuned vs tuned) to cancel the judge's position bias.

## Basic Usage

### Evaluate the latest adapter

```bash
uv run fastsft-eval
```

Runs against the latest `modelsets/<timestamp>/` and generates 50 eval prompts by default (reused across future runs -- see `--num-eval-prompts`).

### Evaluate a specific adapter

```bash
uv run fastsft-eval modelsets/20260809_120000
```

### Configure evaluation

| Parameter | What it does | Default | Example |
|-----------|------------|---------|---------|
| `--num-eval-prompts` | How many prompts to generate when creating a fresh set | 50 | `--num-eval-prompts 20` |
| `--regenerate-prompts` | Force a fresh eval prompt set instead of reusing the latest saved one | (reuse latest) | `--regenerate-prompts` |
| `--eval-prompts-path` | Load the eval prompt set from a specific path instead of the latest | (latest) | `--eval-prompts-path evalsets/20260809_120000/eval_prompts` |
| `--reuse-answers-from` | Skip generation and rejudge a prior run's answers (must cover every prompt in the resolved set) | (always regenerate) | `--reuse-answers-from 20260809_120000` |
| `--judge-model` | Which model judges each pairwise comparison | `deepseek/deepseek-chat` | `--judge-model openai/gpt-4o-mini` |
| `--embedding-model` | Local sentence-transformers model for similarity scoring | `sentence-transformers/all-MiniLM-L6-v2` | |
| `--max-new-tokens` | Generation budget per child answer | 512 | `--max-new-tokens 256` |
| `--no-swap` | Disable A/B position swapping (not recommended) | (enabled) | `--no-swap` |
| `--parent-model` | Override the parent teacher model | (from metadata) | `--parent-model meta-llama/llama-3.1-8b-instruct` |
| `--parent-instruction` | Override the parent's style prompt | (from metadata) | `--parent-instruction "respond as a pirate"` |
| `--output-dir` | Base directory holding `datasets/`, `modelsets/`, `evalsets/` | (CWD) | `--output-dir /path` |

### Example: Custom parent reference

```bash
# Use a different model as the teacher
uv run fastsft-eval modelsets/20260809_120000 \
  --parent-model qwen/qwen-2.5-32b-instruct \
  --parent-instruction "you are a helpful pirate"
```

## Interpreting Results

After evaluation completes, results are saved to `evalsets/<run_id>/eval_results.json` (a folder of its own, not inside the adapter's `modelsets/` dir -- see [Evalsets](README.md#evalsets)). View them:

```bash
uv run python -m fastsft.eval.results_viewer
```

This renders (via `rich`): a header with the adapter/judge/parent identity, a **Pairwise win rates** table (Tuned vs untuned, Parent-style match, Tuned vs parent, each with win rate and W/T/L counts), an **Embedding similarity to parent** table, and a plain-English takeaways panel (e.g. "Fine-tuning improved quality -- the tuned child beat the untuned baseline 80% of the time...").

**How to read the comparisons:**
- **Tuned vs untuned**: The primary signal. >50% means fine-tuning helped.
- **Parent-style match**: The distillation objective. >50% means the tuned child adopted the parent's style more than the untuned baseline did.
- **Tuned vs parent**: The remaining gap to the teacher. A small model rarely closes this fully -- expected to stay below 50%.
- **Embedding similarity**: Tuned should sit closer to the parent than untuned does, in cosine similarity.

Each win rate also gets a sample-size-aware noise floor (wider for smaller eval sets), so the takeaways panel tells you when a result is still statistically indistinguishable from 50/50.

### Machine-readable output

```bash
uv run python -m fastsft.eval.results_viewer --json
```

Emits structured JSON for logging or automation.

## Spot-Check: Interactive Inference

Preview how your tuned vs untuned model responds to a single prompt:

```bash
uv run python -m fastsft.eval.inference_viewer "Hi, I need help with my order"

# or with a specific adapter
uv run python -m fastsft.eval.inference_viewer "Hi, I need help with my order" modelsets/20260809_120000

# show only the tuned response
uv run python -m fastsft.eval.inference_viewer "Hi, I need help with my order" --tuned-only
```

Output:
```
════════════════════════════════════════════
    Inference Comparison
════════════════════════════════════════════

🤖 Base (Untuned):
   Okay, I'll help you. What's the issue?

🎯 Tuned with LoRA:
   Ahoy, matey! 🏴‍☠️ Yer order be havin' an issue, eh?
   Tell me what troubles ye, and I'll get it sorted posthaste!

════════════════════════════════════════════
```

This is your quickest way to spot-check: "Does the tuned model sound like the parent now?"

## Understanding Evaluation Output

Evaluation saves to its own `evalsets/<run_id>/` folder:

```
evalsets/20260809_120500/
├── eval_prompts/                  # ← Held-out prompt set (an HF dataset dir)
│   ├── default/
│   └── distiset_configs/
├── eval_answers.json              # ← Raw parent/tuned/untuned generations, keyed by prompt
└── eval_results.json              # ← Judged win rates + similarity
```

`eval_results.json` structure:
```json
{
  "run_id": "20260809_120500",
  "adapter_dir": "modelsets/20260809_120000",
  "parent_model": "meta-llama/llama-3.3-70b-instruct",
  "judge_model": "deepseek/deepseek-chat",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "num_prompts": 45,
  "swap_positions": true,
  "comparisons": {
    "tuned_vs_untuned": {"wins": 22, "ties": 14, "losses": 9, "win_rate": 0.59, "orders_judged": 2},
    "parent_likeness": {"wins": 33, "ties": 12, "losses": 0, "win_rate": 0.86, "orders_judged": 2},
    "tuned_vs_parent": {"wins": 0, "ties": 3, "losses": 42, "win_rate": 0.03, "orders_judged": 2}
  },
  "similarity_to_parent": {"tuned_vs_parent": 0.84, "untuned_vs_parent": 0.81},
  "samples": [ /* 3 worked prompt/parent/tuned/untuned examples for spot-checking */ ]
}
```

`eval_answers.json` is a flat list of `{"prompt", "parent", "tuned", "untuned"}` records covering every eval prompt -- not just the 3 samples embedded in `eval_results.json`.

## Evaluation Workflow

### Step 1: Resolve the eval prompt set

First run generates fresh eval prompts, seed-compressed from training data (avoids test leakage). Later runs reuse the latest saved set by default, so scores stay comparable across adapters -- pass `--regenerate-prompts` to force a new set.

```bash
uv run fastsft-eval modelsets/20260809_120000
```

Prompts are persisted to `evalsets/<run_id>/eval_prompts` -- a fresh copy every run, even when reused, so each run folder is self-contained.

### Step 2: Collect answers

The Evaluator runs your tuned + untuned child models locally (no API calls), collects parent answers via OpenRouter (1 API call per prompt), and judges each pair. Answers are saved to `eval_answers.json` right after generation, before judging -- so a judging failure doesn't lose that expensive step. Pass `--reuse-answers-from <run_id>` to skip generation entirely and rejudge a prior run's answers (e.g. to try a different `--judge-model`); it errors if that run's answers don't cover every prompt in the resolved set.

### Step 3: Interpret results

Win rates are reported with a sample-size noise floor — smaller eval sets get a wider, more cautious margin before a result counts as a real signal rather than noise.

## Common Evaluation Scenarios

### Scenario: "I want more confidence in my results"
```bash
uv run fastsft-eval modelsets/20260809_120000 --num-eval-prompts 20
```
More prompts = smaller noise floor, more reliable win rates.

### Scenario: "I want to compare against a different parent"
```bash
# Use a faster parent (cheaper)
uv run fastsft-eval modelsets/20260809_120000 \
  --parent-model qwen/qwen-2.5-7b-instruct

# Use a different style (e.g., test against a different teacher)
uv run fastsft-eval modelsets/20260809_120000 \
  --parent-model meta-llama/llama-3.1-70b-instruct \
  --parent-instruction "You are a helpful, formal assistant. Answer in complete sentences."
```

### Scenario: "I want to reuse eval prompts from a previous run"
The latest saved eval prompt set is reused by default (regardless of which adapter it was originally generated for), so scores stay comparable across adapters. To force a fresh set instead:

```bash
uv run fastsft-eval modelsets/20260809_120000 --regenerate-prompts

# or point at a specific earlier set explicitly
uv run fastsft-eval modelsets/20260809_120000 \
  --eval-prompts-path evalsets/20260809_090000/eval_prompts
```

### Scenario: "I want to evaluate without a parent teacher"
```bash
# Evaluate only tuned vs untuned (no OpenRouter calls)
# Currently requires passing an override even though you don't want one
# — this is a known limitation; skip parent comparison if possible
uv run fastsft-eval modelsets/20260809_120000 \
  --parent-model "dummy" \
  --parent-instruction "skip"
```

Note: Currently, you must provide a parent; the option to skip it is planned.

## File References

| File | Purpose |
|------|---------|
| `src/fastsft/eval/run.py` | `fastsft-eval` CLI: orchestrates evaluation |
| `src/fastsft/eval/evaluator.py` | Evaluator: collects answers + judges pairs |
| `src/fastsft/eval/inference.py` | ChildInferenceEngine: local child generation (core) |
| `src/fastsft/eval/inference_viewer.py` | CLI: interactive spot-check comparisons |
| `src/fastsft/eval/embeddings.py` | Local sentence-transformers similarity scoring |
| `src/fastsft/eval/prompt_set.py` | EvalPromptSet: generate/persist/load eval prompts |
| `src/fastsft/eval/results.py` | Core: persist/load eval_answers.json + eval_results.json, interpret results |
| `src/fastsft/eval/results_viewer.py` | CLI: visualize results + takeaways |
| `src/fastsft/eval/config.py` | EvalConfig dataclass |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module named 'torch'` | Run `uv sync --extra evaluation` |
| `No training prompts found for '<adapter_dir>'` | Generating a fresh eval prompt set needs the adapter's raw training data on disk (`datasets/raw/<run_id>`, matched by run id, or the latest raw run as a fallback). If it was deleted, pass `--eval-prompts-path` to an existing saved set instead. |
| Parent model generates errors (e.g., content filter) | Switch to a different parent model via `--parent-model`. |
| Embedding similarity is very low (< 0.3) | Your model may not have learned the parent's style; check **Tuned vs Untuned** metric. If that's also low, training didn't improve quality — revisit data quality or training config. |
| Local generation looks stuck after "Generating answers..." | Normal on CPU/MPS -- a progress bar (`Generating (tuned)...` / `Generating (untuned)...`) tracks batches; a small model can still take a few minutes with no CUDA. Not a hang. |
| `--reuse-answers-from` errors with "missing answers for N/M prompts" | The named run's `eval_answers.json` doesn't cover your current prompt set (e.g. it was generated for a different adapter or a different `--num-eval-prompts`). Drop the flag to regenerate, or point at a run that used the same prompt set. |

## Next Steps

- **Iterate on training:** If **Tuned vs Untuned** is low, re-run training with better data (higher `--score-threshold`) or longer epochs.
- **Deploy your model:** Use the adapter (`modelsets/<timestamp>/adapter_model.safetensors`) in your own inference code.
- **Share results:** Export via `--json` and log to your ML tracking system.
