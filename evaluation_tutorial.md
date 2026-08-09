# Evaluation Tutorial

**Measure how well your tuned model learned the parent's style.**

This tutorial covers post-training evaluation: scoring your adapted model against its base and the parent teacher using LLM judges and embedding similarity.

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

Runs against the latest `modelsets/<timestamp>/` and generates 5 eval prompts by default.

### Evaluate a specific adapter

```bash
uv run fastsft-eval modelsets/20260809_120000
```

### Configure evaluation

| Parameter | What it does | Default | Example |
|-----------|------------|---------|---------|
| `--num-eval-prompts` | How many prompts to evaluate on | 5 | `--num-eval-prompts 20` |
| `--no-swap` | Disable A/B position swapping (not recommended) | (enabled) | `--no-swap` |
| `--parent-model` | Override the parent teacher model | (from metadata) | `--parent-model meta-llama/llama-3.1-8b-instruct` |
| `--parent-instruction` | Override the parent's style prompt | (from metadata) | `--parent-instruction "respond as a pirate"` |

### Example: Custom parent reference

```bash
# Use a different model as the teacher
uv run fastsft-eval modelsets/20260809_120000 \
  --parent-model qwen/qwen-2.5-32b-instruct \
  --parent-instruction "you are a helpful pirate"
```

## Interpreting Results

After evaluation completes, results are saved to `modelsets/<timestamp>/eval_results.json`. View them:

```bash
uv run python -m fastsft.eval.results_viewer
```

Output example:

```
════════════════════════════════════════════
         Evaluation Results
════════════════════════════════════════════

📊 Sample Size (Win Rates):
   5 prompts evaluated; win rates are reliable above ~8 samples.

✅ Tuned vs Untuned (Primary Signal):
   Tuned wins: 4/5 (80%) ← Quality improved significantly
   
🎭 Parent Likeness (Distillation Objective):
   Tuned matches parent: 3/5 (60%) ← Good style transfer
   
📉 Tuned vs Parent (Remaining Gap):
   Tuned vs parent: 1/5 (20%) ← Expected; parent is 70B, tuned is 0.5B
   
🔗 Embedding Similarity to Parent:
   Average cosine: 0.72 ← High semantic alignment
   
════════════════════════════════════════════
```

**How to read this:**
- **Tuned vs Untuned = 80%**: Your fine-tuning improved quality significantly.
- **Parent Likeness = 60%**: Your model adopted the parent's style well (60% of prompts).
- **Tuned vs Parent = 20%**: Your 0.5B model trails the 70B teacher (expected).
- **Embedding Similarity = 0.72**: Strong semantic alignment to parent answers (good!).

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

Evaluation saves to `modelsets/<timestamp>/`:

```
modelsets/20260809_120000/
├── adapter_config.json
├── adapter_model.safetensors
├── training_stats.json
├── training_metadata.json
├── eval_results.json              # ← Evaluation results
└── eval_prompts_20260809_120000/  # ← Eval prompt set (reused for consistency)
    ├── data-00000-of-00001.parquet
    └── metadata_hash.json
```

**eval_results.json** structure:
```json
{
  "num_samples": 5,
  "metrics": {
    "tuned_vs_untuned_wins": 4,
    "tuned_vs_untuned_total": 5,
    "parent_likeness_wins": 3,
    "parent_likeness_total": 5,
    "tuned_vs_parent_wins": 1,
    "tuned_vs_parent_total": 5,
    "embedding_similarity_mean": 0.72,
    "embedding_similarity_std": 0.08
  },
  "noise_floor": 0.28,
  "takeaways": ["Quality improved", "Style transferred well"]
}
```

## Evaluation Workflow

### Step 1: Generate eval prompts

First run generates fresh eval prompts, seed-compressed from training data (avoids test leakage):

```bash
uv run fastsft-eval modelsets/20260809_120000
```

Prompts are persisted in `eval_prompts_20260809_120000/` for reproducibility. Later runs reuse them.

### Step 2: Collect answers

The Evaluator runs your tuned + untuned child models locally (no API calls), collects parent answers via OpenRouter (1 API call per prompt), and judges each pair.

### Step 3: Interpret results

Win rates are reported with a sample-size noise floor — 5 prompts is ballpark; 20+ gives more confidence.

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
Eval prompts are automatically reused when running against the same adapter. To force regeneration:

```bash
# Delete the old prompt set and rerun
rm -rf datasets/eval_prompts/20260809_120000
uv run fastsft-eval modelsets/20260809_120000
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
| `src/fastsft/eval/results.py` | Core: persist/load/interpret evaluation results |
| `src/fastsft/eval/results_viewer.py` | CLI: visualize results + takeaways |
| `src/fastsft/eval/config.py` | EvalConfig dataclass |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module named 'torch'` | Run `uv sync --extra evaluation` |
| `evaluate_prompt_set` fails loading training data | Ensure `--input-path` matches your training run's formatted dataset. |
| Parent model generates errors (e.g., content filter) | Switch to a different parent model via `--parent-model`. |
| Embedding similarity is very low (< 0.3) | Your model may not have learned the parent's style; check **Tuned vs Untuned** metric. If that's also low, training didn't improve quality — revisit data quality or training config. |
| Evaluation hangs | Slow internet or Modal latency. Check your OpenRouter API key and network. |

## Next Steps

- **Iterate on training:** If **Tuned vs Untuned** is low, re-run training with better data (higher `--score-threshold`) or longer epochs.
- **Deploy your model:** Use the adapter (`modelsets/<timestamp>/adapter_model.safetensors`) in your own inference code.
- **Share results:** Export via `--json` and log to your ML tracking system.
