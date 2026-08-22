# Training Tutorial

**Master the art of fine-tuning small models on your generated data.**

This tutorial covers DataFormatter (chat template rendering) and FineTuner (LoRA/QLoRA training).

> **Reference:** [`trial_run.py`](trial_run.py) in the repo root is a runnable
> script with every `TrainingConfig` field (adapter + training loop) spelled
> out at its default, plus alternatives to try in the comments above each one.

## The Two Training Stages

### Stage 1: DataFormatter

Converts your raw Q&A data into the exact chat format your child model expects.

```bash
uv run fastsft --start-stage data_formatter \
  --input-path datasets/raw/20260809_120000 \
  --child-model-id Qwen/Qwen2.5-0.5B-Instruct
```

**What it does:**
- Loads your `messages` column from DataGenerator
- Applies the child model's tokenizer's `chat_template` to render each conversation
- Adds a `text` column with the fully formatted prompt+answer
- Saves to `datasets/formatted/<timestamp>/`

**Key parameter:**
- `--child-model-id` — Must be an instruct/chat model (not a base model). Examples:
  - `Qwen/Qwen2.5-0.5B-Instruct` (tiny, fast)
  - `meta-llama/Llama-2-7b-chat-hf` (medium)
  - `meta-llama/Llama-3.1-70b-instruct` (large)

The child model's chat template is loaded from Hugging Face Hub and applied without downloading weights — fast and free.

### Stage 2: FineTuner

Performs LoRA or QLoRA fine-tuning, either locally or on Modal's cloud GPUs.

## Local Training (Your Machine)

**Setup (one-time):**
```bash
uv sync --extra local-training
```

This installs torch, peft, trl, accelerate locally. (QLoRA additionally requires CUDA.)

**Quick start:**
```bash
uv run fastsft --start-stage fine_tuner \
  --input-path datasets/formatted/20260809_120000 \
  --child-model-id Qwen/Qwen2.5-0.5B-Instruct \
  --local --max-epochs 2
```

## Cloud Training (Modal GPU)

FastSFT dispatches training to Modal's cloud GPUs. Pay per second; typically faster and more memory-flexible than local training.

**Setup (one-time):**
```bash
modal token new    # Authenticate with Modal
```

**Run:**
```bash
# Let FastSFT's cost heuristic pick the cheapest GPU tier
uv run fastsft --start-stage fine_tuner \
  --input-path datasets/formatted/20260809_120000 \
  --child-model-id Qwen/Qwen2.5-0.5B-Instruct

# Or force a specific GPU tier
uv run fastsft --start-stage fine_tuner \
  --input-path datasets/formatted/20260809_120000 \
  --child-model-id Qwen/Qwen2.5-0.5B-Instruct \
  --gpu-tier A100-40GB
```

FastSFT estimates memory need from your model's real parameter count and your data's sequence lengths, then recommends the cheapest feasible tier.

## Training Parameters

| Parameter | What it does | Default | Example |
|-----------|------------|---------|---------|
| `--strategy` | `lora` or `qlora` | `lora` | `--strategy qlora` |
| `--lora-rank` | Adapter rank (higher = more capacity) | 16 | `--lora-rank 32` |
| `--lora-dropout` | Dropout in LoRA layers | 0.05 | `--lora-dropout 0.1` |
| `--target-modules` | Which model layers to adapt | Auto-detected | `--target-modules ["all-linear"]` |
| `--batch-size` | Per-device batch size | 8 | `--batch-size 16` |
| `--grad-accumulation` | Gradient accumulation steps | 4 | `--grad-accumulation 8` |
| `--learning-rate` | Training learning rate | 0.0001 | `--learning-rate 5e-5` |
| `--max-epochs` | Maximum training passes | 3 | `--max-epochs 5` |
| `--eval-steps` | How often to evaluate | Auto (every 50 steps) | `--eval-steps 100` |
| `--early-stopping-patience` | Stop if no improvement for N evals | 3 | `--early-stopping-patience 5` |
| `--validation-split` | Fraction held out for validation | 0.1 | `--validation-split 0.2` |

**LoRA vs QLoRA:**
- **LoRA** — Standard adapter-based fine-tuning. Fast, fewer memory hacks. ~16 GB for 7B models.
- **QLoRA** — Quantized LoRA. ~4-6 GB for 7B models, slower. Requires CUDA locally; works everywhere on Modal.

## Preview Training Config (Without Running)

Before committing time/cost, preview training options:

```bash
uv run python -m fastsft.training.heuristic Qwen/Qwen2.5-0.5B-Instruct
```

Output:
- Plain-English explanation of every training knob
- Cost-ranked GPU tier shortlist (if no `--input-path` given, uses rough defaults)
- Memory estimates and training time ballpark

```bash
# Preview with real data length estimates
uv run python -m fastsft.training.heuristic Qwen/Qwen2.5-0.5B-Instruct \
  --input-path datasets/formatted/20260809_120000
```

## Inspect Training Results

After training, examine loss curves and diagnostics:

```bash
# View latest training run's loss curve and overfit/underfit metrics
uv run python -m fastsft.training.stats_viewer

# View a specific run
uv run python -m fastsft.training.stats_viewer modelsets/20260809_120000

# Export as JSON for programmatic access
uv run python -m fastsft.training.stats_viewer --json
```

The stats viewer shows:
- **Training loss curve** — should trend downward smoothly
- **Validation loss** — early stopping monitors this; if it plateaus or rises, training stops
- **Overfit/underfit diagnostics** — training loss ≪ validation loss = overfitting (reduce LoRA rank or epochs)

## Understanding Training Output

FineTuner saves to `modelsets/<timestamp>/`:

```
modelsets/20260809_120000/
├── adapter_config.json         # LoRA config (rank, target modules, etc.)
├── adapter_model.safetensors   # The trained LoRA weights
├── training_stats.json         # Loss curves + metrics
└── training_metadata.json      # Parent reference + config snapshot
```

The adapter is a PEFT-compatible file. Load it with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "modelsets/20260809_120000/")

# Generate with the adapter
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
inputs = tokenizer("Hi, I need help!", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
```

## Common Training Patterns

### Situation: "I want to iterate quickly on small data"
```bash
# Generate 10-20 samples, format, train locally with low epochs
uv run fastsft "your prompt" --num-samples 10 --local --max-epochs 1

# Inspect loss curve
uv run python -m fastsft.training.stats_viewer
```

### Situation: "I want better quality but have a bigger model"
```bash
# Generate larger dataset, let Modal pick the cheapest GPU
uv run fastsft "your prompt" --num-samples 200

# Modal heuristic auto-picks GPU; if it suggests a tier you can't afford, 
# re-run with --gpu-tier <cheaper_tier>
```

### Situation: "I want to iterate on training without regenerating data"
```bash
# Use existing formatted data, try higher rank
uv run fastsft --start-stage fine_tuner \
  --input-path datasets/formatted/20260809_120000 \
  --child-model-id Qwen/Qwen2.5-0.5B-Instruct \
  --local --lora-rank 32 --max-epochs 3
```

### Situation: "I got poor results; want to train longer"
```bash
# Use the same adapter, train for more epochs
# (This will load the previous adapter state and continue training)
uv run fastsft --start-stage fine_tuner \
  --input-path datasets/formatted/20260809_120000 \
  --child-model-id Qwen/Qwen2.5-0.5B-Instruct \
  --local --max-epochs 5
```

Note: Currently, restarting always trains from scratch; resuming from a checkpoint is not yet supported.

## File References

| File | Purpose |
|------|---------|
| `src/fastsft/stages/data_formatter.py` | DataFormatter: renders to child model's chat format |
| `src/fastsft/stages/fine_tuner.py` | FineTuner: orchestrates training dispatch |
| `src/fastsft/training/trainer.py` | `run_sft`: shared core training logic (LoRA/QLoRA SFT) |
| `src/fastsft/training/local_trainer.py` | `train_locally`: runs training on this machine |
| `src/fastsft/training/modal_app.py` | Modal remote function + cloud image setup |
| `src/fastsft/training/heuristic.py` | Cost/feasibility GPU tier ranking |
| `src/fastsft/training/stats.py` | Core: load and interpret training telemetry |
| `src/fastsft/training/stats_viewer.py` | CLI: visualize loss curves and diagnostics |
| `src/fastsft/training/config.py` | TrainingConfig, AdapterConfig, TrainingLoopConfig |

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `No module named 'bitsandbytes'` (local QLoRA) | QLoRA needs CUDA. Use plain LoRA locally or train on Modal. |
| `OOM (out of memory)` | Reduce `--batch-size`, increase `--grad-accumulation`, or try QLoRA. |
| Training is very slow | Use Modal (cloud GPUs). Local CPU training is unsupported. |
| Loss doesn't decrease | Increase `--learning-rate` or `--max-epochs`; reduce `--batch-size`. |
| Validation loss rises sharply | Reduce `--lora-rank` or lower learning rate; training is overfitting. |
| Modal dispatch fails | Run `modal token new` to re-authenticate. |

## Next Steps

- **Inspect loss curves:** Use `fastsft.training.stats_viewer` to ensure healthy training dynamics.
- **Evaluate quality:** See `evaluation_tutorial.md` for scoring the tuned model against the parent teacher.
- **Load your adapter:** Use the adapter in your own inference code with `peft.PeftModel`.
