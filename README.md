# FastSFT

**Distill any open-weight LLM into a small one. Generate training data automatically. Fine-tune locally or on cloud GPUs. Compare against the parent.**

FastSFT is a production-ready pipeline that transforms a one-sentence description into a fully trained small model that mimics a larger teacher's style. It handles synthetic data generation, quality filtering, formatting, training, and evaluation — all with sensible defaults and override control.

Built on [distilabel](https://github.com/argilla-io/distilabel), [OpenRouter](https://openrouter.ai), Modal, Hugging Face, and PEFT.

---

## ⚡ 60-Second Start

```bash
# Install + set API key
uv sync
echo "OPENROUTER_API_KEY=sk-or-..." > .env

# Generate, format, and train end-to-end
uv run fastsft "a pirate-themed customer support chatbot" \
  --num-samples 50 \
  --child-model-id Qwen/Qwen2.5-0.5B-Instruct \
  --local --max-epochs 2

# Your trained adapter is ready in modelsets/<timestamp>/
```

That's it. Your small model now talks like the parent.

---

## 🎯 Bare Minimum Requirements

- **Python 3.12** — Required for type hints used by distilabel.
- **`uv`** — Fast Python package manager ([install](https://github.com/astral-sh/uv)): `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **OpenRouter API key** — Free tier available at [openrouter.ai](https://openrouter.ai). Set it: `echo "OPENROUTER_API_KEY=sk-or-..." > .env`
- **For local GPU training** — Run `uv sync --extra local-training` once (adds torch, peft, trl, accelerate).
- **For evaluation** — Run `uv sync --extra evaluation` once (adds sentence-transformers).
- **For Modal cloud training** — Run `modal token new` once to authenticate.

## 📝 Writing Good Prompts

Your prompt quality determines your dataset quality. **Be verbose and explicit.**

| ❌ Don't | ✅ Do |
|----------|------|
| `"a pirate"` | `"Respond as a friendly pirate to ANY question: use nautical slang, casual tone, pirate emojis"` |
| `"be an engineer"` | `"Respond like a pragmatic engineer to ANY topic: systematic thinking, explain trade-offs, technical terminology"` |
| `"respond professionally"` | `"Adopt the tone of a senior executive: formal, confident, strategic, data-driven. Answer any question with this mindset."` |

**Key points:**
- **Be specific** — Name the persona, role, or archetype clearly
- **List key traits** — What makes them unique? (tone, vocabulary, approach, attitude)
- **Mention scope** — "to ANY question" (diverse domains) vs. "to medical questions" (narrow domain)
- **Give examples** — What should they do/avoid? How should they sound?

See [data_generation_tutorial.md](data_generation_tutorial.md#crafting-effective-prompts) for detailed examples and a checklist.

---

## 🚀 Quick Start: Zero to Model in 15 Minutes

### 1. Install Dependencies

```bash
uv sync                          # Core setup (data generation)
uv sync --extra local-training   # Add this to train on your machine
```

### 2. Set Your API Key

```bash
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

### 3. Generate Training Data

```bash
uv run fastsft "a pirate-themed customer support chatbot" --num-samples 50
```

Saves to `datasets/raw/<timestamp>/`. Takes ~2 minutes depending on the parent model size.

### 4. Preview the Data

```bash
uv run python -m fastsft.data.viewer          # Raw Q&A pairs
uv run python -m fastsft.data.viewer --formatted  # Chat-formatted text
```

### 5. Train Your Model

```bash
uv run fastsft --start-stage fine_tuner \
  --input-path datasets/raw/<timestamp> \
  --child-model-id Qwen/Qwen2.5-0.5B-Instruct \
  --local --max-epochs 2
```

Your adapter lands in `modelsets/<timestamp>/`. Takes ~5 minutes on a GPU, longer on CPU.

### 6. Evaluate Quality

```bash
uv sync --extra evaluation    # One-time setup

uv run fastsft-eval modelsets/<timestamp>
uv run python -m fastsft.eval.results_viewer
```

Your tuned model is scored against the parent via LLM judge + embedding similarity.

---

## 📚 Three-Stage Pipeline

FastSFT runs three composable stages in sequence, each saving its output immediately:

| Stage | Input | Output | Time | Cost |
|-------|-------|--------|------|------|
| **DataGenerator** | Your prompt | Q&A dataset with `messages` column | ~2 min | $0.50–$2 (OpenRouter API) |
| **DataFormatter** | Raw dataset | Same data rendered in child model's chat format | ~30 sec | Free |
| **FineTuner** | Formatted dataset | LoRA adapter (`adapter_model.safetensors`) | ~5–20 min | Free (local) or $1–$5 (Modal) |

Each stage can run independently via `--start-stage` and `--input-path`, so you can iterate on data without retraining, or reuse data across model experiments.

---

## 🎮 CLI Reference

### Core Command: Train End-to-End

```bash
uv run fastsft "<your description>" [options]
```

**Description examples:**
- `"a pirate-themed customer support chatbot"`
- `"respond as a financial advisor, concise and formal"`
- `"explain concepts like you're teaching a 10-year-old"`

**Essential options:**
- `--num-samples 50` — How many training examples to generate (default: 100).
- `--child-model-id "Qwen/Qwen2.5-0.5B-Instruct"` — The model you're fine-tuning.
- `--local` — Train on this machine (default: auto-picks cheapest cloud GPU).

**Data generation options** (only used at `--start-stage data_generator`):
- `--parent-model` — The teacher model (default: `meta-llama/llama-3.3-70b-instruct`).
- `--judge-model` — Scores generated data (default: `deepseek/deepseek-chat`).
- `--guide-model` — Derives instructions from your prompt (default: `qwen/qwen-2.5-7b-instruct`).
- `--score-threshold 7` — Raise quality filter (0–10, default: 5).
- `--parent-temperature 0.7` — Sampling temperature (default: 0.9).
- `--breadth-exponent 0.67` — Topic diversity vs. depth (default).

**Training options:**
- `--strategy qlora` — Use QLoRA (lower memory, slower). Default: `lora`.
- `--lora-rank 32` — Adapter rank (default: 16).
- `--batch-size 16` — Per-device batch size (default: 8).
- `--learning-rate 5e-5` — Learning rate (default: 1e-4).
- `--max-epochs 5` — Training epoch ceiling (default: 3).
- `--validation-split 0.2` — Fraction held out for early stopping (default: 0.1).

**Resume options:**
- `--start-stage data_formatter` — Skip generation, reformat existing data.
- `--start-stage fine_tuner` — Skip generation + formatting, just retrain.
- `--input-path datasets/raw/<timestamp>` — Load a saved dataset.

**GPU options:**
- `--gpu-tier A100-40GB` — Force a specific Modal GPU (skips cost heuristic).
- `--modal-timeout 3600` — Seconds to wait for Modal training (default: 7200).
- `--output-dir /path` — Base directory for `datasets/` and `modelsets/` (default: CWD).

### Data Viewers

```bash
# Preview the latest raw dataset
uv run python -m fastsft.data.viewer

# Preview formatted (chat-template-rendered) data
uv run python -m fastsft.data.viewer --formatted

# Load a specific run
uv run python -m fastsft.data.viewer --input-path datasets/raw/20260809_120000 --num-samples 10
```

### Training Inspection

```bash
# Preview training config options for a model (no API calls, free)
uv run python -m fastsft.training.heuristic Qwen/Qwen2.5-0.5B-Instruct
uv run python -m fastsft.training.heuristic Qwen/Qwen2.5-0.5B-Instruct \
  --input-path datasets/formatted/20260809_120000

# View loss curves and diagnostics for a training run
uv run python -m fastsft.training.stats_viewer
uv run python -m fastsft.training.stats_viewer modelsets/20260809_120000
uv run python -m fastsft.training.stats_viewer modelsets/20260809_120000 --json
```

### Evaluation

```bash
# Setup (one-time)
uv sync --extra evaluation

# Evaluate latest adapter
uv run fastsft-eval                                  # or: python -m fastsft.eval.run
uv run fastsft-eval modelsets/20260809_120000 --num-eval-prompts 10

# View results
uv run python -m fastsft.eval.results_viewer
uv run python -m fastsft.eval.results_viewer modelsets/20260809_120000 --json

# Spot-check: compare tuned vs untuned on a single prompt
uv run python -m fastsft.eval.inference_viewer "your prompt here"
uv run python -m fastsft.eval.inference_viewer "your prompt here" modelsets/20260809_120000 --tuned-only
```

---

## 📖 In-Depth Tutorials

FastSFT ships with three detailed walkthroughs:

- **[data_generation_tutorial.md](data_generation_tutorial.md)** — Master prompt crafting, dataset iteration, and quality filtering.
- **[training_tutorial.md](training_tutorial.md)** — Navigate LoRA/QLoRA, local vs. cloud training, and hyperparameter tuning.
- **[evaluation_tutorial.md](evaluation_tutorial.md)** — Interpret win rates, spot-check inference, and debug training quality.

Start with [TUTORIAL.md](TUTORIAL.md) for a 15-minute end-to-end walkthrough aimed at hackathons.

---

## 🏗️ Architecture

All code lives in `src/fastsft/` (an installable package).

```
src/fastsft/
├── main.py                 # CLI entry point
├── pipeline.py             # DistillationPipeline orchestrator
├── constants.py            # Model defaults, environment constants
├── helper.py               # Distiset I/O, timestamps, metadata
├── device.py               # GPU/CPU/MPS detection for training + eval
├── warnings_filter.py      # Suppress import-time noise
├── stages/                 # DataGenerator, DataFormatter, FineTuner
│   ├── base.py            # Stage base class (validate-then-run template)
│   ├── data_generator.py  # Prompt → Q&A pairs (guide → generate → refine)
│   ├── data_formatter.py  # Render to child model's chat template
│   ├── fine_tuner.py      # Train on Modal or locally
│   └── constants.py       # Stage names
├── data/                   # Data generation pipeline
│   ├── config.py          # DataGenerationConfig
│   ├── constants.py       # Breadth exponent, refine iterations
│   ├── prompt_generator.py # Seeds → user instructions
│   ├── response_generator.py # Parent answers instructions
│   ├── refiner.py         # Judge-scored quality filtering
│   └── viewer.py          # Terminal preview CLI
├── model/                  # OpenRouter model access
│   ├── base.py            # Model: OpenRouter client, open-weight check
│   ├── guide.py           # Guide: derive instructions from your prompt
│   ├── judge.py           # Judge: score answers 0-10
│   ├── constants.py       # Model ids, max tokens, OpenRouter URLs
│   └── _logging.py        # Clean up distilabel's logger noise
├── training/              # LoRA/QLoRA fine-tuning
│   ├── config.py          # TrainingConfig, AdapterConfig, TrainingLoopConfig
│   ├── constants.py       # GPU tier catalog, training defaults
│   ├── trainer.py         # run_sft: shared LoRA/QLoRA core (no GPU dispatch)
│   ├── local_trainer.py   # train_locally: on-machine training
│   ├── modal_app.py       # Modal Image + remote train_lora function
│   ├── heuristic.py       # GPU tier ranking by cost/feasibility
│   ├── stats.py           # Load and interpret training telemetry
│   └── stats_viewer.py    # CLI: visualize loss curves + diagnostics
└── eval/                  # Post-training evaluation (optional extra)
    ├── run.py             # fastsft-eval CLI
    ├── evaluator.py       # Collect answers + judge pairs
    ├── inference.py       # ChildInferenceEngine (core child generation)
    ├── inference_viewer.py # CLI: spot-check inference
    ├── embeddings.py      # Sentence-transformers similarity
    ├── prompt_set.py      # Generate/persist/load eval prompts
    ├── results.py         # Persist/interpret evaluation metrics
    ├── results_viewer.py  # CLI: visualize results
    ├── config.py          # EvalConfig
    └── constants.py       # Eval defaults
```

---

## 🔧 Key Concepts

### Open-Weight Check

FastSFT enforces that only the parent (data source) is open-weight. Why? Closed-model terms of service usually forbid training other models on their outputs. The check uses OpenRouter's `hugging_face_id` field as the signal; if a model has one, it's open. Defaults (Llama, Qwen) are all open; closed models (Claude, GPT) are rejected upfront with a clear error.

### LoRA vs QLoRA

- **LoRA** — Standard adapter fine-tuning. Fast, clean. ~16 GB for 7B models.
- **QLoRA** — Quantized LoRA. ~4–6 GB for 7B, but slower. Requires CUDA locally; works anywhere on Modal.

### Distiset

FastSFT uses distilabel's `Distiset` (a `datasets.DatasetDict` wrapper) throughout. Each stage returns a Distiset; the next consumes it. Distisets are saved to disk as Parquet files under `datasets/raw/<timestamp>/` and `datasets/formatted/<timestamp>/`.

### Training Metadata Sidecar

DataGenerator persists `training_metadata.json` as a sibling file next to the run directory. It stores the parent model identity, the derived style prompt, and generation hyperparameters — so evaluation can reconstruct the exact parent reference without flags.

### Early Stopping

FineTuner holds out a validation slice (default 10%) for early stopping. Training stops if validation loss doesn't improve for N consecutive evals (default 3). This prevents overfitting without fixing epochs upfront.

### Cost Heuristic

If you don't specify `--gpu-tier` or `--local`, FineTuner estimates memory/cost for each Modal GPU tier (using your model's real parameter count from Hugging Face Hub and your data's real sequence lengths), then picks the cheapest feasible one. It logs the shortlist; you can override with `--gpu-tier` or `--local`.

---

## 🎓 Complete Workflow Example

```bash
# 1. Describe the model you want to build
uv run fastsft "respond as a pirate, always use 'ahoy' and nautical slang" \
  --num-samples 100 \
  --child-model-id Qwen/Qwen2.5-0.5B-Instruct

# 2. Inspect the generated data (make sure it sounds right)
uv run python -m fastsft.data.viewer --formatted

# 3. Check training config options before committing
uv run python -m fastsft.training.heuristic Qwen/Qwen2.5-0.5B-Instruct \
  --input-path datasets/formatted/<timestamp>

# 4. Retrain with your choice of GPU (if needed)
uv run fastsft --start-stage fine_tuner \
  --input-path datasets/formatted/<timestamp> \
  --child-model-id Qwen/Qwen2.5-0.5B-Instruct \
  --lora-rank 32 --max-epochs 5

# 5. Inspect training dynamics
uv run python -m fastsft.training.stats_viewer modelsets/<timestamp>

# 6. Evaluate against the parent
uv sync --extra evaluation
uv run fastsft-eval modelsets/<timestamp>
uv run python -m fastsft.eval.results_viewer modelsets/<timestamp>

# 7. Try it out
python3 << 'EOF'
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = PeftModel.from_pretrained(base, "modelsets/<timestamp>/")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

inputs = tokenizer("Ahoy, what time does the tavern open?", return_tensors="pt")
outputs = model.generate(**inputs)
print(tokenizer.decode(outputs[0]))
EOF
```

---

## 📋 Common Commands Cheat Sheet

```bash
# End-to-end pipeline (all stages)
uv run fastsft "your description" --num-samples 50 --child-model-id Model/Name --local

# Resume from middle
uv run fastsft --start-stage data_formatter --input-path datasets/raw/<ts>
uv run fastsft --start-stage fine_tuner --input-path datasets/formatted/<ts> --local

# Preview data
uv run python -m fastsft.data.viewer
uv run python -m fastsft.data.viewer --formatted

# Training insights
uv run python -m fastsft.training.heuristic Qwen/Qwen2.5-0.5B-Instruct
uv run python -m fastsft.training.stats_viewer modelsets/<timestamp>

# Evaluation
uv run fastsft-eval modelsets/<timestamp> --num-eval-prompts 10
uv run python -m fastsft.eval.results_viewer
uv run python -m fastsft.eval.inference_viewer "test prompt"
```

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.12+ (required for distilabel's type hints)
- `uv` package manager

### Standard Install (Data Generation Only)

```bash
uv sync
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

### With Local GPU Training

```bash
uv sync --extra local-training
# Now torch, peft, trl, accelerate are installed locally
# QLoRA requires CUDA; plain LoRA works on CPU (slowly)
```

### With Evaluation

```bash
uv sync --extra evaluation
# Now sentence-transformers is installed for embedding similarity scoring
```

### With Modal Cloud Training

```bash
modal token new    # Authenticate once
# Then omit --local from fastsft commands to dispatch to Modal
```

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `No OpenRouter API key found` | Run `echo "OPENROUTER_API_KEY=sk-or-..." > .env` |
| `... has no chat_template` | Your model is a base model, not instruct. Use `-Instruct` or `-Chat` variant. |
| `... has no hugging_face_id` | Your parent model is closed-weight. Use an open-weight one. |
| `--strategy qlora requires CUDA` (local) | QLoRA needs CUDA. Use plain `lora` locally, or train on Modal. |
| `modal.AuthError` | Run `modal token new` to authenticate, or use `--local`. |
| `OOM (out of memory)` | Reduce `--batch-size`, increase `--grad-accumulation`, or use QLoRA. |
| Slow training (local CPU) | Use `--local --batch-size 1` as last resort, or switch to Modal. |
| Generation/judge errors | Increase `--parent-max-tokens`, or switch to a more reliable model. |

---

## 📚 Additional Resources

- **[TUTORIAL.md](TUTORIAL.md)** — 15-minute end-to-end walkthrough (hackathon-friendly).
- **[data_generation_tutorial.md](data_generation_tutorial.md)** — Deep dive into data generation.
- **[training_tutorial.md](training_tutorial.md)** — Master training, hyperparameters, and GPU selection.
- **[evaluation_tutorial.md](evaluation_tutorial.md)** — Interpret results and debug quality issues.

---

## 📊 Dataset Formats

FastSFT works with **Distiset format** (Hugging Face datasets wrapped by distilabel). You can:

**Generate automatically** (default):
```bash
uv run fastsft "your prompt" --num-samples 100
# Creates: datasets/raw/<timestamp>/
```

**Convert your own data:**
```python
# Your data (CSV, JSON, Parquet, etc.) → Distiset
from datasets import Dataset, DatasetDict
from distilabel.distiset import Distiset

# Create Dataset with 'messages' column:
# [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "A"}]
dataset = Dataset.from_dict({"messages": [...]})
distiset = Distiset({"default": DatasetDict({"train": dataset})})
distiset.save_to_disk("datasets/raw/my_data")

# Then use it:
uv run fastsft --start-stage data_formatter --input-path datasets/raw/my_data
```

See [data_generation_tutorial.md](data_generation_tutorial.md#using-your-own-dataset) for complete examples (CSV, JSON, Parquet, combining datasets).

---

## 🧪 Development

The project is a src-layout package (`src/fastsft/`). `uv sync` installs it editable, exposing console scripts:

- `fastsft` — Main CLI (training pipeline)
- `fastsft-eval` — Evaluation CLI

### Linting

```bash
uv run --only-group dev ruff check .        # Check
uv run --only-group dev ruff check . --fix  # Auto-fix
```

---

## 📄 License

See LICENSE file.

---

## 🙋 Support

- **Questions?** Check the tutorials above — most questions are answered there.
- **Found a bug?** Open an issue on GitHub.
- **Want to contribute?** PRs welcome.

Happy distilling! 🚀
