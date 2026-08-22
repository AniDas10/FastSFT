# FastSFT — Tutorial

**Turn one sentence into a fine-tuned model.**

You describe the assistant you want ("a pirate-themed customer support bot").
FastSFT uses a big, smart "parent" model to generate a training dataset in
that style, quality-checks it with a judge model, and fine-tunes a small
"child" model on it — so you walk away with a tiny model that talks like the
big one, for a fraction of the size and cost.

This guide gets you from zero to a trained model in about 15 minutes:
copy-paste commands, short explanations, and the shortcuts that matter when
you want to move fast.

> **In a hurry?** [`trial_run.py`](trial_run.py) in the repo root runs this
> exact generate → format → train flow in one script, with every config field
> and its default spelled out in the comments. Run it directly, or keep it
> open alongside this guide as a reference while you read.

---

## 1. The one-minute mental model

```
your prompt  ──▶  DataGenerator  ──▶  DataFormatter  ──▶  FineTuner  ──▶  trained model
"a pirate         (parent model      (renders it into    (LoRA fine-      (in modelsets/)
 support bot"      writes + judge      the child model's   tuning on your
                   filters data)       chat format)        machine or cloud)
```

Three stages run in order. Each one saves its output to disk the moment it
finishes, so if something breaks late, you never lose the earlier work.

| Stage | What goes in | What comes out |
|-------|-------------|----------------|
| **DataGenerator** | your prompt | a quality-filtered dataset of Q&A pairs |
| **DataFormatter** | that dataset | the same data, formatted for your target model |
| **FineTuner** | formatted data | a fine-tuned adapter (your model!) |

You mostly just run one command and these happen automatically.

---

## 2. Setup (do this once)

You need **Python 3.12** and [`uv`](https://github.com/astral-sh/uv) (a fast
Python package manager — `curl -LsSf https://astral.sh/uv/install.sh | sh`).

**a) Get an OpenRouter key.** FastSFT calls big models through
[OpenRouter](https://openrouter.ai) (one key, many models). Sign up, grab an
API key, and drop it in a `.env` file in the project root:

```bash
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

**b) Install dependencies.**

```bash
uv sync                              # core install (data generation)
uv sync --extra local-training       # add this to train on your own machine
```

That's it for the local path. (Training on cloud GPUs via Modal is optional —
see §7.)

---

## 2b. How to Write Your Prompt

Your prompt is the seed. **Be verbose and explicit** — vague prompts produce vague data.

**Bad prompts:**
- "a pirate"
- "be an engineer"
- "sound smart"

**Good prompts:**
- "Respond as a friendly pirate to ANY question: use nautical slang, casual tone, always stay in character"
- "Respond like a pragmatic engineer to ANY topic: break problems into steps, explain trade-offs, assume technical knowledge"
- "Talk like a poet: rich metaphors, emotional depth, elegant prose, but never pretentious or flowery"

**What to include:**
1. **Role/persona** — who should the model act as?
2. **Key traits** — what defines their voice? (tone, vocabulary, approach)
3. **Scope** — "to ANY question" (learns across all topics) vs. "to medical questions" (stays in domain)
4. **Examples** — what should they do/avoid?

See [data_generation_tutorial.md](data_generation_tutorial.md#crafting-effective-prompts) for more examples and a checklist.

---

## 3. Your first run

```bash
uv run fastsft "a pirate-themed customer support chatbot" \
  --num-samples 20 \
  --child-model-id "Qwen/Qwen2.5-0.5B-Instruct" \
  --local --max-epochs 1
```

What each part means:

- `"a pirate-themed customer support chatbot"` — describe the *style and
  domain* you want. Be specific; the parent model takes this literally.
- `--num-samples 20` — how many training examples to generate. Start small
  (20–50) so you can iterate fast, then scale up.
- `--child-model-id` — the small model you're fine-tuning, as a
  [Hugging Face](https://huggingface.co) repo id. `Qwen2.5-0.5B-Instruct` is a
  great default to start with: tiny, fast, and trains on a laptop.
- `--local` — train on *this* machine instead of the cloud. Perfect for quick iteration.
- `--max-epochs 1` — a ceiling on training passes (it stops early on its own).
  Keep it low for a quick first run.

When it finishes, your model lands in `modelsets/<timestamp>/` as a LoRA
adapter (`adapter_config.json` + `adapter_model.safetensors`).

> **Tip:** the very first run downloads the child model's weights, so give it a
> minute. Later runs reuse the cache.

---

## 4. Look at what it made

Never trust a dataset you haven't eyeballed. Preview the latest run:

```bash
uv run python -m fastsft.data.viewer               # the raw generated Q&A
uv run python -m fastsft.data.viewer --formatted   # the same data, chat-formatted
```

You'll see the parent model's questions and pirate-flavored answers in a nice
terminal panel. If the style is off, tweak your prompt and regenerate — this is
the fastest, cheapest thing to iterate on.

---

## 5. What's happening under the hood (optional)

You don't need this to use FastSFT, but it helps when you're explaining or debugging what happened.

**DataGenerator** does four things:

1. **Guide** — a small model reads your prompt and writes the *instructions*
   for everyone else: how the parent should answer, how the judge should
   score, and a list of diverse seed topics so your data isn't repetitive.
2. **PromptGenerator** — expands those seeds into `--num-samples` user
   questions, spanning simple to complex.
3. **ResponseGenerator** — the parent model answers each question *in your
   style*.
4. **DataRefiner** — the judge model scores every answer 0–10, throws out the
   weak ones, and regenerates them. That's your quality filter.

**DataFormatter** renders each Q&A into the exact chat format your child model
expects (using its own tokenizer's chat template).

**FineTuner** runs [LoRA](https://arxiv.org/abs/2106.09685) fine-tuning — it
trains a small set of adapter weights instead of the whole model, which is why
it's fast and fits in modest memory. It holds out a slice of your data for
validation and stops training automatically when it stops improving.

---

## 6. Knobs worth turning early

| Flag | Why you'd use it |
|------|------------------|
| `--num-samples 100` | More data = better model. Bump it up once your style looks right. |
| `--child-model-id ...` | Pick your target model. Must be an *instruct/chat* model, not a base model. |
| `--parent-model ...` | The model writing your data. Default is Llama 3.3 70B; swap for cheaper/faster. |
| `--score-threshold 7` | Raise the quality bar (0–10). Higher = stricter filtering, fewer but better samples. |
| `--max-epochs 3` | Let it train longer for a tiny dataset. |
| `--lora-rank 32` | Bigger adapter = more capacity to learn (needs `--local` or `--gpu-tier`). |

Run `uv run fastsft --help` to see every flag.

> **Gotcha:** the parent model must be **open-weight** (FastSFT checks this).
> Closed models' terms of service usually forbid training on their outputs, so
> FastSFT refuses them with a clear error. The defaults are all fine.

---

## 7. Training on cloud GPUs (when your laptop isn't enough)

If your child model is too big to train locally, FastSFT can dispatch training
to [Modal](https://modal.com) (pay-per-second cloud GPUs):

```bash
modal token new          # authenticate once
uv run fastsft "..." --num-samples 100   # no --local → trains on Modal
```

Without `--gpu-tier`, FastSFT runs a **cost heuristic**: it estimates each GPU
tier's memory need from your model's real size and your data's real length,
then picks the *cheapest tier that fits*. Want to preview that before spending
anything?

```bash
uv run python -m fastsft.training.heuristic Qwen/Qwen2.5-0.5B-Instruct
```

This prints a plain-English explanation of every training knob plus a
cost-ranked shortlist — a great sanity check before a real run.

To force a specific GPU: `--gpu-tier A100-40GB`.

---

## 8. Inspect, Edit, and Resume

Data generation is expensive (API calls cost $). Before spending more, inspect and fix:

```bash
# 1. View the raw Q&A pairs
uv run python -m fastsft.data.viewer

# 2. If quality is good → skip to training
uv run fastsft --start-stage fine_tuner --input-path datasets/formatted/<timestamp> --local

# 3. If quality needs fixing → edit the Parquet file and resume
python3 << 'EOF'
from datasets import load_dataset

# Load the raw data
ds = load_dataset("parquet", data_files="datasets/raw/<timestamp>/train/data-00000-of-00001.parquet")
df = ds["train"].to_pandas()

# Remove or fix bad samples
df = df.drop([5, 12, 23])  # Remove samples 5, 12, 23 (bad quality)
df.loc[0, "generation"] = "Better answer here..."  # Fix sample 0

# Save back
df.to_parquet("datasets/raw/<timestamp>/train/data-00000-of-00001.parquet")
EOF

# 4. Resume from formatting (skip generation, format the edited data)
uv run fastsft --start-stage data_formatter --input-path datasets/raw/<timestamp>

# 5. Then train
uv run fastsft --start-stage fine_tuner --input-path datasets/formatted/<timestamp> --local
```

This workflow saves $1-2 per iteration by avoiding regeneration. See
[data_generation_tutorial.md](data_generation_tutorial.md#advanced-manually-edit--resume)
for the full guide.

---

## 9. Recommended playbook

A battle-tested order of operations when you're short on time:

1. **Prove the loop first.** Run §3 with `--num-samples 10 --local
   --max-epochs 1`. Get *a* model out end-to-end before optimizing anything.
2. **Fix the style, cheaply.** Iterate on your prompt + `--score-threshold`,
   checking `fastsft.data.viewer` each time. Don't train while doing this.
3. **Scale the data.** Once the generated answers look right, bump
   `--num-samples` to 100+.
4. **Train for real.** Re-run from `--start-stage fine_tuner` with a couple
   epochs. Save the earlier stages' work.
5. **Try it out.** Load the adapter from `modelsets/<timestamp>/` with
   `transformers` + `peft` and see your tiny model doing the thing.

---

## 10. When something breaks

| Symptom | Fix |
|---------|-----|
| `No OpenRouter API key found` | Create `.env` with `OPENROUTER_API_KEY=...` (§2). |
| `... has no chat_template` | Your `--child-model-id` is a base model. Use an `-Instruct`/`-Chat` version. |
| `... has no hugging_face_id` | Your `--parent-model` is closed-weight. Use an open one (the default is fine). |
| `--strategy qlora requires CUDA` | QLoRA needs an NVIDIA GPU. Use plain `lora` locally, or train on Modal. |
| `modal.AuthError` | Run `modal token new`, or add `--local` to skip the cloud. |
| Structured-output / empty-response errors | The guide/judge model needs tool-call support. Stick to the defaults. |

---

**That's the whole tool.** Describe a model, generate data, fine-tune, try it out.
Now go build something. 🏴‍☠️
