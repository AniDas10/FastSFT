# FastSFT

**Teach a small, weak model to answer like a big, smart one.**

- 🧠 **Parent** — a big, powerful model. The teacher.
- 🐣 **Child** — a small, weak model. The student you're upgrading.

```
Parent  ──ask tough questions──▶  Q&A dataset  ──train──▶  Child
(big,        (judge filters                      (small,
 smart)       out weak answers)                    fast)
```

Ask the Parent hard questions in the style you want, keep only the good answers, and train the Child on them. Result: a tiny model that mimics the big one's style — for a fraction of the size and cost. This is called **distillation**.

**Why bother, instead of just calling the Parent every time?** Cheaper (no per-token API bill), fits on your laptop, and yours to keep — no rate limits, no vendor lock-in.

---

## Run it

```bash
uv sync --extra local-training
echo "OPENROUTER_API_KEY=sk-or-..." > .env   # free key at openrouter.ai
uv run trial_run.py
```

That's the whole pipeline — generate data, filter it, and train — running end to end with sensible defaults.

Open [`trial_run.py`](trial_run.py) to play around: change the `PROMPT` to whatever persona you want, swap the `CHILD_MODEL_ID`, tweak a config value. Every field is commented with what it does and what else you could try. Edit, save, rerun.

---

## Want more?

- **[TUTORIAL.md](TUTORIAL.md)** — a slower, narrated walkthrough (CLI usage, troubleshooting, cloud training).
- **[data_generation_tutorial.md](data_generation_tutorial.md)**, **[training_tutorial.md](training_tutorial.md)**, **[evaluation_tutorial.md](evaluation_tutorial.md)** — deep dives on each pipeline stage.

---

Built on [distilabel](https://github.com/argilla-io/distilabel), [OpenRouter](https://openrouter.ai), Modal, Hugging Face, and PEFT.

Personal project, built for fun to learn how these models work under the hood. Have fun with it — not actively maintained.
