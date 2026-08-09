# FastSFT

A CLI tool that distills a large LLM into a small one via LoRA/QLoRA
fine-tuning on [Modal](https://modal.com). Given a freeform description of
the dataset you want and a target ("child") model, it derives its own
generation/judging instructions, generates a varied and quality-filtered
dataset, renders it into the child model's chat format, and fine-tunes it —
sensible defaults for a first run, full override control via config objects
and CLI flags for everything else.

Built on [distilabel](https://github.com/argilla-io/distilabel) (generation
engine), [OpenRouter](https://openrouter.ai) (LLM backend), Modal (remote GPU
training), and Hugging Face `transformers` (chat templates, tokenization).

`DistillationPipeline` (`pipeline.py`) runs three composable stages in
sequence, each independently usable and skippable via `start_stage`:

1. **DataGenerator** — prompt in, quality-filtered raw dataset out.
2. **DataFormatter** — raw dataset in, chat-template-rendered dataset out.
3. **FineTuner** — formatted dataset in, fine-tuned LoRA/QLoRA adapter out,
   trained remotely on Modal, or locally on your own machine via `--local`.

A separate **evaluation module** (`eval/`, run after training) scores a trained
adapter against its untuned base and its parent teacher — LLM-judge win rates
plus embedding similarity. See [Evaluation](#evaluation).

New here? [`TUTORIAL.md`](TUTORIAL.md) is a gentler, task-first walkthrough
(zero to a trained model in ~15 minutes). This README is the reference.

## Quick start

```bash
# .env holds OPENROUTER_API_KEY=sk-or-...; `modal token new` once for Modal auth
uv run main.py "a pirate-themed customer support chatbot" --num-samples 50 --child-model-id "Qwen/Qwen2.5-0.5B-Instruct"

# preview the latest run
uv run python -m data.viewer               # raw `messages`
uv run python -m data.viewer --formatted   # rendered `text`

# inspect a finished training run's loss curve + telemetry (latest under modelsets/)
uv run python -m training.stats_viewer
uv run python -m training.stats_viewer modelsets/<timestamp>
uv run python -m training.stats_viewer --json    # machine-readable

# preview training-config options for a model before running anything
uv run python -m training.heuristic Qwen/Qwen2.5-0.5B-Instruct
uv run python -m training.heuristic Qwen/Qwen2.5-0.5B-Instruct --input-path datasets/formatted/<timestamp>

# start mid-pipeline from a saved dataset
uv run main.py --start-stage data_formatter --input-path datasets/raw/<timestamp>

# override the training config directly (skips the cost heuristic)
uv run main.py "..." --gpu-tier A100-80GB --strategy qlora --lora-rank 32 --max-epochs 5

# train locally instead of on Modal (needs `uv sync --extra local-training` once)
uv run main.py "..." --local --max-epochs 2

# evaluate the trained adapter (needs `uv sync --extra evaluation` once)
uv run python -m eval.run                       # latest adapter under modelsets/
uv run python -m eval.results_viewer            # render the win rates + takeaways
uv run python -m eval.inference_viewer "hi"     # spot-check tuned vs untuned on one prompt
```

`main.py` saves each stage's output the moment it completes —
`datasets/raw/<timestamp>/`, `datasets/formatted/<timestamp>/`,
`modelsets/<timestamp>/` — sharing one timestamp per run, so a later stage's
failure never loses an earlier one's saved output.

## Stages

### DataGenerator (`stages/data_generator.py`)

`run(prompt) -> Distiset` with a `messages` column. Four steps:

1. **Guide** derives `parent_instruction` (answer persona/style),
   `judge_instruction` (scoring rubric), and `sample_instructions` (diverse
   seed topics).
2. **PromptGenerator** expands the seeds into exactly `num_samples` user
   instructions: seeds give breadth (distinct topics), each seed is expanded
   into depth (simple→complex variants). Breadth `B = ceil(N ** breadth_exponent)`
   (default 2/3, so breadth-leaning); samples are allocated evenly across
   seeds, and any per-row under-delivery is topped up (or raises rather than
   silently returning fewer).
3. **ResponseGenerator** generates one answer per instruction via the parent
   model with `parent_instruction` applied, so answers carry the persona (one
   API call per instruction, not `n > 1`).
4. **DataRefiner** scores each answer 0-10, drops those below
   `score_threshold`, re-answers the same instruction, and loops up to
   `MAX_REFINE_ITERATIONS`.

Rows are converted to the `messages` schema (`[{role, content}, ...]`) before
returning, keeping `DataFormatter` decoupled from internal column names.

### DataFormatter (`stages/data_formatter.py`)

`run(distiset) -> Distiset`: adds a `text` column by rendering each row's
`messages` through the child model's own chat template
(`AutoTokenizer.from_pretrained(child_model_id).apply_chat_template(..., tokenize=False)`).
The tokenizer is loaded lazily on first run. Raises `ValueError` if the child
model ships no `chat_template` (e.g. a base model). Depends only on the
`messages` column, so any stage or hand-supplied dataset producing it works.

### FineTuner (`stages/fine_tuner.py`)

`run(formatted_distiset) -> str` (path to the trained adapter). Three steps:

1. **Resolve a `TrainingConfig`** — the caller-supplied one if given;
   otherwise, with `--local`, defaults (no Modal tier to rank — there's only
   one "tier": this machine); otherwise rank candidates by cost/feasibility
   via `training.heuristic.recommend_configs` and take the cheapest feasible
   one (logs the shortlist either way). The heuristic estimates memory/cost
   from the child model's real parameter count (Hugging Face Hub metadata, no
   weight download) and the dataset's real sequence lengths — it ranks by
   *feasibility*, never quality, which is only knowable empirically.
2. **Split** a validation slice (`loop.validation_split`) from the formatted
   dataset for early stopping to monitor.
3. **Train** via the shared core (`training/trainer.py::run_sft`): loads the
   base model (4-bit quantized if `strategy="qlora"`), wraps it in a
   `peft.LoraConfig`, trains via TRL's `SFTTrainer` with early stopping on the
   held-out split — epoch count is never fixed upfront, `loop.max_epochs` is
   only a ceiling. Either **dispatched to Modal** (`training/modal_app.py::train_lora`,
   which then downloads the adapter locally), or, with **`--local`**, run
   directly on this machine (`training/local_trainer.py::train_locally`) —
   auto-detects `cuda`/`mps`/`cpu` and writes straight to a local temp dir, no
   download step. QLoRA locally requires CUDA (`bitsandbytes`); otherwise only
   `lora` is allowed, checked upfront with a clear error.

### Stage (`stages/base.py`)

Shared base: `verbose`/`_log`, a `name` each subclass sets from
`stages/constants.py`, a `run()` template method that validates
(`_validate_input`) then runs (`_run`), and a `save_output(output, run_id)`
hook (base no-ops; `DataGenerator`/`DataFormatter` save a `Distiset` via
`save_distiset`, `FineTuner` copies the downloaded adapter into
`modelsets/<run_id>/`). Per-stage input checks: DataGenerator (non-empty
prompt), DataFormatter (`messages` column), FineTuner (`text` column).

## Configuration

Every tunable knob has a sensible default — `uv run main.py "a prompt"` with
no flags works end to end. Overriding anything routes through the relevant
stage's own config object, never a loose flat argument:

- **`DataGenerationConfig`** (`data/config.py`) — `guide_model`,
  `parent_model`, `judge_model`, `num_samples`, `breadth_exponent`,
  `score_threshold`, and a nested `parent_generation: ParentGenerationConfig`
  (`temperature`, `max_tokens`). CLI: `--guide-model`, `--parent-model`,
  `--judge-model`, `--num-samples`, `--breadth-exponent`, `--score-threshold`,
  `--parent-temperature`, `--parent-max-tokens` — all only used at the
  default `--start-stage`.
- **`TrainingConfig`** (`training/config.py`) — `gpu_tier`, `strategy`
  (`lora`/`qlora`), a nested `adapter: AdapterConfig` (`rank`,
  `target_modules`, `dropout`) and `loop: TrainingLoopConfig` (`batch_size`,
  `grad_accumulation`, `learning_rate`, `max_epochs`, `eval_steps`,
  `early_stopping_patience`, `validation_split`, `mask_prompt_loss`), plus
  `modal_timeout_seconds`. `mask_prompt_loss` (default on) excludes the
  prompt (system+user) tokens from the loss so training/eval loss reflects the
  answer only; `run_sft` picks a model-agnostic way to do this at runtime
  (`completion_only_loss`, else the template's `assistant_only_loss`, else a
  logged fall back to whole-sequence loss). `target_modules` accepts
  `["all-linear"]` to auto-target every linear layer on architectures the
  default attention-projection names don't fit.
  CLI: `--gpu-tier` (dispatch to Modal, skipping the cost heuristic) and
  `--local` (train on this machine instead — needs `uv sync --extra
  local-training`) are the two opt-in triggers, mutually exclusive.
  `--strategy`, `--lora-rank`, `--target-modules`, `--lora-dropout`,
  `--batch-size`, `--grad-accumulation`, `--learning-rate`, `--max-epochs`,
  `--eval-steps`, `--early-stopping-patience`, `--validation-split` take
  effect alongside either one (each falls back to its default if not given);
  `--modal-timeout` only makes sense with `--gpu-tier`.
  `helper.py::validate_training_flags` errors if an override flag is passed
  without one of `--gpu-tier`/`--local`, or if both dispatch targets are
  given together.

Programmatic callers pass the same config objects directly:
`DistillationPipeline(generation=DataGenerationConfig(...), training=TrainingConfig(...))`.
See `training/heuristic.py`'s standalone CLI (below) for exploring
`TrainingConfig` options before committing to a run.

## Architecture

```
constants.py         -- entry-point constants (output layout, model-id defaults)
warnings_filter.py   -- import-time warning suppression (transformers, pydantic)
device.py            -- local torch accelerator/dtype detection (training + eval)
helper.py            -- Distiset load/save/shape + run-folder timestamp helpers
validation_checks.py -- CLI argument validation for main.py / eval.run
findings.py          -- Finding: shared diagnostic record (stdlib; stats + eval)
findings_view.py     -- shared `rich` rendering for Findings (both viewers)
progress.py          -- shared rich console + ProgressLogger (stages, Evaluator, CLIs)
pipeline.py          -- DistillationPipeline: runs STAGE_ORDER from start_stage
stages/
  constants.py       -- stage names: STAGE_ORDER, STAGE_NAMES
  base.py            -- Stage: validate-then-run template, save_output hook
  data_generator.py  -- DataGenerator: guide -> prompts -> answers -> refine
  data_formatter.py  -- DataFormatter: chat-template rendering
  fine_tuner.py      -- FineTuner: resolve config -> split -> train (Modal or local)
model/
  constants.py       -- OpenRouter URLs, max tokens, score threshold, guide prompt
  base.py            -- Model: OpenRouter-backed role
  _logging.py        -- distilabel root-logger cleanup for Model.run_pipeline
  guide.py           -- Guide(Model): derives instructions + seed topics
  judge.py           -- Judge(Model): 0-10 scoring
data/
  constants.py          -- breadth exponent, refine iterations, prompt-gen prompt
  config.py             -- DataGenerationConfig, ParentGenerationConfig
  prompt_generator.py   -- PromptGenerator: seeds -> user instructions (breadth x depth)
  response_generator.py -- ResponseGenerator: instructions -> styled answers
  refiner.py            -- DataRefiner: score -> drop -> re-answer loop
  viewer.py             -- DataViewer: terminal preview
training/
  constants.py       -- Modal GPU tier catalog, LoRA/training defaults
  config.py          -- TrainingConfig, AdapterConfig, TrainingLoopConfig
  heuristic.py       -- recommend_configs: cost/feasibility ranking (+ standalone CLI)
  trainer.py         -- run_sft: shared LoRA/QLoRA SFT core (Modal + local)
  modal_app.py       -- Modal App/Image/Volume + train_lora remote function
  local_trainer.py   -- train_locally (--local, needs local-training extra)
  stats.py           -- core: load/structure/diagnose a run's telemetry (pure-stdlib library)
  stats_viewer.py    -- rich terminal rendering + `python -m` CLI over stats.py (loss curve, --json)
eval/                -- post-training evaluation (needs the `evaluation` extra)
  constants.py       -- eval defaults, judge rubrics, results filename
  config.py          -- EvalConfig: adapter + parent/judge/embedding models, knobs
  run.py             -- `python -m eval.run` CLI: resolve prompts/parent -> Evaluator -> save
  evaluator.py       -- Evaluator: parent/tuned/untuned answers -> judge win rates + similarity
  prompt_set.py      -- EvalPromptSet: held-out eval prompts (generate/persist/load)
  inference.py       -- ChildInferenceEngine: local tuned/untuned generation (core)
  inference_viewer.py-- rich `python -m` spot-check over inference.py
  embeddings.py      -- local sentence embeddings for parent-similarity
  results.py         -- core: persist/load/interpret results (pure-stdlib library)
  results_viewer.py  -- rich terminal rendering + `python -m` CLI over results.py (--json)
main.py              -- CLI entry point
```

### Model (`model/base.py`)

Base for any OpenRouter-backed role. `model_id` is public; `_api_key` /
`_temperature` / `_max_tokens` are protected. `set_instruction` /
`get_instruction` manage the per-instance system prompt. `build_llm()` and
`run_pipeline()` (shared load-rows + `TextGeneration`) drive OpenRouter.
`assert_structured_output()` raises a clear error if a structured-output call
returns empty.

Open-weight check: on first use (`build_llm`), rejects any model without a
`hugging_face_id` on OpenRouter — the open-weight signal, since closed-weight
ToS usually forbid training on outputs. Applies to the parent only; `Guide`
and `Judge` set `_enforce_open_weight = False` (their output isn't training
data). The catalog fetch is `@lru_cache`d. Constructors do no I/O.

### DistillationPipeline (`pipeline.py`)

Runs `STAGE_ORDER` (`["data_generator", "data_formatter", "fine_tuner"]`) from
`start_stage`, threading each output into the next. Only stages from
`start_stage` onward are constructed (so `start_stage="fine_tuner"` never
fetches a tokenizer). `run()` is a generator that **yields `(stage, output)`
as each stage completes**, so `main.py` persists each output the moment it's
produced (via `stage.save_output`) — a later stage's failure can't lose an
earlier stage's saved output.

Constructor takes config objects, not a flat argument list, so adding a knob
never touches `DistillationPipeline`'s own signature:

- `child_model_id`, `start_stage`, `verbose` — shared/orchestration, stay
  top-level (`child_model_id` is used by both `DataFormatter` and `FineTuner`,
  so it can't live inside either stage's own config without risking the two
  disagreeing).
- `generation: Optional[DataGenerationConfig]` — `None` uses static defaults;
  only relevant when `start_stage="data_generator"`.
- `training: Optional[TrainingConfig]` — `None` lets `FineTuner`'s cost
  heuristic decide; supply one to skip it.

- `start_stage="data_generator"` (default): `run(prompt)`.
- `="data_formatter"`: `run(raw_dataset)` — needs a `messages` column.
- `="fine_tuner"`: `run(formatted_dataset)` — needs a `text` column.

### Guide / Judge / generators

- **Guide**: `generate_instructions(user_input, num_seeds) -> GuideInstructions`
  (`parent_instruction`, `judge_instruction`, `sample_instructions`).
- **Judge**: `score_samples(id->text) -> id->score` (0-10, structured output);
  `failed_sample_count`.
- **PromptGenerator**: `generate(seeds) -> List[str]`; `seed_count(N, breadth_exponent)`
  gives the breadth.
- **ResponseGenerator**: `generate(instructions) -> Distiset`.
- **DataRefiner**: `refine(distiset, threshold) -> Distiset`.

### DataViewer (`data/viewer.py`)

`python -m data.viewer [--formatted] [--input-path ...] [--num-samples N]`. Defaults
to the latest timestamped run under `datasets/raw/` (or `datasets/formatted/`).
Run as a module, not a bare script — it imports project-root modules.

### Training config heuristic (`training/heuristic.py`)

Also runnable standalone:

```bash
uv run python -m training.heuristic <child_model_id> [--input-path datasets/formatted/<ts>] [--top-n N]
```

Prints a plain-English explanation of every `TrainingConfig` knob
(`KNOB_DESCRIPTIONS`) followed by a cost-ranked shortlist for that model —
using a real formatted dataset's sequence lengths (`--input-path`) or a rough
fallback length if none given. Lets you explore options before running
`DataGenerator`/`DataFormatter`/`FineTuner` at all, or spending anything on
Modal.

## Evaluation

After training, `eval/` scores an adapter against two baselines — its own
untuned base and the parent teacher — needing `uv sync --extra evaluation`
(torch, peft, accelerate, sentence-transformers). Like the stages, the core
logic is kept free of `rich` and split from its presentation layer.

```bash
uv run python -m eval.run [adapter_dir] [--num-eval-prompts N] [--no-swap]
uv run python -m eval.results_viewer [adapter_dir] [--json]
uv run python -m eval.inference_viewer "a prompt" [adapter_dir] [--tuned-only]
```

`eval.run` (`eval/run.py`) resolves the eval prompt set (reuse the latest saved
one for comparability, or generate + persist a fresh one seeded from the
training questions and deduped against them to prevent leakage), reconstructs
the parent teacher from the run's `training_metadata.json` sidecar (identity,
style prompt, and generation recipe — so the reference answers like the *actual*
teacher), runs the `Evaluator`, and writes `eval_results.json` next to the
adapter. For each prompt the `Evaluator` (`eval/evaluator.py`) collects three
answers — parent (OpenRouter), tuned and untuned child (both local, one base
load with the adapter toggled via `PeftModel.disable_adapter()`) — and reports:

- **Tuned vs untuned** — the primary signal: did fine-tuning improve quality?
- **Parent-style match** — the distillation objective: is the tuned child more
  like the parent's style than untuned? (reference-judged against the parent).
- **Tuned vs parent** — the remaining gap to the teacher.
- **Embedding similarity to parent** — distillation fidelity in embedding space.

Each pair is judged in both A/B orders to cancel the judge's position bias
(disable with `--no-swap`), and win rates are reported against a sample-size
noise floor so a thin eval set isn't over-read. `eval/results.py` turns the raw
numbers into plain-English takeaways (the same core/presentation split as
`training/stats.py`), and `eval.results_viewer --json` emits them
machine-readably.

## Known pitfalls

- **`n > 1` ignored**: many OpenRouter providers ignore the multi-generation
  param; we generate one row per sample.
- **Structured output returns `None`**: usually truncation
  (`DEFAULT_MAX_TOKENS = 1024`; distilabel's default 128 truncates JSON) or a
  model that doesn't reliably support tool calls. Surfaced via
  `assert_structured_output`.
- **`warnings_filter` must import first**: it suppresses import-time warnings
  and sets `TRANSFORMERS_NO_ADVISORY_WARNINGS`, so every entry point imports
  it before distilabel/transformers.
- **`No module named 'bs4'`**: harmless — distilabel optionally scrapes arxiv
  for citations; unused here.
- **Timestamped save folders**: raw and formatted saves compute their
  timestamps independently, so they can differ by a second across a boundary
  (benign, back-to-back in practice).
- **Python 3.12**: required (distilabel's instructor integration uses
  `typing.TypeAlias`).
- **Modal auth required for FineTuner to actually train**: `modal` is a local
  dependency, but dispatching still needs `modal token new` run once. Without
  it, the pipeline runs through DataGenerator/DataFormatter and FineTuner's
  config resolution fine, then fails with a clear `modal.AuthError` at the
  dispatch call.
- **`.with_options(gpu=..., timeout=...)` per-call override** (`stages/fine_tuner.py`):
  verify this against your installed `modal` SDK version — Modal's per-call
  resource override API has changed across versions.
- **`MODAL_GPU_TIERS` pricing/VRAM figures are illustrative**
  (`training/constants.py`) — verify against modal.com/pricing; both the tier
  list and pricing drift over time.
- **CLI training flags use `arg if arg is not None else DEFAULT`**, not
  `arg or DEFAULT` — several (e.g. `--lora-dropout 0`) have a meaningful `0`/
  `0.0` value that `or` would silently discard.
- **`--local` needs `uv sync --extra local-training`** (torch, peft, trl,
  accelerate) — kept out of the main dependency list so Modal-only users keep
  a lean install. `bitsandbytes` isn't included; QLoRA locally only works if
  CUDA is detected (`training/local_trainer.py::detect_device`), otherwise
  `--strategy qlora --local` errors upfront rather than failing deep inside
  `bitsandbytes`.

## Defaults

- `DEFAULT_PARENT_MODEL = meta-llama/llama-3.3-70b-instruct`
- `DEFAULT_JUDGE_MODEL = deepseek/deepseek-chat` — different family from the
  parent, to avoid self-preference bias.
- `DEFAULT_GUIDE_MODEL = qwen/qwen-2.5-7b-instruct` — small; must support tool
  calls.
- `DEFAULT_CHILD_MODEL_ID = Qwen/Qwen2.5-0.5B-Instruct` — a Hugging Face repo
  id, not an OpenRouter id; used by DataFormatter/FineTuner via `AutoTokenizer`.
- Training defaults (GPU tier catalog, LoRA rank/dropout, epochs, etc.) live
  in `training/constants.py` — run `uv run python -m training.heuristic
  <child_model_id>` for a live, explained view of each one.

When changing a model, verify tool-call support empirically through the real
code path — OpenRouter's listing isn't reliable on its own.

## Environment

- `.env` holds `OPENROUTER_API_KEY` (gitignored; `.env.example` is the template).
- Python 3.12 via `uv` (`uv run main.py ...`).
- `modal` is a local dependency (client SDK only) — run `modal token new` once
  to authenticate. By default, training runs on Modal's infra: `torch`,
  `peft`, `trl`, `bitsandbytes`, `accelerate` are declared in the Modal
  `Image` (`training/modal_app.py`) and not required locally.
- For `--local` training, install the same stack locally instead:
  `uv sync --extra local-training` (torch, peft, trl, accelerate;
  `bitsandbytes`/QLoRA stays CUDA-only, so local QLoRA needs a CUDA machine).
- For the evaluation module, `uv sync --extra evaluation` (torch, peft,
  accelerate, sentence-transformers) — local child inference plus embeddings.
- Key local deps: `distilabel[instructor,openai]`, `pydantic`,
  `python-dotenv`, `rich`, `requests`, `datasets`, `transformers`, `modal`.

## Development

Lint before pushing — the same check CI runs (`.github/workflows/lint.yml`):

```bash
uv run --only-group dev ruff check .        # lint (installs only ruff)
uv run --only-group dev ruff check . --fix  # apply safe autofixes
```

`ruff` is pinned (`<0.17`) in `pyproject.toml`'s `dev` dependency group and the
enabled rule set is declared explicitly under `[tool.ruff.lint]`, so the lint
result is reproducible rather than tracking ruff's evolving defaults. `--only-group
dev` installs just `ruff`, not the ML stack, so the check stays fast.
