# LLM-Distillator

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
   trained remotely on Modal.

## Quick start

```bash
# .env holds OPENROUTER_API_KEY=sk-or-...; `modal token new` once for Modal auth
uv run main.py "a pirate-themed customer support chatbot" --num-samples 50 --child-model-id "Qwen/Qwen2.5-0.5B-Instruct"

# preview the latest run
uv run python -m data.viewer               # raw `messages`
uv run python -m data.viewer --formatted   # rendered `text`

# preview training-config options for a model before running anything
uv run python -m training.heuristic Qwen/Qwen2.5-0.5B-Instruct
uv run python -m training.heuristic Qwen/Qwen2.5-0.5B-Instruct --input-path datasets/formatted/<timestamp>

# start mid-pipeline from a saved dataset
uv run main.py --start-stage data_formatter --input-path datasets/raw/<timestamp>

# override the training config directly (skips the cost heuristic)
uv run main.py "..." --gpu-tier A100-80GB --strategy qlora --lora-rank 32 --max-epochs 5
```

`main.py` saves each stage's output the moment it completes —
`datasets/raw/<timestamp>/`, `datasets/formatted/<timestamp>/`,
`models/<timestamp>/` — sharing one timestamp per run, so a later stage's
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

`run(formatted_distiset) -> str` (path to the downloaded adapter). Three steps:

1. **Resolve a `TrainingConfig`** — the caller-supplied one if given;
   otherwise rank candidates by cost/feasibility via
   `training.heuristic.recommend_configs` and take the cheapest feasible one
   (logs the shortlist either way). The heuristic estimates memory/cost from
   the child model's real parameter count (Hugging Face Hub metadata, no
   weight download) and the dataset's real sequence lengths — it ranks by
   *feasibility*, never quality, which is only knowable empirically.
2. **Split** a validation slice (`loop.validation_split`) from the formatted
   dataset for early stopping to monitor.
3. **Dispatch to Modal** (`training/modal_app.py::train_lora`): loads the
   base model (4-bit quantized if `strategy="qlora"`), wraps it in a
   `peft.LoraConfig`, trains via TRL's `SFTTrainer` with early stopping on the
   held-out split — epoch count is never fixed upfront, `loop.max_epochs` is
   only a ceiling — then downloads the trained adapter locally.

### Stage (`stages/base.py`)

Shared base: `verbose`/`_log`, a `name` each subclass sets from
`stages/constants.py`, a `run()` template method that validates
(`_validate_input`) then runs (`_run`), and a `save_output(output, run_id)`
hook (base no-ops; `DataGenerator`/`DataFormatter` save a `Distiset` via
`save_distiset`, `FineTuner` copies the downloaded adapter into
`models/<run_id>/`). Per-stage input checks: DataGenerator (non-empty
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
  `early_stopping_patience`, `validation_split`), plus `modal_timeout_seconds`.
  CLI: `--gpu-tier` is the opt-in trigger that skips the cost heuristic
  entirely; `--strategy`, `--lora-rank`, `--target-modules`, `--lora-dropout`,
  `--batch-size`, `--grad-accumulation`, `--learning-rate`, `--max-epochs`,
  `--eval-steps`, `--early-stopping-patience`, `--validation-split`,
  `--modal-timeout` only take effect alongside `--gpu-tier` (each falls back
  to its default if not given; `helper.py::validate_training_flags` errors if
  any is passed without `--gpu-tier`).

Programmatic callers pass the same config objects directly:
`DistillationPipeline(generation=DataGenerationConfig(...), training=TrainingConfig(...))`.
See `training/heuristic.py`'s standalone CLI (below) for exploring
`TrainingConfig` options before committing to a run.

## Architecture

```
constants.py         -- entry-point constants (output layout, model-id defaults)
warnings_filter.py   -- import-time warning suppression (transformers, pydantic)
helper.py            -- CLI load/save/validate/timestamp helpers
pipeline.py          -- DistillationPipeline: runs STAGE_ORDER from start_stage
stages/
  constants.py       -- stage names: STAGE_ORDER, STAGE_NAMES
  base.py            -- Stage: validate-then-run template, save_output hook
  data_generator.py  -- DataGenerator: guide -> prompts -> answers -> refine
  data_formatter.py  -- DataFormatter: chat-template rendering
  fine_tuner.py      -- FineTuner: resolve config -> split -> dispatch to Modal
model/
  constants.py       -- OpenRouter URLs, max tokens, score threshold, guide prompt
  base.py            -- Model: OpenRouter-backed role
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
  modal_app.py       -- Modal App/Image/Volume + train_lora remote function
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

`python -m data.viewer [--formatted] [--path ...] [--num-samples N]`. Defaults
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
  to authenticate. Training itself runs on Modal's infra: `torch`, `peft`,
  `trl`, `bitsandbytes`, `accelerate` are declared only in the Modal `Image`
  (`training/modal_app.py`), never installed locally.
- Key local deps: `distilabel[instructor,openai]`, `pydantic`,
  `python-dotenv`, `rich`, `requests`, `datasets`, `transformers`, `modal`.
