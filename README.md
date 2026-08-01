# LLM-Distillator

A CLI tool that distills a large LLM into a small one. Given a freeform
description of the dataset you want and a target ("child") model, it derives
its own generation/judging instructions, generates a varied and
quality-filtered dataset, renders it into the child model's chat format, and
(eventually) fine-tunes it.

Built on [distilabel](https://github.com/argilla-io/distilabel) (generation
engine), [OpenRouter](https://openrouter.ai) (LLM backend), and Hugging Face
`transformers` (chat templates).

`DistillationPipeline` (`pipeline.py`) runs three composable stages in
sequence, each independently usable and skippable via `start_stage`:

1. **DataGenerator** — prompt in, quality-filtered raw dataset out.
2. **DataFormatter** — raw dataset in, chat-template-rendered dataset out.
3. **FineTuner** — formatted dataset in, fine-tuned model out. *Scaffold only.*

## Quick start

```bash
# .env holds OPENROUTER_API_KEY=sk-or-...
uv run main.py "a pirate-themed customer support chatbot" --num-samples 50 --child-model-id "Qwen/Qwen2.5-0.5B-Instruct"

# preview the latest run
uv run python -m data.viewer               # raw `messages`
uv run python -m data.viewer --formatted   # rendered `text`

# start mid-pipeline from a saved dataset
uv run main.py --start-stage data_formatter --input-path datasets/raw/<timestamp>
```

`FineTuner` raises `NotImplementedError`; `main.py` catches it and still saves
`DataGenerator`'s output to `datasets/raw/<timestamp>/` and `DataFormatter`'s
to `datasets/formatted/<timestamp>/`.

## Stages

### DataGenerator (`stages/data_generator.py`)

`run(prompt) -> Distiset` with a `messages` column. Four steps:

1. **Guide** derives `parent_instruction` (answer persona/style),
   `judge_instruction` (scoring rubric), and `sample_instructions` (diverse
   seed topics).
2. **PromptGenerator** expands the seeds into exactly `num_samples` user
   instructions: seeds give breadth (distinct topics), each seed is expanded
   into depth (simple→complex variants). Breadth `B = ceil(N ** BREADTH_EXPONENT)`
   (2/3, so breadth-leaning); samples are allocated evenly across seeds, and
   any per-row under-delivery is topped up (or raises rather than silently
   returning fewer).
3. **ResponseGenerator** generates one answer per instruction via the parent
   model with `parent_instruction` applied, so answers carry the persona (one
   API call per instruction, not `n > 1`).
4. **DataRefiner** scores each answer 0-10, drops those below
   `DEFAULT_SCORE_THRESHOLD`, re-answers the same instruction, and loops up to
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

### FineTuner (`stages/fine_tuner.py`) — scaffold

`run(formatted_distiset)` raises `NotImplementedError`. Contract: the input
has a `text` column rendered for `child_model_id`; the training framework and
hyperparameters are a separate follow-up.

### Stage (`stages/base.py`)

Shared base: `verbose`/`_log`, a `name` each subclass sets from
`stages/constants.py`, and a `run()` template method that validates
(`_validate_input`) then runs (`_run`). Per-stage input checks: DataGenerator
(non-empty prompt), DataFormatter (`messages` column), FineTuner (`text` column).

## Architecture

```
constants.py         -- entry-point constants (output layout, model-id defaults)
warnings_filter.py   -- import-time warning suppression (transformers, pydantic)
helper.py            -- CLI load/save/validate/timestamp helpers
pipeline.py          -- DistillationPipeline: runs STAGE_ORDER from start_stage
stages/
  constants.py       -- stage names: STAGE_ORDER, STAGE_NAMES
  base.py            -- Stage: validate-then-run template
  data_generator.py  -- DataGenerator: guide -> prompts -> answers -> refine
  data_formatter.py  -- DataFormatter: chat-template rendering
  fine_tuner.py      -- FineTuner: training (scaffold)
model/
  constants.py       -- OpenRouter URLs, max tokens, score threshold, guide prompt
  base.py            -- Model: OpenRouter-backed role
  guide.py           -- Guide(Model): derives instructions + seed topics
  judge.py           -- Judge(Model): 0-10 scoring
data/
  constants.py          -- breadth exponent, refine iterations, prompt-gen prompt
  prompt_generator.py   -- PromptGenerator: seeds -> user instructions (breadth x depth)
  response_generator.py -- ResponseGenerator: instructions -> styled answers
  refiner.py            -- DataRefiner: score -> drop -> re-answer loop
  viewer.py             -- DataViewer: terminal preview
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
produced (via `stage.save_output`) — a later stage's failure (e.g. FineTuner's
`NotImplementedError`) can't lose an earlier stage's saved output.

- `start_stage="data_generator"` (default): `run(prompt)`.
- `="data_formatter"`: `run(raw_dataset)` — needs a `messages` column.
- `="fine_tuner"`: `run(formatted_dataset)` — needs a `text` column.

### Guide / Judge / generators

- **Guide**: `generate_instructions(user_input, num_seeds) -> GuideInstructions`
  (`parent_instruction`, `judge_instruction`, `sample_instructions`).
- **Judge**: `score_samples(id->text) -> id->score` (0-10, structured output);
  `failed_sample_count`.
- **PromptGenerator**: `generate(seeds) -> List[str]`; `seed_count(N)` gives
  the breadth.
- **ResponseGenerator**: `generate(instructions) -> Distiset`.
- **DataRefiner**: `refine(distiset) -> Distiset`.

### DataViewer (`data/viewer.py`)

`python -m data.viewer [--formatted] [--path ...] [--num-samples N]`. Defaults
to the latest timestamped run under `datasets/raw/` (or `datasets/formatted/`).
Run as a module, not a bare script — it imports project-root modules.

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

## Model defaults

- `DEFAULT_PARENT_MODEL = meta-llama/llama-3.3-70b-instruct`
- `DEFAULT_JUDGE_MODEL = deepseek/deepseek-chat` — different family from the
  parent, to avoid self-preference bias.
- `DEFAULT_GUIDE_MODEL = qwen/qwen-2.5-7b-instruct` — small; must support tool
  calls.
- `DEFAULT_CHILD_MODEL_ID = Qwen/Qwen2.5-0.5B-Instruct` — a Hugging Face repo
  id, not an OpenRouter id; used by DataFormatter/FineTuner via `AutoTokenizer`.

When changing a model, verify tool-call support empirically through the real
code path — OpenRouter's listing isn't reliable on its own.

## Environment

- `.env` holds `OPENROUTER_API_KEY` (gitignored; `.env.example` is the template).
- Python 3.12 via `uv` (`uv run main.py ...`).
- Key deps: `distilabel[instructor,openai]`, `pydantic`, `python-dotenv`,
  `rich`, `requests`, `datasets`, `transformers`. PyTorch isn't needed until
  FineTuner's training lands.
