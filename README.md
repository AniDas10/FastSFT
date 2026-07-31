# LLM-Distillator

A CLI tool that distills a large LLM into a small one via fine-tuning, using
[distilabel](https://github.com/argilla-io/distilabel) as the dataset-
generation engine, [OpenRouter](https://openrouter.ai) as the LLM backend for
generation/judging, and Hugging Face `transformers` for chat-template
rendering. Given a freeform description of the dataset you want and a target
("child") model to fine-tune, it derives its own generation/judging
instructions, generates and quality-filters samples, renders them into that
child model's exact chat format, and (eventually) fine-tunes it.

The top-level `DistillationPipeline` (`pipeline.py`) is three composable
mini-pipelines, run in sequence, each independently usable and each
skippable if you want to bring your own data at that point:

1. **DataGenerator** (`stages/data_generator.py`) — prompt in, quality-filtered raw dataset out.
2. **DataFormatter** (`stages/data_formatter.py`) — raw dataset in, chat-template-rendered dataset out.
3. **FineTuner** (`stages/fine_tuner.py`) — formatted dataset in, fine-tuned child model out. **Scaffold only — not yet implemented** (see below).

## Quick start

```bash
# .env holds OPENROUTER_API_KEY=sk-or-...
uv run main.py "a pirate-themed customer support chatbot" --num-samples 50 --child-model-id "Qwen/Qwen2.5-0.5B-Instruct"

# preview the most recent run (auto-detects the latest timestamped folder)
uv run python -m data.view               # raw `messages`
uv run python -m data.view --formatted   # rendered `text` column
```

Since `FineTuner` isn't implemented yet, a default end-to-end run will
currently raise `NotImplementedError` once it reaches that stage — this is
expected (see "Mini-pipeline stages" below), and `main.py` catches it so it
can still save what was produced. `DataGenerator`'s output is saved to
`datasets/raw/<timestamp>/` and `DataFormatter`'s to
`datasets/formatted/<timestamp>/` (same timestamp for a given run, so the two
stay associated) — both fully working and independently usable.

## Mini-pipeline stages

### `DataGenerator` (`stages/data_generator.py`)

Exactly what the whole pipeline used to be before it was split into stages.
`run(prompt: str) -> Distiset`, three internal steps:
1. **Guide** takes your freeform prompt and, in one LLM call, derives
   `parent_instruction` (system prompt for the generation model),
   `judge_instruction` (system prompt for the quality judge), and
   `sample_instruction` (the actual per-sample instruction sent on every
   generation call — deliberately rewritten to ask for **one** item, since
   your raw prompt is usually phrased as "a dataset of X" and sending that
   verbatim to the model N times causes it to generate a whole batch inside
   each individual sample; see "Known pitfalls" below).
2. **Parent model** (`Model`) generates `--num-samples` raw completions of
   `sample_instruction`, one independent API call per sample (not via the
   API's `n` param — many OpenRouter providers silently ignore `n > 1`).
3. **Judge** scores every sample 0-10 against `judge_instruction`, drops
   anything below `DEFAULT_SCORE_THRESHOLD`, and regenerates exactly that
   many replacements via the parent model — looping up to
   `MAX_REFINE_ITERATIONS` times or until nothing fails.

Internally, steps 1-3 all work in terms of distilabel's own
`instruction`/`generation` field convention (`TextGeneration`'s fixed
input/output names) — that's an implementation detail of how this stage
talks to the parent/judge models via OpenRouter, not part of its output
contract. Before returning, `run()` converts each row into the generic
`messages` schema (`[{"role": "user", "content": ...}, {"role": "assistant",
"content": ...}]`) that `DataFormatter` (and any future stage) actually
consumes — this is what keeps `DataFormatter` decoupled from `DataGenerator`
specifically, rather than coupled to its exact internal column names and
assuming every upstream stage is single-turn Q&A.

### `DataFormatter` (`stages/data_formatter.py`)

`run(distiset: Distiset) -> Distiset`: adds a `text` column by rendering
each row's `messages` through the **child model's own**
chat template — `AutoTokenizer.from_pretrained(child_model_id).apply_chat_template(messages, tokenize=False)`.
Schema-agnostic beyond the `messages` column itself — doesn't care how many
turns there are or which stage produced them, so any stage (or hand-supplied
dataset via `skip_generation`) that emits `messages` in this shape works,
not just `DataGenerator`.

Why this is designed the way it is (all settled through discussion before
building it):
- Nearly every open instruct/chat model on Hugging Face ships a
  `chat_template` (Jinja2, in `tokenizer_config.json`) alongside its weights.
  Rendering through it is *automatic* per model id — no per-model format
  mapping (`user:`/`assistant:` vs `<|user|>`/`<|assistant|>` etc.) is
  hand-coded anywhere in this codebase.
- This is keyed off the specific checkpoint's tokenizer, not a model
  "family" name — two fine-tunes of the same base model can define
  different templates, so `DataFormatter` always fetches the tokenizer for
  the exact `child_model_id` it's given, never infers it.
- `tokenize=False` is essential: output is human-readable rendered text
  (viewable via `data/view.py --formatted`), not token ids. Numeric
  tokenization is `FineTuner`'s concern at train time, not `DataFormatter`'s.
- This matters more for fine-tuning than it would for simple text generation:
  SFT frameworks mask the training loss to only the assistant's response,
  locating that boundary by pattern-matching the model's own role-marker
  tokens. Training data that doesn't use the literal format a model's
  template expects can silently break loss-masking, and separately, the
  model never learns the format it'll actually be prompted with at
  inference (whoever uses the fine-tuned model later will prompt it with
  *its own* native template, not an ad hoc one).
- The tokenizer is **tightly coupled** to `child_model_id`, not a separate
  swappable choice — a model's embeddings were trained against its own
  vocabulary, so using a different model's tokenizer produces token ids
  the target model's weights were never trained to understand.
- Raises `ValueError` if the given `child_model_id` has no `chat_template`
  (e.g. a base/non-instruct model) rather than silently producing garbage.
- Deliberately its **own stage**, not folded into `FineTuner`: keeps the
  rendered dataset inspectable before training even starts, is what makes
  the `skip_formatting` semantics below meaningful, and keeps `FineTuner`
  itself framework-agnostic (it never needs to know about chat templates —
  just tokenize the `text` column and train).

### `FineTuner` (`stages/fine_tuner.py`) — scaffold only, not implemented

`run(formatted_distiset)` currently raises `NotImplementedError`. The
interface and skip-flag wiring are final; the actual training logic
(framework choice, hyperparameters, hardware assumptions) was deliberately
deferred as its own follow-up task rather than decided as a side effect of
this restructuring. Its contract, once implemented: assume
`formatted_distiset` already has a `text` column correctly rendered for
`child_model_id` (via `DataFormatter`, or supplied directly through
`skip_formatting`) and do no reformatting of its own.

### `Stage` (`stages/base.py`)

Shared base for the three mini-pipeline stages:
- `verbose` storage and `_log(message)` (prints only if verbose), mirroring
  how `model/base.py`'s `Model` anchors `Judge`/`Guide`. `DataGenerator`,
  `DataFormatter`, and `FineTuner` all inherit from it instead of each
  re-declaring their own identical `_log` method.
- `_validate_input(data)`: every stage must implement this, called at the
  top of its own `run()` before doing any real work -- this is what makes
  each stage's input contract explicit and self-defending regardless of
  caller (the top-level `DistillationPipeline`, or the stage used
  standalone). Not an `abc.abstractmethod` (no new dependency needed) --
  the base implementation just raises `NotImplementedError`, so a subclass
  that forgets to override it fails loudly instead of silently validating
  nothing. Each stage's own check:
  - `DataGenerator`: `prompt` must be a non-empty, non-whitespace string.
  - `DataFormatter`: the input `Distiset` must have a `messages` column
    (a non-empty list of `{"role": ..., "content": ...}` dicts per row) --
    this specifically guards the `skip_generation` path, where a
    hand-supplied raw dataset is the likeliest place a malformed schema
    shows up (otherwise it'd surface as a cryptic `KeyError` deep inside
    `Dataset.map()`).
  - `FineTuner`: the input `Distiset` must have a `text` column (i.e. it
    was actually rendered via `DataFormatter`, or a `skip_formatting`
    caller supplied an equivalent one) -- checked even though `run()` still
    just raises `NotImplementedError` today, so the contract is already
    validated ahead of whenever real training logic lands.

Each stage's other constructor args (`guide_model`/`parent_model`/
`judge_model`/`num_samples` on `DataGenerator`, `child_model_id`/`tokenizer`
on `DataFormatter`, `child_model_id` on `FineTuner`) are all protected
(`_`-prefixed) -- nothing outside each class ever reads them back, same
reasoning already applied to `Model`'s `_api_key`/`_temperature`/
`_max_tokens`. `DataGenerator` also validates `num_samples > 0` in its
constructor (a config value, not a `run()` input, so it's checked in
`__init__` rather than `_validate_input`).

## Architecture

```
constants.py          -- every constant/default lives here, nowhere else
warnings_filter.py     -- single source of truth for the pydantic warning filter
helper.py               -- CLI-level load/save/validate/timestamp helpers (main.py)
pipeline.py             -- DistillationPipeline: orchestrates the 3 stages + skip flags
stages/
  base.py              -- Stage: shared verbose/_log base for the 3 stages below
  data_generator.py     -- DataGenerator(Stage): guide -> generate -> refine
  data_formatter.py      -- DataFormatter(Stage): chat-template rendering for the child model
  fine_tuner.py           -- FineTuner(Stage): trains the child model (scaffold only)
model/
  base.py             -- Model: base class for any OpenRouter-backed role
  judge.py             -- Judge(Model): scoring + free-text evaluation
  guide.py             -- Guide(Model): derives parent/judge/sample instructions
data/
  generator.py         -- SyntheticDatasetGenerator: raw sample generation
  refactor.py           -- DatasetRefactor: quality-filter + regenerate loop
  view.py               -- DatasetViewer: terminal preview of a saved run/stage output
main.py                -- CLI entry point: argparse (_input_args) + helper + DistillationPipeline
```

### `model.Model` (model/base.py)

Base class for any OpenRouter-backed model role. Handles:
- `model_id` (public — used in error messages, harmless to read) /
  `_api_key` / `_temperature` / `_max_tokens` (protected — read only inside
  this class; `api_key` in particular is secret material, so it isn't
  exposed as a public attribute).
- **Open-weight enforcement**: on construction, live-checks OpenRouter's
  `/models` catalog and rejects any model without a `hugging_face_id`.
  OpenRouter has no "allows distillation" flag; `hugging_face_id` presence
  is the closest real signal (open-weight models like Llama/Qwen/Mistral
  have one, closed ones like GPT/Claude/Gemini don't) — closed-weight
  provider ToS typically forbid using outputs to train other models.
  This check is unconditional, no opt-out. The catalog fetch
  (`_fetch_openrouter_models`) is `@lru_cache`d — fetched once per process,
  not once per Model/Judge/Guide construction.
- `set_instruction()` / `get_instruction()`: per-instance system-prompt
  override, falling back to `_instruction()` (empty by default; `Judge`
  overrides it). Not an abstract method — `Model` is directly usable for
  a role that needs nothing extra (the "parent" role has no subclass).
- `build_llm(structured_output=None)`: constructs the `OpenAILLM` pointed at
  OpenRouter, with `max_new_tokens` and `temperature` wired through.
- `run_pipeline(data, system_prompt, structured_output=None, name=...)`:
  shared "load rows, run one TextGeneration task" pipeline, used by
  `Judge.evaluate`/`score_samples`, `Guide.generate_instructions`, and
  `SyntheticDatasetGenerator.generate` (via `self.model.run_pipeline(...)`).
  Public, not protected — `SyntheticDatasetGenerator` holds a `Model` by
  composition rather than inheritance, so it can't reach a protected member.
- `_assert_structured_output(generation, sample_id=None)`: shared guard used
  by `Guide`/`Judge` — raises a clear `RuntimeError` if a structured-output
  call comes back empty (see pitfalls below), instead of a cryptic Pydantic
  `ValidationError` on `None`.

### `DistillationPipeline` (pipeline.py)

Top-level orchestrator wiring `DataGenerator -> DataFormatter -> FineTuner`
(see "Mini-pipeline stages" above for what each does).
`run(prompt=None, raw_dataset=None, formatted_dataset=None)`:
- Default (no skip flags): `prompt` is required, all three stages run.
- `skip_generation=True`: bring your own raw dataset via `raw_dataset`
  (`instruction`/`generation` columns) — `DataFormatter` and `FineTuner`
  still run.
- `skip_formatting=True`: bring your own already-formatted dataset via
  `formatted_dataset` (must already have a `text` column rendered for
  `child_model_id`) — implies skipping generation too, only `FineTuner` runs.
  You take on the "correctly formatted for this child model" guarantee
  yourself in this case, same as `DataFormatter` would otherwise provide it.

`main.py` exposes this via `--skip-generation`/`--raw-dataset-path` and
`--skip-formatting`/`--formatted-dataset-path` (each pair loaded via
`helper.load_data`). `run()` stores whatever it produces on
`self.raw_dataset`/`self.formatted_dataset` (even a user-supplied one passed
straight through) *before* handing off to `FineTuner` — this is what lets
`main.py` still save both stages' output to disk even though `FineTuner`
always raises for now (see below).

**Stages that won't run aren't constructed.** `__init__` only builds
`DataGenerator` if neither skip flag is set, and only builds `DataFormatter`
if `skip_formatting=False` (`FineTuner` is always built -- every flow ends
there). This matters because `DataFormatter.__init__` does a real network
fetch (the child model's tokenizer via `AutoTokenizer.from_pretrained`) --
previously `DistillationPipeline` built all three stages unconditionally,
so e.g. `skip_formatting=True` with an unreachable/invalid `child_model_id`
would fail at construction even though `DataFormatter` is never used in
that flow. `self.data_generator`/`self.data_formatter` are `None` in exactly
the cases where `run()` would never call them, so no extra guard is needed
at the call sites.

`run()` also validates that each of `prompt`/`raw_dataset`/`formatted_dataset`
is provided **if and only if** the current skip flags will actually use it —
providing one that would otherwise be silently ignored raises a clear
`ValueError` instead (e.g. passing `raw_dataset` alongside
`skip_formatting=True` — a common near-mistake, since it looks like "bring
your own raw dataset" but `skip_formatting` actually skips `DataGenerator`
*and* `DataFormatter`, so the raw dataset would never be used; the error
message points at `skip_generation=True` instead, which is what that case
actually wants).

### `helper.py`

CLI-level helpers shared by `main.py` (kept out of `main()` so it stays a
thin argparse + wiring layer):
- `current_timestamp() -> str`: `datetime.now()` formatted as
  `RUN_TIMESTAMP_FORMAT`.
- `load_data(path) -> Optional[Distiset]`: `Distiset.load_from_disk(path)`
  if `path` is truthy, else `None` -- used for `--raw-dataset-path`/
  `--formatted-dataset-path`, which are optional.
- `save_data(dataset, subdir, label)`: no-ops if `dataset` is `None`
  (a stage that didn't run), else saves to
  `DEFAULT_OUTPUT_DIR/subdir/<current_timestamp()>` and prints a
  confirmation. Calls `current_timestamp()` itself per call rather than
  taking it as a param -- the two `save_data` calls in `main.py` (raw,
  formatted) can in principle land in folders with slightly different
  timestamps if they straddle a one-second boundary, since nothing forces
  them to share one; in practice this is a non-issue since the calls are
  back-to-back with no work in between.
- `validate_skip_flags(args, parser)`: CLI-arg-presence checks (prompt
  required unless a skip flag is set; `--skip-generation` requires
  `--raw-dataset-path`; `--skip-formatting` requires
  `--formatted-dataset-path`), calling `parser.error()` for a clean
  usage+exit rather than raising. Deliberately only checks *presence* of
  CLI args -- `DistillationPipeline._validate_inputs` separately
  re-validates the loaded values regardless of caller, so the same
  guarantees hold for non-CLI/programmatic use of `DistillationPipeline`.

### `warnings_filter.py`

A one-line module: `warnings.filterwarnings("ignore",
category=UnsupportedFieldAttributeWarning)`. `model/base.py`, `main.py`, and
`data/view.py` each do `import warnings_filter  # noqa: F401` as their very
first import (see pitfalls below for why it must run before distilabel is
imported anywhere). Single source of truth instead of the filter call being
copy-pasted three times.

### `Judge` (model/judge.py) — `Model` subclass

- `evaluate(samples: List[str], prompt=None) -> Distiset` — free-text verdict
  per sample.
- `score_samples(samples: Dict[id, text], prompt=None) -> Dict[id, float]` —
  forces a numeric 0-10 score via structured output (`Score` schema); `id`
  is passed through the pipeline as a pass-through column (confirmed
  distilabel does `{**input, **output}`, so extra fields survive un-reordered).
- `failed_sample_count(scores, threshold=DEFAULT_SCORE_THRESHOLD) -> int`.

### `Guide` (model/guide.py) — `Model` subclass

`generate_instructions(user_input: str) -> GuideInstructions` — one
structured-output call producing `parent_instruction`, `judge_instruction`,
`sample_instruction` (see pipeline flow above for why all three exist).

### `SyntheticDatasetGenerator` (data/generator.py)

`generate(prompt) -> Distiset`: builds `num_samples` *identical* input rows
and asks for one generation each (not `n=num_samples` in a single call —
see pitfalls). Output schema: `instruction, generation, distilabel_metadata,
model_name`. No `id` column here — ids only appear downstream if something
adds them. This is `instruction`/`generation`-shaped internally (distilabel's
own field convention) — `DataGenerator.run()` converts to `messages` as its
very last step, after `DatasetRefactor` below, so this schema never reaches
`DataFormatter` directly.

### `DatasetRefactor` (data/refactor.py)

`refine(distiset, threshold=DEFAULT_SCORE_THRESHOLD) -> Distiset`: the
quality-filter + regenerate loop described in "Pipeline flow" step 3, still
working in the same `instruction`/`generation` schema as `SyntheticDatasetGenerator`
above (converted to `messages` afterwards by `DataGenerator.run()`).
Derives the regeneration prompt from `distiset["default"]["train"][0]["instruction"]`
(all rows share it, since generation always repeats one instruction).

### `DatasetViewer` (data/view.py)

`raw_samples(n=5)` prints the first `n` samples' `messages` column
(a `DataGenerator` output, from `datasets/raw/`); `formatted_samples(n=5)`
prints the `text` column instead (a `DataFormatter` output, from
`datasets/formatted/`) — pick via `--formatted`. When `--path` isn't given,
defaults to the most recent timestamped folder under `datasets/raw/` or
`datasets/formatted/` respectively (`DatasetViewer(kind="raw"|"formatted")`
controls which subdir is searched; folder names are
`RUN_TIMESTAMP_FORMAT`-formatted, so lexicographic sort = chronological).
**Must be run as `python -m data.view`, not `python data/view.py`** — it
imports `constants` (a project-root module), and running it as a bare script
only puts `data/`'s own directory on `sys.path`, not the project root.

## Known pitfalls (already solved, don't re-debug these)

- **"Each sample contains N items instead of one"**: your prompt/instruction
  was phrased as "a dataset of X" and got sent verbatim per sample — the
  model dutifully produced a whole batch each time. Fixed by `Guide`
  deriving `sample_instruction` (singular-framed) separately from the
  style-describing `parent_instruction`/`judge_instruction`. Considered
  instead making the *system* prompt say "always give exactly one, ignore
  plural phrasing" — rejected because it creates a conflict the model must
  resolve correctly on every call, whereas rewriting the user-turn text to
  not be plural in the first place removes the conflict entirely.
- **distilabel's `n` parameter for multiple generations per call**: many
  OpenRouter providers silently ignore `n > 1` and return only 1 choice.
  Fixed by generating one row per sample instead of relying on `n`.
- **Structured output silently returns `None`**: usually one of two causes,
  both surfaced now via `Model._assert_structured_output`'s clear error
  message instead of a cryptic Pydantic trace:
  - `max_new_tokens` too low (distilabel/OpenAILLM defaults to 128) —
    truncates the JSON mid-object. Fixed via `DEFAULT_MAX_TOKENS = 1024`.
  - The model/provider doesn't reliably support tool calls (needed for
    `instructor`-based structured output). Verified per-model via repeated
    live testing before picking defaults — see below. **Tried and reverted**:
    OpenRouter's `extra_body: {"provider": {"require_parameters": true}}`
    routing hint, meant to restrict routing to tool-capable providers — this
    made things *worse* (it requires the provider to support *every*
    parameter in the request, not just tools, and no provider satisfied the
    full combination for the models tried). Don't re-attempt this without
    reproducing the failure first.
- **`pydantic.warnings.UnsupportedFieldAttributeWarning` noise**: from
  distilabel's own pydantic model definitions, fires at *import time*. The
  filter (`warnings_filter.py`) must run before the first `distilabel`
  import anywhere in the process — every possible entry point
  (`model/base.py`, `main.py`, `data/view.py`) does `import warnings_filter`
  as its very first import. Placing it after an import in the same file, or
  in a module that isn't imported first, silently doesn't work.
- **`Untracked error: No module named 'bs4'`**: harmless. distilabel tries to
  auto-generate citations/README metadata for every `Distiset` by scraping
  arxiv pages with BeautifulSoup; `bs4` isn't installed and none of our
  steps reference arxiv papers anyway, so this is a no-op print, not a
  failure. Not worth installing `bs4` just to silence it.
- **Python 3.9 → 3.12**: `distilabel`'s `instructor` structured-output
  integration does `from typing import TypeAlias`, which doesn't exist
  before Python 3.10. The project was bumped from 3.9 to 3.12 (`.python-version`,
  `pyproject.toml`), which `uv` fetches automatically — no system install
  needed. `rich` was also relaxed from `>=15.0.0` to `>=13.7.0` since
  `instructor` caps it below 15.

## Model defaults (constants.py) and why

- `DEFAULT_PARENT_MODEL = "meta-llama/llama-3.3-70b-instruct"`
- `DEFAULT_JUDGE_MODEL = "deepseek/deepseek-chat"` — deliberately a
  different model family than the parent (self-preference bias risk if the
  same model both generates and judges). `qwen/qwen-2.5-72b-instruct` was
  tried first and looked fine in isolation, but reliably failed structured
  output specifically when a system message was present (which `Judge`
  always sends) — caught via live testing with the *exact* system+user
  pattern `Judge` uses, not just a bare user-only call. Verified
  `deepseek-chat` reliable across multiple runs through the real
  `Judge.score_samples()` path before committing to it.
- `DEFAULT_GUIDE_MODEL = "qwen/qwen-2.5-7b-instruct"` — small/cheap is fine
  here since Guide only writes instruction text, not dataset content.
  `mistralai/mistral-nemo` was tried first (had tool-capable endpoints on
  paper) but failed unpredictably in live testing; `qwen-2.5-7b-instruct`
  verified reliable across 8/8 live runs across two different prompts before
  becoming the default.
- `DEFAULT_CHILD_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"` — a **Hugging Face
  repo id**, not an OpenRouter model id (different id space entirely, even
  where the two happen to look similar) — used by `DataFormatter`/`FineTuner`
  via `AutoTokenizer`/model loading, not via `Model`/OpenRouter at all.
  Small, open, and ships a known `chat_template`, so it works standalone
  with no extra setup. `Model._assert_open_weight`'s OpenRouter-catalog
  check does not apply to it — fine-tuning a model already requires having
  its open weights, so there's nothing separate to enforce.
  (Note: an earlier version of this project had a similar "child model"
  concept that was built and then deleted; it's since been revived properly
  as the `DataFormatter`/`FineTuner` stages described above.)

**Whenever picking/changing a default model**: don't trust "supports tool
calls" from OpenRouter's model listing alone — verify empirically through
the actual code path that will use it (`Judge.score_samples()` /
`Guide.generate_instructions()`), ideally 3-8 repeated live calls, since
per-provider routing flakiness is real and doesn't always show up in one test.

## Environment

- `.env` holds `OPENROUTER_API_KEY` (gitignored; `.env.example` is the
  committed template).
- Python 3.12 (`.python-version`), managed via `uv` — `uv run main.py ...`
  handles the venv automatically.
- Key dependencies: `distilabel[instructor,openai]`, `pydantic`,
  `python-dotenv`, `rich`, `requests`, `datasets`, `transformers` (for
  `DataFormatter`'s `AutoTokenizer`; PyTorch is not installed and not needed
  for tokenizer/chat-template use — only `FineTuner`'s eventual training
  implementation would need it).
