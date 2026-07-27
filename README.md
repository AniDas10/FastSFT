# LLM-Distillator

A CLI tool that generates synthetic datasets for LLM distillation, using
[distilabel](https://github.com/argilla-io/distilabel) as the pipeline engine
and [OpenRouter](https://openrouter.ai) as the LLM backend. Given a freeform
description of the dataset you want, it derives its own generation/judging
instructions, generates samples, filters out low-quality ones, and
regenerates replacements — all before saving to disk.

## Quick start

```bash
# .env holds OPENROUTER_API_KEY=sk-or-...
uv run main.py "a pirate-themed customer support chatbot" --num-samples 50

# preview the most recent run
uv run python -m data.view
uv run python -m data.view --num-samples 10
```

`main.py` prints `[1/4]`..`[4/4]` progress markers as it runs. Output is saved
to `datasets/<timestamp>/` (a fresh timestamped folder every run — nothing is
overwritten).

## Pipeline flow (main.py)

1. **Guide** takes your freeform prompt and, in one LLM call, derives three
   things: `parent_instruction` (system prompt for the generation model),
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
4. The refined `Distiset` (same sample count as requested, all above
   threshold) is saved to `datasets/<timestamp>/`.

## Architecture

```
constants.py          -- every constant/default lives here, nowhere else
warnings_filter.py     -- single source of truth for the pydantic warning filter
pipeline.py            -- DistillationPipeline: guide -> generate -> refine
model/
  base.py             -- Model: base class for any OpenRouter-backed role
  judge.py             -- Judge(Model): scoring + free-text evaluation
  guide.py             -- Guide(Model): derives parent/judge/sample instructions
data/
  generator.py         -- SyntheticDatasetGenerator: raw sample generation
  refactor.py           -- DatasetRefactor: quality-filter + regenerate loop
  view.py               -- DatasetViewer: terminal preview of a saved run
main.py                -- CLI entry point: argparse + DistillationPipeline + save
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

`run(prompt) -> Distiset`: runs steps `[1/4]`-`[3/4]` (guide instructions,
raw generation, quality refinement) and returns the refined `Distiset`.
Extracted out of `main.py` so the pipeline is reusable outside the CLI
(a script, a notebook, ...); `main.py` itself only parses arguments and
handles `[4/4]` (`save_to_disk`), since the output path is a CLI-specific
concern.

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
adds them.

### `DatasetRefactor` (data/refactor.py)

`refine(distiset, threshold=DEFAULT_SCORE_THRESHOLD) -> Distiset`: the
quality-filter + regenerate loop described in "Pipeline flow" step 3.
Derives the regeneration prompt from `distiset["default"]["train"][0]["instruction"]`
(all rows share it, since generation always repeats one instruction).

### `DatasetViewer` (data/view.py)

`raw_samples(n=5)` prints the first `n` samples of a saved run. Defaults to
the most recent folder under `datasets/` (folder names are
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
- The "child model" concept (an OpenRouter id resolved to a Hugging Face
  tokenizer, used to render samples through a target model's chat template
  for fine-tuning) **was built, then deleted entirely**. It never fit the
  `Model` abstraction (no API calls, no temperature, no instruction — purely
  a tokenizer lookup), and `data/refactor.py`'s purpose pivoted entirely to
  quality-filtering + regeneration instead of child-model reformatting.
  `transformers` is still listed in `pyproject.toml` but nothing in the
  codebase imports it anymore — safe to `uv remove transformers` if desired.
  If chat-template rendering for a specific target model is wanted again,
  it should probably be its own small module, not a `Model` subclass.

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
  `python-dotenv`, `rich`, `requests`.
