# Data Generation Tutorial

**Learn how to generate quality training data with FastSFT's DataGenerator.**

This tutorial walks through the data generation pipeline, from describing your desired dataset to inspecting the quality-filtered results.

## Quick Overview

FastSFT's **DataGenerator** (`src/fastsft/stages/data_generator.py`) transforms a one-sentence description into a quality-filtered dataset of Q&A pairs. It runs four internal steps:

1. **Guide** — derives the style prompt, scoring rubric, and seed topics from your description.
2. **PromptGenerator** — expands seeds into diverse user questions (simple to complex variants).
3. **ResponseGenerator** — generates styled answers from a parent model.
4. **DataRefiner** — scores answers 0-10, discards weak ones, regenerates until threshold met.

The output is a Distiset with a `messages` column (each row: `[{role: user, content: ...}, {role: assistant, content: ...}]`).

## Bare Minimum Setup

```bash
# Install core dependencies
uv sync

# Set your OpenRouter API key
echo "OPENROUTER_API_KEY=sk-or-..." > .env
```

## Generate Your First Dataset

```bash
uv run fastsft "a pirate-themed customer support chatbot" --num-samples 50
```

This command:
- Launches DataGenerator (since `--start-stage` defaults to `data_generator`)
- Generates 50 training examples matching your description
- Saves output to `datasets/raw/<timestamp>/`

## Key Parameters

| Parameter | What it does | Default | Example |
|-----------|------------|---------|---------|
| `--num-samples` | How many examples to generate | 100 | `--num-samples 50` |
| `--parent-model` | OpenRouter model id for generation | `meta-llama/llama-3.3-70b-instruct` | `--parent-model qwen/qwen-2.5-32b-instruct` |
| `--judge-model` | Model for quality scoring | `deepseek/deepseek-chat` | `--judge-model anthropic/claude-3-sonnet` |
| `--guide-model` | Small model deriving instructions | `qwen/qwen-2.5-7b-instruct` | `--guide-model meta-llama/llama-3.1-8b-instruct` |
| `--score-threshold` | Min quality score (0-10) before regeneration | 5 | `--score-threshold 7` |
| `--parent-temperature` | Sampling temperature for parent | 0.9 | `--parent-temperature 0.7` |
| `--breadth-exponent` | Diversity vs. depth tradeoff | 0.67 | `--breadth-exponent 0.8` |

**Breadth vs. Depth**: The `breadth_exponent` controls topic diversity.
- `breadth = ceil(num_samples ** exponent)`
- Default 0.67 is breadth-leaning: for 100 samples → ~21 seed topics, ~5 variants per topic.
- Increase to prioritize topic diversity; decrease for deeper exploration of fewer topics.

## Preview Your Data

After generation completes, inspect what was created:

```bash
# View raw generated Q&A pairs
uv run python -m fastsft.data.viewer

# View options: --num-samples N, --input-path datasets/raw/<timestamp>
uv run python -m fastsft.data.viewer --input-path datasets/raw/20260809_120000 --num-samples 10
```

The viewer displays:
- User questions and assistant answers in a formatted panel
- Sample IDs for tracing back to your data
- Quick feedback loop before formatting + training

### **Advanced: Manually Edit & Resume**

You can inspect the raw Parquet files, edit them, and resume from the next stage:

**Step 1: Locate the dataset**
```bash
ls datasets/raw/20260809_120000/train/
# Output: data-00000-of-00001.parquet
```

**Step 2: Load and inspect in Python**
```python
from datasets import load_dataset

# Load the generated dataset
ds = load_dataset("parquet", data_files="datasets/raw/20260809_120000/train/data-00000-of-00001.parquet")
df = ds["train"].to_pandas()

# View the data
print(df[["instruction", "generation"]].head(10))

# Find low-quality samples
for idx, row in df.iterrows():
    print(f"\n[{idx}] Q: {row['instruction']}")
    print(f"    A: {row['generation'][:100]}...")
```

**Step 3: Edit the Parquet file** (if needed)
```python
# Remove bad samples
bad_indices = [5, 12, 23]  # Example: samples you want to remove
df_clean = df.drop(bad_indices).reset_index(drop=True)

# Or manually fix answers
df_clean.loc[0, "generation"] = "Better answer here..."

# Save back
df_clean.to_parquet("datasets/raw/20260809_120000/train/data-00000-of-00001.parquet")
```

**Step 4: Resume from formatting**
```bash
# Skip data generation, start from formatting
uv run fastsft --start-stage data_formatter \
  --input-path datasets/raw/20260809_120000 \
  --child-model-id Qwen/Qwen2.5-0.5B-Instruct
```

**Why do this:**
- 💰 Skip expensive data generation ($1-2 saved)
- ⚡ Iterate fast on dataset quality
- 🎯 Fix edge cases the judge missed
- 📊 Maintain full control over training data

**Example workflow:**
```bash
# 1. Generate initial dataset
uv run fastsft "your prompt" --num-samples 50

# 2. Inspect it
uv run python -m fastsft.data.viewer --formatted

# 3. If quality is good → train immediately
uv run fastsft --start-stage fine_tuner \
  --input-path datasets/formatted/<timestamp>

# 4. If quality is mediocre → edit and resume
# (Python script to fix Parquet file)
uv run fastsft --start-stage data_formatter \
  --input-path datasets/raw/<timestamp>

# 5. Then train
uv run fastsft --start-stage fine_tuner \
  --input-path datasets/formatted/<timestamp>
```

## Iterate on Dataset Quality

Once you've inspected the data:

**1. Improve style/tone:** Adjust your initial prompt description.
```bash
uv run fastsft "a friendly, enthusiastic pirate support agent (uses 'ahoy', 'landlubber')" --num-samples 50
```

**2. Raise the quality bar:** Increase `--score-threshold`.
```bash
uv run fastsft "..." --num-samples 50 --score-threshold 7
```

**3. Reuse a dataset, adjust only generation:** Skip regeneration if the style is good.
```bash
# Re-use raw data, skip data_generator
uv run fastsft --start-stage data_formatter --input-path datasets/raw/20260809_120000
```

## Using Your Own Dataset

FastSFT accepts **Distiset format** (the native format used internally). You can:

1. **Let DataGenerator create the data** (default) — automatically produces Distiset
2. **Convert your own data to Distiset** — use `--start-stage data_formatter`

### Distiset Format

A Distiset is a wrapper around Hugging Face datasets. It's saved on disk as:
```
datasets/raw/<timestamp>/
├── default/
│   ├── train/
│   │   ├── data-00000-of-00001.arrow  (Apache Arrow format)
│   │   ├── dataset_info.json
│   │   └── state.json
│   └── dataset_dict.json
└── distiset_configs/
    └── config.json
```

### Converting Your Data to Distiset

**Required format:** Each row must have a `messages` column containing:
```python
[
    {"role": "user", "content": "the question"},
    {"role": "assistant", "content": "the answer"}
]
```

#### **Example 1: Convert CSV to Distiset**

```python
import pandas as pd
from datasets import Dataset, DatasetDict
from distilabel.distiset import Distiset

# Load your CSV
df = pd.read_csv("my_data.csv")  # Has columns: "question", "answer"

# Transform to messages format
def to_messages(row):
    return {
        "messages": [
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]}
        ]
    }

df_transformed = df.apply(to_messages, axis=1, result_type="expand")

# Convert to Dataset
dataset = Dataset.from_dict(df_transformed)

# Wrap in DatasetDict
dataset_dict = DatasetDict({"train": dataset})

# Wrap in Distiset
distiset = Distiset({"default": dataset_dict})

# Save
distiset.save_to_disk("datasets/raw/my_custom_dataset")

print(f"Saved {len(dataset)} samples to datasets/raw/my_custom_dataset")
```

#### **Example 2: Convert Parquet to Distiset**

```python
from datasets import Dataset, DatasetDict
from distilabel.distiset import Distiset

# Load Parquet
dataset = Dataset.from_parquet("my_data.parquet")

# Ensure it has "messages" column (convert from other formats if needed)
if "messages" not in dataset.column_names:
    # If columns are "question" and "answer"
    def make_messages(row):
        return {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}
            ]
        }
    dataset = dataset.map(make_messages, remove_columns=["question", "answer"])

# Wrap and save
distiset = Distiset({"default": DatasetDict({"train": dataset})})
distiset.save_to_disk("datasets/raw/my_parquet_data")
```

#### **Example 3: Convert JSON to Distiset**

```python
import json
from datasets import Dataset, DatasetDict
from distilabel.distiset import Distiset

# Load JSON (assume list of dicts with "question" and "answer" keys)
with open("my_data.json") as f:
    data = json.load(f)

# Convert to messages format
messages_data = []
for item in data:
    messages_data.append({
        "messages": [
            {"role": "user", "content": item["question"]},
            {"role": "assistant", "content": item["answer"]}
        ]
    })

# Create Dataset and Distiset
dataset = Dataset.from_dict(messages_data)
distiset = Distiset({"default": DatasetDict({"train": dataset})})
distiset.save_to_disk("datasets/raw/my_json_data")
```

#### **Example 4: Combine Auto-Generated + Manual Data**

```python
from datasets import Dataset, DatasetDict, concatenate_datasets
from distilabel.distiset import Distiset

# Load auto-generated data
auto_distiset = Distiset.load_from_disk("datasets/raw/20260809_120000")
auto_dataset = auto_distiset["default"]["train"]

# Load your custom data
custom_distiset = Distiset.load_from_disk("datasets/raw/my_custom_data")
custom_dataset = custom_distiset["default"]["train"]

# Combine
combined = concatenate_datasets([auto_dataset, custom_dataset])

# Save
final_distiset = Distiset({"default": DatasetDict({"train": combined})})
final_distiset.save_to_disk("datasets/raw/combined_data")

print(f"Combined {len(auto_dataset)} + {len(custom_dataset)} = {len(combined)} samples")
```

### Use Your Custom Dataset

Once you've converted your data to Distiset format:

```bash
# Resume from data formatting
uv run fastsft --start-stage data_formatter \
  --input-path datasets/raw/my_custom_dataset \
  --child-model-id Qwen/Qwen2.5-0.5B-Instruct

# Then train
uv run fastsft --start-stage fine_tuner \
  --input-path datasets/formatted/<timestamp> \
  --child-model-id Qwen/Qwen2.5-0.5B-Instruct --local
```

## Understanding the Output

DataGenerator saves to `datasets/raw/<timestamp>/`:

```
datasets/raw/20260809_120000/
├── train/
│   └── data-00000-of-00001.parquet    # Q&A pairs in Parquet format
├── test/
│   └── data-00000-of-00001.parquet    # Validation split (used internally)
└── 20260809_120000.training_metadata.json  # Sidecar: teacher reference
```

The `training_metadata.json` file stores:
- `parent_model`: The model used for generation (e.g., `meta-llama/llama-3.3-70b-instruct`)
- `parent_instruction`: The derived style prompt (how the parent should answer)
- `temperature` & `max_tokens`: Generation hyperparameters

This sidecar is critical for later evaluation — it lets the eval module reconstruct the exact parent reference for comparison.

## File References

| File | Purpose |
|------|---------|
| `src/fastsft/stages/data_generator.py` | DataGenerator class orchestrating all 4 steps |
| `src/fastsft/model/guide.py` | Guide: derives instructions from your prompt |
| `src/fastsft/data/prompt_generator.py` | PromptGenerator: expands seeds → instructions |
| `src/fastsft/data/response_generator.py` | ResponseGenerator: parent answers instructions |
| `src/fastsft/data/refiner.py` | DataRefiner: judge-scored quality filtering |
| `src/fastsft/data/viewer.py` | DataViewer CLI for inspecting results |
| `src/fastsft/data/config.py` | DataGenerationConfig dataclass |

## Crafting Effective Prompts

The quality of your generated dataset depends entirely on how well you describe what you want. **Be verbose and explicit.**

### ❌ Vague Prompts (Don't Do This)

These prompts are too short and ambiguous:
- "a pirate"
- "be smart"
- "sound professional"
- "respond like an engineer"

The Guide model has to guess what you mean, and ambiguity leads to inconsistent data.

### ✅ Effective Prompts (Do This)

Describe:
1. **The persona/role** — who should the model act as?
2. **Key characteristics** — what defines their voice/tone/style?
3. **Examples of behavior** — what should they do/avoid?
4. **Scope** — is this specific domain or "anything"?

**Example 1: Pirate Support Bot**
```
"Respond as a friendly pirate-themed customer support agent. Always use 
nautical slang (ahoy, matey, landlubber, shiver me timbers), adopt a casual 
and jovial tone, and frame problems as 'challenges to overcome like the high seas'. 
End responses with a relevant pirate emoji. You help with ANY customer issue, 
from billing to technical problems — just with pirate flair."
```

**Example 2: Software Engineer**
```
"Respond like a pragmatic backend engineer to ANY question (not just software). 
Use technical thinking: break problems into steps, explain trade-offs, recommend 
practical solutions. Assume the user wants efficiency and correctness. Tone: 
no-nonsense, direct, helpful. Use technical terminology but explain it clearly."
```

**Example 3: Academic Mathematician**
```
"Respond as a distinguished mathematics scholar. Be precise and rigorous, 
define terms before using them, reason in clear logical steps. Occasionally 
note historical context or philosophical implications. Use formal, elegant prose. 
Answer ANY question (not just math) with this scholarly mindset — even 'how to 
cook pasta' or 'best way to organize a bookshelf'."
```

**Example 4: Startup Founder Mentor**
```
"Respond as an experienced startup founder mentoring someone. Be encouraging 
but realistic, share practical insights from building companies, focus on 
execution over theory. Your tone: warm, direct, slightly irreverent. You answer 
about startups, business, leadership, AND general life/career advice — always 
bringing founder perspective."
```

### 📋 Prompt Checklist

Before running data generation, ask yourself:

- [ ] **Persona clear?** "Respond as a [specific role/archetype]"
- [ ] **Key traits listed?** (at least 3-5: tone, vocabulary, approach, attitude)
- [ ] **Scope explicit?** ("to ANY topic" vs. "to medical questions")
- [ ] **Behavioral examples?** ("always do X", "never do Y")
- [ ] **What they should sound like?** (formal/casual, technical/simple, etc.)
- [ ] **What they should NOT sound like?** (avoid sarcasm? too verbose?)

### 🎯 Key Insight: Domain Diversity

**If your prompt says "anything" or "everything"**, the Guide will now automatically generate seed topics from diverse domains:
- Software engineering → questions about code, cooking, home repair, finance, philosophy
- Pirate → questions about sailing, trading, survival, history, humor, life lessons
- Poet → questions about nature, love, technology, society, personal growth

**If your prompt is specific to one domain** (e.g., "answer medical questions"), we stay in that domain.

So be explicit:
- ✅ "Respond like a doctor to ANY question" → diverse domains
- ✅ "Respond like a doctor to medical questions only" → medical domain
- ❌ "be a doctor" → ambiguous, Guide will guess

## Common Issues

| Issue | Fix |
|-------|-----|
| `No OpenRouter API key found` | Run `echo "OPENROUTER_API_KEY=sk-or-..." > .env` |
| Generation times out or is slow | Use a faster parent model, or reduce `--num-samples` |
| Quality is poor / off-topic | Your prompt was too vague. Use the checklist above and be more explicit. Good example: `"a pirate support bot: always calls users 'captain', uses nautical slang, casual tone, responds to ANY customer issue"` |
| Too many samples rejected (low scores) | The judge instruction was unclear. Try a more detailed prompt with concrete traits. Lower `--score-threshold` only as a last resort. |
| Data is only about one domain | Your prompt didn't clearly say "anything" or "all topics". Try: `"Respond like an engineer to ANY question, even non-software ones"` |
| `... has no chat_template` on later stages | This error is from DataFormatter, meaning you changed to an incompatible `--child-model-id` after generation — rerun generation or use the same model. |

## Next Steps

- **Inspect the data:** Use `fastsft.data.viewer` to visually confirm quality and style.
- **Move to formatting:** Once satisfied, the next stage (DataFormatter) renders this into your child model's exact chat format — see `training_tutorial.md`.
- **Explore defaults:** Run `uv run python -m fastsft.training.heuristic <model_id>` to preview training config options before committing.
