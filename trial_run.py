"""FastSFT quickstart: edit values below, and then `uv run trial_run.py`. All set!
(Or don't edit anything and run it, All set either way!)

Pre-req:
Ensure you have uv: check with `uv --version`, or install via `brew install uv`
Ensure you have an OpenRouter API key: sign up free at https://openrouter.ai -> Keys -> Create Key
Set it as an environment variable:
    echo "OPENROUTER_API_KEY=sk-or-..." > .env

Now run:
1) uv sync --extra local-training
2) uv run trial_run.py
"""

# Side-effect import, must precede distilabel/transformers.
import fastsft.warnings_filter  # noqa: F401

import os

from fastsft.data.config import DataGenerationConfig, ParentGenerationConfig
from fastsft.helper import current_timestamp
from fastsft.pipeline import DistillationPipeline
from fastsft.progress import log
from fastsft.training.config import AdapterConfig, TrainingConfig, TrainingLoopConfig

if not os.environ.get("OPENROUTER_API_KEY"):
    raise SystemExit(
        "Missing OPENROUTER_API_KEY. Put it in a .env file: "
        'echo "OPENROUTER_API_KEY=sk-or-..." > .env'
    )

# Freeform description of the Q&A dataset you want generated so child can look at how the parent answers these and learn.
PROMPT = "Answer everything related to any topic like a poet. Keep responses rhyming, and add funny callbacks"

# Alternatives to play around with: 'meta-llama/Llama-3.2-1B-Instruct' /
# 'Qwen/Qwen2.5-1.5B-Instruct' / any Hugging Face repo id.
#
# The below is the default if you don't set one:
CHILD_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# See data_generation_tutorial.md for what each field does and how to tune it.
# Alternatives to play around with in this config:
# DataGenerationConfig(
#     guide_model='meta-llama/llama-3.1-8b-instruct'/'qwen/qwen-2.5-7b-instruct'/any OpenRouter model id,
#     parent_model='qwen/qwen-2.5-72b-instruct'/'meta-llama/llama-3.3-70b-instruct'/any open-weight OpenRouter model id,
#     judge_model='openai/gpt-4o-mini'/'deepseek/deepseek-chat'/any OpenRouter model id,
#     num_samples=(1...any number of samples you want),
#     breadth_exponent=(0...1),
#     score_threshold=(0...10),
#     parent_generation=ParentGenerationConfig(
#         temperature=(0...1),
#         max_tokens=(1...any number of tokens you want),
#     ),
# )
#
# The below are the default configs if you don't set any:
generation = DataGenerationConfig(
    guide_model="qwen/qwen-2.5-7b-instruct",
    parent_model="meta-llama/llama-3.3-70b-instruct",
    judge_model="deepseek/deepseek-chat",
    num_samples=100,
    breadth_exponent=0.85,
    score_threshold=5.0,
    parent_generation=ParentGenerationConfig(
        temperature=0.9,
        max_tokens=1024,
    ),
)

# See training_tutorial.md for what each field does and how to tune it.
# Alternatives to play around with in this config:
# TrainingConfig(
#     gpu_tier='local'/'L4'/'A10G'/'A100-40GB'/'A100-80GB'/'H100',
#     strategy='lora'/'qlora',
#     adapter=AdapterConfig(
#         rank=(1...any rank you want),
#         target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']/['all-linear'],
#         dropout=(0...1),
#     ),
#     loop=TrainingLoopConfig(
#         batch_size=(1...any batch size you want),
#         grad_accumulation=(1...any number of steps you want),
#         learning_rate=(0...1),
#         max_epochs=(1...any number of epochs you want),
#         eval_steps=(1...any number of steps you want),
#         early_stopping_patience=(1...any number of evals you want),
#         validation_split=(0...1),
#         mask_prompt_loss=True/False,
#     ),
#     modal_timeout_seconds=(1...any number of seconds you want),
# )
#
# The below are the default configs if you don't set any:
training = TrainingConfig(
    gpu_tier="local",
    strategy="lora",
    adapter=AdapterConfig(
        rank=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        dropout=0.05,
    ),
    loop=TrainingLoopConfig(
        batch_size=8,
        grad_accumulation=1,
        learning_rate=2e-4,
        max_epochs=10,
        eval_steps=20,
        early_stopping_patience=3,
        validation_split=0.15,
        mask_prompt_loss=True,
    ),
    modal_timeout_seconds=3600,
)

pipeline = DistillationPipeline(
    child_model_id=CHILD_MODEL_ID,
    generation=generation,
    training=training,
    # False + a Modal gpu_tier above trains on Modal instead.
    local_training=True,
)

run_id = current_timestamp()
for stage, output in pipeline.run(PROMPT):
    path = stage.save_output(output, run_id)
    if path:
        log(f"Saved {stage.name} output to '{path}'")

log(
    "\nDone. Explore the run with:\n"
    "  uv run python -m fastsft.data.viewer            # preview the generated dataset\n"
    "  uv run python -m fastsft.training.stats_viewer  # loss curves for this training run\n"
    "  uv run fastsft-eval                              # score the adapter vs. base vs. parent\n"
    "  uv run python -m fastsft.eval.results_viewer     # view evaluation results\n"
)
