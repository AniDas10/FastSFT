"""Shared LoRA/QLoRA SFT training core -- used by both the Modal remote
function (training/modal_app.py) and local training (training/local_trainer.py).

Heavy ML imports (torch, peft, trl, transformers) are deferred inside
run_sft's body, not at module level, so this file stays importable without
those packages installed unless training is actually invoked.

Loss masking is model-agnostic: nothing here assumes a particular chat format.
`_resolve_loss_masking` probes the installed TRL and the child model's own
tokenizer to decide HOW to keep prompt tokens out of the loss (see it).
"""

import json
import os
from collections.abc import Callable
from typing import Any

from fastsft.training.constants import ALL_LINEAR_TARGET, QLORA


def _write_training_stats(state: Any, output_dir: str, loss_masking: str) -> None:
    """Persists a compact telemetry summary (loss history, best checkpoint,
    epochs run, and how the loss was masked) next to the adapter, so it travels
    with the adapter on both the local and Modal paths and can be surfaced later
    (see training/stats.py)."""
    stats = {
        "log_history": state.log_history,
        "best_metric": state.best_metric,
        "best_model_checkpoint": state.best_model_checkpoint,
        "global_step": state.global_step,
        "epoch": state.epoch,
        "num_train_epochs": getattr(state, "num_train_epochs", None),
        # "completion" | "assistant" | "full" -- lets the stats viewer explain
        # whether the loss reflects the answer only or the whole sequence.
        "loss_masking": loss_masking,
    }
    with open(os.path.join(output_dir, "training_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def _resolve_lora_targets(target_modules: list[str]) -> Any:
    """PEFT treats the string "all-linear" specially (auto-target every linear
    layer, regardless of a model's naming). A list wouldn't trigger that, so
    unwrap a lone ["all-linear"] into the bare string; otherwise pass through."""
    if target_modules == [ALL_LINEAR_TARGET]:
        return ALL_LINEAR_TARGET
    return target_modules


def _supports_assistant_mask(tokenizer: Any, messages: Any) -> bool:
    """True if the model's own chat template can emit a per-token assistant mask
    (it has `{% generation %}` markers). Model-agnostic: we don't read the
    template text, we ask the tokenizer to produce the mask and check it marks
    some -- but not all -- tokens as the assistant's."""
    if not messages:
        return False
    try:
        out = tokenizer.apply_chat_template(
            messages, return_assistant_tokens_mask=True, return_dict=True, tokenize=True
        )
    except Exception:
        return False
    mask = out.get("assistant_masks")
    return bool(mask) and 0 < sum(mask) < len(mask)


def _resolve_loss_masking(
    sft_fields: set[str],
    tokenizer: Any,
    sample_messages: Any,
    log: Callable[[str], None],
) -> str:
    """Chooses a model-agnostic way to exclude prompt tokens from the loss, in
    three tiers, best first. Every tier is feature-detected, so an older TRL or
    an unusual template simply falls through to the next one:

    1. "completion" -- split each row into (prompt, completion) and let TRL mask
       the prompt (`completion_only_loss`). Works for ANY chat model via its own
       template; no per-family strings, no template markers. Single-turn, which
       is exactly what this pipeline produces.
    2. "assistant" -- the template exposes a native per-turn assistant mask
       (`assistant_only_loss`); multi-turn-safe. Used when tier 1 is unavailable
       but the template has `{% generation %}` markers.
    3. "full" -- neither is available; fall back to loss over the whole sequence
       (logged, never silent), so training still runs on any model.
    """
    has_prompt_completion = sample_messages and len(sample_messages) >= 2
    if "completion_only_loss" in sft_fields and has_prompt_completion:
        log("answer-only loss via completion_only_loss (prompt/completion split).")
        return "completion"

    if "assistant_only_loss" in sft_fields and _supports_assistant_mask(tokenizer, sample_messages):
        log("answer-only loss via the template's native assistant mask (assistant_only_loss).")
        return "assistant"

    log(
        "could not mask prompt tokens for this model -- training on loss over the whole "
        "sequence (system + user + answer)."
    )
    return "full"


def _build_dataset(dataset_cls: Any, rows: list[dict], mode: str) -> Any:
    """Projects rows into the shape each masking mode expects: (prompt,
    completion) message lists for "completion", raw `messages` for "assistant",
    or the pre-rendered `text` for "full"."""
    if mode == "completion":
        data = [
            {"prompt": row["messages"][:-1], "completion": row["messages"][-1:]}
            for row in rows
        ]
    elif mode == "assistant":
        data = [{"messages": row["messages"]} for row in rows]
    else:
        data = [{"text": row["text"]} for row in rows]
    return dataset_cls.from_list(data)


def run_sft(
    child_model_id: str,
    train_rows: list[dict],
    eval_rows: list[dict],
    config: dict[str, Any],
    output_dir: str,
    device_map: Any,
    torch_dtype: Any,
) -> None:
    """Runs LoRA/QLoRA SFT on `child_model_id`, saving the trained adapter to
    `output_dir`. `device_map`/`torch_dtype` are the caller's responsibility --
    Modal always trains on a requested CUDA GPU (`device_map="auto"`,
    `torch.bfloat16`); local training auto-detects cuda/mps/cpu and picks
    accordingly (see training/local_trainer.py).
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        EarlyStoppingCallback,
    )
    from trl import SFTConfig, SFTTrainer

    def log(message: str) -> None:
        print(f"[trainer] {message}")

    adapter = config["adapter"]
    loop = config["loop"]

    tokenizer = AutoTokenizer.from_pretrained(child_model_id)
    # Batching needs a pad token; many base models ship without one. Falling
    # back to eos is the standard, model-agnostic fix.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if config["strategy"] == QLORA:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = AutoModelForCausalLM.from_pretrained(
        child_model_id,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=adapter["rank"],
            lora_alpha=adapter["rank"] * 2,
            target_modules=_resolve_lora_targets(adapter["target_modules"]),
            lora_dropout=adapter["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    # Decide how (or whether) to mask prompt tokens out of the loss.
    mode = "full"
    if not loop.get("mask_prompt_loss", True):
        log("prompt masking disabled by config -- training on loss over the whole sequence.")
    elif train_rows:
        mode = _resolve_loss_masking(
            set(SFTConfig.__dataclass_fields__),
            tokenizer,
            train_rows[0].get("messages"),
            log,
        )

    # Only the "full" mode trains on the pre-rendered `text` field; the masking
    # modes hand TRL structured data (prompt/completion or messages) and let it
    # apply the template. The masking flags are only set when their mode is
    # active, so an older TRL without the field is never handed an unknown kwarg.
    masking_flags: dict[str, bool] = {}
    if mode == "completion":
        masking_flags["completion_only_loss"] = True
    elif mode == "assistant":
        masking_flags["assistant_only_loss"] = True

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=loop["batch_size"],
        gradient_accumulation_steps=loop["grad_accumulation"],
        learning_rate=loop["learning_rate"],
        num_train_epochs=loop["max_epochs"],
        eval_strategy="steps",
        eval_steps=loop["eval_steps"],
        save_strategy="steps",
        save_steps=loop["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataset_text_field=("text" if mode == "full" else None),
        report_to=[],
        **masking_flags,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=_build_dataset(Dataset, train_rows, mode),
        eval_dataset=_build_dataset(Dataset, eval_rows, mode),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=loop["early_stopping_patience"])
        ],
    )
    trainer.train()
    trainer.save_model(output_dir)
    _write_training_stats(trainer.state, output_dir, mode)
