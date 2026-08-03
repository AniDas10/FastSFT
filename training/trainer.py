"""Shared LoRA/QLoRA SFT training core -- used by both the Modal remote
function (training/modal_app.py) and local training (training/local_trainer.py).

Heavy ML imports (torch, peft, trl, transformers) are deferred inside
run_sft's body, not at module level, so this file stays importable without
those packages installed unless training is actually invoked.
"""

from typing import Any

from training.constants import QLORA


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

    AutoTokenizer.from_pretrained(child_model_id)  # validates the tokenizer loads

    adapter = config["adapter"]
    loop = config["loop"]

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
            target_modules=adapter["target_modules"],
            lora_dropout=adapter["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

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
        dataset_text_field="text",
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows),
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=loop["early_stopping_patience"])
        ],
    )
    trainer.train()
    trainer.save_model(output_dir)
