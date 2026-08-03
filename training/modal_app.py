"""Modal app + remote LoRA/QLoRA SFT training function.

Heavy ML imports (torch, peft, trl, bitsandbytes) are deferred to inside
train_lora's body -- this module must stay importable locally without those
packages installed. They live only in the Modal Image below, per the
"Modal handles all compute" design: pyproject.toml only adds `modal` locally.
"""

import modal

from training.constants import QLORA

app = modal.App("llm-distillator-finetune")

image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets"
)

ADAPTER_VOLUME_PATH = "/adapters"
adapter_volume = modal.Volume.from_name(
    "llm-distillator-adapters", create_if_missing=True
)


@app.function(
    image=image,
    volumes={ADAPTER_VOLUME_PATH: adapter_volume},
    timeout=3600,
)
def train_lora(
    child_model_id: str,
    train_rows: list,
    eval_rows: list,
    config: dict,
    run_id: str,
) -> str:
    """Runs LoRA/QLoRA SFT on the child model; returns the path (inside the
    Modal Volume) to a .tar.gz of the trained adapter.

    GPU tier and timeout are set per-call via `.with_options(gpu=..., timeout=...)`
    at the caller (see stages/fine_tuner.py) -- verify this call pattern
    against your installed `modal` SDK version, since Modal's per-call
    resource override API has changed across versions. Not exercised by this
    session's verification (no live Modal dispatch was run, per instruction).
    """
    import tarfile

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
        torch_dtype=torch.bfloat16,
        device_map="auto",
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

    output_dir = f"{ADAPTER_VOLUME_PATH}/{run_id}"
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

    tar_path = f"{output_dir}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(output_dir, arcname=".")
    adapter_volume.commit()
    return tar_path
