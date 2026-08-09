"""Modal app + remote LoRA/QLoRA SFT training function.

The actual training logic lives in training/trainer.py::run_sft (shared with
local training) -- this module only holds Modal-specific plumbing (Image,
Volume, per-call GPU/timeout, tar packaging). Heavy ML imports stay deferred
inside train_lora's body, so this module stays importable locally without
those packages installed. They live only in the Modal Image below, per the
"Modal handles all compute" design: pyproject.toml only adds `modal` locally
(torch/peft/trl/accelerate are also available via the optional
`local-training` extra, for --local training on your own machine).
"""

import modal

app = modal.App("fastsft-finetune")

# The remote fn calls `from fastsft.training.trainer import run_sft`, so the
# `fastsft` package must exist in the image. Under the src/ layout Modal no
# longer auto-mounts loose top-level files, so add the installed package's
# source explicitly. (Only fastsft's own code is added; the ML wheels come from
# pip_install above.) Verify against your installed modal version at the e2e step.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets"
    )
    .add_local_python_source("fastsft")
)

ADAPTER_VOLUME_PATH = "/adapters"
adapter_volume = modal.Volume.from_name(
    "fastsft-adapters", create_if_missing=True
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
    resource override API has changed across versions.
    """
    import tarfile

    import torch

    from fastsft.training.trainer import run_sft

    output_dir = f"{ADAPTER_VOLUME_PATH}/{run_id}"
    run_sft(
        child_model_id=child_model_id,
        train_rows=train_rows,
        eval_rows=eval_rows,
        config=config,
        output_dir=output_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tar_path = f"{output_dir}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(output_dir, arcname=".")
    adapter_volume.commit()
    return tar_path
