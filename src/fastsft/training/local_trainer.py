"""Local (non-Modal) LoRA SFT training -- for exercising FineTuner before
setting up Modal, or for training on your own hardware.

Requires the `local-training` optional dependency group:
`uv sync --extra local-training` (torch, peft, trl, accelerate).

QLoRA (bitsandbytes) is CUDA-first -- on non-CUDA hardware only `lora` is
supported here, validated upfront (see assert_strategy_supported) rather
than left to fail deep inside bitsandbytes.
"""

import tempfile
from typing import Any

from fastsft.device import detect_device, dtype_for_device
from fastsft.training.constants import QLORA


def assert_strategy_supported(strategy: str, device: str) -> None:
    """QLoRA needs bitsandbytes, which needs CUDA -- fail clearly upfront,
    not deep inside a quantization error."""
    if strategy == QLORA and device != "cuda":
        raise ValueError(
            f"--strategy qlora requires CUDA for local training (bitsandbytes); "
            f"this machine only has '{device}'. Use --strategy lora, or drop "
            "--local to train on Modal instead."
        )


def train_locally(
    child_model_id: str,
    train_rows: list[dict],
    eval_rows: list[dict],
    config: dict[str, Any],
) -> str:
    """Runs LoRA/QLoRA SFT on this machine; returns the local directory the
    trained adapter was saved to."""
    from fastsft.training.trainer import run_sft

    device = detect_device()
    assert_strategy_supported(config["strategy"], device)

    torch_dtype = dtype_for_device(device)
    device_map = {"": device}  # single-device placement, no multi-GPU sharding

    output_dir = tempfile.mkdtemp(prefix="finetuner_local_")
    run_sft(
        child_model_id=child_model_id,
        train_rows=train_rows,
        eval_rows=eval_rows,
        config=config,
        output_dir=output_dir,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    return output_dir
