"""Local torch runtime detection, shared by training (training/local_trainer.py,
stages/fine_tuner.py) and evaluation (eval/inference.py): which accelerator this
machine has, and the dtype to load models in on it.

Torch is imported lazily inside each function, so this module stays importable
without torch (it lives only in the local-training / evaluation optional extras).
The `TYPE_CHECKING` import below is erased at runtime -- it exists only so
`dtype_for_device`'s return type is real, not `Any`.

Also runnable directly to report what this machine offers:

    uv run python -m fastsft.device
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def detect_device() -> str:
    """Detect 'cuda', 'mps', or 'cpu' on this machine."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for_device(device: str) -> "torch.dtype":
    """Return torch dtype for device: bf16 on accelerators (cuda/mps), fp32 on cpu."""
    import torch

    return torch.bfloat16 if device in ("cuda", "mps") else torch.float32


def main() -> None:
    """Report detected device, dtype, and torch version."""
    device = detect_device()
    print(f"Device: {device}")
    print(f"Dtype:  {dtype_for_device(device)}")

    import torch

    print(f"Torch:  {torch.__version__}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
