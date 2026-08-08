"""Local torch runtime detection, shared by training (training/local_trainer.py,
stages/fine_tuner.py) and evaluation (eval/inference.py): which accelerator this
machine has, and the dtype to load models in on it.

Torch is imported lazily inside each function, so this module stays importable
without torch (it lives only in the local-training / evaluation optional extras).

Also runnable directly to report what this machine offers:

    uv run python -m device
"""


def detect_device() -> str:
    """Returns 'cuda', 'mps', or 'cpu' -- whichever this machine actually has."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def dtype_for_device(device: str):
    """The torch dtype to load models in on `device`: bf16 on a real accelerator
    (cuda/mps), fp32 on cpu. (fp32 on mps is unstable in some torch builds, so
    accelerators use bf16 consistently.)"""
    import torch

    return torch.bfloat16 if device in ("cuda", "mps") else torch.float32


def main():
    """Reports the detected device and the dtype models will load in."""
    device = detect_device()
    print(f"Device: {device}")
    print(f"Dtype:  {dtype_for_device(device)}")

    import torch

    print(f"Torch:  {torch.__version__}")
    if device == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(0)}")


if __name__ == "__main__":
    main()
