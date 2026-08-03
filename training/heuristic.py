"""Cost/feasibility heuristic for ranking candidate LoRA/QLoRA training configs.

Ranks candidates by estimated Modal $/hour among those that plausibly fit in
a tier's VRAM. It does NOT and cannot predict training quality -- that's only
knowable empirically, via validation-loss monitoring during the actual run
(see FineTuner: it holds out a validation split and lets the Modal job's
early stopping decide epoch count and reveal real quality).
"""

import argparse

from huggingface_hub import model_info
from rich.console import Console
from rich.table import Table
from transformers import AutoConfig, AutoTokenizer

from helper import load_data
from training.config import AdapterConfig, TrainingConfig, TrainingLoopConfig
from training.constants import (
    ACTIVATION_BYTES_PER_TOKEN_PER_LAYER_UNIT,
    ADAPTER_BASE_OVERHEAD_GB,
    ADAPTER_PER_RANK_GB,
    BATCH_SIZE_CANDIDATES,
    BYTES_PER_PARAM,
    DEFAULT_FALLBACK_SEQ_LEN,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_RANK,
    LORA,
    MODAL_GPU_TIERS,
    QLORA,
    SAFETY_MARGIN,
    TARGET_EFFECTIVE_BATCH,
    TOP_N_CONFIGS,
)

console = Console()

# (TrainingConfig field path, one-line explanation) -- printed by
# `python -m training.heuristic` as a quick reference for what each knob
# does and how it affects cost/quality/feasibility. Ordered to match
# TrainingConfig's own field order.
KNOB_DESCRIPTIONS: list[tuple[str, str]] = [
    (
        "gpu_tier",
        ("Which Modal GPU tier to train on -- sets available VRAM and $/hour. "
        "The heuristic picks the cheapest tier that fits; override for more "
        "headroom or a faster GPU."),
    ),
    (
        "strategy",
        ("lora (full-precision frozen base) or qlora (4-bit quantized base -- "
        "less memory, slight quality risk). The heuristic prefers lora when "
        "it fits, falling back to qlora only if needed."),
    ),
    (
        "adapter.rank",
        ("How expressive the LoRA adapter is. Higher = can learn more complex "
        "changes, at the cost of a bit more memory/compute and higher "
        "overfitting risk on small datasets."),
    ),
    (
        "adapter.target_modules",
        ("Which weight matrices get adapted (e.g. attention projections). "
        "Adding more (e.g. MLP layers) increases adaptation capacity and "
        "adapter size."),
    ),
    (
        "adapter.dropout",
        ("Regularization on the adapter. Higher reduces overfitting risk "
        "(useful for small datasets or many epochs), at some cost to how "
        "fast the adapter learns."),
    ),
    (
        "loop.batch_size",
        ("Samples processed per training step. Larger = more stable "
        "gradients, more memory. The heuristic picks the largest that fits "
        "your GPU tier."),
    ),
    (
        "loop.grad_accumulation",
        ("Simulates a larger effective batch size without the memory cost, "
        "by accumulating gradients over several steps before updating "
        "weights."),
    ),
    (
        "loop.learning_rate",
        ("Step size for weight updates. Too high destabilizes training; too "
        "low slows convergence."),
    ),
    (
        "loop.max_epochs",
        ("Upper bound on training passes -- early stopping decides the "
        "ACTUAL count. This is just a cost ceiling: lower it for a cheap "
        "experiment, raise it for a tiny dataset that needs more passes."),
    ),
    (
        "loop.eval_steps",
        ("How often (in steps) the model is checked against the held-out "
        "validation set. Too infrequent risks overshooting the best "
        "checkpoint; too frequent adds overhead."),
    ),
    (
        "loop.early_stopping_patience",
        ("Non-improving evaluations tolerated before stopping. Lower = stops "
        "sooner (cheaper, riskier); higher = more patient, costs more "
        "compute."),
    ),
    (
        "loop.validation_split",
        ("Fraction of the dataset held out to monitor for early stopping. "
        "Larger = more reliable signal, less data left to actually train on."),
    ),
    (
        "modal_timeout_seconds",
        ("Hard kill-time for the Modal job. Raise this for bigger datasets "
        "or slower/cheaper GPU tiers, or the job gets killed mid-training."),
    ),
]


def recommend_configs(
    child_model_id: str, sample_texts: list[str], top_n: int
) -> list[TrainingConfig]:
    """Returns up to `top_n` candidate TrainingConfigs, cheapest first, among
    (gpu_tier, strategy) combinations that plausibly fit `child_model_id`.
    """
    param_count, hidden_size, num_layers = _model_metadata(child_model_id)
    max_seq_len = _max_sequence_length(child_model_id, sample_texts)

    candidates: list[TrainingConfig] = []
    for gpu_tier, (vram_gb, usd_per_hour) in MODAL_GPU_TIERS.items():
        for strategy in (LORA, QLORA):
            batch_size = _best_batch_size(
                vram_gb, param_count, strategy, hidden_size, num_layers, max_seq_len
            )
            if batch_size is not None:
                grad_accumulation = max(1, -(-TARGET_EFFECTIVE_BATCH // batch_size))
                est_memory_gb = _estimate_memory_gb(
                    param_count, strategy, hidden_size, num_layers, max_seq_len, batch_size
                )
                candidates.append(
                    TrainingConfig(
                        gpu_tier=gpu_tier,
                        strategy=strategy,
                        adapter=AdapterConfig(rank=DEFAULT_LORA_RANK),
                        loop=TrainingLoopConfig(
                            batch_size=batch_size,
                            grad_accumulation=grad_accumulation,
                            learning_rate=DEFAULT_LEARNING_RATE,
                        ),
                        est_memory_gb=round(est_memory_gb, 2),
                        est_usd_per_hour=usd_per_hour,
                    )
                )
                break  # prefer lora over qlora once one strategy fits this tier

    if not candidates:
        raise RuntimeError(
            f"No Modal GPU tier in the catalog can fit '{child_model_id}' even "
            "at batch_size=1. Consider a smaller child model."
        )

    candidates.sort(key=lambda c: c.est_usd_per_hour)
    return candidates[:top_n]


def _model_metadata(child_model_id: str) -> tuple[int, int, int]:
    """Returns (param_count, hidden_size, num_layers) without downloading weights."""
    info = model_info(child_model_id)
    if info.safetensors is None:
        raise ValueError(
            f"'{child_model_id}' has no safetensors metadata on the Hub -- "
            "can't determine its parameter count without downloading weights."
        )
    param_count = info.safetensors.total

    config = AutoConfig.from_pretrained(child_model_id)
    hidden_size = config.hidden_size
    num_layers = getattr(config, "num_hidden_layers", None) or config.num_layers
    return param_count, hidden_size, num_layers


def _max_sequence_length(child_model_id: str, sample_texts: list[str]) -> int:
    """Longest tokenized sample, or DEFAULT_FALLBACK_SEQ_LEN if none given."""
    if not sample_texts:
        return DEFAULT_FALLBACK_SEQ_LEN
    tokenizer = AutoTokenizer.from_pretrained(child_model_id)
    return max(len(tokenizer.encode(text)) for text in sample_texts)


def _best_batch_size(
    vram_gb: float,
    param_count: int,
    strategy: str,
    hidden_size: int,
    num_layers: int,
    max_seq_len: int,
) -> int | None:
    """Largest candidate batch size that fits `vram_gb`, or None if even
    batch_size=1 doesn't fit."""
    for batch_size in BATCH_SIZE_CANDIDATES:
        estimate = _estimate_memory_gb(
            param_count, strategy, hidden_size, num_layers, max_seq_len, batch_size
        )
        if estimate <= vram_gb:
            return batch_size
    return None


def _estimate_memory_gb(
    param_count: int,
    strategy: str,
    hidden_size: int,
    num_layers: int,
    max_seq_len: int,
    batch_size: int,
    lora_rank: int = DEFAULT_LORA_RANK,
) -> float:
    """Deliberately coarse memory estimate -- good enough to RANK candidates,
    not a precise predictor (real usage also depends on the training
    framework, attention implementation, and other runtime factors not
    modeled here).
    """
    base_gb = param_count * BYTES_PER_PARAM[strategy] / 1e9
    activation_gb = (
        hidden_size * num_layers * max_seq_len * batch_size
        * ACTIVATION_BYTES_PER_TOKEN_PER_LAYER_UNIT / 1e9
    )
    adapter_gb = ADAPTER_BASE_OVERHEAD_GB + lora_rank * ADAPTER_PER_RANK_GB
    return (base_gb + activation_gb + adapter_gb) * SAFETY_MARGIN


def _print_knob_glossary() -> None:
    """Prints a quick, scannable explanation of every TrainingConfig knob."""
    table = Table(title="TrainingConfig knobs", show_lines=True)
    table.add_column("Knob", style="bold cyan", no_wrap=True)
    table.add_column("What it affects")
    for name, description in KNOB_DESCRIPTIONS:
        table.add_row(name, description)
    console.print(table)


def _print_shortlist(shortlist: list[TrainingConfig]) -> None:
    """Prints the ranked shortlist as a table."""
    table = Table(title="Ranked training configs (cheapest first)")
    table.add_column("#", justify="right")
    table.add_column("GPU Tier", style="bold")
    table.add_column("Strategy")
    table.add_column("Rank", justify="right")
    table.add_column("Batch", justify="right")
    table.add_column("$/hr", justify="right", style="green")
    table.add_column("Est. GB", justify="right")
    for i, cfg in enumerate(shortlist):
        table.add_row(
            str(i),
            cfg.gpu_tier,
            cfg.strategy,
            str(cfg.adapter.rank),
            str(cfg.loop.batch_size),
            f"${cfg.est_usd_per_hour}",
            f"{cfg.est_memory_gb}",
        )
    console.print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Preview cost-ranked training configs for a child model, "
        "without running DataGenerator/DataFormatter/FineTuner."
    )
    parser.add_argument("child_model_id", help="Hugging Face repo id of the child model.")
    parser.add_argument(
        "--input-path",
        default=None,
        help="Path to a saved formatted Distiset (with a 'text' column) to "
        "measure real sequence lengths from. Omit for a rough preview using "
        "a fallback sequence length.",
    )
    parser.add_argument(
        "--top-n", type=int, default=TOP_N_CONFIGS, help="How many candidates to show."
    )
    args = parser.parse_args()

    _print_knob_glossary()

    sample_texts = []
    if args.input_path:
        distiset = load_data(args.input_path)
        sample_texts = [row["text"] for row in distiset["default"]["train"]]

    source = f"{len(sample_texts)} real samples" if sample_texts else "no dataset -- rough preview"
    console.print(f"\n[bold]Ranking training configs for '{args.child_model_id}'[/bold] ({source})...\n")

    shortlist = recommend_configs(args.child_model_id, sample_texts, top_n=args.top_n)
    _print_shortlist(shortlist)


if __name__ == "__main__":
    main()
