"""View samples from a saved synthetic dataset in the terminal."""

import warnings_filter  # noqa: F401

import argparse
import os
from typing import Optional

from distilabel.distiset import Distiset
from rich.console import Console
from rich.panel import Panel

from constants import DEFAULT_OUTPUT_DIR

console = Console()


def _latest_run_path(base_dir: str = DEFAULT_OUTPUT_DIR) -> str:
    """Returns the most recent timestamped run folder under `base_dir`.

    Run folder names are `RUN_TIMESTAMP_FORMAT`-formatted, which sorts
    lexicographically in chronological order, so the last one alphabetically
    is also the most recent.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"No '{base_dir}' directory found. Run main.py first, or pass path=..."
        )

    runs = sorted(
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    )
    if not runs:
        raise FileNotFoundError(f"No runs found under '{base_dir}'.")

    return os.path.join(base_dir, runs[-1])


class DatasetViewer:
    """Loads a `Distiset` saved via `SyntheticDatasetGenerator` and previews samples."""

    def __init__(
        self,
        path: Optional[str] = None,
        config: str = "default",
        split: str = "train",
    ):
        distiset = Distiset.load_from_disk(path or _latest_run_path())
        self.dataset = distiset[config][split]

    def raw_samples(self, n: int = 5) -> None:
        """Prints the first `n` raw (instruction, generation) samples as generated."""
        for i, row in enumerate(self.dataset.select(range(min(n, len(self.dataset))))):
            body = row.get("generation", "")
            title = row.get("instruction", f"Sample {i}")
            console.print(Panel(body, title=f"[{i}] {title}", expand=False))


def main():
    parser = argparse.ArgumentParser(description="Preview samples from a saved distilabel dataset.")
    parser.add_argument(
        "--path",
        default=None,
        help=f"Directory the dataset was saved to (default: latest run under {DEFAULT_OUTPUT_DIR}/).",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to display.")
    args = parser.parse_args()

    DatasetViewer(args.path).raw_samples(args.num_samples)


if __name__ == "__main__":
    main()
