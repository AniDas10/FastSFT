"""View samples from a saved synthetic dataset in the terminal."""

import warnings_filter  # noqa: F401

import argparse
import os

from rich.console import Console
from rich.panel import Panel

from constants import DEFAULT_OUTPUT_DIR, FORMATTED_OUTPUT_SUBDIR, RAW_OUTPUT_SUBDIR
from helper import load_data

console = Console()

_ROLE_COLORS = {"user": "cyan", "assistant": "magenta"}


def _format_message(message: dict) -> str:
    role = message.get("role", "?")
    color = _ROLE_COLORS.get(role, "white")
    return f"[bold {color}]{role}[/bold {color}]: {message.get('content', '')}"


def _latest_run_path(base_dir: str) -> str:
    """Returns the most recent timestamped run folder under `base_dir`."""
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


class DataViewer:
    """Loads a saved `Distiset` and previews samples."""

    def __init__(self, path: str | None = None, kind: str = "raw"):
        if path is None:
            subdir = FORMATTED_OUTPUT_SUBDIR if kind == "formatted" else RAW_OUTPUT_SUBDIR
            path = _latest_run_path(os.path.join(DEFAULT_OUTPUT_DIR, subdir))
        distiset = load_data(path)
        self.dataset = distiset["default"]["train"]

    def raw_samples(self, n: int = 5) -> None:
        """Prints the first `n` raw samples' `messages` as generated."""
        for i, row in enumerate(self.dataset.select(range(min(n, len(self.dataset))))):
            messages = row.get("messages", [])
            body = "\n\n".join(_format_message(m) for m in messages)
            console.print(
                Panel(
                    body,
                    title=f"[bold cyan][{i}][/bold cyan]",
                    border_style="cyan",
                    expand=False,
                )
            )

    def formatted_samples(self, n: int = 5) -> None:
        """Prints the first `n` samples' `text` column."""
        for i, row in enumerate(self.dataset.select(range(min(n, len(self.dataset))))):
            console.print(
                Panel(
                    row.get("text", ""),
                    title=f"[bold cyan][{i}][/bold cyan]",
                    border_style="cyan",
                    expand=False,
                )
            )


def main():
    parser = argparse.ArgumentParser(description="Preview samples from a saved distilabel dataset.")
    parser.add_argument(
        "--path",
        default=None,
        help=f"Directory the dataset was saved to (default: latest run under "
        f"{DEFAULT_OUTPUT_DIR}/{RAW_OUTPUT_SUBDIR}/ or {DEFAULT_OUTPUT_DIR}/"
        f"{FORMATTED_OUTPUT_SUBDIR}/, depending on --formatted).",
    )
    parser.add_argument("--num-samples", type=int, default=5, help="Number of samples to display.")
    parser.add_argument(
        "--formatted",
        action="store_true",
        help="Show the DataFormatter-rendered 'text' column instead of raw 'messages'.",
    )
    args = parser.parse_args()

    kind = "formatted" if args.formatted else "raw"
    viewer = DataViewer(args.path, kind=kind)
    if args.formatted:
        viewer.formatted_samples(args.num_samples)
    else:
        viewer.raw_samples(args.num_samples)


if __name__ == "__main__":
    main()
