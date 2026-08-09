"""Terminal spot-check for a child adapter -- the `rich` presentation layer over
eval/inference.py's ChildInferenceEngine, and the `python -m` entry point:

    uv run python -m fastsft.eval.inference_viewer "your prompt here"

Prints the tuned answer (and the untuned baseline for comparison) so you can
eyeball what the adapter changed. Kept separate from the engine so the core
inference logic carries no `rich` dependency and stays reusable on its own; this
module imports that core one-directionally.
"""

import fastsft.warnings_filter  # noqa: F401

import argparse

from rich.console import Console
from rich.panel import Panel

from fastsft.eval.constants import DEFAULT_MAX_NEW_TOKENS
from fastsft.eval.inference import ChildInferenceEngine
from fastsft.helper import latest_run_path, modelsets_dir

console = Console()


def _input_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Registers CLI arguments on `parser` and returns the parsed args."""
    parser.add_argument("prompt", help="User prompt to generate an answer for.")
    parser.add_argument(
        "adapter_dir",
        nargs="?",
        default=None,
        help="Adapter directory to load (default: latest run under modelsets/).",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--tuned-only",
        action="store_true",
        help="Skip the untuned baseline; print only the tuned answer.",
    )
    return parser.parse_args()


def render(adapter_dir: str, prompt: str, tuned: str, untuned: str | None) -> None:
    """Renders the spot-check: a header, the tuned answer, and (unless skipped)
    the untuned baseline for side-by-side comparison."""
    console.print(
        Panel.fit(
            f"[bold]Adapter spot-check[/bold]\n[dim]{adapter_dir}[/dim]\n\n"
            f"[bold cyan]prompt[/bold cyan]  {prompt}",
            border_style="cyan",
        )
    )
    console.print(
        Panel(tuned or "[dim](empty)[/dim]", title="tuned  (adapter applied)", border_style="green")
    )
    if untuned is not None:
        console.print(
            Panel(
                untuned or "[dim](empty)[/dim]",
                title="untuned  (base only)",
                border_style="bright_black",
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Spot-check a child adapter on one prompt: print the tuned "
        "answer (and the untuned baseline for comparison)."
    )
    args = _input_args(parser)

    adapter_dir = args.adapter_dir or latest_run_path(modelsets_dir())
    engine = ChildInferenceEngine(adapter_dir, max_new_tokens=args.max_new_tokens)

    tuned = engine.generate_tuned([args.prompt])[0]
    untuned = None if args.tuned_only else engine.generate_untuned([args.prompt])[0]
    render(adapter_dir, args.prompt, tuned, untuned)


if __name__ == "__main__":
    main()
