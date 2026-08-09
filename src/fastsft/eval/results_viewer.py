"""Terminal rendering + CLI for a finished evaluation run -- the `rich`
presentation layer over eval/results.py's core logic, and the `python -m`
entry point:

    uv run python -m fastsft.eval.results_viewer [adapter_dir] [--json]

Kept separate from the core so the data/logic (persist, load, interpret) carries
no `rich` dependency and stays reusable on its own; this module imports that
core one-directionally.
"""

import fastsft.warnings_filter  # noqa: F401

import argparse

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from fastsft.constants import MODELSETS_OUTPUT_DIR
from fastsft.eval.results import interpret, load_results, results_as_json
from fastsft.findings_view import findings_panel
from fastsft.helper import latest_run_path

console = Console()


def _comparisons_table(results: dict) -> Table:
    table = Table(title="Pairwise win rates", title_style="bold", show_lines=True)
    table.add_column("Comparison", style="bold cyan", no_wrap=True)
    table.add_column("Tuned win rate", justify="right", style="green")
    table.add_column("W / T / L", justify="right")
    table.add_column("What it means")

    descriptions = {
        "tuned_vs_untuned": "Primary quality signal: did fine-tuning beat the "
        "untuned child? >50% means it helped.",
        "parent_likeness": "Distillation objective: is the tuned child more like "
        "the parent's style than the untuned child? >50% = style is transferring.",
        "tuned_vs_parent": "Gap to the teacher: how often the small tuned child "
        "matches the parent. A gap is expected.",
    }
    labels = {
        "tuned_vs_untuned": "Tuned vs untuned (quality)",
        "parent_likeness": "Parent-style match",
        "tuned_vs_parent": "Tuned vs parent (quality)",
    }
    for key, comparison in results.get("comparisons", {}).items():
        table.add_row(
            labels.get(key, key),
            f"{comparison['win_rate']:.0%}",
            f"{comparison['wins']} / {comparison['ties']} / {comparison['losses']}",
            descriptions.get(key, ""),
        )
    return table


def _similarity_table(results: dict) -> Table | None:
    similarity = results.get("similarity_to_parent") or {}
    tuned = similarity.get("tuned_vs_parent")
    untuned = similarity.get("untuned_vs_parent")
    if tuned is None and untuned is None:
        return None

    table = Table(title="Embedding similarity to parent", title_style="bold", show_lines=True)
    table.add_column("Pair", style="bold cyan", no_wrap=True)
    table.add_column("Mean cosine", justify="right", style="green")
    table.add_column("What it means")
    table.add_row(
        "Tuned ↔ parent",
        "n/a" if tuned is None else f"{tuned:.3f}",
        "Distillation fidelity: higher = the tuned child's answers sit closer "
        "to the parent's.",
    )
    table.add_row(
        "Untuned ↔ parent",
        "n/a" if untuned is None else f"{untuned:.3f}",
        "The baseline distance -- tuning should raise the tuned figure above this.",
    )
    return table


def _interpretation(results: dict) -> Panel:
    return findings_panel(interpret(results))


def _samples_panel(results: dict) -> Panel | None:
    samples = results.get("samples") or []
    if not samples:
        return None
    blocks = []
    for i, sample in enumerate(samples):
        blocks.append(
            Text.from_markup(
                f"[bold cyan][{i}] prompt:[/bold cyan] {sample['prompt']}\n\n"
                f"[bold green]tuned:[/bold green] {sample['tuned']}\n\n"
                f"[dim]untuned:[/dim] {sample['untuned']}\n\n"
                f"[bold magenta]parent:[/bold magenta] {sample['parent']}"
            )
        )
    return Panel(
        Group(*blocks),
        title="Example answers (tuned / untuned / parent)",
        border_style="cyan",
    )


def render(results: dict, adapter_dir: str) -> None:
    console.print(
        Panel.fit(
            f"[bold]Evaluation results[/bold]\n[dim]{adapter_dir}[/dim]\n"
            f"{results.get('num_prompts', 0)} prompts · judge "
            f"{results.get('judge_model', '?')} · parent {results.get('parent_model', '?')}",
            border_style="cyan",
        )
    )
    console.print(_comparisons_table(results))
    similarity_table = _similarity_table(results)
    if similarity_table is not None:
        console.print(similarity_table)
    console.print(_interpretation(results))
    samples_panel = _samples_panel(results)
    if samples_panel is not None:
        console.print(samples_panel)


def main():
    parser = argparse.ArgumentParser(
        description="Show a finished evaluation run's win rates, similarity, and takeaways."
    )
    parser.add_argument(
        "adapter_dir",
        nargs="?",
        default=None,
        help="Adapter directory to read (default: latest run under modelsets/).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the results (plus findings) as machine-readable JSON.",
    )
    args = parser.parse_args()
    adapter_dir = args.adapter_dir or latest_run_path(MODELSETS_OUTPUT_DIR)
    results = load_results(adapter_dir)
    if args.json:
        print(results_as_json(results))
    else:
        render(results, adapter_dir)


if __name__ == "__main__":
    main()
