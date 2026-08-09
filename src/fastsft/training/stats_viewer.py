"""Terminal rendering + CLI for training telemetry -- the `rich` presentation
layer over training/stats.py's core logic, and the `python -m` entry point:

    uv run python -m fastsft.training.stats_viewer [adapter_dir] [--json]

Kept separate from the core so the data/logic (loading, structuring, diagnosing)
carries no `rich` dependency and stays reusable on its own; this module imports
that core one-directionally.
"""

import fastsft.warnings_filter  # noqa: F401

import argparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from fastsft.findings_view import findings_panel
from fastsft.helper import latest_run_path, modelsets_dir
from fastsft.training.stats import (
    RunInterpreter,
    load_stats,
    series,
    stats_as_json,
)

console = Console()

# Metric -> plain-English "what it means", shown alongside each number so the
# table reads as an explanation, not just a dump of figures.
METRIC_DESCRIPTIONS = {
    "Validation loss": (
        "Error on held-out examples the model never trained on -- the honest "
        "signal. Lower is better. If it climbs while training loss keeps "
        "falling, the model is overfitting."
    ),
    "Best validation loss": (
        "The lowest validation loss reached. `load_best_model_at_end` kept THIS "
        "checkpoint -- not necessarily the final one."
    ),
    "Training loss": (
        "Error on the examples the model trained on. Should fall steadily; "
        "falling then flattening means it has fit the training data."
    ),
    "Token accuracy": (
        "Fraction of next-token predictions correct on held-out data. An "
        "intuitive companion to loss -- higher is better."
    ),
    "Epochs run": (
        "Full passes over the dataset actually completed, vs the max allowed. "
        "Fewer than the max means early stopping ended training once it stopped "
        "improving."
    ),
    "Optimizer steps": "Total weight updates performed across the whole run.",
}

def _chart(chart_series: list, width: int = 56, height: int = 12) -> str:
    """Plots (step, loss) points as a terminal line chart. `chart_series` is a
    list of (points, style, marker); losses share one y-scale so curves are
    comparable. Higher loss sits at the top; the y-axis shows the loss range."""
    points = [pt for pts, _, _ in chart_series for pt in pts]
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)

    grid = [[" "] * width for _ in range(height)]
    for pts, style, marker in chart_series:
        for x, y in pts:
            col = 0 if xmax == xmin else round((x - xmin) / (xmax - xmin) * (width - 1))
            row = 0 if ymax == ymin else round((ymax - y) / (ymax - ymin) * (height - 1))
            grid[row][col] = f"[{style}]{marker}[/{style}]"

    rows = []
    for r in range(height):
        label = f"{ymax:6.3f}" if r == 0 else f"{ymin:6.3f}" if r == height - 1 else " " * 6
        rows.append(f"{label} │" + "".join(grid[r]))
    rows.append(" " * 7 + "└" + "─" * width)
    left, right = f"step {xmin}", str(xmax)
    rows.append(" " * 8 + left + " " * max(1, width - len(left) - len(right)) + right)
    return "\n".join(rows)


def _chart_panel(train_pts: list, eval_pts: list) -> Panel:
    chart_series = []
    if train_pts:
        chart_series.append((train_pts, "dim cyan", "·"))
    if eval_pts:
        chart_series.append((eval_pts, "bold magenta", "●"))
    legend = "[dim cyan]· training loss[/]     [bold magenta]● validation loss[/]"
    return Panel(
        _chart(chart_series) + "\n\n" + legend,
        title="Loss over training  (down and to the right = learning)",
        border_style="cyan",
    )


def _summary_table(stats: dict, train_pts: list, eval_pts: list, acc_pts: list) -> Table:
    table = Table(title="Summary", title_style="bold", show_lines=True)
    table.add_column("Metric", style="bold cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="green")
    table.add_column("What it means")

    best = stats.get("best_metric")
    if best is None and eval_pts:
        best = min(y for _, y in eval_pts)

    rows: list[tuple[str, str]] = []
    if eval_pts:
        rows.append(("Validation loss", f"{eval_pts[0][1]:.3f} → {eval_pts[-1][1]:.3f}"))
    if best is not None:
        rows.append(("Best validation loss", f"{best:.3f}"))
    if train_pts:
        rows.append(("Training loss", f"{train_pts[0][1]:.3f} → {train_pts[-1][1]:.3f}"))
    if acc_pts:
        rows.append(("Token accuracy", f"{acc_pts[-1][1]:.1%}"))
    epoch, max_ep = stats.get("epoch"), stats.get("num_train_epochs")
    if epoch is not None:
        rows.append(("Epochs run", f"{epoch:.2f}" + (f" / {int(max_ep)}" if max_ep else "")))
    if stats.get("global_step") is not None:
        rows.append(("Optimizer steps", str(stats["global_step"])))

    for name, value in rows:
        table.add_row(name, value, METRIC_DESCRIPTIONS.get(name, ""))
    return table


def _interpretation(stats: dict) -> Panel:
    """Renders the named diagnostic checks as a plain-English takeaways panel."""
    return findings_panel(RunInterpreter(stats).run())


def render(stats: dict, adapter_dir: str) -> None:
    log = stats.get("log_history", [])
    train_pts, eval_pts = series(log, "loss"), series(log, "eval_loss")
    acc_pts = series(log, "eval_mean_token_accuracy")

    console.print(
        Panel.fit(f"[bold]Training telemetry[/bold]\n[dim]{adapter_dir}[/dim]", border_style="cyan")
    )
    if len(eval_pts) >= 2 or len(train_pts) >= 2:
        console.print(_chart_panel(train_pts, eval_pts))
    else:
        console.print(
            Panel(
                "Only one evaluation was recorded, so there's no curve to plot yet. "
                "Train longer (--max-epochs) or evaluate more often (smaller --eval-steps).",
                title="Loss curve",
                border_style="yellow",
            )
        )
    console.print(_summary_table(stats, train_pts, eval_pts, acc_pts))
    console.print(_interpretation(stats))


def main():
    parser = argparse.ArgumentParser(
        description="Show a finished training run's loss curve and telemetry."
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
        help="Emit the parsed stats as machine-readable JSON instead of the rich view.",
    )
    args = parser.parse_args()
    adapter_dir = args.adapter_dir or latest_run_path(modelsets_dir())
    stats = load_stats(adapter_dir)
    if args.json:
        print(stats_as_json(stats, adapter_dir))
    else:
        render(stats, adapter_dir)


if __name__ == "__main__":
    main()
