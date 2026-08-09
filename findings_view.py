"""Shared `rich` rendering for diagnostic Findings -- the presentation half of
findings.py, used by both training/stats_viewer.py and eval/results_viewer.py.
"""

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from findings import Finding

# status -> (rich style, icon); "info" renders as a dim note.
_STATUS_STYLE = {"good": ("green", "✓"), "warn": ("yellow", "!")}


def format_finding(finding: Finding) -> str:
    """A finding as rich-markup text: a colored icon and its message (or a dim
    note for "info")."""
    if finding.status == "info":
        return f"[dim]{finding.message}[/dim]"
    style, icon = _STATUS_STYLE[finding.status]
    return f"[{style}]{icon}[/{style}] {finding.message}"


def findings_panel(findings: list[Finding], title: str = "How to read this") -> Panel:
    """Renders a list of findings as a titled takeaways panel."""
    return Panel(
        Group(*(Text.from_markup(format_finding(f)) for f in findings)),
        title=title,
        border_style="cyan",
    )
