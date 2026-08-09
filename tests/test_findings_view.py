"""Tier-1 unit tests for fastsft.findings_view (rich rendering of Findings)."""

import pytest
from rich.panel import Panel

from fastsft.findings import Finding
from fastsft.findings_view import findings_panel, format_finding


@pytest.mark.parametrize(
    "status,expected",
    [
        ("good", "[green]✓[/green] all good"),
        ("warn", "[yellow]![/yellow] all good"),
    ],
)
def test_format_finding_good_and_warn(status, expected):
    assert format_finding(Finding(status, "all good")) == expected


def test_format_finding_info_is_dim():
    # "info" renders as a plain dim note -- no icon.
    assert format_finding(Finding("info", "a note")) == "[dim]a note[/dim]"


def test_format_finding_preserves_message_text():
    msg = "loss fell 1.234 -> 0.567 (-54%)"
    assert msg in format_finding(Finding("good", msg))


def test_findings_panel_is_panel_with_default_title():
    panel = findings_panel([Finding("good", "x"), Finding("warn", "y")])
    assert isinstance(panel, Panel)
    assert panel.title == "How to read this"
    assert panel.border_style == "cyan"


def test_findings_panel_custom_title():
    panel = findings_panel([Finding("info", "x")], title="Summary")
    assert panel.title == "Summary"


def test_findings_panel_one_renderable_per_finding():
    findings = [Finding("good", "a"), Finding("warn", "b"), Finding("info", "c")]
    panel = findings_panel(findings)
    # The panel wraps a rich Group; it holds one Text per finding.
    assert len(panel.renderable.renderables) == len(findings)


def test_findings_panel_empty():
    panel = findings_panel([])
    assert isinstance(panel, Panel)
    assert len(panel.renderable.renderables) == 0
