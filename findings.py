"""The shared diagnostic-finding record, produced by the training-stats and
evaluation interpreters (training/stats.py, eval/results.py) and consumed by
their viewers.

Pure stdlib -- no `rich` -- so the core logic modules stay presentation-free.
Rich rendering for a Finding lives in findings_view.py.
"""

from dataclasses import dataclass


@dataclass
class Finding:
    """One diagnostic takeaway: a severity and a message.

    `status` is "good" | "warn" | "info"; the message carries no styling markup,
    so the terminal renderer (findings_view.py) and the JSON output share it.
    """

    status: str
    message: str
