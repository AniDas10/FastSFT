"""One shared `rich` console for main-process progress output, so the pipeline
stages, the Evaluator, and the CLIs (main.py, eval/run.py) log in a single
consistent style that matches the terminal viewers (findings_view.py, the
*_viewer.py modules).

The trainer core (training/trainer.py) and the Modal job (training/modal_app.py)
stay on plain print() on purpose -- they run inside the Modal image / remote
worker, kept free of a `rich` dependency.
"""

from rich.console import Console

console = Console()


def log(message: str) -> None:
    """Print one progress line through the shared console. Markup and
    highlighting are off, so bracketed step markers like "[1/4]" and quoted
    file paths render literally rather than being parsed as rich markup."""
    console.print(message, markup=False, highlight=False)


class ProgressLogger:
    """Verbosity-gated progress logging over the shared console -- the `_log`
    helper shared by Stage and Evaluator (inherit this, call self._log)."""

    def __init__(self, verbose: bool = True):
        self._verbose = verbose

    def _log(self, message: str) -> None:
        if self._verbose:
            log(message)
