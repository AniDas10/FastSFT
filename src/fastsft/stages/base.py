"""Shared base for DistillationPipeline's mini-pipeline stages."""

from typing import Any

from fastsft.progress import ProgressLogger, rule


class Stage(ProgressLogger):
    """Base for pipeline stages: shared logging and a validate-then-run template.

    Subclasses implement _validate_input and _run; run() is defined once here
    and not overridden. Progress logging (self._log) comes from ProgressLogger.
    """

    # Canonical stage name, set by each subclass from stages/constants.py.
    name: str = ""
    # Human-readable label for the start/end partition rule bracketing the stage
    # in the run output. Defaults to a title-cased `name` if a subclass omits it.
    title: str = ""

    def __init__(self, verbose: bool = True):
        if not self.name:
            raise NotImplementedError(
                f"{type(self).__name__} must set a non-empty class attribute "
                "`name` (its canonical stage name from stages/constants.py)."
            )
        super().__init__(verbose=verbose)

    def _banner_title(self) -> str:
        return self.title or f"{self.name.replace('_', ' ').title()} Stage"

    def _validate_input(self, data: Any) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _validate_input()."
        )

    def _run(self, data: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement _run().")

    def run(self, data: Any) -> Any:
        """Bracket the stage with start/end partition rules, validate, then run."""
        if self._verbose:
            rule(self._banner_title())
        self._validate_input(data)
        output = self._run(data)
        if self._verbose:
            rule(f"{self._banner_title()} complete", style="dim")
        return output

    def save_output(self, output: Any, run_id: str) -> str | None:
        """Persists this stage's output; returns the path, or None if there's
        nothing to persist. Overridden by stages that produce a saved artifact."""
        return None
