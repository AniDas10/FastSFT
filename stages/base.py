"""Shared base for DistillationPipeline's mini-pipeline stages."""

import os
from typing import Any, Optional

from distilabel.distiset import Distiset

from constants import DEFAULT_OUTPUT_DIR


def save_distiset(dataset: Distiset, subdir: str, run_id: str) -> str:
    """Saves `dataset` under DEFAULT_OUTPUT_DIR/subdir/run_id; returns the path."""
    path = os.path.join(DEFAULT_OUTPUT_DIR, subdir, run_id)
    dataset.save_to_disk(path)
    return path


class Stage:
    """Base for pipeline stages: shared logging and a validate-then-run template.

    Subclasses implement _validate_input and _run; run() is defined once here
    and not overridden.
    """

    # Canonical stage name, set by each subclass from stages/constants.py.
    # __init__ rejects a subclass that leaves it unset.
    name: str = ""

    def __init__(self, verbose: bool = True):
        if not self.name:
            raise NotImplementedError(
                f"{type(self).__name__} must set a non-empty class attribute "
                "`name` (its canonical stage name from stages/constants.py)."
            )
        self._verbose = verbose

    def _log(self, message: str) -> None:
        if self._verbose:
            print(message)

    def _validate_input(self, data: Any) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _validate_input()."
        )

    def _run(self, data: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement _run().")

    def run(self, data: Any) -> Any:
        """Validate the input contract, then run."""
        self._validate_input(data)
        return self._run(data)

    def save_output(self, output: Any, run_id: str) -> Optional[str]:
        """Persists this stage's output; returns the path, or None if there's
        nothing to persist. Overridden by stages that produce a saved artifact."""
        return None
