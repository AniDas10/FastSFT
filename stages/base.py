"""Shared base for DistillationPipeline's mini-pipeline stages."""

from typing import Any


class Stage:
    """Common verbose/_log plumbing shared by DataGenerator, DataFormatter,
    and FineTuner -- mirrors how model/base.py's Model anchors Judge/Guide.

    Every stage must implement _validate_input, called at the top of its
    own run() before doing any real work. This is what makes each stage's
    input contract explicit and self-defending regardless of caller (the
    top-level DistillationPipeline, or the stage used standalone) -- not an
    abstract method (no `abc` dependency), but the base implementation
    raises NotImplementedError so a subclass that forgets to override it
    fails loudly rather than silently validating nothing.
    """

    def __init__(self, verbose: bool = True):
        self._verbose = verbose

    def _log(self, message: str) -> None:
        if self._verbose:
            print(message)

    def _validate_input(self, data: Any) -> None:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _validate_input()."
        )
