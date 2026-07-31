"""FineTuner mini-pipeline: trains the child model on a formatted dataset."""

from distilabel.distiset import Distiset

from stages.base import Stage
from stages.constants import FINE_TUNER


class FineTuner(Stage):
    """Fine-tunes the child model on a chat-template-rendered dataset.
    Scaffold only -- training not yet implemented.
    """

    name = FINE_TUNER

    def __init__(self, child_model_id: str, verbose: bool = True):
        super().__init__(verbose=verbose)
        self._child_model_id = child_model_id

    def _validate_input(self, formatted_distiset: Distiset) -> None:
        train = formatted_distiset["default"]["train"]
        if "text" not in train.column_names:
            raise ValueError(
                "FineTuner.run() requires a 'text' column in the input "
                "distiset, rendered via the child model's chat template "
                "(see DataFormatter)."
            )

    def _run(self, formatted_distiset: Distiset):
        raise NotImplementedError("FineTuner training isn't implemented yet.")
