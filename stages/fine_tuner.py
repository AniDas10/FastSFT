"""FineTuner mini-pipeline: trains the child model on a formatted dataset."""

from distilabel.distiset import Distiset

from stages.base import Stage


class FineTuner(Stage):
    """Third stage of the DistillationPipeline.

    Fine-tunes the child model on a dataset already rendered to its chat
    template (see DataFormatter). Scaffold only: training logic (framework,
    hyperparameters) is a deliberately deferred follow-up task, not part of
    this pass.
    """

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

    def run(self, formatted_distiset: Distiset):
        self._validate_input(formatted_distiset)
        raise NotImplementedError(
            "FineTuner training isn't implemented yet. formatted_distiset is "
            "expected to already have a 'text' column rendered via the child "
            "model's chat template (see DataFormatter) -- training logic "
            "(framework, hyperparameters) is a separate follow-up task."
        )
