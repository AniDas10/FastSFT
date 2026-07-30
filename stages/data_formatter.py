"""DataFormatter mini-pipeline: renders a dataset into a target model's chat format."""

from datasets import DatasetDict
from distilabel.distiset import Distiset
from transformers import AutoTokenizer

from stages.base import Stage


class DataFormatter(Stage):
    """Second stage of the DistillationPipeline.

    Renders each row's `messages` (a list of `{"role": ..., "content": ...}`
    dicts) into the exact chat-template text the target ("child") model
    expects for fine-tuning, via that model's own Hugging Face tokenizer --
    every open instruct/chat model ships a `chat_template` alongside its
    weights, so this needs no per-model formatting logic of its own (see
    README for why).

    Deliberately schema-agnostic beyond the `messages` column itself: it
    doesn't care how many turns there are or which stage produced them, so
    it isn't coupled to DataGenerator specifically -- any stage (or hand-
    supplied dataset via skip_generation) that emits `messages` in this
    shape works.
    """

    def __init__(self, child_model_id: str, verbose: bool = True):
        super().__init__(verbose=verbose)
        self._child_model_id = child_model_id
        self._tokenizer = AutoTokenizer.from_pretrained(child_model_id)
        if self._tokenizer.chat_template is None:
            raise ValueError(
                f"'{child_model_id}' has no chat_template -- it's likely a "
                "base (non-instruct) model. DataFormatter needs a chat/"
                "instruct model that ships a chat_template."
            )

    def _validate_input(self, distiset: Distiset) -> None:
        train = distiset["default"]["train"]
        if "messages" not in train.column_names:
            raise ValueError(
                "DataFormatter.run() requires a 'messages' column (a list of "
                "{'role': ..., 'content': ...} dicts) in the input distiset; "
                f"got columns: {train.column_names}. This usually means a "
                "hand-supplied raw_dataset (skip_generation) doesn't match "
                "the expected schema."
            )
        if len(train) > 0:
            first = train[0]["messages"]
            valid = isinstance(first, list) and first and all(
                isinstance(m, dict) and "role" in m and "content" in m for m in first
            )
            if not valid:
                raise ValueError(
                    "DataFormatter.run() requires each row's 'messages' to be "
                    "a non-empty list of {'role': ..., 'content': ...} dicts; "
                    f"row 0 was: {first!r}."
                )

    def run(self, distiset: Distiset) -> Distiset:
        """Adds a `text` column: each row's `messages` rendered through the
        child model's chat template. `tokenize=False` keeps the output as
        plain text (viewable), not token ids -- numeric tokenization is
        FineTuner's concern at train time.
        """
        self._validate_input(distiset)
        self._log(
            f"Formatting dataset for child model '{self._child_model_id}'..."
        )
        train = distiset["default"]["train"]

        def render(row):
            return {
                "text": self._tokenizer.apply_chat_template(
                    row["messages"], tokenize=False
                )
            }

        train = train.map(render)
        self._log(f"Done: formatted {len(train)} samples.")
        return Distiset({"default": DatasetDict({"train": train})})
