"""DataFormatter mini-pipeline: renders a dataset into a target model's chat format."""

from distilabel.distiset import Distiset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from fastsft.constants import FORMATTED_OUTPUT_SUBDIR
from fastsft.helper import convert_to_distiset, save_distiset
from fastsft.stages.base import Stage
from fastsft.stages.constants import DATA_FORMATTER


class DataFormatter(Stage):
    """Renders each row's `messages` into the child model's chat-template text
    via that model's Hugging Face tokenizer.
    """

    name = DATA_FORMATTER
    title = "Data Formatting Stage"

    def __init__(self, child_model_id: str, verbose: bool = True):
        super().__init__(verbose=verbose)
        self._child_model_id = child_model_id
        self._tokenizer: PreTrainedTokenizerBase | None = None

    def _load_tokenizer(self) -> PreTrainedTokenizerBase:
        """Loads, validates, and caches the child model's tokenizer on first use."""
        if self._tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(self._child_model_id)
            if tokenizer.chat_template is None:
                raise ValueError(
                    f"'{self._child_model_id}' has no chat_template -- DataFormatter "
                    "needs a chat/instruct model, not a base model."
                )
            self._tokenizer = tokenizer
        return self._tokenizer

    def save_output(self, output: Distiset, run_id: str) -> str:
        return save_distiset(output, FORMATTED_OUTPUT_SUBDIR, run_id)

    def _validate_input(self, distiset: Distiset) -> None:
        train = distiset["default"]["train"]
        if "messages" not in train.column_names:
            raise ValueError(
                "DataFormatter.run() requires a 'messages' column (a list of "
                "{'role': ..., 'content': ...} dicts) in the input distiset; "
                f"got columns: {train.column_names}. This usually means a "
                "hand-supplied dataset (via DistillationPipeline's "
                "start_stage) doesn't match the expected schema."
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

    def _run(self, distiset: Distiset) -> Distiset:
        """Adds a `text` column: `messages` rendered through the chat template."""
        self._log(
            f"Formatting dataset for child model '{self._child_model_id}'..."
        )
        tokenizer = self._load_tokenizer()
        train = distiset["default"]["train"]

        def render(row: dict) -> dict:
            return {
                "text": tokenizer.apply_chat_template(row["messages"], tokenize=False)
            }

        train = train.map(render)
        self._log(f"Done: formatted {len(train)} samples.")
        return convert_to_distiset(train)
