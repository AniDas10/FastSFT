"""Local inference over the child model -- tuned (adapter applied) and untuned
(base only) -- to produce the answer pairs the evaluation scores.

Requires the `evaluation` optional dependency group:
`uv sync --extra evaluation` (torch, peft, accelerate, sentence-transformers).

Everything but the base weights is recovered from the adapter directory itself
(a standard PEFT save): the base model id via PeftConfig, and the tokenizer +
exact chat template from the saved files -- so only the adapter dir is needed.
Base weights load once; the adapter is toggled off in place via
PeftModel.disable_adapter(), so tuned and untuned answers share a single load.

The `python -m` spot-check CLI (rich-rendered) lives in eval/inference_viewer.py;
this module only powers it.
"""

from fastsft.device import detect_device, dtype_for_device
from fastsft.eval.constants import DEFAULT_INFERENCE_BATCH_SIZE, DEFAULT_MAX_NEW_TOKENS


class ChildInferenceEngine:
    """Loads a child adapter (and its frozen base) once and batch-generates
    answers with the adapter either applied (tuned) or bypassed (untuned)."""

    def __init__(
        self,
        adapter_dir: str,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        batch_size: int = DEFAULT_INFERENCE_BATCH_SIZE,
    ):
        self._adapter_dir = adapter_dir
        self._max_new_tokens = max_new_tokens
        self._batch_size = batch_size
        self._device = detect_device()
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        """Loads base weights + adapter + tokenizer on first use (idempotent)."""
        if self._model is not None:
            return

        from peft import PeftConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        base_id = PeftConfig.from_pretrained(self._adapter_dir).base_model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(self._adapter_dir)
        # Batched decoder-only generation needs a pad token and left padding so
        # each sequence's real tokens sit flush against the generated ones.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        base = AutoModelForCausalLM.from_pretrained(
            base_id,
            dtype=dtype_for_device(self._device),
            device_map={"": self._device},
        )
        self._model = PeftModel.from_pretrained(base, self._adapter_dir)
        self._model.eval()
        self._tokenizer = tokenizer

    def generate_tuned(self, prompts: list[str]) -> list[str]:
        """One answer per prompt from the fine-tuned child (adapter applied)."""
        self._load()
        return self._generate(prompts)

    def generate_untuned(self, prompts: list[str]) -> list[str]:
        """One answer per prompt from the untuned child (base weights only)."""
        self._load()
        with self._model.disable_adapter():
            return self._generate(prompts)

    def _generate(self, prompts: list[str]) -> list[str]:
        answers: list[str] = []
        for start in range(0, len(prompts), self._batch_size):
            answers.extend(self._generate_batch(prompts[start : start + self._batch_size]))
        return answers

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        import torch

        tokenizer = self._tokenizer
        # No system prompt: training examples carried none, so the child answers
        # from the bare user turn. add_generation_prompt cues the assistant turn.
        conversations = [[{"role": "user", "content": prompt}] for prompt in prompts]
        inputs = tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the tokens generated past the (left-padded) prompt.
        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, prompt_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [answer.strip() for answer in decoded]
