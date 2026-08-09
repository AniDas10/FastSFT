"""Shared access to OpenRouter's model catalog, and the base Model class."""

import fastsft.warnings_filter  # noqa: F401

import os
from functools import lru_cache

import requests
from distilabel.distiset import Distiset
from distilabel.models.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
from dotenv import load_dotenv

from fastsft.constants import DEFAULT_PARENT_MODEL
from fastsft.model._logging import detach_stale_queue_handlers
from fastsft.model.constants import (
    DEFAULT_MAX_TOKENS,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODELS_URL,
)

load_dotenv()


@lru_cache(maxsize=1)
def _fetch_openrouter_models() -> dict:
    """Fetch OpenRouter's model catalog."""
    response = requests.get(OPENROUTER_MODELS_URL, timeout=30)
    response.raise_for_status()
    return {m["id"]: m for m in response.json()["data"]}


class Model:
    """An OpenRouter-backed model playing a generation/judging role.

    Handles api key, temperature, open-weight validation, and per-instance
    instruction storage. Usable directly (as the parent) or subclassed (see Judge).
    """

    _enforce_open_weight: bool = True

    def __init__(
        self,
        model_id: str = DEFAULT_PARENT_MODEL,
        api_key: str | None = None,
        temperature: float = 0.9,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.model_id = model_id
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self._api_key:
            raise ValueError(
                "No OpenRouter API key found. Set OPENROUTER_API_KEY or pass api_key=..."
            )
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._custom_instruction: str | None = None
        self._open_weight_verified = False

    def _ensure_open_weight(self) -> None:
        """Run open-weight check once, on first use."""
        if self._enforce_open_weight and not self._open_weight_verified:
            self._assert_open_weight()
            self._open_weight_verified = True

    def _assert_open_weight(self) -> None:
        """Reject models without hugging_face_id on OpenRouter."""
        models = _fetch_openrouter_models()

        info = models.get(self.model_id)
        if info is None:
            raise ValueError(f"Model '{self.model_id}' was not found on OpenRouter.")

        if not info.get("hugging_face_id"):
            raise ValueError(
                f"Model '{self.model_id}' has no hugging_face_id on OpenRouter, "
                "meaning it's closed-weight and its provider's terms likely forbid "
                "using outputs to train/distill other models. Pick an open-weight "
                "model instead."
            )

    def _instruction(self) -> str:
        """Default instruction for this role (empty; overridden in subclasses)."""
        return ""

    def set_instruction(self, instruction: str) -> None:
        self._custom_instruction = instruction

    def get_instruction(self) -> str:
        if self._custom_instruction is not None:
            return self._custom_instruction
        return self._instruction()

    def assert_structured_output(
        self, generation: str | None, sample_id: str | None = None
    ) -> str:
        """Raises a clear error if a structured-output call returned nothing."""
        if not generation:
            context = f" for sample '{sample_id}'" if sample_id is not None else ""
            raise RuntimeError(
                f"Model '{self.model_id}' returned no structured output{context}. "
                "This usually means the response was truncated or it doesn't "
                "reliably support tool calls -- try a different model_id or "
                "increase max_tokens."
            )
        return generation

    def assert_generation(
        self, generation: str | None, sample_id: str | None = None
    ) -> str:
        """Raise error if generation returned empty (filtered/refused by provider)."""
        if not generation:
            context = f" for sample '{sample_id}'" if sample_id is not None else ""
            raise RuntimeError(
                f"Model '{self.model_id}' returned no generation{context}. "
                "This usually means the response was empty, filtered, or "
                "refused by the provider -- try a different model_id or prompt."
            )
        return generation

    def build_llm(self, structured_output: dict | None = None) -> OpenAILLM:
        self._ensure_open_weight()
        return OpenAILLM(
            model=self.model_id,
            base_url=OPENROUTER_BASE_URL,
            api_key=self._api_key,
            generation_kwargs={
                "temperature": self._temperature,
                "max_new_tokens": self._max_tokens,
            },
            structured_output=structured_output,
        )

    def run_pipeline(
        self,
        data: list[dict],
        system_prompt: str,
        structured_output: dict | None = None,
        name: str = "pipeline",
    ) -> Distiset:
        """Runs a single-step LoadDataFromDicts -> TextGeneration pipeline over `data`."""
        with Pipeline(name=name) as pipeline:
            load_data = LoadDataFromDicts(data=data)
            task = TextGeneration(
                llm=self.build_llm(structured_output=structured_output),
                system_prompt=system_prompt,
            )
            load_data >> task

        distiset = pipeline.run(use_cache=False)
        detach_stale_queue_handlers()
        return distiset
