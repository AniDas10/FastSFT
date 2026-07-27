"""Shared access to OpenRouter's model catalog, and the base Model class."""

import warnings_filter  # noqa: F401

import os
from functools import lru_cache
from typing import List, Optional

import requests
from dotenv import load_dotenv

from distilabel.distiset import Distiset
from distilabel.models.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration

from constants import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_PARENT_MODEL,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODELS_URL,
)

load_dotenv()


@lru_cache(maxsize=1)
def _fetch_openrouter_models() -> dict:
    """Fetches OpenRouter's model catalog, keyed by model id.

    Cached for the life of the process -- every Model/Judge/Guide
    construction calls this to validate open-weight status, and the catalog
    doesn't change meaningfully within a single run.
    """
    response = requests.get(OPENROUTER_MODELS_URL, timeout=30)
    response.raise_for_status()
    return {m["id"]: m for m in response.json()["data"]}


class Model:
    """An OpenRouter-backed model playing a generation/judging role.

    Handles common initialization (api key, temperature, live open-weight
    validation) and per-instance instruction (system prompt) storage. Usable
    directly for a role that needs nothing extra (e.g. as the "parent"
    generation model), or subclassed for roles that add real behavior of
    their own (see Judge).
    """

    def __init__(
        self,
        model_id: str = DEFAULT_PARENT_MODEL,
        api_key: Optional[str] = None,
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
        self._custom_instruction: Optional[str] = None

        self._assert_open_weight()

    def _assert_open_weight(self) -> None:
        """Rejects models OpenRouter doesn't mirror on Hugging Face.

        OpenRouter's API has no explicit "allows distillation" flag. The
        closest real signal it exposes is `hugging_face_id`: it's set only
        for open-weight models (Llama, Qwen, Mistral, ...) whose licenses
        permit using outputs to train/distill other models, and is `null`
        for closed-weight models (GPT, Claude, Gemini, ...) whose provider
        ToS typically forbid that. Required for every model role.
        """
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
        """This role's default instruction (system prompt).

        Empty by default -- instructions are expected to come from
        set_instruction() (e.g. via model/guide.py's Guide) rather than a
        generic hardcoded fallback. Subclasses may still override this for
        role-specific defaults (see Judge).
        """
        return ""

    def set_instruction(self, instruction: str) -> None:
        self._custom_instruction = instruction

    def get_instruction(self) -> str:
        if self._custom_instruction is not None:
            return self._custom_instruction
        return self._instruction()

    def _assert_structured_output(
        self, generation: Optional[str], sample_id: Optional[str] = None
    ) -> str:
        """Raises a clear error if a structured-output call returned nothing.

        A `None`/empty `generation` here almost always means the response
        was truncated (raise max_tokens) or the model doesn't reliably
        support tool calls (try a different model_id) -- both Guide and
        Judge hit this, so it's centralized here rather than duplicated.
        """
        if not generation:
            context = f" for sample '{sample_id}'" if sample_id is not None else ""
            raise RuntimeError(
                f"Model '{self.model_id}' returned no structured output{context}. "
                "This usually means the response was truncated or it doesn't "
                "reliably support tool calls -- try a different model_id or "
                "increase max_tokens."
            )
        return generation

    def build_llm(self, structured_output: Optional[dict] = None) -> OpenAILLM:
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
        data: List[dict],
        system_prompt: str,
        structured_output: Optional[dict] = None,
        name: str = "pipeline",
    ) -> Distiset:
        """Runs a single-step generation pipeline over `data`.

        Shared by every role (Judge.evaluate/score_samples,
        Guide.generate_instructions, SyntheticDatasetGenerator.generate) --
        all of them are "load rows, run one TextGeneration task" and
        differed only in `data`/`system_prompt`/`structured_output`. Public
        (not protected) because it's called both by Model subclasses and by
        SyntheticDatasetGenerator, which holds a Model by composition rather
        than inheritance.
        """
        with Pipeline(name=name) as pipeline:
            load_data = LoadDataFromDicts(data=data)
            task = TextGeneration(
                llm=self.build_llm(structured_output=structured_output),
                system_prompt=system_prompt,
            )
            load_data >> task

        return pipeline.run(use_cache=False)
