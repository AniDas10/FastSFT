"""FineTuner mini-pipeline: LoRA/QLoRA SFT on the child model, dispatched to Modal."""

import os
import shutil
import tarfile
import tempfile
import uuid
from dataclasses import asdict
from typing import Optional

from distilabel.distiset import Distiset

from constants import MODELS_OUTPUT_DIR
from stages.base import Stage
from stages.constants import FINE_TUNER
from training.config import TrainingConfig
from training.constants import TOP_N_CONFIGS
from training.heuristic import recommend_configs
from training.modal_app import adapter_volume, train_lora


class FineTuner(Stage):
    """Fine-tunes the child model via LoRA/QLoRA SFT, dispatched to Modal.

    Uses `training_config` if the caller supplies one; otherwise ranks
    candidates by cost/feasibility (see training.heuristic) and takes the
    cheapest feasible one. Either way, carves a validation split for early
    stopping and dispatches training to a Modal GPU container.
    """

    name = FINE_TUNER

    def __init__(
        self,
        child_model_id: str,
        training_config: Optional[TrainingConfig] = None,
        verbose: bool = True,
    ):
        super().__init__(verbose=verbose)
        self._child_model_id = child_model_id
        self._training_config = training_config

    def _validate_input(self, formatted_distiset: Distiset) -> None:
        train = formatted_distiset["default"]["train"]
        if "text" not in train.column_names:
            raise ValueError(
                "FineTuner.run() requires a 'text' column in the input "
                "distiset, rendered via the child model's chat template "
                "(see DataFormatter)."
            )

    def _run(self, formatted_distiset: Distiset) -> str:
        # Resolved before splitting: the heuristic measures sequence lengths
        # from the full dataset, and the resolved config's own
        # loop.validation_split then decides how the split is carved.
        chosen = self._resolve_training_config(formatted_distiset)
        train_ds, eval_ds = self._split_validation(
            formatted_distiset, chosen.loop.validation_split
        )
        self._log(
            f"[1/3] Split into {len(train_ds)} train / {len(eval_ds)} eval "
            f"({chosen.loop.validation_split:.0%} held out)."
        )

        self._log(f"[2/3] Dispatching training to Modal ({chosen.gpu_tier})...")
        job_id = uuid.uuid4().hex[:12]
        tar_path = train_lora.with_options(
            gpu=chosen.gpu_tier, timeout=chosen.modal_timeout_seconds
        ).remote(
            child_model_id=self._child_model_id,
            train_rows=train_ds.to_list(),
            eval_rows=eval_ds.to_list(),
            config=asdict(chosen),
            run_id=job_id,
        )
        self._log("[2/3] Done: training completed on Modal.")

        self._log("[3/3] Downloading trained adapter...")
        adapter_dir = self._download_adapter(tar_path)
        self._log(f"[3/3] Done: adapter downloaded to '{adapter_dir}'.")

        return adapter_dir

    def _resolve_training_config(self, formatted_distiset: Distiset) -> TrainingConfig:
        """Uses the caller-supplied TrainingConfig if given; otherwise ranks
        candidates via the cost heuristic and takes the cheapest feasible one."""
        if self._training_config is not None:
            self._log(
                f"[1/3] Using caller-supplied config: "
                f"{self._training_config.gpu_tier} / "
                f"{self._training_config.strategy}."
            )
            return self._training_config

        self._log("[1/3] Ranking candidate training configs...")
        sample_texts = [row["text"] for row in formatted_distiset["default"]["train"]]
        shortlist = recommend_configs(
            self._child_model_id, sample_texts, top_n=TOP_N_CONFIGS
        )
        for i, cfg in enumerate(shortlist):
            self._log(
                f"    [{i}] {cfg.gpu_tier} / {cfg.strategy} / "
                f"rank {cfg.adapter.rank} / batch {cfg.loop.batch_size} -- "
                f"~${cfg.est_usd_per_hour}/hr, ~{cfg.est_memory_gb}GB estimated"
            )
        chosen = shortlist[0]
        self._log(
            f"[1/3] Done: selected {chosen.gpu_tier} / {chosen.strategy} "
            "(cheapest feasible)."
        )
        return chosen

    def save_output(self, output: str, run_id: str) -> str:
        """Copies the downloaded adapter directory into
        MODELS_OUTPUT_DIR/run_id and returns that path."""
        destination = os.path.join(MODELS_OUTPUT_DIR, run_id)
        shutil.copytree(output, destination, dirs_exist_ok=True)
        return destination

    def _split_validation(self, distiset: Distiset, validation_split: float):
        """Carves a held-out validation slice from the formatted dataset,
        for the Modal job's early stopping to monitor."""
        train = distiset["default"]["train"]
        split = train.train_test_split(test_size=validation_split, seed=42)
        return split["train"], split["test"]

    def _download_adapter(self, tar_path: str) -> str:
        """Downloads the trained adapter tarball from the Modal Volume and
        extracts it to a local temp directory; returns that path."""
        local_tar_path = tempfile.mktemp(suffix=".tar.gz")
        with open(local_tar_path, "wb") as f:
            for chunk in adapter_volume.read_file(tar_path):
                f.write(chunk)

        extract_dir = tempfile.mkdtemp(prefix="finetuner_adapter_")
        with tarfile.open(local_tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        return extract_dir
