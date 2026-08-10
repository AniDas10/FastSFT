"""FineTuner mini-pipeline: LoRA/QLoRA SFT on the child model, dispatched to
Modal, or trained locally via --local."""

import os
import shutil
import tarfile
import tempfile
import uuid
from dataclasses import asdict
from typing import Any

from distilabel.distiset import Distiset

from fastsft.helper import modelsets_dir
from fastsft.stages.base import Stage
from fastsft.stages.constants import FINE_TUNER
from fastsft.training.config import AdapterConfig, TrainingConfig, TrainingLoopConfig
from fastsft.training.constants import TOP_N_CONFIGS
from fastsft.training.heuristic import recommend_configs
from fastsft.training.modal_app import adapter_volume, app, train_lora


class FineTuner(Stage):
    """LoRA/QLoRA SFT on child model, locally or on Modal cloud GPUs."""

    name = FINE_TUNER
    title = "Fine-Tuning Stage"

    def __init__(
        self,
        child_model_id: str,
        training_config: TrainingConfig | None = None,
        local_training: bool = False,
        verbose: bool = True,
    ):
        super().__init__(verbose=verbose)
        self._child_model_id = child_model_id
        self._training_config = training_config
        self._local_training = local_training

    def _validate_input(self, formatted_distiset: Distiset) -> None:
        train = formatted_distiset["default"]["train"]
        if "text" not in train.column_names:
            raise ValueError(
                "FineTuner.run() requires a 'text' column in the input "
                "distiset, rendered via the child model's chat template "
                "(see DataFormatter)."
            )

    def _run(self, formatted_distiset: Distiset) -> str:
        chosen = self._resolve_training_config(formatted_distiset)
        train_ds, eval_ds = self._split_validation(
            formatted_distiset, chosen.loop.validation_split
        )
        self._log(
            f"[1/3] Split into {len(train_ds)} train / {len(eval_ds)} eval "
            f"({chosen.loop.validation_split:.0%} held out)."
        )

        if self._local_training:
            return self._run_local(chosen, train_ds, eval_ds)
        return self._run_modal(chosen, train_ds, eval_ds)

    def _run_modal(self, chosen: TrainingConfig, train_ds: Any, eval_ds: Any) -> str:
        self._log(f"[2/3] Dispatching training to Modal ({chosen.gpu_tier})...")
        job_id = uuid.uuid4().hex[:12]
        with app.run():
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

    def _run_local(self, chosen: TrainingConfig, train_ds: Any, eval_ds: Any) -> str:
        from fastsft.training.local_trainer import train_locally

        self._log(f"[2/3] Training locally ({chosen.gpu_tier})...")
        adapter_dir = train_locally(
            child_model_id=self._child_model_id,
            train_rows=train_ds.to_list(),
            eval_rows=eval_ds.to_list(),
            config=asdict(chosen),
        )
        self._log(f"[2/3] Done: adapter saved to '{adapter_dir}'.")
        return adapter_dir

    def _resolve_training_config(self, formatted_distiset: Distiset) -> TrainingConfig:
        """Resolve training config: caller-supplied, local defaults, or cost-ranked."""
        if self._training_config is not None:
            self._log(
                f"[1/3] Using caller-supplied config: "
                f"{self._training_config.gpu_tier} / "
                f"{self._training_config.strategy}."
            )
            return self._training_config

        if self._local_training:
            from fastsft.device import detect_device

            device = detect_device()
            self._log(f"[1/3] Training locally on detected device: {device}.")
            return TrainingConfig(
                gpu_tier=f"local ({device})",
                adapter=AdapterConfig(),
                loop=TrainingLoopConfig(),
            )

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
        """Copy adapter to modelsets_dir()/run_id."""
        destination = os.path.join(modelsets_dir(), run_id)
        shutil.copytree(output, destination, dirs_exist_ok=True)
        return destination

    def _split_validation(self, distiset: Distiset, validation_split: float) -> tuple[Any, Any]:
        """Hold out validation slice for early stopping."""
        train = distiset["default"]["train"]
        split = train.train_test_split(test_size=validation_split, seed=42)
        return split["train"], split["test"]

    def _download_adapter(self, tar_path: str) -> str:
        """Download and extract adapter tarball from Modal Volume."""
        fd, local_tar_path = tempfile.mkstemp(suffix=".tar.gz")
        with os.fdopen(fd, "wb") as f:
            for chunk in adapter_volume.read_file(tar_path):
                f.write(chunk)

        extract_dir = tempfile.mkdtemp(prefix="finetuner_adapter_")
        with tarfile.open(local_tar_path, "r:gz") as tar:
            # filter="data" blocks path-traversal/special-file entries and is
            # the default in Python 3.14; set explicitly to silence the 3.12
            # deprecation warning and lock the behavior in.
            tar.extractall(extract_dir, filter="data")
        return extract_dir
