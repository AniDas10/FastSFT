import warnings_filter  # noqa: F401

import argparse

from constants import (
    DEFAULT_CHILD_MODEL_ID,
    DEFAULT_GUIDE_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PARENT_MODEL,
)
from data.config import DataGenerationConfig, ParentGenerationConfig
from data.constants import BREADTH_EXPONENT
from helper import (
    current_timestamp,
    load_data,
    validate_start_stage,
    validate_training_flags,
)
from model.constants import DEFAULT_MAX_TOKENS, DEFAULT_SCORE_THRESHOLD
from pipeline import DistillationPipeline
from stages.constants import STAGE_NAMES, STAGE_ORDER
from training.config import AdapterConfig, TrainingConfig, TrainingLoopConfig
from training.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_GRAD_ACCUMULATION,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_MODAL_TIMEOUT_SECONDS,
    DEFAULT_STRATEGY,
    DEFAULT_VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE,
    EVAL_STEPS,
    LORA,
    LORA_TARGET_MODULES,
    MAX_EPOCHS,
    MODAL_GPU_TIERS,
    QLORA,
)


def _input_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Registers CLI arguments on `parser` and returns the parsed args."""
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Freeform description of the dataset you want (style, domain, tone). "
        f"Only used when --start-stage={STAGE_ORDER[0]} (the default).",
    )
    parser.add_argument(
        "--guide-model",
        default=DEFAULT_GUIDE_MODEL,
        help=f"OpenRouter model id for the guide. Only used when "
        f"--start-stage={STAGE_ORDER[0]} (the default).",
    )
    parser.add_argument(
        "--parent-model",
        default=DEFAULT_PARENT_MODEL,
        help=f"OpenRouter model id for generation. Only used when "
        f"--start-stage={STAGE_ORDER[0]} (the default).",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"OpenRouter model id for judging. Only used when "
        f"--start-stage={STAGE_ORDER[0]} (the default).",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help=f"Number of samples to generate. Only used when "
        f"--start-stage={STAGE_ORDER[0]} (the default).",
    )
    parser.add_argument(
        "--breadth-exponent",
        type=float,
        default=BREADTH_EXPONENT,
        help="Breadth (distinct seed topics) = ceil(num_samples ** exponent); "
        "higher favors more topics over depth per topic. Only used when "
        f"--start-stage={STAGE_ORDER[0]} (the default).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=DEFAULT_SCORE_THRESHOLD,
        help="Judge score (0-10) below which a sample is regenerated. Only "
        f"used when --start-stage={STAGE_ORDER[0]} (the default).",
    )
    parser.add_argument(
        "--parent-temperature",
        type=float,
        default=0.9,
        help="Sampling temperature for the parent model's generations. Only "
        f"used when --start-stage={STAGE_ORDER[0]} (the default).",
    )
    parser.add_argument(
        "--parent-max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max tokens per parent model generation. Only used when "
        f"--start-stage={STAGE_ORDER[0]} (the default).",
    )
    parser.add_argument(
        "--child-model-id",
        default=DEFAULT_CHILD_MODEL_ID,
        help="Hugging Face repo id of the target/child model to format for and fine-tune.",
    )
    parser.add_argument(
        "--start-stage",
        choices=STAGE_NAMES,
        default=STAGE_ORDER[0],
        help="Which stage to start the pipeline at -- every stage before it is "
        "skipped and runs from --input-path's dataset instead. Requires "
        "--input-path for anything other than the default.",
    )
    parser.add_argument(
        "--input-path",
        default=None,
        help="Path to a saved Distiset to use as input for --start-stage "
        f"(required unless --start-stage={STAGE_ORDER[0]}, which uses the "
        "prompt argument instead).",
    )
    parser.add_argument(
        "--gpu-tier",
        choices=list(MODAL_GPU_TIERS),
        default=None,
        help="Override FineTuner's training GPU tier, skipping the cost "
        "heuristic entirely. The other training flags below fill in around "
        "this (defaulting if not given). Omit to let the heuristic pick the "
        "cheapest feasible tier automatically.",
    )
    parser.add_argument(
        "--strategy",
        choices=[LORA, QLORA],
        default=None,
        help=f"Training strategy, only used with --gpu-tier (default: {DEFAULT_STRATEGY}).",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help=f"LoRA rank, only used with --gpu-tier (default: {DEFAULT_LORA_RANK}).",
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="LoRA target module names, only used with --gpu-tier "
        f"(default: {' '.join(LORA_TARGET_MODULES)}).",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        help=f"LoRA dropout, only used with --gpu-tier (default: {DEFAULT_LORA_DROPOUT}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Per-device training batch size, only used with --gpu-tier "
        f"(default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--grad-accumulation",
        type=int,
        default=None,
        help=f"Gradient accumulation steps, only used with --gpu-tier "
        f"(default: {DEFAULT_GRAD_ACCUMULATION}).",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=f"Learning rate, only used with --gpu-tier (default: {DEFAULT_LEARNING_RATE}).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Upper bound on training epochs (early stopping decides the "
        f"actual count), only used with --gpu-tier (default: {MAX_EPOCHS}).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=None,
        help=f"Steps between validation evaluations, only used with --gpu-tier "
        f"(default: {EVAL_STEPS}).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Non-improving evaluations tolerated before stopping, only used "
        f"with --gpu-tier (default: {EARLY_STOPPING_PATIENCE}).",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=None,
        help="Fraction of the formatted dataset held out for validation, "
        f"only used with --gpu-tier (default: {DEFAULT_VALIDATION_SPLIT}).",
    )
    parser.add_argument(
        "--modal-timeout",
        type=int,
        default=None,
        help="Modal job timeout in seconds, only used with --gpu-tier "
        f"(default: {DEFAULT_MODAL_TIMEOUT_SECONDS}).",
    )
    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser(
        description="Run the distillation pipeline: " + " -> ".join(STAGE_ORDER) + "."
    )
    args = _input_args(parser)
    validate_start_stage(args, parser)
    validate_training_flags(args, parser)

    pipeline_input = (
        args.prompt if args.start_stage == STAGE_ORDER[0] else load_data(args.input_path)
    )

    # Only relevant when starting from DataGenerator -- omitted otherwise so
    # DistillationPipeline's own default applies (mirrors how `training`
    # below is only built when the caller actually wants to override it).
    generation = (
        DataGenerationConfig(
            guide_model=args.guide_model,
            parent_model=args.parent_model,
            judge_model=args.judge_model,
            num_samples=args.num_samples,
            breadth_exponent=args.breadth_exponent,
            score_threshold=args.score_threshold,
            parent_generation=ParentGenerationConfig(
                temperature=args.parent_temperature,
                max_tokens=args.parent_max_tokens,
            ),
        )
        if args.start_stage == STAGE_ORDER[0]
        else None
    )

    # None (the default) lets FineTuner's cost heuristic pick the cheapest
    # feasible config; --gpu-tier opts into a fully explicit one instead.
    # `is not None` (not `or`) throughout: 0/0.0 are meaningful values for
    # several of these (e.g. --lora-dropout 0), not "unset".
    training = (
        TrainingConfig(
            gpu_tier=args.gpu_tier,
            strategy=args.strategy if args.strategy is not None else DEFAULT_STRATEGY,
            adapter=AdapterConfig(
                rank=args.lora_rank if args.lora_rank is not None else DEFAULT_LORA_RANK,
                target_modules=(
                    args.target_modules
                    if args.target_modules is not None
                    else list(LORA_TARGET_MODULES)
                ),
                dropout=(
                    args.lora_dropout
                    if args.lora_dropout is not None
                    else DEFAULT_LORA_DROPOUT
                ),
            ),
            loop=TrainingLoopConfig(
                batch_size=(
                    args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZE
                ),
                grad_accumulation=(
                    args.grad_accumulation
                    if args.grad_accumulation is not None
                    else DEFAULT_GRAD_ACCUMULATION
                ),
                learning_rate=(
                    args.learning_rate
                    if args.learning_rate is not None
                    else DEFAULT_LEARNING_RATE
                ),
                max_epochs=args.max_epochs if args.max_epochs is not None else MAX_EPOCHS,
                eval_steps=args.eval_steps if args.eval_steps is not None else EVAL_STEPS,
                early_stopping_patience=(
                    args.early_stopping_patience
                    if args.early_stopping_patience is not None
                    else EARLY_STOPPING_PATIENCE
                ),
                validation_split=(
                    args.validation_split
                    if args.validation_split is not None
                    else DEFAULT_VALIDATION_SPLIT
                ),
            ),
            modal_timeout_seconds=(
                args.modal_timeout
                if args.modal_timeout is not None
                else DEFAULT_MODAL_TIMEOUT_SECONDS
            ),
        )
        if args.gpu_tier is not None
        else None
    )

    pipeline = DistillationPipeline(
        child_model_id=args.child_model_id,
        generation=generation,
        training=training,
        start_stage=args.start_stage,
    )
    # One run_id for the whole run, so each stage's output folder shares it.
    # Each stage is saved the moment it completes, so a later stage's failure
    # (e.g. FineTuner's NotImplementedError) can't lose an earlier one's output.
    run_id = current_timestamp()
    try:
        for stage, output in pipeline.run(pipeline_input):
            path = stage.save_output(output, run_id)
            if path:
                print(f"Saved {stage.name} output to '{path}'")
    except NotImplementedError as e:
        # FineTuner unimplemented; earlier stages are already saved.
        print(f"Note: {e}")


if __name__ == "__main__":
    main()
