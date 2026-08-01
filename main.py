import warnings_filter  # noqa: F401

import argparse

from constants import (
    DEFAULT_CHILD_MODEL_ID,
    DEFAULT_GUIDE_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PARENT_MODEL,
)
from helper import current_timestamp, load_data, validate_start_stage
from pipeline import DistillationPipeline
from stages.constants import STAGE_NAMES, STAGE_ORDER


def _input_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Registers CLI arguments on `parser` and returns the parsed args."""
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Freeform description of the dataset you want (style, domain, tone). "
        f"Only used when --start-stage={STAGE_ORDER[0]} (the default).",
    )
    parser.add_argument("--guide-model", default=DEFAULT_GUIDE_MODEL, help="OpenRouter model id for the guide.")
    parser.add_argument("--parent-model", default=DEFAULT_PARENT_MODEL, help="OpenRouter model id for generation.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="OpenRouter model id for judging.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate.")
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
    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser(
        description="Run the distillation pipeline: " + " -> ".join(STAGE_ORDER) + "."
    )
    args = _input_args(parser)
    validate_start_stage(args, parser)

    pipeline_input = (
        args.prompt if args.start_stage == STAGE_ORDER[0] else load_data(args.input_path)
    )

    pipeline = DistillationPipeline(
        child_model_id=args.child_model_id,
        guide_model=args.guide_model,
        parent_model=args.parent_model,
        judge_model=args.judge_model,
        num_samples=args.num_samples,
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
