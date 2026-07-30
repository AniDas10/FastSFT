import warnings_filter  # noqa: F401

import argparse

from constants import (
    DEFAULT_CHILD_MODEL_ID,
    DEFAULT_GUIDE_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_PARENT_MODEL,
    FORMATTED_OUTPUT_SUBDIR,
    RAW_OUTPUT_SUBDIR,
)
from helper import load_data, save_data, validate_skip_flags
from pipeline import DistillationPipeline


def _input_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
    """Registers all CLI arguments on `parser` and returns the parsed args."""
    parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Freeform description of the dataset you want (style, domain, tone). "
        "Not needed if --skip-generation or --skip-formatting is set.",
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
        "--skip-generation",
        action="store_true",
        help="Skip DataGenerator; supply your own raw dataset via --raw-dataset-path.",
    )
    parser.add_argument(
        "--raw-dataset-path",
        default=None,
        help="Path to a saved raw Distiset (required with --skip-generation).",
    )
    parser.add_argument(
        "--skip-formatting",
        action="store_true",
        help="Skip DataGenerator and DataFormatter; supply your own dataset already "
        "formatted for --child-model-id via --formatted-dataset-path.",
    )
    parser.add_argument(
        "--formatted-dataset-path",
        default=None,
        help="Path to a saved formatted Distiset (required with --skip-formatting).",
    )
    return parser.parse_args()


def main():
    parser = argparse.ArgumentParser(
        description="Run the distillation pipeline: DataGenerator -> DataFormatter -> FineTuner."
    )
    args = _input_args(parser)
    validate_skip_flags(args, parser)

    raw_dataset = load_data(args.raw_dataset_path)
    formatted_dataset = load_data(args.formatted_dataset_path)

    pipeline = DistillationPipeline(
        child_model_id=args.child_model_id,
        guide_model=args.guide_model,
        parent_model=args.parent_model,
        judge_model=args.judge_model,
        num_samples=args.num_samples,
        skip_generation=args.skip_generation,
        skip_formatting=args.skip_formatting,
    )
    try:
        pipeline.run(
            prompt=args.prompt,
            raw_dataset=raw_dataset,
            formatted_dataset=formatted_dataset,
        )
    except NotImplementedError as e:
        # FineTuner is a scaffold for now -- still save whatever DataGenerator/
        # DataFormatter produced before it raised.
        print(f"Note: {e}")

    save_data(pipeline.raw_dataset, RAW_OUTPUT_SUBDIR, "raw")
    save_data(pipeline.formatted_dataset, FORMATTED_OUTPUT_SUBDIR, "formatted")


if __name__ == "__main__":
    main()
