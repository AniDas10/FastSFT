import warnings_filter  # noqa: F401

import argparse
import os
from datetime import datetime

from constants import (
    DEFAULT_GUIDE_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARENT_MODEL,
    RUN_TIMESTAMP_FORMAT,
)
from pipeline import DistillationPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic dataset from a prompt using distilabel + OpenRouter."
    )
    parser.add_argument("prompt", help="Freeform description of the dataset you want (style, domain, tone).")
    parser.add_argument("--guide-model", default=DEFAULT_GUIDE_MODEL, help="OpenRouter model id for the guide.")
    parser.add_argument("--parent-model", default=DEFAULT_PARENT_MODEL, help="OpenRouter model id for generation.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="OpenRouter model id for judging.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to generate.")
    default_output = os.path.join(DEFAULT_OUTPUT_DIR, datetime.now().strftime(RUN_TIMESTAMP_FORMAT))
    parser.add_argument(
        "--output",
        default=default_output,
        help=f"Directory to save the dataset to (default: {DEFAULT_OUTPUT_DIR}/<timestamp>).",
    )
    args = parser.parse_args()

    if not args.prompt.strip():
        parser.error("prompt must not be empty.")

    pipeline = DistillationPipeline(
        guide_model=args.guide_model,
        parent_model=args.parent_model,
        judge_model=args.judge_model,
        num_samples=args.num_samples,
    )
    refined_distiset = pipeline.run(args.prompt)

    print(f"[4/4] Saving dataset to '{args.output}'...")
    refined_distiset.save_to_disk(args.output)
    print(f"[4/4] Done: saved {args.num_samples} samples to '{args.output}'")


if __name__ == "__main__":
    main()
