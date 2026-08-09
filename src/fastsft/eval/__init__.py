"""Extrinsic, judge-based evaluation of a fine-tuned child adapter.

Standalone (not a DistillationPipeline stage): runs *after* training against a
saved adapter, loading the parent (via OpenRouter), the tuned child, and the
untuned child at once. Entry point: `python -m fastsft.eval.run [adapter_dir]`;
view a finished run with `python -m fastsft.eval.results_viewer`.
"""
