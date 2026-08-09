"""Import guardrail: every module in the package must import cleanly, and the
pipeline must construct without doing any I/O.

Catches broken or mis-typed imports from mechanical refactors (e.g. the src/
move) the moment CI runs, instead of in the field down some un-exercised CLI
path. It needs only the main dependencies -- the modules defer torch / peft /
trl / sentence-transformers inside functions, so importing them here stays
light. If someone hoists one of those heavy imports to module top-level, this
test starts failing without the optional extras installed -- which is the
deferral discipline working as intended.
"""

import importlib
import pkgutil

import fastsft


def test_every_module_imports():
    failures = []
    for module in pkgutil.walk_packages(fastsft.__path__, prefix="fastsft."):
        try:
            importlib.import_module(module.name)
        except Exception as exc:  # noqa: BLE001 -- report every failure, don't stop at the first
            failures.append(f"{module.name}: {exc!r}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)


def test_pipeline_constructs_without_io():
    from fastsft.pipeline import DistillationPipeline

    # Starting at the last stage builds only FineTuner -- no tokenizer fetch, no
    # network -- so construction is a pure wiring check.
    pipeline = DistillationPipeline(start_stage="fine_tuner")
    assert pipeline.stages
