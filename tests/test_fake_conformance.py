"""Guardrail: the hand-written fakes in conftest must keep the same public
method signatures as the real collaborators they stand in for.

The suite deliberately prefers fakes over mocks -- they read clearly and enforce
the real interface -- but that only holds while the fake mirrors the real class.
A silent drift (the real `score_samples`/`generate_untuned`/... signature
changes, the fake doesn't) would let every test stay green against a contract
that no longer exists. This pins fake <-> real parity so drift fails loudly here
instead of hiding a stale fake.

Scope: only the PUBLIC methods a fake defines are checked (the surface the code
under test actually calls). Constructors are intentionally exempt -- a fake is
built from scripted answers, not real weights/credentials, so `__init__`
legitimately differs. Annotations are ignored (the fakes carry none by design);
parameter names, kinds, and default values are compared.
"""

import inspect


def _public_params(func):
    """(name, kind, default) per parameter of `func`, minus `self` and minus
    type annotations -- the part of the signature a caller must satisfy."""
    return [
        (p.name, p.kind, p.default)
        for p in inspect.signature(func).parameters.values()
        if p.name != "self"
    ]


def _assert_conforms(fake_cls, real_cls):
    mismatches = []
    for name, fake_method in inspect.getmembers(fake_cls, predicate=inspect.isfunction):
        if name.startswith("_"):  # skip dunders (incl. __init__) and privates
            continue
        real_method = getattr(real_cls, name, None)
        if not callable(real_method):
            mismatches.append(f"{name}: not a method on {real_cls.__name__}")
            continue
        fake_params, real_params = _public_params(fake_method), _public_params(real_method)
        if fake_params != real_params:
            mismatches.append(f"{name}: fake {fake_params} != real {real_params}")
    assert not mismatches, (
        f"{fake_cls.__name__} has drifted from {real_cls.__name__}:\n  "
        + "\n  ".join(mismatches)
    )


def test_fake_judge_matches_real_judge(fake_judge):
    from fastsft.model.judge import Judge

    _assert_conforms(fake_judge, Judge)


def test_fake_inference_engine_matches_real_engine(fake_inference_engine):
    from fastsft.eval.inference import ChildInferenceEngine

    _assert_conforms(fake_inference_engine, ChildInferenceEngine)
