"""Fixtures local to the model-layer Tier-2 tests.

The model layer talks to OpenRouter via distilabel; these tests patch that edge
and never hit the network. `fake_distiset` builds the minimal
`["default"]["train"]` shape the roles iterate over, so run_pipeline can be
stubbed to return canned generations.
"""

import pytest


@pytest.fixture
def fake_distiset():
    """Factory: rows -> object supporting `d["default"]["train"]` iteration.

    Each row is a plain dict (e.g. {"generation": ..., "id": ...}); the returned
    value mimics just the slice the model roles read from a real Distiset.
    """

    def _make(rows):
        return {"default": {"train": list(rows)}}

    return _make
