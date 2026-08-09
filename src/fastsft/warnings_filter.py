"""Suppresses import-time warnings from distilabel and transformers.
Must be imported before either (directly or transitively) is.
"""

import os
import warnings

from pydantic.warnings import (
    PydanticDeprecationWarning,
    UnsupportedFieldAttributeWarning,
)

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
# distilabel reads Pydantic's `model_fields` off instances (deprecated in 2.11,
# raised as PydanticDeprecatedSince211 -- a PydanticDeprecationWarning subclass)
# and imports the deprecated `distilabel.llms` shim. Both are library-internal
# noise we can't fix upstream; filter them so they don't bury real warnings.
warnings.filterwarnings("ignore", category=PydanticDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"distilabel")

# distilabel runs each pipeline step in its own SPAWNED subprocess (a fresh
# interpreter) that never sees the in-process filters above, so the workers
# re-emit Pydantic's schema-build warnings. PYTHONWARNINGS is inherited via the
# environment and reaches those children -- but at interpreter startup it can
# only name BUILT-IN categories (it can't import Pydantic's warning classes that
# early: `Invalid -W option ... invalid module name`), and the messages contain
# commas (the PYTHONWARNINGS entry separator), so we can scope neither by class
# nor by message. We therefore ignore the two built-in BASE categories in the
# children: UserWarning (UnsupportedFieldAttributeWarning's base) and
# DeprecationWarning (the model_fields deprecation's base). These subprocesses
# run only distilabel/pydantic internals -- not our code -- and the main process
# keeps the precise class filters above, so this masks nothing of ours.
_child_filters = ("ignore::UserWarning", "ignore::DeprecationWarning")
os.environ["PYTHONWARNINGS"] = ",".join(
    filter(None, [os.environ.get("PYTHONWARNINGS", ""), *_child_filters])
)

# Silences transformers' advisory warnings (e.g. "PyTorch was not found").
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Quiet distilabel's own INFO chatter (the per-stage "loading steps" lines) so
# our stages' "[1/4] ..." progress reads cleanly; WARNING still lets real
# problems through. Read dynamically by distilabel and inherited by its worker
# subprocesses. setdefault so an explicit DISTILABEL_LOG_LEVEL=DEBUG still wins.
os.environ.setdefault("DISTILABEL_LOG_LEVEL", "WARNING")
# Silence the `datasets` map/save progress bars (the "Map: 100%" / "Saving the
# dataset" lines) for the same clean look; inherited by subprocesses too.
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
