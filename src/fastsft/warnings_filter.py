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
# distilabel's own Pydantic/shim deprecation noise -- library-internal, can't fix upstream.
warnings.filterwarnings("ignore", category=PydanticDeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"distilabel")

# distilabel's spawned subprocesses don't see the filters above; PYTHONWARNINGS can only name built-in categories at startup, so silence the two base categories there too.
_child_filters = ("ignore::UserWarning", "ignore::DeprecationWarning")
os.environ["PYTHONWARNINGS"] = ",".join(
    filter(None, [os.environ.get("PYTHONWARNINGS", ""), *_child_filters])
)

# Silences transformers' advisory warnings (e.g. "PyTorch was not found").
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Quiets distilabel's INFO chatter so our own "[1/4] ..." progress reads cleanly.
os.environ.setdefault("DISTILABEL_LOG_LEVEL", "WARNING")
# Silences `datasets`' map/save progress bars for the same clean look.
os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
