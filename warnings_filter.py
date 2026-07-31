"""Suppresses import-time warnings from distilabel and transformers.
Must be imported before either (directly or transitively) is.
"""

import os
import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)

# Silences transformers' advisory warnings (e.g. "PyTorch was not found").
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
