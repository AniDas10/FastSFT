"""Suppresses a noisy pydantic warning distilabel triggers at import time.

Import this module first, before anything that (directly or transitively)
imports distilabel -- the filter only works if it runs before distilabel's
own pydantic model definitions are evaluated.
"""

import warnings

from pydantic.warnings import UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
