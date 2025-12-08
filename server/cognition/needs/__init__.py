"""
Murph - Needs System Module
Sims-style needs that drive autonomous behavior.
"""

from .need import Need
from .personality import Personality, PRESETS
from .needs_system import NeedsSystem, DEFAULT_NEEDS

__all__ = [
    "Need",
    "Personality",
    "NeedsSystem",
    "PRESETS",
    "DEFAULT_NEEDS",
]
