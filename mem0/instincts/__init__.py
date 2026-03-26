"""
Instincts System - ECC auto-applied high-confidence patterns.

Automatically applies high-confidence patterns without user intervention.
"""

from .instincts import Instinct, InstinctRegistry
from .auto_applier import InstinctApplier
from .decay import InstinctDecayEngine

__all__ = [
    "Instinct",
    "InstinctRegistry",
    "InstinctApplier",
    "InstinctDecayEngine",
]
