"""
Pattern Detection Layer - ECC Learning & Analysis

Analyzes observations to extract reusable patterns and skills.
"""

from .pattern_detector import PatternDetector
from .pattern import Pattern, PatternType, PatternEvidence
from .skill_extractor import SkillExtractor

__all__ = [
    "PatternDetector",
    "Pattern",
    "PatternType",
    "PatternEvidence",
    "SkillExtractor",
]
