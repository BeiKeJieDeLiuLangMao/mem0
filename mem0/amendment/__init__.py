"""
Amendment Proposal System - ECC-style memory optimization.

Proposes and manages memory improvements based on observed patterns.
"""

from .proposer import AmendmentProposer
from .models import AmendmentProposal, AmendmentStatus

__all__ = [
    "AmendmentProposer",
    "AmendmentProposal",
    "AmendmentStatus",
]
