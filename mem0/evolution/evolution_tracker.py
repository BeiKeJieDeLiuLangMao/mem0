"""
Evolution tracking for memory amendments.

Tracks how memories change over time with version history.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from ..observation.models import Observation


@dataclass
class AmendmentProposal:
    """
    A proposed amendment to a memory.

    ECC-style amendment with:
    - Evidence backing
    - Confidence scoring
    - Expected impact
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_id: str = ""
    proposed_change: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    evidence: List[str] = field(default_factory=list)

    # Scoring
    confidence: float = 0.0
    expected_impact: float = 0.0  # -1 to 1

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    project_id: str = "global"
    agent_id: str = "default"
    status: str = "pending"  # pending, applied, rejected

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "proposed_change": self.proposed_change,
            "reasoning": self.reasoning,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "expected_impact": self.expected_impact,
            "created_at": self.created_at.isoformat(),
            "project_id": self.project_id,
            "agent_id": self.agent_id,
            "status": self.status,
        }


@dataclass
class MemoryVersion:
    """A version of a memory."""

    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_id: str = ""
    content: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    created_by: str = "system"  # system, user, or amendment_id

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "memory_id": self.memory_id,
            "content": self.content,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
        }


class EvolutionTracker:
    """
    Tracks memory evolution and amendments.

    ECC-style evolution with:
    - Version history
    - Amendment proposals
    - Rollback capability
    """

    def __init__(self):
        """Initialize evolution tracker."""
        self._versions: List[MemoryVersion] = []
        self._proposals: List[AmendmentProposal] = []

    async def record_version(
        self,
        memory_id: str,
        content: Dict[str, Any],
        created_by: str = "system",
    ) -> MemoryVersion:
        """
        Record a new version of a memory.

        Args:
            memory_id: ID of the memory
            content: Memory content at this version
            created_by: What created this version

        Returns:
            The created MemoryVersion
        """
        version = MemoryVersion(
            memory_id=memory_id,
            content=content,
            created_by=created_by,
        )

        self._versions.append(version)
        return version

    async def propose_amendment(
        self,
        memory_id: str,
        proposed_change: Dict[str, Any],
        reasoning: str,
        evidence: List[str],
        confidence: float,
        expected_impact: float,
        project_id: str = "global",
        agent_id: str = "default",
    ) -> AmendmentProposal:
        """
        Propose an amendment to a memory.

        Args:
            memory_id: ID of the memory to amend
            proposed_change: The change to apply
            reasoning: Why this change is needed
            evidence: Observation IDs supporting the change
            confidence: Confidence in the amendment (0-1)
            expected_impact: Expected impact (-1 to 1)
            project_id: Project identifier
            agent_id: Agent identifier

        Returns:
            The created AmendmentProposal
        """
        proposal = AmendmentProposal(
            memory_id=memory_id,
            proposed_change=proposed_change,
            reasoning=reasoning,
            evidence=evidence,
            confidence=confidence,
            expected_impact=expected_impact,
            project_id=project_id,
            agent_id=agent_id,
        )

        self._proposals.append(proposal)
        return proposal

    async def apply_amendment(
        self,
        proposal_id: str,
        current_content: Dict[str, Any],
    ) -> Optional[MemoryVersion]:
        """
        Apply an approved amendment proposal.

        Args:
            proposal_id: ID of the proposal to apply
            current_content: Current memory content

        Returns:
            New MemoryVersion if applied, None if proposal not found
        """
        proposal = self._get_proposal(proposal_id)
        if not proposal or proposal.status != "pending":
            return None

        # Apply change
        new_content = {**current_content, **proposal.proposed_change}

        # Record version
        version = await self.record_version(
            memory_id=proposal.memory_id,
            content=new_content,
            created_by=f"amendment:{proposal.id}",
        )

        # Update proposal status
        proposal.status = "applied"

        return version

    async def rollback(
        self,
        memory_id: str,
        version_id: str,
    ) -> Optional[MemoryVersion]:
        """
        Rollback a memory to a previous version.

        Args:
            memory_id: ID of the memory
            version_id: Target version ID

        Returns:
            New MemoryVersion representing the rollback, or None
        """
        # Find target version
        target = None
        for v in self._versions:
            if v.version_id == version_id and v.memory_id == memory_id:
                target = v
                break

        if not target:
            return None

        # Create rollback version
        return await self.record_version(
            memory_id=memory_id,
            content=target.content,
            created_by=f"rollback:{version_id}",
        )

    def get_version_history(self, memory_id: str) -> List[MemoryVersion]:
        """Get all versions of a memory."""
        return [
            v for v in self._versions
            if v.memory_id == memory_id
        ]

    def get_pending_proposals(
        self,
        project_id: Optional[str] = None,
        agent_id: Optional[str] = None,
    ) -> List[AmendmentProposal]:
        """Get pending amendment proposals."""
        proposals = [
            p for p in self._proposals
            if p.status == "pending"
        ]

        if project_id:
            proposals = [p for p in proposals if p.project_id == project_id]

        if agent_id:
            proposals = [p for p in proposals if p.agent_id == agent_id]

        # Sort by confidence (high confidence first)
        proposals.sort(key=lambda p: p.confidence, reverse=True)

        return proposals

    def _get_proposal(self, proposal_id: str) -> Optional[AmendmentProposal]:
        """Get proposal by ID."""
        for p in self._proposals:
            if p.id == proposal_id:
                return p
        return None
