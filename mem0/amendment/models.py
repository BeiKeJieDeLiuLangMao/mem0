"""
Models for amendment proposals.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class AmendmentStatus(Enum):
    """Status of an amendment proposal."""
    PENDING = "pending"
    APPROVED = "approved"
    APPLIED = "applied"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class AmendmentType(Enum):
    """Types of amendments."""
    CONTENT_UPDATE = "content_update"  # Update memory content
    MERGE = "merge"  # Merge similar memories
    SPLIT = "split"  # Split complex memories
    DEPRECATE = "deprecate"  # Mark as outdated
    REPRIORITIZE = "reprioritize"  # Change importance/score
    CLARIFY = "clarify"  # Add clarifying context


@dataclass
class AmendmentProposal:
    """
    A proposal to amend a memory.

    ECC-style proposal with:
    - Evidence backing
    - Impact assessment
    - AI-generated reasoning
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    amendment_type: AmendmentType = AmendmentType.CONTENT_UPDATE

    # Target
    memory_id: str = ""
    current_content: Dict[str, Any] = field(default_factory=dict)

    # Proposed change
    proposed_change: Dict[str, Any] = field(default_factory=dict)

    # Evidence and reasoning
    reasoning: str = ""
    evidence_observation_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0

    # Impact assessment
    expected_impact: float = 0.0  # -1 to 1
    impact_description: str = ""

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    created_by: str = "system"  # system, user_id, or pattern_id
    project_id: str = "global"
    agent_id: str = "default"
    status: AmendmentStatus = AmendmentStatus.PENDING

    # Application tracking
    applied_at: Optional[datetime] = None
    applied_version_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "amendment_type": self.amendment_type.value,
            "memory_id": self.memory_id,
            "current_content": self.current_content,
            "proposed_change": self.proposed_change,
            "reasoning": self.reasoning,
            "evidence_observation_ids": self.evidence_observation_ids,
            "confidence": self.confidence,
            "expected_impact": self.expected_impact,
            "impact_description": self.impact_description,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "project_id": self.project_id,
            "agent_id": self.agent_id,
            "status": self.status.value,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "applied_version_id": self.applied_version_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AmendmentProposal":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            amendment_type=AmendmentType(data["amendment_type"]),
            memory_id=data["memory_id"],
            current_content=data["current_content"],
            proposed_change=data["proposed_change"],
            reasoning=data["reasoning"],
            evidence_observation_ids=data["evidence_observation_ids"],
            confidence=data["confidence"],
            expected_impact=data["expected_impact"],
            impact_description=data["impact_description"],
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            project_id=data["project_id"],
            agent_id=data.get("agent_id", "default"),
            status=AmendmentStatus(data["status"]),
            applied_at=datetime.fromisoformat(data["applied_at"]) if data.get("applied_at") else None,
            applied_version_id=data.get("applied_version_id"),
        )


@dataclass
class AmendmentBatch:
    """A batch of related amendments."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    proposals: List[AmendmentProposal] = field(default_factory=list)
    batch_reasoning: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    project_id: str = "global"
    agent_id: str = "default"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "proposals": [p.to_dict() for p in self.proposals],
            "batch_reasoning": self.batch_reasoning,
            "created_at": self.created_at.isoformat(),
            "project_id": self.project_id,
            "agent_id": self.agent_id,
        }
