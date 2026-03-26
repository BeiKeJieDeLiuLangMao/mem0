"""
Data models for observations.

ECC-inspired observation data structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
import json


class ObservationType(Enum):
    """Types of observation events."""
    ADD_INITIATED = "ADD_INITIATED"
    ADD_COMPLETED = "ADD_COMPLETED"
    UPDATE_INITIATED = "UPDATE_INITIATED"
    UPDATE_COMPLETED = "UPDATE_COMPLETED"
    DELETE_INITIATED = "DELETE_INITIATED"
    DELETE_COMPLETED = "DELETE_COMPLETED"
    SEARCH_INITIATED = "SEARCH_INITIATED"
    SEARCH_COMPLETED = "SEARCH_COMPLETED"
    FEEDBACK = "FEEDBACK"


@dataclass
class Observation:
    """
    A single observation event.

    Similar to ECC's observation structure but adapted for memory operations.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    event_type: ObservationType = ObservationType.ADD_INITIATED
    project_id: str = "global"
    agent_id: str = "default"
    session_id: str = "default"
    user_id: str = "default"
    data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    redacted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "project_id": self.project_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "data": self.data,
            "metadata": self.metadata,
            "confidence": self.confidence,
            "redacted": self.redacted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        """Create from dictionary."""
        event_type = ObservationType(data["event_type"])
        timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            id=data["id"],
            timestamp=timestamp,
            event_type=event_type,
            project_id=data.get("project_id", "global"),
            agent_id=data.get("agent_id", "default"),
            session_id=data.get("session_id", "default"),
            user_id=data.get("user_id", "default"),
            data=data.get("data", {}),
            metadata=data.get("metadata", {}),
            confidence=data.get("confidence", 0.0),
            redacted=data.get("redacted", False),
        )

    def to_jsonl(self) -> str:
        """Convert to JSONL format."""
        return json.dumps(self.to_dict())


@dataclass
class ProjectInfo:
    """
    Project information for isolation.

    ECC-style project detection using Git remote URL hash.
    """
    project_id: str
    project_name: str
    git_remote_url: Optional[str] = None
    git_branch: Optional[str] = None
    detected_at: datetime = field(default_factory=lambda: datetime.utcnow())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "git_remote_url": self.git_remote_url,
            "git_branch": self.git_branch,
            "detected_at": self.detected_at.isoformat(),
        }
