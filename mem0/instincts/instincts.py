"""
Instincts - auto-applied high-confidence patterns.

ECC-style instincts: patterns so reliable they're applied automatically.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Callable, Optional
import uuid


class InstinctType(Enum):
    """Types of instincts."""
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    ERROR_PREVENTION = "error_prevention"
    PREFERENCE_APPLICATION = "preference_application"
    RESOURCE_OPTIMIZATION = "resource_optimization"


@dataclass
class InstinctTrigger:
    """Condition that triggers an instinct."""
    trigger_type: str  # "method", "pattern", "context"
    condition: Dict[str, Any] = field(default_factory=dict)

    def matches(self, context: Dict[str, Any]) -> bool:
        """Check if trigger matches context."""
        if self.trigger_type == "method":
            return context.get("method") == self.condition.get("method")
        elif self.trigger_type == "pattern":
            pattern = self.condition.get("pattern", "")
            return pattern in context.get("prompt", "")
        return False


@dataclass
class InstinctAction:
    """Action to take when instinct is triggered."""
    action_type: str  # "modify_args", "prepend", "append", "replace"
    content: Any

    def apply(self, original: Any) -> Any:
        """Apply action to original value."""
        if self.action_type == "modify_args":
            if isinstance(original, dict) and isinstance(self.content, dict):
                return {**original, **self.content}
            return original
        elif self.action_type == "prepend":
            if isinstance(original, str) and isinstance(self.content, str):
                return self.content + original
            return original
        elif self.action_type == "append":
            if isinstance(original, str) and isinstance(self.content, str):
                return original + self.content
            return original
        elif self.action_type == "replace":
            return self.content
        return original


@dataclass
class Instinct:
    """
    An auto-applied high-confidence pattern.

    ECC-style instincts:
    - Confidence threshold: >= 0.95 (11+ observations)
    - Auto-apply on trigger match
    - Track effectiveness
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    instinct_type: InstinctType = InstinctType.WORKFLOW_OPTIMIZATION

    # Trigger and action
    trigger: InstinctAction = field(default_factory=lambda: InstinctAction("modify_args", {}))
    action: InstinctAction = field(default_factory=lambda: InstinctAction("modify_args", {}))

    # Evidence and confidence
    source_pattern_id: str = ""
    confidence: float = 0.0
    observation_count: int = 0

    # Effectiveness tracking
    times_applied: int = 0
    times_successful: int = 0
    success_rate: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.utcnow())
    project_id: str = "global"
    user_id: str = "default"
    agent_id: str = "default"
    enabled: bool = True

    # Decay fields
    last_confirmed_at: Optional[datetime] = None
    last_contradicted_at: Optional[datetime] = None
    decay_rate: float = 0.02  # weekly decay

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "instinct_type": self.instinct_type.value,
            "trigger": {
                "action_type": self.trigger.action_type,
                "content": self.trigger.content,
            },
            "action": {
                "action_type": self.action.action_type,
                "content": self.action.content,
            },
            "source_pattern_id": self.source_pattern_id,
            "confidence": self.confidence,
            "observation_count": self.observation_count,
            "times_applied": self.times_applied,
            "times_successful": self.times_successful,
            "success_rate": self.success_rate,
            "created_at": self.created_at.isoformat(),
            "project_id": self.project_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "enabled": self.enabled,
            "last_confirmed_at": self.last_confirmed_at.isoformat() if self.last_confirmed_at else None,
            "last_contradicted_at": self.last_contradicted_at.isoformat() if self.last_contradicted_at else None,
            "decay_rate": self.decay_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Instinct":
        """Create from dictionary."""
        trigger = InstinctAction(
            data["trigger"]["action_type"],
            data["trigger"]["content"],
        )
        action = InstinctAction(
            data["action"]["action_type"],
            data["action"]["content"],
        )

        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            instinct_type=InstinctType(data["instinct_type"]),
            trigger=trigger,
            action=action,
            source_pattern_id=data["source_pattern_id"],
            confidence=data["confidence"],
            observation_count=data["observation_count"],
            times_applied=data["times_applied"],
            times_successful=data["times_successful"],
            success_rate=data["success_rate"],
            created_at=datetime.fromisoformat(data["created_at"]),
            project_id=data["project_id"],
            user_id=data["user_id"],
            agent_id=data.get("agent_id", "default"),
            enabled=data["enabled"],
            last_confirmed_at=datetime.fromisoformat(data["last_confirmed_at"]) if data.get("last_confirmed_at") else None,
            last_contradicted_at=datetime.fromisoformat(data["last_contradicted_at"]) if data.get("last_contradicted_at") else None,
            decay_rate=data.get("decay_rate", 0.02),
        )


class InstinctRegistry:
    """
    Registry for managing instincts.

    ECC-style registry with:
    - Project-specific instincts
    - Global instincts
    - Effectiveness tracking
    """

    def __init__(self, confidence_threshold: float = 0.95):
        """
        Initialize registry.

        Args:
            confidence_threshold: Minimum confidence for instinct creation
        """
        self.confidence_threshold = confidence_threshold
        self._instincts: List[Instinct] = []

    async def register_instinct(self, instinct: Instinct) -> bool:
        """
        Register a new instinct.

        Args:
            instinct: Instinct to register

        Returns:
            True if registered (meets threshold), False otherwise
        """
        if instinct.confidence < self.confidence_threshold:
            return False

        # Check for duplicates
        for existing in self._instincts:
            if (
                existing.project_id == instinct.project_id and
                existing.name == instinct.name
            ):
                # Update existing
                existing.confidence = instinct.confidence
                existing.observation_count = instinct.observation_count
                return True

        self._instincts.append(instinct)
        return True

    def get_instincts(
        self,
        project_id: str,
        enabled_only: bool = True,
    ) -> List[Instinct]:
        """
        Get instincts for a project.

        Args:
            project_id: Project identifier
            enabled_only: Only return enabled instincts

        Returns:
            List of applicable instincts
        """
        instincts = [i for i in self._instincts if i.project_id == project_id]

        if enabled_only:
            instincts = [i for i in instincts if i.enabled]

        return instincts

    def get_global_instincts(self, enabled_only: bool = True) -> List[Instinct]:
        """Get global instincts."""
        instincts = [i for i in self._instincts if i.project_id == "global"]

        if enabled_only:
            instincts = [i for i in instincts if i.enabled]

        return instincts

    def get_instincts_by_agent_id(
        self,
        agent_id: str,
        enabled_only: bool = True,
    ) -> List[Instinct]:
        """
        Get instincts for a specific agent.

        Args:
            agent_id: Agent identifier
            enabled_only: Only return enabled instincts

        Returns:
            List of applicable instincts
        """
        instincts = [i for i in self._instincts if i.agent_id == agent_id]

        if enabled_only:
            instincts = [i for i in instincts if i.enabled]

        return instincts

    async def record_application(
        self,
        instinct_id: str,
        successful: bool,
    ) -> None:
        """
        Record instinct application for effectiveness tracking.

        Args:
            instinct_id: ID of the instinct
            successful: Whether the application was successful
        """
        for instinct in self._instincts:
            if instinct.id == instinct_id:
                instinct.times_applied += 1
                if successful:
                    instinct.times_successful += 1

                # Update success rate
                if instinct.times_applied > 0:
                    instinct.success_rate = (
                        instinct.times_successful / instinct.times_applied
                    )
                break

    async def disable_underperforming(self, min_success_rate: float = 0.7) -> int:
        """
        Disable instincts with low success rates.

        Args:
            min_success_rate: Minimum success rate threshold

        Returns:
            Number of instincts disabled
        """
        disabled = 0

        for instinct in self._instincts:
            if (
                instinct.enabled and
                instinct.times_applied >= 5 and
                instinct.success_rate < min_success_rate
            ):
                instinct.enabled = False
                disabled += 1

        return disabled
