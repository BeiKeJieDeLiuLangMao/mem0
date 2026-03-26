"""
Pattern models for learning layer.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class PatternType(Enum):
    """Types of learnable patterns."""

    # Workflow patterns
    WORKFLOW_SEQUENCE = "workflow_sequence"
    WORKFLOW_CONDITIONAL = "workflow_conditional"
    WORKFLOW_ITERATIVE = "workflow_iterative"

    # Code patterns
    CODE_STRUCTURE = "code_structure"
    CODE_IDIOM = "code_idiom"
    CODE_REFACTORING = "code_refactoring"

    # User preference patterns
    USER_PREFERENCE = "user_preference"
    USER_BEHAVIOR = "user_behavior"

    # Error handling patterns
    ERROR_RECOVERY = "error_recovery"
    ERROR_PREVENTION = "error_prevention"

    # Decision patterns
    DECISION_CRITERIA = "decision_criteria"
    DECISION_TRADEOFF = "decision_tradeoff"


@dataclass
class PatternEvidence:
    """
    Evidence supporting a pattern.

    Links pattern to specific observations that demonstrate it.
    """

    observation_ids: List[str] = field(default_factory=list)
    example_count: int = 0
    confidence_distribution: List[float] = field(default_factory=list)
    context_snippets: List[Dict[str, Any]] = field(default_factory=list)

    def get_average_confidence(self) -> float:
        """Calculate average confidence from evidence."""
        if not self.confidence_distribution:
            return 0.0
        return sum(self.confidence_distribution) / len(self.confidence_distribution)


@dataclass
class Pattern:
    """
    A learned pattern from observations.

    ECC-style pattern with:
    - Type classification
    - Evidence backing
    - Confidence score
    - Extracted knowledge
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    pattern_type: PatternType = PatternType.WORKFLOW_SEQUENCE
    name: str = ""
    description: str = ""
    extracted_at: datetime = field(default_factory=lambda: datetime.utcnow())

    # Evidence and confidence
    evidence: PatternEvidence = field(default_factory=PatternEvidence)
    confidence: float = 0.0
    frequency: int = 0

    # Pattern content
    trigger_condition: Dict[str, Any] = field(default_factory=dict)
    pattern_content: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: Optional[str] = None

    # Metadata
    project_id: str = "global"
    user_id: str = "default"
    agent_id: str = "default"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "pattern_type": self.pattern_type.value,
            "name": self.name,
            "description": self.description,
            "extracted_at": self.extracted_at.isoformat(),
            "evidence": {
                "observation_ids": self.evidence.observation_ids,
                "example_count": self.evidence.example_count,
                "confidence_distribution": self.evidence.confidence_distribution,
                "context_snippets": self.evidence.context_snippets,
            },
            "confidence": self.confidence,
            "frequency": self.frequency,
            "trigger_condition": self.trigger_condition,
            "pattern_content": self.pattern_content,
            "expected_outcome": self.expected_outcome,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pattern":
        """Create from dictionary."""
        evidence_data = data.get("evidence", {})
        evidence = PatternEvidence(
            observation_ids=evidence_data.get("observation_ids", []),
            example_count=evidence_data.get("example_count", 0),
            confidence_distribution=evidence_data.get("confidence_distribution", []),
            context_snippets=evidence_data.get("context_snippets", []),
        )

        return cls(
            id=data["id"],
            pattern_type=PatternType(data["pattern_type"]),
            name=data["name"],
            description=data["description"],
            extracted_at=datetime.fromisoformat(data["extracted_at"]),
            evidence=evidence,
            confidence=data.get("confidence", 0.0),
            frequency=data.get("frequency", 0),
            trigger_condition=data.get("trigger_condition", {}),
            pattern_content=data.get("pattern_content", {}),
            expected_outcome=data.get("expected_outcome"),
            project_id=data.get("project_id", "global"),
            user_id=data.get("user_id", "default"),
            agent_id=data.get("agent_id", "default"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class SkillCandidate:
    """
    A candidate skill extracted from patterns.

    Skills are higher-level abstractions built from patterns.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    trigger_phrases: List[str] = field(default_factory=list)
    extracted_at: datetime = field(default_factory=lambda: datetime.utcnow())

    # Source patterns
    source_pattern_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0

    # Skill content
    instructions: str = ""
    examples: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    project_id: str = "global"
    agent_id: str = "default"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "trigger_phrases": self.trigger_phrases,
            "extracted_at": self.extracted_at.isoformat(),
            "source_pattern_ids": self.source_pattern_ids,
            "confidence": self.confidence,
            "instructions": self.instructions,
            "examples": self.examples,
            "project_id": self.project_id,
            "agent_id": self.agent_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SkillCandidate":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            trigger_phrases=data.get("trigger_phrases", []),
            extracted_at=datetime.fromisoformat(data["extracted_at"]),
            source_pattern_ids=data.get("source_pattern_ids", []),
            confidence=data.get("confidence", 0.0),
            instructions=data.get("instructions", ""),
            examples=data.get("examples", []),
            project_id=data.get("project_id", "global"),
            agent_id=data.get("agent_id", "default"),
            tags=data.get("tags", []),
        )
