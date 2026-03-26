"""
Core AI Learning integration for mem0.

⚠️ DEPRECATED: This module is deprecated in favor of mem0.ailearn.enhanced.EnhancedAILearn.
The enhanced version includes:
- Turn-aware pattern detection
- LLM-powered skill extraction
- Cross-session analysis

Use `from mem0.ailearn.enhanced import EnhancedAILearn, enable_enhanced_ailearn` instead.

This module is kept for backward compatibility and will be removed in a future version.

Three-layer self-learning architecture:
1. Observation Layer - 100% reliable capture
2. Learning Layer - Pattern detection & skill extraction
3. Evolution Layer - Health monitoring & amendments
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..observation import (
    MemoryObservationHook,
    FileObservationStore,
    ProjectDetector,
    enable_observation,
)
from ..learning import PatternDetector, SkillExtractor
from ..evolution import HealthMonitor, MetricsCollector, EvolutionTracker
from ..amendment import AmendmentProposer
from ..instincts import InstinctRegistry, InstinctApplier


class Mem0AILearn:
    """
    Main AI Learning integration for mem0.

    Three-layer architecture:
    1. Observation Layer - 100% reliable capture
    2. Learning Layer - Pattern detection & skill extraction
    3. Evolution Layer - Health monitoring & amendments
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: str = "default",
        user_id: str = "default",
        auto_learn: bool = True,
        auto_amend: bool = False,
    ):
        """
        Initialize AI Learning integration.

        Args:
            storage_path: Base path for AI Learning data storage
            project_id: Project identifier (auto-detected if None)
            session_id: Session identifier
            user_id: User identifier
            auto_learn: Enable automatic pattern learning
            auto_amend: Enable automatic amendment application
        """
        # Storage setup
        if storage_path is None:
            import os
            storage_path = os.path.expanduser("~/.mem0/ailearn")

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Project detection
        if project_id is None:
            detector = ProjectDetector()
            project_info = detector.detect()
            self.project_id = project_info.project_id
        else:
            self.project_id = project_id

        self.session_id = session_id
        self.user_id = user_id

        # Configuration
        self.auto_learn = auto_learn
        self.auto_amend = auto_amend

        # Initialize layers
        self._init_observation_layer()
        self._init_learning_layer()
        self._init_evolution_layer()
        self._init_instincts_layer()

    def _init_observation_layer(self):
        """Initialize observation layer."""
        obs_path = self.storage_path / "observations"
        self.observation_store = FileObservationStore(obs_path)
        self.observation_hook = MemoryObservationHook(
            project_id=self.project_id,
            session_id=self.session_id,
            user_id=self.user_id,
            storage_backend=self.observation_store,
        )

    def _init_learning_layer(self):
        """Initialize learning layer."""
        self.pattern_detector = PatternDetector()
        self.skill_extractor = SkillExtractor()

    def _init_evolution_layer(self):
        """Initialize evolution layer."""
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor(self.metrics_collector)
        self.evolution_tracker = EvolutionTracker()
        self.amendment_proposer = AmendmentProposer()

    def _init_instincts_layer(self):
        """Initialize instincts layer."""
        self.instinct_registry = InstinctRegistry()
        self.instinct_applier = InstinctApplier(self.instinct_registry)

    def wrap_memory(self, memory_instance: Any) -> Any:
        """
        Wrap a Memory instance with AI Learning observation.

        Args:
            memory_instance: mem0 Memory instance

        Returns:
            Wrapped Memory with AI Learning enabled
        """
        return self.observation_hook.wrap_memory(memory_instance)

    async def analyze_and_learn(
        self,
        limit: Optional[int] = 1000,
    ) -> Dict[str, Any]:
        """
        Run learning analysis on recent observations.

        Args:
            limit: Maximum observations to analyze

        Returns:
            Analysis results with patterns and skills
        """
        # Get recent observations
        observations = await self.observation_store.get_by_project(
            self.project_id,
            limit=limit,
        )

        # Detect patterns
        patterns = await self.pattern_detector.detect_patterns(observations)

        # Extract skills
        skills = await self.skill_extractor.extract_skills(patterns)

        # Generate amendments
        proposals = await self.amendment_proposer.propose_amendments(
            observations,
            patterns,
        )

        # Auto-promote high-confidence patterns to instincts
        for pattern in patterns:
            if pattern.confidence >= 0.95:
                await self._promote_to_instinct(pattern)

        return {
            "observations_analyzed": len(observations),
            "patterns_detected": len(patterns),
            "skills_extracted": len(skills),
            "amendments_proposed": len(proposals),
            "patterns": [p.to_dict() for p in patterns[:10]],
            "skills": [s.to_dict() for s in skills[:10]],
            "amendments": [a.to_dict() for a in proposals[:10]],
        }

    async def _promote_to_instinct(self, pattern) -> None:
        """Promote a high-confidence pattern to an instinct."""
        # This would create an instinct from the pattern
        pass

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get current system health status.

        Returns:
            Health status with metrics and alerts
        """
        status, alerts = await self.health_monitor.get_health_status()

        metrics = await self.metrics_collector.get_current_metrics()

        return {
            "status": status.value,
            "metrics": {
                "total_adds": metrics.total_adds,
                "total_updates": metrics.total_updates,
                "total_deletes": metrics.total_deletes,
                "total_searches": metrics.total_searches,
                "add_success_rate": metrics.add_success_rate,
                "overall_trend": metrics.overall_trend,
            },
            "alerts": [
                {
                    "severity": a.severity,
                    "message": a.message,
                }
                for a in alerts
            ],
        }

    async def apply_amendment(
        self,
        proposal_id: str,
        memory_content: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Apply an approved amendment proposal.

        Args:
            proposal_id: ID of the proposal to apply
            memory_content: Current memory content

        Returns:
            Updated memory content, or None if proposal not found
        """
        # This would look up the proposal and apply it
        # For now, stub implementation
        return None

    async def flush(self) -> None:
        """Flush all pending observations."""
        await self.observation_hook.flush()

    async def shutdown(self) -> None:
        """Shutdown AI Learning system gracefully."""
        await self.flush()


def enable_ailearn(
    memory_instance,
    storage_path: Optional[str] = None,
    project_id: Optional[str] = None,
    session_id: str = "default",
    user_id: str = "default",
    auto_learn: bool = True,
) -> Mem0AILearn:
    """
    Enable AI Learning for a mem0 Memory instance.

    Args:
        memory_instance: mem0 Memory instance
        storage_path: Path for AI Learning data storage
        project_id: Project identifier (auto-detected if None)
        session_id: Session identifier
        user_id: User identifier
        auto_learn: Enable automatic learning

    Returns:
        Mem0AILearn instance for managing AI Learning features

    Example:
        >>> from mem0 import Memory
        >>> from mem0.ailearn import enable_ailearn
        >>>
        >>> memory = Memory()
        >>> ailearn = enable_ailearn(memory)
        >>>
        >>> # Use memory normally - all operations are observed
        >>> memory.add("User prefers Python", user_id="alice")
        >>>
        >>> # Analyze and learn from observations
        >>> results = await ailearn.analyze_and_learn()
        >>>
        >>> # Check health
        >>> health = await ailearn.get_health_status()
    """
    ailearn = Mem0AILearn(
        storage_path=storage_path,
        project_id=project_id,
        session_id=session_id,
        user_id=user_id,
        auto_learn=auto_learn,
    )

    # Wrap memory
    ailearn.wrap_memory(memory_instance)

    return ailearn
