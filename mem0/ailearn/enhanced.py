"""
Enhanced AI Learning with Turn-Aware Analysis

整合增强的模式检测和技能提取，使用 LLM 分析对话上下文。
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..observation.models import Observation
from ..learning.turn_aware_detector import TurnAwarePatternDetector
from ..learning.enhanced_skill_extractor import EnhancedSkillExtractor
from ..amendment.proposer import AmendmentProposer
from ..evolution.health_monitor import HealthMonitor
from ..evolution.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class EnhancedAILearn:
    """
    Enhanced AI Learning with turn-aware analysis and LLM support.

    Features:
    1. Turn-aware pattern detection (跨 Turn 事务分析)
    2. LLM-powered skill extraction
    3. Cross-session pattern recognition
    4. Enhanced amendment proposals
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        project_id: Optional[str] = None,
        session_id: str = "default",
        user_id: str = "default",
        llm_client: Optional[Any] = None,  # OpenAI client
        llm_model: str = "gpt-4.1-nano-2025-04-14",
        auto_learn: bool = True,
        auto_amend: bool = False,
    ):
        """
        Initialize Enhanced AI Learning.

        Args:
            storage_path: Base path for data storage
            project_id: Project identifier
            session_id: Session identifier
            user_id: User identifier
            llm_client: LLM client for analysis
            llm_model: Model to use
            auto_learn: Enable automatic learning
            auto_amend: Enable automatic amendments
        """
        from pathlib import Path
        import os

        # Storage setup
        if storage_path is None:
            storage_path = os.path.expanduser("~/.mem0/ailearn")

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Project detection
        if project_id is None:
            from ..observation.collectors.project_detector import ProjectDetector
            detector = ProjectDetector()
            project_info = detector.detect()
            self.project_id = project_info.project_id
        else:
            self.project_id = project_id

        self.session_id = session_id
        self.user_id = user_id

        # LLM client
        self.llm_client = llm_client
        self.llm_model = llm_model

        # Configuration
        self.auto_learn = auto_learn
        self.auto_amend = auto_amend

        # Initialize components
        self._init_components()

    def _init_components(self):
        """Initialize learning components."""
        from ..observation import (
            MemoryObservationHook,
            FileObservationStore,
        )

        # Observation layer
        obs_path = self.storage_path / "observations"
        self.observation_store = FileObservationStore(obs_path)
        self.observation_hook = MemoryObservationHook(
            project_id=self.project_id,
            session_id=self.session_id,
            user_id=self.user_id,
            storage_backend=self.observation_store,
        )

        # Enhanced learning layer
        self.pattern_detector = TurnAwarePatternDetector(
            llm_client=self.llm_client,
            min_confidence=0.3,
        )
        self.skill_extractor = EnhancedSkillExtractor(
            llm_client=self.llm_client,
            model=self.llm_model,
        )

        # Evolution layer
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor(self.metrics_collector)
        self.amendment_proposer = AmendmentProposer()

    def wrap_memory(self, memory_instance: Any) -> Any:
        """Wrap memory instance with observation hooks."""
        return self.observation_hook.wrap_memory(memory_instance)

    async def analyze_and_learn(
        self,
        turns: List[Dict[str, Any]],  # New parameter: conversation turns
        limit: Optional[int] = 1000,
    ) -> Dict[str, Any]:
        """
        Run enhanced learning analysis.

        Args:
            turns: Conversation turns from OpenMemory
            limit: Maximum observations to analyze

        Returns:
            Analysis results
        """
        # Get recent observations
        observations = await self.observation_store.get_by_project(
            self.project_id,
            limit=limit,
        )

        # Detect patterns with turn-aware analysis
        patterns = await self.pattern_detector.detect_patterns(
            observations,
            turns,
        )

        # Extract skills with LLM
        # Build turn contexts for skill extraction
        turn_contexts = self._group_turns_by_session(turns)
        skills = await self.skill_extractor.extract_skills(
            patterns,
            turn_contexts,
        )

        # Generate amendments
        proposals = await self.amendment_proposer.propose_amendments(
            observations,
            patterns,
        )

        # Auto-promote high-confidence patterns
        for pattern in patterns:
            if pattern.confidence >= 0.95:
                await self._promote_to_instinct(pattern)

        return {
            "observations_analyzed": len(observations),
            "turns_analyzed": len(turns),
            "patterns_detected": len(patterns),
            "skills_extracted": len(skills),
            "amendments_proposed": len(proposals),
            "patterns": [p.to_dict() for p in patterns[:10]],
            "skills": [s.to_dict() for s in skills[:10]],
            "amendments": [a.to_dict() for a in proposals[:10]],
        }

    def _group_turns_by_session(
        self,
        turns: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group turns by session_id."""
        from collections import defaultdict

        grouped = defaultdict(list)
        for turn in turns:
            session_id = turn.get("session_id", "default")
            grouped[session_id].append(turn)

        return dict(grouped)

    async def _promote_to_instinct(self, pattern) -> None:
        """Promote high-confidence pattern to instinct."""
        # TODO: Implement instinct promotion
        pass

    async def get_health_status(self) -> Dict[str, Any]:
        """Get current system health status."""
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
        """Apply an approved amendment proposal."""
        # TODO: Implement amendment application
        return None

    async def flush(self) -> None:
        """Flush all pending observations."""
        await self.observation_hook.flush()

    async def shutdown(self) -> None:
        """Shutdown Enhanced AI Learning gracefully."""
        await self.flush()


def enable_enhanced_ailearn(
    memory_instance,
    turns: List[Dict[str, Any]],  # New parameter
    llm_client: Optional[Any] = None,
    llm_model: str = "gpt-4.1-nano-2025-04-14",
    storage_path: Optional[str] = None,
    project_id: Optional[str] = None,
    session_id: str = "default",
    user_id: str = "default",
    auto_learn: bool = True,
) -> EnhancedAILearn:
    """
    Enable Enhanced AI Learning for a mem0 Memory instance.

    Args:
        memory_instance: mem0 Memory instance
        turns: Conversation turns for analysis
        llm_client: LLM client (optional, from server.py)
        llm_model: Model to use
        storage_path: Path for data storage
        project_id: Project identifier
        session_id: Session identifier
        user_id: User identifier
        auto_learn: Enable automatic learning

    Returns:
        EnhancedAILearn instance

    Example:
        >>> from mem0 import Memory
        >>> from mem0.ailearn.enhanced import enable_enhanced_ailearn
        >>>
        >>> # Get LLM client
        >>> from openai import OpenAI
        >>> client = OpenAI(
        ...     api_key='your-key',
        ...     base_url='https://your-endpoint/v1'
        ... )
        >>>
        >>> # Get turns from OpenMemory
        >>> turns = fetch_turns_from_db()
        >>>
        >>> memory = Memory()
        >>> ailearn = enable_enhanced_ailearn(
        ...     memory,
        ...     turns=turns,
        ...     llm_client=client
        ... )
        >>>
        >>> # Use memory normally
        >>> results = await ailearn.analyze_and_learn(turns)
    """
    ailearn = EnhancedAILearn(
        storage_path=storage_path,
        project_id=project_id,
        session_id=session_id,
        user_id=user_id,
        llm_client=llm_client,
        llm_model=llm_model,
        auto_learn=auto_learn,
    )

    # Wrap memory
    ailearn.wrap_memory(memory_instance)

    return ailearn
