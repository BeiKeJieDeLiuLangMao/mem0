"""
Enhanced Pattern Detection with Turn-Aware Analysis

核心思想：
1. 读取多个 Turns，重建完整的对话上下文
2. 识别跨 Turn 的事务边界（一个完整任务可能跨越多个 Turn）
3. 分析事务序列，发现更长的工作流模式
4. 识别决策点（用户在不同情境下的选择）
5. 提取跨会话的复用技能
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from ..observation.models import Observation, ObservationType
from .pattern import Pattern, PatternType, PatternEvidence

logger = logging.getLogger(__name__)


class TurnAwarePatternDetector:
    """
    Turn-aware pattern detector that analyzes complete conversations.

    Key features:
    - Reconstructs conversation context from multiple turns
    - Identifies transaction boundaries across turns
    - Detects longer workflow patterns
    - Extracts decision points and user preferences
    """

    def __init__(
        self,
        llm_client: Optional[Any] = None,  # OpenAI client from server.py
        min_confidence: float = 0.3,
        max_turns_per_analysis: int = 50,
    ):
        """
        Initialize turn-aware detector.

        Args:
            llm_client: LLM client for semantic analysis
            min_confidence: Minimum confidence for pattern retention
            max_turns_per_analysis: Maximum turns to analyze in one batch
        """
        self.llm_client = llm_client
        self.min_confidence = min_confidence
        self.max_turns_per_analysis = max_turns_per_analysis

    async def detect_patterns(
        self,
        observations: List[Observation],
        turns: List[Dict[str, Any]],  # Turns from OpenMemory database
    ) -> List[Pattern]:
        """
        Detect patterns with turn-aware analysis.

        Args:
            observations: Memory operation observations
            turns: Conversation turns from /api/v1/turns

        Returns:
            List of detected patterns
        """
        patterns = []

        # 1. 重建会话上下文
        sessions = self._reconstruct_sessions(turns, observations)

        # 2. 分析每个会话的事务序列
        for session_id, session_data in sessions.items():
            session_patterns = await self._analyze_session(session_id, session_data)
            patterns.extend(session_patterns)

        # 3. 跨会话模式识别
        cross_session_patterns = await self._detect_cross_session_patterns(sessions)
        patterns.extend(cross_session_patterns)

        # 4. 过滤和排序
        patterns = [p for p in patterns if p.confidence >= self.min_confidence]
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        return patterns

    def _reconstruct_sessions(
        self,
        turns: List[Dict[str, Any]],
        observations: List[Observation],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Reconstruct complete session contexts.

        Returns:
            {
                session_id: {
                    "turns": [...],
                    "observations": [...],
                    "start_time": datetime,
                    "end_time": datetime,
                    "user_id": str,
                    "agent_id": str,
                }
            }
        """
        sessions = defaultdict(lambda: {
            "turns": [],
            "observations": [],
            "start_time": None,
            "end_time": None,
            "user_id": None,
            "agent_id": None,
        })

        # Group turns by session
        for turn in turns:
            session_id = turn.get("session_id", "default")
            sessions[session_id]["turns"].append(turn)

            # Track metadata
            if not sessions[session_id]["user_id"]:
                sessions[session_id]["user_id"] = turn.get("user_id")
            if not sessions[session_id]["agent_id"]:
                sessions[session_id]["agent_id"] = turn.get("agent_id")

        # Group observations by session
        for obs in observations:
            session_id = obs.session_id
            sessions[session_id]["observations"].append(obs)

        # Calculate time bounds
        for session_id, data in sessions.items():
            all_timestamps = []

            for turn in data["turns"]:
                ts = turn.get("created_at")
                if ts:
                    all_timestamps.append(datetime.fromisoformat(ts))

            for obs in data["observations"]:
                all_timestamps.append(obs.timestamp)

            if all_timestamps:
                data["start_time"] = min(all_timestamps)
                data["end_time"] = max(all_timestamps)

        return dict(sessions)

    async def _analyze_session(
        self,
        session_id: str,
        session_data: Dict[str, Any],
    ) -> List[Pattern]:
        """
        Analyze a single session for patterns.

        Key analyses:
        1. Transaction boundary detection
        2. Workflow sequence extraction
        3. Decision point identification
        4. User preference extraction
        """
        patterns = []

        turns = session_data["turns"]
        observations = session_data["observations"]

        if not turns or not observations:
            return patterns

        # 1. 识别事务边界
        transactions = self._identify_transactions(turns, observations)

        # 2. 分析每个事务
        for transaction in transactions:
            transaction_patterns = await self._analyze_transaction(transaction)
            patterns.extend(transaction_patterns)

        # 3. 检测决策点
        decision_patterns = await self._detect_decision_points(turns, observations)
        patterns.extend(decision_patterns)

        # 4. 提取用户偏好
        preference_patterns = await self._extract_preferences(turns, observations)
        patterns.extend(preference_patterns)

        return patterns

    def _identify_transactions(
        self,
        turns: List[Dict[str, Any]],
        observations: List[Observation],
    ) -> List[Dict[str, Any]]:
        """
        Identify transaction boundaries across turns.

        A transaction = a complete unit of work that accomplishes a goal.

        Heuristics:
        - Time gap > 5 minutes suggests new transaction
        - Topic change suggests new transaction
        - Memory operations cluster indicates transaction activity
        """
        transactions = []

        # Sort by time
        sorted_turns = sorted(
            turns,
            key=lambda t: datetime.fromisoformat(t.get("created_at", ""))
        )

        current_transaction = {
            "turns": [],
            "observations": [],
            "start_time": None,
            "end_time": None,
            "topic": None,
        }

        for i, turn in enumerate(sorted_turns):
            turn_time = datetime.fromisoformat(turn.get("created_at", ""))

            # Check if this starts a new transaction
            is_new_transaction = (
                not current_transaction["turns"]  # First turn
                or (turn_time - current_transaction["end_time"]).total_seconds() > 300  # 5 min gap
                or self._detect_topic_change(turn, current_transaction["turns"][-1])
            )

            if is_new_transaction and current_transaction["turns"]:
                # Finalize current transaction
                transactions.append(current_transaction)
                current_transaction = {
                    "turns": [],
                    "observations": [],
                    "start_time": turn_time,
                    "end_time": turn_time,
                    "topic": self._extract_topic(turn),
                }
            else:
                current_transaction["end_time"] = turn_time

            current_transaction["turns"].append(turn)

            # Find related observations
            turn_obs = self._find_observations_for_turn(
                turn,
                observations,
                window_seconds=60,  # Look 60s before/after turn
            )
            current_transaction["observations"].extend(turn_obs)

        if current_transaction["turns"]:
            transactions.append(current_transaction)

        return transactions

    def _detect_topic_change(self, turn1: Dict, turn2: Dict) -> bool:
        """
        Detect if topic changed between turns.

        Uses semantic analysis if LLM available, otherwise heuristic.
        """
        if not self.llm_client:
            # Fallback: check if message content overlaps
            messages1 = self._extract_user_messages(turn1)
            messages2 = self._extract_user_messages(turn2)

            # Simple heuristic: low word overlap suggests topic change
            words1 = set(messages1.lower().split())
            words2 = set(messages2.lower().split())

            if not words1 or not words2:
                return False

            overlap = len(words1 & words2) / max(len(words1), len(words2))
            return overlap < 0.2  # Less than 20% overlap = topic change

        # TODO: Use LLM for semantic similarity
        return False

    def _extract_topic(self, turn: Dict) -> str:
        """Extract topic from turn."""
        messages = turn.get("messages", [])
        if not messages:
            return "unknown"

        # Get last user message
        for msg in reversed(messages):
            if msg.get("role") == "user":
                text = msg.get("content", "")[:100]
                return text[:50] + "..." if len(text) > 50 else text

        return "unknown"

    def _extract_user_messages(self, turn: Dict) -> str:
        """Extract all user messages from a turn."""
        messages = turn.get("messages", [])
        user_messages = [
            m.get("content", "")
            for m in messages
            if m.get("role") == "user"
        ]
        return " ".join(user_messages)

    def _find_observations_for_turn(
        self,
        turn: Dict,
        observations: List[Observation],
        window_seconds: int = 60,
    ) -> List[Observation]:
        """Find observations that occurred during a turn."""
        turn_time = datetime.fromisoformat(turn.get("created_at", ""))

        return [
            obs
            for obs in observations
            if abs((obs.timestamp - turn_time).total_seconds()) <= window_seconds
        ]

    async def _analyze_transaction(
        self,
        transaction: Dict[str, Any],
    ) -> List[Pattern]:
        """
        Analyze a transaction for patterns.

        Detects:
        1. Workflow patterns (sequence of operations)
        2. Task accomplishment patterns
        3. Resource usage patterns
        """
        patterns = []

        turns = transaction["turns"]
        observations = transaction["observations"]

        if not turns:
            return patterns

        # Build transaction representation
        transaction_summary = self._build_transaction_summary(transaction)

        # Detect workflow patterns
        workflow_pattern = await self._detect_workflow_pattern(
            transaction,
            transaction_summary,
        )
        if workflow_pattern:
            patterns.append(workflow_pattern)

        # Detect task patterns
        task_pattern = await self._detect_task_pattern(
            transaction,
            transaction_summary,
        )
        if task_pattern:
            patterns.append(task_pattern)

        return patterns

    def _build_transaction_summary(self, transaction: Dict) -> Dict[str, Any]:
        """Build a summary of transaction for analysis."""
        turns = transaction["turns"]
        observations = transaction["observations"]

        # Extract user intents
        user_messages = [
            self._extract_user_messages(turn)
            for turn in turns
        ]

        # Extract memory operations
        memory_operations = [
            {
                "type": obs.event_type.value,
                "method": obs.data.get("method", "unknown"),
                "timestamp": obs.timestamp.isoformat(),
            }
            for obs in observations
        ]

        # Extract tool calls
        tool_calls = [
            call
            for turn in turns
            for msg in turn.get("messages", [])
            for call in msg.get("tool_calls", [])
        ]

        return {
            "user_messages": user_messages,
            "memory_operations": memory_operations,
            "tool_calls": tool_calls,
            "turn_count": len(turns),
            "duration_seconds": (
                (transaction["end_time"] - transaction["start_time"]).total_seconds()
                if transaction["start_time"] and transaction["end_time"]
                else 0
            ),
            "topic": transaction.get("topic", "unknown"),
        }

    async def _detect_workflow_pattern(
        self,
        transaction: Dict,
        summary: Dict[str, Any],
    ) -> Optional[Pattern]:
        """
        Detect workflow patterns in transaction.

        A workflow pattern = a repeatable sequence of operations
        that accomplishes a specific goal.
        """
        if not self.llm_client:
            # Fallback: sequence-based detection
            return self._detect_sequence_pattern(transaction, summary)

        # TODO: Use LLM to identify semantic workflows
        return None

    def _detect_sequence_pattern(
        self,
        transaction: Dict,
        summary: Dict[str, Any],
    ) -> Optional[Pattern]:
        """Detect patterns based on operation sequences."""
        operations = summary["memory_operations"]

        if len(operations) < 2:
            return None

        # Extract operation sequence
        sequence = [op["type"] for op in operations]

        # Build pattern key
        sequence_key = "->".join(sequence)

        return Pattern(
            pattern_type=PatternType.WORKFLOW_SEQUENCE,
            name=f"Workflow: {sequence_key}",
            description=f"Transaction with {len(operations)} operations over {summary['duration_seconds']:.0f}s",
            confidence=0.5,  # Base confidence, will be updated with frequency
            frequency=1,
            evidence=PatternEvidence(
                observation_ids=[],
                example_count=1,
                confidence_distribution=[0.5],
                context_snippets=[{
                    "transaction_summary": summary,
                    "turn_count": len(transaction["turns"]),
                }],
            ),
            pattern_content={
                "sequence": sequence,
                "operations": operations,
                "duration_seconds": summary["duration_seconds"],
            },
            project_id="openmemory",
            user_id=transaction["turns"][0].get("user_id") if transaction["turns"] else "default",
        )

    async def _detect_task_pattern(
        self,
        transaction: Dict,
        summary: Dict[str, Any],
    ) -> Optional[Pattern]:
        """
        Detect task accomplishment patterns.

        A task pattern = what the user was trying to accomplish
        and how they went about it.
        """
        if not self.llm_client:
            return None

        # TODO: Use LLM to extract task intent
        return None

    async def _detect_decision_points(
        self,
        turns: List[Dict[str, Any]],
        observations: List[Observation],
    ) -> List[Pattern]:
        """
        Detect decision points in user behavior.

        A decision point = where user made a choice between alternatives.
        """
        patterns = []

        # Look for patterns like:
        # - Search → Choose specific result
        # - Multiple tools tried → One succeeded
        # - Error → Retry with different parameters

        # TODO: Implement decision point detection
        return patterns

    async def _extract_preferences(
        self,
        turns: List[Dict[str, Any]],
        observations: List[Observation],
    ) -> List[Pattern]:
        """
        Extract user preferences from behavior.

        Looks for:
        - Consistent choices
        - Repeated queries
        - Preferred tools/methods
        """
        patterns = []

        # TODO: Implement preference extraction
        return patterns

    async def _detect_cross_session_patterns(
        self,
        sessions: Dict[str, Dict[str, Any]],
    ) -> List[Pattern]:
        """
        Detect patterns that span multiple sessions.

        Looks for:
        - Repeated workflows across time
        - Long-term behavior trends
        - Consistent preferences
        """
        patterns = []

        # TODO: Implement cross-session analysis
        return patterns
