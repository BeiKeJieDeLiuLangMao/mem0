"""
Pattern detection from observations.

⚠️ DEPRECATED: This module is deprecated in favor of mem0.learning.turn_aware_detector.
The enhanced version includes:
- Turn-aware pattern detection
- Transaction boundary identification
- Cross-turn workflow analysis

Implements ECC-style pattern extraction with confidence scoring.
"""

import asyncio
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from .pattern import Pattern, PatternType, PatternEvidence
from ..observation.models import Observation, ObservationType


class PatternDetector:
    """
    Detects patterns from observation stream.

    ECC-style pattern detection:
    - Frequency-based confidence scoring
    - Evidence accumulation
    - Multi-type pattern recognition
    """

    # Confidence thresholds (from ECC)
    CONFIDENCE_THRESHOLDS = {
        "min": 0.3,      # 1-2 observations
        "low": 0.7,      # 3-5 observations
        "medium": 0.85,  # 6-10 observations
        "high": 0.95,    # 11+ observations
    }

    # Frequency to confidence mapping
    FREQUENCY_CONFIDENCE = {
        range(1, 3): 0.3,
        range(3, 6): 0.7,
        range(6, 11): 0.85,
        range(11, 1000): 0.95,
    }

    def __init__(self, min_confidence: float = 0.3):
        """
        Initialize detector.

        Args:
            min_confidence: Minimum confidence for pattern retention
        """
        self.min_confidence = min_confidence
        self._pattern_handlers: Dict[PatternType, Callable] = {
            PatternType.WORKFLOW_SEQUENCE: self._detect_workflow_sequence,
            PatternType.USER_PREFERENCE: self._detect_user_preference,
            PatternType.ERROR_RECOVERY: self._detect_error_recovery,
        }

    async def detect_patterns(
        self,
        observations: List[Observation],
    ) -> List[Pattern]:
        """
        Detect patterns from observations.

        Args:
            observations: List of observations to analyze

        Returns:
            List of detected patterns with confidence scores
        """
        patterns = []

        # Group by event type for efficient processing
        by_type = self._group_by_type(observations)

        # Run each detector type
        for pattern_type, handler in self._pattern_handlers.items():
            detected = await handler(observations, by_type)
            patterns.extend(detected)

        # Filter by confidence
        patterns = [p for p in patterns if p.confidence >= self.min_confidence]

        # Sort by confidence
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        return patterns

    def _group_by_type(
        self,
        observations: List[Observation],
    ) -> Dict[ObservationType, List[Observation]]:
        """Group observations by event type."""
        grouped = defaultdict(list)
        for obs in observations:
            grouped[obs.event_type].append(obs)
        return dict(grouped)

    async def _detect_workflow_sequence(
        self,
        observations: List[Observation],
        by_type: Dict[ObservationType, List[Observation]],
    ) -> List[Pattern]:
        """
        Detect workflow sequence patterns.

        Looks for repeated sequences of operations.
        """
        patterns = []

        # Find completed ADD operations
        add_completed = by_type.get(ObservationType.ADD_COMPLETED, [])

        # Group by user/session to identify individual workflows
        by_user = self._group_by_metadata(add_completed, "user_id")
        by_session = self._group_by_metadata(add_completed, "session_id")

        # Analyze each user's workflow patterns
        for user_id, user_obs in by_user.items():
            # Extract operation sequences
            sequences = self._extract_sequences(user_obs)

            # Find repeating sequences
            sequence_counts = defaultdict(list)
            for seq in sequences:
                key = self._sequence_key(seq)
                sequence_counts[key].append(seq)

            # Create patterns from frequent sequences
            for key, seq_list in sequence_counts.items():
                freq = len(seq_list)
                if freq >= 2:
                    confidence = self._frequency_to_confidence(freq)

                    pattern = Pattern(
                        pattern_type=PatternType.WORKFLOW_SEQUENCE,
                        name=f"User {user_id} workflow pattern",
                        description=f"Repeated workflow sequence observed {freq} times",
                        confidence=confidence,
                        frequency=freq,
                        evidence=self._build_evidence(seq_list),
                        pattern_content={"sequence": seq_list[0]},
                        project_id=seq_list[0][0].project_id if seq_list else "global",
                        user_id=user_id,
                    )
                    patterns.append(pattern)

        return patterns

    async def _detect_user_preference(
        self,
        observations: List[Observation],
        by_type: Dict[ObservationType, List[Observation]],
    ) -> List[Pattern]:
        """
        Detect user preference patterns.

        Looks for consistent choices in user behavior.
        """
        patterns = []

        # Analyze feedback observations
        feedback_obs = [o for o in observations if o.event_type == ObservationType.FEEDBACK]

        # Group by memory_id to find patterns in feedback
        by_memory = defaultdict(list)
        for obs in feedback_obs:
            memory_id = obs.data.get("memory_id")
            if memory_id:
                by_memory[memory_id].append(obs)

        # Look for consistent ratings
        for memory_id, mem_feedback in by_memory.items():
            if len(mem_feedback) >= 2:
                ratings = [f.data.get("rating") for f in mem_feedback if f.data.get("rating") is not None]

                if ratings:
                    avg_rating = sum(ratings) / len(ratings)

                    pattern = Pattern(
                        pattern_type=PatternType.USER_PREFERENCE,
                        name=f"Memory {memory_id[:8]} preference",
                        description=f"User consistently rates this memory {avg_rating:.2f}",
                        confidence=self._frequency_to_confidence(len(mem_feedback)),
                        frequency=len(mem_feedback),
                        evidence=self._build_evidence(mem_feedback),
                        pattern_content={
                            "memory_id": memory_id,
                            "average_rating": avg_rating,
                            "rating_count": len(ratings),
                        },
                    )
                    patterns.append(pattern)

        return patterns

    async def _detect_error_recovery(
        self,
        observations: List[Observation],
        by_type: Dict[ObservationType, List[Observation]],
    ) -> List[Pattern]:
        """
        Detect error recovery patterns.

        Looks for successful recoveries after failures.
        """
        patterns = []

        # Find failed operations
        failed_ops = []
        for obs in observations:
            if obs.data.get("success") is False and obs.data.get("error"):
                failed_ops.append(obs)

        # For each failure, look for subsequent successful attempts
        for failed in failed_ops:
            method = failed.data.get("method")

            # Find subsequent successful attempts of same method
            subsequent = [
                o for o in observations
                if o.timestamp > failed.timestamp
                and o.data.get("method") == method
                and o.data.get("success") is True
            ]

            if subsequent:
                # Find the first successful retry
                retry = min(subsequent, key=lambda o: o.timestamp)

                # Calculate time to recovery
                recovery_time = (retry.timestamp - failed.timestamp).total_seconds()

                pattern = Pattern(
                    pattern_type=PatternType.ERROR_RECOVERY,
                    name=f"{method} error recovery",
                    description=f"Recovery pattern for {method} failures",
                    confidence=self._frequency_to_confidence(1),  # Single instance
                    frequency=1,
                    evidence=self._build_evidence([failed, retry]),
                    pattern_content={
                        "error_type": failed.data.get("error_type"),
                        "recovery_time_seconds": recovery_time,
                        "failed_method": method,
                    },
                )
                patterns.append(pattern)

        return patterns

    def _group_by_metadata(
        self,
        observations: List[Observation],
        key: str,
    ) -> Dict[str, List[Observation]]:
        """Group observations by metadata key."""
        grouped = defaultdict(list)
        for obs in observations:
            value = getattr(obs, key, "default")
            grouped[value].append(obs)
        return dict(grouped)

    def _extract_sequences(self, observations: List[Observation]) -> List[List[Observation]]:
        """Extract operation sequences from observations."""
        # Simple implementation: group by timestamp proximity
        sequences = []
        current_seq = []

        sorted_obs = sorted(observations, key=lambda o: o.timestamp)

        for obs in sorted_obs:
            if not current_seq:
                current_seq.append(obs)
            else:
                # If within 5 minutes, continue sequence
                time_diff = (obs.timestamp - current_seq[-1].timestamp).total_seconds()
                if time_diff < 300:  # 5 minutes
                    current_seq.append(obs)
                else:
                    if len(current_seq) >= 2:
                        sequences.append(current_seq)
                    current_seq = [obs]

        if len(current_seq) >= 2:
            sequences.append(current_seq)

        return sequences

    def _sequence_key(self, sequence: List[Observation]) -> str:
        """Create a key for sequence comparison."""
        methods = [obs.data.get("method", "unknown") for obs in sequence]
        return "->".join(methods)

    def _build_evidence(
        self,
        observations: List[Observation],
    ) -> PatternEvidence:
        """Build evidence object from observations."""
        return PatternEvidence(
            observation_ids=[obs.id for obs in observations],
            example_count=len(observations),
            confidence_distribution=[obs.confidence for obs in observations],
            context_snippets=[
                {
                    "timestamp": obs.timestamp.isoformat(),
                    "event_type": obs.event_type.value,
                    "data": obs.data,
                }
                for obs in observations[:5]  # Limit to 5 snippets
            ],
        )

    def _frequency_to_confidence(self, frequency: int) -> float:
        """Map observation frequency to confidence score."""
        for freq_range, confidence in self.FREQUENCY_CONFIDENCE.items():
            if frequency in freq_range:
                return confidence
        return 0.95  # Default high confidence for very frequent patterns
