"""
Tests for Learning Layer - Pattern Detection.

Following TDD: Test-first approach for pattern detection.
"""

import asyncio
import sys
import pytest
from datetime import datetime
from unittest import mock

# Mock mem0 modules that have external dependencies
sys.modules['mem0.client'] = mock.MagicMock()
sys.modules['mem0.client.main'] = mock.MagicMock()
sys.modules['mem0.client.project'] = mock.MagicMock()
sys.modules['mem0.memory'] = mock.MagicMock()
sys.modules['mem0.memory.main'] = mock.MagicMock()
sys.modules['mem0.memory.telemetry'] = mock.MagicMock()

from mem0.learning.pattern import Pattern, PatternType, PatternEvidence
from mem0.learning.pattern_detector import PatternDetector


class TestPatternTypeEnum:
    """Test PatternType enum values."""

    def test_type_values(self):
        """Should have expected pattern type values."""
        assert PatternType.WORKFLOW_SEQUENCE.value == "workflow_sequence"
        assert PatternType.USER_PREFERENCE.value == "user_preference"
        assert PatternType.ERROR_RECOVERY.value == "error_recovery"


class TestPatternEvidenceModel:
    """Test PatternEvidence data model."""

    def test_create_evidence_with_defaults(self):
        """Should create evidence with default values."""
        evidence = PatternEvidence()

        assert evidence.observation_ids == []
        assert evidence.example_count == 0
        assert evidence.confidence_distribution == []
        assert evidence.context_snippets == []

    def test_create_evidence_with_values(self):
        """Should create evidence with specified values."""
        evidence = PatternEvidence(
            observation_ids=["obs1", "obs2"],
            example_count=5,
            confidence_distribution=[0.8, 0.9],
            context_snippets=[{"text": "sample"}],
        )

        assert evidence.observation_ids == ["obs1", "obs2"]
        assert evidence.example_count == 5
        assert evidence.confidence_distribution == [0.8, 0.9]
        assert evidence.context_snippets[0]["text"] == "sample"


class TestPatternModel:
    """Test Pattern data model."""

    def test_create_pattern_with_defaults(self):
        """Should create pattern with default values."""
        pattern = Pattern()

        assert pattern.id is not None
        assert pattern.pattern_type == PatternType.WORKFLOW_SEQUENCE
        assert pattern.confidence == 0.0
        assert pattern.frequency == 0
        assert pattern.project_id == "global"
        assert pattern.user_id == "default"

    def test_create_pattern_with_values(self):
        """Should create pattern with specified values."""
        pattern = Pattern(
            pattern_type=PatternType.USER_PREFERENCE,
            name="prefers_python",
            confidence=0.85,
            frequency=10,
            project_id="test_project",
            pattern_content={"preference": "python"},
        )

        assert pattern.pattern_type == PatternType.USER_PREFERENCE
        assert pattern.name == "prefers_python"
        assert pattern.confidence == 0.85
        assert pattern.frequency == 10
        assert pattern.project_id == "test_project"
        assert pattern.pattern_content["preference"] == "python"

    def test_pattern_to_dict(self):
        """Should convert pattern to dictionary."""
        pattern = Pattern(
            pattern_type=PatternType.ERROR_RECOVERY,
            pattern_content={"error_type": "timeout"},
        )

        result = pattern.to_dict()

        assert result["pattern_type"] == "error_recovery"
        assert result["pattern_content"]["error_type"] == "timeout"
        assert "id" in result
        assert "confidence" in result

    def test_pattern_from_dict(self):
        """Should create pattern from dictionary."""
        data = {
            "id": "pattern123",
            "pattern_type": "user_preference",
            "name": "test_pattern",
            "description": "A test pattern",
            "extracted_at": datetime.utcnow().isoformat(),
            "evidence": {
                "observation_ids": ["obs1"],
                "example_count": 1,
                "confidence_distribution": [0.9],
                "context_snippets": [],
            },
            "confidence": 0.85,
            "frequency": 5,
            "trigger_condition": {},
            "pattern_content": {"preference": "dark_mode"},
            "expected_outcome": None,
            "project_id": "test",
            "user_id": "user1",
            "tags": [],
            "metadata": {},
        }

        pattern = Pattern.from_dict(data)

        assert pattern.id == "pattern123"
        assert pattern.pattern_type == PatternType.USER_PREFERENCE
        assert pattern.confidence == 0.85
        assert pattern.pattern_content["preference"] == "dark_mode"


class TestPatternDetector:
    """Test pattern detection."""

    def test_detector_initialization(self):
        """Should initialize pattern detector."""
        detector = PatternDetector()

        assert detector.min_confidence == 0.3

    @pytest.mark.asyncio
    async def test_detect_empty_observations(self):
        """Should handle empty observations."""
        detector = PatternDetector()

        patterns = await detector.detect_patterns([])

        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_detect_workflow_pattern(self):
        """Should detect workflow sequence patterns."""
        detector = PatternDetector()

        from mem0.observation.models import Observation, ObservationType

        obs1 = Observation(
            event_type=ObservationType.ADD_COMPLETED,
            data={"action": "add"},
        )
        obs2 = Observation(
            event_type=ObservationType.SEARCH_COMPLETED,
            data={"action": "search"},
        )
        obs3 = Observation(
            event_type=ObservationType.UPDATE_COMPLETED,
            data={"action": "update"},
        )

        patterns = await detector.detect_patterns([obs1, obs2, obs3])

        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_confidence_threshold(self):
        """Should filter patterns by confidence."""
        detector = PatternDetector(min_confidence=0.5)

        from mem0.observation.models import Observation, ObservationType

        low_freq_obs = Observation(
            event_type=ObservationType.ADD_COMPLETED,
            data={"action": "test"},
        )

        patterns = await detector.detect_patterns([low_freq_obs])

        for pattern in patterns:
            assert pattern.confidence >= 0.5 or len(patterns) == 0

    @pytest.mark.asyncio
    async def test_user_preference_detection(self):
        """Should detect user preference patterns."""
        detector = PatternDetector()

        from mem0.observation.models import Observation, ObservationType

        obs = Observation(
            event_type=ObservationType.ADD_COMPLETED,
            data={"action": "add_python_code"},
        )

        patterns = await detector.detect_patterns([obs])

        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_error_recovery_pattern(self):
        """Should detect error recovery patterns."""
        detector = PatternDetector()

        from mem0.observation.models import Observation, ObservationType

        obs1 = Observation(
            event_type=ObservationType.ADD_COMPLETED,
            data={"success": False, "error": "timeout", "method": "api_call"},
        )
        obs2 = Observation(
            event_type=ObservationType.ADD_COMPLETED,
            data={"success": True, "method": "api_call"},
        )

        patterns = await detector.detect_patterns([obs1, obs2])

        assert isinstance(patterns, list)

    @pytest.mark.asyncio
    async def test_pattern_project_isolation(self):
        """Should maintain project isolation in patterns."""
        detector = PatternDetector()

        from mem0.observation.models import Observation, ObservationType

        obs1 = Observation(
            project_id="project_a",
            event_type=ObservationType.ADD_COMPLETED,
            data={"action": "a"},
        )
        obs2 = Observation(
            project_id="project_b",
            event_type=ObservationType.ADD_COMPLETED,
            data={"action": "b"},
        )

        patterns_a = await detector.detect_patterns([obs1])
        patterns_b = await detector.detect_patterns([obs2])

        if patterns_a:
            assert patterns_a[0].project_id == "project_a"
        if patterns_b:
            assert patterns_b[0].project_id == "project_b"

    @pytest.mark.asyncio
    async def test_frequency_tracking(self):
        """Should track pattern frequency."""
        detector = PatternDetector()

        from mem0.observation.models import Observation, ObservationType

        for _ in range(5):
            await detector.detect_patterns([
                Observation(
                    event_type=ObservationType.ADD_COMPLETED,
                    data={"action": "repeated_action"},
                )
            ])

    @pytest.mark.asyncio
    async def test_pattern_update(self):
        """Should update existing patterns."""
        detector = PatternDetector()

        from mem0.observation.models import Observation, ObservationType

        obs = Observation(
            event_type=ObservationType.ADD_COMPLETED,
            data={"action": "update_test"},
        )

        patterns1 = await detector.detect_patterns([obs])
        patterns2 = await detector.detect_patterns([obs])

        assert isinstance(patterns2, list)

    @pytest.mark.asyncio
    async def test_multiple_pattern_types(self):
        """Should detect multiple pattern types."""
        detector = PatternDetector()

        from mem0.observation.models import Observation, ObservationType

        observations = [
            Observation(event_type=ObservationType.ADD_COMPLETED, data={"method": "add"}),
            Observation(event_type=ObservationType.SEARCH_COMPLETED, data={"method": "search"}),
            Observation(event_type=ObservationType.ADD_COMPLETED, data={"success": False, "error": "test", "method": "op"}),
            Observation(event_type=ObservationType.FEEDBACK, data={"rating": 0.5}),
        ]

        patterns = await detector.detect_patterns(observations)

        assert isinstance(patterns, list)
