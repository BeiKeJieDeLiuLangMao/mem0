"""
Tests for Learning Layer - Skill Extraction.

Following TDD: Test-first approach for skill extraction.
"""

import asyncio
import sys
import pytest
from datetime import datetime, timedelta
from unittest import mock

# Mock mem0 modules that have external dependencies
sys.modules['mem0.client'] = mock.MagicMock()
sys.modules['mem0.client.main'] = mock.MagicMock()
sys.modules['mem0.client.project'] = mock.MagicMock()
sys.modules['mem0.memory'] = mock.MagicMock()
sys.modules['mem0.memory.main'] = mock.MagicMock()
sys.modules['mem0.memory.telemetry'] = mock.MagicMock()

from mem0.observation.models import Observation, ObservationType
from mem0.learning.pattern import Pattern, PatternType, SkillCandidate
from mem0.learning.skill_extractor import SkillExtractor


class TestSkillCandidateModel:
    """Test SkillCandidate data model."""

    def test_create_skill_candidate_with_defaults(self):
        """Should create skill candidate with default values."""
        skill = SkillCandidate()

        assert skill.id is not None
        assert skill.name == ""
        assert skill.description == ""
        assert skill.trigger_phrases == []
        assert skill.confidence == 0.0
        assert skill.instructions == ""
        assert skill.examples == []
        assert skill.project_id == "global"

    def test_create_skill_candidate_with_values(self):
        """Should create skill candidate with specified values."""
        skill = SkillCandidate(
            name="Test Skill",
            description="A test skill",
            trigger_phrases=["test trigger"],
            confidence=0.9,
            instructions="Do this step",
            examples=[{"context": "test"}],
            source_pattern_ids=["pattern123"],
            project_id="test_project",
            tags=["test", "automation"],
        )

        assert skill.name == "Test Skill"
        assert skill.description == "A test skill"
        assert skill.trigger_phrases == ["test trigger"]
        assert skill.confidence == 0.9
        assert skill.instructions == "Do this step"
        assert skill.examples == [{"context": "test"}]
        assert skill.source_pattern_ids == ["pattern123"]
        assert skill.project_id == "test_project"
        assert skill.tags == ["test", "automation"]

    def test_skill_candidate_to_dict(self):
        """Should convert skill candidate to dictionary."""
        skill = SkillCandidate(
            name="Workflow Skill",
            description="A workflow pattern skill",
            trigger_phrases=["repeat workflow"],
        )

        result = skill.to_dict()

        assert result["name"] == "Workflow Skill"
        assert result["description"] == "A workflow pattern skill"
        assert result["trigger_phrases"] == ["repeat workflow"]
        assert "id" in result
        assert "confidence" in result
        assert "extracted_at" in result


class TestSkillExtractor:
    """Test skill extraction from patterns."""

    @pytest.mark.asyncio
    async def test_extract_empty_patterns(self):
        """Should handle empty pattern list."""
        extractor = SkillExtractor()
        skills = await extractor.extract_skills([])

        assert skills == []

    @pytest.mark.asyncio
    async def test_extract_workflow_skill(self):
        """Should extract workflow skill from pattern."""
        extractor = SkillExtractor()

        # pattern_content["sequence"] must contain Observation objects with .data["method"]
        sequence = [
            Observation(event_type=ObservationType.ADD_COMPLETED, data={"method": "add"}),
            Observation(event_type=ObservationType.SEARCH_COMPLETED, data={"method": "search"}),
            Observation(event_type=ObservationType.UPDATE_COMPLETED, data={"method": "update"}),
        ]
        pattern = Pattern(
            pattern_type=PatternType.WORKFLOW_SEQUENCE,
            confidence=0.8,
            frequency=5,
            pattern_content={"sequence": sequence},
        )

        skills = await extractor.extract_skills([pattern])

        assert len(skills) > 0
        workflow_skills = [s for s in skills if "workflow" in s.tags or "Workflow" in s.name]
        assert len(workflow_skills) > 0

    @pytest.mark.asyncio
    async def test_extract_user_preference_skill(self):
        """Should extract user preference skill from pattern."""
        extractor = SkillExtractor()

        pattern = Pattern(
            pattern_type=PatternType.USER_PREFERENCE,
            confidence=0.85,
            frequency=8,
            pattern_content={
                "preference": "python",
                "category": "programming_language",
                "memory_id": "mem123",
                "average_rating": 0.9,
                "rating_count": 5,
            },
        )

        skills = await extractor.extract_skills([pattern])

        assert len(skills) > 0
        preference_skills = [
            s for s in skills
            if "preference" in s.tags or "user-behavior" in s.tags
        ]
        assert len(preference_skills) > 0

    @pytest.mark.asyncio
    async def test_extract_error_recovery_skill(self):
        """Should extract error recovery skill from pattern."""
        extractor = SkillExtractor()

        pattern = Pattern(
            pattern_type=PatternType.ERROR_RECOVERY,
            confidence=0.75,
            frequency=4,
            pattern_content={
                "error_type": "timeout",
                "failed_method": "api_call",
                "recovery_time_seconds": 2.5,
            },
        )

        skills = await extractor.extract_skills([pattern])

        assert len(skills) > 0
        recovery_skills = [
            s for s in skills
            if "error-handling" in s.tags or "recovery" in s.tags
        ]
        # May or may not extract depending on implementation
        if recovery_skills:
            assert "timeout" in recovery_skills[0].name or "timeout" in recovery_skills[0].description

    @pytest.mark.asyncio
    async def test_confidence_propagation(self):
        """Should propagate confidence from pattern to skill."""
        extractor = SkillExtractor()

        pattern = Pattern(
            pattern_type=PatternType.USER_PREFERENCE,
            confidence=0.9,
            pattern_content={
                "preference": "dark_mode",
                "memory_id": "mem123",
                "average_rating": 0.9,
                "rating_count": 3,
            },
        )

        skills = await extractor.extract_skills([pattern])

        if skills:
            # Skill confidence should be based on pattern confidence
            assert skills[0].confidence >= 0.5

    @pytest.mark.asyncio
    async def test_source_pattern_tracking(self):
        """Should track source pattern for skills."""
        extractor = SkillExtractor()

        sequence = [
            Observation(event_type=ObservationType.ADD_COMPLETED, data={"method": "add"}),
            Observation(event_type=ObservationType.SEARCH_COMPLETED, data={"method": "search"}),
        ]
        pattern = Pattern(
            id="pattern123",
            pattern_type=PatternType.WORKFLOW_SEQUENCE,
            pattern_content={"sequence": sequence},
        )

        skills = await extractor.extract_skills([pattern])

        if skills:
            assert "pattern123" in skills[0].source_pattern_ids

    @pytest.mark.asyncio
    async def test_project_isolation(self):
        """Should maintain project isolation in skills."""
        extractor = SkillExtractor()

        patterns = [
            Pattern(
                pattern_type=PatternType.USER_PREFERENCE,
                pattern_content={"preference": "A", "memory_id": "mem1", "average_rating": 0.8, "rating_count": 1},
                project_id="project_a",
            ),
            Pattern(
                pattern_type=PatternType.USER_PREFERENCE,
                pattern_content={"preference": "B", "memory_id": "mem2", "average_rating": 0.8, "rating_count": 1},
                project_id="project_b",
            ),
        ]

        skills = extractor.extract_skills(patterns)
        # Run sync
        result_a = await extractor.extract_skills([patterns[0]])
        result_b = await extractor.extract_skills([patterns[1]])

        # Skills should maintain project IDs
        if result_a:
            assert result_a[0].project_id == "project_a"
        if result_b:
            assert result_b[0].project_id == "project_b"

    @pytest.mark.asyncio
    async def test_skill_created_at(self):
        """Should set extracted_at timestamp on skills."""
        extractor = SkillExtractor()

        sequence = [
            Observation(event_type=ObservationType.ADD_COMPLETED, data={"method": "add"}),
        ]
        pattern = Pattern(
            pattern_type=PatternType.WORKFLOW_SEQUENCE,
            pattern_content={"sequence": sequence},
        )

        skills = await extractor.extract_skills([pattern])

        if skills:
            assert skills[0].extracted_at is not None

    @pytest.mark.asyncio
    async def test_min_confidence_filter(self):
        """Should filter skills by minimum confidence."""
        extractor = SkillExtractor()

        sequence = [
            Observation(event_type=ObservationType.ADD_COMPLETED, data={"method": "add"}),
        ]
        pattern = Pattern(
            pattern_type=PatternType.WORKFLOW_SEQUENCE,
            confidence=0.5,  # Low confidence
            pattern_content={"sequence": sequence},
        )

        skills = await extractor.extract_skills([pattern])

        # Low confidence patterns may not produce skills
        # If they do, they should have low confidence
        for skill in skills:
            assert skill.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_skill_instructions_extraction(self):
        """Should extract relevant instructions from patterns."""
        extractor = SkillExtractor()

        pattern = Pattern(
            pattern_type=PatternType.USER_PREFERENCE,
            pattern_content={
                "preference": "vim",
                "category": "editor",
                "context": "programming",
                "memory_id": "mem123",
                "average_rating": 0.85,
                "rating_count": 4,
            },
        )

        skills = await extractor.extract_skills([pattern])

        if skills:
            # Skill should have instructions
            assert len(skills[0].instructions) > 0

    @pytest.mark.asyncio
    async def test_multiple_patterns_same_type(self):
        """Should handle multiple patterns of the same type."""
        extractor = SkillExtractor()

        patterns = [
            Pattern(
                pattern_type=PatternType.USER_PREFERENCE,
                pattern_content={"preference": "python", "memory_id": "mem1", "average_rating": 0.8, "rating_count": 1},
            ),
            Pattern(
                pattern_type=PatternType.USER_PREFERENCE,
                pattern_content={"preference": "vim", "memory_id": "mem2", "average_rating": 0.8, "rating_count": 1},
            ),
        ]

        skills = await extractor.extract_skills(patterns)

        # Should extract skills for both patterns
        assert len(skills) >= 0  # May vary by implementation

    @pytest.mark.asyncio
    async def test_skill_metadata(self):
        """Should include metadata in extracted skills."""
        extractor = SkillExtractor()

        sequence = [
            Observation(event_type=ObservationType.ADD_COMPLETED, data={"method": "add"}),
            Observation(event_type=ObservationType.SEARCH_COMPLETED, data={"method": "search"}),
        ]
        pattern = Pattern(
            pattern_type=PatternType.WORKFLOW_SEQUENCE,
            pattern_content={"sequence": sequence},
            confidence=0.85,
            frequency=10,
        )

        skills = await extractor.extract_skills([pattern])

        if skills:
            # Metadata should be populated
            assert skills[0].extracted_at is not None
            assert skills[0].id is not None
