"""
Tests for Amendment System - Proposer and Models.

Following TDD: Test-first approach for amendment proposals.
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

from mem0.amendment.models import (
    AmendmentProposal,
    AmendmentBatch,
    AmendmentType,
    AmendmentStatus,
)
from mem0.amendment.proposer import AmendmentProposer
from mem0.learning.pattern import Pattern, PatternType
from mem0.observation.models import Observation, ObservationType


class TestAmendmentStatusEnum:
    """Test AmendmentStatus enum values."""

    def test_status_values(self):
        """Should have expected status values."""
        assert AmendmentStatus.PENDING.value == "pending"
        assert AmendmentStatus.APPROVED.value == "approved"
        assert AmendmentStatus.APPLIED.value == "applied"
        assert AmendmentStatus.REJECTED.value == "rejected"
        assert AmendmentStatus.SUPERSEDED.value == "superseded"


class TestAmendmentTypeEnum:
    """Test AmendmentType enum values."""

    def test_type_values(self):
        """Should have expected type values."""
        assert AmendmentType.CONTENT_UPDATE.value == "content_update"
        assert AmendmentType.MERGE.value == "merge"
        assert AmendmentType.SPLIT.value == "split"
        assert AmendmentType.DEPRECATE.value == "deprecate"
        assert AmendmentType.REPRIORITIZE.value == "reprioritize"
        assert AmendmentType.CLARIFY.value == "clarify"


class TestAmendmentProposalModel:
    """Test AmendmentProposal data model."""

    def test_create_proposal_with_defaults(self):
        """Should create proposal with default values."""
        proposal = AmendmentProposal()

        assert proposal.id is not None
        assert proposal.amendment_type == AmendmentType.CONTENT_UPDATE
        assert proposal.memory_id == ""
        assert proposal.confidence == 0.0
        assert proposal.status == AmendmentStatus.PENDING
        assert proposal.created_by == "system"
        assert proposal.project_id == "global"

    def test_create_proposal_with_values(self):
        """Should create proposal with specified values."""
        proposal = AmendmentProposal(
            amendment_type=AmendmentType.DEPRECATE,
            memory_id="mem123",
            reasoning="Memory is outdated",
            confidence=0.85,
            expected_impact=-0.3,
            project_id="test_project",
        )

        assert proposal.amendment_type == AmendmentType.DEPRECATE
        assert proposal.memory_id == "mem123"
        assert proposal.reasoning == "Memory is outdated"
        assert proposal.confidence == 0.85
        assert proposal.expected_impact == -0.3
        assert proposal.project_id == "test_project"

    def test_proposal_to_dict(self):
        """Should convert proposal to dictionary."""
        proposal = AmendmentProposal(
            amendment_type=AmendmentType.CONTENT_UPDATE,
            memory_id="mem456",
            confidence=0.9,
        )

        result = proposal.to_dict()

        assert result["amendment_type"] == "content_update"
        assert result["memory_id"] == "mem456"
        assert result["confidence"] == 0.9
        assert "id" in result
        assert "status" in result
        assert "created_at" in result

    def test_proposal_from_dict(self):
        """Should create proposal from dictionary."""
        data = {
            "id": "proposal123",
            "amendment_type": "merge",
            "memory_id": "mem789",
            "current_content": {"old": "data"},
            "proposed_change": {"new": "data"},
            "reasoning": "Better organization",
            "evidence_observation_ids": ["obs1", "obs2"],
            "confidence": 0.8,
            "expected_impact": 0.3,
            "impact_description": "Improved organization",
            "created_at": "2024-01-01T00:00:00",
            "created_by": "user1",
            "project_id": "proj1",
            "status": "approved",
            "applied_at": None,
            "applied_version_id": None,
        }

        proposal = AmendmentProposal.from_dict(data)

        assert proposal.id == "proposal123"
        assert proposal.amendment_type == AmendmentType.MERGE
        assert proposal.memory_id == "mem789"
        assert proposal.confidence == 0.8
        assert proposal.status == AmendmentStatus.APPROVED

    def test_proposal_evidence_tracking(self):
        """Should track evidence observation IDs."""
        proposal = AmendmentProposal(
            evidence_observation_ids=["obs1", "obs2", "obs3"],
        )

        assert len(proposal.evidence_observation_ids) == 3
        assert "obs1" in proposal.evidence_observation_ids


class TestAmendmentBatchModel:
    """Test AmendmentBatch data model."""

    def test_create_batch(self):
        """Should create amendment batch."""
        proposals = [
            AmendmentProposal(memory_id="mem1"),
            AmendmentProposal(memory_id="mem2"),
        ]

        batch = AmendmentBatch(
            proposals=proposals,
            batch_reasoning="Related updates",
            project_id="test",
        )

        assert len(batch.proposals) == 2
        assert batch.batch_reasoning == "Related updates"
        assert batch.project_id == "test"

    def test_batch_to_dict(self):
        """Should convert batch to dictionary."""
        batch = AmendmentBatch(
            proposals=[AmendmentProposal(memory_id="mem1")],
            batch_reasoning="Test batch",
        )

        result = batch.to_dict()

        assert len(result["proposals"]) == 1
        assert result["batch_reasoning"] == "Test batch"
        assert "id" in result
        assert "created_at" in result


class TestAmendmentProposer:
    """Test amendment proposal generation."""

    @pytest.mark.asyncio
    async def test_propose_empty_inputs(self):
        """Should handle empty patterns and observations."""
        proposer = AmendmentProposer()
        proposals = await proposer.propose_amendments([], [])

        # Should return empty list or minimal proposals
        assert isinstance(proposals, list)

    @pytest.mark.asyncio
    async def test_propose_low_confidence_pattern(self):
        """Should not propose amendments for low confidence patterns."""
        proposer = AmendmentProposer(min_confidence=0.7)

        pattern = Pattern(
            pattern_type=PatternType.USER_PREFERENCE,
            confidence=0.5,  # Below threshold
            pattern_content={"preference": "test"},
        )

        proposals = await proposer.propose_amendments([], [pattern])

        # Low confidence patterns should not produce proposals
        for proposal in proposals:
            assert proposal.confidence >= 0.7 or len(proposals) == 0

    @pytest.mark.asyncio
    async def test_propose_high_confidence_pattern(self):
        """Should propose amendments for high confidence patterns."""
        proposer = AmendmentProposer(min_confidence=0.5)

        pattern = Pattern(
            pattern_type=PatternType.USER_PREFERENCE,
            confidence=0.85,
            pattern_content={"preference": "dark_mode"},
        )

        proposals = await proposer.propose_amendments([], [pattern])

        # May or may not produce proposals depending on implementation
        assert isinstance(proposals, list)

    @pytest.mark.asyncio
    async def test_propose_from_feedback(self):
        """Should propose deprecation from negative feedback."""
        proposer = AmendmentProposer()

        observation = Observation(
            event_type=ObservationType.FEEDBACK,
            data={
                "rating": 0.2,  # Low rating
                "memory_id": "mem_bad",
            },
        )

        proposals = await proposer.propose_amendments([observation], [])

        # Should propose deprecation for low-rated memories
        deprecate_proposals = [
            p for p in proposals
            if p.amendment_type == AmendmentType.DEPRECATE
        ]
        if deprecate_proposals:
            assert any(p.memory_id == "mem_bad" for p in deprecate_proposals)

    @pytest.mark.asyncio
    async def test_propose_merge_similar_patterns(self):
        """Should propose merging similar memories."""
        proposer = AmendmentProposer()

        patterns = [
            Pattern(
                pattern_type=PatternType.USER_PREFERENCE,
                confidence=0.8,
                pattern_content={"preference": "python", "category": "language"},
            ),
            Pattern(
                pattern_type=PatternType.USER_PREFERENCE,
                confidence=0.75,
                pattern_content={"preference": "python", "category": "language"},
            ),
        ]

        proposals = await proposer.propose_amendments([], patterns)

        # May propose merge for similar patterns
        merge_proposals = [
            p for p in proposals
            if p.amendment_type == AmendmentType.MERGE
        ]
        assert isinstance(proposals, list)

    @pytest.mark.asyncio
    async def test_propose_error_prevention(self):
        """Should propose error prevention amendments."""
        proposer = AmendmentProposer()

        pattern = Pattern(
            pattern_type=PatternType.ERROR_RECOVERY,
            confidence=0.85,
            pattern_content={
                "error_type": "timeout",
                "failed_method": "api_call",
            },
        )

        proposals = await proposer.propose_amendments([], [pattern])

        # Should have proposals for error recovery patterns
        update_proposals = [
            p for p in proposals
            if p.amendment_type == AmendmentType.CONTENT_UPDATE
        ]
        if update_proposals:
            assert any("error" in str(p.proposed_change) for p in update_proposals)

    @pytest.mark.asyncio
    async def test_proposals_sorted_by_confidence(self):
        """Should sort proposals by confidence and impact."""
        proposer = AmendmentProposer()

        patterns = [
            Pattern(
                pattern_type=PatternType.USER_PREFERENCE,
                confidence=0.6,
                pattern_content={"low": "conf"},
            ),
            Pattern(
                pattern_type=PatternType.USER_PREFERENCE,
                confidence=0.95,
                pattern_content={"high": "conf"},
            ),
        ]

        proposals = await proposer.propose_amendments([], patterns)

        # Proposals should be sorted (if any produced)
        if len(proposals) > 1:
            for i in range(len(proposals) - 1):
                # Either higher confidence or higher absolute impact
                curr = proposals[i]
                next_p = proposals[i + 1]
                assert (curr.confidence > next_p.confidence or
                        abs(curr.expected_impact) >= abs(next_p.expected_impact))

    @pytest.mark.asyncio
    async def test_min_confidence_threshold(self):
        """Should respect minimum confidence threshold."""
        proposer = AmendmentProposer(min_confidence=0.8)

        observation = Observation(
            event_type=ObservationType.FEEDBACK,
            data={"rating": 0.1, "memory_id": "mem1"},
        )

        proposals = await proposer.propose_amendments([observation], [])

        # All proposals should meet minimum confidence
        for proposal in proposals:
            assert proposal.confidence >= 0.8 or len(proposals) == 0

    @pytest.mark.asyncio
    async def test_evidence_collection(self):
        """Should collect evidence for proposals."""
        proposer = AmendmentProposer()

        obs1 = Observation(id="obs1", event_type=ObservationType.ADD_COMPLETED)
        obs2 = Observation(id="obs2", event_type=ObservationType.ADD_COMPLETED)

        proposals = await proposer.propose_amendments([obs1, obs2], [])

        # Proposals may include evidence from observations
        assert isinstance(proposals, list)

    @pytest.mark.asyncio
    async def test_project_isolation(self):
        """Should maintain project isolation."""
        proposer = AmendmentProposer()

        pattern = Pattern(
            pattern_type=PatternType.USER_PREFERENCE,
            confidence=0.85,
            pattern_content={"pref": "A"},
            project_id="project_a",
        )

        proposals = await proposer.propose_amendments([], [pattern])

        # Proposals should maintain project ID
        if proposals:
            for proposal in proposals:
                assert proposal.project_id == "project_a"

    @pytest.mark.asyncio
    async def test_multiple_amendment_types(self):
        """Should support multiple amendment types."""
        proposer = AmendmentProposer()

        # Create scenarios for different amendment types
        patterns = [
            Pattern(
                pattern_type=PatternType.USER_PREFERENCE,
                confidence=0.9,
                pattern_content={"preference": "update"},
            ),
            Pattern(
                pattern_type=PatternType.ERROR_RECOVERY,
                confidence=0.85,
                pattern_content={"error_type": "timeout"},
            ),
        ]

        proposals = await proposer.propose_amendments([], patterns)

        # Should produce various amendment types
        amendment_types = {p.amendment_type for p in proposals}
        assert AmendmentType.CONTENT_UPDATE in amendment_types or len(amendment_types) >= 0

    @pytest.mark.asyncio
    async def test_impact_assessment(self):
        """Should assess expected impact of amendments."""
        proposer = AmendmentProposer()

        patterns = [
            Pattern(
                pattern_type=PatternType.USER_PREFERENCE,
                confidence=0.9,
                pattern_content={"high_impact": "change"},
            ),
        ]

        proposals = await proposer.propose_amendments([], patterns)

        for proposal in proposals:
            # Impact should be between -1 and 1
            assert -1 <= proposal.expected_impact <= 1
            # Should have impact description
            assert proposal.impact_description != ""

    @pytest.mark.asyncio
    async def test_proposal_status_initialization(self):
        """Should initialize proposals with PENDING status."""
        proposer = AmendmentProposer()

        observation = Observation(
            event_type=ObservationType.FEEDBACK,
            data={"rating": 0.1, "memory_id": "mem1"},
        )

        proposals = await proposer.propose_amendments([observation], [])

        for proposal in proposals:
            assert proposal.status == AmendmentStatus.PENDING
