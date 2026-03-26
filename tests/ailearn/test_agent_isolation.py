"""
Tests for Agent Isolation (project_id → agent_id).

Tests that all models support agent_id for OpenClaw agent-level isolation.
"""

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


class TestPatternAgentId:
    """Pattern must have agent_id field."""

    def test_pattern_has_agent_id_field(self):
        """Pattern dataclass must have agent_id field."""
        from mem0.learning.pattern import Pattern
        p = Pattern()
        assert hasattr(p, 'agent_id')
        assert p.agent_id == "default"

    def test_pattern_agent_id_in_to_dict(self):
        """to_dict() must include agent_id."""
        from mem0.learning.pattern import Pattern
        p = Pattern(agent_id="my-agent")
        d = p.to_dict()
        assert "agent_id" in d
        assert d["agent_id"] == "my-agent"

    def test_pattern_agent_id_in_from_dict(self):
        """from_dict() must restore agent_id."""
        from mem0.learning.pattern import Pattern
        data = {
            "id": "p1",
            "pattern_type": "workflow_sequence",
            "name": "Test",
            "description": "",
            "extracted_at": datetime.utcnow().isoformat(),
            "evidence": {
                "observation_ids": [],
                "example_count": 0,
                "confidence_distribution": [],
                "context_snippets": [],
            },
            "confidence": 0.5,
            "frequency": 1,
            "trigger_condition": {},
            "pattern_content": {},
            "expected_outcome": None,
            "project_id": "global",
            "user_id": "default",
            "agent_id": "my-agent",
            "tags": [],
            "metadata": {},
        }
        p = Pattern.from_dict(data)
        assert p.agent_id == "my-agent"


class TestSkillCandidateAgentId:
    """SkillCandidate must have agent_id field."""

    def test_skill_candidate_has_agent_id_field(self):
        """SkillCandidate dataclass must have agent_id field."""
        from mem0.learning.pattern import SkillCandidate
        sc = SkillCandidate()
        assert hasattr(sc, 'agent_id')
        assert sc.agent_id == "default"

    def test_skill_candidate_agent_id_in_to_dict(self):
        """to_dict() must include agent_id."""
        from mem0.learning.pattern import SkillCandidate
        sc = SkillCandidate(name="TestSkill", agent_id="agent-123")
        d = sc.to_dict()
        assert "agent_id" in d
        assert d["agent_id"] == "agent-123"

    def test_skill_candidate_agent_id_in_from_dict(self):
        """from_dict() must restore agent_id."""
        from mem0.learning.pattern import SkillCandidate
        data = {
            "id": "s1",
            "name": "TestSkill",
            "description": "",
            "trigger_phrases": [],
            "extracted_at": datetime.utcnow().isoformat(),
            "source_pattern_ids": [],
            "confidence": 0.7,
            "instructions": "Do X",
            "examples": [],
            "project_id": "global",
            "agent_id": "agent-456",
            "tags": [],
        }
        sc = SkillCandidate.from_dict(data)
        assert sc.agent_id == "agent-456"


class TestInstinctAgentId:
    """Instinct must have agent_id field."""

    def test_instinct_has_agent_id_field(self):
        """Instinct dataclass must have agent_id field."""
        from mem0.instincts.instincts import Instinct
        i = Instinct()
        assert hasattr(i, 'agent_id')
        assert i.agent_id == "default"

    def test_instinct_has_decay_fields(self):
        """Instinct must have decay-related fields."""
        from mem0.instincts.instincts import Instinct
        i = Instinct()
        assert hasattr(i, 'last_confirmed_at')
        assert hasattr(i, 'last_contradicted_at')
        assert hasattr(i, 'decay_rate')

    def test_instinct_decay_fields_in_to_dict(self):
        """to_dict() must include decay fields."""
        from mem0.instincts.instincts import Instinct
        now = datetime.utcnow()
        i = Instinct(
            name="TestInstinct",
            agent_id="decay-agent",
            last_confirmed_at=now,
            decay_rate=0.05,
        )
        d = i.to_dict()
        assert "agent_id" in d
        assert "last_confirmed_at" in d
        assert "decay_rate" in d
        assert d["agent_id"] == "decay-agent"
        assert d["decay_rate"] == 0.05

    def test_instinct_decay_fields_in_from_dict(self):
        """from_dict() must restore decay fields."""
        from mem0.instincts.instincts import Instinct
        now = datetime.utcnow()
        data = {
            "id": "i1",
            "name": "TestInstinct",
            "description": "",
            "instinct_type": "workflow_optimization",
            "trigger": {"action_type": "modify_args", "content": {}},
            "action": {"action_type": "modify_args", "content": {}},
            "source_pattern_id": "",
            "confidence": 0.8,
            "observation_count": 5,
            "times_applied": 0,
            "times_successful": 0,
            "success_rate": 0.0,
            "created_at": now.isoformat(),
            "project_id": "global",
            "user_id": "default",
            "agent_id": "decay-agent",
            "enabled": True,
            "last_confirmed_at": now.isoformat(),
            "last_contradicted_at": None,
            "decay_rate": 0.02,
        }
        i = Instinct.from_dict(data)
        assert i.agent_id == "decay-agent"
        assert i.decay_rate == 0.02


class TestInstinctRegistryAgentId:
    """InstinctRegistry must support agent_id filtering."""

    def test_get_instincts_by_agent_id(self):
        """get_instincts must support agent_id parameter."""
        from mem0.instincts.instincts import Instinct, InstinctRegistry, InstinctAction
        registry = InstinctRegistry()

        instinct_a = Instinct(
            name="InstA",
            agent_id="agent-a",
            confidence=1.0,
            trigger=InstinctAction("modify_args", {}),
        )
        instinct_b = Instinct(
            name="InstB",
            agent_id="agent-b",
            confidence=1.0,
            trigger=InstinctAction("modify_args", {}),
        )

        import asyncio
        asyncio.run(registry.register_instinct(instinct_a))
        asyncio.run(registry.register_instinct(instinct_b))

        # Should filter by agent_id
        agent_a_instincts = registry.get_instincts_by_agent_id("agent-a")
        assert len(agent_a_instincts) == 1
        assert agent_a_instincts[0].name == "InstA"


class TestAmendmentProposalAgentId:
    """AmendmentProposal must have agent_id field."""

    def test_amendment_proposal_has_agent_id_field(self):
        """AmendmentProposal must have agent_id field."""
        from mem0.amendment.models import AmendmentProposal
        a = AmendmentProposal()
        assert hasattr(a, 'agent_id')
        assert a.agent_id == "default"

    def test_amendment_proposal_agent_id_in_to_dict(self):
        """to_dict() must include agent_id."""
        from mem0.amendment.models import AmendmentProposal
        a = AmendmentProposal(agent_id="amend-agent")
        d = a.to_dict()
        assert "agent_id" in d
        assert d["agent_id"] == "amend-agent"

    def test_amendment_proposal_agent_id_in_from_dict(self):
        """from_dict() must restore agent_id."""
        from mem0.amendment.models import AmendmentProposal, AmendmentStatus
        data = {
            "id": "a1",
            "amendment_type": "content_update",
            "memory_id": "m1",
            "current_content": {},
            "proposed_change": {},
            "reasoning": "Needs update",
            "evidence_observation_ids": [],
            "confidence": 0.6,
            "expected_impact": 0.1,
            "impact_description": "",
            "created_at": datetime.utcnow().isoformat(),
            "created_by": "system",
            "project_id": "global",
            "agent_id": "amend-agent",
            "status": "pending",
            "applied_at": None,
            "applied_version_id": None,
        }
        a = AmendmentProposal.from_dict(data)
        assert a.agent_id == "amend-agent"


class TestAmendmentBatchAgentId:
    """AmendmentBatch must have agent_id field."""

    def test_amendment_batch_has_agent_id_field(self):
        """AmendmentBatch must have agent_id field."""
        from mem0.amendment.models import AmendmentBatch
        b = AmendmentBatch()
        assert hasattr(b, 'agent_id')
        assert b.agent_id == "default"

    def test_amendment_batch_agent_id_in_to_dict(self):
        """to_dict() must include agent_id."""
        from mem0.amendment.models import AmendmentBatch
        b = AmendmentBatch(agent_id="batch-agent")
        d = b.to_dict()
        assert "agent_id" in d
        assert d["agent_id"] == "batch-agent"


class TestObservationAgentId:
    """Observation must have agent_id field."""

    def test_observation_has_agent_id_field(self):
        """Observation must have agent_id field."""
        from mem0.observation.models import Observation
        o = Observation()
        assert hasattr(o, 'agent_id')
        assert o.agent_id == "default"

    def test_observation_agent_id_in_to_dict(self):
        """to_dict() must include agent_id."""
        from mem0.observation.models import Observation
        o = Observation(agent_id="obs-agent")
        d = o.to_dict()
        assert "agent_id" in d
        assert d["agent_id"] == "obs-agent"

    def test_observation_agent_id_in_from_dict(self):
        """from_dict() must restore agent_id."""
        from mem0.observation.models import Observation, ObservationType
        data = {
            "id": "o1",
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "ADD_COMPLETED",
            "project_id": "global",
            "agent_id": "obs-agent",
            "session_id": "default",
            "user_id": "default",
            "data": {},
            "metadata": {},
            "confidence": 0.9,
            "redacted": False,
        }
        o = Observation.from_dict(data)
        assert o.agent_id == "obs-agent"


class TestEvolutionTrackerAmendmentProposalAgentId:
    """EvolutionTracker's AmendmentProposal must have agent_id."""

    def test_evolution_tracker_amendment_has_agent_id(self):
        """EvolutionTracker.AmendmentProposal must have agent_id."""
        from mem0.evolution.evolution_tracker import AmendmentProposal
        a = AmendmentProposal()
        assert hasattr(a, 'agent_id')
        assert a.agent_id == "default"

    def test_evolution_tracker_propose_with_agent_id(self):
        """propose_amendment must accept agent_id parameter."""
        from mem0.evolution.evolution_tracker import EvolutionTracker
        tracker = EvolutionTracker()

        import asyncio
        proposal = asyncio.run(tracker.propose_amendment(
            memory_id="m1",
            proposed_change={"content": "new"},
            reasoning="test",
            evidence=[],
            confidence=0.7,
            expected_impact=0.1,
            agent_id="evo-agent",
        ))
        assert proposal.agent_id == "evo-agent"

    def test_evolution_tracker_get_pending_by_agent_id(self):
        """get_pending_proposals must support agent_id filtering."""
        from mem0.evolution.evolution_tracker import EvolutionTracker
        tracker = EvolutionTracker()

        import asyncio
        asyncio.run(tracker.propose_amendment(
            memory_id="m1",
            proposed_change={},
            reasoning="r1",
            evidence=[],
            confidence=0.7,
            expected_impact=0.1,
            agent_id="evo-agent-a",
        ))
        asyncio.run(tracker.propose_amendment(
            memory_id="m2",
            proposed_change={},
            reasoning="r2",
            evidence=[],
            confidence=0.8,
            expected_impact=0.1,
            agent_id="evo-agent-b",
        ))

        pending = tracker.get_pending_proposals(agent_id="evo-agent-a")
        assert len(pending) == 1
        assert pending[0].agent_id == "evo-agent-a"


class TestFileObservationStoreAgentsPath:
    """FileObservationStore must use agents/ path prefix."""

    def test_store_uses_agents_path(self):
        """Storage path must use agents/ prefix, not projects/."""
        import tempfile
        from pathlib import Path
        from mem0.observation.storage.observation_store import FileObservationStore

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileObservationStore(tmpdir)
            path = store._get_agent_path("my-agent")

            # Path must contain "agents" not "projects"
            assert "agents" in str(path)
            assert "projects" not in str(path)
            assert "my-agent" in str(path)
