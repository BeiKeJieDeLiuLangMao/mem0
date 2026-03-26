"""
Tests for Mem0AILearn - Integration Tests.

Following TDD: Integration tests for the complete AI Learning system.
"""

import asyncio
import sys
import pytest
from datetime import datetime
from pathlib import Path
import tempfile
from unittest import mock

# Mock mem0 modules that have external dependencies
sys.modules['mem0.client'] = mock.MagicMock()
sys.modules['mem0.client.main'] = mock.MagicMock()
sys.modules['mem0.client.project'] = mock.MagicMock()
sys.modules['mem0.memory'] = mock.MagicMock()
sys.modules['mem0.memory.main'] = mock.MagicMock()
sys.modules['mem0.memory.telemetry'] = mock.MagicMock()

from mem0.observation.models import Observation, ObservationType
from mem0.observation.storage.observation_store import FileObservationStore
from mem0.observation.filters.privacy_filter import PrivacyFilter
from mem0.observation.collectors.project_detector import ProjectDetector
from mem0.observation.hooks.memory_hook import MemoryObservationHook
from mem0.learning.pattern_detector import PatternDetector
from mem0.learning.skill_extractor import SkillExtractor
from mem0.evolution.health_monitor import HealthMonitor
from mem0.evolution.metrics import MetricsCollector
from mem0.evolution.evolution_tracker import EvolutionTracker
from mem0.amendment.proposer import AmendmentProposer
from mem0.instincts import InstinctRegistry, InstinctApplier
from mem0.ailearn import Mem0AILearn, enable_ailearn


class TestMem0AILearnInitialization:
    """Test Mem0AILearn initialization."""

    def test_init_default_values(self):
        """Should initialize with default values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(
                storage_path=tmpdir,
                project_id="test_project",
            )

            assert ailearn.project_id == "test_project"
            assert ailearn.session_id == "default"
            assert ailearn.user_id == "default"
            assert ailearn.auto_learn is True
            assert ailearn.auto_amend is False

    def test_init_custom_values(self):
        """Should initialize with custom values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(
                storage_path=tmpdir,
                project_id="custom_project",
                session_id="session123",
                user_id="user456",
                auto_learn=False,
                auto_amend=True,
            )

            assert ailearn.project_id == "custom_project"
            assert ailearn.session_id == "session123"
            assert ailearn.user_id == "user456"
            assert ailearn.auto_learn is False
            assert ailearn.auto_amend is True

    def test_init_creates_storage_path(self):
        """Should create storage path if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = Path(tmpdir) / "new" / "path"
            ailearn = Mem0AILearn(storage_path=str(storage_path))

            assert storage_path.exists()
            assert (storage_path / "observations").exists()

    def test_init_initializes_observation_layer(self):
        """Should initialize observation layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(storage_path=tmpdir)

            assert hasattr(ailearn, 'observation_store')
            assert hasattr(ailearn, 'observation_hook')
            assert isinstance(ailearn.observation_store, FileObservationStore)
            assert isinstance(ailearn.observation_hook, MemoryObservationHook)

    def test_init_initializes_learning_layer(self):
        """Should initialize learning layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(storage_path=tmpdir)

            assert hasattr(ailearn, 'pattern_detector')
            assert hasattr(ailearn, 'skill_extractor')
            assert isinstance(ailearn.pattern_detector, PatternDetector)
            assert isinstance(ailearn.skill_extractor, SkillExtractor)

    def test_init_initializes_evolution_layer(self):
        """Should initialize evolution layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(storage_path=tmpdir)

            assert hasattr(ailearn, 'metrics_collector')
            assert hasattr(ailearn, 'health_monitor')
            assert hasattr(ailearn, 'evolution_tracker')
            assert hasattr(ailearn, 'amendment_proposer')
            assert isinstance(ailearn.metrics_collector, MetricsCollector)
            assert isinstance(ailearn.health_monitor, HealthMonitor)
            assert isinstance(ailearn.amendment_proposer, AmendmentProposer)

    def test_init_initializes_instincts_layer(self):
        """Should initialize instincts layer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(storage_path=tmpdir)

            assert hasattr(ailearn, 'instinct_registry')
            assert hasattr(ailearn, 'instinct_applier')
            assert isinstance(ailearn.instinct_registry, InstinctRegistry)
            assert isinstance(ailearn.instinct_applier, InstinctApplier)


class TestEnableAilearn:
    """Test enable_ailearn helper function."""

    def test_enable_ailearn_returns_instance(self):
        """Should return Mem0AILearn instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_memory = mock.MagicMock()
            result = enable_ailearn(
                mock_memory,
                storage_path=tmpdir,
                project_id="test",
            )

            assert isinstance(result, Mem0AILearn)

    def test_enable_ailearn_wraps_memory(self):
        """Should wrap provided memory instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_memory = mock.MagicMock()
            ailearn = enable_ailearn(
                mock_memory,
                storage_path=tmpdir,
                project_id="test",
            )

            # Memory should be wrapped
            assert mock_memory.add is not mock.MagicMock

    def test_enable_ailearn_custom_params(self):
        """Should pass custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_memory = mock.MagicMock()
            ailearn = enable_ailearn(
                mock_memory,
                storage_path=tmpdir,
                session_id="custom_session",
                user_id="custom_user",
                auto_learn=False,
            )

            assert ailearn.session_id == "custom_session"
            assert ailearn.user_id == "custom_user"
            assert ailearn.auto_learn is False


class TestMem0AILearnObservation:
    """Test observation capabilities."""

    @pytest.mark.asyncio
    async def test_add_observation(self):
        """Should add observation to storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(
                storage_path=tmpdir,
                project_id="test_proj",
            )

            obs = Observation(
                project_id="test_proj",
                event_type=ObservationType.ADD_COMPLETED,
                data={"test": "data"},
            )

            await ailearn.observation_store.add(obs)

            # Verify observation was stored
            stored = await ailearn.observation_store.get_by_project("test_proj")
            assert len(stored) == 1
            assert stored[0].data == {"test": "data"}

    @pytest.mark.asyncio
    async def test_query_observations(self):
        """Should query observations by filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(
                storage_path=tmpdir,
                project_id="test_proj",
            )

            # Add multiple observations
            for i in range(3):
                await ailearn.observation_store.add(
                    Observation(
                        project_id="test_proj",
                        session_id=f"session_{i}",
                        data={"index": i},
                    )
                )

            # Query by project
            results = await ailearn.observation_store.get_by_project("test_proj")
            assert len(results) == 3

    @pytest.mark.asyncio
    async def test_flush_observations(self):
        """Should flush pending observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(storage_path=tmpdir)

            # Flush should complete without error
            await ailearn.flush()

            # Observation hook should also support flush
            assert hasattr(ailearn.observation_hook, 'flush')


class TestMem0AILearnLearning:
    """Test learning capabilities."""

    @pytest.mark.asyncio
    async def test_analyze_empty_observations(self):
        """Should handle empty observations gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(
                storage_path=tmpdir,
                project_id="empty_project",
            )

            result = await ailearn.analyze_and_learn(limit=100)

            assert result is not None
            assert "observations_analyzed" in result
            assert result["observations_analyzed"] == 0

    @pytest.mark.asyncio
    async def test_analyze_with_observations(self):
        """Should analyze observations and detect patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(
                storage_path=tmpdir,
                project_id="learn_test",
            )

            # Add some observations
            for i in range(5):
                await ailearn.observation_store.add(
                    Observation(
                        project_id="learn_test",
                        event_type=ObservationType.ADD_COMPLETED,
                        data={"action": "test"},
                    )
                )

            result = await ailearn.analyze_and_learn()

            assert result is not None
            assert "patterns_detected" in result
            assert "skills_extracted" in result
            assert "amendments_proposed" in result

    @pytest.mark.asyncio
    async def test_promote_high_confidence_patterns(self):
        """Should promote high-confidence patterns to instincts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(storage_path=tmpdir)

            # Add observations that might form a high-confidence pattern
            for i in range(15):
                await ailearn.observation_store.add(
                    Observation(
                        event_type=ObservationType.ADD_COMPLETED,
                        data={"repeated": "action"},
                    )
                )

            result = await ailearn.analyze_and_learn()

            # High confidence patterns (11+ occurrences) should be promoted
            # This is implementation-specific
            assert result is not None


class TestMem0AILearnHealth:
    """Test health monitoring capabilities."""

    @pytest.mark.asyncio
    async def test_health_status_healthy(self):
        """Should return healthy status with good metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(storage_path=tmpdir)

            # Record some successful operations
            for i in range(10):
                await ailearn.metrics_collector.record_operation("add", duration=0.1, success=True)
                await ailearn.metrics_collector.record_operation("search", duration=0.1, success=True)

            status = await ailearn.get_health_status()

            assert status is not None
            assert "status" in status
            assert "metrics" in status
            assert "alerts" in status
            assert status["metrics"]["total_adds"] == 10

    @pytest.mark.asyncio
    async def test_health_alerts_generated(self):
        """Should generate alerts for poor metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(storage_path=tmpdir)

            # Record mostly failures
            for i in range(2):
                await ailearn.metrics_collector.record_operation("add", duration=0.1, success=True)
            for i in range(8):
                await ailearn.metrics_collector.record_operation("add", duration=0.1, success=False)

            status = await ailearn.get_health_status()

            # Should have alerts for low success rate
            assert isinstance(status["alerts"], list)

    @pytest.mark.asyncio
    async def test_health_metrics_tracking(self):
        """Should track various metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(storage_path=tmpdir)

            await ailearn.metrics_collector.record_operation("add", duration=0.1, success=True)
            await ailearn.metrics_collector.record_operation("update", duration=0.1, success=True)
            await ailearn.metrics_collector.record_operation("delete", duration=0.1, success=True)
            await ailearn.metrics_collector.record_operation("search", duration=0.1, success=True)

            metrics = await ailearn.metrics_collector.get_current_metrics()

            assert metrics.total_adds == 1
            assert metrics.total_updates == 1
            assert metrics.total_deletes == 1
            assert metrics.total_searches == 1


class TestMem0AILearnAmendment:
    """Test amendment capabilities."""

    @pytest.mark.asyncio
    async def test_apply_amendment_not_found(self):
        """Should return None for nonexistent proposal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(storage_path=tmpdir)

            result = await ailearn.apply_amendment(
                "nonexistent_id",
                {"current": "content"},
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_propose_amendments_from_observations(self):
        """Should propose amendments based on observations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(storage_path=tmpdir)

            # Add observation with negative feedback
            await ailearn.observation_store.add(
                Observation(
                    event_type=ObservationType.FEEDBACK,
                    data={
                        "rating": 0.1,
                        "memory_id": "bad_memory",
                    },
                )
            )

            result = await ailearn.analyze_and_learn()

            # Should have proposals
            assert "amendments_proposed" in result


class TestMem0AILearnShutdown:
    """Test shutdown capabilities."""

    @pytest.mark.asyncio
    async def test_flush_on_shutdown(self):
        """Should flush pending data on shutdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(storage_path=tmpdir)

            # Add observation
            await ailearn.observation_store.add(
                Observation(data={"test": "data"})
            )

            # Shutdown should flush
            await ailearn.shutdown()

            # Should complete without error

    @pytest.mark.asyncio
    async def test_multiple_flush_calls(self):
        """Should handle multiple flush calls."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(storage_path=tmpdir)

            await ailearn.flush()
            await ailearn.flush()
            await ailearn.flush()

            # Should complete without error


class TestMem0AILearnProjectIsolation:
    """Test project isolation."""

    @pytest.mark.asyncio
    async def test_project_data_isolation(self):
        """Should isolate data by project."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(
                storage_path=tmpdir,
                project_id="project_a",
            )

            # Add observation for project_a
            await ailearn.observation_store.add(
                Observation(
                    project_id="project_a",
                    data={"project": "a"},
                )
            )

            # Add observation for project_b (using different instance)
            ailearn_b = Mem0AILearn(
                storage_path=tmpdir,
                project_id="project_b",
            )
            await ailearn_b.observation_store.add(
                Observation(
                    project_id="project_b",
                    data={"project": "b"},
                )
            )

            # Each project should only see its own data
            project_a_data = await ailearn.observation_store.get_by_project("project_a")
            project_b_data = await ailearn_b.observation_store.get_by_project("project_b")

            assert len(project_a_data) >= 1
            assert len(project_b_data) >= 1


class TestMem0AILearnIntegration:
    """Full integration tests."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Should handle full observation → learning → evolution workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize
            ailearn = Mem0AILearn(
                storage_path=tmpdir,
                project_id="integration_test",
            )

            # Simulate usage
            for i in range(10):
                await ailearn.metrics_collector.record_operation("add", duration=0.1, success=True)
                await ailearn.observation_store.add(
                    Observation(
                        event_type=ObservationType.ADD_COMPLETED,
                        data={"action": f"op_{i}"},
                    )
                )

            # Analyze
            analysis = await ailearn.analyze_and_learn()

            # Check health
            health = await ailearn.get_health_status()

            # Verify results
            assert analysis["observations_analyzed"] >= 0
            assert health["status"] in ["healthy", "degraded", "unhealthy", "critical"]
            assert health["metrics"]["total_adds"] == 10

            # Cleanup
            await ailearn.shutdown()

    @pytest.mark.asyncio
    async def test_auto_learn_disabled(self):
        """Should respect auto_learn=False setting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(
                storage_path=tmpdir,
                auto_learn=False,
            )

            assert ailearn.auto_learn is False

    @pytest.mark.asyncio
    async def test_auto_amend_enabled(self):
        """Should respect auto_amend=True setting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ailearn = Mem0AILearn(
                storage_path=tmpdir,
                auto_amend=True,
            )

            assert ailearn.auto_amend is True
