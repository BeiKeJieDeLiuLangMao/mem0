"""
Tests for Observation Layer.

Following TDD: Test-first approach for observation capture.
"""

import asyncio
import sys
import pytest
from datetime import datetime
from pathlib import Path
import tempfile
from unittest import mock

# Mock mem0 modules that have external dependencies to avoid import errors
sys.modules['mem0.client'] = mock.MagicMock()
sys.modules['mem0.client.main'] = mock.MagicMock()
sys.modules['mem0.client.project'] = mock.MagicMock()
sys.modules['mem0.memory'] = mock.MagicMock()
sys.modules['mem0.memory.main'] = mock.MagicMock()
sys.modules['mem0.memory.telemetry'] = mock.MagicMock()

from mem0.observation.models import (
    Observation,
    ObservationType,
    ProjectInfo,
)
from mem0.observation.filters.privacy_filter import PrivacyFilter
from mem0.observation.storage.observation_store import FileObservationStore
from mem0.observation.storage.buffer import ObservationBuffer


class TestObservationModel:
    """Test Observation data model."""

    def test_create_observation_with_defaults(self):
        """Should create observation with default values."""
        obs = Observation()

        assert obs.id is not None
        assert isinstance(obs.timestamp, datetime)
        assert obs.event_type == ObservationType.ADD_INITIATED
        assert obs.project_id == "global"
        assert obs.session_id == "default"
        assert obs.user_id == "default"
        assert obs.data == {}
        assert obs.metadata == {}
        assert obs.confidence == 0.0
        assert obs.redacted is False

    def test_create_observation_with_values(self):
        """Should create observation with specified values."""
        now = datetime.utcnow()
        obs = Observation(
            event_type=ObservationType.ADD_COMPLETED,
            project_id="test_project",
            session_id="test_session",
            user_id="test_user",
            data={"key": "value"},
            confidence=0.85,
            redacted=True,
        )

        assert obs.event_type == ObservationType.ADD_COMPLETED
        assert obs.project_id == "test_project"
        assert obs.session_id == "test_session"
        assert obs.user_id == "test_user"
        assert obs.data == {"key": "value"}
        assert obs.confidence == 0.85
        assert obs.redacted is True

    def test_observation_to_dict(self):
        """Should convert observation to dictionary."""
        obs = Observation(
            event_type=ObservationType.SEARCH_INITIATED,
            data={"query": "test"},
        )

        result = obs.to_dict()

        assert result["event_type"] == "SEARCH_INITIATED"
        assert result["data"] == {"query": "test"}
        assert "timestamp" in result
        assert "id" in result

    def test_observation_from_dict(self):
        """Should create observation from dictionary."""
        data = {
            "id": "test-id",
            "timestamp": "2024-01-01T00:00:00",
            "event_type": "ADD_COMPLETED",
            "project_id": "proj1",
            "session_id": "sess1",
            "user_id": "user1",
            "data": {"test": "data"},
            "metadata": {},
            "confidence": 0.9,
            "redacted": False,
        }

        obs = Observation.from_dict(data)

        assert obs.id == "test-id"
        assert obs.event_type == ObservationType.ADD_COMPLETED
        assert obs.project_id == "proj1"
        assert obs.confidence == 0.9

    def test_observation_to_jsonl(self):
        """Should convert observation to JSONL format."""
        obs = Observation(data={"key": "value"})

        jsonl = obs.to_jsonl()

        assert isinstance(jsonl, str)
        assert '"key": "value"' in jsonl or '"key":"value"' in jsonl


class TestProjectInfo:
    """Test ProjectInfo model."""

    def test_create_project_info(self):
        """Should create project info."""
        info = ProjectInfo(
            project_id="abc123",
            project_name="test-project",
            git_remote_url="https://github.com/test/repo",
            git_branch="main",
        )

        assert info.project_id == "abc123"
        assert info.project_name == "test-project"
        assert info.git_remote_url == "https://github.com/test/repo"
        assert info.git_branch == "main"

    def test_project_info_to_dict(self):
        """Should convert to dictionary."""
        info = ProjectInfo(
            project_id="xyz",
            project_name="my-project",
        )

        result = info.to_dict()

        assert result["project_id"] == "xyz"
        assert result["project_name"] == "my-project"
        assert "detected_at" in result


class TestPrivacyFilter:
    """Test PII detection and redaction."""

    def test_redact_email(self):
        """Should redact email addresses."""
        filter = PrivacyFilter()
        text = "Contact user@example.com for support"

        result = filter.redact(text)

        assert "[REDACTED_EMAIL]" in result
        assert "user@example.com" not in result

    def test_redact_phone(self):
        """Should redact phone numbers."""
        filter = PrivacyFilter()
        text = "Call me at (555) 123-4567"

        result = filter.redact(text)

        assert "[REDACTED_PHONE]" in result
        assert "(555) 123-4567" not in result

    def test_redact_ssn(self):
        """Should redact SSNs."""
        filter = PrivacyFilter()
        text = "SSN: 123-45-6789"

        result = filter.redact(text)

        assert "[REDACTED_SSN]" in result
        assert "123-45-6789" not in result

    def test_redact_api_key(self):
        """Should redact API keys."""
        filter = PrivacyFilter()
        text = "api_key=sk-1234567890abcdef"

        result = filter.redact(text)

        assert "[REDACTED_API_KEY]" in result
        assert "sk-1234567890abcdef" not in result

    def test_redact_bearer_token(self):
        """Should redact bearer tokens."""
        filter = PrivacyFilter()
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"

        result = filter.redact(text)

        assert "[REDACTED_TOKEN]" in result
        assert "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result

    def test_redact_ip_address(self):
        """Should redact IP addresses."""
        filter = PrivacyFilter()
        text = "Server at 192.168.1.1 is down"

        result = filter.redact(text)

        assert "[REDACTED_IP]" in result
        assert "192.168.1.1" not in result

    def test_scan_string(self):
        """Should scan string and return redaction status."""
        filter = PrivacyFilter()

        has_pii, redacted = filter.scan("Email: test@example.com")

        assert has_pii is True
        assert "[REDACTED_EMAIL]" in redacted

    def test_scan_dict(self):
        """Should scan dictionary and redact PII."""
        filter = PrivacyFilter()
        data = {
            "email": "user@example.com",
            "name": "John Doe",
        }

        has_pii, redacted = filter.scan(data)

        assert has_pii is True
        assert redacted["email"] == "[REDACTED_EMAIL]"
        assert redacted["name"] == "John Doe"

    def test_scan_list(self):
        """Should scan list and redact PII."""
        filter = PrivacyFilter()
        data = ["user@example.com", "normal text", "(555) 123-4567"]

        has_pii, redacted = filter.scan(data)

        assert has_pii is True
        assert isinstance(redacted, list)
        assert "[REDACTED_EMAIL]" in redacted[0]
        assert "[REDACTED_PHONE]" in redacted[2]

    def test_scan_no_pii(self):
        """Should return unchanged data when no PII found."""
        filter = PrivacyFilter()
        data = {"message": "Hello world"}

        has_pii, redacted = filter.scan(data)

        assert has_pii is False
        assert redacted == data


class TestObservationBuffer:
    """Test observation buffering."""

    @pytest.mark.asyncio
    async def test_buffer_initialization(self):
        """Should initialize buffer with default settings."""
        buffer = ObservationBuffer()

        assert buffer.size() == 0
        assert buffer.flush_size == 1000

    @pytest.mark.asyncio
    async def test_add_observation(self):
        """Should add observation to buffer."""
        buffer = ObservationBuffer()
        obs = Observation()

        flushed = await buffer.add(obs)

        assert buffer.size() == 1
        assert flushed is False  # Not at flush threshold

    @pytest.mark.asyncio
    async def test_auto_flush_on_size(self):
        """Should auto-flush when reaching flush_size."""
        flush_called = []

        async def mock_storage(observations):
            flush_called.append(observations)

        buffer = ObservationBuffer(
            flush_size=3,
            storage_backend=mock_storage,
        )

        # Add observations up to flush size
        for i in range(3):
            await buffer.add(Observation(data={"index": i}))

        assert len(flush_called) == 1
        assert len(flush_called[0]) == 3
        assert buffer.size() == 0

    @pytest.mark.asyncio
    async def test_manual_flush(self):
        """Should manually flush buffer."""
        flush_called = []

        async def mock_storage(observations):
            flush_called.append(observations)

        buffer = ObservationBuffer(storage_backend=mock_storage)

        await buffer.add(Observation())
        await buffer.add(Observation())

        await buffer.flush()

        assert len(flush_called) == 1
        assert len(flush_called[0]) == 2

    @pytest.mark.asyncio
    async def test_get_all_buffered(self):
        """Should get all buffered observations without clearing."""
        buffer = ObservationBuffer()

        obs1 = Observation(data={"id": 1})
        obs2 = Observation(data={"id": 2})

        await buffer.add(obs1)
        await buffer.add(obs2)

        all_obs = await buffer.get_all()

        assert len(all_obs) == 2
        assert buffer.size() == 2  # Should not clear


class TestFileObservationStore:
    """Test file-based observation storage."""

    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileObservationStore(tmpdir)
            yield store

    @pytest.mark.asyncio
    async def test_add_observation(self, temp_storage):
        """Should store observation to file."""
        obs = Observation(
            project_id="test_proj",
            data={"test": "data"},
        )

        await temp_storage.add(obs)

        # Verify file was created
        project_path = temp_storage._get_project_path("test_proj")
        assert project_path.exists()

    @pytest.mark.asyncio
    async def test_add_batch_observations(self, temp_storage):
        """Should store multiple observations atomically."""
        observations = [
            Observation(project_id="proj1", data={"id": 1}),
            Observation(project_id="proj1", data={"id": 2}),
            Observation(project_id="proj2", data={"id": 3}),
        ]

        await temp_storage.add_batch(observations)

        # Check project 1
        proj1_obs = await temp_storage.get_by_project("proj1")
        assert len(proj1_obs) == 2

        # Check project 2
        proj2_obs = await temp_storage.get_by_project("proj2")
        assert len(proj2_obs) == 1

    @pytest.mark.asyncio
    async def test_get_by_project(self, temp_storage):
        """Should retrieve observations by project."""
        obs1 = Observation(project_id="proj_a", data={"seq": 1})
        obs2 = Observation(project_id="proj_a", data={"seq": 2})
        obs3 = Observation(project_id="proj_b", data={"seq": 1})

        await temp_storage.add(obs1)
        await temp_storage.add(obs2)
        await temp_storage.add(obs3)

        results = await temp_storage.get_by_project("proj_a")

        assert len(results) == 2
        assert results[0].data["seq"] == 1
        assert results[1].data["seq"] == 2

    @pytest.mark.asyncio
    async def test_get_by_project_with_limit(self, temp_storage):
        """Should limit results when specified."""
        for i in range(10):
            await temp_storage.add(
                Observation(project_id="proj_x", data={"index": i})
            )

        results = await temp_storage.get_by_project("proj_x", limit=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_get_nonexistent_project(self, temp_storage):
        """Should return empty list for non-existent project."""
        results = await temp_storage.get_by_project("nonexistent")

        assert results == []

    @pytest.mark.asyncio
    async def test_query_observations(self, temp_storage):
        """Should query observations with filters."""
        await temp_storage.add(
            Observation(
                project_id="proj1",
                session_id="sess1",
                user_id="user1",
                data={"value": 1},
            )
        )
        await temp_storage.add(
            Observation(
                project_id="proj1",
                session_id="sess2",
                user_id="user1",
                data={"value": 2},
            )
        )

        # Query by session
        results = await temp_storage.query({
            "session_id": "sess1",
        })

        assert len(results) == 1
        assert results[0].session_id == "sess1"

    @pytest.mark.asyncio
    async def test_iterate_observations(self, temp_storage):
        """Should stream observations efficiently."""
        observations = []
        for i in range(5):
            obs = Observation(
                project_id="iter_proj",
                data={"index": i},
            )
            observations.append(obs)
            await temp_storage.add(obs)

        streamed = []
        async for obs in temp_storage.iterate("iter_proj"):
            streamed.append(obs)

        assert len(streamed) == 5

    @pytest.mark.asyncio
    async def test_iterate_all_projects(self, temp_storage):
        """Should iterate across all projects."""
        await temp_storage.add(Observation(project_id="p1", data={}))
        await temp_storage.add(Observation(project_id="p2", data={}))
        await temp_storage.add(Observation(project_id="p1", data={}))

        all_obs = []
        async for obs in temp_storage.iterate():
            all_obs.append(obs)

        assert len(all_obs) == 3
