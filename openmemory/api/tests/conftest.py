"""
Test configuration for OpenMemory API tests.
"""

import os
import pytest
from fastapi.testclient import TestClient

# Set test environment variables before importing app
os.environ["OPENAI_API_KEY"] = "test-key-for-testing"
os.environ["USER"] = "test_user"

from main import app


@pytest.fixture
def client():
    """Create a test client for the API."""
    return TestClient(app)


@pytest.fixture
def test_user_id():
    """Test user ID."""
    return "test_user"


@pytest.fixture
def test_agent_id():
    """Test agent ID."""
    return "test_agent"


@pytest.fixture
def mock_graph_data():
    """Mock graph data for testing."""
    return [
        {
            "source": "coffee",
            "relationship": "LIKES",
            "target": "yishu"
        },
        {
            "source": "yishu",
            "relationship": "WORKS_ON",
            "target": "openclaw"
        },
        {
            "source": "openclaw",
            "relationship": "USES",
            "target": "mem0"
        }
    ]


@pytest.fixture
def mock_graph_stats():
    """Mock graph statistics for testing."""
    return {
        "nodes": 15,
        "relations": 12,
        "relation_types": {
            "LIKES": 3,
            "WORKS_ON": 5,
            "USES": 2,
            "RELATED_TO": 2
        }
    }


@pytest.fixture
def mock_agents_list():
    """Mock agents list for testing."""
    return {
        "agents": [
            {"agent_id": "main", "memory_count": 150},
            {"agent_id": "planner", "memory_count": 45},
            {"agent_id": "researcher", "memory_count": 78}
        ]
    }
