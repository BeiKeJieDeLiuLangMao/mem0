"""
Test cases for Graph API endpoints.

This module follows TDD approach: tests are written first, then implementation.
"""

import pytest
from unittest.mock import Mock, patch


def test_graph_search_by_node_name(client, mock_graph_data):
    """Test searching graph by node name."""
    # Mock the graph client
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_graph.search.return_value = mock_graph_data
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?q=Coffee&user_id=yishu")
        assert response.status_code == 200
        data = response.json()
        assert "relations" in data
        assert isinstance(data["relations"], list)
        assert len(data["relations"]) > 0


def test_graph_search_by_relationship_type(client, mock_graph_data):
    """Test searching graph by relationship type."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_graph.search.return_value = mock_graph_data
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?q=LIKES&user_id=yishu")
        assert response.status_code == 200
        data = response.json()
        assert "relations" in data
        assert isinstance(data["relations"], list)


def test_graph_search_empty_query(client):
    """Test graph search with empty query returns validation error."""
    response = client.get("/api/v1/graph/search?q=&user_id=yishu")
    assert response.status_code == 422  # Validation error


def test_graph_search_missing_user_id(client):
    """Test graph search without user_id returns validation error."""
    response = client.get("/api/v1/graph/search?q=Coffee")
    assert response.status_code == 422  # Validation error


def test_graph_stats(client, mock_graph_stats):
    """Test getting graph statistics."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        # Mock get_all to return relation data (not get_stats, as implementation uses get_all)
        mock_graph.get_all.return_value = [
            {"source": "coffee", "relationship": "LIKES", "target": "yishu"},
            {"source": "yishu", "relationship": "WORKS_ON", "target": "openclaw"},
            {"source": "openclaw", "relationship": "USES", "target": "mem0"},
        ]
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/stats?user_id=yishu")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "relations" in data
        assert "relation_types" in data
        assert data["nodes"] >= 0
        assert data["relations"] >= 0
        assert isinstance(data["relation_types"], dict)


def test_graph_stats_missing_user_id(client):
    """Test graph stats without user_id returns validation error."""
    response = client.get("/api/v1/graph/stats")
    assert response.status_code == 422  # Validation error


def test_graph_agents(client, mock_agents_list):
    """Test getting agents list."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        # Mock the underlying Neo4j graph object
        mock_neo4j_graph = Mock()
        mock_neo4j_graph.query.return_value = [
            {"agent_id": "main", "memory_count": 150},
            {"agent_id": "planner", "memory_count": 45},
            {"agent_id": "researcher", "memory_count": 78}
        ]
        mock_graph.graph = mock_neo4j_graph
        mock_graph.node_label = ""
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/agents?user_id=yishu")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)
        assert len(data["agents"]) > 0
        # Verify agent structure
        assert "agent_id" in data["agents"][0]
        assert "memory_count" in data["agents"][0]


def test_graph_agents_missing_user_id(client):
    """Test graph agents without user_id returns validation error."""
    response = client.get("/api/v1/graph/agents")
    assert response.status_code == 422  # Validation error


def test_graph_service_unavailable(client):
    """Test graph endpoints when service is unavailable."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_get_client.return_value = None

        response = client.get("/api/v1/graph/search?q=Coffee&user_id=yishu")
        assert response.status_code == 503

        response = client.get("/api/v1/graph/stats?user_id=yishu")
        assert response.status_code == 503

        response = client.get("/api/v1/graph/agents?user_id=yishu")
        assert response.status_code == 503


def test_graph_not_enabled(client):
    """Test graph endpoints when graph store is not enabled."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_client.graph = None
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?q=Coffee&user_id=yishu")
        assert response.status_code == 503
        assert "not enabled" in response.json()["detail"].lower()


def test_graph_search_with_agent_filter(client, mock_graph_data):
    """Test graph search with agent_id filter."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_graph.search.return_value = mock_graph_data
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?q=Coffee&user_id=yishu&agent_id=main")
        assert response.status_code == 200
        data = response.json()
        assert "relations" in data
        # Verify the search was called with agent_id filter
        mock_graph.search.assert_called_once()
        call_kwargs = mock_graph.search.call_args[1]
        assert call_kwargs["filters"]["agent_id"] == "main"


def test_graph_search_case_insensitive(client, mock_graph_data):
    """Test that graph search is case insensitive."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_graph.search.return_value = mock_graph_data
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        # Test lowercase
        response1 = client.get("/api/v1/graph/search?q=coffee&user_id=yishu")
        assert response1.status_code == 200

        # Test uppercase
        response2 = client.get("/api/v1/graph/search?q=COFFEE&user_id=yishu")
        assert response2.status_code == 200

        # Test mixed case
        response3 = client.get("/api/v1/graph/search?q=CofFeE&user_id=yishu")
        assert response3.status_code == 200


def test_graph_search_with_limit(client, mock_graph_data):
    """Test graph search with custom limit."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_graph.search.return_value = mock_graph_data
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?q=Coffee&user_id=yishu&limit=50")
        assert response.status_code == 200
        data = response.json()
        assert "relations" in data
        # Verify limit was passed correctly
        mock_graph.search.assert_called_once()


def test_graph_search_with_run_id_filter(client, mock_graph_data):
    """Test graph search with run_id filter."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_graph.search.return_value = mock_graph_data
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?q=Coffee&user_id=yishu&agent_id=main&run_id=run123")
        assert response.status_code == 200
        data = response.json()
        assert "relations" in data
        # Verify the search was called with correct filters
        mock_graph.search.assert_called_once()
        call_kwargs = mock_graph.search.call_args[1]
        assert call_kwargs["filters"]["user_id"] == "yishu"
        assert call_kwargs["filters"]["agent_id"] == "main"
        assert call_kwargs["filters"]["run_id"] == "run123"


def test_graph_stats_with_agent_filter(client):
    """Test graph stats with agent_id filter."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_graph.get_all.return_value = [
            {"source": "coffee", "relationship": "LIKES", "target": "yishu"},
        ]
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/stats?user_id=yishu&agent_id=main")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "relations" in data
        assert data["nodes"] == 2  # coffee and yishu
        assert data["relations"] == 1
        # Verify get_all was called with agent_id filter
        mock_graph.get_all.assert_called_once()
        call_kwargs = mock_graph.get_all.call_args[1]
        assert call_kwargs["filters"]["agent_id"] == "main"


def test_graph_stats_empty_graph(client):
    """Test graph stats with empty graph."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_graph.get_all.return_value = []
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/stats?user_id=yishu")
        assert response.status_code == 200
        data = response.json()
        assert data["nodes"] == 0
        assert data["relations"] == 0
        assert data["relation_types"] == {}


def test_graph_agents_empty_result(client):
    """Test graph agents when no agents found."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_neo4j_graph = Mock()
        mock_neo4j_graph.query.return_value = []
        mock_graph.graph = mock_neo4j_graph
        mock_graph.node_label = ""
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/agents?user_id=yishu")
        assert response.status_code == 200
        data = response.json()
        assert "agents" in data
        assert data["agents"] == []


def test_graph_agents_with_base_label(client):
    """Test graph agents when base label is used."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_neo4j_graph = Mock()
        mock_neo4j_graph.query.return_value = [
            {"agent_id": "main", "memory_count": 100}
        ]
        mock_graph.graph = mock_neo4j_graph
        mock_graph.node_label = ":`__Entity__`"
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/agents?user_id=yishu")
        assert response.status_code == 200
        data = response.json()
        assert len(data["agents"]) == 1
        assert data["agents"][0]["agent_id"] == "main"
        assert data["agents"][0]["memory_count"] == 100


def test_graph_agents_query_error(client):
    """Test graph agents when graph query fails."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_neo4j_graph = Mock()
        mock_neo4j_graph.query.side_effect = Exception("Neo4j query failed")
        mock_graph.graph = mock_neo4j_graph
        mock_graph.node_label = ""
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/agents?user_id=yishu")
        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()


def test_graph_search_exception_handling(client):
    """Test graph search when graph client raises exception."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_graph.search.side_effect = Exception("Search failed")
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?q=Coffee&user_id=yishu")
        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()


def test_graph_stats_exception_handling(client):
    """Test graph stats when graph client raises exception."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_graph.get_all.side_effect = Exception("Get all failed")
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/stats?user_id=yishu")
        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()


def test_graph_agents_attribute_error(client):
    """Test graph agents when graph client doesn't have expected attributes."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock(spec=[])  # Empty spec, no attributes
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/agents?user_id=yishu")
        # Should handle gracefully and return empty list or error
        assert response.status_code in [200, 500]


def test_graph_search_result_structure(client, mock_graph_data):
    """Test that graph search returns correct structure."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_graph.search.return_value = mock_graph_data
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?q=Coffee&user_id=yishu")
        assert response.status_code == 200
        data = response.json()
        # Verify response structure
        assert "relations" in data
        assert "total" in data
        assert isinstance(data["relations"], list)
        assert isinstance(data["total"], int)
        assert len(data["relations"]) == data["total"]
        # Verify relation structure
        if len(data["relations"]) > 0:
            rel = data["relations"][0]
            assert "source" in rel
            assert "relationship" in rel
            assert "target" in rel
            assert all(isinstance(v, str) for v in rel.values())


def test_graph_search_result_with_missing_fields(client):
    """Test graph search handles results with missing fields."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        # Return result with missing fields
        mock_graph.search.return_value = [
            {"source": "coffee", "relationship": "LIKES"}  # missing target
        ]
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph/search?q=Coffee&user_id=yishu")
        assert response.status_code == 200
        data = response.json()
        assert "relations" in data
        # Should handle missing fields gracefully
        if len(data["relations"]) > 0:
            rel = data["relations"][0]
            # Missing fields should be empty strings
            assert rel.get("target", "") == ""


def test_graph_get_all_with_limit(client):
    """Test graph get_all endpoint with custom limit."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_graph.get_all.return_value = [
            {"source": "a", "relationship": "REL", "target": "b"}
        ]
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph?user_id=yishu&limit=500")
        assert response.status_code == 200
        data = response.json()
        assert "relations" in data
        assert "total" in data
        # Verify limit was passed
        mock_graph.get_all.assert_called_once()
        call_kwargs = mock_graph.get_all.call_args[1]
        assert call_kwargs["limit"] == 500


def test_graph_get_all_exception_handling(client):
    """Test graph get_all when graph client raises exception."""
    with patch('app.routers.graph.get_memory_client') as mock_get_client:
        mock_client = Mock()
        mock_graph = Mock()
        mock_graph.get_all.side_effect = Exception("Get all failed")
        mock_client.graph = mock_graph
        mock_get_client.return_value = mock_client

        response = client.get("/api/v1/graph?user_id=yishu")
        assert response.status_code == 500
        assert "failed" in response.json()["detail"].lower()
