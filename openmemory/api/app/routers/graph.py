"""
Graph memory API router.

Exposes the Neo4j graph store via REST for the MemOS mission-control frontend.
"""

import logging
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.utils.memory import get_memory_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/graph", tags=["graph"])


class GraphRelation(BaseModel):
    source: str
    relationship: str
    target: str


class GraphResponse(BaseModel):
    relations: List[GraphRelation]
    total: int


class GraphSearchResponse(BaseModel):
    relations: List[GraphRelation]
    total: int


class GraphStatsResponse(BaseModel):
    nodes: int
    relations: int
    relation_types: dict


class AgentInfo(BaseModel):
    agent_id: str
    memory_count: int


class GraphAgentsResponse(BaseModel):
    agents: List[AgentInfo]


# GET /api/v1/graph
@router.get("/", response_model=GraphResponse)
async def get_graph(
    user_id: str = Query(..., description="User identifier"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    limit: int = Query(200, ge=1, le=1000, description="Max relations to return"),
):
    """
    Retrieve all entity relationships from the Neo4j graph store.

    Returns a list of (source, relationship, target) triples that can be
    rendered as a force-directed graph on the frontend.
    """
    try:
        # Direct Neo4j query to handle old user_id format (yishu:agent:xxx)
        from neo4j import GraphDatabase

        # Build Neo4j query
        if agent_id:
            # Filter by specific agent - match user_id ending with :agent:{agent_id}
            query = """
                MATCH (n)-[r]->(m)
                WHERE n.user_id = $user_id_pattern
                RETURN n.name AS source, type(r) AS relationship, m.name AS target
                LIMIT $limit
            """
            params = {"user_id_pattern": f"{user_id}:agent:{agent_id}", "limit": limit}
        else:
            # Match any user_id that starts with the base user_id
            query = """
                MATCH (n)-[r]->(m)
                WHERE n.user_id STARTS WITH $user_id_prefix
                RETURN n.name AS source, type(r) AS relationship, m.name AS target
                LIMIT $limit
            """
            params = {"user_id_prefix": user_id, "limit": limit}

        # Execute query
        driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ.get("NEO4J_PASSWORD", "mem0password"))
        )

        with driver.session() as session:
            result = session.run(query, params)
            relations = [
                {"source": record["source"], "relationship": record["relationship"], "target": record["target"]}
                for record in result
            ]

        driver.close()

        return GraphResponse(
            relations=[
                GraphRelation(source=r["source"], relationship=r["relationship"], target=r["target"])
                for r in relations
            ],
            total=len(relations),
        )
    except Exception as e:
        logger.error(f"Graph get_all failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph query failed: {e}")


# GET /api/v1/graph/search
@router.get("/search", response_model=GraphSearchResponse)
async def search_graph(
    q: str = Query(..., min_length=1, description="Search query (node name or relationship type)"),
    user_id: str = Query(..., description="User identifier"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    limit: int = Query(100, ge=1, le=1000, description="Max results to return"),
):
    """
    Search the graph for nodes or relationships matching the query.

    Performs fuzzy matching on node names and relationship types.
    """
    try:
        client = get_memory_client()
    except Exception as e:
        logger.error(f"Failed to get memory client: {e}")
        raise HTTPException(status_code=503, detail="Memory service unavailable")

    if client is None:
        raise HTTPException(
            status_code=503,
            detail="Memory client not initialized. "
            "Ensure Neo4j is configured and the server has been restarted.",
        )

    if not hasattr(client, "graph") or client.graph is None:
        raise HTTPException(
            status_code=503,
            detail="Graph store is not enabled. "
            "Configure a graph_store (e.g. Neo4j) in the Mem0 configuration.",
        )

    try:
        filters = {"user_id": user_id}
        if agent_id:
            filters["agent_id"] = agent_id

        # Use the graph's search method with the query
        search_results = client.graph.search(query=q, filters=filters, limit=limit)

        return GraphSearchResponse(
            relations=[
                GraphRelation(
                    source=r.get("source", ""),
                    relationship=r.get("relationship", ""),
                    target=r.get("target", "")
                )
                for r in search_results
            ],
            total=len(search_results),
        )
    except Exception as e:
        logger.error(f"Graph search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph search failed: {e}")


# GET /api/v1/graph/stats
@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats(
    user_id: str = Query(..., description="User identifier"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
):
    """
    Get statistics about the graph.

    Returns node count, relation count, and relation type distribution.
    """
    try:
        # Direct Neo4j query for stats
        from neo4j import GraphDatabase

        # Build query to count nodes and relations for user
        if agent_id:
            # Filter by specific agent
            query = """
                MATCH (n {user_id: $user_id_pattern})
                OPTIONAL MATCH (n)-[r]->()
                RETURN count(DISTINCT n) as node_count, count(r) as relation_count
            """
            params = {"user_id_pattern": f"{user_id}:agent:{agent_id}"}
        else:
            # Match any user_id that starts with the base user_id
            query = """
                MATCH (n)
                WHERE n.user_id STARTS WITH $user_id_prefix
                OPTIONAL MATCH (n)-[r]->()
                RETURN count(DISTINCT n) as node_count, count(r) as relation_count
            """
            params = {"user_id_prefix": user_id}

        # Execute query
        driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ.get("NEO4J_PASSWORD", "mem0password"))
        )

        try:
            with driver.session() as session:
                result = session.run(query, params)
                record = result.single()
                node_count = record["node_count"]
                relation_count = record["relation_count"]

            # Get relation types distribution (respect agent_id filter)
            if agent_id:
                type_query = """
                    MATCH (n {user_id: $user_id_pattern})-[r]->()
                    RETURN type(r) as rel_type, count(*) as count
                """
                type_params = {"user_id_pattern": f"{user_id}:agent:{agent_id}"}
            else:
                type_query = """
                    MATCH (n)-[r]->()
                    WHERE n.user_id STARTS WITH $user_id_prefix
                    RETURN type(r) as rel_type, count(*) as count
                """
                type_params = {"user_id_prefix": user_id}

            with driver.session() as session:
                result = session.run(type_query, type_params)
                relation_types = {record["rel_type"]: record["count"] for record in result}
        finally:
            driver.close()

        return GraphStatsResponse(
            nodes=node_count,
            relations=relation_count,
            relation_types=relation_types,
        )
    except Exception as e:
        logger.error(f"Graph stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph stats failed: {e}")


# GET /api/v1/graph/agents
@router.get("/agents", response_model=GraphAgentsResponse)
async def get_graph_agents(
    user_id: str = Query(..., description="User identifier"),
):
    """
    Get list of agents with their memory counts.

    Returns all agent_ids that have memories in the graph for this user.
    """
    try:
        # Direct Neo4j query to get agents and their memory counts
        from neo4j import GraphDatabase

        # Query to get distinct user_ids and count their associated memories/nodes
        # Extract agent_id in Python from user_id format: "yishu:agent:main" -> "main"
        query = """
            MATCH (n)
            WHERE n.user_id STARTS WITH $user_id_prefix
            AND n.user_id CONTAINS ':agent:'
            RETURN n.user_id as user_id, count(*) as memory_count
            ORDER BY memory_count DESC
        """

        # Execute query
        driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.environ.get("NEO4J_USERNAME", "neo4j"), os.environ.get("NEO4J_PASSWORD", "mem0password"))
        )

        try:
            with driver.session() as session:
                result = session.run(query, {"user_id_prefix": user_id})
                # Extract agent_id from user_id in Python
                agents = []
                for record in result:
                    user_id_val = record["user_id"]
                    if ":agent:" in user_id_val:
                        agent_id = user_id_val.split(":agent:")[-1]
                        memory_count = record["memory_count"]
                        agents.append(AgentInfo(agent_id=agent_id, memory_count=memory_count))
        finally:
            driver.close()

        return GraphAgentsResponse(agents=agents)
    except Exception as e:
        logger.error(f"Graph agents failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph agents query failed: {e}")
