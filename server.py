#!/usr/bin/env python3
"""
mem0 FastAPI 服务器启动脚本
为 OpenClaw 插件提供 API 接口
使用 Qdrant 向量数据库
"""

import os
import sys
import uvicorn
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import sqlite3

app = FastAPI(title="mem0 API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置
SCRIPT_DIR = Path(__file__).parent
QUEUE_DB = SCRIPT_DIR / "mem0_queue.db"

QDRANT_HOST = "http://127.0.0.1:6333"
COLLECTION_NAME = "memories"
EMBEDDING_DIM = 1536

# 懒加载的 Qdrant 客户端
_qdrant_client = None


def get_qdrant_client():
    """懒加载获取 Qdrant 客户端"""
    global _qdrant_client
    if _qdrant_client is None:
        from qdrant_client import QdrantClient
        _qdrant_client = QdrantClient(url=QDRANT_HOST)
    return _qdrant_client


def ensure_collection():
    """确保 collection 存在"""
    client = get_qdrant_client()
    collections = client.get_collections().collections
    names = [c.name for c in collections]
    if COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={"": {"size": EMBEDDING_DIM, "distance": "Cosine"}},
        )


def get_embedding(text: str) -> Optional[List[float]]:
    """调用 FAI 嵌入 API 生成向量"""
    import openai
    try:
        client = openai.OpenAI(
            api_key='fai-2-977-cdc6435fbca2',
            base_url='https://trip-llm.alibaba-inc.com/api/openai/v1'
        )
        resp = client.embeddings.create(
            model='text-embedding-3-small',
            input=text
        )
        return resp.data[0].embedding
    except Exception as e:
        print(f"[WARN] Embedding failed: {e}", file=sys.stderr)
        return None


def add_memory_to_qdrant(text: str, user_id: str, agent_id: str = None, metadata: dict = None):
    """添加记忆到 Qdrant"""
    import uuid
    from qdrant_client.models import PointStruct, Payload

    ensure_collection()
    client = get_qdrant_client()

    mem_metadata = (metadata or {}).copy()
    mem_metadata.update({
        "data": text,
        "userId": user_id,
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "source": "openclaw_plugin"
    })
    if agent_id:
        mem_metadata["agentId"] = agent_id

    embedding = get_embedding(text)
    memory_id = str(uuid.uuid4())

    point = PointStruct(
        id=memory_id,
        vector={"": embedding} if embedding else None,
        payload=mem_metadata
    )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[point]
    )
    return memory_id


def search_memories_vector(query: str, user_id: str, limit: int = 5, agent_id: str = None) -> List[Dict]:
    """在 Qdrant 中向量搜索，可按 agent_id 过滤"""
    ensure_collection()
    client = get_qdrant_client()

    query_vec = get_embedding(query)
    if not query_vec:
        return []

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=limit * 3,
    )

    memories = []
    for r in results.points:
        payload = r.payload or {}
        if user_id and user_id not in (payload.get('userId', '') or ''):
            continue
        if agent_id is not None and payload.get('agentId') != agent_id:
            continue
        memories.append({
            "id": r.id,
            "memory": payload.get('data', '') or payload.get('memory', ''),
            "score": r.score,
            "metadata": payload
        })
        if len(memories) >= limit:
            break
    return memories


def get_all_memories(user_id: str, agent_id: str = None) -> List[Dict]:
    """从 Qdrant 获取所有记忆，可按 agent_id 过滤"""
    ensure_collection()
    client = get_qdrant_client()

    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=None,
        limit=1000,
        with_vectors=False,
        with_payload=True,
    )

    memories = []
    for point in (results[0] or []):
        payload = point.payload or {}
        # 兼容两种数据结构：插件格式（驼峰）和 HTTP API 格式（下划线）
        user_id_field = payload.get('user_id') or payload.get('userId', '')
        if user_id and user_id not in str(user_id_field):
            continue
        # agent_id 过滤：两个字段有一个匹配就保留
        aid_snake = payload.get('agent_id')
        aid_camel = payload.get('agentId')
        if agent_id is not None and aid_snake != agent_id and aid_camel != agent_id:
            continue
        memories.append({
            "id": point.id,
            "memory": payload.get('data', '') or payload.get('memory', ''),
            "metadata": payload
        })
    return memories


def get_all_agents(user_id: str) -> List[Dict]:
    """获取所有有记忆的 agent 列表及其统计"""
    ensure_collection()
    client = get_qdrant_client()

    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=None,
        limit=1000,
        with_vectors=False,
        with_payload=True,
    )

    agent_stats = {}
    for point in (results[0] or []):
        payload = point.payload or {}
        if user_id and user_id not in (payload.get('userId', '') or ''):
            continue
        aid = payload.get('agentId') or payload.get('agent_id') or 'unknown'
        if aid not in agent_stats:
            agent_stats[aid] = {"agent_id": aid, "count": 0, "latest": None}
        agent_stats[aid]["count"] += 1
        created = payload.get('createdAt')
        if created:
            if not agent_stats[aid]["latest"] or created > agent_stats[aid]["latest"]:
                agent_stats[aid]["latest"] = created

    return list(agent_stats.values())


def get_memory_stats_by_agent(user_id: str, agent_id: str = None) -> Dict:
    """获取记忆统计，可按 agent_id 过滤"""
    ensure_collection()
    client = get_qdrant_client()

    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=None,
        limit=1000,
        with_vectors=False,
        with_payload=True,
    )

    total = 0
    by_source = {}
    by_agent = {}
    for point in (results[0] or []):
        payload = point.payload or {}
        if user_id and user_id not in (payload.get('userId', '') or ''):
            continue
        aid_snake = payload.get('agent_id')
        aid_camel = payload.get('agentId')
        aid = aid_camel or aid_snake or 'unknown'
        if agent_id is not None and aid_snake != agent_id and aid_camel != agent_id:
            continue
        total += 1
        source = payload.get('source', 'unknown') or 'unknown'
        by_source[source] = by_source.get(source, 0) + 1
        if agent_id is None:  # 全局统计时才按 agent 分组
            if aid not in by_agent:
                by_agent[aid] = {"agent_id": aid, "count": 0}
            by_agent[aid]["count"] += 1

    return {
        "total": total,
        "by_source": by_source,
        "by_agent": list(by_agent.values()) if agent_id is None else [],
        "user_id": user_id,
        "agent_id": agent_id
    }


def delete_memory_from_qdrant(memory_id: str) -> bool:
    """从 Qdrant 删除记忆"""
    from qdrant_client.models import PointIdsList
    ensure_collection()
    client = get_qdrant_client()
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=PointIdsList(points=[memory_id]),
    )
    return True


# =============================================================================
# Pydantic 模型
# =============================================================================

class MemoryAddRequest(BaseModel):
    text: str
    user_id: str = "yishu"
    agent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MemoryItem(BaseModel):
    text: str
    user_id: Optional[str] = None
    agent_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MemoryBatchRequest(BaseModel):
    items: List[MemoryItem]
    user_id: str = "yishu"

class MemoryResponse(BaseModel):
    id: str
    memory: str
    metadata: Optional[Dict[str, Any]] = None


def _get_queue_stats():
    """从 SQLite 队列数据库获取统计"""
    if not QUEUE_DB.exists():
        return {"pending": 0, "done": 0, "error": 0, "total": 0}
    conn = sqlite3.connect(QUEUE_DB)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT status, COUNT(*) FROM memory_queue GROUP BY status")
        stats = dict(cursor.fetchall())
        cursor.execute("SELECT COUNT(*) FROM memory_queue")
        total = cursor.fetchone()[0]
        conn.close()
        return {
            "pending": stats.get("pending", 0),
            "done": stats.get("done", 0),
            "error": stats.get("error", 0),
            "total": total,
        }
    except Exception:
        conn.close()
        return {"pending": 0, "done": 0, "error": 0, "total": 0}


# =============================================================================
# API 端点
# =============================================================================

@app.post("/api/v1/memories/raw/batch")
async def add_memories_raw_batch(request: MemoryBatchRequest):
    """批量快速添加记忆到 SQLite 队列"""
    try:
        conn = sqlite3.connect(QUEUE_DB)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memory_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                user_id TEXT DEFAULT 'yishu',
                agent_id TEXT,
                metadata TEXT,
                status TEXT DEFAULT 'pending',
                mem0_id TEXT,
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                processed_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON memory_queue(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON memory_queue(agent_id)")
        try:
            conn.execute("ALTER TABLE memory_queue ADD COLUMN agent_id TEXT")
        except Exception:
            pass

        records = []
        for item in request.items:
            metadata_json = json.dumps(item.metadata or {}, ensure_ascii=False)
            records.append((item.text, item.user_id or request.user_id, item.agent_id, metadata_json))

        conn.executemany(
            "INSERT INTO memory_queue (text, user_id, agent_id, metadata) VALUES (?, ?, ?, ?)",
            records,
        )
        conn.commit()
        conn.close()
        return {"status": "queued", "count": len(records), "queue_db": str(QUEUE_DB)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memories/queue-status")
async def get_queue_status():
    """获取队列状态"""
    return _get_queue_stats()


@app.post("/api/v1/memories")
async def add_memory(request: MemoryAddRequest):
    """直接添加记忆到 Qdrant"""
    try:
        memory_id = add_memory_to_qdrant(
            text=request.text,
            user_id=request.user_id,
            agent_id=request.agent_id,
            metadata=request.metadata
        )
        return {"status": "added", "id": memory_id, "text": request.text[:50]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memories")
async def get_memories(user_id: str = "yishu", agent_id: str = None):
    """获取所有记忆，可按 agent_id 过滤"""
    try:
        return get_all_memories(user_id, agent_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memories/search")
async def search_memories(query: str, user_id: str = "yishu", limit: int = 5, agent_id: str = None):
    """搜索记忆（向量搜索），可按 agent_id 过滤"""
    try:
        return search_memories_vector(query, user_id, limit, agent_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memories/stats")
async def get_memory_stats(user_id: str = "yishu", agent_id: str = None):
    """获取记忆统计，可按 agent_id 过滤"""
    try:
        return get_memory_stats_by_agent(user_id, agent_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memories/agents")
async def get_agents(user_id: str = "yishu"):
    """获取所有有记忆的 agent 列表"""
    try:
        return {"agents": get_all_agents(user_id)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/memories/{memory_id}")
async def delete_memory(memory_id: str):
    """删除记忆"""
    try:
        delete_memory_from_qdrant(memory_id)
        return {"status": "deleted", "id": memory_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok", "service": "mem0"}


@app.get("/dashboard")
async def dashboard():
    """返回 dashboard HTML 页面"""
    from fastapi.responses import FileResponse
    return FileResponse(SCRIPT_DIR / "dashboard.html")


@app.get("/")
async def root():
    """重定向到 dashboard"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/dashboard")


# =============================================================================
# AI Learning REST Endpoints
# =============================================================================

@app.get("/ailearn/skills")
async def get_skills(
    user_id: str = Query(..., description="User identifier"),
    type: Optional[str] = Query(None, alias="type"),
    limit: int = Query(50, ge=1, le=200),
) -> Dict[str, Any]:
    return {"skills": [], "total": 0, "user_id": user_id}


@app.get("/ailearn/patterns")
async def get_patterns(
    user_id: str = Query(..., description="User identifier"),
    type: Optional[str] = Query(None, alias="type"),
    limit: int = Query(100, ge=1, le=500),
) -> Dict[str, Any]:
    return {"patterns": [], "total": 0, "user_id": user_id}


@app.get("/ailearn/amendments")
async def get_amendments(
    user_id: str = Query(..., description="User identifier"),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> Dict[str, Any]:
    amendments = []
    if status:
        amendments = [a for a in amendments if a.get("status") == status]
    return {"amendments": amendments[:limit], "total": len(amendments), "user_id": user_id}


@app.post("/ailearn/amendments/{amendment_id}/approve", status_code=202)
async def approve_amendment(amendment_id: str) -> Dict[str, Any]:
    return {"id": amendment_id, "status": "approved", "message": f"Amendment {amendment_id} approved"}


@app.post("/ailearn/amendments/{amendment_id}/reject", status_code=202)
async def reject_amendment(amendment_id: str) -> Dict[str, Any]:
    return {"id": amendment_id, "status": "rejected", "message": f"Amendment {amendment_id} rejected"}


class AnalyzeRequest(BaseModel):
    user_id: str
    agent_id: Optional[str] = None


@app.post("/ailearn/analyze", status_code=202)
async def trigger_analysis(request: AnalyzeRequest) -> Dict[str, Any]:
    return {"status": "triggered", "user_id": request.user_id, "agent_id": request.agent_id, "message": "Analysis triggered successfully"}


@app.get("/ailearn/health")
async def get_ai_health(
    user_id: str = Query(..., description="User identifier"),
) -> Dict[str, Any]:
    return {
        "status": "healthy",
        "total_observations": 0,
        "total_patterns_detected": 0,
        "total_skills_extracted": 0,
        "total_amendments_pending": 0,
        "success_rate": 0.0,
        "avg_response_time_ms": 0.0,
        "last_analysis_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/ailearn/skills/effectiveness")
async def get_skill_effectiveness(
    user_id: str = Query(..., description="User identifier"),
    limit: int = Query(50, ge=1, le=200),
) -> Dict[str, Any]:
    return {"skills": [], "user_id": user_id}


# =============================================================================
# Graph (Neo4j) 端点
# =============================================================================

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "mem0password")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")


def _get_neo4j_driver():
    """懒加载获取 Neo4j 驱动"""
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))


@app.get("/api/v1/graph/agents")
async def get_graph_agents(
    user_id: str = Query(..., description="User prefix (e.g. yishu)"),
):
    """
    获取当前用户的可选 agent 列表（图谱隔离维度），
    用于填充图谱视图的 agent 选择下拉框。
    """
    try:
        driver = _get_neo4j_driver()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {e}")

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.user_id STARTS WITH $user_id
                RETURN DISTINCT n.user_id AS user_id, count(*) AS node_count
                ORDER BY node_count DESC
                """,
                {"user_id": user_id}
            ).data()
        driver.close()
        # 提取 agent 名称（去掉前缀）
        prefix = user_id + ":agent:"
        agents = []
        for row in result:
            uid = row["user_id"]
            if uid.startswith(prefix):
                agents.append({
                    "agent_id": uid,
                    "label": uid[len(prefix):],
                    "node_count": row["node_count"],
                })
            else:
                agents.append({
                    "agent_id": uid,
                    "label": uid,
                    "node_count": row["node_count"],
                })
        return {"agents": agents}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Graph agents query failed: {e}")


@app.get("/api/v1/graph")
async def get_graph(
    user_id: str = Query(..., description="User identifier"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    limit: int = Query(200, ge=1, le=1000, description="Max relations"),
):
    """
    从 Neo4j 图数据库获取实体关系，返回 {source, relationship, target} 列表，
    可直接用于 force-directed graph 可视化。
    """
    try:
        driver = _get_neo4j_driver()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {e}")

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            params = {"user_id": user_id, "limit": limit}

            cypher = """
            MATCH (n)-[r]->(m)
            WHERE n.user_id STARTS WITH $user_id AND m.user_id STARTS WITH $user_id
            AND (r.valid IS NULL OR r.valid = true)
            RETURN n.name AS source, type(r) AS relationship, m.name AS target
            LIMIT $limit
            """

            result = session.run(cypher, params)
            records = [dict(r) for r in result]

        driver.close()
        return {"relations": records, "total": len(records)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Graph query failed: {e}")


@app.get("/api/v1/graph/stats")
async def get_graph_stats(
    user_id: str = Query(..., description="User identifier"),
):
    """
    获取图谱统计：节点数、关系数、关系类型分布。
    """
    try:
        driver = _get_neo4j_driver()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Neo4j unavailable: {e}")

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            # 节点数
            node_count = session.run(
                "MATCH (n) WHERE n.user_id STARTS WITH $user_id RETURN count(n) AS cnt",
                {"user_id": user_id}
            ).single()["cnt"]

            # 关系数
            rel_count = session.run(
                """
                MATCH (n)-[r]->(m)
                WHERE n.user_id STARTS WITH $user_id
                AND (r.valid IS NULL OR r.valid = true)
                RETURN count(r) AS cnt
                """,
                {"user_id": user_id}
            ).single()["cnt"]

            # 关系类型分布
            rel_types = session.run(
                """
                MATCH (n)-[r]->(m)
                WHERE n.user_id STARTS WITH $user_id
                AND (r.valid IS NULL OR r.valid = true)
                RETURN type(r) AS rel_type, count(*) AS cnt
                ORDER BY cnt DESC
                """,
                {"user_id": user_id}
            ).data()

        driver.close()
        return {
            "nodes": node_count,
            "relations": rel_count,
            "relation_types": rel_types,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Graph stats failed: {e}")


if __name__ == "__main__":
    print("🚀 启动 mem0 FastAPI 服务器...")
    print(f"📡 Qdrant 地址: http://127.0.0.1:6333")
    print(f"🗂️  Collection: {COLLECTION_NAME}")
    print(f"🌐 API 地址: http://localhost:8765")
    print(f"📖 API 文档: http://localhost:8765/docs")
    print()

    uvicorn.run(app, host="0.0.0.0", port=8765, log_level="info")
