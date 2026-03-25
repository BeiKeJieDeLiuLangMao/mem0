#!/usr/bin/env python3
"""
mem0 后台队列处理器 (SQLite 持久化 + 指数退避重试)

改进:
- 指数退避重试机制（最多 3 次，间隔 10s/30s/90s）
- 错误分类：rate_limit / permanent / transient
- 重试恢复命令：--recover-errors 重置 error 项以便重试
- 增大并发 worker：3 → 8
- 详细统计和进度报告
- 支持 agent_id 隔离
"""

import sys
import time
import json
import uuid
import signal
import logging
import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Optional, List, Dict

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Qdrant 配置
QDRANT_HOST = "http://127.0.0.1:6333"
COLLECTION_NAME = "memories"
EMBEDDING_DIM = 1536

# 配置
SCRIPT_DIR = Path(__file__).parent
QUEUE_DB = SCRIPT_DIR / "mem0_queue.db"
BATCH_SIZE = 100
MAX_WORKERS = 8
MAX_RETRIES = 3
RETRY_DELAYS = [10, 30, 90]  # 秒

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# 懒加载 Qdrant 客户端
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
        logger.warning(f"Embedding failed: {e}")
        return None


def add_to_qdrant(text: str, user_id: str, agent_id: str = None, metadata: dict = None) -> str:
    """直接添加记忆到 Qdrant"""
    from qdrant_client.models import PointStruct

    ensure_collection()
    client = get_qdrant_client()

    mem_metadata = (metadata or {}).copy()
    mem_metadata.update({
        "data": text,
        "user_id": user_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "openclaw_queue"
    })
    if agent_id:
        mem_metadata["agent_id"] = agent_id

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


running = True
lock = Lock()


def init_queue_db():
    """初始化/升级队列数据库，添加重试相关字段和 agent_id"""
    conn = sqlite3.connect(QUEUE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            user_id TEXT DEFAULT 'yishu',
            agent_id TEXT,
            metadata TEXT,
            status TEXT DEFAULT 'pending',
            retry_count INTEGER DEFAULT 0,
            last_error TEXT,
            mem0_id TEXT,
            error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            processed_at TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON memory_queue(status)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_retry ON memory_queue(retry_count)")
    # 添加 agent_id 索引（兼容已有表）
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON memory_queue(agent_id)")
    except Exception:
        pass  # 索引已存在
    # 迁移：已有表可能缺少 agent_id 列
    try:
        conn.execute("ALTER TABLE memory_queue ADD COLUMN agent_id TEXT")
    except Exception:
        pass  # 列已存在
    conn.commit()
    conn.close()


def _classify_error(error_msg: str) -> str:
    """将错误归类为 rate_limit / permanent / transient"""
    msg = error_msg.lower()
    if any(k in msg for k in ["429", "too many requests", "rate limit", "retry-after"]):
        return "rate_limit"
    if any(k in msg for k in ["401", "403", "404", "500", "502", "503", "timeout"]):
        return "transient"
    return "permanent"


def process_single(record: dict) -> dict:
    """处理单条记录，包含重试延迟"""
    retry_count = record.get("retry_count", 0)

    # 如果需要等待退避延迟
    if retry_count > 0 and retry_count <= len(RETRY_DELAYS):
        delay = RETRY_DELAYS[retry_count - 1]
        logger.info(f"  record {record['id']} 等待退避 {delay}s (retry {retry_count})")
        time.sleep(delay)

    try:
        metadata = {}
        if record.get("metadata"):
            try:
                metadata = json.loads(record["metadata"]) if isinstance(record["metadata"], str) else record["metadata"]
            except Exception:
                metadata = {}

        memory_id = add_to_qdrant(
            text=record["text"],
            user_id=record["user_id"],
            agent_id=record.get("agent_id"),
            metadata=metadata,
        )
        return {"success": True, "id": memory_id, "retry_count": retry_count}
    except Exception as e:
        error_str = str(e)
        error_type = _classify_error(error_str)

        # rate_limit 错误额外等待
        if error_type == "rate_limit":
            logger.warning(f"  record {record['id']} 遇到限流，等待 120s...")
            time.sleep(120)

        # 是否值得重试
        should_retry = error_type != "permanent" and retry_count < MAX_RETRIES
        next_retry = retry_count + 1 if should_retry else retry_count

        return {
            "success": False,
            "error": error_str[:500],
            "error_type": error_type,
            "should_retry": should_retry,
            "retry_count": next_retry,
        }


def process_batch() -> Optional[dict]:
    """处理一批 pending/error 记录"""
    conn = sqlite3.connect(QUEUE_DB)
    cursor = conn.cursor()

    # 优先取 pending，其次取可重试的 error
    cursor.execute("""
        SELECT id, text, user_id, agent_id, metadata, retry_count
        FROM memory_queue
        WHERE status = 'pending'
           OR (status = 'error' AND retry_count < ? AND last_error != 'PERMANENT')
        ORDER BY
            CASE status WHEN 'pending' THEN 0 ELSE 1 END,
            id
        LIMIT ?
    """, (MAX_RETRIES, BATCH_SIZE))
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return None

    records = [
        {"id": r[0], "text": r[1], "user_id": r[2], "agent_id": r[3], "metadata": r[4], "retry_count": r[5] or 0}
        for r in rows
    ]

    results = {"success": 0, "retry": 0, "error": 0}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single, rec): rec for rec in records}

        for future in as_completed(futures):
            record = futures[future]
            record_id = record["id"]
            try:
                result = future.result(timeout=180)
            except Exception as e:
                result = {
                    "success": False,
                    "error": f"future_timeout: {str(e)[:200]}",
                    "should_retry": True,
                    "retry_count": record["retry_count"] + 1,
                }

            conn2 = sqlite3.connect(QUEUE_DB)
            if result["success"]:
                conn2.execute("""
                    UPDATE memory_queue
                    SET status = 'done', mem0_id = ?, processed_at = ?,
                        last_error = NULL
                    WHERE id = ?
                """, (result["id"], datetime.now().isoformat(), record_id))
                results["success"] += 1
            elif result.get("should_retry"):
                conn2.execute("""
                    UPDATE memory_queue
                    SET status = 'pending', retry_count = ?,
                        last_error = ?
                    WHERE id = ?
                """, (result["retry_count"], result.get("error", "unknown")[:500], record_id))
                results["retry"] += 1
            else:
                conn2.execute("""
                    UPDATE memory_queue
                    SET status = 'error', retry_count = ?,
                        last_error = 'PERMANENT', error = ?
                    WHERE id = ?
                """, (result["retry_count"], result.get("error", "unknown")[:500], record_id))
                results["error"] += 1
            conn2.commit()
            conn2.close()

    return results


def recover_errors(limit: int = 5000) -> int:
    """
    将 error 项重置为 pending，以便重新处理。
    跳过标记为 PERMANENT 的错误。
    返回重置的数量。
    """
    conn = sqlite3.connect(QUEUE_DB)
    cursor = conn.cursor()
    # SQLite UPDATE 不支持 LIMIT，用子查询分批处理
    cursor.execute("""
        UPDATE memory_queue
        SET status = 'pending', last_error = NULL
        WHERE id IN (
            SELECT id FROM memory_queue
            WHERE status = 'error'
              AND (last_error != 'PERMANENT' OR last_error IS NULL)
            LIMIT ?
        )
    """, (limit,))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    logger.info(f"已重置 {affected} 条 error → pending")
    return affected


def get_stats() -> dict:
    """获取队列统计"""
    conn = sqlite3.connect(QUEUE_DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT status,
               COUNT(*) as cnt,
               SUM(CASE WHEN last_error = 'PERMANENT' THEN 1 ELSE 0 END) as permanent,
               SUM(retry_count) as total_retries
        FROM memory_queue
        GROUP BY status
    """)
    rows = cur.fetchall()
    conn.close()
    stats = {}
    for row in rows:
        stats[row[0]] = {"count": row[1], "permanent": row[2], "total_retries": row[3]}
    return stats


def signal_handler(signum, frame):
    global running
    running = False


def main():
    global running
    parser = argparse.ArgumentParser(description="mem0 队列处理器")
    parser.add_argument("--recover", action="store_true", help="将 error 项重置为 pending 并退出")
    parser.add_argument("--recover-limit", type=int, default=5000, help="每次最多重置多少条")
    parser.add_argument("--once", action="store_true", help="只跑一个批次后退出（用于测试）")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    init_queue_db()

    if args.recover:
        count = recover_errors(args.recover_limit)
        print(f"已重置 {count} 条 error → pending")
        return

    stats = get_stats()
    pending = stats.get("pending", {}).get("count", 0)
    error = stats.get("error", {}).get("count", 0)
    done = stats.get("done", {}).get("count", 0)
    total = pending + error + done

    logger.info(f"mem0 队列处理器启动 | 并发={MAX_WORKERS} | 最大重试={MAX_RETRIES}")
    logger.info(f"队列: pending={pending} error={error} done={done} total={total}")

    if pending == 0 and error > 0:
        logger.info("无 pending 项，error 项可先用 --recover 恢复")

    total_success = 0
    total_retry = 0
    total_error = 0

    while running:
        results = process_batch()

        if results:
            total_success += results["success"]
            total_retry += results["retry"]
            total_error += results["error"]
            logger.info(
                f"批次  +{results['success']} ✓  ~{results['retry']} ⟳  "
                f"✗{results['error']}  |  "
                f"累计: ✓{total_success}  ⟳{total_retry}  ✗{total_error}"
            )
        else:
            stats = get_stats()
            pending = stats.get("pending", {}).get("count", 0)
            error = stats.get("error", {}).get("count", 0)
            if pending == 0 and error == 0:
                logger.info("队列已空，3秒后退出")
                time.sleep(3)
                break
            logger.info(f"队列空或全为 error，10秒后重检  (pending={pending} error={error})")
            time.sleep(10)

        if args.once:
            break

    logger.info(
        f"处理完成: ✓{total_success}  ⟳{total_retry}  "
        f"✗{total_error}  (残留 error 需 review 或 --recover)"
    )


if __name__ == "__main__":
    main()
