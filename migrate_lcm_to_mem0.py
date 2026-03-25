#!/usr/bin/env python3
"""
OpenClaw LCM 到 mem0 的数据迁移脚本
- 步骤1: 从 LCM 读取数据，批量写入 SQLite 队列数据库（快速）
- 步骤2: 后台处理器从队列读取，调用 mem0 嵌入（异步）
"""

import sqlite3
import json
import time
from datetime import datetime
from pathlib import Path

# 配置
LCM_DB_PATH = Path.home() / ".openclaw" / "lcm.db"
QUEUE_DB = Path(__file__).parent / "mem0_queue.db"
BATCH_SIZE = 1000  # 每批写入数据库的记录数

def init_queue_db():
    """初始化队列数据库"""
    conn = sqlite3.connect(QUEUE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            user_id TEXT DEFAULT 'yishu',
            metadata TEXT,
            status TEXT DEFAULT 'pending',
            mem0_id TEXT,
            error TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            processed_at TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_status ON memory_queue(status)")
    conn.commit()
    conn.close()

def fetch_lcm_data():
    """从 LCM 数据库获取所有消息和总结"""
    conn = sqlite3.connect(LCM_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 获取所有消息
    cursor.execute("""
        SELECT m.message_id, m.role, m.content, m.created_at, m.conversation_id,
               c.session_id, c.session_key
        FROM messages m
        LEFT JOIN conversations c ON m.conversation_id = c.conversation_id
        ORDER BY m.created_at
    """)
    messages = [dict(row) for row in cursor.fetchall()]

    # 获取所有总结
    cursor.execute("""
        SELECT summary_id, conversation_id, kind, depth, content,
               token_count, earliest_at, latest_at, descendant_count,
               created_at
        FROM summaries
        ORDER BY created_at
    """)
    summaries = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return messages, summaries

def migrate_messages(messages):
    """迁移消息到队列数据库"""
    print(f"\n📝 步骤1: 将 {len(messages)} 条消息写入队列数据库...")

    conn = sqlite3.connect(QUEUE_DB)
    start_time = time.time()

    # 批量插入
    records = []
    for msg in messages:
        content = f"[{msg['role']}] {msg['content']}"
        metadata = json.dumps({
            "type": "message",
            "message_id": msg['message_id'],
            "conversation_id": msg['conversation_id'],
            "session_id": msg.get('session_id'),
            "session_key": msg.get('session_key'),
            "created_at": msg['created_at'],
            "role": msg['role']
        }, ensure_ascii=False)

        records.append((content, "yishu", metadata))

    conn.executemany("""
        INSERT INTO memory_queue (text, user_id, metadata) VALUES (?, ?, ?)
    """, records)

    conn.commit()
    elapsed = time.time() - start_time

    print(f"✅ 消息写入完成: {len(messages)} 条, 耗时 {elapsed:.2f}秒")
    conn.close()
    return len(messages)

def migrate_summaries(summaries):
    """迁移总结到队列数据库"""
    print(f"\n📊 步骤2: 将 {len(summaries)} 个总结写入队列数据库...")

    conn = sqlite3.connect(QUEUE_DB)
    start_time = time.time()

    records = []
    for summary in summaries:
        content = f"[会话总结 depth={summary['depth']}] {summary['content']}"
        metadata = json.dumps({
            "type": "summary",
            "summary_id": summary['summary_id'],
            "conversation_id": summary['conversation_id'],
            "kind": summary['kind'],
            "depth": summary['depth'],
            "token_count": summary.get('token_count'),
            "created_at": summary['created_at']
        }, ensure_ascii=False)

        records.append((content, "yishu", metadata))

    conn.executemany("""
        INSERT INTO memory_queue (text, user_id, metadata) VALUES (?, ?, ?)
    """, records)

    conn.commit()
    elapsed = time.time() - start_time

    print(f"✅ 总结写入完成: {len(summaries)} 个, 耗时 {elapsed:.2f}秒")
    conn.close()
    return len(summaries)

def get_queue_stats():
    """获取队列统计"""
    conn = sqlite3.connect(QUEUE_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT status, COUNT(*) FROM memory_queue GROUP BY status")
    stats = dict(cursor.fetchall())
    cursor.execute("SELECT COUNT(*) FROM memory_queue")
    total = cursor.fetchone()[0]
    conn.close()
    return stats, total

def main():
    print("🚀 OpenClaw LCM → mem0 数据迁移工具")
    print(f"=" * 50)
    print(f"LCM 数据库: {LCM_DB_PATH}")
    print(f"队列数据库: {QUEUE_DB}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 初始化队列数据库
    init_queue_db()

    # 获取数据
    print("\n📂 从 LCM 读取数据...")
    messages, summaries = fetch_lcm_data()
    print(f"   消息: {len(messages)} 条")
    print(f"   总结: {len(summaries)} 个")

    # 迁移
    msg_count = migrate_messages(messages)
    sum_count = migrate_summaries(summaries)

    # 统计
    stats, total = get_queue_stats()
    pending = stats.get('pending', 0)

    print("\n" + "=" * 50)
    print("✅ 迁移完成!")
    print(f"   消息: {msg_count} 条")
    print(f"   总结: {sum_count} 个")
    print(f"   总计: {total} 条记录在队列中 (状态: {stats})")
    print(f"\n📋 下一步: 启动后台处理器处理嵌入")
    print(f"   cd mem0 && python3 process_queue.py")
    print(f"\n📊 查看队列状态:")
    print(f"   sqlite3 {QUEUE_DB} 'SELECT status, COUNT(*) FROM memory_queue GROUP BY status'")

if __name__ == "__main__":
    main()
