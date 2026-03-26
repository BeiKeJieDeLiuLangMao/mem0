# Mem0 - AI Agent 持久化记忆层

## 项目概述

Mem0 是 OpenClaw Team OS 的记忆后端服务，提供 AI Agent 的长期记忆存储、向量搜索和知识图谱功能。

## 技术栈

- **API 框架**: FastAPI (Python)
- **向量数据库**: Qdrant (http://127.0.0.1:6333)
- **图数据库**: Neo4j (bolt://127.0.0.1:7687)
- **关系数据库**: SQLite (openmemory.db)
- **LLM 提供商**: OpenAI / Ollama
- **嵌入模型**: OpenAI text-embedding-3-small (1536 维)

## 目录结构

```
mem0/
├── openmemory/
│   └── api/
│       ├── main.py              # FastAPI 应用入口
│       ├── app/
│       │   ├── database.py       # SQLAlchemy 配置
│       │   ├── models.py         # ORM 模型定义
│       │   ├── routers/          # API 路由
│       │   │   ├── memories.py   # 记忆 CRUD
│       │   │   ├── graph.py      # Neo4j 图搜索
│       │   │   ├── turns.py      # 对话轮次
│       │   │   ├── ailearn.py    # AI 学习
│       │   │   └── ...
│       │   └── utils/
│       │       └── memory.py     # Mem0 客户端工具
├── openmemory.db                 # SQLite 主数据库
├── mem0_queue.db                 # 队列数据库（已废弃）
└── memory.db                     # 历史记录（测试数据）
```

## 开发命令

### 环境设置

```bash
cd mem0

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 安装依赖
pip install -e ".[dev]"
```

### 启动服务

```bash
# 启动 OpenMemory API 服务器
cd openmemory/api
uvicorn main:app --reload --port 8000

# 或使用 Makefile
make dev
```

### 测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_memory.py

# 带覆盖率报告
pytest --cov=mem0 --cov-report=html
```

### 代码质量

```bash
# Lint
ruff check

# 格式化
ruff format

# 类型检查
mypy .
```

## API 端点

### 记忆管理

```bash
# 添加记忆
POST /api/v1/memories
{
  "text": "用户喜欢编程",
  "user_id": "yishu",
  "metadata": { "category": "preference" }
}

# 搜索记忆（向量搜索）
GET /api/v1/memories/search?query=用户爱好&limit=5

# 获取所有记忆
GET /api/v1/memories?user_id=yishu

# 删除记忆
DELETE /api/v1/memories/{memory_id}
```

### 图搜索 (Neo4j)

```bash
# 图遍历查询
POST /api/v1/graph/search
{
  "query": "MATCH (n:person {name: 'yishu'})-[*..2]-(m) RETURN m",
  "user_id": "yishu"
}
```

### 对话轮次

```bash
# 保存对话轮次
POST /api/v1/turns
{
  "session_id": "session-123",
  "user_id": "yishu",
  "messages": [...],
  "source": "openclaw"
}

# 获取对话历史
GET /api/v1/turns?session_id=session-123
```

## 数据库架构

### openmemory.db (SQLite)

| 表名 | 用途 | 关键字段 |
|------|------|----------|
| `users` | 用户账户 | user_id, name, email |
| `apps` | 应用管理 | owner_id, name, is_active |
| `memories` | 记忆元数据 | user_id, app_id, content, state |
| `turns` | 对话轮次 | session_id, user_id, messages (JSON) |
| `categories` | 记忆分类 | name, description |
| `memory_categories` | 多对多关联 | memory_id, category_id |
| `configs` | 系统配置 | key, value (JSON) |
| `access_controls` | 访问控制 | subject_type, object_id, effect |

### Qdrant 向量数据库

- **Collection**: `memories`
- **向量维度**: 1536
- **距离度量**: Cosine

Payload 结构：
```json
{
  "user_id": "yishu",
  "data": "记忆内容",
  "memory_type": "fact",  // "fact" | "summary"
  "turn_id": "uuid",
  "agent_id": "agent-name",
  "created_at": "ISO-8601"
}
```

### Neo4j 图数据库

**连接**: `bolt://127.0.0.1:7687` (用户: neo4j, 密码: mem0password)

主要节点类型：
- `person` - 用户实体
- `event` - 事件
- `concept` - 概念
- `agent_id` - Agent 标识

## 配置

### 环境变量

```bash
# LLM 配置
LLM_PROVIDER=openai          # 或 ollama
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...

# 嵌入模型配置
EMBEDDER_PROVIDER=openai
EMBEDDER_MODEL=text-embedding-3-small

# 向量数据库
QDRANT_HOST=127.0.0.1
QDRANT_PORT=6333
QDRANT_COLLECTION=memories

# 图数据库
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=mem0password

# API 服务
PORT=8000
```

### Docker 服务

```bash
# 启动依赖服务（Qdrant + Neo4j）
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f qdrant
docker-compose logs -f neo4j
```

## 常见问题

### Qdrant 连接失败

```bash
# 检查 Qdrant 是否运行
curl http://127.0.0.1:6333/collections

# 重启 Qdrant
docker restart qdrant
```

### Neo4j 连接失败

```bash
# 检查 Neo4j 是否运行
docker exec neo4j-mem0 cypher-shell -u neo4j -p mem0password "RETURN 1"

# 查看 Neo4j 日志
docker logs neo4j-mem0
```

### 向量索引未构建

```bash
# 检查索引状态
curl http://127.0.0.1:6333/collections/memories | jq '.result.indexed_vectors_count'

# 创建索引（如果需要）
# Qdrant 会在后台自动索引
```

## 清理测试数据

```bash
# 清理 memory.db 测试历史
sqlite3 memory.db "DELETE FROM memory_history;"

# 清理 openmemory.db 测试 turns
sqlite3 openmemory.db "DELETE FROM turns WHERE source LIKE '%test%';"

# 删除废弃的队列数据库
rm mem0_queue.db

# 查看 Qdrant 测试数据
curl -s -X POST "http://127.0.0.1:6333/collections/memories/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{"limit": 100}' | jq '.result.points[] | {id, payload}'
```

## 开发工作流

1. **特性开发**: 创建功能分支 → 编写测试 → 实现功能 → 运行测试
2. **代码审查**: 提交 PR 前运行 `ruff check` 和 `pytest`
3. **提交规范**: 使用 Conventional Commits (`feat:`, `fix:`, `docs:`)

## 相关文档

- [API 文档](http://localhost:8000/docs) - FastAPI 自动生成的 Swagger UI
- [Qdrant 文档](https://qdrant.tech/documentation/)
- [Neo4j 文档](https://neo4j.com/docs/)
- [根目录 CLAUDE.md](../CLAUDE.md) - 项目整体架构
