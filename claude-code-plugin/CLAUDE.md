# CLAUDE.md - Mem0 Claude Code Plugin

## 概述

这是一个 Claude Code 插件，通过 hooks 将对话存储到 mem0 openmemory 系统。

## 架构

- **召回**: `UserPromptSubmit` hook - 用户发出消息后，基于输入查询相关记忆
- **存储**: `Stop` hook - 每轮回复后，将对话存储到 turns 表

## 文件结构

```
claude-code-plugin/
├── config.sh           # 配置（API URL 等）
├── lib/api.sh         # HTTP API 封装
├── mem0-retrieve.sh   # 召回 Hook
├── mem0-store.sh      # 存储 Hook
├── install.sh         # 安装脚本
└── README.md          # 使用说明
```

## 开发命令

```bash
# 测试 API 连接
curl http://localhost:8765/health

# 手动测试召回
echo '{"message":{"role":"user","content":"测试消息"}}' | bash mem0-retrieve.sh

# 测试存储（需要 transcript 文件）
echo '{"transcript_path":"/tmp/test_transcript.jsonl"}' | bash mem0-store.sh
```

## 配置

通过环境变量覆盖：
- `MEM0_API_URL` - API 地址（默认: http://localhost:8765）
- `MEM0_DEBUG=1` - 启用调试日志
