#!/usr/bin/env bash
# 测试 mem0 API 是否返回 source 字段

API_URL="${MEM0_API_URL:-http://localhost:8765}"

echo "测试 /api/v1/memories/ API 是否返回 source 字段..."
echo "API URL: $API_URL"
echo ""

# 获取前 5 条记忆
response=$(curl -s "$API_URL/api/v1/memories/?user_id=yishu&limit=5")

# 检查是否有 source 字段
echo "响应示例:"
echo "$response" | jq '.items[0]' 2>/dev/null || echo "JSON 解析失败"

# 检查 source 字段
has_source=$(echo "$response" | jq '.items[0].source' 2>/dev/null)

if [[ "$has_source" != "null" ]] && [[ -n "$has_source" ]]; then
    echo ""
    echo "✅ 成功: API 返回了 source 字段"
    echo "Source 值: $has_source"
else
    echo ""
    echo "❌ 失败: API 没有返回 source 字段"
    echo "请检查:"
    echo "  1. 后端服务是否重启"
    echo "  2. 代码修改是否正确应用"
fi

# 统计所有记忆的 source 分布
echo ""
echo "所有记忆的 source 分布:"
echo "$response" | jq -r '.items[] | .source // "null"' | sort | uniq -c
