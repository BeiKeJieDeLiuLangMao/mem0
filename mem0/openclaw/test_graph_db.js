#!/usr/bin/env node

/**
 * 测试 Mem0 OSS 模式下的图数据库连接
 */

const { Memory } = require('mem0ai/oss');

async function testGraphDB() {
  const config = {
    version: 'v1.1',
    embedder: {
      provider: 'openai',
      config: {
        model: 'text-embedding-3-small',
        embeddingDims: 1536,
        apiKey: 'fai-2-977-cdc6435fbca2',
        baseURL: 'https://trip-llm.alibaba-inc.com/api/openai/v1/'
      }
    },
    llm: {
      provider: 'openai',
      config: {
        model: 'gpt-5-mini',
        apiKey: 'fai-2-977-cdc6435fbca2',
        baseURL: 'https://trip-llm.alibaba-inc.com/api/fai/v1'
      }
    },
    vectorStore: {
      provider: 'qdrant',
      config: {
        location: 'http://127.0.0.1:6333',
        collectionName: 'memories'
      }
    },
    graphStore: {
      provider: 'neo4j',
      config: {
        url: 'bolt://localhost:7687',
        username: 'neo4j',
        password: 'mem0password'
      }
    },
    disableHistory: true
  };

  console.log('🔧 初始化 Memory 类...');
  const memory = new Memory(config);

  // 等待初始化完成
  await new Promise(resolve => setTimeout(resolve, 2000));

  console.log('✅ Memory 类创建成功');
  console.log(`📊 enableGraph: ${memory.enableGraph}`);
  console.log(`📊 graphMemory: ${memory.graphMemory ? '已初始化' : '未初始化'}`);

  // 测试添加记忆
  console.log('\n📝 添加测试记忆...');
  try {
    const result = await memory.add(
      '我是逸殊，我的同事是枫日，我们在做 OpenClaw Mem0 集成项目',
      {
        userId: 'yishu:agent:main',
        metadata: { source: 'graph_test' }
      }
    );

    console.log('✅ 记忆添加成功');
    console.log('📋 结果:', JSON.stringify(result, null, 2));
  } catch (error) {
    console.error('❌ 添加记忆失败:', error.message);
    console.error('详细错误:', error);
  }

  // 搜索记忆
  console.log('\n🔍 搜索记忆...');
  try {
    const searchResult = await memory.search('逸殊的同事是谁？', {
      userId: 'yishu:agent:main',
      limit: 5
    });

    console.log('✅ 搜索成功');
    console.log('📋 搜索结果:', JSON.stringify(searchResult, null, 2));
  } catch (error) {
    console.error('❌ 搜索失败:', error.message);
  }

  console.log('\n✅ 测试完成');
}

testGraphDB().catch(console.error);
