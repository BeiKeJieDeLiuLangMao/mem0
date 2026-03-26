/**
 * Shared type definitions for the OpenClaw Mem0 plugin.
 * Open-Source mode only — no cloud dependencies.
 */

export type Mem0Mode = "open-source";

export type Mem0Config = {
  mode: Mem0Mode;
  // OSS-specific
  customPrompt?: string;
  customCategories?: Record<string, string>;
  enableGraph?: boolean;
  oss?: {
    embedder?: { provider: string; config: Record<string, unknown> };
    vectorStore?: { provider: string; config: Record<string, unknown> };
    llm?: { provider: string; config: Record<string, unknown> };
    graphStore?: { provider: string; config: Record<string, unknown> };
    historyDbPath?: string;
    disableHistory?: boolean;
  };
  // Shared
  userId: string;
  autoCapture: boolean;
  autoRecall: boolean;
  searchThreshold: number;
  topK: number;
};

export interface AddOptions {
  user_id: string;
  agent_id?: string;
  run_id?: string;
  source?: string;
}

export interface SearchOptions {
  user_id: string;
  run_id?: string;
  top_k?: number;
  threshold?: number;
  limit?: number;
  keyword_search?: boolean;
  reranking?: boolean;
  source?: string;
}

export interface ListOptions {
  user_id: string;
  run_id?: string;
  page_size?: number;
  source?: string;
}

export interface MemoryItem {
  id: string;
  memory: string;
  user_id?: string;
  score?: number;
  categories?: string[];
  metadata?: Record<string, unknown>;
  created_at?: string;
  updated_at?: string;
}

export interface AddResultItem {
  id: string;
  memory: string;
  event: "ADD" | "UPDATE" | "DELETE" | "NOOP";
}

export interface AddResult {
  results: AddResultItem[];
}

export interface Mem0Provider {
  add(
    messages: Array<{ role: string; content: string }>,
    options: AddOptions,
  ): Promise<AddResult>;
  search(query: string, options: SearchOptions): Promise<MemoryItem[]>;
  get(memoryId: string): Promise<MemoryItem>;
  getAll(options: ListOptions): Promise<MemoryItem[]>;
  delete(memoryId: string): Promise<void>;
}

// ============================================================================
// OSS Config (for pure TypeScript implementation)
// ============================================================================

export interface OSSEmbedderConfig {
  provider: string;
  config: {
    apiKey?: string;
    baseURL?: string;
    model?: string;
    embeddingDims?: number;
    [key: string]: unknown;
  };
}

export interface OSSLLMConfig {
  provider: string;
  config: {
    apiKey?: string;
    baseURL?: string;
    model?: string;
    [key: string]: unknown;
  };
}

export interface OSSVectorStoreConfig {
  provider: string;
  config: {
    url?: string;
    collectionName?: string;
    apiKey?: string;
    dimension?: number;
    [key: string]: unknown;
  };
}

export interface OSSGraphStoreConfig {
  provider: string;
  config: {
    url?: string;
    username?: string;
    password?: string;
    [key: string]: unknown;
  };
  customPrompt?: string;
  llm?: OSSLLMConfig;
}

export interface OSSConfig {
  embedder?: OSSEmbedderConfig;
  llm?: OSSLLMConfig;
  vectorStore?: OSSVectorStoreConfig;
  graphStore?: OSSGraphStoreConfig;
  historyDbPath?: string;
  disableHistory?: boolean;
  customPrompt?: string;
}
