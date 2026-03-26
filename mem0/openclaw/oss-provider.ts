/**
 * OSS Provider - Pure TypeScript implementation.
 * Replicates mem0ai's Memory class behavior without depending on mem0ai.
 */

import { v4 as uuidv4 } from "uuid";
import type {
  AddOptions,
  SearchOptions,
  ListOptions,
  MemoryItem,
  AddResult,
  AddResultItem,
} from "./types.ts";
import type { Mem0Provider } from "./types.ts";
import { QdrantVectorStore } from "./vector-store.ts";
import { Neo4jGraphStore } from "./graph-store.ts";
import {
  OpenAIEmbedder,
  OpenAILLM,
  EntityExtractor,
} from "./entity-extractor.ts";
import {
  getFactRetrievalMessages,
  removeCodeBlocks,
} from "./prompts.ts";
import type { OSSConfig } from "./types.ts";

// ============================================================================
// Types
// ============================================================================

interface MemoryPayload {
  data: string;
  hash: string;
  userId?: string;
  agentId?: string;
  runId?: string;
  createdAt?: string;
  updatedAt?: string;
  [key: string]: unknown;
}

interface MemoryAction {
  id: string;
  text: string;
  event: "ADD" | "UPDATE" | "DELETE" | "NONE";
  old_memory?: string;
}

interface SearchResult {
  results: MemoryItem[];
  relations?: unknown[];
}

// ============================================================================
// OSS Provider (Pure TypeScript Memory)
// ============================================================================

export class OSSProvider implements Mem0Provider {
  private vectorStore: QdrantVectorStore | null = null;
  private graphStore: Neo4jGraphStore | null = null;
  private embedder: OpenAIEmbedder | null = null;
  private llm: OpenAILLM | null = null;
  private entityExtractor: EntityExtractor | null = null;
  private config: OSSConfig;
  private _initPromise: Promise<void> | null = null;
  private _initError: Error | null = null;
  private customPrompt: string | undefined;

  constructor(config: OSSConfig) {
    this.config = config;
    this.customPrompt = config.customPrompt;
  }

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------

  private async ensureInitialized(): Promise<void> {
    if (this._initPromise) {
      await this._initPromise;
      if (this._initError) {
        this._initError = null;
        this._initPromise = this._doInitialize().catch((err) => {
          this._initError = err instanceof Error ? err : new Error(String(err));
          throw this._initError;
        });
        await this._initPromise;
        if (this._initError) throw this._initError;
      }
      return;
    }
    this._initPromise = this._doInitialize().catch((err) => {
      this._initError = err instanceof Error ? err : new Error(String(err));
      throw this._initError;
    });
    await this._initPromise;
  }

  private async _doInitialize(): Promise<void> {
    // Initialize embedder
    const embedderConfig = this.config.embedder?.config as {
      apiKey?: string;
      baseURL?: string;
      model?: string;
      embeddingDims?: number;
    } | undefined;

    this.embedder = new OpenAIEmbedder({
      apiKey: embedderConfig?.apiKey,
      baseURL: embedderConfig?.baseURL,
      model: embedderConfig?.model || "text-embedding-3-small",
      embeddingDims: embedderConfig?.embeddingDims || 1536,
    });

    // Auto-detect embedding dimension if not provided
    let dimension = embedderConfig?.embeddingDims;
    if (!dimension) {
      try {
        const probe = await this.embedder.embed("dimension probe");
        dimension = probe.length;
      } catch (error) {
        throw new Error(
          `Failed to auto-detect embedding dimension: ${error instanceof Error ? error.message : error}. Please set 'embeddingDims' in embedder.config explicitly.`,
        );
      }
    }

    // Initialize LLM
    const llmConfig = this.config.llm?.config as {
      apiKey?: string;
      baseURL?: string;
      model?: string;
    } | undefined;

    this.llm = new OpenAILLM({
      apiKey: llmConfig?.apiKey,
      baseURL: llmConfig?.baseURL,
      model: llmConfig?.model || "gpt-4o-mini",
    });

    // Initialize entity extractor
    this.entityExtractor = new EntityExtractor(this.llm);

    // Initialize vector store
    const vectorConfig = this.config.vectorStore?.config as {
      url?: string;
      collectionName?: string;
      apiKey?: string;
    } | undefined;

    this.vectorStore = new QdrantVectorStore({
      url: vectorConfig?.url || "http://127.0.0.1:6333",
      collectionName: (vectorConfig?.collectionName as string) || "memories",
      apiKey: vectorConfig?.apiKey,
      dimension,
    });

    await this.vectorStore.initialize();

    // Initialize graph store if configured
    const graphConfig = this.config.graphStore?.config as {
      url?: string;
      username?: string;
      password?: string;
    } | undefined;

    if (graphConfig?.url && graphConfig?.username && graphConfig?.password) {
      this.graphStore = new Neo4jGraphStore({
        url: graphConfig.url,
        username: graphConfig.username,
        password: graphConfig.password,
      });
      // Set embedder for graph store
      this.graphStore.setEmbedder(this.embedder);
    }
  }

  // --------------------------------------------------------------------------
  // Memory operations
  // --------------------------------------------------------------------------

  async add(
    messages: Array<{ role: string; content: string }>,
    options: AddOptions,
  ): Promise<AddResult> {
    await this.ensureInitialized();

    const filters: Record<string, unknown> = {};
    const metadata: Record<string, unknown> = {};

    if (options.user_id) {
      filters.userId = options.user_id;
      metadata.userId = options.user_id;
    }
    if (options.agent_id) {
      filters.agentId = options.agent_id;
      metadata.agentId = options.agent_id;
    }
    if (options.run_id) {
      filters.runId = options.run_id;
      metadata.runId = options.run_id;
    }

    if (!options.user_id && !options.agent_id && !options.run_id) {
      throw new Error(
        "One of user_id, agent_id, or run_id is required!",
      );
    }

    // Parse messages
    const parsedMessages = Array.isArray(messages)
      ? messages
      : [{ role: "user", content: messages }];

    // Step 1: Extract facts from messages
    const facts = await this.extractFacts(parsedMessages, options.user_id);

    // Step 2: Process facts with vector store (add/update/delete)
    const results = await this.processFacts(facts, metadata, filters);

    // Step 3: Extract entities and relationships for graph store
    if (this.graphStore && options.user_id) {
      try {
        const textContent = parsedMessages.map((m) => m.content).join("\n");
        const entityMap = await this.entityExtractor!.extractEntities(
          textContent,
          options.user_id,
        );
        const relations = await this.entityExtractor!.extractRelations(
          textContent,
          entityMap,
          options.user_id,
          this.config.graphStore?.customPrompt,
        );

        // Add to graph store
        await this.graphStore.addEntities(
          relations,
          options.user_id,
          entityMap,
        );
      } catch (error) {
        console.error("[oss-provider] Error adding to graph store:", error);
      }
    }

    return { results };
  }

  private async extractFacts(
    messages: Array<{ role: string; content: string }>,
    userId?: string,
  ): Promise<string[]> {
    if (!this.llm) throw new Error("LLM not initialized");

    const parsedMessages = messages.map((m) => m.content).join("\n");

    let systemPrompt: string;
    let userPrompt: string;

    if (this.customPrompt) {
      systemPrompt = this.customPrompt.toLowerCase().includes("json")
        ? this.customPrompt
        : `${this.customPrompt}

You MUST return a valid JSON object with a 'facts' key containing an array of strings.`;
      userPrompt = `Input:\n${parsedMessages}`;
    } else {
      const [sys, user] = getFactRetrievalMessages(messages);
      systemPrompt = sys.content;
      userPrompt = user.content;
    }

    const response = await this.llm.generateResponse(
      [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      { type: "json_object" },
    );

    const cleanResponse = removeCodeBlocks(response.content || "");
    try {
      const parsed = JSON.parse(cleanResponse);
      return parsed.facts || [];
    } catch {
      console.error("[oss-provider] Failed to parse facts:", cleanResponse);
      return [];
    }
  }

  private async processFacts(
    facts: string[],
    metadata: Record<string, unknown>,
    filters: Record<string, unknown>,
  ): Promise<AddResultItem[]> {
    if (!this.vectorStore || !this.embedder) {
      throw new Error("Vector store not initialized");
    }

    const results: AddResultItem[] = [];

    for (const fact of facts) {
      // Embed the fact
      const embedding = await this.embedder.embed(fact);

      // Search for existing similar memories
      const existingMemories = await this.vectorStore.search(
        embedding,
        5,
        filters,
      );

      // Check if similar memory exists (for deduplication)
      const existingFact = existingMemories.find(
        (mem) => mem.score > 0.95 && (mem.payload.data as string) === fact,
      );

      if (existingFact) {
        // Skip if very similar memory already exists
        continue;
      }

      // Add new memory
      const memoryId = uuidv4();
      const payload: MemoryPayload = {
        data: fact,
        hash: this.hashText(fact),
        ...metadata,
        createdAt: new Date().toISOString(),
      };

      await this.vectorStore.insert([embedding], [memoryId], [payload]);

      results.push({
        id: memoryId,
        memory: fact,
        event: "ADD",
      });
    }

    return results;
  }

  private hashText(text: string): string {
    // Simple hash for deduplication
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = (hash << 5) - hash + char;
      hash = hash & hash;
    }
    return Math.abs(hash).toString(16);
  }

  async search(query: string, options: SearchOptions): Promise<MemoryItem[]> {
    await this.ensureInitialized();

    if (!this.vectorStore || !this.embedder) {
      throw new Error("Vector store not initialized");
    }

    const filters: Record<string, unknown> = {};
    if (options.user_id) filters.userId = options.user_id;
    if (options.agent_id) filters.agentId = options.agent_id;
    if (options.run_id) filters.runId = options.run_id;

    if (!options.user_id && !options.agent_id && !options.run_id) {
      throw new Error(
        "One of user_id, agent_id, or run_id is required!",
      );
    }

    // Embed the query
    const queryEmbedding = await this.embedder.embed(query);

    // Search vector store
    const memories = await this.vectorStore.search(
      queryEmbedding,
      options.limit ?? options.top_k ?? 100,
      filters,
    );

    // Apply threshold filter
    let threshold = options.threshold ?? 0;
    const results = memories
      .filter((mem) => mem.score >= threshold)
      .map((mem) => this.mapToMemoryItem(mem));

    return results;
  }

  async get(memoryId: string): Promise<MemoryItem> {
    await this.ensureInitialized();

    if (!this.vectorStore) {
      throw new Error("Vector store not initialized");
    }

    const memory = await this.vectorStore.get(memoryId);
    if (!memory) {
      throw new Error(`Memory not found: ${memoryId}`);
    }

    return this.mapToMemoryItem(memory);
  }

  async getAll(options: ListOptions): Promise<MemoryItem[]> {
    await this.ensureInitialized();

    if (!this.vectorStore) {
      throw new Error("Vector store not initialized");
    }

    const filters: Record<string, unknown> = {};
    if (options.user_id) filters.userId = options.user_id;
    if (options.agent_id) filters.agentId = options.agent_id;
    if (options.run_id) filters.runId = options.run_id;

    const [memories] = await this.vectorStore.list(
      filters,
      options.page_size ?? 100,
    );

    return memories.map((mem) => this.mapToMemoryItem(mem));
  }

  async delete(memoryId: string): Promise<void> {
    await this.ensureInitialized();

    if (!this.vectorStore) {
      throw new Error("Vector store not initialized");
    }

    await this.vectorStore.delete(memoryId);
  }

  // --------------------------------------------------------------------------
  // Helpers
  // --------------------------------------------------------------------------

  private mapToMemoryItem(mem: {
    id: string;
    payload: Record<string, unknown>;
    score?: number;
  }): MemoryItem {
    const payload = mem.payload as MemoryPayload;
    const excludedKeys = new Set([
      "userId",
      "agentId",
      "runId",
      "hash",
      "data",
      "createdAt",
      "updatedAt",
    ]);

    const metadata: Record<string, unknown> = {};
    for (const [key, value] of Object.entries(payload)) {
      if (!excludedKeys.has(key)) {
        metadata[key] = value;
      }
    }

    return {
      id: mem.id,
      memory: payload.data || "",
      user_id: payload.userId,
      score: mem.score,
      metadata,
      created_at: payload.createdAt,
      updated_at: payload.updatedAt,
    };
  }
}
