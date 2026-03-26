/**
 * Qdrant vector store implementation.
 * Replicates mem0ai's Qdrant vector store behavior in pure TypeScript.
 */

import { QdrantClient } from "@qdrant/js-client-rest";

// ============================================================================
// Types
// ============================================================================

export interface VectorSearchResult {
  id: string;
  payload: Record<string, unknown>;
  score: number;
}

export interface VectorStoreConfig {
  url?: string;
  apiKey?: string;
  collectionName: string;
  dimension?: number;
}

// ============================================================================
// Qdrant Vector Store
// ============================================================================

export class QdrantVectorStore {
  private client: QdrantClient;
  private collectionName: string;
  private dimension: number;
  private _initPromise: Promise<void> | null = null;

  constructor(config: VectorStoreConfig) {
    const params: Record<string, unknown> = {};

    if (config.url) {
      params.url = config.url;
    }
    if (config.apiKey) {
      params.apiKey = config.apiKey;
    }

    this.client = new QdrantClient(params);
    this.collectionName = config.collectionName || "memories";
    this.dimension = config.dimension || 1536;
  }

  // --------------------------------------------------------------------------
  // Initialization
  // --------------------------------------------------------------------------

  async initialize(): Promise<void> {
    if (!this._initPromise) {
      this._initPromise = this._doInitialize();
    }
    return this._initPromise;
  }

  private async _doInitialize(): Promise<void> {
    try {
      await this.ensureCollection(this.collectionName, this.dimension);
      // memory_migrations collection for backward compatibility
      await this.ensureCollection("memory_migrations", 1);
    } catch (error) {
      console.error("[qdrant] Initialization error:", error);
      throw error;
    }
  }

  private async ensureCollection(name: string, size: number): Promise<void> {
    try {
      await this.client.createCollection(name, {
        vectors: {
          size,
          distance: "Cosine",
        },
      });
    } catch (error: unknown) {
      const err = error as { status?: number; message?: string };
      if (err?.status === 409) {
        // Collection already exists - verify dimension
        if (name === this.collectionName) {
          try {
            const collectionInfo = await this.client.getCollection(name);
            const vectorConfig =
              collectionInfo.config?.params?.vectors;
            if (vectorConfig && vectorConfig.size !== size) {
              throw new Error(
                `Collection ${name} exists but has wrong vector size. Expected: ${size}, got: ${vectorConfig.size}`,
              );
            }
          } catch (verifyError: unknown) {
            const ve = verifyError as { message?: string };
            if (ve?.message?.includes("wrong vector size")) {
              throw verifyError;
            }
            console.warn(
              `[qdrant] Collection '${name}' exists but dimension verification failed: ${ve?.message || verifyError}. Proceeding anyway.`,
            );
          }
        }
      } else {
        throw error;
      }
    }
  }

  // --------------------------------------------------------------------------
  // Filter helpers
  // --------------------------------------------------------------------------

  private createFilter(
    filters?: Record<string, unknown>,
  ): Record<string, unknown> | undefined {
    if (!filters) return undefined;

    const conditions: Array<Record<string, unknown>> = [];

    for (const [key, value] of Object.entries(filters)) {
      if (
        typeof value === "object" &&
        value !== null &&
        "gte" in value &&
        "lte" in value
      ) {
        conditions.push({
          key,
          range: {
            gte: (value as { gte: unknown }).gte,
            lte: (value as { lte: unknown }).lte,
          },
        });
      } else {
        conditions.push({
          key,
          match: { value },
        });
      }
    }

    return conditions.length ? { must: conditions } : undefined;
  }

  // --------------------------------------------------------------------------
  // CRUD operations
  // --------------------------------------------------------------------------

  /**
   * Insert vectors with payloads.
   */
  async insert(
    vectors: number[][],
    ids: string[],
    payloads: Array<Record<string, unknown>>,
  ): Promise<void> {
    const points = vectors.map((vector, idx) => ({
      id: ids[idx],
      vector,
      payload: payloads[idx] || {},
    }));

    await this.client.upsert(this.collectionName, { points });
  }

  /**
   * Search for similar vectors.
   */
  async search(
    query: number[],
    limit: number = 5,
    filters?: Record<string, unknown>,
  ): Promise<VectorSearchResult[]> {
    const queryFilter = this.createFilter(filters);

    const results = await this.client.search(this.collectionName, {
      vector: query,
      filter: queryFilter,
      limit,
    });

    return results.map((hit) => ({
      id: String(hit.id),
      payload: hit.payload || {},
      score: hit.score,
    }));
  }

  /**
   * Get a single vector by ID.
   */
  async get(vectorId: string): Promise<VectorSearchResult | null> {
    const results = await this.client.retrieve(this.collectionName, {
      ids: [vectorId],
      with_payload: true,
    });

    if (!results.length) return null;

    return {
      id: vectorId,
      payload: results[0].payload || {},
      score: 0, // No score for get operations
    };
  }

  /**
   * Update a vector and its payload.
   */
  async update(
    vectorId: string,
    vector: number[],
    payload: Record<string, unknown>,
  ): Promise<void> {
    await this.client.upsert(this.collectionName, {
      points: [{ id: vectorId, vector, payload }],
    });
  }

  /**
   * Delete a vector by ID.
   */
  async delete(vectorId: string): Promise<void> {
    await this.client.delete(this.collectionName, {
      points: [vectorId],
    });
  }

  /**
   * List all vectors with optional filters.
   */
  async list(
    filters?: Record<string, unknown>,
    limit: number = 100,
  ): Promise<[VectorSearchResult[], number]> {
    const response = await this.client.scroll(this.collectionName, {
      limit,
      filter: this.createFilter(filters),
      with_payload: true,
      with_vectors: false,
    });

    const results = response.points.map((point) => ({
      id: String(point.id),
      payload: point.payload || {},
      score: 0,
    }));

    return [results, response.points.length];
  }

  /**
   * Generate a UUID v4.
   */
  generateUUID(): string {
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(
      /[xy]/g,
      (c) => {
        const r = (Math.random() * 16) | 0;
        const v = c === "x" ? r : (r & 3) | 8;
        return v.toString(16);
      },
    );
  }
}
