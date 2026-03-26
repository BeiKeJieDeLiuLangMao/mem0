/**
 * Neo4j graph store implementation.
 * Replicates mem0ai's graph store behavior in pure TypeScript.
 */

import neo4j, { Driver, Session } from "neo4j-driver";

// ============================================================================
// Types
// ============================================================================

export interface GraphEntity {
  source: string;
  relationship: string;
  destination: string;
}

export interface GraphRelation {
  source: string;
  source_id: string;
  relationship: string;
  relation_id: string;
  destination: string;
  destination_id: string;
  similarity: number;
}

export interface GraphStoreConfig {
  url: string;
  username: string;
  password: string;
  embedder?: {
    provider: string;
    config: Record<string, unknown>;
  };
  llm?: {
    provider: string;
    config: Record<string, unknown>;
  };
  threshold?: number;
}

// ============================================================================
// Neo4j Graph Store
// ============================================================================

export class Neo4jGraphStore {
  private driver: Driver;
  private embedder: { embed(text: string): Promise<number[]> };
  private threshold: number;

  constructor(config: GraphStoreConfig) {
    if (!config.url || !config.username || !config.password) {
      throw new Error("Neo4j configuration is incomplete");
    }

    this.driver = neo4j.driver(
      config.url,
      neo4j.auth.basic(config.username, config.password),
    );

    this.threshold = config.threshold ?? 0.7;

    // Initialize embedder - in the full implementation this would be
    // initialized with the actual embedder from config
    // For now we create a placeholder
    this.embedder = {
      async embed(text: string): Promise<number[]> {
        // This should be replaced with actual embedding call
        // The embedder is typically passed through config
        throw new Error("Embedder not initialized. Call setEmbedder() first.");
      },
    };
  }

  /**
   * Set the embedder after construction (allows async initialization).
   */
  setEmbedder(embedder: { embed(text: string): Promise<number[]> }): void {
    this.embedder = embedder;
  }

  /**
   * Normalize entity names (lowercase, replace spaces with underscores).
   */
  private normalizeEntity(entity: string): string {
    return entity.toLowerCase().replace(/ /g, "_");
  }

  /**
   * Normalize a list of graph entities.
   */
  normalizeEntities(
    entities: GraphEntity[],
  ): GraphEntity[] {
    return entities.map((item) => ({
      ...item,
      source: this.normalizeEntity(item.source),
      relationship: this.normalizeEntity(item.relationship),
      destination: this.normalizeEntity(item.destination),
    }));
  }

  /**
   * Add entities to the graph.
   */
  async addEntities(
    entities: GraphEntity[],
    userId: string,
    entityTypeMap: Record<string, string>,
  ): Promise<void> {
    const session = this.driver.session();
    try {
      for (const item of entities) {
        const { source, destination, relationship } = item;
        const sourceType = entityTypeMap[source] || "unknown";
        const destType = entityTypeMap[destination] || "unknown";

        const sourceEmbedding = await this.embedder.embed(source);
        const destEmbedding = await this.embedder.embed(destination);

        const sourceNodeResult = await this.searchNodeByEmbedding(
          sourceEmbedding,
          userId,
          0.9,
        );
        const destNodeResult = await this.searchNodeByEmbedding(
          destEmbedding,
          userId,
          0.9,
        );

        let cypher: string;
        let params: Record<string, unknown>;

        if (
          destNodeResult.length === 0 && sourceNodeResult.length > 0
        ) {
          // Destination is new, source exists
          cypher = `
            MATCH (source)
            WHERE elementId(source) = $source_id
            MERGE (destination:${destType} {name: $destination_name, user_id: $user_id})
            ON CREATE SET
                destination.created = timestamp(),
                destination.embedding = $destination_embedding
            MERGE (source)-[r:${relationship}]->(destination)
            ON CREATE SET r.created = timestamp()
            RETURN source.name AS source, type(r) AS relationship, destination.name AS target
          `;
          params = {
            source_id: sourceNodeResult[0].elementId,
            destination_name: destination,
            destination_embedding: destEmbedding,
            user_id: userId,
          };
        } else if (
          destNodeResult.length > 0 && sourceNodeResult.length === 0
        ) {
          // Source is new, destination exists
          cypher = `
            MATCH (destination)
            WHERE elementId(destination) = $destination_id
            MERGE (source:${sourceType} {name: $source_name, user_id: $user_id})
            ON CREATE SET
                source.created = timestamp(),
                source.embedding = $source_embedding
            MERGE (source)-[r:${relationship}]->(destination)
            ON CREATE SET r.created = timestamp()
            RETURN source.name AS source, type(r) AS relationship, destination.name AS target
          `;
          params = {
            destination_id: destNodeResult[0].elementId,
            source_name: source,
            source_embedding: sourceEmbedding,
            user_id: userId,
          };
        } else if (
          sourceNodeResult.length > 0 && destNodeResult.length > 0
        ) {
          // Both nodes exist
          cypher = `
            MATCH (source)
            WHERE elementId(source) = $source_id
            MATCH (destination)
            WHERE elementId(destination) = $destination_id
            MERGE (source)-[r:${relationship}]->(destination)
            ON CREATE SET
                r.created_at = timestamp(),
                r.updated_at = timestamp()
            RETURN source.name AS source, type(r) AS relationship, destination.name AS target
          `;
          params = {
            source_id: sourceNodeResult[0]?.elementId,
            destination_id: destNodeResult[0]?.elementId,
            user_id: userId,
          };
        } else {
          // Both nodes are new
          cypher = `
            MERGE (n:${sourceType} {name: $source_name, user_id: $user_id})
            ON CREATE SET n.created = timestamp(), n.embedding = $source_embedding
            ON MATCH SET n.embedding = $source_embedding
            MERGE (m:${destType} {name: $dest_name, user_id: $user_id})
            ON CREATE SET m.created = timestamp(), m.embedding = $dest_embedding
            ON MATCH SET m.embedding = $dest_embedding
            MERGE (n)-[rel:${relationship}]->(m)
            ON CREATE SET rel.created = timestamp()
            RETURN n.name AS source, type(rel) AS relationship, m.name AS target
          `;
          params = {
            source_name: source,
            dest_name: destination,
            source_embedding: sourceEmbedding,
            dest_embedding: destEmbedding,
            user_id: userId,
          };
        }

        await session.run(cypher, params);
      }
    } finally {
      await session.close();
    }
  }

  /**
   * Search for a node by its embedding vector.
   */
  private async searchNodeByEmbedding(
    embedding: number[],
    userId: string,
    threshold: number = 0.9,
  ): Promise<Array<{ elementId: string }>> {
    const session = this.driver.session();
    try {
      const cypher = `
        MATCH (candidate)
        WHERE candidate.embedding IS NOT NULL
        AND candidate.user_id = $user_id

        WITH candidate,
            round(
                reduce(dot = 0.0, i IN range(0, size(candidate.embedding)-1) |
                    dot + candidate.embedding[i] * $embedding[i]) /
                (sqrt(reduce(l2 = 0.0, i IN range(0, size(candidate.embedding)-1) |
                    l2 + candidate.embedding[i] * candidate.embedding[i])) *
                sqrt(reduce(l2 = 0.0, i IN range(0, size($embedding)-1) |
                    l2 + $embedding[i] * $embedding[i])))
                , 4) AS similarity
        WHERE similarity >= $threshold

        WITH candidate, similarity
        ORDER BY similarity DESC
        LIMIT 1

        RETURN elementId(candidate) as element_id
      `;

      const result = await session.run(cypher, {
        embedding,
        user_id: userId,
        threshold,
      });

      return result.records.map((record) => ({
        elementId: record.get("element_id").toString(),
      }));
    } finally {
      await session.close();
    }
  }

  /**
   * Search the graph for related entities.
   */
  async search(
    query: string,
    userId: string,
    limit: number = 100,
  ): Promise<GraphRelation[]> {
    const session = this.driver.session();
    try {
      const queryEmbedding = await this.embedder.embed(query);
      const resultRelations: GraphRelation[] = [];

      // Search for nodes similar to the query
      const nodeSearchResult = await session.run(
        `
        MATCH (n)
        WHERE n.embedding IS NOT NULL AND n.user_id = $user_id
        WITH n,
            round(reduce(dot = 0.0, i IN range(0, size(n.embedding)-1) | dot + n.embedding[i] * $embedding[i]) /
            (sqrt(reduce(l2 = 0.0, i IN range(0, size(n.embedding)-1) | l2 + n.embedding[i] * n.embedding[i])) *
            sqrt(reduce(l2 = 0.0, i IN range(0, size($embedding)-1) | l2 + $embedding[i] * $embedding[i]))), 4) AS similarity
        WHERE similarity >= $threshold
        RETURN elementId(n) as node_id, n.name as name, similarity
        ORDER BY similarity DESC
        LIMIT toInteger($limit)
        `,
        {
          embedding: queryEmbedding,
          user_id: userId,
          threshold: this.threshold,
          limit,
        },
      );

      const nodeIds = nodeSearchResult.records.map((r) =>
        r.get("node_id").toString(),
      );

      // For each found node, get its relationships
      for (const nodeId of nodeIds) {
        const relResult = await session.run(
          `
          MATCH (n)
          WHERE elementId(n) = $node_id
          MATCH (n)-[r]->(m)
          RETURN n.name AS source, elementId(n) AS source_id, type(r) AS relationship,
                 elementId(r) AS relation_id, m.name AS destination, elementId(m) AS destination_id
          UNION ALL
          MATCH (m)-[r]->(n)
          WHERE elementId(n) = $node_id
          RETURN m.name AS source, elementId(m) AS source_id, type(r) AS relationship,
                 elementId(r) AS relation_id, n.name AS destination, elementId(n) AS destination_id
          `,
          { node_id: nodeId },
        );

        for (const record of relResult.records) {
          resultRelations.push({
            source: record.get("source"),
            source_id: record.get("source_id").toString(),
            relationship: record.get("relationship"),
            relation_id: record.get("relation_id").toString(),
            destination: record.get("destination"),
            destination_id: record.get("destination_id").toString(),
            similarity: 0, // Similarity from node search already filtered
          });
        }
      }

      return resultRelations;
    } finally {
      await session.close();
    }
  }

  /**
   * Delete a relationship from the graph.
   */
  async deleteRelation(
    source: string,
    relationship: string,
    destination: string,
    userId: string,
  ): Promise<void> {
    const session = this.driver.session();
    try {
      const cypher = `
        MATCH (n {name: $source_name, user_id: $user_id})
        -[r:${relationship}]->
        (m {name: $dest_name, user_id: $user_id})
        DELETE r
        RETURN n.name AS source, m.name AS target, type(r) AS relationship
      `;

      await session.run(cypher, {
        source_name: source,
        dest_name: destination,
        user_id: userId,
      });
    } finally {
      await session.close();
    }
  }

  /**
   * Close the driver connection.
   */
  async close(): Promise<void> {
    await this.driver.close();
  }
}
