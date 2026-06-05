import fs from "fs/promises";
import path from "path";

import { JsonObject, JsonValue, ToolExecutionResult } from "../../core/types";
import { ToolRegistry } from "../../core/toolRegistry";
import { MemoryMcpBridge } from "./backend";
import { buildAddObservationsTool } from "./addObservations";
import { buildCreateEntitiesTool } from "./createEntities";
import { buildCreateRelationsTool } from "./createRelations";
import { buildOpenNodesTool } from "./openNodes";
import { buildReadGraphTool } from "./readGraph";
import { buildSearchNodesTool } from "./searchNodes";
import { MemoryToolProxy } from "./types";

interface MemoryEntity {
  name: string;
  entityType: string;
  observations: string[];
}

interface MemoryRelation {
  from: string;
  to: string;
  relationType: string;
}

interface MemoryGraph {
  entities: MemoryEntity[];
  relations: MemoryRelation[];
}

function emptyGraph(): MemoryGraph {
  return {
    entities: [],
    relations: [],
  };
}

function sanitizeGraph(raw: unknown): MemoryGraph {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return emptyGraph();
  }
  const payload = raw as Record<string, unknown>;
  const entities = Array.isArray(payload.entities)
    ? payload.entities
        .map((item) => sanitizeEntity(item))
        .filter((item): item is MemoryEntity => item !== null)
    : [];
  const relations = Array.isArray(payload.relations)
    ? payload.relations
        .map((item) => sanitizeRelation(item))
        .filter((item): item is MemoryRelation => item !== null)
    : [];
  return { entities, relations };
}

function sanitizeEntity(raw: unknown): MemoryEntity | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }
  const payload = raw as Record<string, unknown>;
  const name = String(payload.name || "").trim();
  const entityType = String(payload.entityType || "").trim();
  if (!name || !entityType) {
    return null;
  }
  const observations = Array.isArray(payload.observations)
    ? payload.observations.map((item) => String(item || "").trim()).filter((item) => Boolean(item))
    : [];
  return { name, entityType, observations };
}

function sanitizeRelation(raw: unknown): MemoryRelation | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return null;
  }
  const payload = raw as Record<string, unknown>;
  const from = String(payload.from || "").trim();
  const to = String(payload.to || "").trim();
  const relationType = String(payload.relationType || "").trim();
  if (!from || !to || !relationType) {
    return null;
  }
  return { from, to, relationType };
}

async function readGraphFromFile(memoryFilePath: string): Promise<MemoryGraph> {
  try {
    const raw = await fs.readFile(memoryFilePath, "utf8");
    return sanitizeGraph(JSON.parse(raw));
  } catch {
    return emptyGraph();
  }
}

async function writeGraphToFile(memoryFilePath: string, graph: MemoryGraph): Promise<void> {
  await fs.mkdir(path.dirname(memoryFilePath), { recursive: true });
  await fs.writeFile(memoryFilePath, `${JSON.stringify(graph, null, 2)}\n`, "utf8");
}

function bridgeReturnedError(data: JsonValue): boolean {
  if (!data || typeof data !== "object" || Array.isArray(data)) {
    return false;
  }
  const maybeError = (data as Record<string, unknown>)["isError"];
  return maybeError === true;
}

function normalizeEntityName(value: unknown): string {
  return String(value || "").trim();
}

function findEntityByName(graph: MemoryGraph, name: string): MemoryEntity | undefined {
  const normalizedName = name.toLowerCase();
  return graph.entities.find((entity) => entity.name.toLowerCase() === normalizedName);
}

function toGraphResponse(graph: MemoryGraph): JsonObject {
  return {
    entities: graph.entities.map((entity) => ({
      name: entity.name,
      entityType: entity.entityType,
      observations: entity.observations,
    })),
    relations: graph.relations.map((relation) => ({
      from: relation.from,
      to: relation.to,
      relationType: relation.relationType,
    })),
  };
}

function searchGraph(graph: MemoryGraph, query: string): MemoryGraph {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) {
    return graph;
  }
  const matchingEntityNames = new Set(
    graph.entities
      .filter((entity) => {
        if (entity.name.toLowerCase().includes(normalizedQuery)) {
          return true;
        }
        if (entity.entityType.toLowerCase().includes(normalizedQuery)) {
          return true;
        }
        return entity.observations.some((observation) => observation.toLowerCase().includes(normalizedQuery));
      })
      .map((entity) => entity.name),
  );

  const relations = graph.relations.filter(
    (relation) => matchingEntityNames.has(relation.from) || matchingEntityNames.has(relation.to),
  );
  relations.forEach((relation) => {
    matchingEntityNames.add(relation.from);
    matchingEntityNames.add(relation.to);
  });

  const entities = graph.entities.filter((entity) => matchingEntityNames.has(entity.name));
  return {
    entities,
    relations,
  };
}

function openNodes(graph: MemoryGraph, names: string[]): MemoryGraph {
  const targetNames = new Set(names.map((name) => name.trim().toLowerCase()).filter((name) => Boolean(name)));
  const entities = graph.entities.filter((entity) => targetNames.has(entity.name.toLowerCase()));
  const selectedEntityNames = new Set(entities.map((entity) => entity.name));
  const relations = graph.relations.filter(
    (relation) => selectedEntityNames.has(relation.from) || selectedEntityNames.has(relation.to),
  );
  return {
    entities,
    relations,
  };
}

export function registerMemoryTools(registry: ToolRegistry, bridge: MemoryMcpBridge, memoryFilePath: string): void {
  async function fallbackCall(upstreamToolName: string, args: JsonObject): Promise<JsonValue> {
    const graph = await readGraphFromFile(memoryFilePath);

    if (upstreamToolName === "read_graph") {
      return toGraphResponse(graph);
    }

    if (upstreamToolName === "search_nodes") {
      const query = String(args.query || "");
      return toGraphResponse(searchGraph(graph, query));
    }

    if (upstreamToolName === "open_nodes") {
      const names = Array.isArray(args.names) ? args.names.map((item) => String(item || "")) : [];
      return toGraphResponse(openNodes(graph, names));
    }

    if (upstreamToolName === "create_entities") {
      const entities = Array.isArray(args.entities) ? args.entities : [];
      let created = 0;
      for (const item of entities) {
        const entity = sanitizeEntity(item);
        if (!entity) {
          continue;
        }
        if (findEntityByName(graph, entity.name)) {
          continue;
        }
        graph.entities.push(entity);
        created += 1;
      }
      await writeGraphToFile(memoryFilePath, graph);
      return {
        content: [{ type: "text", text: `Created ${created} entities.` }],
        structuredContent: {
          created,
          ...toGraphResponse(graph),
        },
      };
    }

    if (upstreamToolName === "create_relations") {
      const relations = Array.isArray(args.relations) ? args.relations : [];
      let created = 0;
      for (const item of relations) {
        const relation = sanitizeRelation(item);
        if (!relation) {
          continue;
        }
        const exists = graph.relations.some(
          (current) =>
            current.from === relation.from &&
            current.to === relation.to &&
            current.relationType === relation.relationType,
        );
        if (exists) {
          continue;
        }
        graph.relations.push(relation);
        created += 1;
      }
      await writeGraphToFile(memoryFilePath, graph);
      return {
        content: [{ type: "text", text: `Created ${created} relations.` }],
        structuredContent: {
          created,
          ...toGraphResponse(graph),
        },
      };
    }

    if (upstreamToolName === "add_observations") {
      const observations = Array.isArray(args.observations) ? args.observations : [];
      let added = 0;
      for (const item of observations) {
        if (!item || typeof item !== "object" || Array.isArray(item)) {
          continue;
        }
        const payload = item as Record<string, unknown>;
        const entityName = normalizeEntityName(payload.entityName);
        if (!entityName) {
          continue;
        }
        const contents = Array.isArray(payload.contents)
          ? payload.contents.map((entry) => String(entry || "").trim()).filter((entry) => Boolean(entry))
          : [];
        if (contents.length === 0) {
          continue;
        }
        let entity = findEntityByName(graph, entityName);
        if (!entity) {
          entity = {
            name: entityName,
            entityType: "unknown",
            observations: [],
          };
          graph.entities.push(entity);
        }
        for (const content of contents) {
          if (entity.observations.includes(content)) {
            continue;
          }
          entity.observations.push(content);
          added += 1;
        }
      }
      await writeGraphToFile(memoryFilePath, graph);
      return {
        content: [{ type: "text", text: `Added ${added} observations.` }],
        structuredContent: {
          added,
          ...toGraphResponse(graph),
        },
      };
    }

    throw new Error(`Unknown memory tool '${upstreamToolName}'.`);
  }

  const proxy: MemoryToolProxy = async ({ upstreamToolName, args }): Promise<ToolExecutionResult> => {
    try {
      const data = await bridge.callTool(upstreamToolName, args);
      if (bridgeReturnedError(data)) {
        throw new Error("Memory MCP bridge returned tool-level error.");
      }
      return {
        ok: true,
        data,
      };
    } catch (error) {
      try {
        const fallbackData = await fallbackCall(upstreamToolName, args);
        return {
          ok: true,
          data: fallbackData,
        };
      } catch (fallbackError) {
        return {
          ok: false,
          error: `Memory tool '${upstreamToolName}' failed: ${String(error)}; fallback failed: ${String(fallbackError)}`,
        };
      }
    }
  };

  registry.register(buildReadGraphTool(proxy));
  registry.register(buildSearchNodesTool(proxy));
  registry.register(buildOpenNodesTool(proxy));
  registry.register(buildCreateEntitiesTool(proxy));
  registry.register(buildCreateRelationsTool(proxy));
  registry.register(buildAddObservationsTool(proxy));
}
