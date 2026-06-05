import { ToolDefinition } from "../../core/types";
import { MemoryToolProxy } from "./types";

function sanitizeEntities(rawEntities: unknown): Array<{ name: string; entityType: string; observations: string[] }> {
  if (!Array.isArray(rawEntities)) {
    throw new Error("Field 'entities' must be an array.");
  }
  const entities: Array<{ name: string; entityType: string; observations: string[] }> = [];
  for (const item of rawEntities) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      continue;
    }
    const record = item as Record<string, unknown>;
    const name = String(record.name || "").trim();
    const entityType = String(record.entityType || "").trim();
    if (!name || !entityType) {
      continue;
    }
    const observations = Array.isArray(record.observations)
      ? record.observations.map((entry) => String(entry || "").trim()).filter((entry) => Boolean(entry))
      : [];
    entities.push({ name, entityType, observations });
  }
  if (entities.length === 0) {
    throw new Error("Field 'entities' must contain at least one valid entity.");
  }
  return entities;
}

export function buildCreateEntitiesTool(proxy: MemoryToolProxy): ToolDefinition {
  return {
    name: "memory.create_entities",
    description: "Create one or more entities in memory graph.",
    inputSchema: {
      type: "object",
      properties: {
        entities: {
          type: "array",
          items: {
            type: "object",
            properties: {
              name: { type: "string" },
              entityType: { type: "string" },
              observations: {
                type: "array",
                items: { type: "string" },
              },
            },
            required: ["name", "entityType"],
            additionalProperties: false,
          },
        },
      },
      required: ["entities"],
      additionalProperties: false,
    },
    execute: async (args, context) => {
      const entities = sanitizeEntities(args.entities);
      return proxy({
        upstreamToolName: "create_entities",
        args: { entities },
        context,
      });
    },
  };
}
