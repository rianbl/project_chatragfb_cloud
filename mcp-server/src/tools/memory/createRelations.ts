import { ToolDefinition } from "../../core/types";
import { MemoryToolProxy } from "./types";

function sanitizeRelations(rawRelations: unknown): Array<{ from: string; to: string; relationType: string }> {
  if (!Array.isArray(rawRelations)) {
    throw new Error("Field 'relations' must be an array.");
  }
  const relations: Array<{ from: string; to: string; relationType: string }> = [];
  for (const item of rawRelations) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      continue;
    }
    const record = item as Record<string, unknown>;
    const from = String(record.from || "").trim();
    const to = String(record.to || "").trim();
    const relationType = String(record.relationType || "").trim();
    if (!from || !to || !relationType) {
      continue;
    }
    relations.push({ from, to, relationType });
  }
  if (relations.length === 0) {
    throw new Error("Field 'relations' must contain at least one valid relation.");
  }
  return relations;
}

export function buildCreateRelationsTool(proxy: MemoryToolProxy): ToolDefinition {
  return {
    name: "memory.create_relations",
    description: "Create one or more relations in memory graph.",
    inputSchema: {
      type: "object",
      properties: {
        relations: {
          type: "array",
          items: {
            type: "object",
            properties: {
              from: { type: "string" },
              to: { type: "string" },
              relationType: { type: "string" },
            },
            required: ["from", "to", "relationType"],
            additionalProperties: false,
          },
        },
      },
      required: ["relations"],
      additionalProperties: false,
    },
    execute: async (args, context) => {
      const relations = sanitizeRelations(args.relations);
      return proxy({
        upstreamToolName: "create_relations",
        args: { relations },
        context,
      });
    },
  };
}
