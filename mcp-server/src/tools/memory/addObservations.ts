import { ToolDefinition } from "../../core/types";
import { MemoryToolProxy } from "./types";

function sanitizeObservations(rawObservations: unknown): Array<{ entityName: string; contents: string[] }> {
  if (!Array.isArray(rawObservations)) {
    throw new Error("Field 'observations' must be an array.");
  }
  const observations: Array<{ entityName: string; contents: string[] }> = [];
  for (const item of rawObservations) {
    if (!item || typeof item !== "object" || Array.isArray(item)) {
      continue;
    }
    const record = item as Record<string, unknown>;
    const entityName = String(record.entityName || "").trim();
    if (!entityName) {
      continue;
    }
    const contents = Array.isArray(record.contents)
      ? record.contents.map((entry) => String(entry || "").trim()).filter((entry) => Boolean(entry))
      : [];
    if (contents.length === 0) {
      continue;
    }
    observations.push({ entityName, contents });
  }
  if (observations.length === 0) {
    throw new Error("Field 'observations' must contain at least one valid observation item.");
  }
  return observations;
}

export function buildAddObservationsTool(proxy: MemoryToolProxy): ToolDefinition {
  return {
    name: "memory.add_observations",
    description: "Add observations to existing entities in memory graph.",
    inputSchema: {
      type: "object",
      properties: {
        observations: {
          type: "array",
          items: {
            type: "object",
            properties: {
              entityName: { type: "string" },
              contents: {
                type: "array",
                items: { type: "string" },
              },
            },
            required: ["entityName", "contents"],
            additionalProperties: false,
          },
        },
      },
      required: ["observations"],
      additionalProperties: false,
    },
    execute: async (args, context) => {
      const observations = sanitizeObservations(args.observations);
      return proxy({
        upstreamToolName: "add_observations",
        args: { observations },
        context,
      });
    },
  };
}
