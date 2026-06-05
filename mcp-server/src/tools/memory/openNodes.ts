import { ToolDefinition } from "../../core/types";
import { MemoryToolProxy } from "./types";

function sanitizeNames(rawNames: unknown): string[] {
  if (!Array.isArray(rawNames)) {
    throw new Error("Field 'names' must be an array of strings.");
  }
  const names = rawNames
    .map((name) => String(name || "").trim())
    .filter((name) => Boolean(name));
  if (names.length === 0) {
    throw new Error("Field 'names' must contain at least one value.");
  }
  return names;
}

export function buildOpenNodesTool(proxy: MemoryToolProxy): ToolDefinition {
  return {
    name: "memory.open_nodes",
    description: "Open memory graph nodes by exact names.",
    inputSchema: {
      type: "object",
      properties: {
        names: {
          type: "array",
          items: { type: "string" },
          description: "Entity names to open.",
        },
      },
      required: ["names"],
      additionalProperties: false,
    },
    execute: async (args, context) => {
      const names = sanitizeNames(args.names);
      return proxy({
        upstreamToolName: "open_nodes",
        args: { names },
        context,
      });
    },
  };
}
