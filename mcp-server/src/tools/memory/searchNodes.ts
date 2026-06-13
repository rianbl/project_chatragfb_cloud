import { ToolDefinition } from "../../core/types";
import { MemoryToolProxy } from "./types";

export function buildSearchNodesTool(proxy: MemoryToolProxy): ToolDefinition {
  return {
    name: "memory.search_nodes",
    description:
      "Search remembered facts by keyword across entity name/type/observations. Use for memory lookup questions.",
    inputSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search query for entity names, entity types, and observations.",
        },
      },
      required: ["query"],
      additionalProperties: false,
    },
    execute: async (args, context) => {
      const query = String(args.query || "").trim();
      if (!query) {
        throw new Error("Field 'query' is required.");
      }
      return proxy({
        upstreamToolName: "search_nodes",
        args: { query },
        context,
      });
    },
  };
}
