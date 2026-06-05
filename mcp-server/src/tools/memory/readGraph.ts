import { ToolDefinition } from "../../core/types";
import { MemoryToolProxy } from "./types";

export function buildReadGraphTool(proxy: MemoryToolProxy): ToolDefinition {
  return {
    name: "memory.read_graph",
    description: "Read the complete memory graph (entities and relations).",
    inputSchema: {
      type: "object",
      properties: {},
      additionalProperties: false,
    },
    execute: async (_args, context) =>
      proxy({
        upstreamToolName: "read_graph",
        args: {},
        context,
      }),
  };
}
