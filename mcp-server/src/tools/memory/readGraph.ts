import { ToolDefinition } from "../../core/types";
import { MemoryToolProxy } from "./types";

export function buildReadGraphTool(proxy: MemoryToolProxy): ToolDefinition {
  return {
    name: "memory.read_graph",
    description:
      "Read all remembered knowledge (entities and relations). Use when the user asks what is in memory.",
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
