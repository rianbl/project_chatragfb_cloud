import { ToolDefinition } from "../../core/types";

export function buildHttpFetchTool(): ToolDefinition {
  return {
    name: "http.fetch",
    description: "Placeholder for future outbound HTTP fetch operations.",
    inputSchema: {
      type: "object",
      properties: {
        url: { type: "string" },
      },
      required: ["url"],
      additionalProperties: false,
    },
    execute: async () => ({
      ok: false,
      error: "http.fetch is not enabled yet.",
    }),
  };
}
