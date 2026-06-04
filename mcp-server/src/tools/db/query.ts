import { ToolDefinition } from "../../core/types";

export function buildDbQueryTool(): ToolDefinition {
  return {
    name: "db.query",
    description: "Placeholder for future database query execution.",
    inputSchema: {
      type: "object",
      properties: {
        sql: { type: "string" },
      },
      required: ["sql"],
      additionalProperties: false,
    },
    execute: async () => ({
      ok: false,
      error: "db.query is not enabled yet.",
    }),
  };
}
