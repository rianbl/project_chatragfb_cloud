import { ToolDefinition } from "../../core/types";

export function buildGitDiffTool(): ToolDefinition {
  return {
    name: "git.diff",
    description: "Placeholder for future git diff queries.",
    inputSchema: {
      type: "object",
      properties: {
        target: { type: "string" },
      },
      required: [],
      additionalProperties: false,
    },
    execute: async () => ({
      ok: false,
      error: "git.diff is not enabled yet.",
    }),
  };
}
