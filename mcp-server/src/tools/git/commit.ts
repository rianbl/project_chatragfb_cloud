import { ToolDefinition } from "../../core/types";

export function buildGitCommitTool(): ToolDefinition {
  return {
    name: "git.commit",
    description: "Placeholder for future git commit automation.",
    inputSchema: {
      type: "object",
      properties: {
        message: { type: "string" },
      },
      required: ["message"],
      additionalProperties: false,
    },
    execute: async () => ({
      ok: false,
      error: "git.commit is not enabled yet.",
    }),
  };
}
