import { ToolDefinition } from "../../core/types";
import { sanitizeToolPathArgs } from "../../utils/sanitizePath";
import { FilesystemToolProxy } from "./types";

export function buildDeleteFileTool(proxy: FilesystemToolProxy): ToolDefinition {
  return {
    name: "filesystem.delete_file",
    description: "Delete a file under the configured filesystem root.",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "Relative file path from MCP_FS_ROOT." },
      },
      required: ["path"],
      additionalProperties: false,
    },
    execute: async (args, context) => {
      const sanitizedArgs = sanitizeToolPathArgs(args, ["path"]);
      return proxy({
        upstreamToolName: "delete_file",
        args: sanitizedArgs,
        context,
      });
    },
  };
}
