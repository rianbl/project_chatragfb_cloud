import { ToolDefinition } from "../../core/types";
import { sanitizeToolPathArgs } from "../../utils/sanitizePath";
import { FilesystemToolProxy } from "./types";

export function buildReadFileTool(proxy: FilesystemToolProxy): ToolDefinition {
  return {
    name: "filesystem.read_file",
    description: "Read a UTF-8 text file from the configured filesystem root.",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "Relative path from MCP_FS_ROOT." },
      },
      required: ["path"],
      additionalProperties: false,
    },
    execute: async (args, context) => {
      const sanitizedArgs = sanitizeToolPathArgs(args, ["path"]);
      return proxy({
        upstreamToolName: "read_file",
        args: sanitizedArgs,
        context,
      });
    },
  };
}
