import { ToolDefinition } from "../../core/types";
import { sanitizeToolPathArgs } from "../../utils/sanitizePath";
import { FilesystemToolProxy } from "./types";

export function buildReadFileTool(proxy: FilesystemToolProxy): ToolDefinition {
  return {
    name: "filesystem.read_file",
    description:
      "Read exact UTF-8 file content by path. Use for precise inspection/debugging or literal file output.",
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
