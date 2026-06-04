import { ToolDefinition } from "../../core/types";
import { sanitizeToolPathArgs } from "../../utils/sanitizePath";
import { FilesystemToolProxy } from "./types";

export function buildWriteFileTool(proxy: FilesystemToolProxy): ToolDefinition {
  return {
    name: "filesystem.write_file",
    description: "Write UTF-8 text content to a file under the configured filesystem root.",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "Relative path from MCP_FS_ROOT." },
        content: { type: "string", description: "UTF-8 content to be written." },
      },
      required: ["path", "content"],
      additionalProperties: false,
    },
    execute: async (args, context) => {
      const sanitizedArgs = sanitizeToolPathArgs(args, ["path"]);
      return proxy({
        upstreamToolName: "write_file",
        args: sanitizedArgs,
        context,
      });
    },
  };
}
