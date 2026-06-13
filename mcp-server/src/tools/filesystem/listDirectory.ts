import { ToolDefinition } from "../../core/types";
import { sanitizeToolPathArgs } from "../../utils/sanitizePath";
import { FilesystemToolProxy } from "./types";

export function buildListDirectoryTool(proxy: FilesystemToolProxy): ToolDefinition {
  return {
    name: "filesystem.list_directory",
    description:
      "List files/directories under a path. Use when the user asks to browse, list, or discover files.",
    inputSchema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "Relative directory path from MCP_FS_ROOT. Use '.' for root.",
          default: ".",
        },
      },
      required: [],
      additionalProperties: false,
    },
    execute: async (args, context) => {
      const mergedArgs = {
        path: typeof args.path === "string" ? args.path : ".",
      };
      const sanitizedArgs = sanitizeToolPathArgs(mergedArgs, ["path"]);
      return proxy({
        upstreamToolName: "list_directory",
        args: sanitizedArgs,
        context,
      });
    },
  };
}
