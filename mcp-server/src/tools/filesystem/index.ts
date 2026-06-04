import { ToolRegistry } from "../../core/toolRegistry";
import fs from "fs/promises";
import path from "path";
import { JsonObject, JsonValue, ToolExecutionResult } from "../../core/types";
import { resolveFilesystemRoot } from "../../utils/fsRoot";
import { FilesystemMcpBridge } from "./backend";
import { buildListDirectoryTool } from "./listDirectory";
import { buildReadFileTool } from "./readFile";
import { FilesystemToolProxy } from "./types";
import { buildWriteFileTool } from "./writeFile";

export function registerFilesystemTools(registry: ToolRegistry, bridge: FilesystemMcpBridge): void {
  const fsRoot = resolveFilesystemRoot();

  function safeJoinRoot(relativePath: string): string {
    const resolved = path.resolve(fsRoot, relativePath || ".");
    const normalizedRoot = path.resolve(fsRoot);
    if (resolved !== normalizedRoot && !resolved.startsWith(`${normalizedRoot}${path.sep}`)) {
      throw new Error("Path is outside allowed root.");
    }
    return resolved;
  }

  async function fallbackCall(upstreamToolName: string, args: JsonObject): Promise<JsonValue> {
    if (upstreamToolName === "read_file") {
      const relPath = String(args.path || ".");
      const absPath = safeJoinRoot(relPath);
      const content = await fs.readFile(absPath, "utf8");
      return {
        content: [
          {
            type: "text",
            text: content,
          },
        ],
      };
    }

    if (upstreamToolName === "write_file") {
      const relPath = String(args.path || ".");
      const content = String(args.content || "");
      const absPath = safeJoinRoot(relPath);
      await fs.mkdir(path.dirname(absPath), { recursive: true });
      await fs.writeFile(absPath, content, "utf8");
      return {
        content: [
          {
            type: "text",
            text: `File written: ${relPath}`,
          },
        ],
      };
    }

    if (upstreamToolName === "list_directory") {
      const relPath = String(args.path || ".");
      const absPath = safeJoinRoot(relPath);
      const entries = await fs.readdir(absPath, { withFileTypes: true });
      return {
        entries: entries.map((entry) => ({
          name: entry.name,
          is_directory: entry.isDirectory(),
        })),
      };
    }

    throw new Error(`Unknown filesystem tool '${upstreamToolName}'.`);
  }

  function buildBridgeArgs(upstreamToolName: string, args: JsonObject): JsonObject {
    if (upstreamToolName === "read_file" || upstreamToolName === "write_file" || upstreamToolName === "list_directory") {
      const relPath = String(args.path || ".");
      return {
        ...args,
        path: safeJoinRoot(relPath),
      };
    }
    return args;
  }

  function bridgeReturnedError(data: JsonValue): boolean {
    if (!data || typeof data !== "object" || Array.isArray(data)) {
      return false;
    }
    const maybeError = (data as Record<string, unknown>)["isError"];
    return maybeError === true;
  }

  const proxy: FilesystemToolProxy = async ({ upstreamToolName, args }): Promise<ToolExecutionResult> => {
    try {
      const bridgeArgs = buildBridgeArgs(upstreamToolName, args);
      const data = await bridge.callTool(upstreamToolName, bridgeArgs);
      if (bridgeReturnedError(data)) {
        throw new Error("Filesystem MCP bridge returned tool-level error.");
      }
      return {
        ok: true,
        data,
      };
    } catch (error) {
      try {
        const fallbackData = await fallbackCall(upstreamToolName, args);
        return {
          ok: true,
          data: fallbackData,
        };
      } catch (fallbackError) {
        return {
          ok: false,
          error: `Filesystem tool '${upstreamToolName}' failed: ${String(error)}; fallback failed: ${String(fallbackError)}`,
        };
      }
    }
  };

  registry.register(buildReadFileTool(proxy));
  registry.register(buildWriteFileTool(proxy));
  registry.register(buildListDirectoryTool(proxy));
}
