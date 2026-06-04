import { ToolRegistry } from "../../core/toolRegistry";
import { buildHttpFetchTool } from "./fetch";

export function registerHttpTools(registry: ToolRegistry): void {
  const enabled = (process.env.MCP_ENABLE_HTTP_TOOLS || "").toLowerCase() === "true";
  if (!enabled) {
    return;
  }
  registry.register(buildHttpFetchTool());
}
