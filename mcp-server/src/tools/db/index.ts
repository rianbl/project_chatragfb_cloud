import { ToolRegistry } from "../../core/toolRegistry";
import { buildDbQueryTool } from "./query";

export function registerDbTools(registry: ToolRegistry): void {
  const enabled = (process.env.MCP_ENABLE_DB_TOOLS || "").toLowerCase() === "true";
  if (!enabled) {
    return;
  }
  registry.register(buildDbQueryTool());
}
