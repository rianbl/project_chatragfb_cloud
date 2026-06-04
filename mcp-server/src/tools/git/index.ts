import { ToolRegistry } from "../../core/toolRegistry";
import { buildGitCommitTool } from "./commit";
import { buildGitDiffTool } from "./diff";

export function registerGitTools(registry: ToolRegistry): void {
  const enabled = (process.env.MCP_ENABLE_GIT_TOOLS || "").toLowerCase() === "true";
  if (!enabled) {
    return;
  }
  registry.register(buildGitCommitTool());
  registry.register(buildGitDiffTool());
}
