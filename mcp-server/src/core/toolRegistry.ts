import { ToolDefinition, ToolExecutionContext, ToolExecutionResult, ToolManifestItem, JsonObject } from "./types";

export class ToolRegistry {
  private readonly tools = new Map<string, ToolDefinition>();

  register(tool: ToolDefinition): void {
    if (this.tools.has(tool.name)) {
      throw new Error(`Tool '${tool.name}' already registered.`);
    }
    this.tools.set(tool.name, tool);
  }

  list(): ToolManifestItem[] {
    return Array.from(this.tools.values()).map((tool) => ({
      name: tool.name,
      description: tool.description,
      inputSchema: tool.inputSchema,
    }));
  }

  has(toolName: string): boolean {
    return this.tools.has(toolName);
  }

  async execute(toolName: string, args: JsonObject, context: ToolExecutionContext): Promise<ToolExecutionResult> {
    const tool = this.tools.get(toolName);
    if (!tool) {
      return {
        ok: false,
        error: `Tool '${toolName}' is not registered.`,
      };
    }

    return tool.execute(args, context);
  }
}
