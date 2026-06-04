export type JsonValue = string | number | boolean | null | JsonObject | JsonArray;
export interface JsonObject {
  [key: string]: JsonValue;
}
export interface JsonArray extends Array<JsonValue> {}

export interface ToolExecutionContext {
  requestId: string;
  userId?: string;
  tenantId?: string;
  metadata: Record<string, string>;
}

export interface ToolExecutionResult {
  ok: boolean;
  data?: JsonValue;
  error?: string;
}

export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: JsonObject;
  execute: (args: JsonObject, context: ToolExecutionContext) => Promise<ToolExecutionResult>;
}

export interface ToolManifestItem {
  name: string;
  description: string;
  inputSchema: JsonObject;
}
