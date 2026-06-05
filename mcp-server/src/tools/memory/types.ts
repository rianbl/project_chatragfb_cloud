import { JsonObject, ToolExecutionContext, ToolExecutionResult } from "../../core/types";

export interface MemoryProxyInput {
  upstreamToolName: string;
  args: JsonObject;
  context: ToolExecutionContext;
}

export type MemoryToolProxy = (input: MemoryProxyInput) => Promise<ToolExecutionResult>;
