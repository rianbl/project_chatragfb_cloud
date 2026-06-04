import { JsonObject, ToolExecutionContext, ToolExecutionResult } from "../../core/types";

export interface FilesystemProxyInput {
  upstreamToolName: string;
  args: JsonObject;
  context: ToolExecutionContext;
}

export type FilesystemToolProxy = (input: FilesystemProxyInput) => Promise<ToolExecutionResult>;
