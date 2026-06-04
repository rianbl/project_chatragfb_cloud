import fs from "fs";
import path from "path";
import { ChildProcessWithoutNullStreams, spawn } from "child_process";

import { JsonObject, JsonValue } from "../../core/types";
import { logger } from "../../utils/logger";

interface JsonRpcResponse {
  id?: number | string;
  result?: JsonValue;
  error?: {
    code?: number;
    message?: string;
    data?: JsonValue;
  };
  method?: string;
  params?: JsonValue;
}

export class FilesystemMcpBridge {
  private process: ChildProcessWithoutNullStreams | null = null;
  private readonly pending = new Map<
    number,
    {
      resolve: (value: JsonValue) => void;
      reject: (reason?: unknown) => void;
      timeout: NodeJS.Timeout;
    }
  >();
  private stdoutBuffer = "";
  private readonly stdioMode = (process.env.MCP_STDIO_MODE || "jsonl").toLowerCase();
  private nextId = 1;

  constructor(
    private readonly allowedRoots: string[],
    private readonly protocolVersion: string = process.env.MCP_PROTOCOL_VERSION || "2025-03-26",
  ) {}

  async start(): Promise<void> {
    if (this.process) {
      return;
    }

    const entry = this.resolveFilesystemServerEntry();
    const args = [entry, ...this.allowedRoots];
    logger.info("Starting prebuilt filesystem MCP server.", { entry, allowedRoots: this.allowedRoots });
    this.process = spawn(process.execPath, args, {
      stdio: ["pipe", "pipe", "pipe"],
      env: process.env,
    });

    this.process.stdout.on("data", (chunk: Buffer) => this.handleStdout(chunk));
    this.process.stderr.on("data", (chunk: Buffer) => {
      logger.debug("filesystem-server stderr", { message: chunk.toString("utf8").trim() });
    });
    this.process.on("close", (code: number | null, signal: NodeJS.Signals | null) => {
      const reason = new Error(`Filesystem MCP process closed (code=${String(code)} signal=${String(signal)}).`);
      for (const pending of this.pending.values()) {
        clearTimeout(pending.timeout);
        pending.reject(reason);
      }
      this.pending.clear();
      this.process = null;
    });

    await this.request("initialize", {
      protocolVersion: this.protocolVersion,
      capabilities: {},
      clientInfo: {
        name: "project-chatragfb-mcp-server",
        version: "0.1.0",
      },
    }, Number(process.env.MCP_INIT_TIMEOUT_MS || 12000));
    await this.notify("notifications/initialized", {});
    logger.info("Prebuilt filesystem MCP server initialized.");
  }

  async stop(): Promise<void> {
    if (!this.process) {
      return;
    }
    this.process.kill("SIGTERM");
    this.process = null;
  }

  async listTools(): Promise<JsonObject[]> {
    const response = await this.request("tools/list", {});
    const payload = (response as JsonObject) || {};
    const tools = payload["tools"];
    return Array.isArray(tools) ? (tools as JsonObject[]) : [];
  }

  async callTool(name: string, args: JsonObject): Promise<JsonValue> {
    const response = await this.request("tools/call", {
      name,
      arguments: args,
    });
    return response;
  }

  private resolveFilesystemServerEntry(): string {
    let packageJsonPath: string;
    try {
      packageJsonPath = require.resolve("@modelcontextprotocol/server-filesystem/package.json");
    } catch (_error) {
      const resolvedEntrypoint = require.resolve("@modelcontextprotocol/server-filesystem");
      packageJsonPath = this.findNearestPackageJson(path.dirname(resolvedEntrypoint));
    }
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, "utf8")) as {
      bin?: string | Record<string, string>;
    };
    const binField = packageJson.bin;
    if (!binField) {
      throw new Error("Could not resolve filesystem server bin from package metadata.");
    }
    const binRelativePath = typeof binField === "string" ? binField : Object.values(binField)[0];
    if (!binRelativePath) {
      throw new Error("Filesystem server bin field is empty.");
    }
    return path.resolve(path.dirname(packageJsonPath), binRelativePath);
  }

  private findNearestPackageJson(startDir: string): string {
    let current = startDir;
    while (true) {
      const candidate = path.join(current, "package.json");
      if (fs.existsSync(candidate)) {
        return candidate;
      }
      const parent = path.dirname(current);
      if (parent === current) {
        throw new Error("Could not locate package.json for filesystem MCP server package.");
      }
      current = parent;
    }
  }

  private async request(method: string, params: JsonObject, timeoutMs?: number): Promise<JsonValue> {
    if (!this.process) {
      throw new Error("Filesystem MCP process is not started.");
    }
    const id = this.nextId++;
    const message = {
      jsonrpc: "2.0",
      id,
      method,
      params,
    };
    const responsePromise = new Promise<JsonValue>((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Timed out waiting for MCP response for method '${method}'.`));
      }, timeoutMs ?? Number(process.env.MCP_REQUEST_TIMEOUT_MS || 12000));
      this.pending.set(id, { resolve, reject, timeout });
    });
    this.sendMessage(message);
    return responsePromise;
  }

  private async notify(method: string, params: JsonObject): Promise<void> {
    this.sendMessage({
      jsonrpc: "2.0",
      method,
      params,
    });
  }

  private sendMessage(payload: JsonObject): void {
    if (!this.process) {
      throw new Error("Filesystem MCP process is not started.");
    }
    const json = JSON.stringify(payload);
    if (this.stdioMode === "content-length") {
      const body = Buffer.from(json, "utf8");
      const header = Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, "utf8");
      this.process.stdin.write(Buffer.concat([header, body]));
      return;
    }
    this.process.stdin.write(`${json}\n`);
  }

  private handleStdout(chunk: Buffer): void {
    this.stdoutBuffer += chunk.toString("utf8");
    this.drainContentLengthMessages();
    this.drainJsonLineMessages();
  }

  private drainContentLengthMessages(): void {
    while (true) {
      const headerEnd = this.stdoutBuffer.indexOf("\r\n\r\n");
      if (headerEnd < 0) {
        return;
      }
      const headerText = this.stdoutBuffer.slice(0, headerEnd);
      if (!/Content-Length:/i.test(headerText)) {
        return;
      }
      const lengthMatch = /Content-Length:\s*(\d+)/i.exec(headerText);
      if (!lengthMatch) {
        this.stdoutBuffer = this.stdoutBuffer.slice(headerEnd + 4);
        continue;
      }
      const contentLength = Number(lengthMatch[1]);
      const bodyStart = headerEnd + 4;
      const totalLength = bodyStart + contentLength;
      if (this.stdoutBuffer.length < totalLength) {
        return;
      }
      const body = this.stdoutBuffer.slice(bodyStart, totalLength);
      this.stdoutBuffer = this.stdoutBuffer.slice(totalLength);
      this.handleMessage(body);
    }
  }

  private drainJsonLineMessages(): void {
    while (true) {
      const newlineIndex = this.stdoutBuffer.indexOf("\n");
      if (newlineIndex < 0) {
        return;
      }
      const line = this.stdoutBuffer.slice(0, newlineIndex).trim();
      this.stdoutBuffer = this.stdoutBuffer.slice(newlineIndex + 1);
      if (!line) {
        continue;
      }
      if (/^Content-Length:/i.test(line)) {
        continue;
      }
      if (!(line.startsWith("{") && line.endsWith("}"))) {
        logger.debug("Ignoring non-JSON stdout line from filesystem server.", { line });
        continue;
      }
      this.handleMessage(line);
    }
  }

  private handleMessage(rawMessage: string): void {
    let parsed: JsonRpcResponse;
    try {
      parsed = JSON.parse(rawMessage) as JsonRpcResponse;
    } catch (error) {
      logger.warn("Failed to parse MCP message.", { rawMessage, error: String(error) });
      return;
    }

    const responseId = Number(parsed.id);
    if (!Number.isInteger(responseId)) {
      return;
    }
    const pending = this.pending.get(responseId);
    if (!pending) {
      return;
    }
    this.pending.delete(responseId);
    clearTimeout(pending.timeout);

    if (parsed.error) {
      pending.reject(new Error(parsed.error.message || "Unknown MCP error."));
      return;
    }
    pending.resolve(parsed.result as JsonValue);
  }
}
