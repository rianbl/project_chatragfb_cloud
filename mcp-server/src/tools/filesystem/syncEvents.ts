import http from "http";
import https from "https";

import { logger } from "../../utils/logger";

type FilesystemEventOperation = "upsert" | "delete";

interface FilesystemSyncEventPayload {
  operation: FilesystemEventOperation;
  path: string;
  requestId: string;
  timestamp: string;
}

function isSyncEnabled(): boolean {
  return String(process.env.MCP_SYNC_EVENTS_ENABLED || "true").toLowerCase() !== "false";
}

function resolveEndpoint(): URL {
  const baseUrl = String(process.env.MCP_SYNC_APP_URL || "http://app:8080").trim();
  const eventPath = String(process.env.MCP_SYNC_EVENTS_PATH || "/internal/filesystem/events").trim();
  return new URL(eventPath, baseUrl.endsWith("/") ? baseUrl : `${baseUrl}/`);
}

function postJson(url: URL, body: string, timeoutMs: number, token: string): Promise<void> {
  const client = url.protocol === "https:" ? https : http;
  const headers: Record<string, string | number> = {
    "content-type": "application/json",
    "content-length": Buffer.byteLength(body),
  };
  if (token) {
    headers["x-internal-token"] = token;
  }

  return new Promise((resolve, reject) => {
    const req = client.request(
      {
        protocol: url.protocol,
        hostname: url.hostname,
        port: url.port || (url.protocol === "https:" ? "443" : "80"),
        path: `${url.pathname}${url.search}`,
        method: "POST",
        headers,
      },
      (res) => {
        let responseBody = "";
        res.setEncoding("utf8");
        res.on("data", (chunk: string) => {
          responseBody += chunk;
        });
        res.on("end", () => {
          const statusCode = Number(res.statusCode || 0);
          if (statusCode >= 200 && statusCode < 300) {
            resolve();
            return;
          }
          reject(new Error(`Sync webhook failed with status=${statusCode}: ${responseBody.slice(0, 300)}`));
        });
      },
    );

    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Sync webhook timed out after ${timeoutMs}ms.`));
    });
    req.on("error", reject);
    req.write(body);
    req.end();
  });
}

export async function emitFilesystemSyncEvent(
  operation: FilesystemEventOperation,
  path: string,
  requestId: string,
): Promise<void> {
  if (!isSyncEnabled()) {
    return;
  }
  const relativePath = String(path || "").trim();
  if (!relativePath) {
    return;
  }

  const endpoint = resolveEndpoint();
  const payload: FilesystemSyncEventPayload = {
    operation,
    path: relativePath,
    requestId,
    timestamp: new Date().toISOString(),
  };
  const timeoutMs = Number(process.env.MCP_SYNC_TIMEOUT_MS || 5000);
  const token = String(process.env.MCP_SYNC_API_TOKEN || "");

  logger.info("Dispatching filesystem sync event.", {
    operation,
    path: relativePath,
    requestId,
    endpoint: endpoint.toString(),
  });
  await postJson(endpoint, JSON.stringify(payload), timeoutMs, token);
}

