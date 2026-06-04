import { randomUUID } from "crypto";
import { Request } from "express";

import { ToolExecutionContext } from "./types";

export function buildExecutionContext(req: Request): ToolExecutionContext {
  const headerRequestId = req.header("x-request-id");
  const headerUserId = req.header("x-user-id");
  const headerTenantId = req.header("x-tenant-id");

  return {
    requestId: headerRequestId || randomUUID(),
    userId: headerUserId || undefined,
    tenantId: headerTenantId || undefined,
    metadata: {
      ip: String(req.ip || ""),
      userAgent: String(req.header("user-agent") || ""),
      method: String(req.method || ""),
      route: String(req.path || ""),
    },
  };
}
