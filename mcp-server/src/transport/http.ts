import express, { Express, NextFunction, Request, Response } from "express";

import { buildExecutionContext } from "../core/context";
import { ToolRegistry } from "../core/toolRegistry";
import { JsonObject } from "../core/types";
import { logger } from "../utils/logger";

interface HttpTransportOptions {
  registry: ToolRegistry;
}

export function createHttpApp(options: HttpTransportOptions): Express {
  const app = express();
  app.use(express.json({ limit: "1mb" }));

  app.get("/health", (_req: Request, res: Response) => {
    res.status(200).json({
      status: "ok",
      tools_count: options.registry.list().length,
      timestamp: new Date().toISOString(),
    });
  });

  app.get("/tools", (_req: Request, res: Response) => {
    res.status(200).json({
      tools: options.registry.list(),
    });
  });

  app.post("/tools/:toolName", async (req: Request, res: Response) => {
    const toolName = String(req.params.toolName || "").trim();
    if (!toolName) {
      res.status(400).json({ ok: false, error: "Tool name is required." });
      return;
    }
    if (!options.registry.has(toolName)) {
      res.status(404).json({ ok: false, error: `Tool '${toolName}' not found.` });
      return;
    }

    const payload = (req.body || {}) as Record<string, unknown>;
    const args = (payload.arguments || {}) as JsonObject;
    const executionContext = buildExecutionContext(req);
    logger.info("HTTP tool execution requested.", { toolName, requestId: executionContext.requestId });

    try {
      const result = await options.registry.execute(toolName, args, executionContext);
      const statusCode = result.ok ? 200 : 422;
      res.status(statusCode).json(result);
    } catch (err) {
      logger.error("Tool execution error.", { toolName, error: String(err) });
      res.status(500).json({ ok: false, error: String(err) });
    }
  });

  app.use((error: unknown, _req: Request, res: Response, _next: NextFunction) => {
    logger.error("Unhandled HTTP transport error.", { error: String(error) });
    res.status(500).json({
      ok: false,
      error: String(error),
    });
  });

  return app;
}
