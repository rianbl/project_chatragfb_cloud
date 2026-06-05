import { ToolRegistry } from "./core/toolRegistry";
import { createHttpApp } from "./transport/http";
import { registerFilesystemTools } from "./tools/filesystem";
import { FilesystemMcpBridge } from "./tools/filesystem/backend";
import { resolveFilesystemRoot } from "./utils/fsRoot";
import { logger } from "./utils/logger";

async function bootstrap(): Promise<void> {
  const httpPort = Number(process.env.MCP_HTTP_PORT || 8090);
  const fsRoot = resolveFilesystemRoot();

  const filesystemBridge = new FilesystemMcpBridge([fsRoot]);
  let bridgeConnected = true;
  try {
    await filesystemBridge.start();
  } catch (error) {
    bridgeConnected = false;
    logger.warn("Failed to initialize prebuilt filesystem MCP bridge. Local fallback will be used.", {
      error: String(error),
    });
  }

  const registry = new ToolRegistry();
  registerFilesystemTools(registry, filesystemBridge);

  const app = createHttpApp({ registry });
  app.listen(httpPort, () => {
    logger.info("MCP HTTP transport listening.", {
      port: httpPort,
      fsRoot,
      bridgeConnected,
      registeredTools: registry.list().map((tool) => tool.name),
    });
  });

  const shutdown = async (): Promise<void> => {
    logger.info("Shutdown signal received. Stopping MCP bridge.");
    await filesystemBridge.stop();
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

bootstrap().catch((error) => {
  logger.error("Failed to start MCP server.", { error: String(error) });
  process.exit(1);
});
