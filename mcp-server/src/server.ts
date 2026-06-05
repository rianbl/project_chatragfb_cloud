import { ToolRegistry } from "./core/toolRegistry";
import { createHttpApp } from "./transport/http";
import { registerFilesystemTools } from "./tools/filesystem";
import { FilesystemMcpBridge } from "./tools/filesystem/backend";
import { registerMemoryTools } from "./tools/memory";
import { MemoryMcpBridge } from "./tools/memory/backend";
import { resolveFilesystemRoot } from "./utils/fsRoot";
import { logger } from "./utils/logger";
import { resolveMemoryFilePath } from "./utils/memoryStore";

async function bootstrap(): Promise<void> {
  const httpPort = Number(process.env.MCP_HTTP_PORT || 8090);
  const fsRoot = resolveFilesystemRoot();
  const memoryEnabled = String(process.env.MCP_MEMORY_ENABLED || "true").toLowerCase() !== "false";
  const memoryFilePath = memoryEnabled ? resolveMemoryFilePath() : "";

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

  const memoryBridge = memoryEnabled ? new MemoryMcpBridge(memoryFilePath) : null;
  let memoryBridgeConnected = false;
  if (memoryBridge) {
    try {
      await memoryBridge.start();
      memoryBridgeConnected = true;
    } catch (error) {
      logger.warn("Failed to initialize prebuilt memory MCP bridge. Local fallback will be used.", {
        error: String(error),
      });
    }
  }

  const registry = new ToolRegistry();
  registerFilesystemTools(registry, filesystemBridge);
  if (memoryBridge) {
    registerMemoryTools(registry, memoryBridge, memoryFilePath);
  }

  const app = createHttpApp({ registry });
  app.listen(httpPort, () => {
    logger.info("MCP HTTP transport listening.", {
      port: httpPort,
      fsRoot,
      bridgeConnected,
      memoryEnabled,
      memoryFilePath: memoryEnabled ? memoryFilePath : undefined,
      memoryBridgeConnected,
      registeredTools: registry.list().map((tool) => tool.name),
    });
  });

  const shutdown = async (): Promise<void> => {
    logger.info("Shutdown signal received. Stopping MCP bridges.");
    if (memoryBridge) {
      await memoryBridge.stop();
    }
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
