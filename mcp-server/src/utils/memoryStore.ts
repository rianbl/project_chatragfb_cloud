import fs from "fs";
import path from "path";

const DEFAULT_MEMORY_FILE_PATH = "/workspace/memory/memory.json";

export function resolveMemoryFilePath(): string {
  const configuredPath = process.env.MCP_MEMORY_FILE_PATH || DEFAULT_MEMORY_FILE_PATH;
  const absolutePath = path.resolve(configuredPath);
  const parentDirectory = path.dirname(absolutePath);
  if (!fs.existsSync(parentDirectory)) {
    fs.mkdirSync(parentDirectory, { recursive: true });
  }
  if (!fs.existsSync(absolutePath)) {
    fs.writeFileSync(absolutePath, JSON.stringify({ entities: [], relations: [] }, null, 2), "utf8");
  }
  return absolutePath;
}
