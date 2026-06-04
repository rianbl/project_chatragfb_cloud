import fs from "fs";
import path from "path";

export function resolveFilesystemRoot(): string {
  const configuredRoot = process.env.MCP_FS_ROOT || "/workspace/uploads";
  const root = path.resolve(configuredRoot);
  if (!fs.existsSync(root)) {
    fs.mkdirSync(root, { recursive: true });
  }
  return root;
}
