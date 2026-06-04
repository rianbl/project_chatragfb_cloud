import path from "path";

export function sanitizeRelativePath(rawPath: string): string {
  const normalized = path.posix.normalize(String(rawPath || "").replace(/\\/g, "/")).trim();
  if (!normalized || normalized === ".") {
    return ".";
  }
  if (normalized.startsWith("/") || normalized.startsWith("..") || normalized.includes("/../")) {
    throw new Error("Path is outside allowed root.");
  }
  return normalized;
}

export function sanitizeToolPathArgs<T extends Record<string, unknown>>(args: T, keys: string[]): T {
  const clone: Record<string, unknown> = { ...args };
  for (const key of keys) {
    const value = clone[key];
    if (typeof value === "string") {
      clone[key] = sanitizeRelativePath(value);
    }
  }
  return clone as T;
}
