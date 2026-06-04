type LogLevel = "INFO" | "WARN" | "ERROR" | "DEBUG";

function formatMessage(level: LogLevel, message: string, extra?: unknown): string {
  const timestamp = new Date().toISOString();
  if (extra === undefined) {
    return `${timestamp} [${level}] ${message}`;
  }
  return `${timestamp} [${level}] ${message} ${JSON.stringify(extra)}`;
}

export const logger = {
  info: (message: string, extra?: unknown): void => {
    console.log(formatMessage("INFO", message, extra));
  },
  warn: (message: string, extra?: unknown): void => {
    console.warn(formatMessage("WARN", message, extra));
  },
  error: (message: string, extra?: unknown): void => {
    console.error(formatMessage("ERROR", message, extra));
  },
  debug: (message: string, extra?: unknown): void => {
    if ((process.env.LOG_LEVEL || "").toLowerCase() === "debug") {
      console.log(formatMessage("DEBUG", message, extra));
    }
  },
};
