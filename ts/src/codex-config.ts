import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

export interface CodexConfig {
  codexHome: string;
  configPath: string;
  model?: string;
  modelContextWindow?: number;
  modelAutoCompactTokenLimit?: number;
}

function expandHome(p: string): string {
  if (p === "~") return os.homedir();
  if (p.startsWith("~/")) return path.join(os.homedir(), p.slice(2));
  return p;
}

export function resolveCodexHome(raw?: string | null): string {
  return expandHome(raw || process.env.CODEX_HOME || path.join(os.homedir(), ".codex"));
}

export function loadCodexConfig(rawCodexHome?: string | null): CodexConfig {
  const codexHome = resolveCodexHome(rawCodexHome);
  const configPath = path.join(codexHome, "config.toml");
  const config: CodexConfig = { codexHome, configPath };

  let text: string;
  try {
    text = fs.readFileSync(configPath, "utf-8");
  } catch {
    return config;
  }

  const model = parseTomlString(text, "model");
  if (model) config.model = model;

  const modelContextWindow = parseTomlInteger(text, "model_context_window");
  if (modelContextWindow != null) config.modelContextWindow = modelContextWindow;

  const modelAutoCompactTokenLimit = parseTomlInteger(text, "model_auto_compact_token_limit");
  if (modelAutoCompactTokenLimit != null) {
    config.modelAutoCompactTokenLimit = modelAutoCompactTokenLimit;
  }

  return config;
}

function parseTomlString(text: string, key: string): string | undefined {
  const match = text.match(new RegExp(`^\\s*${escapeRegExp(key)}\\s*=\\s*["']([^"']+)["']\\s*(?:#.*)?$`, "m"));
  return match?.[1];
}

function parseTomlInteger(text: string, key: string): number | undefined {
  const match = text.match(new RegExp(`^\\s*${escapeRegExp(key)}\\s*=\\s*([0-9][0-9_]*)\\s*(?:#.*)?$`, "m"));
  if (!match) return undefined;
  const value = Number(match[1].replace(/_/g, ""));
  return Number.isSafeInteger(value) && value > 0 ? value : undefined;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
