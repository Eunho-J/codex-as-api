import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { TextDecoder } from "node:util";
import { parse, type TomlTable } from "smol-toml";

export interface CodexConfig {
  codexHome: string;
  configPath: string;
  model?: string;
  modelReasoningEffort?: string;
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

  let bytes: Buffer;
  try {
    bytes = fs.readFileSync(configPath);
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === "ENOENT") return config;
    throw err;
  }

  const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  const document = parse(text, { integersAsBigInt: true });
  const model = optionalRootString(document, "model");
  if (model) config.model = model;

  const modelReasoningEffort = optionalRootString(document, "model_reasoning_effort");
  if (modelReasoningEffort !== undefined) {
    if (modelReasoningEffort.length === 0) {
      throw new Error("model_reasoning_effort must be a non-empty TOML string");
    }
    config.modelReasoningEffort = modelReasoningEffort;
  }

  const modelContextWindow = optionalPositiveRootInteger(document, "model_context_window");
  if (modelContextWindow != null) config.modelContextWindow = modelContextWindow;

  const modelAutoCompactTokenLimit = optionalPositiveRootInteger(
    document,
    "model_auto_compact_token_limit",
  );
  if (modelAutoCompactTokenLimit != null) {
    config.modelAutoCompactTokenLimit = modelAutoCompactTokenLimit;
  }

  return config;
}

function optionalRootString(document: TomlTable, key: string): string | undefined {
  if (!Object.hasOwn(document, key)) return undefined;
  const value = document[key];
  if (typeof value !== "string") {
    throw new Error(`${key} must be a TOML string`);
  }
  return value;
}

function optionalPositiveRootInteger(document: TomlTable, key: string): number | undefined {
  if (!Object.hasOwn(document, key)) return undefined;
  const value = document[key];
  if (typeof value !== "bigint" || value <= 0n || value > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`${key} must be a positive TOML integer`);
  }
  return Number(value);
}
