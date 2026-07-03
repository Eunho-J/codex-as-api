import * as crypto from "node:crypto";
import * as path from "node:path";
import capabilityData from "../../config/model-capabilities.json";

export const RESPONSES_LITE_ENV = "CODEX_AS_API_RESPONSES_LITE";
export const CODEX_METADATA_ENV = "CODEX_AS_API_CODEX_METADATA";
export const LITE_HEADER_NAME = "x-openai-internal-codex-responses-lite";
export const LITE_HEADER_VALUE = "true";
export const TURN_METADATA_KEY = "x-codex-turn-metadata";
export const INSTALLATION_ID_KEY = "x-codex-installation-id";
export const WINDOW_ID_KEY = "x-codex-window-id";
export const SESSION_ID_KEY = "session_id";
export const THREAD_ID_KEY = "thread_id";
export const TURN_ID_KEY = "turn_id";

type ResponsesLiteMode = "off" | "on" | "auto";

export interface ModelCapability {
  useResponsesLite: boolean;
  supportsParallelToolCalls: boolean;
  supportVerbosity: boolean;
  defaultVerbosity: string | null;
  serviceTiers: string[];
  defaultServiceTier: string | null;
  source: string;
}

const UNKNOWN_CAPABILITY: ModelCapability = {
  useResponsesLite: false,
  supportsParallelToolCalls: false,
  supportVerbosity: false,
  defaultVerbosity: null,
  serviceTiers: [],
  defaultServiceTier: null,
  source: "unknown",
};

const INSTALLATION_NAMESPACE = "d2c81270-8f15-5e8d-a5c4-4cdbf2c21fd0";
const SESSION_ID = crypto.randomUUID();
const THREAD_ID = crypto.randomUUID();
const WINDOW_ID = crypto.randomUUID();

let cachedCapabilities: Record<string, ModelCapability> | null = null;

export function capabilityForModel(model?: string): ModelCapability {
  if (!model) return UNKNOWN_CAPABILITY;
  return loadModelCapabilities()[model] ?? UNKNOWN_CAPABILITY;
}

export function resolveResponsesLiteMode(value?: boolean | string): ResponsesLiteMode {
  const raw = value ?? process.env[RESPONSES_LITE_ENV] ?? "auto";
  if (typeof raw === "boolean") return raw ? "on" : "off";
  const normalized = raw.trim().toLowerCase();
  if (["true", "1", "yes", "on"].includes(normalized)) return "on";
  if (["false", "0", "no", "off"].includes(normalized)) return "off";
  if (normalized === "auto") return "auto";
  throw new Error("responses_lite must be one of: off, on, auto");
}

export function useResponsesLite(model: string, value?: boolean | string): boolean {
  const mode = resolveResponsesLiteMode(value);
  if (mode === "on") return true;
  if (mode === "off") return false;
  return capabilityForModel(model).useResponsesLite;
}

export function resolveCodexMetadataEnabled(value?: boolean): boolean {
  if (value != null) return value;
  const raw = (process.env[CODEX_METADATA_ENV] ?? "off").trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(raw)) return true;
  if (["0", "false", "no", "off", ""].includes(raw)) return false;
  throw new Error("codex_metadata must be on or off");
}

export function applyModelCapabilityFields(
  payload: Record<string, unknown>,
  model: string,
  text?: Record<string, unknown>,
  serviceTier?: string,
): void {
  const capability = capabilityForModel(model);
  if (capability.supportVerbosity) {
    const mergedText: Record<string, unknown> = { ...(text ?? {}) };
    if (mergedText.verbosity == null && capability.defaultVerbosity != null) {
      mergedText.verbosity = capability.defaultVerbosity;
    }
    if (Object.keys(mergedText).length > 0) payload.text = mergedText;
  } else if (text != null) {
    payload.text = { ...text };
  }

  if (serviceTier != null && serviceTier !== "default" && capability.serviceTiers.includes(serviceTier)) {
    payload.service_tier = serviceTier;
  }
}

export function shouldEnableParallelToolCalls(opts: {
  model: string;
  requested?: boolean;
  responsesLite: boolean;
}): boolean {
  if (opts.responsesLite || opts.requested !== true) return false;
  return capabilityForModel(opts.model).supportsParallelToolCalls;
}

export function buildCodexClientMetadata(opts: {
  authJsonPath?: string;
  existing?: Record<string, string>;
}): Record<string, string> {
  const metadata = { ...(opts.existing ?? {}) };
  const rawPath = opts.authJsonPath ?? "~/.codex/auth.json";
  const expandedPath = rawPath.startsWith("~/")
    ? path.join(process.env.HOME ?? "", rawPath.slice(2))
    : rawPath;
  const installationId = uuidV5(`codex-as-api:${path.resolve(expandedPath)}`, INSTALLATION_NAMESPACE);
  const turnId = crypto.randomUUID();
  const turnMetadata = {
    installation_id: installationId,
    session_id: SESSION_ID,
    thread_id: THREAD_ID,
    turn_id: turnId,
    window_id: WINDOW_ID,
    source: "codex-as-api",
  };
  metadata[INSTALLATION_ID_KEY] = installationId;
  metadata[SESSION_ID_KEY] = SESSION_ID;
  metadata[THREAD_ID_KEY] = THREAD_ID;
  metadata[TURN_ID_KEY] = turnId;
  metadata[WINDOW_ID_KEY] = WINDOW_ID;
  metadata[TURN_METADATA_KEY] = JSON.stringify(turnMetadata);
  return metadata;
}

export function stripImageDetailFields(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripImageDetailFields);
  if (typeof value === "object" && value !== null) {
    const source = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const [key, child] of Object.entries(source)) {
      if (key === "detail" && source.type === "input_image") continue;
      out[key] = stripImageDetailFields(child);
    }
    return out;
  }
  return value;
}

function loadModelCapabilities(): Record<string, ModelCapability> {
  if (cachedCapabilities !== null) return cachedCapabilities;
  const models = (capabilityData as { models?: Record<string, Record<string, unknown>> }).models ?? {};
  cachedCapabilities = Object.fromEntries(
    Object.entries(models).map(([name, value]) => [name, capabilityFromRecord(value)]),
  );
  return cachedCapabilities;
}

function capabilityFromRecord(value: Record<string, unknown>): ModelCapability {
  return {
    useResponsesLite: value.use_responses_lite === true,
    supportsParallelToolCalls: value.supports_parallel_tool_calls === true,
    supportVerbosity: value.support_verbosity === true,
    defaultVerbosity: typeof value.default_verbosity === "string" ? value.default_verbosity : null,
    serviceTiers: Array.isArray(value.service_tiers) ? value.service_tiers.filter((v): v is string => typeof v === "string") : [],
    defaultServiceTier: typeof value.default_service_tier === "string" ? value.default_service_tier : null,
    source: typeof value.source === "string" ? value.source : "unknown",
  };
}

function uuidV5(name: string, namespace: string): string {
  const namespaceBytes = Buffer.from(namespace.replace(/-/g, ""), "hex");
  const hash = crypto.createHash("sha1").update(namespaceBytes).update(name).digest();
  hash[6] = (hash[6] & 0x0f) | 0x50;
  hash[8] = (hash[8] & 0x3f) | 0x80;
  const hex = hash.subarray(0, 16).toString("hex");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20, 32)}`;
}
