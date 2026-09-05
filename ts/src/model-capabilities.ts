import * as crypto from "node:crypto";
import * as path from "node:path";
import { performance } from "node:perf_hooks";
import {
  ChatGPTOAuthCatalogUnavailableError,
  ChatGPTOAuthInvalidRequestError,
  ChatGPTOAuthProtocolError,
  resolveAuthPath,
} from "./auth.js";

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
export const DEFAULT_MODEL_CATALOG_TTL_MS = 5 * 60 * 1000;

type ResponsesLiteMode = "off" | "on" | "auto";

export interface ModelServiceTier {
  readonly id: string;
  readonly name: string;
  readonly description: string;
}

export interface ReasoningEffortPreset {
  readonly effort: string;
  readonly description: string;
}

export interface ModelCapability {
  readonly slug: string;
  readonly displayName: string;
  readonly description: string | null;
  readonly defaultReasoningEffort?: string;
  readonly supportedReasoningEfforts: readonly ReasoningEffortPreset[];
  readonly priority: number;
  readonly visibility: "list" | "hide" | "none";
  readonly supportedInApi: boolean;
  readonly useResponsesLite: boolean;
  readonly supportsImageDetailOriginal: boolean;
  readonly supportVerbosity: boolean;
  readonly defaultVerbosity: string | null;
  readonly supportsReasoningSummaryParameter: boolean;
  readonly defaultReasoningSummary: "auto" | "concise" | "detailed" | "none";
  readonly compHash?: string;
  readonly contextWindow?: number;
  readonly maxContextWindow?: number;
  readonly autoCompactTokenLimit?: number;
  readonly effectiveContextWindowPercent: number;
  readonly serviceTiers: readonly ModelServiceTier[];
  readonly defaultServiceTier: string | null;
  readonly inputModalities: readonly ("text" | "image" | "audio")[];
  readonly multiAgentReasoningEffort?: string;
}

export interface ModelCatalogSnapshot {
  readonly key: string;
  readonly etag: string | null;
  readonly fetchedAt: number;
  readonly expiresAt: number;
  readonly models: readonly ModelCapability[];
  readonly defaultModel: ModelCapability | null;
}

export interface ModelCatalogFetchResult {
  readonly value: unknown;
  readonly etag: string | null;
}

export function modelCatalogCacheKey(
  accountId: string,
  baseUrl: string,
  clientVersion: string,
): string {
  return JSON.stringify([accountId, baseUrl, clientVersion]);
}

export function accountIdFromModelCatalogCacheKey(key: string): string {
  let fields: unknown;
  try {
    fields = JSON.parse(key);
  } catch (err) {
    throw new ChatGPTOAuthProtocolError(`invalid internal model catalog cache key: ${String(err)}`);
  }
  if (
    !Array.isArray(fields)
    || fields.length !== 3
    || fields.some((field) => typeof field !== "string")
    || fields.some((field) => (field as string).length === 0)
  ) {
    throw new ChatGPTOAuthProtocolError("invalid internal model catalog cache key");
  }
  return fields[0] as string;
}

interface CacheEntry {
  readonly snapshot: ModelCatalogSnapshot;
  readonly expiresAtMonotonic: number;
}

const INSTALLATION_NAMESPACE = "d2c81270-8f15-5e8d-a5c4-4cdbf2c21fd0";
const WINDOW_ID = crypto.randomUUID();

export class ModelCatalogCache {
  private readonly entries = new Map<string, CacheEntry>();
  private readonly flights = new Map<string, Promise<ModelCatalogSnapshot>>();
  private readonly revisions = new Map<string, number>();

  constructor(
    private readonly ttlMs = DEFAULT_MODEL_CATALOG_TTL_MS,
    private readonly wallNow: () => number = Date.now,
    private readonly monotonicNow: () => number = () => performance.now(),
  ) {
    if (!Number.isFinite(ttlMs) || ttlMs <= 0) {
      throw new ChatGPTOAuthInvalidRequestError("model catalog TTL must be a positive finite number");
    }
  }

  async get(
    key: string,
    fetchCatalog: () => Promise<ModelCatalogFetchResult>,
  ): Promise<ModelCatalogSnapshot> {
    const current = this.entries.get(key);
    if (current != null && current.expiresAtMonotonic > this.monotonicNow()) {
      return current.snapshot;
    }

    const existingFlight = this.flights.get(key);
    if (existingFlight != null) return existingFlight;

    const observedRevision = this.revisions.get(key) ?? 0;
    const flight = (async () => {
      const result = await fetchCatalog();
      const fetchedAt = this.wallNow();
      const snapshot = parseModelCatalog(result.value, {
        key,
        etag: result.etag,
        fetchedAt,
        expiresAt: fetchedAt + this.ttlMs,
      });
      if ((this.revisions.get(key) ?? 0) !== observedRevision) {
        throw new ChatGPTOAuthCatalogUnavailableError(
          "model catalog was invalidated while a refresh was in flight",
        );
      }
      this.entries.set(key, {
        snapshot,
        expiresAtMonotonic: this.monotonicNow() + this.ttlMs,
      });
      return snapshot;
    })();
    this.flights.set(key, flight);
    try {
      return await flight;
    } finally {
      this.flights.delete(key);
    }
  }

  invalidateOnEtagMismatch(key: string, etag: string | null): void {
    const normalizedEtag = etag?.trim();
    if (normalizedEtag == null || normalizedEtag.length === 0) return;
    const entry = this.entries.get(key);
    if (entry == null || entry.snapshot.etag === normalizedEtag) return;
    this.revisions.set(key, (this.revisions.get(key) ?? 0) + 1);
    this.entries.delete(key);
  }
}

export function parseModelCatalog(
  value: unknown,
  metadata: { key: string; etag: string | null; fetchedAt: number; expiresAt: number },
): ModelCatalogSnapshot {
  const rawModels = isRecord(value) && Array.isArray(value.models)
    ? value.models
    : null;
  if (rawModels == null) {
    throw new ChatGPTOAuthCatalogUnavailableError("upstream model catalog must be an object with a models array");
  }
  const models = rawModels.map((raw, index) => parseModelInfo(raw, index));
  const slugs = new Set<string>();
  for (const model of models) {
    if (slugs.has(model.slug)) {
      throw new ChatGPTOAuthCatalogUnavailableError(
        "upstream model catalog contains a duplicate slug",
      );
    }
    slugs.add(model.slug);
  }

  const orderedModels = models
    .map((model, index) => ({ model, index }))
    .sort((left, right) => left.model.priority - right.model.priority || left.index - right.index)
    .map(({ model }) => model);
  const defaultModel = orderedModels.find((model) => model.visibility === "list") ?? null;
  const etag = typeof metadata.etag === "string" && metadata.etag.trim().length > 0
    ? metadata.etag.trim()
    : null;

  return Object.freeze({
    key: metadata.key,
    etag,
    fetchedAt: metadata.fetchedAt,
    expiresAt: metadata.expiresAt,
    models: Object.freeze([...models]),
    defaultModel,
  });
}

export function modelFromSnapshot(
  snapshot: ModelCatalogSnapshot,
  slug: string,
): ModelCapability | undefined {
  return snapshot.models.find((model) => model.slug === slug);
}

function parseModelInfo(value: unknown, index: number): ModelCapability {
  const field = `models[${index}]`;
  const record = requireRecord(value, field);
  const slug = requireString(record.slug, `${field}.slug`);
  const displayName = requireString(record.display_name, `${field}.display_name`);
  const description = optionalString(record.description, `${field}.description`) ?? null;
  const defaultReasoningEffort = optionalReasoningEffort(
    record.default_reasoning_level,
    `${field}.default_reasoning_level`,
  );
  const supportedReasoningEfforts = requireArray(record.supported_reasoning_levels, `${field}.supported_reasoning_levels`).map(
    (preset, presetIndex) => {
      const presetRecord = requireRecord(preset, `${field}.supported_reasoning_levels[${presetIndex}]`);
      return Object.freeze({
        effort: requireReasoningEffort(presetRecord.effort, `${field}.supported_reasoning_levels[${presetIndex}].effort`),
        description: requireString(presetRecord.description, `${field}.supported_reasoning_levels[${presetIndex}].description`),
      });
    },
  );
  const visibility = requireEnum(record.visibility, `${field}.visibility`, ["list", "hide", "none"] as const);
  const supportedInApi = requireBoolean(record.supported_in_api, `${field}.supported_in_api`);
  const priority = requireInt32(record.priority, `${field}.priority`);
  const supportVerbosity = requireBoolean(record.support_verbosity, `${field}.support_verbosity`);
  const defaultVerbosity = optionalEnum(
    record.default_verbosity,
    `${field}.default_verbosity`,
    ["low", "medium", "high"] as const,
  ) ?? null;
  const compHash = optionalString(record.comp_hash, `${field}.comp_hash`);
  const supportsReasoningSummaryParameter = booleanWithMissingDefault(
    record.supports_reasoning_summary_parameter,
    `${field}.supports_reasoning_summary_parameter`,
    true,
  );
  const defaultReasoningSummary = Object.hasOwn(record, "default_reasoning_summary")
    ? requireEnum(
      record.default_reasoning_summary,
      `${field}.default_reasoning_summary`,
      ["auto", "concise", "detailed", "none"] as const,
    )
    : "auto";
  const serviceTiers = arrayWithMissingDefault(
    record.service_tiers,
    `${field}.service_tiers`,
    [],
  ).map((tier, tierIndex) => {
    const tierRecord = requireRecord(tier, `${field}.service_tiers[${tierIndex}]`);
    return Object.freeze({
      id: requireString(tierRecord.id, `${field}.service_tiers[${tierIndex}].id`),
      name: requireString(tierRecord.name, `${field}.service_tiers[${tierIndex}].name`),
      description: requireString(tierRecord.description, `${field}.service_tiers[${tierIndex}].description`),
    });
  });
  const defaultServiceTier = optionalString(record.default_service_tier, `${field}.default_service_tier`);
  const inputModalities = arrayWithMissingDefault(
    record.input_modalities,
    `${field}.input_modalities`,
    ["text", "image"],
  ).map(
    (modality, modalityIndex) => requireEnum(
      modality,
      `${field}.input_modalities[${modalityIndex}]`,
      ["text", "image", "audio"] as const,
    ),
  );
  const contextWindow = optionalInteger(record.context_window, `${field}.context_window`);
  const maxContextWindow = optionalInteger(record.max_context_window, `${field}.max_context_window`);
  const autoCompactTokenLimit = optionalInteger(record.auto_compact_token_limit, `${field}.auto_compact_token_limit`);
  const effectiveContextWindowPercent = Object.hasOwn(record, "effective_context_window_percent")
    ? requireInteger(record.effective_context_window_percent, `${field}.effective_context_window_percent`)
    : 95;
  const multiAgentReasoningEffort = optionalReasoningEffort(
    record.multi_agent_reasoning_effort,
    `${field}.multi_agent_reasoning_effort`,
  );
  return Object.freeze({
    slug,
    displayName,
    description,
    defaultReasoningEffort,
    supportedReasoningEfforts: Object.freeze(supportedReasoningEfforts),
    priority,
    visibility,
    supportedInApi,
    useResponsesLite: booleanWithMissingDefault(
      record.use_responses_lite,
      `${field}.use_responses_lite`,
      false,
    ),
    supportsImageDetailOriginal: booleanWithMissingDefault(
      record.supports_image_detail_original,
      `${field}.supports_image_detail_original`,
      false,
    ),
    supportVerbosity,
    defaultVerbosity,
    supportsReasoningSummaryParameter,
    defaultReasoningSummary,
    ...(compHash == null ? {} : { compHash }),
    ...(contextWindow == null ? {} : { contextWindow }),
    ...(maxContextWindow == null ? {} : { maxContextWindow }),
    ...(autoCompactTokenLimit == null ? {} : { autoCompactTokenLimit }),
    effectiveContextWindowPercent,
    serviceTiers: Object.freeze(serviceTiers),
    defaultServiceTier: defaultServiceTier ?? null,
    inputModalities: Object.freeze(inputModalities),
    ...(multiAgentReasoningEffort == null ? {} : { multiAgentReasoningEffort }),
  });
}

export function resolveResponsesLiteMode(value?: boolean | string): ResponsesLiteMode {
  const raw = value ?? process.env[RESPONSES_LITE_ENV] ?? "auto";
  if (typeof raw === "boolean") return raw ? "on" : "off";
  if (typeof raw !== "string") {
    throw new ChatGPTOAuthInvalidRequestError("responses_lite must be one of: off, on, auto");
  }
  const normalized = raw.trim().toLowerCase();
  if (["true", "1", "yes", "on"].includes(normalized)) return "on";
  if (["false", "0", "no", "off"].includes(normalized)) return "off";
  if (normalized === "auto") return "auto";
  throw new ChatGPTOAuthInvalidRequestError("responses_lite must be one of: off, on, auto");
}

export function useResponsesLite(capability: ModelCapability, value?: boolean | string): boolean {
  const mode = resolveResponsesLiteMode(value);
  if (mode === "on") return true;
  if (mode === "off") return false;
  return capability.useResponsesLite;
}

export function resolveCodexMetadataEnabled(value?: boolean): boolean {
  if (value != null) return value;
  const raw = (process.env[CODEX_METADATA_ENV] ?? "off").trim().toLowerCase();
  if (["1", "true", "yes", "on"].includes(raw)) return true;
  if (["0", "false", "no", "off"].includes(raw)) return false;
  throw new ChatGPTOAuthInvalidRequestError("codex_metadata must be on or off");
}

export function applyModelCapabilityFields(
  payload: Record<string, unknown>,
  capability: ModelCapability,
  text?: Record<string, unknown>,
  serviceTier?: string,
): void {
  const mergedText: Record<string, unknown> = { ...(text ?? {}) };
  if (mergedText.verbosity == null) delete mergedText.verbosity;
  if (
    mergedText.verbosity != null
    && (
      typeof mergedText.verbosity !== "string"
      || !["low", "medium", "high"].includes(mergedText.verbosity)
    )
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "text.verbosity must be one of: low, medium, high",
    );
  }
  if (capability.supportVerbosity) {
    if (!Object.hasOwn(mergedText, "verbosity") && capability.defaultVerbosity != null) {
      mergedText.verbosity = capability.defaultVerbosity;
    }
    if (Object.keys(mergedText).length > 0) payload.text = mergedText;
  } else if (text?.verbosity != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "text.verbosity is not supported by the selected model",
    );
  } else if (Object.keys(mergedText).length > 0) {
    payload.text = mergedText;
  }

  if (serviceTier != null) {
    if (serviceTier === "default") {
      delete payload.service_tier;
      return;
    }
    const wireServiceTier = serviceTier === "fast" ? "priority" : serviceTier;
    if (!capability.serviceTiers.some((tier) => tier.id === wireServiceTier)) {
      throw new ChatGPTOAuthInvalidRequestError(
        "service_tier is not supported by the selected model",
      );
    }
    payload.service_tier = wireServiceTier;
  }
}

export function shouldEnableParallelToolCalls(opts: {
  requested?: boolean;
  responsesLite: boolean;
}): boolean {
  if (opts.requested != null && typeof opts.requested !== "boolean") {
    throw new ChatGPTOAuthInvalidRequestError("parallel_tool_calls must be a boolean when provided");
  }
  if (opts.requested === true && opts.responsesLite) {
    throw new ChatGPTOAuthInvalidRequestError("parallel_tool_calls=true cannot be represented by Responses Lite");
  }
  return opts.requested === true;
}

export function buildCodexClientMetadata(opts: {
  authJsonPath?: string;
  existing?: Record<string, unknown>;
}): Record<string, string> {
  const existing = opts.existing ?? {};
  const sessionId = requireMetadataIdentity(existing[SESSION_ID_KEY], `client_metadata.${SESSION_ID_KEY}`);
  const threadId = existing[THREAD_ID_KEY] == null
    ? sessionId
    : requireMetadataIdentity(existing[THREAD_ID_KEY], `client_metadata.${THREAD_ID_KEY}`);
  const metadata = { ...existing } as Record<string, string>;
  const installationId = uuidV5(
    `codex-as-api:${path.resolve(resolveAuthPath(opts.authJsonPath))}`,
    INSTALLATION_NAMESPACE,
  );
  const turnId = crypto.randomUUID();
  const turnMetadata = {
    installation_id: installationId,
    session_id: sessionId,
    thread_id: threadId,
    turn_id: turnId,
    window_id: WINDOW_ID,
    source: "codex-as-api",
  };
  metadata[INSTALLATION_ID_KEY] = installationId;
  metadata[SESSION_ID_KEY] = sessionId;
  metadata[THREAD_ID_KEY] = threadId;
  metadata[TURN_ID_KEY] = turnId;
  metadata[WINDOW_ID_KEY] = WINDOW_ID;
  metadata[TURN_METADATA_KEY] = JSON.stringify(turnMetadata);
  return metadata;
}

function requireMetadataIdentity(value: unknown, field: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be a non-empty string when codex_metadata is enabled`);
  }
  return value;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireRecord(value: unknown, field: string): Record<string, unknown> {
  if (!isRecord(value)) throw new ChatGPTOAuthCatalogUnavailableError(`${field} must be an object`);
  return value;
}

function requireArray(value: unknown, field: string): unknown[] {
  if (!Array.isArray(value)) throw new ChatGPTOAuthCatalogUnavailableError(`${field} must be an array`);
  return value;
}

function arrayWithMissingDefault(
  value: unknown,
  field: string,
  defaultValue: readonly unknown[],
): unknown[] {
  return value === undefined ? [...defaultValue] : requireArray(value, field);
}

function requireString(value: unknown, field: string): string {
  if (typeof value !== "string") throw new ChatGPTOAuthCatalogUnavailableError(`${field} must be a string`);
  return value;
}

function requireNonEmptyString(value: unknown, field: string): string {
  const result = requireString(value, field);
  if (result.trim().length === 0) throw new ChatGPTOAuthCatalogUnavailableError(`${field} must not be empty`);
  return result;
}

function requireReasoningEffort(value: unknown, field: string): string {
  const result = requireString(value, field);
  if (result.length === 0) {
    throw new ChatGPTOAuthCatalogUnavailableError(`${field} must not be empty`);
  }
  return result;
}

function optionalReasoningEffort(value: unknown, field: string): string | undefined {
  return value == null ? undefined : requireReasoningEffort(value, field);
}

function optionalString(value: unknown, field: string): string | undefined {
  return value == null ? undefined : requireString(value, field);
}

function requireBoolean(value: unknown, field: string): boolean {
  if (typeof value !== "boolean") throw new ChatGPTOAuthCatalogUnavailableError(`${field} must be a boolean`);
  return value;
}

function optionalBoolean(value: unknown, field: string): boolean | undefined {
  return value == null ? undefined : requireBoolean(value, field);
}

function booleanWithMissingDefault(
  value: unknown,
  field: string,
  defaultValue: boolean,
): boolean {
  return value === undefined ? defaultValue : requireBoolean(value, field);
}

function requireInteger(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value)) {
    throw new ChatGPTOAuthCatalogUnavailableError(`${field} must be a safe integer`);
  }
  return value;
}

function requireInt32(value: unknown, field: string): number {
  const result = requireInteger(value, field);
  if (result < -2_147_483_648 || result > 2_147_483_647) {
    throw new ChatGPTOAuthCatalogUnavailableError(`${field} must be a 32-bit integer`);
  }
  return result;
}

function optionalInteger(value: unknown, field: string): number | undefined {
  return value == null ? undefined : requireInteger(value, field);
}

function requireEnum<const T extends readonly string[]>(value: unknown, field: string, allowed: T): T[number] {
  if (typeof value !== "string" || !allowed.includes(value)) {
    throw new ChatGPTOAuthCatalogUnavailableError(`${field} must be one of: ${allowed.join(", ")}`);
  }
  return value as T[number];
}

function optionalEnum<const T extends readonly string[]>(
  value: unknown,
  field: string,
  allowed: T,
): T[number] | undefined {
  return value == null ? undefined : requireEnum(value, field, allowed);
}

function uuidV5(name: string, namespace: string): string {
  const namespaceBytes = Buffer.from(namespace.replace(/-/g, ""), "hex");
  const hash = crypto.createHash("sha1").update(namespaceBytes).update(name).digest();
  hash[6] = (hash[6] & 0x0f) | 0x50;
  hash[8] = (hash[8] & 0x3f) | 0x80;
  const hex = hash.subarray(0, 16).toString("hex");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20, 32)}`;
}
