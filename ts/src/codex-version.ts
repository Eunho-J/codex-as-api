import * as os from "node:os";
import { performance } from "node:perf_hooks";
import packageMetadata from "../package.json";
import {
  ChatGPTOAuthCatalogUnavailableError,
  ChatGPTOAuthInvalidRequestError,
} from "./auth.js";
import { decodeUtf8Strict, parseJsonStrict } from "./utf8-json.js";

export { ChatGPTOAuthCatalogUnavailableError } from "./auth.js";

export const CODEX_NPM_LATEST_URL =
  "https://registry.npmjs.org/@openai%2fcodex/latest";
export const CODEX_NPM_PACKAGE_NAME = "@openai/codex";
export const CODEX_CLI_ORIGINATOR = "codex_cli_rs";
export const CODEX_VERSION_FETCH_TIMEOUT_MS = 5_000;
export const CODEX_VERSION_CACHE_TTL_MS = 60 * 60 * 1_000;
export const CODEX_VERSION_RETRY_BACKOFF_MS = 60 * 1_000;
export const MAX_CODEX_VERSION_LENGTH = 64;

const STRICT_STABLE_SEMVER_RE =
  /^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/;

export type CodexSemverTuple = readonly [number, number, number];
export type CodexVersionFetch = (
  input: string,
  init?: RequestInit,
) => Promise<Response>;
export type CodexVersionClock = () => number;
export type CodexVersionProvider = (() => string | Promise<string>) | { resolve(): string | Promise<string> };

export interface CodexVersionResolverOptions {
  readonly fetchImpl?: CodexVersionFetch;
  readonly now?: CodexVersionClock;
  readonly ttlMs?: number;
  readonly retryBackoffMs?: number;
  readonly timeoutMs?: number;
  readonly registryUrl?: string;
}

export class ChatGPTOAuthCodexVersionUnavailableError extends ChatGPTOAuthCatalogUnavailableError {
  constructor(message: string, options?: ErrorOptions) {
    super(message);
    this.name = "ChatGPTOAuthCodexVersionUnavailableError";
    if (options?.cause !== undefined) {
      this.cause = options.cause;
    }
  }
}

export function parseCodexSemver(value: unknown): CodexSemverTuple | null {
  if (typeof value !== "string") return null;
  if (
    value.length === 0
    || value.length > MAX_CODEX_VERSION_LENGTH
    || value !== value.trim()
  ) {
    return null;
  }
  const match = STRICT_STABLE_SEMVER_RE.exec(value);
  if (match == null) return null;
  const major = Number(match[1]);
  const minor = Number(match[2]);
  const patch = Number(match[3]);
  if (
    !Number.isSafeInteger(major)
    || !Number.isSafeInteger(minor)
    || !Number.isSafeInteger(patch)
    || major < 0
    || minor < 0
    || patch < 0
  ) {
    return null;
  }
  return Object.freeze([major, minor, patch] as const);
}

export function isValidCodexVersion(value: unknown): value is string {
  return parseCodexSemver(value) !== null;
}

export function validateCodexVersion(
  value: unknown,
  source = "npm @openai/codex version",
): string {
  if (parseCodexSemver(value) === null) {
    throw new ChatGPTOAuthCodexVersionUnavailableError(
      `${source} must be a strict stable three-component semantic version`,
    );
  }
  return value as string;
}

export function requireExplicitCodexVersion(
  value: unknown,
  source = "Codex compatibility version",
): string {
  if (parseCodexSemver(value) === null) {
    throw new ChatGPTOAuthInvalidRequestError(
      `${source} must be a strict stable three-component semantic version`,
    );
  }
  return value as string;
}

function compareSemverTuples(
  left: CodexSemverTuple,
  right: CodexSemverTuple,
): -1 | 0 | 1 {
  for (let index = 0; index < 3; index++) {
    if (left[index]! < right[index]!) return -1;
    if (left[index]! > right[index]!) return 1;
  }
  return 0;
}

export function compareCodexVersions(left: string, right: string): -1 | 0 | 1 {
  const leftSemver = parseCodexSemver(left);
  const rightSemver = parseCodexSemver(right);
  if (leftSemver === null || rightSemver === null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "compared Codex versions must be strict stable three-component semantic versions",
    );
  }
  return compareSemverTuples(leftSemver, rightSemver);
}

interface JsonObjectFrame {
  readonly type: "object";
  readonly keys: Set<string>;
  expectingKey: boolean;
}

interface JsonArrayFrame {
  readonly type: "array";
}

type JsonContainerFrame = JsonObjectFrame | JsonArrayFrame;

function assertNoDuplicateJsonKeys(text: string): void {
  const stack: JsonContainerFrame[] = [];
  let index = 0;
  const length = text.length;

  while (index < length) {
    const ch = text[index]!;
    if (ch === " " || ch === "\t" || ch === "\r" || ch === "\n") {
      index++;
      continue;
    }
    if (ch === "{") {
      stack.push({ type: "object", keys: new Set<string>(), expectingKey: true });
      index++;
      continue;
    }
    if (ch === "}") {
      stack.pop();
      index++;
      continue;
    }
    if (ch === "[") {
      stack.push({ type: "array" });
      index++;
      continue;
    }
    if (ch === "]") {
      stack.pop();
      index++;
      continue;
    }
    if (ch === ",") {
      const top = stack[stack.length - 1];
      if (top !== undefined && top.type === "object") {
        top.expectingKey = true;
      }
      index++;
      continue;
    }
    if (ch === ":") {
      const top = stack[stack.length - 1];
      if (top !== undefined && top.type === "object") {
        top.expectingKey = false;
      }
      index++;
      continue;
    }
    if (ch === '"') {
      const start = index;
      index++;
      while (index < length) {
        const current = text[index]!;
        if (current === "\\") {
          index += 2;
          continue;
        }
        if (current === '"') {
          index++;
          break;
        }
        index++;
      }
      const top = stack[stack.length - 1];
      if (top !== undefined && top.type === "object" && top.expectingKey) {
        const rawKey = text.slice(start, index);
        const decodedKey = JSON.parse(rawKey) as string;
        if (top.keys.has(decodedKey)) {
          throw new SyntaxError("JSON object contains duplicate key");
        }
        top.keys.add(decodedKey);
        top.expectingKey = false;
      }
      continue;
    }
    while (index < length) {
      const current = text[index]!;
      if (
        current === " "
        || current === "\t"
        || current === "\r"
        || current === "\n"
        || current === "{"
        || current === "}"
        || current === "["
        || current === "]"
        || current === ":"
        || current === ","
        || current === '"'
      ) {
        break;
      }
      index++;
    }
  }
}

export function parseJsonStrictNoDuplicateKeys(text: string): unknown {
  const parsed = parseJsonStrict(text);
  assertNoDuplicateJsonKeys(text);
  return parsed;
}

export function parseCodexNpmLatestPayload(
  payload: string | Uint8Array,
): string {
  let text: string;
  if (typeof payload === "string") {
    text = payload;
  } else {
    try {
      text = decodeUtf8Strict(payload);
    } catch (err) {
      throw new ChatGPTOAuthCodexVersionUnavailableError(
        "npm @openai/codex metadata response is not valid UTF-8",
        { cause: err },
      );
    }
  }

  let parsed: unknown;
  try {
    parsed = parseJsonStrictNoDuplicateKeys(text);
  } catch (err) {
    throw new ChatGPTOAuthCodexVersionUnavailableError(
      "npm @openai/codex metadata response is not valid strict JSON",
      { cause: err },
    );
  }

  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new ChatGPTOAuthCodexVersionUnavailableError(
      "npm @openai/codex metadata response must be a JSON object",
    );
  }

  const record = parsed as Record<string, unknown>;
  if (record.name !== CODEX_NPM_PACKAGE_NAME) {
    throw new ChatGPTOAuthCodexVersionUnavailableError(
      `npm @openai/codex metadata response has unexpected package name`,
    );
  }

  return validateCodexVersion(
    record.version,
    "npm @openai/codex metadata version",
  );
}

const CODEX_AS_API_PACKAGE_VERSION = requireExplicitCodexVersion(
  packageMetadata.version,
  "codex-as-api package version",
);

function codexOsName(): string {
  const platform = os.platform();
  switch (platform) {
    case "darwin":
      return "Mac OS";
    case "win32":
      return "Windows";
    case "linux":
      return "Linux";
    default:
      return platform;
  }
}

function codexOsInfo(): string {
  return `${codexOsName()} ${os.release()}; ${os.arch()}`;
}

function sanitizeHeaderValue(value: string): string {
  return value.replace(/[^\x20-\x7E]/g, "_");
}

export function codexUserAgentForVersion(
  version: string,
  codexAsApiVersion = CODEX_AS_API_PACKAGE_VERSION,
): string {
  const normalizedVersion = requireExplicitCodexVersion(
    version,
    "Codex compatibility version",
  );
  const normalizedPackageVersion = requireExplicitCodexVersion(
    codexAsApiVersion,
    "codex-as-api package version",
  );
  return sanitizeHeaderValue(
    `${CODEX_CLI_ORIGINATOR}/${normalizedVersion} (${codexOsInfo()}) codex-as-api/${normalizedPackageVersion}`,
  );
}

export function codexCliHeadersForVersion(
  version: string,
  codexAsApiVersion = CODEX_AS_API_PACKAGE_VERSION,
): Record<string, string> {
  return {
    originator: CODEX_CLI_ORIGINATOR,
    "User-Agent": codexUserAgentForVersion(version, codexAsApiVersion),
  };
}

function validatePositiveFiniteNumber(value: number, name: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new ChatGPTOAuthInvalidRequestError(
      `${name} must be a positive finite number`,
    );
  }
  return value;
}

function validateRegistryUrl(url: string): string {
  if (typeof url !== "string" || url.length === 0 || url !== url.trim()) {
    throw new ChatGPTOAuthInvalidRequestError(
      "registryUrl must be a non-empty URL without surrounding whitespace",
    );
  }
  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    throw new ChatGPTOAuthInvalidRequestError(
      "registryUrl must be a valid HTTP(S) URL",
    );
  }
  if (
    (parsed.protocol !== "https:" && parsed.protocol !== "http:")
    || parsed.hostname.length === 0
    || parsed.username.length > 0
    || parsed.password.length > 0
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "registryUrl must be an HTTP(S) URL without credentials",
    );
  }
  return url;
}

async function cancelResponseBodySafely(response: Response): Promise<void> {
  try {
    await response.body?.cancel();
  } catch {
    // Ignore cleanup errors when discarding an unusable upstream response body.
  }
}

export class CodexVersionResolver {
  private fetchImpl: CodexVersionFetch;
  private now: CodexVersionClock;
  private ttlMs: number;
  private retryBackoffMs: number;
  private timeoutMs: number;
  private registryUrl: string;

  private lastGoodVersion: string | null = null;
  private lastGoodSemver: CodexSemverTuple | null = null;
  private expiresAt: number | null = null;
  private retryNotBefore: number | null = null;
  private inFlight: Promise<string> | null = null;
  private revision = 0;

  constructor(options: CodexVersionResolverOptions = {}) {
    this.fetchImpl = options.fetchImpl ?? ((input, init) => fetch(input, init));
    this.now = options.now ?? (() => performance.now());
    this.ttlMs = validatePositiveFiniteNumber(
      options.ttlMs ?? CODEX_VERSION_CACHE_TTL_MS,
      "ttlMs",
    );
    this.retryBackoffMs = validatePositiveFiniteNumber(
      options.retryBackoffMs ?? CODEX_VERSION_RETRY_BACKOFF_MS,
      "retryBackoffMs",
    );
    this.timeoutMs = validatePositiveFiniteNumber(
      options.timeoutMs ?? CODEX_VERSION_FETCH_TIMEOUT_MS,
      "timeoutMs",
    );
    this.registryUrl = validateRegistryUrl(
      options.registryUrl ?? CODEX_NPM_LATEST_URL,
    );
  }

  configure(options: CodexVersionResolverOptions = {}): void {
    this.fetchImpl = options.fetchImpl ?? ((input, init) => fetch(input, init));
    this.now = options.now ?? (() => performance.now());
    this.ttlMs = validatePositiveFiniteNumber(
      options.ttlMs ?? CODEX_VERSION_CACHE_TTL_MS,
      "ttlMs",
    );
    this.retryBackoffMs = validatePositiveFiniteNumber(
      options.retryBackoffMs ?? CODEX_VERSION_RETRY_BACKOFF_MS,
      "retryBackoffMs",
    );
    this.timeoutMs = validatePositiveFiniteNumber(
      options.timeoutMs ?? CODEX_VERSION_FETCH_TIMEOUT_MS,
      "timeoutMs",
    );
    this.registryUrl = validateRegistryUrl(
      options.registryUrl ?? CODEX_NPM_LATEST_URL,
    );
    this.reset();
  }

  reset(): void {
    this.revision++;
    this.lastGoodVersion = null;
    this.lastGoodSemver = null;
    this.expiresAt = null;
    this.retryNotBefore = null;
    this.inFlight = null;
  }

  cachedVersion(): string | null {
    return this.lastGoodVersion;
  }

  async getVersion(): Promise<string> {
    return this.resolve();
  }

  async resolve(): Promise<string> {
    const currentNow = this.now();
    if (
      this.lastGoodVersion !== null
      && this.expiresAt !== null
      && currentNow < this.expiresAt
    ) {
      return this.lastGoodVersion;
    }
    if (
      this.lastGoodVersion !== null
      && this.retryNotBefore !== null
      && currentNow < this.retryNotBefore
    ) {
      return this.lastGoodVersion;
    }
    if (this.inFlight !== null) {
      return this.inFlight;
    }

    const observedRevision = this.revision;
    const flight = this.performLookup(observedRevision);
    this.inFlight = flight;
    try {
      return await flight;
    } finally {
      if (this.inFlight === flight) {
        this.inFlight = null;
      }
    }
  }

  async headers(
    codexAsApiVersion = CODEX_AS_API_PACKAGE_VERSION,
  ): Promise<Record<string, string>> {
    const version = await this.resolve();
    return codexCliHeadersForVersion(version, codexAsApiVersion);
  }

  private async performLookup(observedRevision: number): Promise<string> {
    try {
      const fetchedVersion = await this.fetchLatestVersion();
      const fetchedSemver = parseCodexSemver(fetchedVersion);
      if (fetchedSemver === null) {
        throw new ChatGPTOAuthCodexVersionUnavailableError(
          "npm @openai/codex metadata version is invalid",
        );
      }
      if (this.revision !== observedRevision) {
        if (
          this.lastGoodVersion !== null
          && this.lastGoodSemver !== null
          && compareSemverTuples(fetchedSemver, this.lastGoodSemver) < 0
        ) {
          return this.lastGoodVersion;
        }
        return fetchedVersion;
      }
      const completedAt = this.now();
      if (
        this.lastGoodVersion === null
        || this.lastGoodSemver === null
        || compareSemverTuples(fetchedSemver, this.lastGoodSemver) >= 0
      ) {
        this.lastGoodVersion = fetchedVersion;
        this.lastGoodSemver = fetchedSemver;
      }
      this.expiresAt = completedAt + this.ttlMs;
      this.retryNotBefore = null;
      return this.lastGoodVersion;
    } catch (err) {
      const unavailableError =
        err instanceof ChatGPTOAuthCodexVersionUnavailableError
          ? err
          : new ChatGPTOAuthCodexVersionUnavailableError(
              `failed to resolve latest @openai/codex version from npm: ${
                err instanceof Error ? err.message : String(err)
              }`,
              { cause: err },
            );

      if (this.revision === observedRevision && this.lastGoodVersion !== null) {
        this.retryNotBefore = this.now() + this.retryBackoffMs;
        return this.lastGoodVersion;
      }
      throw unavailableError;
    }
  }

  private async fetchLatestVersion(): Promise<string> {
    const controller = new AbortController();
    let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
    const timeoutPromise = new Promise<never>((_resolve, reject) => {
      timeoutHandle = setTimeout(() => {
        const timeoutErr = new ChatGPTOAuthCodexVersionUnavailableError(
          `npm @openai/codex version lookup timed out after ${this.timeoutMs}ms`,
        );
        controller.abort(timeoutErr);
        reject(timeoutErr);
      }, this.timeoutMs);
      // Keep the lookup deadline alive even if fetch has no active I/O handle.
      // The finally block clears it as soon as the lookup settles.
    });

    const fetchAndParse = (async (): Promise<string> => {
      let response: Response;
      try {
        response = await this.fetchImpl(this.registryUrl, {
          method: "GET",
          redirect: "manual",
          headers: {
            Accept: "application/json",
          },
          signal: controller.signal,
        });
      } catch (err) {
        if (err instanceof ChatGPTOAuthCodexVersionUnavailableError) {
          throw err;
        }
        throw new ChatGPTOAuthCodexVersionUnavailableError(
          `npm @openai/codex version lookup request failed: ${
            err instanceof Error ? err.message : String(err)
          }`,
          { cause: err },
        );
      }

      if (response.status !== 200 || !response.ok) {
        await cancelResponseBodySafely(response);
        throw new ChatGPTOAuthCodexVersionUnavailableError(
          `npm @openai/codex version lookup failed with HTTP ${response.status}`,
        );
      }

      let bytes: Uint8Array;
      try {
        bytes = new Uint8Array(await response.arrayBuffer());
      } catch (err) {
        if (err instanceof ChatGPTOAuthCodexVersionUnavailableError) {
          throw err;
        }
        throw new ChatGPTOAuthCodexVersionUnavailableError(
          `failed to read npm @openai/codex version response body: ${
            err instanceof Error ? err.message : String(err)
          }`,
          { cause: err },
        );
      }

      return parseCodexNpmLatestPayload(bytes);
    })();

    try {
      return await Promise.race([fetchAndParse, timeoutPromise]);
    } finally {
      if (timeoutHandle !== undefined) {
        clearTimeout(timeoutHandle);
      }
    }
  }
}

export const defaultCodexVersionResolver = new CodexVersionResolver();

export async function resolveCodexCliVersion(
  resolver: CodexVersionProvider = defaultCodexVersionResolver,
): Promise<string> {
  const version = await (typeof resolver === "function" ? resolver() : resolver.resolve());
  return validateCodexVersion(version, "Codex version resolver");
}

export async function resolveLatestCodexVersion(
  resolver: CodexVersionProvider = defaultCodexVersionResolver,
): Promise<string> {
  return resolveCodexCliVersion(resolver);
}

export async function resolveCodexCliHeaders(
  resolver: CodexVersionProvider = defaultCodexVersionResolver,
): Promise<Record<string, string>> {
  return codexCliHeadersForVersion(await resolveCodexCliVersion(resolver));
}

export function getCachedCodexCliVersion(): string | null {
  return defaultCodexVersionResolver.cachedVersion();
}

export function resetCodexVersionCache(
  options?: CodexVersionResolverOptions,
): void {
  if (options !== undefined) {
    defaultCodexVersionResolver.configure(options);
  } else {
    defaultCodexVersionResolver.configure();
  }
}
