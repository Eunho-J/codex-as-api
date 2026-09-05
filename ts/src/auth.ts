import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import * as crypto from "node:crypto";
import {
  decodeUtf8Strict,
  parseJsonResponseStrict,
  parseJsonStrict,
} from "./utf8-json.js";

const CHATGPT_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const DEFAULT_AUTH_PATH = "~/.codex/auth.json";
const DEFAULT_REFRESH_URL = "https://auth.openai.com/oauth/token";
const REFRESH_URL_OVERRIDE_ENV = "CODEX_REFRESH_TOKEN_URL_OVERRIDE";
const REFRESH_WINDOW_MS = 5 * 60 * 1000;
const refreshFlights = new Map<string, Promise<ChatGPTTokenData>>();
const UNSUPPORTED_ROOT_CREDENTIAL_FIELDS = [
  "access_token",
  "refresh_token",
  "id_token",
  "chatgptAuthTokens",
] as const;

const SECRET_KEYS = [
  "access_token",
  "refresh_token",
  "id_token",
  "Authorization",
  "authorization",
  "ChatGPT-Account-Id",
  "chatgpt-account-id",
];

export class ChatGPTOAuthError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ChatGPTOAuthError";
  }
}

export class ChatGPTOAuthInvalidRequestError extends ChatGPTOAuthError {
  constructor(message: string) {
    super(message);
    this.name = "ChatGPTOAuthInvalidRequestError";
  }
}

export class ChatGPTOAuthMissingError extends ChatGPTOAuthError {
  constructor(message: string) {
    super(message);
    this.name = "ChatGPTOAuthMissingError";
  }
}

export class ChatGPTOAuthRefreshError extends ChatGPTOAuthError {
  constructor(message: string) {
    super(message);
    this.name = "ChatGPTOAuthRefreshError";
  }
}

export class ChatGPTOAuthUpstreamError extends ChatGPTOAuthError {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ChatGPTOAuthUpstreamError";
    this.status = status;
  }
}

export class ChatGPTOAuthUnavailableError extends ChatGPTOAuthError {
  constructor(message: string) {
    super(message);
    this.name = "ChatGPTOAuthUnavailableError";
  }
}

export class ChatGPTOAuthCatalogUnavailableError extends ChatGPTOAuthError {
  constructor(message: string) {
    super(message);
    this.name = "ChatGPTOAuthCatalogUnavailableError";
  }
}

export class ChatGPTOAuthModelNotFoundError extends ChatGPTOAuthError {
  readonly model: string;

  constructor(model: string) {
    super("requested model is not available in the authenticated upstream catalog");
    this.name = "ChatGPTOAuthModelNotFoundError";
    this.model = model;
  }
}

export class ChatGPTOAuthProtocolError extends ChatGPTOAuthError {
  constructor(message: string) {
    super(message);
    this.name = "ChatGPTOAuthProtocolError";
  }
}

export interface ChatGPTTokenData {
  auth_path: string;
  access_token: string;
  refresh_token: string;
  id_token: string;
  account_id: string;
  plan_type: string | null;
  user_id: string | null;
  fedramp: boolean;
  access_expires_at: Date | null;
}

export function isTokenExpired(token: ChatGPTTokenData): boolean {
  return token.access_expires_at !== null && token.access_expires_at <= new Date();
}

export function tokenExpiresWithin(
  token: ChatGPTTokenData,
  windowMs = REFRESH_WINDOW_MS,
): boolean {
  return token.access_expires_at !== null
    && token.access_expires_at.getTime() <= Date.now() + windowMs;
}

function expandHome(p: string): string {
  if (p.startsWith("~/") || p === "~") {
    return path.join(os.homedir(), p.slice(1));
  }
  return p;
}

export function resolveAuthPath(raw?: string | null): string {
  if (raw != null) {
    if (raw.trim().length === 0) {
      throw new ChatGPTOAuthError("ChatGPT OAuth auth path must not be blank");
    }
    return expandHome(raw);
  }
  const codexHome = process.env["CODEX_HOME"];
  if (codexHome !== undefined) {
    if (codexHome.trim().length === 0) {
      throw new ChatGPTOAuthError("CODEX_HOME must not be blank");
    }
    return path.join(expandHome(codexHome), "auth.json");
  }
  return expandHome(DEFAULT_AUTH_PATH);
}

function refreshEndpointFromEnvironment(): string {
  const override = process.env[REFRESH_URL_OVERRIDE_ENV];
  const endpoint = override ?? DEFAULT_REFRESH_URL;
  if (endpoint.length === 0) {
    throw new ChatGPTOAuthRefreshError(`${REFRESH_URL_OVERRIDE_ENV} must not be blank`);
  }
  if (endpoint.trim() !== endpoint) {
    throw new ChatGPTOAuthRefreshError(
      `${REFRESH_URL_OVERRIDE_ENV} must not contain surrounding whitespace`,
    );
  }
  if (/[\p{White_Space}\p{Cc}]/u.test(endpoint)) {
    throw new ChatGPTOAuthRefreshError(
      `${REFRESH_URL_OVERRIDE_ENV} must not contain raw whitespace or control characters`,
    );
  }
  if (/%(?![0-9A-Fa-f]{2})/.test(endpoint)) {
    throw new ChatGPTOAuthRefreshError(
      `${REFRESH_URL_OVERRIDE_ENV} must not contain malformed percent encoding`,
    );
  }
  let parsed: URL;
  try {
    parsed = new URL(endpoint);
  } catch {
    throw new ChatGPTOAuthRefreshError(
      `${REFRESH_URL_OVERRIDE_ENV} must be an absolute HTTP(S) URL`,
    );
  }
  const hasExplicitAuthority = endpoint
    .split("://", 2)[1]
    ?.split(/[/?#]/, 1)[0]
    .length;
  if (
    (parsed.protocol !== "https:" && parsed.protocol !== "http:")
    || parsed.hostname.length === 0
    || !hasExplicitAuthority
    || parsed.username.length > 0
    || parsed.password.length > 0
    || parsed.search.length > 0
    || parsed.hash.length > 0
  ) {
    throw new ChatGPTOAuthRefreshError(
      `${REFRESH_URL_OVERRIDE_ENV} must be an absolute HTTP(S) URL without credentials, query, or fragment`,
    );
  }
  return endpoint;
}

export function validateAuthEnvironment(): void {
  refreshEndpointFromEnvironment();
}

function jwtClaims(jwt: string): Record<string, unknown> {
  const parts = jwt.split(".");
  if (
    parts.length !== 3
    || parts.some((part) => part.length === 0)
  ) {
    throw new ChatGPTOAuthError("invalid ChatGPT OAuth JWT structure");
  }
  if (
    !/^[A-Za-z0-9_-]+$/.test(parts[1])
    || parts[1].length % 4 === 1
    || Buffer.from(parts[1], "base64url").toString("base64url") !== parts[1]
  ) {
    throw new ChatGPTOAuthError("invalid ChatGPT OAuth JWT payload");
  }
  const payload = parts[1] + "=".repeat((4 - (parts[1].length % 4)) % 4);
  let decoded: string;
  try {
    decoded = new TextDecoder("utf-8", { fatal: true }).decode(
      Buffer.from(payload, "base64url"),
    );
  } catch {
    throw new ChatGPTOAuthError("invalid ChatGPT OAuth JWT payload");
  }
  let value: unknown;
  try {
    value = parseJsonStrict(decoded);
  } catch {
    throw new ChatGPTOAuthError("invalid ChatGPT OAuth JWT payload");
  }
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ChatGPTOAuthError("invalid ChatGPT OAuth JWT claims");
  }
  return value as Record<string, unknown>;
}

function expiration(jwt: string): Date | null {
  const claims = jwtClaims(jwt);
  const exp = claims["exp"];
  if (exp == null) {
    return null;
  }
  if (!Number.isSafeInteger(exp)) {
    throw new ChatGPTOAuthError("ChatGPT OAuth JWT exp claim must be an integer");
  }
  const value = new Date((exp as number) * 1000);
  if (Number.isNaN(value.getTime())) {
    throw new ChatGPTOAuthError("ChatGPT OAuth JWT exp claim is out of range");
  }
  return value;
}

function authClaims(jwt: string): Record<string, unknown> {
  const claims = jwtClaims(jwt);
  const value = claims["https://api.openai.com/auth"];
  if (value === undefined) return {};
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  throw new ChatGPTOAuthError("ChatGPT OAuth JWT auth claim must be an object");
}

export function redactText(text: string, ...values: (string | null | undefined)[]): string {
  let redacted = String(text);
  const filtered = values.filter((v): v is string => typeof v === "string" && v.length > 0);
  filtered.sort((a, b) => b.length - a.length);
  const marker = filtered.some((value) => "***".includes(value)) ? "" : "***";
  for (const value of filtered) {
    redacted = redacted.split(value).join(marker);
  }
  let removedBoundaryMatch = true;
  while (removedBoundaryMatch) {
    removedBoundaryMatch = false;
    for (const value of filtered) {
      if (redacted.includes(value)) {
        redacted = redacted.split(value).join("");
        removedBoundaryMatch = true;
      }
    }
  }
  return redacted;
}

export function loadTokenData(authJsonPath?: string | null): ChatGPTTokenData {
  const authPath = resolveAuthPath(authJsonPath !== undefined ? (authJsonPath ?? null) : undefined);
  let rawBytes: Buffer;
  try {
    rawBytes = fs.readFileSync(authPath);
  } catch (err: unknown) {
    if (err instanceof Error && "code" in err && (err as NodeJS.ErrnoException).code === "ENOENT") {
      throw new ChatGPTOAuthMissingError("ChatGPT OAuth auth file not found");
    }
    throw new ChatGPTOAuthMissingError("ChatGPT OAuth auth file is unavailable");
  }
  let raw: string;
  try {
    raw = decodeUtf8Strict(rawBytes);
  } catch {
    throw new ChatGPTOAuthError("ChatGPT OAuth auth file is not valid UTF-8");
  }
  let data: unknown;
  try {
    data = parseJsonStrict(raw);
  } catch {
    throw new ChatGPTOAuthError("ChatGPT OAuth auth file is invalid JSON");
  }
  return tokenDataFromDocument(data, authPath);
}

function tokenDataFromDocument(data: unknown, authPath: string): ChatGPTTokenData {
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new ChatGPTOAuthError("ChatGPT OAuth auth file root must be an object");
  }
  const d = data as Record<string, unknown>;
  const mode = d["auth_mode"];
  if (
    mode !== undefined
    && mode !== null
    && mode !== "chatgpt"
  ) {
    throw new ChatGPTOAuthError("ChatGPT OAuth auth_mode is unsupported");
  }
  const tokens = d["tokens"];
  if (typeof tokens !== "object" || tokens === null || Array.isArray(tokens)) {
    if (Object.hasOwn(d, "tokens")) {
      throw new ChatGPTOAuthError("ChatGPT OAuth auth file tokens must be an object");
    }
    throw new ChatGPTOAuthError(unsupportedAuthSchemaMessage(d));
  }
  if (hasUnsupportedRootCredentialFields(d)) {
    throw new ChatGPTOAuthError(
      "ChatGPT OAuth auth file mixes unsupported root credential fields with managed tokens",
    );
  }
  const t = tokens as Record<string, unknown>;
  const access_token = t["access_token"];
  const refresh_token = t["refresh_token"];
  const id_token = t["id_token"];
  for (const [name, value] of [
    ["access_token", access_token],
    ["refresh_token", refresh_token],
    ["id_token", id_token],
  ] as const) {
    if (typeof value !== "string" || value.trim().length === 0) {
      throw new ChatGPTOAuthError(`ChatGPT OAuth ${name} is missing`);
    }
  }
  if (!isVisibleAsciiHeaderValue(access_token as string)) {
    throw new ChatGPTOAuthError(
      "ChatGPT OAuth access_token is invalid for an HTTP header",
    );
  }
  const idAuth = authClaims(id_token as string);
  const accessAuth = authClaims(access_token as string);
  const accountSources = new Map<string, unknown>([
    ["account_id", t["account_id"]],
    ["id_token", idAuth["chatgpt_account_id"]],
    ["access_token", accessAuth["chatgpt_account_id"]],
  ]);
  for (const [source, value] of accountSources) {
    if (value !== undefined && value !== null && (typeof value !== "string" || value.trim().length === 0)) {
      throw new ChatGPTOAuthError(`ChatGPT OAuth ${source} account id is invalid`);
    }
  }
  const accountIds = new Set(
    [...accountSources.values()].filter((value): value is string => typeof value === "string" && value.trim().length > 0),
  );
  if (accountIds.size === 0) {
    throw new ChatGPTOAuthError("ChatGPT OAuth account id not available; rerun codex login");
  }
  if (accountIds.size !== 1) {
    throw new ChatGPTOAuthError("ChatGPT OAuth token account ids do not match");
  }
  const account_id = [...accountIds][0];
  if (!isVisibleAsciiHeaderValue(account_id)) {
    throw new ChatGPTOAuthError(
      "ChatGPT OAuth account id is invalid for an HTTP header",
    );
  }
  const plan = matchingOptionalStringClaim([
    [idAuth["chatgpt_plan_type"], "id_token chatgpt_plan_type"],
    [accessAuth["chatgpt_plan_type"], "access_token chatgpt_plan_type"],
  ], "ChatGPT OAuth plan claims");
  const user = matchingOptionalStringClaim([
    [idAuth["chatgpt_user_id"], "id_token chatgpt_user_id"],
    [idAuth["user_id"], "id_token user_id"],
    [accessAuth["chatgpt_user_id"], "access_token chatgpt_user_id"],
    [accessAuth["user_id"], "access_token user_id"],
  ], "ChatGPT OAuth user claims");
  const idFedramp = optionalBooleanClaim(
    idAuth["chatgpt_account_is_fedramp"],
    "id_token chatgpt_account_is_fedramp",
  );
  const accessFedramp = optionalBooleanClaim(
    accessAuth["chatgpt_account_is_fedramp"],
    "access_token chatgpt_account_is_fedramp",
  );
  if (
    idFedramp !== undefined
    && accessFedramp !== undefined
    && idFedramp !== accessFedramp
  ) {
    throw new ChatGPTOAuthError("ChatGPT OAuth FedRAMP claims do not match");
  }
  const fedramp = idFedramp ?? accessFedramp ?? false;
  return {
    auth_path: authPath,
    access_token: access_token as string,
    refresh_token: refresh_token as string,
    id_token: id_token as string,
    account_id: account_id as string,
    plan_type: plan ?? null,
    user_id: user ?? null,
    fedramp,
    access_expires_at: expiration(access_token as string),
  };
}

function hasUnsupportedRootCredentialFields(data: Record<string, unknown>): boolean {
  return UNSUPPORTED_ROOT_CREDENTIAL_FIELDS.some((key) => Object.hasOwn(data, key));
}

function isVisibleAsciiHeaderValue(value: string): boolean {
  return /^[\x21-\x7E]+$/.test(value);
}

function optionalBooleanClaim(value: unknown, field: string): boolean | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "boolean") {
    throw new ChatGPTOAuthError(`ChatGPT OAuth ${field} must be a boolean`);
  }
  return value;
}

function matchingOptionalStringClaim(
  sources: readonly (readonly [unknown, string])[],
  group: string,
): string | undefined {
  const values = new Set<string>();
  for (const [value, field] of sources) {
    if (value === undefined) continue;
    if (typeof value !== "string" || value.trim().length === 0) {
      throw new ChatGPTOAuthError(`ChatGPT OAuth ${field} must be a non-empty string`);
    }
    values.add(value);
  }
  if (values.size > 1) {
    throw new ChatGPTOAuthError(`${group} do not match`);
  }
  return values.values().next().value;
}

function unsupportedAuthSchemaMessage(data: Record<string, unknown>): string {
  const hasFileTokens = ["tokens", "access_token", "refresh_token", "id_token"].some((key) => key in data);
  if ("personal_access_token" in data && !hasFileTokens) {
    return "ChatGPT OAuth personal_access_token-only auth is not supported; rerun codex login to create file-backed tokens";
  }
  if ("agent_identity" in data && !hasFileTokens) {
    return "ChatGPT OAuth agent_identity-only auth is not supported; rerun codex login to create file-backed tokens";
  }
  if ("bedrock_api_key" in data && !hasFileTokens) {
    return "ChatGPT OAuth bedrock_api_key-only auth is not supported by the ChatGPT OAuth backend";
  }
  return "ChatGPT OAuth file-backed ChatGPT OAuth tokens are required; rerun codex login";
}

export function isAuthLocallyAvailable(authJsonPath?: string | null): boolean {
  try {
    const data = loadTokenData(authJsonPath);
    return Boolean(data.access_token && data.account_id);
  } catch (err) {
    if (err instanceof ChatGPTOAuthError) {
      return false;
    }
    throw err;
  }
}

const NO_WRITE_FAILURE = Symbol("no auth write failure");

function writeAuthJson(filePath: string, data: Record<string, unknown>): void {
  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = path.join(dir, `.${path.basename(filePath)}.tmp-${process.pid}-${crypto.randomUUID()}`);
  const payload = JSON.stringify(data, null, 2) + "\n";
  const fd = fs.openSync(tmp, "w", 0o600);
  let failure: unknown = NO_WRITE_FAILURE;
  try {
    fs.writeFileSync(fd, payload, "utf-8");
    fs.fsyncSync(fd);
  } catch (err) {
    failure = err;
  }
  try {
    fs.closeSync(fd);
  } catch (err) {
    failure = combineCleanupFailure(failure, err, "failed to write and close ChatGPT OAuth auth file");
  }
  if (failure === NO_WRITE_FAILURE) {
    try {
      fs.renameSync(tmp, filePath);
    } catch (err) {
      failure = err;
    }
  }
  if (failure === NO_WRITE_FAILURE) {
    try {
      fsyncAuthDirectory(dir);
    } catch (err) {
      failure = err;
    }
  }
  try {
    fs.unlinkSync(tmp);
  } catch (err) {
    if (!(err instanceof Error && "code" in err && (err as NodeJS.ErrnoException).code === "ENOENT")) {
      failure = combineCleanupFailure(failure, err, "failed to write and clean up ChatGPT OAuth auth file");
    }
  }
  if (failure !== NO_WRITE_FAILURE) throw failure;
}

function fsyncAuthDirectory(dir: string): void {
  // Windows does not support opening a directory for fsync. Unix failures are
  // durability failures and must remain visible rather than being downgraded.
  if (process.platform === "win32") return;
  let dirFd: number | undefined;
  let failure: unknown = NO_WRITE_FAILURE;
  try {
    dirFd = fs.openSync(dir, "r");
    fs.fsyncSync(dirFd);
  } catch (err) {
    failure = err;
  } finally {
    if (dirFd !== undefined) {
      try {
        fs.closeSync(dirFd);
      } catch (err) {
        failure = combineCleanupFailure(
          failure,
          err,
          "failed to sync and close ChatGPT OAuth auth directory",
        );
      }
    }
  }
  if (failure !== NO_WRITE_FAILURE) throw failure;
}

function combineCleanupFailure(primary: unknown, cleanup: unknown, message: string): unknown {
  return primary === NO_WRITE_FAILURE ? cleanup : new AggregateError([primary, cleanup], message);
}

export async function tokenForRequest(authJsonPath?: string | null): Promise<ChatGPTTokenData> {
  const current = loadTokenData(authJsonPath);
  if (!tokenExpiresWithin(current)) return current;
  return refreshToken(authJsonPath, current, true);
}

export async function refreshAfterUnauthorized(current: ChatGPTTokenData): Promise<ChatGPTTokenData> {
  return refreshToken(current.auth_path, current, false);
}

function failIfAccountChanged(current: ChatGPTTokenData, latest: ChatGPTTokenData): void {
  if (current.account_id !== latest.account_id) {
    throw new ChatGPTOAuthRefreshError("ChatGPT OAuth account changed while refreshing credentials");
  }
}

function accessTokenChanged(current: ChatGPTTokenData, latest: ChatGPTTokenData): boolean {
  return current.access_token !== latest.access_token;
}

function otherCredentialsChanged(current: ChatGPTTokenData, latest: ChatGPTTokenData): boolean {
  return current.refresh_token !== latest.refresh_token || current.id_token !== latest.id_token;
}

function validateRefreshedTokenAccounts(payload: Record<string, unknown>, accountId: string): void {
  for (const name of ["access_token", "id_token"] as const) {
    const value = payload[name];
    if (typeof value !== "string" || value.trim().length === 0) continue;
    const claimAccount = authClaims(value)["chatgpt_account_id"];
    if (claimAccount !== undefined && (typeof claimAccount !== "string" || claimAccount !== accountId)) {
      throw new ChatGPTOAuthRefreshError(
        `ChatGPT OAuth refreshed ${name} account id does not match current account`,
      );
    }
  }
}

export async function refreshToken(
  authJsonPath?: string | null,
  observed?: ChatGPTTokenData,
  refreshIfExpiring = false,
): Promise<ChatGPTTokenData> {
  const current = observed ?? loadTokenData(authJsonPath);
  const authPath = current.auth_path;
  const existing = refreshFlights.get(authPath);
  if (existing) {
    const refreshed = await existing;
    failIfAccountChanged(current, refreshed);
    return refreshed;
  }

  const flight = performRefresh(current, refreshIfExpiring);
  refreshFlights.set(authPath, flight);
  try {
    return await flight;
  } finally {
    if (refreshFlights.get(authPath) === flight) refreshFlights.delete(authPath);
  }
}

async function performRefresh(
  observed: ChatGPTTokenData,
  refreshIfExpiring: boolean,
): Promise<ChatGPTTokenData> {
  const latest = loadTokenData(observed.auth_path);
  failIfAccountChanged(observed, latest);
  if (accessTokenChanged(observed, latest)) return latest;
  if (refreshIfExpiring && !tokenExpiresWithin(latest)) return latest;
  const current = latest;
  const endpoint = refreshEndpointFromEnvironment();
  const body = JSON.stringify({
    client_id: CHATGPT_OAUTH_CLIENT_ID,
    grant_type: "refresh_token",
    refresh_token: current.refresh_token,
  });
  let response: Response;
  try {
    response = await fetch(endpoint, {
      method: "POST",
      redirect: "manual",
      headers: { "Content-Type": "application/json" },
      body,
      signal: AbortSignal.timeout(30000),
    });
  } catch (err) {
    const redacted = redactText(
      String(err),
      current.access_token,
      current.refresh_token,
      current.id_token,
      current.account_id,
    );
    throw new ChatGPTOAuthUnavailableError(`ChatGPT OAuth token refresh failed: ${redacted}`);
  }
  if (!response.ok) {
    let responseText = "could not read upstream error body";
    try {
      responseText = await response.text();
    } catch {
      // Preserve the upstream HTTP status even when its error body cannot be read.
    }
    const redacted = redactText(
      responseText,
      current.access_token,
      current.refresh_token,
      current.id_token,
      current.account_id,
    );
    if (response.status === 400 || response.status === 401) {
      throw new ChatGPTOAuthRefreshError(
        `ChatGPT OAuth refresh token is invalid; rerun codex login: ${redacted}`
      );
    }
    throw new ChatGPTOAuthUpstreamError(
      response.status,
      `ChatGPT OAuth token refresh failed: HTTP ${response.status}: ${redacted}`,
    );
  }
  let responsePayload: unknown;
  try {
    responsePayload = await parseJsonResponseStrict(response);
  } catch {
    throw new ChatGPTOAuthProtocolError(
      "ChatGPT OAuth token refresh returned invalid JSON",
    );
  }
  if (typeof responsePayload !== "object" || responsePayload === null || Array.isArray(responsePayload)) {
    throw new ChatGPTOAuthProtocolError("ChatGPT OAuth token refresh returned invalid JSON");
  }
  const p = responsePayload as Record<string, unknown>;
  if (typeof p["access_token"] !== "string" || p["access_token"].trim().length === 0) {
    throw new ChatGPTOAuthProtocolError("ChatGPT OAuth token refresh response is missing access_token");
  }
  if (!isVisibleAsciiHeaderValue(p["access_token"])) {
    throw new ChatGPTOAuthProtocolError(
      "ChatGPT OAuth token refresh response access_token is invalid for an HTTP header",
    );
  }
  for (const name of ["id_token", "refresh_token"] as const) {
    const value = p[name];
    if (value !== undefined && (typeof value !== "string" || value.trim().length === 0)) {
      throw new ChatGPTOAuthProtocolError(`ChatGPT OAuth token refresh response has invalid ${name}`);
    }
  }
  validateRefreshedTokenAccounts(p, current.account_id);

  let latestAfterRefresh: ChatGPTTokenData;
  try {
    latestAfterRefresh = loadTokenData(current.auth_path);
  } catch (err) {
    throw new Error("failed to re-read ChatGPT OAuth auth file after refresh", { cause: err });
  }
  failIfAccountChanged(current, latestAfterRefresh);
  if (accessTokenChanged(current, latestAfterRefresh)) return latestAfterRefresh;
  if (otherCredentialsChanged(current, latestAfterRefresh)) {
    throw new ChatGPTOAuthRefreshError(
      "ChatGPT OAuth credentials changed while token refresh was in flight",
    );
  }
  let data: Record<string, unknown>;
  let parsed: unknown;
  try {
    const raw = decodeUtf8Strict(fs.readFileSync(current.auth_path));
    parsed = parseJsonStrict(raw);
  } catch (err) {
    throw new Error("failed to re-read ChatGPT OAuth auth file", { cause: err });
  }
  let exactLatest: ChatGPTTokenData;
  if (
    typeof parsed !== "object"
    || parsed === null
    || Array.isArray(parsed)
    || typeof (parsed as Record<string, unknown>)["tokens"] !== "object"
    || (parsed as Record<string, unknown>)["tokens"] === null
    || Array.isArray((parsed as Record<string, unknown>)["tokens"])
  ) {
    throw new Error("ChatGPT OAuth auth file tokens must be an object");
  }
  try {
    exactLatest = tokenDataFromDocument(parsed, current.auth_path);
    data = parsed as Record<string, unknown>;
  } catch (err) {
    throw new Error("ChatGPT OAuth auth file changed to an invalid schema", { cause: err });
  }
  failIfAccountChanged(current, exactLatest);
  if (accessTokenChanged(current, exactLatest)) return exactLatest;
  if (otherCredentialsChanged(current, exactLatest)) {
    throw new ChatGPTOAuthRefreshError(
      "ChatGPT OAuth credentials changed while token refresh was in flight",
    );
  }
  const tokens = data["tokens"] as Record<string, unknown>;
  const tokenObject = tokens as Record<string, unknown>;
  if (p["id_token"]) {
    tokenObject["id_token"] = p["id_token"];
  }
  tokenObject["access_token"] = p["access_token"];
  if (p["refresh_token"]) {
    tokenObject["refresh_token"] = p["refresh_token"];
  }
  data["last_refresh"] = new Date().toISOString().replace("+00:00", "Z");
  writeAuthJson(current.auth_path, data);
  try {
    return loadTokenData(current.auth_path);
  } catch (err) {
    throw new Error("failed to load persisted ChatGPT OAuth credentials", { cause: err });
  }
}
