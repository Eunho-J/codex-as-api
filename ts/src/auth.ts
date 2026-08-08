import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import * as crypto from "node:crypto";

const CHATGPT_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const DEFAULT_AUTH_PATH = "~/.codex/auth.json";
const DEFAULT_REFRESH_URL = "https://auth.openai.com/oauth/token";
const REFRESH_URL_OVERRIDE_ENV = "CODEX_REFRESH_TOKEN_URL_OVERRIDE";
const REFRESH_WINDOW_MS = 5 * 60 * 1000;
const refreshFlights = new Map<string, Promise<ChatGPTTokenData>>();

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
  const value = raw || process.env["CODEX_HOME"];
  if (value && raw === undefined) {
    return path.join(expandHome(value), "auth.json");
  }
  if (value && raw === null) {
    return path.join(expandHome(value), "auth.json");
  }
  return expandHome(raw || DEFAULT_AUTH_PATH);
}

function jwtClaims(jwt: string): Record<string, unknown> {
  const parts = jwt.split(".");
  if (parts.length < 2 || !parts[1]) {
    return {};
  }
  const payload = parts[1] + "=".repeat((4 - (parts[1].length % 4)) % 4);
  let decoded: string;
  try {
    decoded = Buffer.from(payload, "base64url").toString("utf-8");
  } catch {
    throw new ChatGPTOAuthError("invalid ChatGPT OAuth JWT payload");
  }
  let value: unknown;
  try {
    value = JSON.parse(decoded);
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
  if (typeof exp !== "number") {
    return null;
  }
  return new Date(exp * 1000);
}

function authClaims(jwt: string): Record<string, unknown> {
  const claims = jwtClaims(jwt);
  const value = claims["https://api.openai.com/auth"];
  if (typeof value === "object" && value !== null && !Array.isArray(value)) {
    return value as Record<string, unknown>;
  }
  return {};
}

export function redactText(text: string, ...values: (string | null | undefined)[]): string {
  let redacted = String(text);
  const filtered = values.filter((v): v is string => typeof v === "string" && v.length > 0);
  filtered.sort((a, b) => b.length - a.length);
  for (const value of filtered) {
    while (redacted.includes(value)) {
      redacted = redacted.replace(value, "***");
    }
  }
  return redacted;
}

export function loadTokenData(authJsonPath?: string | null): ChatGPTTokenData {
  const authPath = resolveAuthPath(authJsonPath !== undefined ? (authJsonPath ?? null) : undefined);
  let raw: string;
  try {
    raw = fs.readFileSync(authPath, "utf-8");
  } catch (err: unknown) {
    if (err instanceof Error && "code" in err && (err as NodeJS.ErrnoException).code === "ENOENT") {
      throw new ChatGPTOAuthMissingError(`ChatGPT OAuth auth file not found: ${authPath}`);
    }
    throw err;
  }
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    throw new ChatGPTOAuthError(`ChatGPT OAuth auth file is invalid JSON: ${authPath}`);
  }
  if (typeof data !== "object" || data === null || Array.isArray(data)) {
    throw new ChatGPTOAuthError("ChatGPT OAuth auth file root must be an object");
  }
  const d = data as Record<string, unknown>;
  const mode = d["auth_mode"];
  const validModes = new Set(["chatgpt", "Chatgpt", "chatgpt_auth_tokens", "ChatgptAuthTokens", undefined, null]);
  if (!validModes.has(mode as string | undefined | null)) {
    throw new ChatGPTOAuthError(`ChatGPT OAuth auth_mode required, got ${JSON.stringify(mode)}`);
  }
  let tokens = d["tokens"];
  if (tokens == null) tokens = rootTokensFromLatestAuth(d);
  if (typeof tokens !== "object" || tokens === null || Array.isArray(tokens)) {
    throw new ChatGPTOAuthError(unsupportedAuthSchemaMessage(d));
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
    if (typeof value !== "string" || value === "") {
      throw new ChatGPTOAuthError(`ChatGPT OAuth ${name} is missing`);
    }
  }
  const idAuth = authClaims(id_token as string);
  const accessAuth = authClaims(access_token as string);
  const accountSources = new Map<string, unknown>([
    ["account_id", t["account_id"]],
    ["id_token", idAuth["chatgpt_account_id"]],
    ["access_token", accessAuth["chatgpt_account_id"]],
  ]);
  for (const [source, value] of accountSources) {
    if (value !== undefined && value !== null && (typeof value !== "string" || value.length === 0)) {
      throw new ChatGPTOAuthError(`ChatGPT OAuth ${source} account id is invalid`);
    }
  }
  const accountIds = new Set(
    [...accountSources.values()].filter((value): value is string => typeof value === "string" && value.length > 0),
  );
  if (accountIds.size === 0) {
    throw new ChatGPTOAuthError("ChatGPT OAuth account id not available; rerun codex login");
  }
  if (accountIds.size !== 1) {
    throw new ChatGPTOAuthError("ChatGPT OAuth token account ids do not match");
  }
  const account_id = [...accountIds][0];
  const plan =
    idAuth["chatgpt_plan_type"] || accessAuth["chatgpt_plan_type"];
  const user =
    idAuth["chatgpt_user_id"] ||
    idAuth["user_id"] ||
    accessAuth["chatgpt_user_id"] ||
    accessAuth["user_id"];
  const fedramp = Boolean(
    idAuth["chatgpt_account_is_fedramp"] || accessAuth["chatgpt_account_is_fedramp"]
  );
  return {
    auth_path: authPath,
    access_token: access_token as string,
    refresh_token: refresh_token as string,
    id_token: id_token as string,
    account_id: account_id as string,
    plan_type: typeof plan === "string" ? plan : null,
    user_id: typeof user === "string" ? user : null,
    fedramp,
    access_expires_at: expiration(access_token as string),
  };
}

function rootTokensFromLatestAuth(data: Record<string, unknown>): Record<string, unknown> | null {
  const names = ["access_token", "refresh_token", "id_token", "account_id"];
  if (!names.some((name) => typeof data[name] === "string")) return null;
  return Object.fromEntries(names.filter((name) => name in data).map((name) => [name, data[name]]));
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

function writeAuthJson(filePath: string, data: Record<string, unknown>): void {
  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true });
  const tmp = path.join(dir, `.${path.basename(filePath)}.tmp-${process.pid}-${crypto.randomUUID()}`);
  const payload = JSON.stringify(data, null, 2) + "\n";
  const fd = fs.openSync(tmp, "w", 0o600);
  try {
    fs.writeFileSync(fd, payload, "utf-8");
    fs.fsyncSync(fd);
    fs.closeSync(fd);
    fs.renameSync(tmp, filePath);
  } finally {
    try {
      fs.unlinkSync(tmp);
    } catch {
      // tmp already renamed or cleaned up
    }
  }
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
    if (typeof value !== "string" || value.length === 0) continue;
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
  if (existing) return existing;

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
  const endpoint = process.env[REFRESH_URL_OVERRIDE_ENV] || DEFAULT_REFRESH_URL;
  const body = JSON.stringify({
    client_id: CHATGPT_OAUTH_CLIENT_ID,
    grant_type: "refresh_token",
    refresh_token: current.refresh_token,
  });
  let responsePayload: unknown;
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      signal: AbortSignal.timeout(30000),
    });
    if (!response.ok) {
      const text = await response.text();
      const redacted = redactText(
        text,
        current.access_token,
        current.refresh_token,
        current.id_token,
        current.account_id,
      );
      if (response.status === 401) {
        throw new ChatGPTOAuthRefreshError(
          `ChatGPT OAuth refresh token is invalid; rerun codex login: ${redacted}`
        );
      }
      throw new ChatGPTOAuthRefreshError(
        `ChatGPT OAuth token refresh failed: HTTP ${response.status}: ${redacted}`
      );
    }
    responsePayload = await response.json();
  } catch (err) {
    if (err instanceof ChatGPTOAuthRefreshError) {
      throw err;
    }
    const redacted = redactText(
      String(err),
      current.access_token,
      current.refresh_token,
      current.id_token,
      current.account_id,
    );
    throw new ChatGPTOAuthRefreshError(`ChatGPT OAuth token refresh failed: ${redacted}`);
  }
  if (typeof responsePayload !== "object" || responsePayload === null || Array.isArray(responsePayload)) {
    throw new ChatGPTOAuthRefreshError("ChatGPT OAuth token refresh returned invalid JSON");
  }
  const p = responsePayload as Record<string, unknown>;
  if (typeof p["access_token"] !== "string" || p["access_token"].length === 0) {
    throw new ChatGPTOAuthRefreshError("ChatGPT OAuth token refresh response is missing access_token");
  }
  for (const name of ["id_token", "refresh_token"] as const) {
    const value = p[name];
    if (value !== undefined && (typeof value !== "string" || value.length === 0)) {
      throw new ChatGPTOAuthRefreshError(`ChatGPT OAuth token refresh response has invalid ${name}`);
    }
  }
  validateRefreshedTokenAccounts(p, current.account_id);

  const latestAfterRefresh = loadTokenData(current.auth_path);
  failIfAccountChanged(current, latestAfterRefresh);
  if (accessTokenChanged(current, latestAfterRefresh)) return latestAfterRefresh;
  if (otherCredentialsChanged(current, latestAfterRefresh)) {
    throw new ChatGPTOAuthRefreshError(
      "ChatGPT OAuth credentials changed while token refresh was in flight",
    );
  }
  let data: Record<string, unknown>;
  try {
    const raw = fs.readFileSync(current.auth_path, "utf-8");
    const parsed = JSON.parse(raw) as unknown;
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
      throw new Error("root must be an object");
    }
    data = parsed as Record<string, unknown>;
  } catch (err) {
    throw new ChatGPTOAuthRefreshError(`failed to re-read ChatGPT OAuth auth file: ${err}`);
  }
  const nestedTokens = data["tokens"];
  const tokens = nestedTokens == null ? data : nestedTokens;
  if (typeof tokens !== "object" || tokens === null || Array.isArray(tokens)) {
    throw new ChatGPTOAuthRefreshError("ChatGPT OAuth auth file tokens must be an object");
  }
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
  return loadTokenData(current.auth_path);
}
