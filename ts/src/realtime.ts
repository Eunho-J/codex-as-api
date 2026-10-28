import { randomUUID } from "node:crypto";
import type { IncomingMessage, Server } from "node:http";
import type { Duplex } from "node:stream";
import WebSocket, { WebSocketServer } from "ws";

import {
  ChatGPTOAuthInvalidRequestError,
  ChatGPTOAuthUpstreamError,
  redactText,
  tokenForRequest,
  type ChatGPTTokenData,
} from "./auth.js";
import {
  CHATGPT_OAUTH_DEFAULT_BASE_URL,
  codexCliHeadersForVersion,
  defaultCodexVersionResolver,
  resolveLatestCodexVersion,
  type CodexVersionProvider,
} from "./provider.js";

/** Default model used for ChatGPT Live (realtime) voice calls. */
export const DEFAULT_REALTIME_MODEL = "gpt-live-1-codex";
/** Default voice for ChatGPT Live (realtime) voice calls. */
export const DEFAULT_REALTIME_VOICE = "cove";
/**
 * Default model and voice for standard Realtime calls (`mode: "realtime"`): the
 * function-calling voice model Codex defaults to on its v2 wire (openai/codex
 * #17183), with a voice from that wire's list.
 */
export const DEFAULT_REALTIME_V2_MODEL = "gpt-realtime-1.5";
export const DEFAULT_REALTIME_V2_VOICE = "marin";

/**
 * - `live`: a ChatGPT Live (GPT-Live, frameless bidi) call -- `?intent=quicksilver&
 *   architecture=avas` with `openai-alpha: quicksilver=v2`, `delegation` instead of tools.
 * - `realtime`: a standard Realtime session (`session.type: "realtime"`) with function
 *   tools, e.g. gpt-realtime-1.5. The ChatGPT backend takes it on the bare
 *   `/realtime/calls` route with no intent, architecture or alpha header (measured
 *   2026-10-27: adding any of them is refused with a 400).
 */
export type RealtimeCallMode = "live" | "realtime";
/**
 * Base URL used for the direct (sideband) realtime WebSocket. The call itself is
 * created on the ChatGPT backend, but the media/event sideband lives on the
 * OpenAI API host.
 */
export const REALTIME_SIDEBAND_BASE_URL = "https://api.openai.com/v1";

export const REALTIME_CALL_ROUTE = "/v1/realtime/calls";
export const REALTIME_SIDEBAND_ROUTE_PREFIX = "/v1/realtime/sideband/";

const REALTIME_CALL_QUERY = "intent=quicksilver&architecture=avas";
const MAX_SDP_BYTES = 256 * 1024;
const SIDEBAND_JOIN_ATTEMPTS = 4;
const SIDEBAND_JOIN_BACKOFF_MS = [100, 300, 800];

export interface RealtimeCallSessionPayload {
  instructions?: string;
  model?: string;
  voice?: string;
  audio?: Record<string, unknown>;
  delegation?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface CreateRealtimeCallOptions {
  sdp: string;
  session?: RealtimeCallSessionPayload;
  authPath?: string | null;
  baseUrl?: string;
  sessionId?: string;
  threadId?: string;
  /** Defaults to `realtime` when `session.type` is `"realtime"`, otherwise `live`. */
  mode?: RealtimeCallMode;
  /**
   * Client-generated voice session id, forwarded as `voice-session-id` (codex-rs
   * #53784, Codex 0.164.0). Distinct from the realtime session id (`x-session-id`).
   */
  voiceSessionId?: string;
  fetchImpl?: typeof fetch;
  versionResolver?: CodexVersionProvider;
}

export interface RealtimeCallResult {
  status: number;
  sdp: string;
  callId: string;
  location: string;
  sessionId: string;
  threadId: string;
  voiceSessionId?: string;
}

interface RealtimeAuthContext {
  token: ChatGPTTokenData;
  headers: Record<string, string>;
}

export function resolveRealtimeBaseUrl(raw?: string | null): string {
  const value = (raw ?? process.env.CODEX_AS_API_REALTIME_BASE_URL ?? "").trim();
  if (value.length === 0) return CHATGPT_OAUTH_DEFAULT_BASE_URL;
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new ChatGPTOAuthInvalidRequestError(
      "CODEX_AS_API_REALTIME_BASE_URL must be a valid HTTP(S) URL",
    );
  }
  if (
    (url.protocol !== "https:" && url.protocol !== "http:")
    || url.username.length > 0
    || url.password.length > 0
    || url.search.length > 0
    || url.hash.length > 0
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "CODEX_AS_API_REALTIME_BASE_URL must be an HTTP(S) URL without credentials, query, or fragment",
    );
  }
  return value.replace(/\/+$/, "");
}

export function resolveRealtimeSidebandUrl(callId: string, baseUrl = REALTIME_SIDEBAND_BASE_URL): string {
  const url = new URL(baseUrl);
  url.protocol = url.protocol === "http:" ? "ws:" : "wss:";
  const base = url.toString().replace(/\/+$/, "");
  return `${base}/live/${encodeURIComponent(callId)}`;
}

/** Header-safe client voice session id (same token rules as call ids). */
export function validateVoiceSessionId(value: string): string {
  if (typeof value !== "string" || !/^[A-Za-z0-9_.:-]{1,200}$/.test(value)) {
    throw new ChatGPTOAuthInvalidRequestError("invalid voice session id");
  }
  return value;
}

export function validateCallId(callId: string): string {
  if (typeof callId !== "string" || !/^[A-Za-z0-9_.:-]{1,200}$/.test(callId)) {
    throw new ChatGPTOAuthInvalidRequestError("invalid realtime call id");
  }
  return callId;
}

function requireSdp(sdp: unknown): string {
  if (typeof sdp !== "string") {
    throw new ChatGPTOAuthInvalidRequestError("`sdp` must be a string");
  }
  if (sdp.trim().length === 0) {
    throw new ChatGPTOAuthInvalidRequestError("`sdp` must not be blank");
  }
  if (Buffer.byteLength(sdp, "utf8") > MAX_SDP_BYTES) {
    throw new ChatGPTOAuthInvalidRequestError(
      `\`sdp\` must be at most ${MAX_SDP_BYTES} bytes`,
    );
  }
  if (!sdp.startsWith("v=0")) {
    throw new ChatGPTOAuthInvalidRequestError("`sdp` must be an SDP offer starting with v=0");
  }
  return sdp;
}

export function resolveRealtimeCallMode(
  session: RealtimeCallSessionPayload | undefined,
  mode?: RealtimeCallMode,
): RealtimeCallMode {
  if (mode != null) {
    if (mode !== "live" && mode !== "realtime") {
      throw new ChatGPTOAuthInvalidRequestError("`mode` must be \"live\" or \"realtime\"");
    }
    return mode;
  }
  return session != null && typeof session === "object" && !Array.isArray(session) && session.type === "realtime"
    ? "realtime"
    : "live";
}

export function buildRealtimeSessionPayload(
  session: RealtimeCallSessionPayload | undefined,
  mode: RealtimeCallMode = "live",
): Record<string, unknown> {
  if (session != null && (typeof session !== "object" || Array.isArray(session))) {
    throw new ChatGPTOAuthInvalidRequestError("`session` must be an object");
  }
  const incoming: Record<string, unknown> = { ...(session ?? {}) };
  // The ChatGPT backend mints these itself; client-chosen values make the backend
  // deny the voice session.
  delete incoming.id;
  delete incoming.voice_session_id;
  delete incoming.session_id;

  const audio =
    incoming.audio != null && typeof incoming.audio === "object" && !Array.isArray(incoming.audio)
      ? { ...(incoming.audio as Record<string, unknown>) }
      : {};
  const output =
    audio.output != null && typeof audio.output === "object" && !Array.isArray(audio.output)
      ? { ...(audio.output as Record<string, unknown>) }
      : {};
  const voice = typeof incoming.voice === "string" && incoming.voice.trim().length > 0
    ? incoming.voice
    : typeof output.voice === "string" && output.voice.trim().length > 0
      ? output.voice
      : mode === "realtime" ? DEFAULT_REALTIME_V2_VOICE : DEFAULT_REALTIME_VOICE;
  delete incoming.voice;

  const model = typeof incoming.model === "string" && incoming.model.trim().length > 0
    ? incoming.model
    : mode === "realtime" ? DEFAULT_REALTIME_V2_MODEL : DEFAULT_REALTIME_MODEL;
  if (mode === "realtime") {
    // Tools, not delegation: a standard Realtime session has no `delegation`.
    return { ...incoming, type: "realtime", audio: { ...audio, output: { ...output, voice } }, model };
  }
  const delegation =
    incoming.delegation != null
    && typeof incoming.delegation === "object"
    && !Array.isArray(incoming.delegation)
      ? incoming.delegation
      : { type: "client" };

  return {
    ...incoming,
    audio: { ...audio, output: { ...output, voice } },
    delegation,
    model,
  };
}

async function realtimeAuthHeaders(
  authPath?: string | null,
  versionResolver: CodexVersionProvider = defaultCodexVersionResolver,
  mode: RealtimeCallMode = "live",
): Promise<RealtimeAuthContext> {
  const token = await tokenForRequest(authPath);
  const version = await resolveLatestCodexVersion(versionResolver);
  const headers: Record<string, string> = {
    ...codexCliHeadersForVersion(version),
    version,
    Authorization: `Bearer ${token.access_token}`,
    "ChatGPT-Account-Id": token.account_id,
    "Content-Type": "application/json",
  };
  if (mode === "live") headers["openai-alpha"] = "quicksilver=v2";
  if (token.fedramp) headers["X-OpenAI-Fedramp"] = "true";
  return { token, headers };
}

function decodeCallId(location: string | null): string {
  if (location == null || location.trim().length === 0) {
    throw new ChatGPTOAuthUpstreamError(502, "upstream realtime call response missing Location");
  }
  const path = location.split("?")[0].split("#")[0];
  const segment = path.split("/").filter((part) => part.length > 0).at(-1) ?? "";
  return validateCallId(segment);
}

/**
 * Create a ChatGPT Live (realtime) call and return its SDP answer plus call id.
 *
 * This mirrors `codex-rs`'s `RealtimeCallClient::create_with_session_and_headers`,
 * which is the only call-creation path the ChatGPT backend authorizes for OAuth
 * (non-API-key) accounts: `x-session-id`/`session-id`/`thread-id` must all carry
 * the same id and the session payload must not pick its own voice session.
 */
export async function createRealtimeCall(
  options: CreateRealtimeCallOptions,
): Promise<RealtimeCallResult> {
  const sdp = requireSdp(options.sdp);
  const mode = resolveRealtimeCallMode(options.session, options.mode);
  const session = buildRealtimeSessionPayload(options.session, mode);
  const auth = await realtimeAuthHeaders(options.authPath, options.versionResolver, mode);
  const baseUrl = resolveRealtimeBaseUrl(options.baseUrl);
  const sessionId = options.sessionId ?? randomUUID();
  const threadId = options.threadId ?? sessionId;
  validateCallId(sessionId);
  validateCallId(threadId);
  const voiceSessionId = options.voiceSessionId == null ? undefined : validateVoiceSessionId(options.voiceSessionId);

  const url = mode === "live" ? `${baseUrl}/realtime/calls?${REALTIME_CALL_QUERY}` : `${baseUrl}/realtime/calls`;
  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(url, {
    method: "POST",
    headers: {
      ...auth.headers,
      "x-session-id": sessionId,
      "session-id": sessionId,
      "thread-id": threadId,
      ...(voiceSessionId == null ? {} : { "voice-session-id": voiceSessionId }),
    },
    body: JSON.stringify({ sdp, session }),
  });
  const text = await response.text();
  if (!response.ok) {
    const detail = redactText(text, auth.token.access_token);
    throw new ChatGPTOAuthUpstreamError(
      response.status,
      `upstream realtime call creation failed: HTTP ${response.status}: ${detail}`,
    );
  }
  const callId = decodeCallId(response.headers.get("location"));
  return {
    status: response.status,
    sdp: text,
    callId,
    location: `${REALTIME_CALL_ROUTE}/${callId}`,
    sessionId,
    threadId,
    ...(voiceSessionId == null ? {} : { voiceSessionId }),
  };
}

export interface SidebandRelayOptions {
  callId: string;
  authPath?: string | null;
  sidebandBaseUrl?: string;
  /** Forwarded as `voice-session-id` on the sideband join (existing-call transport). */
  voiceSessionId?: string;
  versionResolver?: CodexVersionProvider;
}

/**
 * Relay a client WebSocket to the OpenAI realtime sideband of an already-created
 * call. Headless clients (a Discord bot, a server-side agent) use this instead of
 * the WebRTC data channel.
 */
export async function relayRealtimeSideband(
  client: WebSocket,
  options: SidebandRelayOptions,
): Promise<void> {
  let callId: string;
  try {
    callId = validateCallId(options.callId);
  } catch {
    closeClient(client, 4400, "invalid realtime call id");
    return;
  }
  let auth: RealtimeAuthContext;
  try {
    auth = await realtimeAuthHeaders(options.authPath, options.versionResolver);
  } catch {
    closeClient(client, 4501, "realtime auth unavailable");
    return;
  }
  const target = resolveRealtimeSidebandUrl(callId, options.sidebandBaseUrl);
  const headers: Record<string, string> = { ...auth.headers };
  delete headers["Content-Type"];
  if (options.voiceSessionId != null) {
    try {
      headers["voice-session-id"] = validateVoiceSessionId(options.voiceSessionId);
    } catch {
      closeClient(client, 4400, "invalid voice session id");
      return;
    }
  }

  let upstream: WebSocket | null = null;
  let lastStatus: number | null = null;
  let lastCode: string | null = null;
  for (let attempt = 0; attempt < SIDEBAND_JOIN_ATTEMPTS; attempt++) {
    const joined = await openUpstream(target, headers);
    if (joined.socket != null) {
      upstream = joined.socket;
      break;
    }
    lastStatus = joined.status;
    lastCode = joined.code;
    if (joined.status !== 404) break;
    const delay =
      SIDEBAND_JOIN_BACKOFF_MS[Math.min(attempt, SIDEBAND_JOIN_BACKOFF_MS.length - 1)];
    await new Promise((resolve) => setTimeout(resolve, delay));
  }

  if (upstream == null) {
    // The close code stays 4403 / 4504; the reason carries what the upstream said, so a
    // caller's log shows the cause (e.g. "HTTP 404 session_id_not_found").
    closeClient(client, lastStatus === 403 ? 4403 : 4504, sidebandFailureReason(lastStatus, lastCode));
    return;
  }

  const socket: WebSocket = upstream;
  socket.on("message", (data, isBinary) => {
    if (client.readyState === WebSocket.OPEN) client.send(data, { binary: isBinary });
  });
  socket.on("close", (code, reason) => {
    closeClient(client, code === 1006 || code === 1005 ? 1011 : code, reason.toString());
  });
  socket.on("error", () => {
    closeClient(client, 1011, "realtime sideband error");
  });
  client.on("message", (data, isBinary) => {
    if (socket.readyState === WebSocket.OPEN) socket.send(data, { binary: isBinary });
  });
  client.on("close", () => {
    try {
      socket.close();
    } catch {
      /* already closed */
    }
  });
  client.on("error", () => {
    try {
      socket.terminate();
    } catch {
      /* already closed */
    }
  });
}

/** Close reason for a failed sideband join: the status and upstream error code when known. */
export function sidebandFailureReason(status: number | null, code: string | null): string {
  const detail = [status == null ? null : `HTTP ${status}`, code].filter((x): x is string => !!x).join(" ");
  return detail ? `realtime sideband unavailable (${detail})` : "realtime sideband unavailable";
}

// A WebSocket close reason is at most 123 bytes of UTF-8.
function clampReason(reason: string): string {
  let out = "";
  for (const ch of reason) {
    if (Buffer.byteLength(out + ch, "utf8") > 123) break;
    out += ch;
  }
  return out;
}

function closeClient(client: WebSocket, code: number, reason: string): void {
  try {
    if (client.readyState === WebSocket.OPEN || client.readyState === WebSocket.CONNECTING) {
      client.close(code, clampReason(reason));
    }
  } catch {
    /* already closed */
  }
}

const UPSTREAM_ERROR_BODY_MAX_BYTES = 1024;
const UPSTREAM_ERROR_BODY_WAIT_MS = 2000;
const ERROR_CODE_RE = /^[A-Za-z0-9_.:-]{1,64}$/;

/** The `error.code` (or `code`) of a refused join's JSON body, when it is a short token. */
export function upstreamErrorCode(body: string): string | null {
  try {
    const parsed = JSON.parse(body) as { error?: { code?: unknown } | string; code?: unknown };
    const code =
      parsed && typeof parsed.error === "object" && parsed.error ? parsed.error.code : parsed?.code;
    return typeof code === "string" && ERROR_CODE_RE.test(code) ? code : null;
  } catch {
    return null;
  }
}

type UpstreamJoin = { socket: WebSocket | null; status: number | null; code: string | null };

function openUpstream(target: string, headers: Record<string, string>): Promise<UpstreamJoin> {
  return new Promise((resolve) => {
    const socket = new WebSocket(target, { headers });
    let settled = false;
    const settle = (value: UpstreamJoin) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    socket.once("open", () => settle({ socket, status: 101, code: null }));
    socket.once("unexpected-response", (_request, response) => {
      // Read a little of the refusal body for its error code, then drop the connection.
      const status = response.statusCode ?? null;
      const chunks: Buffer[] = [];
      let size = 0;
      const finish = () => {
        clearTimeout(timer);
        const body = Buffer.concat(chunks).toString("utf8");
        settle({ socket: null, status, code: upstreamErrorCode(body) });
        try {
          socket.terminate();
        } catch {
          /* ignore */
        }
      };
      const timer = setTimeout(finish, UPSTREAM_ERROR_BODY_WAIT_MS);
      response.on("data", (chunk: Buffer) => {
        if (size >= UPSTREAM_ERROR_BODY_MAX_BYTES) return;
        const piece = chunk.subarray(0, UPSTREAM_ERROR_BODY_MAX_BYTES - size);
        chunks.push(piece);
        size += piece.length;
        if (size >= UPSTREAM_ERROR_BODY_MAX_BYTES) finish();
      });
      response.once("end", finish);
      response.once("error", finish);
    });
    socket.once("error", () => settle({ socket: null, status: null, code: null }));
    socket.once("close", () => settle({ socket: null, status: null, code: null }));
  });
}

export function attachRealtimeSideband(
  server: Server,
  options: {
    authPath?: string | null;
    sidebandBaseUrl?: string;
    versionResolver?: CodexVersionProvider;
  } = {},
): void {
  const wss = new WebSocketServer({ noServer: true });
  server.on("upgrade", (request: IncomingMessage, socket: Duplex, head: Buffer): void => {
    const path = (request.url ?? "").split("?")[0];
    if (!path.startsWith(REALTIME_SIDEBAND_ROUTE_PREFIX)) return;
    const callId = path.slice(REALTIME_SIDEBAND_ROUTE_PREFIX.length);
    wss.handleUpgrade(request, socket, head, (client) => {
      void relayRealtimeSideband(client, {
        callId,
        authPath: options.authPath,
        sidebandBaseUrl: options.sidebandBaseUrl,
        versionResolver: options.versionResolver,
      });
    });
  });
}
