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
/** Default wall-clock deadline in milliseconds for `createRealtimeCall` (request + body read). */
export const REALTIME_CALL_TIMEOUT_MS = 15_000;
/** Default wall-clock deadline in milliseconds for each WebSocket sideband join attempt. */
export const SIDEBAND_HANDSHAKE_TIMEOUT_MS = 10_000;

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
  /** Total wall-clock deadline in milliseconds covering request headers and SDP body read. */
  timeoutMs?: number;
  signal?: AbortSignal;
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

function resolveTimeoutMs(value: number | undefined, fallback: number, name: string): number {
  if (value === undefined) return fallback;
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new ChatGPTOAuthInvalidRequestError(`\`${name}\` must be a positive finite number`);
  }
  return value;
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
  const timeoutMs = resolveTimeoutMs(options.timeoutMs, REALTIME_CALL_TIMEOUT_MS, "timeoutMs");
  const mode = resolveRealtimeCallMode(options.session, options.mode);
  const session = buildRealtimeSessionPayload(options.session, mode);
  const sessionId = options.sessionId ?? randomUUID();
  const threadId = options.threadId ?? sessionId;
  validateCallId(sessionId);
  validateCallId(threadId);
  const voiceSessionId = options.voiceSessionId == null ? undefined : validateVoiceSessionId(options.voiceSessionId);
  const baseUrl = resolveRealtimeBaseUrl(options.baseUrl);
  const auth = await realtimeAuthHeaders(options.authPath, options.versionResolver, mode);

  const url = mode === "live" ? `${baseUrl}/realtime/calls?${REALTIME_CALL_QUERY}` : `${baseUrl}/realtime/calls`;
  const fetchImpl = options.fetchImpl ?? fetch;

  const controller = new AbortController();
  const onCallerAbort = () => controller.abort(options.signal?.reason);
  if (options.signal?.aborted) {
    onCallerAbort();
  } else {
    options.signal?.addEventListener("abort", onCallerAbort, { once: true });
  }

  let timedOut = false;
  let response: Response | undefined;
  let deadlineTimer: ReturnType<typeof setTimeout> | undefined;
  const deadlinePromise = new Promise<never>((_resolve, reject) => {
    deadlineTimer = setTimeout(() => {
      timedOut = true;
      const err = new ChatGPTOAuthUpstreamError(
        504,
        `upstream realtime call creation timed out after ${timeoutMs}ms`,
      );
      controller.abort(err);
      void response?.body?.cancel().catch(() => {});
      reject(err);
    }, timeoutMs);
    deadlineTimer.unref?.();
  });
  deadlinePromise.catch(() => {});

  let text = "";
  try {
    if (options.signal?.aborted) {
      throw new ChatGPTOAuthUpstreamError(499, "realtime call creation aborted");
    }
    try {
      response = await Promise.race([
        fetchImpl(url, {
          method: "POST",
          headers: {
            ...auth.headers,
            "x-session-id": sessionId,
            "session-id": sessionId,
            "thread-id": threadId,
            ...(voiceSessionId == null ? {} : { "voice-session-id": voiceSessionId }),
          },
          body: JSON.stringify({ sdp, session }),
          signal: controller.signal,
        }),
        deadlinePromise,
      ]);
      text = await Promise.race([response.text(), deadlinePromise]);
    } catch (err) {
      if (err instanceof ChatGPTOAuthUpstreamError || err instanceof ChatGPTOAuthInvalidRequestError) {
        throw err;
      }
      if (timedOut) {
        throw new ChatGPTOAuthUpstreamError(
          504,
          `upstream realtime call creation timed out after ${timeoutMs}ms`,
        );
      }
      const detail = redactText(String(err), auth.token.access_token);
      throw new ChatGPTOAuthUpstreamError(
        502,
        `upstream realtime call creation failed: ${detail}`,
      );
    }

    if (!response.ok) {
      const detail = redactText(text, auth.token.access_token);
      throw new ChatGPTOAuthUpstreamError(
        response.status,
        `upstream realtime call creation failed: HTTP ${response.status}: ${detail}`,
      );
    }
  } finally {
    clearTimeout(deadlineTimer);
    options.signal?.removeEventListener("abort", onCallerAbort);
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
  /** Wall-clock deadline in milliseconds for each WebSocket handshake attempt (default: 10000). */
  handshakeTimeoutMs?: number;
  versionResolver?: CodexVersionProvider;
}

function isClientActive(client: WebSocket): boolean {
  return client.readyState === WebSocket.OPEN || client.readyState === WebSocket.CONNECTING;
}

/**
 * Relay a client WebSocket to the OpenAI realtime sideband of an already-created
 * call. Headless clients (a Discord bot, a server-side agent) use this instead of
 * the WebRTC data channel.
 *
 * Resolves once the upstream sideband WebSocket is open and wired to `client`.
 * When validation, authentication, or the upstream WebSocket join fails (or times
 * out), closes `client` with the corresponding 4xxx close frame AND rejects with
 * the underlying error so callers awaiting `relayRealtimeSideband` never mark a
 * call connected without a live sideband.
 */
export function relayRealtimeSideband(
  client: WebSocket,
  options: SidebandRelayOptions,
): Promise<void> {
  const promise = relayRealtimeSidebandImpl(client, options);
  // Mark the returned promise handled so fire-and-forget callers (`void relayRealtimeSideband(...)`)
  // that rely solely on `client`'s `close` event do not trigger an unhandled rejection,
  // while callers that `await` or `.catch()` still observe the rejection.
  promise.catch(() => {});
  return promise;
}

async function relayRealtimeSidebandImpl(
  client: WebSocket,
  options: SidebandRelayOptions,
): Promise<void> {
  let handshakeTimeoutMs: number;
  let callId: string;
  let voiceSessionId: string | undefined;
  try {
    handshakeTimeoutMs = resolveTimeoutMs(
      options.handshakeTimeoutMs,
      SIDEBAND_HANDSHAKE_TIMEOUT_MS,
      "handshakeTimeoutMs",
    );
    callId = validateCallId(options.callId);
  } catch (err) {
    await Promise.resolve();
    closeClient(client, 4400, "invalid realtime call id");
    throw err;
  }
  if (options.voiceSessionId != null) {
    try {
      voiceSessionId = validateVoiceSessionId(options.voiceSessionId);
    } catch (err) {
      await Promise.resolve();
      closeClient(client, 4400, "invalid voice session id");
      throw err;
    }
  }
  let auth: RealtimeAuthContext;
  try {
    auth = await realtimeAuthHeaders(options.authPath, options.versionResolver);
  } catch (err) {
    closeClient(client, 4501, "realtime auth unavailable");
    throw err;
  }
  const target = resolveRealtimeSidebandUrl(callId, options.sidebandBaseUrl);
  const headers: Record<string, string> = { ...auth.headers };
  delete headers["Content-Type"];
  if (voiceSessionId != null) {
    headers["voice-session-id"] = voiceSessionId;
  }

  let upstream: WebSocket | null = null;
  let lastStatus: number | null = null;
  let lastCode: string | null = null;
  for (let attempt = 0; attempt < SIDEBAND_JOIN_ATTEMPTS; attempt++) {
    if (!isClientActive(client)) {
      throw new ChatGPTOAuthInvalidRequestError("realtime sideband client closed before join");
    }
    const joined = await openUpstream(target, headers, handshakeTimeoutMs, client);
    if (joined.socket != null) {
      if (!isClientActive(client)) {
        try {
          joined.socket.terminate();
        } catch {
          /* ignore */
        }
        throw new ChatGPTOAuthInvalidRequestError("realtime sideband client closed before join");
      }
      upstream = joined.socket;
      break;
    }
    lastStatus = joined.status;
    lastCode = joined.code;
    if (joined.code === "client_closed") {
      throw new ChatGPTOAuthInvalidRequestError("realtime sideband client closed before join");
    }
    if (joined.status !== 404) break;
    if (attempt + 1 < SIDEBAND_JOIN_ATTEMPTS) {
      const delay =
        SIDEBAND_JOIN_BACKOFF_MS[Math.min(attempt, SIDEBAND_JOIN_BACKOFF_MS.length - 1)];
      await waitBackoffOrClientClose(delay, client);
    }
  }

  if (upstream == null) {
    // The close code stays 4403 / 4504; the reason carries what the upstream said, so a
    // caller's log shows the cause (e.g. "HTTP 404 session_id_not_found" or "timeout").
    const reason = sidebandFailureReason(lastStatus, lastCode);
    closeClient(client, lastStatus === 403 ? 4403 : 4504, reason);
    const status = lastStatus ?? (lastCode === "timeout" ? 504 : 502);
    throw new ChatGPTOAuthUpstreamError(status, reason);
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

function waitBackoffOrClientClose(delayMs: number, client: WebSocket): Promise<void> {
  if (!isClientActive(client)) return Promise.resolve();
  return new Promise((resolve) => {
    const onClose = () => {
      clearTimeout(timer);
      resolve();
    };
    const timer = setTimeout(() => {
      client.removeListener?.("close", onClose);
      resolve();
    }, delayMs);
    timer.unref?.();
    client.once?.("close", onClose);
  });
}

/** Close reason for a failed sideband join: the status and upstream error code when known. */
export function sidebandFailureReason(status: number | null, code: string | null): string {
  const detail = [status == null ? null : `HTTP ${status}`, code].filter((x): x is string => !!x).join(" ");
  return detail ? `realtime sideband unavailable (${detail})` : "realtime sideband unavailable";
}

// A WebSocket close reason is at most 123 bytes of UTF-8.
export function clampReason(reason: string): string {
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

function openUpstream(
  target: string,
  headers: Record<string, string>,
  handshakeTimeoutMs = SIDEBAND_HANDSHAKE_TIMEOUT_MS,
  client?: WebSocket,
): Promise<UpstreamJoin> {
  if (client != null && !isClientActive(client)) {
    return Promise.resolve({ socket: null, status: null, code: "client_closed" });
  }
  return new Promise((resolve) => {
    const deadlineAt = Date.now() + handshakeTimeoutMs;
    const socket = new WebSocket(target, { headers, handshakeTimeout: handshakeTimeoutMs });
    let settled = false;
    let bodyTimer: ReturnType<typeof setTimeout> | undefined;
    let finishRefusal: (() => void) | null = null;

    const cleanup = () => {
      clearTimeout(handshakeTimer);
      clearTimeout(bodyTimer);
      if (client != null) {
        client.removeListener?.("close", onClientClose);
      }
    };

    const settle = (value: UpstreamJoin) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(value);
    };

    const onClientClose = () => {
      settle({ socket: null, status: null, code: "client_closed" });
      try {
        socket.terminate();
      } catch {
        /* ignore */
      }
    };
    client?.once?.("close", onClientClose);

    const handshakeTimer = setTimeout(() => {
      if (finishRefusal != null) {
        finishRefusal();
        return;
      }
      settle({ socket: null, status: null, code: "timeout" });
      try {
        socket.terminate();
      } catch {
        /* ignore */
      }
    }, handshakeTimeoutMs);
    handshakeTimer.unref?.();

    socket.once("open", () => {
      if (client != null && !isClientActive(client)) {
        settle({ socket: null, status: null, code: "client_closed" });
        try {
          socket.terminate();
        } catch {
          /* ignore */
        }
        return;
      }
      settle({ socket, status: 101, code: null });
    });
    socket.once("unexpected-response", (_request, response) => {
      // Read a little of the refusal body for its error code, then drop the connection.
      const status = response.statusCode ?? null;
      const chunks: Buffer[] = [];
      let size = 0;
      const finish = () => {
        finishRefusal = null;
        clearTimeout(bodyTimer);
        const body = Buffer.concat(chunks).toString("utf8");
        settle({ socket: null, status, code: upstreamErrorCode(body) });
        try {
          socket.terminate();
        } catch {
          /* ignore */
        }
      };
      finishRefusal = finish;
      const remainingMs = Math.max(1, deadlineAt - Date.now());
      bodyTimer = setTimeout(finish, Math.min(UPSTREAM_ERROR_BODY_WAIT_MS, remainingMs));
      bodyTimer.unref?.();
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
    socket.once("error", () => {
      if (finishRefusal != null) {
        finishRefusal();
        return;
      }
      settle({ socket: null, status: null, code: null });
    });
    socket.once("close", () => {
      if (finishRefusal != null) {
        finishRefusal();
        return;
      }
      settle({ socket: null, status: null, code: null });
    });
  });
}

export function attachRealtimeSideband(
  server: Server,
  options: {
    authPath?: string | null;
    sidebandBaseUrl?: string;
    handshakeTimeoutMs?: number;
    versionResolver?: CodexVersionProvider;
  } = {},
): void {
  const wss = new WebSocketServer({ noServer: true });
  server.on("upgrade", (request: IncomingMessage, socket: Duplex, head: Buffer): void => {
    const path = (request.url ?? "").split("?")[0];
    if (!path.startsWith(REALTIME_SIDEBAND_ROUTE_PREFIX)) return;
    const callId = path.slice(REALTIME_SIDEBAND_ROUTE_PREFIX.length);
    const voiceSessionHeader = request.headers["voice-session-id"];
    const voiceSessionId = Array.isArray(voiceSessionHeader)
      ? voiceSessionHeader.join(",")
      : voiceSessionHeader;
    wss.handleUpgrade(request, socket, head, (client) => {
      void relayRealtimeSideband(client, {
        callId,
        authPath: options.authPath,
        sidebandBaseUrl: options.sidebandBaseUrl,
        versionResolver: options.versionResolver,
        ...(options.handshakeTimeoutMs == null ? {} : { handshakeTimeoutMs: options.handshakeTimeoutMs }),
        ...(voiceSessionId == null ? {} : { voiceSessionId }),
      });
    });
  });
}
