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
  resolveCodexCliVersion,
} from "./provider.js";

/** Default model used for ChatGPT Live (realtime) voice calls. */
export const DEFAULT_REALTIME_MODEL = "gpt-live-1-codex";
/** Default voice for ChatGPT Live (realtime) voice calls. */
export const DEFAULT_REALTIME_VOICE = "cove";
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
  fetchImpl?: typeof fetch;
}

export interface RealtimeCallResult {
  status: number;
  sdp: string;
  callId: string;
  location: string;
  sessionId: string;
  threadId: string;
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

export function buildRealtimeSessionPayload(
  session: RealtimeCallSessionPayload | undefined,
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
      : DEFAULT_REALTIME_VOICE;
  delete incoming.voice;

  const model = typeof incoming.model === "string" && incoming.model.trim().length > 0
    ? incoming.model
    : DEFAULT_REALTIME_MODEL;
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

async function realtimeAuthHeaders(authPath?: string | null): Promise<RealtimeAuthContext> {
  const token = await tokenForRequest(authPath);
  const headers: Record<string, string> = {
    ...codexCliHeadersForVersion(),
    version: resolveCodexCliVersion(),
    Authorization: `Bearer ${token.access_token}`,
    "ChatGPT-Account-Id": token.account_id,
    "Content-Type": "application/json",
    "openai-alpha": "quicksilver=v2",
  };
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
  const session = buildRealtimeSessionPayload(options.session);
  const auth = await realtimeAuthHeaders(options.authPath);
  const baseUrl = resolveRealtimeBaseUrl(options.baseUrl);
  const sessionId = options.sessionId ?? randomUUID();
  const threadId = options.threadId ?? sessionId;
  validateCallId(sessionId);
  validateCallId(threadId);

  const url = `${baseUrl}/realtime/calls?${REALTIME_CALL_QUERY}`;
  const fetchImpl = options.fetchImpl ?? fetch;
  const response = await fetchImpl(url, {
    method: "POST",
    headers: {
      ...auth.headers,
      "x-session-id": sessionId,
      "session-id": sessionId,
      "thread-id": threadId,
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
  };
}

export interface SidebandRelayOptions {
  callId: string;
  authPath?: string | null;
  sidebandBaseUrl?: string;
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
    auth = await realtimeAuthHeaders(options.authPath);
  } catch {
    closeClient(client, 4501, "realtime auth unavailable");
    return;
  }
  const target = resolveRealtimeSidebandUrl(callId, options.sidebandBaseUrl);
  const headers: Record<string, string> = { ...auth.headers };
  delete headers["Content-Type"];

  let upstream: WebSocket | null = null;
  let lastStatus: number | null = null;
  for (let attempt = 0; attempt < SIDEBAND_JOIN_ATTEMPTS; attempt++) {
    const joined = await openUpstream(target, headers);
    if (joined.socket != null) {
      upstream = joined.socket;
      break;
    }
    lastStatus = joined.status;
    if (joined.status !== 404) break;
    const delay =
      SIDEBAND_JOIN_BACKOFF_MS[Math.min(attempt, SIDEBAND_JOIN_BACKOFF_MS.length - 1)];
    await new Promise((resolve) => setTimeout(resolve, delay));
  }

  if (upstream == null) {
    closeClient(client, lastStatus === 403 ? 4403 : 4504, "realtime sideband unavailable");
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

function closeClient(client: WebSocket, code: number, reason: string): void {
  try {
    if (client.readyState === WebSocket.OPEN || client.readyState === WebSocket.CONNECTING) {
      client.close(code, reason.slice(0, 120));
    }
  } catch {
    /* already closed */
  }
}

function openUpstream(
  target: string,
  headers: Record<string, string>,
): Promise<{ socket: WebSocket | null; status: number | null }> {
  return new Promise((resolve) => {
    const socket = new WebSocket(target, { headers });
    let settled = false;
    const settle = (value: { socket: WebSocket | null; status: number | null }) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };
    socket.once("open", () => settle({ socket, status: 101 }));
    socket.once("unexpected-response", (_request, response) => {
      settle({ socket: null, status: response.statusCode ?? null });
      try {
        socket.terminate();
      } catch {
        /* ignore */
      }
    });
    socket.once("error", () => settle({ socket: null, status: null }));
    socket.once("close", () => settle({ socket: null, status: null }));
  });
}

export function attachRealtimeSideband(
  server: Server,
  options: { authPath?: string | null; sidebandBaseUrl?: string } = {},
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
      });
    });
  });
}
