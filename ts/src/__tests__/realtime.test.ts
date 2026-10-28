import { after, before, describe, it } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import { createServer, type IncomingHttpHeaders, type Server } from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import { createApp } from "../server.js";
import {
  ChatGPTOAuthInvalidRequestError,
  ChatGPTOAuthUpstreamError,
} from "../auth.js";
import {
  DEFAULT_REALTIME_MODEL,
  DEFAULT_REALTIME_V2_MODEL,
  DEFAULT_REALTIME_V2_VOICE,
  DEFAULT_REALTIME_VOICE,
  buildRealtimeSessionPayload,
  createRealtimeCall,
  resolveRealtimeBaseUrl,
  resolveRealtimeSidebandUrl,
  relayRealtimeSideband,
  sidebandFailureReason,
  upstreamErrorCode,
  validateCallId,
} from "../realtime.js";
import type { CodexConfig } from "../codex-config.js";
import WebSocket, { WebSocketServer } from "ws";

const TEST_CONFIG: CodexConfig = {
  codexHome: "/test/codex-home",
  configPath: "/test/codex-home/config.toml",
};

const OFFER = [
  "v=0",
  "o=- 1 2 IN IP4 127.0.0.1",
  "s=-",
  "t=0 0",
  "m=audio 9 UDP/TLS/RTP/SAVPF 111",
  "c=IN IP4 0.0.0.0",
  "a=mid:0",
  "a=rtpmap:111 opus/48000/2",
  "",
].join("\r\n");

function makeJwt(payload: Record<string, unknown>): string {
  const header = Buffer.from(JSON.stringify({ alg: "HS256", typ: "JWT" })).toString("base64url");
  const body = Buffer.from(JSON.stringify(payload)).toString("base64url");
  return `${header}.${body}.sig`;
}

function writeAuthFile(): { authPath: string; directory: string } {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-realtime-"));
  const authPath = path.join(directory, "auth.json");
  fs.writeFileSync(authPath, JSON.stringify({
    tokens: {
      access_token: makeJwt({ exp: 9_999_999_999 }),
      refresh_token: "refresh-token",
      id_token: makeJwt({
        exp: 9_999_999_999,
        "https://api.openai.com/auth": {
          chatgpt_account_id: "account-123",
          chatgpt_plan_type: "plus",
          chatgpt_user_id: "user-123",
        },
      }),
    },
  }));
  return { authPath, directory };
}

interface RecordedRealtimeRequest {
  method: string;
  url: string;
  headers: IncomingHttpHeaders;
  body: Record<string, unknown>;
}

interface RealtimeUpstream {
  baseUrl: string;
  requests: RecordedRealtimeRequest[];
  close: () => Promise<void>;
  setResponse: (status: number, body: string, location?: string) => void;
}

async function startRealtimeUpstream(): Promise<RealtimeUpstream> {
  const requests: RecordedRealtimeRequest[] = [];
  let status = 201;
  let responseBody = "v=0\r\no=- 9 9 IN IP4 0.0.0.0\r\n";
  let location = "/v1/realtime/calls/rtc_test123";

  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      requests.push({
        method: req.method ?? "",
        url: req.url ?? "",
        headers: req.headers,
        body: JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}") as Record<string, unknown>,
      });
      res.writeHead(status, {
        "content-type": "text/plain; charset=utf-8",
        location,
      });
      res.end(responseBody);
    });
  });

  const port = await new Promise<number>((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      assert.ok(address && typeof address === "object");
      resolve(address.port);
    });
  });

  return {
    baseUrl: `http://127.0.0.1:${port}/backend-api/codex`,
    requests,
    close: () => new Promise<void>((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    }),
    setResponse: (nextStatus, nextBody, nextLocation) => {
      status = nextStatus;
      responseBody = nextBody;
      if (nextLocation != null) location = nextLocation;
    },
  };
}

describe("realtime call creation", () => {
  it("mirrors the codex wire contract", async () => {
    const upstream = await startRealtimeUpstream();
    const { authPath, directory } = writeAuthFile();
    try {
      const result = await createRealtimeCall({
        sdp: OFFER,
        session: {
          instructions: "hi",
          voice: "marin",
          voice_session_id: "voice_should_be_dropped",
          id: "session_should_be_dropped",
        },
        authPath,
        baseUrl: upstream.baseUrl,
        sessionId: "session-fixed",
        versionResolver: () => "0.205.0",
      });

      assert.equal(result.callId, "rtc_test123");
      assert.equal(result.location, "/v1/realtime/calls/rtc_test123");
      assert.equal(result.sessionId, "session-fixed");
      assert.equal(result.threadId, "session-fixed");
      assert.equal(result.sdp, "v=0\r\no=- 9 9 IN IP4 0.0.0.0\r\n");

      assert.equal(upstream.requests.length, 1);
      const request = upstream.requests[0];
      assert.equal(request.method, "POST");
      assert.equal(request.url, "/backend-api/codex/realtime/calls?intent=quicksilver&architecture=avas");
      assert.equal(request.headers["openai-alpha"], "quicksilver=v2");
      assert.equal(request.headers["x-session-id"], "session-fixed");
      assert.equal(request.headers["session-id"], "session-fixed");
      assert.equal(request.headers["thread-id"], "session-fixed");
      assert.equal(request.headers["chatgpt-account-id"], "account-123");
      assert.equal(String(request.headers.authorization).startsWith("Bearer "), true);
      assert.equal(request.headers["content-type"], "application/json");
      assert.equal(request.headers.version, "0.205.0");
      assert.equal(String(request.headers["user-agent"]).startsWith("codex_cli_rs/0.205.0 ("), true);

      const session = request.body.session as Record<string, unknown>;
      assert.equal(session.model, DEFAULT_REALTIME_MODEL);
      assert.deepEqual(session.audio, { output: { voice: "marin" } });
      assert.deepEqual(session.delegation, { type: "client" });
      assert.equal(session.instructions, "hi");
      assert.equal(Object.hasOwn(session, "id"), false);
      assert.equal(Object.hasOwn(session, "voice_session_id"), false);
      assert.equal(request.body.sdp, OFFER);
    } finally {
      await upstream.close();
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });

  it("forwards a client voice session id as voice-session-id only when given", async () => {
    const upstream = await startRealtimeUpstream();
    const { authPath, directory } = writeAuthFile();
    try {
      const withId = await createRealtimeCall({
        sdp: OFFER,
        authPath,
        baseUrl: upstream.baseUrl,
        sessionId: "session-fixed",
        voiceSessionId: "voice_789",
        versionResolver: () => "0.164.0",
      });
      const withoutId = await createRealtimeCall({
        sdp: OFFER,
        authPath,
        baseUrl: upstream.baseUrl,
        versionResolver: () => "0.164.0",
      });
      assert.equal(withId.voiceSessionId, "voice_789");
      assert.equal(withoutId.voiceSessionId, undefined);
      assert.equal(upstream.requests[0].headers["voice-session-id"], "voice_789");
      assert.equal(upstream.requests[0].headers["x-session-id"], "session-fixed");
      assert.equal(Object.hasOwn(upstream.requests[1].headers, "voice-session-id"), false);
      await assert.rejects(
        createRealtimeCall({
          sdp: OFFER,
          authPath,
          baseUrl: upstream.baseUrl,
          voiceSessionId: "bad\r\nInjected: 1",
          versionResolver: () => "0.164.0",
        }),
        ChatGPTOAuthInvalidRequestError,
      );
      assert.equal(upstream.requests.length, 2, "an invalid id never reaches upstream");
    } finally {
      await upstream.close();
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });

  it("creates a standard Realtime call on the bare route with tools and no alpha header", async () => {
    const upstream = await startRealtimeUpstream();
    const { authPath, directory } = writeAuthFile();
    const tools = [{ type: "function", name: "end_call", description: "hang up", parameters: { type: "object", properties: {} } }];
    try {
      await createRealtimeCall({
        sdp: OFFER,
        session: { type: "realtime", instructions: "hi", tools, tool_choice: "auto", delegation: { type: "client" } as never },
        authPath,
        baseUrl: upstream.baseUrl,
        sessionId: "session-fixed",
        versionResolver: () => "0.205.0",
      });
      await createRealtimeCall({
        sdp: OFFER,
        session: { instructions: "explicit", voice: "cedar" },
        mode: "realtime",
        authPath,
        baseUrl: upstream.baseUrl,
        versionResolver: () => "0.205.0",
      });
      const [inferred, explicit] = upstream.requests;
      assert.equal(inferred.url, "/backend-api/codex/realtime/calls");
      assert.equal(Object.hasOwn(inferred.headers, "openai-alpha"), false);
      assert.equal(inferred.headers["x-session-id"], "session-fixed");
      const session = inferred.body.session as Record<string, unknown>;
      assert.equal(session.type, "realtime");
      assert.equal(session.model, DEFAULT_REALTIME_V2_MODEL);
      assert.deepEqual(session.audio, { output: { voice: DEFAULT_REALTIME_V2_VOICE } });
      assert.deepEqual(session.tools, tools);
      assert.equal(session.tool_choice, "auto");
      assert.deepEqual(session.delegation, { type: "client" }, "a caller-given field passes through untouched");
      assert.equal(explicit.url, "/backend-api/codex/realtime/calls");
      const explicitSession = explicit.body.session as Record<string, unknown>;
      assert.equal(explicitSession.type, "realtime");
      assert.equal(Object.hasOwn(explicitSession, "delegation"), false, "no delegation is injected in realtime mode");
      assert.deepEqual(explicitSession.audio, { output: { voice: "cedar" } });
      await assert.rejects(
        createRealtimeCall({ sdp: OFFER, mode: "bogus" as never, authPath, baseUrl: upstream.baseUrl, versionResolver: () => "0.205.0" }),
        ChatGPTOAuthInvalidRequestError,
      );
      assert.equal(upstream.requests.length, 2, "an unknown mode never reaches upstream");
    } finally {
      await upstream.close();
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });

  it("accepts a call id from a full backend Location path", async () => {
    const upstream = await startRealtimeUpstream();
    const { authPath, directory } = writeAuthFile();
    upstream.setResponse(201, "v=0\r\n", "/backend-api/codex/v1/realtime/calls/rtc_abc_123?intent=quicksilver");
    try {
      const result = await createRealtimeCall({
        sdp: OFFER,
        authPath,
        baseUrl: upstream.baseUrl,
        versionResolver: () => "0.164.0",
      });
      assert.equal(result.callId, "rtc_abc_123");
    } finally {
      await upstream.close();
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });

  it("rejects malformed SDP before calling upstream", async () => {
    const { authPath, directory } = writeAuthFile();
    try {
      await assert.rejects(
        createRealtimeCall({ sdp: "not-sdp", authPath, versionResolver: () => "0.164.0" }),
        ChatGPTOAuthInvalidRequestError,
      );
      await assert.rejects(
        createRealtimeCall({ sdp: "", authPath, versionResolver: () => "0.164.0" }),
        ChatGPTOAuthInvalidRequestError,
      );
      await assert.rejects(
        createRealtimeCall({ sdp: "v=0\r\n", authPath, session: [] as never, versionResolver: () => "0.164.0" }),
        ChatGPTOAuthInvalidRequestError,
      );
    } finally {
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });

  it("maps upstream failures to an upstream error and redacts the token", async () => {
    const upstream = await startRealtimeUpstream();
    const { authPath, directory } = writeAuthFile();
    upstream.setResponse(403, "Voice session access denied.");
    try {
      await assert.rejects(
        createRealtimeCall({
          sdp: OFFER,
          authPath,
          baseUrl: upstream.baseUrl,
          versionResolver: () => "0.164.0",
        }),
        (err: unknown) => {
          assert.ok(err instanceof ChatGPTOAuthUpstreamError);
          assert.equal(err.status, 403);
          assert.equal(err.message.includes("Voice session access denied."), true);
          return true;
        },
      );
    } finally {
      await upstream.close();
      fs.rmSync(directory, { recursive: true, force: true });
    }
  });
});

describe("realtime helpers", () => {
  it("validates call ids", () => {
    assert.equal(validateCallId("rtc_abc-123"), "rtc_abc-123");
    assert.throws(() => validateCallId(""), ChatGPTOAuthInvalidRequestError);
    assert.throws(() => validateCallId("../etc"), ChatGPTOAuthInvalidRequestError);
    assert.throws(() => validateCallId("a b"), ChatGPTOAuthInvalidRequestError);
  });

  it("builds the sideband websocket url from the api host", () => {
    assert.equal(
      resolveRealtimeSidebandUrl("rtc_1", "https://api.openai.com/v1"),
      "wss://api.openai.com/v1/live/rtc_1",
    );
    assert.equal(
      resolveRealtimeSidebandUrl("rtc_1", "http://127.0.0.1:8899/v1/"),
      "ws://127.0.0.1:8899/v1/live/rtc_1",
    );
  });

  it("resolves the realtime base url from the environment", () => {
    const previous = process.env.CODEX_AS_API_REALTIME_BASE_URL;
    try {
      process.env.CODEX_AS_API_REALTIME_BASE_URL = "http://127.0.0.1:9999/backend-api/codex/";
      assert.equal(resolveRealtimeBaseUrl(), "http://127.0.0.1:9999/backend-api/codex");
      delete process.env.CODEX_AS_API_REALTIME_BASE_URL;
      assert.equal(resolveRealtimeBaseUrl(), "https://chatgpt.com/backend-api/codex");
      process.env.CODEX_AS_API_REALTIME_BASE_URL = "ftp://example.com";
      assert.throws(() => resolveRealtimeBaseUrl(), ChatGPTOAuthInvalidRequestError);
    } finally {
      if (previous === undefined) delete process.env.CODEX_AS_API_REALTIME_BASE_URL;
      else process.env.CODEX_AS_API_REALTIME_BASE_URL = previous;
    }
  });

  it("normalizes the session payload", () => {
    const payload = buildRealtimeSessionPayload({ voice: "marin" });
    assert.deepEqual(payload.audio, { output: { voice: "marin" } });
    assert.deepEqual(payload.delegation, { type: "client" });
    assert.equal(payload.model, DEFAULT_REALTIME_MODEL);
    assert.equal(buildRealtimeSessionPayload(undefined).audio instanceof Object, true);
    assert.deepEqual(
      (buildRealtimeSessionPayload({}).audio as Record<string, unknown>).output,
      { voice: DEFAULT_REALTIME_VOICE },
    );
  });
});

describe("POST /v1/realtime/calls", () => {
  let upstream: RealtimeUpstream;
  let server: Server;
  let directory: string;
  let authPath: string;
  let baseUrl: string;
  const previousBaseUrl = process.env.CODEX_AS_API_REALTIME_BASE_URL;

  before(async () => {
    upstream = await startRealtimeUpstream();
    ({ authPath, directory } = writeAuthFile());
    process.env.CODEX_AS_API_REALTIME_BASE_URL = upstream.baseUrl;
    const app = createApp({
      provider: {
        catalogSnapshot: async () => {
          throw new ChatGPTOAuthUpstreamError(503, "catalog unavailable in route tests");
        },
      } as never,
      model: "gpt-5.5",
      codexConfig: TEST_CONFIG,
      authPath,
      versionResolver: () => "0.205.0",
    });
    server = await new Promise<Server>((resolve) => {
      const listening = app.listen(0, "127.0.0.1", () => resolve(listening));
    });
    const address = server.address();
    assert.ok(address && typeof address === "object");
    baseUrl = `http://127.0.0.1:${address.port}`;
  });

  after(async () => {
    if (previousBaseUrl === undefined) delete process.env.CODEX_AS_API_REALTIME_BASE_URL;
    else process.env.CODEX_AS_API_REALTIME_BASE_URL = previousBaseUrl;
    await new Promise<void>((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    });
    await upstream.close();
    fs.rmSync(directory, { recursive: true, force: true });
  });

  it("returns the SDP answer with a rewritten Location", async () => {
    const response = await fetch(`${baseUrl}/v1/realtime/calls`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: OFFER, session: { voice: "cove", instructions: "hi" } }),
    });
    assert.equal(response.status, 201);
    assert.equal(response.headers.get("location"), "/v1/realtime/calls/rtc_test123");
    assert.equal(response.headers.get("content-type"), "text/plain; charset=utf-8");
    assert.equal(response.headers.get("x-realtime-call-id"), "rtc_test123");
    const sessionId = response.headers.get("x-session-id");
    assert.equal(response.headers.get("session-id"), sessionId);
    assert.equal(response.headers.get("thread-id"), sessionId);
    assert.equal((await response.text()).startsWith("v=0"), true);
  });

  it("forwards a voice-session-id request header and echoes it", async () => {
    const response = await fetch(`${baseUrl}/v1/realtime/calls`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "voice-session-id": "voice_server_1" },
      body: JSON.stringify({ sdp: OFFER }),
    });
    assert.equal(response.status, 201);
    assert.equal(response.headers.get("voice-session-id"), "voice_server_1");
    assert.equal(upstream.requests.at(-1)?.headers["voice-session-id"], "voice_server_1");
    await response.text();
  });

  it("rejects unknown request fields", async () => {
    const response = await fetch(`${baseUrl}/v1/realtime/calls`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sdp: OFFER, model: "gpt-live-1-codex" }),
    });
    assert.equal(response.status, 400);
    const body = await response.json() as { error: { message: string } };
    assert.equal(body.error.message.includes('does not support field "model"'), true);
  });

  it("surfaces upstream failures as JSON errors", async () => {
    upstream.setResponse(403, "Voice session access denied.");
    try {
      const response = await fetch(`${baseUrl}/v1/realtime/calls`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdp: OFFER }),
      });
      assert.equal(response.status, 403);
      const body = await response.json() as { error: { message: string } };
      assert.equal(body.error.message, "upstream request failed");
    } finally {
      upstream.setResponse(201, "v=0\r\n");
    }
  });
});

describe("realtime sideband relay voice session id", () => {
  async function joinHeaders(voiceSessionId?: string): Promise<{ headers: Record<string, string | string[] | undefined> | null; closeCode: number | null }> {
    const { authPath, directory } = writeAuthFile();
    const upstreamHttp = createServer();
    const upstreamWss = new WebSocketServer({ server: upstreamHttp });
    let seen: Record<string, string | string[] | undefined> | null = null;
    upstreamWss.on("connection", (socket, request) => {
      seen = { ...request.headers };
      socket.close(1000, "done");
    });
    await new Promise<void>((resolve) => upstreamHttp.listen(0, "127.0.0.1", resolve));
    const upstreamPort = (upstreamHttp.address() as { port: number }).port;
    const clientHttp = createServer();
    const clientWss = new WebSocketServer({ server: clientHttp });
    await new Promise<void>((resolve) => clientHttp.listen(0, "127.0.0.1", resolve));
    const clientPort = (clientHttp.address() as { port: number }).port;
    let closeCode: number | null = null;
    const relayed = new Promise<void>((resolve) => {
      clientWss.on("connection", (serverSide) => {
        void relayRealtimeSideband(serverSide, {
          callId: "rtc_1",
          authPath,
          sidebandBaseUrl: `http://127.0.0.1:${upstreamPort}/v1`,
          versionResolver: () => "0.164.0",
          ...(voiceSessionId === undefined ? {} : { voiceSessionId }),
        }).finally(resolve);
      });
    });
    const client = new WebSocket(`ws://127.0.0.1:${clientPort}`);
    const closed = new Promise<void>((resolve) => client.on("close", (code) => { closeCode = code; resolve(); }));
    await Promise.race([Promise.all([relayed, closed]), new Promise((r) => setTimeout(r, 3000))]);
    client.terminate();
    await new Promise<void>((resolve) => clientWss.close(() => clientHttp.close(() => resolve())));
    await new Promise<void>((resolve) => upstreamWss.close(() => upstreamHttp.close(() => resolve())));
    fs.rmSync(directory, { recursive: true, force: true });
    return { headers: seen, closeCode };
  }

  it("sends voice-session-id on the sideband join only when given", async () => {
    const withId = await joinHeaders("voice_789");
    assert.equal(withId.headers?.["voice-session-id"], "voice_789");
    const withoutId = await joinHeaders();
    assert.notEqual(withoutId.headers, null);
    assert.equal(Object.hasOwn(withoutId.headers ?? {}, "voice-session-id"), false);
  });

  it("closes the client and never joins upstream for an invalid id", async () => {
    const bad = await joinHeaders("bad id");
    assert.equal(bad.headers, null);
    assert.equal(bad.closeCode, 4400);
  });
});

describe("realtime sideband join failure reason", () => {
  // A fake upstream that refuses the WebSocket upgrade with `status` and `body`.
  async function refusedJoin(status: number, body: string): Promise<{ code: number | null; reason: string; attempts: number }> {
    const { authPath, directory } = writeAuthFile();
    let attempts = 0;
    const upstreamHttp = createServer();
    upstreamHttp.on("upgrade", (_req, socket) => {
      attempts++;
      const payload = Buffer.from(body, "utf8");
      socket.end(`HTTP/1.1 ${status} Refused\r\nContent-Type: application/json\r\nContent-Length: ${payload.length}\r\nConnection: close\r\n\r\n${body}`);
    });
    await new Promise<void>((resolve) => upstreamHttp.listen(0, "127.0.0.1", resolve));
    const upstreamPort = (upstreamHttp.address() as { port: number }).port;
    const clientHttp = createServer();
    const clientWss = new WebSocketServer({ server: clientHttp });
    await new Promise<void>((resolve) => clientHttp.listen(0, "127.0.0.1", resolve));
    const clientPort = (clientHttp.address() as { port: number }).port;
    clientWss.on("connection", (serverSide) => {
      void relayRealtimeSideband(serverSide, {
        callId: "rtc_1",
        authPath,
        sidebandBaseUrl: `http://127.0.0.1:${upstreamPort}/v1`,
        versionResolver: () => "0.164.0",
      });
    });
    const client = new WebSocket(`ws://127.0.0.1:${clientPort}`);
    const result = await new Promise<{ code: number | null; reason: string }>((resolve) => {
      client.on("close", (code, reason) => resolve({ code, reason: reason.toString() }));
      setTimeout(() => resolve({ code: null, reason: "" }), 8000);
    });
    client.terminate();
    await new Promise<void>((resolve) => clientWss.close(() => clientHttp.close(() => resolve())));
    await new Promise<void>((resolve) => upstreamHttp.close(() => resolve()));
    fs.rmSync(directory, { recursive: true, force: true });
    return { ...result, attempts };
  }

  it("404 session_id_not_found: retried, then 4504 with the status and the upstream code in the reason", async () => {
    const r = await refusedJoin(404, JSON.stringify({ error: { code: "session_id_not_found", message: "no such session" } }));
    assert.equal(r.code, 4504);
    assert.equal(r.reason, "realtime sideband unavailable (HTTP 404 session_id_not_found)");
    assert.equal(r.attempts, 4);
  });

  it("403: 4403 as before, reason names the status", async () => {
    const r = await refusedJoin(403, "forbidden");
    assert.equal(r.code, 4403);
    assert.equal(r.reason, "realtime sideband unavailable (HTTP 403)");
    assert.equal(r.attempts, 1);
  });

  it("reason helpers: no detail, code only when it is a short token, clamped to a close frame", () => {
    assert.equal(sidebandFailureReason(null, null), "realtime sideband unavailable");
    assert.equal(sidebandFailureReason(500, null), "realtime sideband unavailable (HTTP 500)");
    assert.equal(upstreamErrorCode('{"error":{"code":"session_id_not_found"}}'), "session_id_not_found");
    assert.equal(upstreamErrorCode('{"code":"rate_limited"}'), "rate_limited");
    assert.equal(upstreamErrorCode('{"error":{"code":"has spaces and is long"}}'), null);
    assert.equal(upstreamErrorCode("<html>"), null);
  });
});
