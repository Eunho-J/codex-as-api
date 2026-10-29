import { test } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as path from "node:path";
import { createServer } from "node:http";
import type { Socket } from "node:net";
import { Duplex } from "node:stream";
import type express from "express";
import { createApp } from "../server.js";
import { DeviceLoginManager } from "../device-login.js";
import { commitLoginCredentials, loadTokenData } from "../auth.js";

const claims = { "https://api.openai.com/auth": { chatgpt_account_id: "fake-account" }, exp: 2000000000 };
const jwt = `e30.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.fake`;
const tokens = { access_token: jwt, id_token: jwt, refresh_token: "new-fake-refresh" };
const initial = { device_auth_id: "fake-device", user_code: "fake-code", interval: 2 };
const authorization = { authorization_code: "fake-code", code_verifier: "fake-verifier", code_challenge: "fake-challenge" };
const turn = () => new Promise<void>((resolve) => setImmediate(resolve));
async function until(predicate: () => boolean) {
  for (let i = 0; i < 200; i++) { if (predicate()) return; await turn(); }
  assert.fail("device flow did not reach expected state");
}

// Exercise the real HTTP parser and Express without opening a network socket.
async function request(app: express.Express, method: string, url: string, body = "") {
  const chunks: Buffer[] = [];
  const socket = new Duplex({ read() {}, write(chunk, _encoding, callback) { chunks.push(Buffer.from(chunk)); callback(); } });
  const server = createServer(app);
  server.emit("connection", socket as unknown as Socket);
  const finished = new Promise<void>((resolve, reject) => { socket.once("finish", resolve); socket.once("error", reject); });
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    socket.push(`${method} ${url} HTTP/1.1\r\nhost: local.test\r\nconnection: close\r\nauthorization: Bearer fake-admin\r\ncontent-type: application/json\r\ncontent-length: ${Buffer.byteLength(body)}\r\n\r\n${body}`);
    await Promise.race([finished, new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error("HTTP request timed out")), 2000); })]);
  } finally { if (timer) clearTimeout(timer); socket.destroy(); }
  const raw = Buffer.concat(chunks).toString("utf8");
  const split = raw.indexOf("\r\n\r\n");
  assert.notEqual(split, -1);
  const [statusLine, ...headerLines] = raw.slice(0, split).split("\r\n");
  const headers = new Map(headerLines.map((line) => {
    const colon = line.indexOf(":");
    assert.notEqual(colon, -1);
    return [line.slice(0, colon).toLowerCase(), line.slice(colon + 1).trim()] as const;
  }));
  let payload = raw.slice(split + 4);
  if (headers.get("transfer-encoding")?.toLowerCase() === "chunked") {
    const end = payload.indexOf("\r\n");
    payload = payload.slice(end + 2, end + 2 + parseInt(payload.slice(0, end), 16));
  }
  assert.equal(headers.get("cache-control"), "no-store");
  assert.ok(headers.get("content-type")?.toLowerCase().startsWith("application/json"));
  assert.equal(statusLine.split(" ")[0], "HTTP/1.1");
  return { status: Number(statusLine.split(" ")[1]), body: JSON.parse(payload) };
}

test("real createApp device login, cancellation races, and body errors", async () => {
  // Keep even HOME and any default Codex paths in a disposable directory under ts/.
  const dir = fs.mkdtempSync(path.join(process.cwd(), ".device-http-test-"));
  const previousHome = process.env.HOME;
  const previousAdmin = process.env.CODEX_AS_API_ADMIN_KEY;
  process.env.HOME = dir;
  process.env.CODEX_AS_API_ADMIN_KEY = "fake-admin";
  try {
    for (const scenario of ["success", "cancel-poll", "cancel-exchange"] as const) {
      const file = path.join(dir, scenario, "auth.json");
      let release!: (response: Response) => void;
      const blocked = new Promise<Response>((resolve) => { release = resolve; });
      const urls: string[] = [];
      let blockedSignal: AbortSignal | undefined;
      const manager = new DeviceLoginManager(file, { fetch: async (url, init) => {
        const endpoint = String(url); urls.push(endpoint);
        if (endpoint.endsWith("/usercode")) return Response.json(initial);
        if (endpoint.endsWith("/deviceauth/token")) {
          if (scenario === "cancel-poll") { blockedSignal = init?.signal as AbortSignal; return blocked; }
          return Response.json(authorization);
        }
        if (endpoint.endsWith("/oauth/token")) {
          blockedSignal = init?.signal as AbortSignal;
          return blocked;
        }
        throw new Error(`unexpected fake transport request: ${endpoint}`);
      } });
      const app = createApp({ provider: {} as never, codexConfig: { codexHome: dir, configPath: path.join(dir, "config.toml") }, authPath: file, loginManager: manager });
      try {
        if (scenario !== "success") commitLoginCredentials(file, { ...tokens, refresh_token: "old-fake-refresh" });
        const before = scenario === "success" ? undefined : fs.readFileSync(file);
        let result = await request(app, "GET", "/auth/status");
        assert.equal(result.status, 200);
        assert.deepEqual(result.body, { status: scenario === "success" ? "unauthenticated" : "authenticated" });
        if (scenario === "success") {
          for (const invalid of ["{", JSON.stringify({ padding: "x".repeat(5000) })]) {
            result = await request(app, "POST", "/auth/login", invalid);
            assert.equal(result.status, 400);
            assert.deepEqual(result.body, { error: { code: "invalid_request", message: "Invalid request body." } });
          }
          assert.equal(urls.length, 0);
        }
        result = await request(app, "POST", "/auth/login", "{}");
        assert.equal(result.status, 202);
        assert.equal(result.body.status, "pending");
        assert.equal(result.body.user_code, initial.user_code);
        const id: string = result.body.id;
        await until(() => blockedSignal !== undefined);
        result = await request(app, "GET", `/auth/login/${id}`);
        assert.equal(result.status, 200); assert.equal(result.body.status, "pending");
        assert.equal(result.body.user_code, undefined);
        if (scenario === "success") {
          assert.equal(fs.existsSync(file), false);
          release(Response.json(tokens));
          await until(() => manager.get(id)?.status === "succeeded");
          result = await request(app, "GET", `/auth/login/${id}`);
          assert.equal(result.body.status, "succeeded");
          assert.equal(loadTokenData(file).refresh_token, tokens.refresh_token);
          result = await request(app, "GET", "/auth/status");
          assert.deepEqual(result.body, { status: "authenticated" });
        } else {
          result = await request(app, "DELETE", `/auth/login/${id}`);
          assert.equal(result.status, 200); assert.equal(result.body.status, "cancelled");
          assert.equal(blockedSignal?.aborted, true);
          release(Response.json(scenario === "cancel-poll" ? authorization : tokens));
          for (let i = 0; i < 10; i++) await turn();
          result = await request(app, "GET", `/auth/login/${id}`);
          assert.equal(result.body.status, "cancelled");
          assert.deepEqual(fs.readFileSync(file), before);
          assert.equal(loadTokenData(file).refresh_token, "old-fake-refresh");
          assert.equal(urls.length, scenario === "cancel-poll" ? 2 : 3);
          result = await request(app, "GET", "/auth/status");
          assert.deepEqual(result.body, { status: "authenticated" });
        }
      } finally { app.locals.closeAuthManagement(); }
    }
  } finally {
    if (previousHome === undefined) delete process.env.HOME; else process.env.HOME = previousHome;
    if (previousAdmin === undefined) delete process.env.CODEX_AS_API_ADMIN_KEY; else process.env.CODEX_AS_API_ADMIN_KEY = previousAdmin;
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
