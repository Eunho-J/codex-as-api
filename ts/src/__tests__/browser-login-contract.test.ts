import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import * as http from "node:http";
import * as net from "node:net";
import * as os from "node:os";
import * as path from "node:path";
import { loadTokenData } from "../auth.js";
import { BrowserLoginManager } from "../browser-login.js";

const turn = () => new Promise<void>((resolve) => setImmediate(resolve));
async function until(predicate: () => boolean) {
  for (let i = 0; i < 100; i++) { if (predicate()) return; await turn(); }
  assert.fail("login did not reach expected state");
}
const payload = Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "fake-account" } })).toString("base64url");
const fakeJwt = `e30.${payload}.signature`;
const tokens = { access_token: fakeJwt, id_token: fakeJwt, refresh_token: "fake-refresh" };
const json = (value: unknown) => new Response(JSON.stringify(value), { headers: { "content-type": "application/json" } });
function temporaryStore() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "browser-login-test-"));
  return { file: path.join(dir, "auth.json"), dispose: () => fs.rmSync(dir, { recursive: true, force: true }) };
}
function callback(redirect: string, query: string, options: { method?: string; host?: string; path?: string } = {}) {
  const target = new URL(redirect);
  return new Promise<{ status: number; headers: http.IncomingHttpHeaders; body: string }>((resolve, reject) => {
    const req = http.request({ hostname: "127.0.0.1", port: Number(target.port), path: `${options.path ?? target.pathname}${query}`, method: options.method ?? "GET", headers: options.host ? { host: options.host } : undefined }, (res) => {
      let body = ""; res.setEncoding("utf8"); res.on("data", (chunk) => { body += chunk; });
      res.on("end", () => resolve({ status: res.statusCode!, headers: res.headers, body }));
    });
    req.on("error", reject); req.end();
  });
}
function authInfo(authUrl: string) {
  const url = new URL(authUrl);
  const redirect = url.searchParams.get("redirect_uri");
  const state = url.searchParams.get("state");
  assert.ok(redirect); assert.ok(state);
  return { url, redirect, state };
}
function fakeClock() {
  let now = 0;
  const timers = new Map<object, { callback: () => void; at: number }>();
  return {
    clock: {
      now: () => now, wallNow: () => 0,
      setTimeout(callback: () => void, ms: number) { const key = {}; timers.set(key, { callback, at: now + ms }); return key; },
      clearTimeout(key: unknown) { timers.delete(key as object); },
    },
    advance(ms: number) { now += ms; for (const [key, timer] of [...timers]) if (timer.at <= now) { timers.delete(key); timer.callback(); } },
    elapseWithoutTimers(ms: number) { now += ms; },
  };
}
async function occupy() {
  const server = net.createServer();
  await new Promise<void>((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  const address = server.address(); assert.ok(address && typeof address !== "string");
  return { port: address.port, close: () => new Promise<void>((resolve, reject) => server.close((err) => err ? reject(err) : resolve())) };
}

describe("browser OAuth contract using loopback and fake issuer", () => {
  it("generates official authorization and S256 PKCE, then durably writes before success", async () => {
    const store = temporaryStore();
    const calls: Array<{ url: string; init: RequestInit }> = [];
    const manager = new BrowserLoginManager(store.file, { ports: [0], fetch: async (input, init) => { calls.push({ url: String(input), init: init! }); return json(tokens); } });
    try {
      const started = await manager.start();
      assert.equal(started.method, "browser"); assert.equal(started.status, "pending");
      const { url, redirect, state } = authInfo(started.auth_url);
      assert.equal(url.origin + url.pathname, "https://auth.openai.com/oauth/authorize");
      assert.equal(url.searchParams.get("response_type"), "code");
      assert.equal(url.searchParams.get("client_id"), "app_EMoamEEZ73f0CkXaXp7hrann");
      assert.equal(url.searchParams.get("scope"), "openid profile email offline_access api.connectors.read api.connectors.invoke");
      assert.equal(url.searchParams.get("code_challenge_method"), "S256");
      assert.equal(url.searchParams.get("id_token_add_organizations"), "true");
      assert.equal(url.searchParams.get("codex_cli_simplified_flow"), "true");
      assert.equal(url.searchParams.get("originator"), "codex_cli_rs");
      const redirectUrl = new URL(redirect);
      assert.equal(redirectUrl.protocol, "http:"); assert.equal(redirectUrl.hostname, "127.0.0.1");
      assert.ok(Number(redirectUrl.port) > 0); assert.equal(redirectUrl.pathname, "/auth/callback");
      const response = await callback(redirect, `?state=${encodeURIComponent(state)}&code=fake-code`);
      assert.equal(response.status, 200);
      assert.equal(response.headers["cache-control"], "no-store");
      assert.equal(response.headers["referrer-policy"], "no-referrer");
      assert.equal(manager.get(started.id)?.status, "succeeded");
      assert.equal(loadTokenData(store.file).refresh_token, "fake-refresh");
      assert.equal(loadTokenData(store.file).account_id, "fake-account");
      if (process.platform !== "win32") assert.equal(fs.statSync(store.file).mode & 0o777, 0o600);
      assert.equal(calls.length, 1); assert.equal(calls[0]!.url, "https://auth.openai.com/oauth/token");
      assert.equal(calls[0]!.init.redirect, "manual");
      const form = new URLSearchParams(String(calls[0]!.init.body));
      assert.equal(form.get("grant_type"), "authorization_code"); assert.equal(form.get("client_id"), "app_EMoamEEZ73f0CkXaXp7hrann");
      assert.equal(form.get("redirect_uri"), redirect); assert.equal(form.get("code"), "fake-code");
      const verifier = form.get("code_verifier"); assert.ok(verifier);
      assert.ok(verifier.length >= 43 && verifier.length <= 128);
      assert.ok([...verifier].every((character) => "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~-".includes(character)));
      assert.equal(crypto.createHash("sha256").update(verifier).digest("base64url"), url.searchParams.get("code_challenge"));
      for (const secret of ["fake-code", "fake-refresh", "fake-account", "code_verifier", "auth_url"]) {
        assert.equal(JSON.stringify(manager.get(started.id)).includes(secret), false);
      }
      assert.equal(manager.cancel(started.id)?.status, "succeeded");
    } finally { manager.close(); store.dispose(); }
  });

  it("rejects malformed or hostile callbacks without consuming the valid session", async () => {
    const store = temporaryStore(); let calls = 0;
    const manager = new BrowserLoginManager(store.file, { ports: [0], fetch: async () => { calls++; return json(tokens); } });
    try {
      const started = await manager.start(); const { redirect, state } = authInfo(started.auth_url);
      const s = encodeURIComponent(state);
      const attacks = [
        { query: "?code=fake-code" }, { query: "?state=&code=fake-code" },
        { query: "?state=wrong-secret&code=fake-code" }, { query: `?state=${s}&state=${s}&code=fake-code` },
        { query: `?state=${s}&code=a&code=b` }, { query: `?state=${s}&error=a&error=b` },
        { query: `?state=${s}&code=a&error=access_denied` },
        { query: `?state=${s}&code=fake-code`, method: "POST" },
        { query: `?state=${s}&code=fake-code`, host: "evil.example" },
        { query: `?state=${s}&code=fake-code`, path: "/unrelated" },
      ];
      for (const attack of attacks) {
        const { query, ...options } = attack; const response = await callback(redirect, query, options);
        assert.notEqual(response.status, 200, JSON.stringify(attack));
        assert.equal(response.headers["cache-control"], "no-store");
        assert.equal(response.headers["referrer-policy"], "no-referrer");
        for (const secret of ["wrong-secret", "fake-code", "access_denied", "evil.example"]) assert.equal(response.body.includes(secret), false);
        assert.equal(manager.get(started.id)?.status, "pending"); assert.equal(calls, 0);
      }
      assert.equal((await callback(redirect, `?state=${s}&code=fake-code`)).status, 200);
      assert.equal(calls, 1); assert.equal(manager.get(started.id)?.status, "succeeded");
    } finally { manager.close(); store.dispose(); }
  });

  it("fails safely on a valid-state denial without exposing provider text", async () => {
    const store = temporaryStore(); let calls = 0;
    const manager = new BrowserLoginManager(store.file, { ports: [0], fetch: async () => { calls++; throw Error("unexpected exchange"); } });
    try {
      const started = await manager.start(); const { redirect, state } = authInfo(started.auth_url);
      const response = await callback(redirect, `?state=${encodeURIComponent(state)}&error=access_denied&error_description=private-denial-text`);
      assert.notEqual(response.status, 200);
      for (const secret of ["private-denial-text", "access_denied"]) assert.equal(response.body.includes(secret), false);
      assert.equal(manager.get(started.id)?.status, "failed"); assert.equal(calls, 0); assert.equal(fs.existsSync(store.file), false);
    } finally { manager.close(); store.dispose(); }
  });

  it("uses the fallback port and reports both ports busy without displacing owners", async () => {
    const one = await occupy(); const two = await occupy(); const store = temporaryStore();
    const fallback = new BrowserLoginManager(store.file, { ports: [one.port, 0], fetch: async () => json(tokens) });
    const busy = new BrowserLoginManager(path.join(path.dirname(store.file), "other.json"), { ports: [one.port, two.port], fetch: async () => json(tokens) });
    try {
      const started = await fallback.start(); const { redirect } = authInfo(started.auth_url);
      assert.notEqual(Number(new URL(redirect).port), one.port);
      await assert.rejects(busy.start(), (err: unknown) => { const e = err as { code?: string; status?: number }; return e.code === "callback_port_unavailable" && e.status === 409; });
      assert.equal(fallback.get(started.id)?.status, "pending");
    } finally { fallback.close(); busy.close(); await one.close(); await two.close(); store.dispose(); }
  });

  it("rejects a concurrent start and closes a cancelled listener before any exchange", async () => {
    const store = temporaryStore(); let calls = 0;
    const manager = new BrowserLoginManager(store.file, { ports: [0], fetch: async () => { calls++; return json(tokens); } });
    try {
      const started = await manager.start(); const { redirect, state } = authInfo(started.auth_url);
      await assert.rejects(manager.start(), (err: unknown) => { const e = err as { code?: string; status?: number }; return e.code === "login_in_progress" && e.status === 409; });
      assert.equal(manager.cancel(started.id)?.status, "cancelled");
      assert.equal(manager.cancel(started.id)?.status, "cancelled");
      await assert.rejects(callback(redirect, `?state=${encodeURIComponent(state)}&code=late`));
      assert.equal(calls, 0); assert.equal(fs.existsSync(store.file), false);
    } finally { manager.close(); store.dispose(); }
  });

  it("claims a callback only once even while the exchange is outstanding", async () => {
    const store = temporaryStore(); let calls = 0;
    let release!: (response: Response) => void; const exchange = new Promise<Response>((resolve) => { release = resolve; });
    const manager = new BrowserLoginManager(store.file, { ports: [0], fetch: async () => { calls++; return exchange; } });
    try {
      const started = await manager.start(); const { redirect, state } = authInfo(started.auth_url);
      const query = `?state=${encodeURIComponent(state)}&code=fake-code`;
      const first = callback(redirect, query); await until(() => calls === 1);
      const second = await callback(redirect, query).catch(() => undefined);
      if (second) assert.notEqual(second.status, 200);
      assert.equal(calls, 1); release(json(tokens));
      assert.equal((await first).status, 200); assert.equal(manager.get(started.id)?.status, "succeeded");
    } finally { manager.close(); store.dispose(); }
  });

  it("expires rather than reporting upstream failure when a request rejects after the deadline", async () => {
    const store = temporaryStore(); const { clock, elapseWithoutTimers } = fakeClock();
    let rejectExchange!: (error: Error) => void; let calls = 0;
    const exchange = new Promise<Response>((_, reject) => { rejectExchange = reject; });
    const manager = new BrowserLoginManager(store.file, { ports: [0], clock, fetch: async () => { calls++; return exchange; } });
    try {
      const started = await manager.start(); const { redirect, state } = authInfo(started.auth_url);
      const response = callback(redirect, `?state=${encodeURIComponent(state)}&code=fake-code`);
      await until(() => calls === 1);
      elapseWithoutTimers(900_000); rejectExchange(new Error("fake failure"));
      assert.notEqual((await response).status, 200);
      assert.equal(manager.get(started.id)?.status, "expired");
      assert.equal(fs.existsSync(store.file), false);
    } finally { manager.close(); store.dispose(); }
  });

  for (const terminal of ["cancelled", "expired"] as const) {
    it(`fences a late exchange after ${terminal}`, async () => {
      const store = temporaryStore(); const { clock, advance } = fakeClock();
      let release!: (response: Response) => void; const exchange = new Promise<Response>((resolve) => { release = resolve; });
      let calls = 0;
      const manager = new BrowserLoginManager(store.file, { ports: [0], clock, fetch: async () => { calls++; return exchange; } });
      try {
        const started = await manager.start(); const { redirect, state } = authInfo(started.auth_url);
        const response = callback(redirect, `?state=${encodeURIComponent(state)}&code=fake-code`).catch(() => undefined);
        await until(() => calls === 1);
        if (terminal === "cancelled") manager.cancel(started.id); else advance(900_000);
        assert.equal(manager.get(started.id)?.status, terminal);
        release(json(tokens)); await response; await turn(); await turn();
        assert.equal(manager.get(started.id)?.status, terminal); assert.equal(fs.existsSync(store.file), false); assert.equal(calls, 1);
        await assert.rejects(callback(redirect, `?state=${encodeURIComponent(state)}&code=late`));
      } finally { manager.close(); store.dispose(); }
    });
  }
});
