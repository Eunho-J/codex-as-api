import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { loadTokenData } from "../auth.js";
import { DeviceLoginError, DeviceLoginManager } from "../device-login.js";

const initial = { device_auth_id: "fake-device-id", user_code: "FAKE-CODE", interval: 1 };
const authorization = { authorization_code: "fake-auth-code", code_verifier: "fake-verifier", code_challenge: "fake-challenge" };
const tokens = { access_token: "fake-access", refresh_token: "fake-refresh", id_token: "fake-id" };
const json = (value: unknown, status = 200) => new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
const turn = () => new Promise<void>((resolve) => setImmediate(resolve));
async function until(predicate: () => boolean) {
  for (let i = 0; i < 50; i++) { if (predicate()) return; await turn(); }
  assert.fail("login did not reach expected state");
}

describe("device login contract with fake transport", () => {
  it("uses fixed endpoints and commits once before reporting success", async () => {
    const calls: Array<{ url: string; init: RequestInit }> = [];
    const committed: unknown[] = [];
    const responses = [json(initial), json(authorization), json(tokens)];
    const transport: typeof fetch = async (input, init) => {
      calls.push({ url: String(input), init: init! });
      const response = responses.shift();
      if (!response) throw new Error("unexpected request");
      return response;
    };
    const manager = new DeviceLoginManager("/fake-test/success-auth.json", { fetch: transport, commit: (_path, value) => committed.push(value) });
    try {
      const started = await manager.start();
      assert.equal(started.user_code, "FAKE-CODE");
      await until(() => manager.get(started.id)?.status === "succeeded");
      assert.deepEqual(committed, [tokens]);
      assert.deepEqual(calls.map((c) => c.url), ["https://auth.openai.com/api/accounts/deviceauth/usercode", "https://auth.openai.com/api/accounts/deviceauth/token", "https://auth.openai.com/oauth/token"]);
      assert.ok(calls.every((c) => c.init.redirect === "manual"));
      const form = new URLSearchParams(String(calls[2]!.init.body));
      assert.equal(form.get("client_id"), "app_EMoamEEZ73f0CkXaXp7hrann");
      assert.equal(form.get("redirect_uri"), "https://auth.openai.com/deviceauth/callback");
      assert.equal(form.get("grant_type"), "authorization_code");
      const snapshot = manager.get(started.id)!;
      assert.equal(snapshot.status, "succeeded");
      for (const secret of ["FAKE-CODE", "fake-device-id", "fake-auth-code", "fake-verifier", "fake-access", "fake-refresh", "fake-id"]) {
        assert.equal(JSON.stringify(snapshot).includes(secret), false);
      }
      assert.equal(manager.cancel(started.id)?.status, "succeeded");
    } finally { manager.close(); }
  });
  it("distinguishes unsupported device flow and malformed initiation", async () => {
    for (const [response, code] of [[new Response("private upstream body", { status: 404 }), "device_login_unsupported"], [json({ ...initial, interval: 0 }), "invalid_login_response"]] as const) {
      const manager = new DeviceLoginManager(`/fake-test/${code}.json`, { fetch: async () => response });
      try {
        await assert.rejects(manager.start(), (err: unknown) => err instanceof DeviceLoginError && err.code === code && err.status === 502 && !err.message.includes("private upstream body"));
      } finally { manager.close(); }
    }
  });
  it("persists a validated fake grant before reporting success", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "device-persist-test-"));
    const file = path.join(dir, "auth.json");
    const payload = Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: "fake-account" } })).toString("base64url");
    const fakeJwt = `e30.${payload}.signature`;
    const replies = [json(initial), json(authorization), json({ access_token: fakeJwt, id_token: fakeJwt, refresh_token: "fake-refresh" })];
    const manager = new DeviceLoginManager(file, { fetch: async () => { const response = replies.shift(); if (!response) throw Error("unexpected request"); return response; } });
    try {
      const started = await manager.start();
      await until(() => manager.get(started.id)?.status !== "pending");
      assert.equal(manager.get(started.id)?.status, "succeeded");
      assert.equal(loadTokenData(file).account_id, "fake-account");
      assert.equal(loadTokenData(file).refresh_token, "fake-refresh");
      if (process.platform !== "win32") assert.equal(fs.statSync(file).mode & 0o777, 0o600);
    } finally { manager.close(); fs.rmSync(dir, { recursive: true, force: true }); }
  });
  for (const terminal of ["cancelled", "expired"] as const) {
    it(`prevents commit if ${terminal} during authorization-code exchange`, async () => {
      let now = 0;
      const timers = new Map<object, { callback: () => void; at: number }>();
      const clock = {
        now: () => now, wallNow: () => 0,
        setTimeout(callback: () => void, ms: number) { const key = {}; timers.set(key, { callback, at: now + ms }); return key; },
        clearTimeout(key: unknown) { timers.delete(key as object); },
      };
      let release!: (response: Response) => void;
      const exchange = new Promise<Response>((resolve) => { release = resolve; });
      let calls = 0, commits = 0;
      const manager = new DeviceLoginManager(`/fake-test/exchange-${terminal}.json`, {
        clock, fetch: async () => { calls++; if (calls === 1) return json(initial); if (calls === 2) return json(authorization); return exchange; },
        commit: () => { commits++; },
      });
      try {
        const started = await manager.start();
        await until(() => calls === 3);
        if (terminal === "cancelled") manager.cancel(started.id);
        else { now = 900_000; for (const timer of [...timers.values()]) if (timer.at <= now) timer.callback(); }
        assert.equal(manager.get(started.id)?.status, terminal);
        release(json(tokens));
        await turn(); await turn();
        assert.equal(manager.get(started.id)?.status, terminal);
        assert.equal(commits, 0);
        assert.equal(calls, 3);
      } finally { manager.close(); }
    });
  }
  it("treats poll 403 and 404 as pending and respects the interval", async () => {
    let now = 0;
    const timers = new Map<object, { callback: () => void; at: number }>();
    const clock = {
      now: () => now, wallNow: () => 0,
      setTimeout(callback: () => void, ms: number) { const key = {}; timers.set(key, { callback, at: now + ms }); return key; },
      clearTimeout(key: unknown) { timers.delete(key as object); },
    };
    let calls = 0;
    const manager = new DeviceLoginManager("/fake-test/poll-pending.json", {
      clock,
      fetch: async () => {
        calls++;
        if (calls === 1) return json({ ...initial, interval: "2" });
        if (calls === 2) return new Response(null, { status: 403 });
        if (calls === 3) return new Response(null, { status: 404 });
        throw new Error("unexpected extra poll");
      },
      commit: () => assert.fail("pending poll must not commit"),
    });
    const advance = (ms: number) => {
      now += ms;
      for (const [key, timer] of [...timers]) if (timer.at <= now) { timers.delete(key); timer.callback(); }
    };
    try {
      const started = await manager.start();
      await until(() => calls === 2 && [...timers.values()].some((timer) => timer.at === 2_000));
      assert.equal(started.interval, 2);
      assert.equal(manager.get(started.id)?.status, "pending");
      advance(1_999); await turn(); assert.equal(calls, 2);
      advance(1); await until(() => calls === 3 && [...timers.values()].some((timer) => timer.at === 4_000));
      assert.equal(manager.get(started.id)?.status, "pending");
      assert.equal(manager.cancel(started.id)?.status, "cancelled");
    } finally { manager.close(); }
  });
  it("expires an outstanding request at the total deadline and ignores late results", async () => {
    let now = 0;
    const timers = new Map<object, { callback: () => void; at: number }>();
    const clock = {
      now: () => now,
      wallNow: () => 1_000,
      setTimeout(callback: () => void, ms: number) { const key = {}; timers.set(key, { callback, at: now + ms }); return key; },
      clearTimeout(key: unknown) { timers.delete(key as object); },
    };
    let release!: (response: Response) => void;
    const poll = new Promise<Response>((resolve) => { release = resolve; });
    let calls = 0, commits = 0;
    const manager = new DeviceLoginManager("/fake-test/expiry-auth.json", {
      clock, fetch: async () => ++calls === 1 ? json(initial) : poll, commit: () => { commits++; },
    });
    try {
      const started = await manager.start();
      assert.equal(started.expires_at, new Date(901_000).toISOString());
      now = 900_000;
      for (const timer of [...timers.values()]) if (timer.at <= now) timer.callback();
      assert.equal(manager.get(started.id)?.status, "expired");
      release(json(authorization));
      await turn(); await turn();
      assert.equal(commits, 0);
      assert.equal(manager.get(started.id)?.status, "expired");
    } finally { manager.close(); }
  });
  it("fences a late poll response after cancellation and rejects concurrent starts", async () => {
    let release!: (response: Response) => void;
    const poll = new Promise<Response>((resolve) => { release = resolve; });
    let calls = 0;
    let commits = 0;
    const manager = new DeviceLoginManager("/fake-test/cancel-auth.json", {
      fetch: async () => ++calls === 1 ? json(initial) : poll,
      commit: () => { commits++; },
    });
    try {
      const started = await manager.start();
      await assert.rejects(manager.start(), (err: unknown) => err instanceof DeviceLoginError && err.code === "login_in_progress" && err.status === 409);
      assert.equal(manager.cancel(started.id)?.status, "cancelled");
      assert.equal(manager.cancel(started.id)?.status, "cancelled");
      release(json(authorization));
      await turn(); await turn();
      assert.equal(manager.get(started.id)?.status, "cancelled");
      assert.equal(commits, 0);
      assert.equal(calls, 2);
    } finally { manager.close(); }
  });
});
