import { test } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { commitLoginCredentials, loadTokenData } from "../auth.js";
import { DeviceLoginManager, type DeviceLoginClock } from "../device-login.js";

const initial = { device_auth_id: "device-id", user_code: "user-code", interval: 2 };
const authorization = { authorization_code: "code", code_verifier: "verifier", code_challenge: "challenge" };
const claims = { "https://api.openai.com/auth": { chatgpt_account_id: "fake-account" }, exp: 2000000000 };
const jwt = `e30.${Buffer.from(JSON.stringify(claims)).toString("base64url")}.fake`;
const tokens = { access_token: jwt, id_token: jwt, refresh_token: "fake-refresh" };
const json = (data: unknown) => Response.json(data);
const turn = () => new Promise<void>((resolve) => setImmediate(resolve));
async function until(condition: () => boolean): Promise<void> {
  for (let i = 0; i < 100; i++) { if (condition()) return; await turn(); }
  assert.fail("expected device flow state not reached");
}
class Clock implements DeviceLoginClock {
  time = 0;
  timers = new Map<object, { at: number; callback: () => void }>();
  now(): number { return this.time; }
  wallNow(): number { return 1_000; }
  setTimeout(callback: () => void, ms: number): unknown {
    const key = {}; this.timers.set(key, { at: this.time + ms, callback }); return key;
  }
  clearTimeout(key: unknown): void { this.timers.delete(key as object); }
  advance(ms: number): void {
    this.time += ms;
    for (const [key, timer] of [...this.timers]) {
      if (timer.at <= this.time) { this.timers.delete(key); timer.callback(); }
    }
  }
}

for (const ending of ["cancelled", "expired"] as const) {
  test(`${ending} during exchange aborts and ignores a late token response`, async () => {
    const clock = new Clock();
    let release!: (response: Response) => void;
    const exchange = new Promise<Response>((resolve) => { release = resolve; });
    let exchangeSignal: AbortSignal | undefined;
    let calls = 0, commits = 0;
    const manager = new DeviceLoginManager(`/fake-device-races/${ending}`, {
      clock,
      fetch: async (_url, init) => {
        calls++;
        if (calls === 1) return json(initial);
        if (calls === 2) return json(authorization);
        if (calls === 3) { exchangeSignal = init?.signal as AbortSignal; return exchange; }
        assert.fail("unexpected request");
      },
      commit: () => { commits++; },
    });
    try {
      const session = await manager.start();
      await until(() => calls === 3);
      if (ending === "cancelled") assert.equal(manager.cancel(session.id)?.status, ending);
      else { clock.advance(900_000); assert.equal(manager.get(session.id)?.status, ending); }
      assert.equal(exchangeSignal?.aborted, true);
      release(json(tokens));
      await turn(); await turn();
      assert.equal(commits, 0);
      assert.equal(manager.get(session.id)?.status, ending);
      assert.equal(calls, 3);
    } finally { manager.close(); }
  });
}

test("overlapping starts are rejected while the initial request is unresolved", async () => {
  const clock = new Clock();
  let release!: (response: Response) => void;
  const initialResponse = new Promise<Response>((resolve) => { release = resolve; });
  let calls = 0;
  const transport: typeof fetch = async () => {
    calls++;
    if (calls === 1) return initialResponse;
    if (calls === 2) return new Response(null, { status: 403 });
    assert.fail("unexpected request");
  };
  const first = new DeviceLoginManager("/fake-device-races/overlap", { clock, fetch: transport });
  const second = new DeviceLoginManager("/fake-device-races/overlap", { clock, fetch: transport });
  try {
    const starting = first.start();
    assert.equal(calls, 1);
    await assert.rejects(second.start(), { code: "login_in_progress", status: 409 });
    assert.equal(calls, 1);
    release(json(initial));
    const session = await starting;
    assert.equal(session.status, "pending");
    assert.equal(session.user_code, initial.user_code);
    await until(() => calls === 2);
    assert.equal(first.get(session.id)?.status, "pending");
  } finally { first.close(); second.close(); }
});

test("huge poll interval is capped by the total deadline", async () => {
  const clock = new Clock();
  let calls = 0, commits = 0;
  const manager = new DeviceLoginManager("/fake-device-races/huge-interval", {
    clock, fetch: async () => {
      calls++;
      if (calls === 1) return json({ ...initial, interval: Number.MAX_SAFE_INTEGER });
      if (calls === 2) return new Response(null, { status: 403 });
      assert.fail("unexpected request");
    }, commit: () => { commits++; },
  });
  try {
    const session = await manager.start();
    await until(() => calls === 2 && clock.timers.size === 2);
    assert.deepEqual([...clock.timers.values()].map((timer) => timer.at), [900_000, 900_000]);
    clock.advance(899_999); await turn();
    assert.equal(manager.get(session.id)?.status, "pending");
    assert.equal(calls, 2);
    clock.advance(1); await turn();
    assert.equal(manager.get(session.id)?.status, "expired");
    assert.equal(calls, 2);
    assert.equal(commits, 0);
    assert.equal(clock.timers.size, 0);
  } finally { manager.close(); }
});

test("deadline expires while the token response body is stalled", async () => {
  const clock = new Clock();
  let body!: ReadableStreamDefaultController<Uint8Array>;
  const stream = new ReadableStream<Uint8Array>({ start(controller) { body = controller; } });
  let calls = 0, commits = 0;
  let exchangeSignal: AbortSignal | undefined;
  const manager = new DeviceLoginManager("/fake-device-races/stalled-body", {
    clock, fetch: async (_url, init) => {
      calls++;
      if (calls === 1) return json(initial);
      if (calls === 2) return json(authorization);
      if (calls === 3) {
        exchangeSignal = init?.signal as AbortSignal;
        return new Response(stream, { headers: { "Content-Type": "application/json" } });
      }
      assert.fail("unexpected request");
    }, commit: () => { commits++; },
  });
  try {
    const session = await manager.start();
    await until(() => calls === 3 && stream.locked);
    clock.advance(899_999); await turn();
    assert.equal(manager.get(session.id)?.status, "pending");
    clock.advance(1); await turn();
    assert.equal(manager.get(session.id)?.status, "expired");
    assert.equal(exchangeSignal?.aborted, true);
    body.enqueue(new TextEncoder().encode(JSON.stringify(tokens)));
    body.close();
    await turn(); await turn();
    assert.equal(manager.get(session.id)?.status, "expired");
    assert.equal(commits, 0);
    assert.equal(calls, 3);
  } finally { manager.close(); }
});

test("403 and 404 polls stay pending until the next interval", async () => {
  const clock = new Clock();
  let calls = 0;
  const manager = new DeviceLoginManager("/fake-device-races/pending", {
    clock, fetch: async () => {
      calls++;
      if (calls === 1) return json(initial);
      if (calls === 2) return new Response(null, { status: 403 });
      if (calls === 3) return new Response(null, { status: 404 });
      throw new Error("unexpected request");
    }, commit: () => assert.fail("pending cannot commit"),
  });
  try {
    const session = await manager.start();
    await until(() => calls === 2 && [...clock.timers.values()].some((t) => t.at === 2_000));
    assert.equal(manager.get(session.id)?.status, "pending");
    clock.advance(1_999); await turn(); assert.equal(calls, 2);
    clock.advance(1);
    await until(() => calls === 3 && [...clock.timers.values()].some((t) => t.at === 4_000));
    assert.equal(manager.get(session.id)?.status, "pending");
    clock.advance(1_999); await turn(); assert.equal(calls, 3);
  } finally { manager.close(); }
});

test("success is reported only after a private real project store is committed", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "device-races-"));
  const file = path.join(dir, "private", "auth.json");
  let release!: (response: Response) => void;
  const exchange = new Promise<Response>((resolve) => { release = resolve; });
  let calls = 0, id = "", observedBeforeSuccess = false;
  const manager = new DeviceLoginManager(file, {
    fetch: async () => {
      calls++;
      if (calls === 1) return json(initial);
      if (calls === 2) return json(authorization);
      if (calls === 3) return exchange;
      throw new Error("unexpected request");
    },
    commit: (store, payload) => {
      commitLoginCredentials(store, payload);
      assert.equal(manager.get(id)?.status, "pending");
      assert.equal(loadTokenData(file).refresh_token, tokens.refresh_token);
      assert.equal(fs.statSync(file).mode & 0o777, 0o600);
      observedBeforeSuccess = true;
    },
  });
  try {
    const session = await manager.start(); id = session.id;
    await until(() => calls === 3);
    assert.equal(fs.existsSync(file), false);
    assert.equal(manager.get(id)?.status, "pending");
    release(json(tokens));
    await until(() => manager.get(id)?.status === "succeeded");
    assert.equal(observedBeforeSuccess, true);
    assert.equal(loadTokenData(file).refresh_token, tokens.refresh_token);
    assert.equal(fs.statSync(path.dirname(file)).mode & 0o777, 0o700);
    assert.equal(manager.cancel(id)?.status, "succeeded");
  } finally { manager.close(); fs.rmSync(dir, { recursive: true, force: true }); }
});
