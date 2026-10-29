import { test } from "node:test";
import assert from "node:assert/strict";
import express from "express";
import { createAuthManagementRouter, type LoginManager } from "../auth-management.js";
import { createApp } from "../server.js";

test("management routes authenticate, reject foreign origins, and expose only safe fields", async () => {
  const previous = process.env.CODEX_AS_API_ADMIN_KEY;
  let starts = 0;
  const record = { id: "test-id", status: "pending" as const, expires_at: "2030-01-01T00:00:00Z", access_token: "secret-token", user_code: "secret-code" };
  const manager: LoginManager = {
    async start() { starts++; return { ...record, verification_url: "https://auth.openai.com/codex/device", interval: 5 }; },
    get(id) { return id === record.id ? record : undefined; },
    cancel(id) { return id === record.id ? { ...record, status: "cancelled" as const } : undefined; },
    close() {},
  };
  const app = express();
  app.use("/auth", createAuthManagementRouter(manager, () => "authenticated"));
  const server = app.listen(0, "127.0.0.1");
  await new Promise<void>((resolve) => server.once("listening", resolve));
  const address = server.address();
  assert.ok(address && typeof address !== "string");
  const base = `http://127.0.0.1:${address.port}`;
  const request = (path: string, options: RequestInit = {}) => fetch(base + path, options);
  const check = async (response: Response, expected: number) => {
    assert.equal(response.status, expected);
    assert.equal(response.headers.get("cache-control"), "no-store");
    return response.json() as Promise<Record<string, unknown>>;
  };
  try {
    delete process.env.CODEX_AS_API_ADMIN_KEY;
    assert.equal((await check(await request("/auth/status"), 503)).error && starts, 0);
    process.env.CODEX_AS_API_ADMIN_KEY = "admin-test";
    await check(await request("/auth/status"), 401);
    const auth = { Authorization: "Bearer admin-test" };
    await check(await request("/auth/login", { method: "POST", headers: { ...auth, Origin: "https://evil.example" } }), 403);
    for (const body of ['{"token":"bad"}', '{"issuer":"https://evil.test"}', '[]', '{', 'null']) {
      await check(await request("/auth/login", { method: "POST", headers: { ...auth, "Content-Type": "application/json" }, body }), 400);
    }
    assert.equal(starts, 0);
    const initial = await check(await request("/auth/login", { method: "POST", headers: auth }), 202);
    assert.equal(initial.user_code, "secret-code");
    assert.ok(!JSON.stringify(initial).includes("secret-token"));
    const result = await check(await request("/auth/login/test-id", { headers: auth }), 200);
    assert.ok(!JSON.stringify(result).includes("secret"));
    const cancelled = await check(await request("/auth/login/test-id", { method: "DELETE", headers: { ...auth, Origin: base } }), 200);
    assert.equal(cancelled.status, "cancelled");
    await check(await request("/auth/login/missing", { headers: auth }), 404);
    assert.deepEqual(await check(await request("/auth/status", { headers: auth }), 200), { status: "authenticated" });
  } finally {
    if (previous === undefined) delete process.env.CODEX_AS_API_ADMIN_KEY; else process.env.CODEX_AS_API_ADMIN_KEY = previous;
    await new Promise<void>((resolve, reject) => server.close((err) => err ? reject(err) : resolve()));
  }
});

test("createApp close cancels pending login before waiting for its request", async () => {
  const previous = process.env.CODEX_AS_API_ADMIN_KEY;
  process.env.CODEX_AS_API_ADMIN_KEY = "fake-admin";
  let started!: () => void;
  const startedPromise = new Promise<void>((resolve) => { started = resolve; });
  let cancel!: () => void;
  let closes = 0;
  const manager: LoginManager = {
    start: () => new Promise((_resolve, reject) => {
      cancel = () => reject(new Error("cancelled"));
      started();
    }),
    get: () => undefined,
    cancel: () => undefined,
    close: () => { closes++; cancel?.(); },
  };
  const app = createApp({
    provider: {} as never,
    codexConfig: { codexHome: "/fake", configPath: "/fake/config.toml" },
    authPath: "/fake/project-auth.json",
    loginManager: manager,
  });
  const server = app.listen(0, "127.0.0.1");
  await new Promise<void>((resolve) => server.once("listening", resolve));
  const address = server.address();
  assert.ok(address && typeof address !== "string");
  try {
    const pending = fetch(`http://127.0.0.1:${address.port}/auth/login`, {
      method: "POST", headers: { Authorization: "Bearer fake-admin", Connection: "close" },
    });
    await startedPromise;
    const closed = new Promise<void>((resolve, reject) => server.close((err) => err ? reject(err) : resolve()));
    assert.equal(closes, 1);
    assert.equal((await pending).status, 502);
    await closed;
    app.locals.closeAuthManagement();
    assert.equal(closes, 1);
  } finally {
    if (previous === undefined) delete process.env.CODEX_AS_API_ADMIN_KEY; else process.env.CODEX_AS_API_ADMIN_KEY = previous;
  }
});
