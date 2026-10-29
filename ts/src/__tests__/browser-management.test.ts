import { test } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { createApp } from "../server.js";
import { CombinedLoginManager } from "../auth-management.js";
import { BrowserLoginManager } from "../browser-login.js";
import { DeviceLoginManager } from "../device-login.js";

test("real management app selects and coordinates browser and device logins", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "browser-management-test-"));
  const oldHome = process.env.HOME;
  const oldAdmin = process.env.CODEX_AS_API_ADMIN_KEY;
  process.env.HOME = dir;
  process.env.CODEX_AS_API_ADMIN_KEY = "test-admin";
  const file = path.join(dir, "auth.json");
  let releaseInitial!: (response: Response) => void;
  let initialStarted!: () => void;
  const started = new Promise<void>((resolve) => { initialStarted = resolve; });
  const initial = new Promise<Response>((resolve) => { releaseInitial = resolve; });
  let requests = 0;
  const transport: typeof fetch = async (input) => {
    requests++;
    const url = String(input);
    if (url.endsWith("/deviceauth/usercode")) { initialStarted(); return initial; }
    if (url.endsWith("/deviceauth/token")) return new Response(null, { status: 403 });
    throw new Error("Unexpected fake upstream request");
  };
  const browser = new BrowserLoginManager(file, { ports: [0], fetch: transport });
  const device = new DeviceLoginManager(file, { fetch: transport });
  const manager = new CombinedLoginManager(browser, device);
  const app = createApp({ provider: {} as never, codexConfig: { codexHome: dir, configPath: path.join(dir, "config.toml") }, authPath: file, loginManager: manager });
  const server = app.listen(0, "127.0.0.1");
  await new Promise<void>((resolve, reject) => { server.once("listening", resolve); server.once("error", reject); });
  const address = server.address();
  assert.ok(address && typeof address !== "string");
  const base = `http://127.0.0.1:${address.port}/auth`;
  async function request(method: string, route: string, body?: string) {
    const response = await fetch(base + route, {
      method,
      headers: { authorization: "Bearer test-admin", ...(body === undefined ? {} : { "content-type": "application/json" }) },
      body,
      signal: AbortSignal.timeout(2000),
    });
    assert.equal(response.headers.get("cache-control"), "no-store");
    return { status: response.status, body: await response.json() as Record<string, any> };
  }
  function snapshot(result: { status: number; body: Record<string, any> }, method: string, status: string) {
    assert.equal(result.status, 200);
    assert.equal(result.body.method, method);
    assert.equal(result.body.status, status);
    assert.deepEqual(Object.keys(result.body).sort(), ["expires_at", "id", "method", "status"]);
    for (const secret of ["fake-user-code", "fake-device-id", "auth_url", "verification_url", "code_verifier", "access_token", "refresh_token", "id_token"]) {
      assert.equal(JSON.stringify(result.body).includes(secret), false);
    }
  }
  try {
    for (const invalid of ["null", "[]", "{", '{"method":"unknown"}', '{"method":null}', '{"method":"browser","token":"secret"}', '{"method":"device","path":"x"}']) {
      const result = await request("POST", "/login", invalid);
      assert.equal(result.status, 400);
      assert.equal(result.body.error.code, "invalid_request");
    }
    assert.equal(requests, 0);
    for (const body of [undefined, "{}", '{"method":"browser"}']) {
      const result = await request("POST", "/login", body);
      assert.equal(result.status, 202);
      assert.equal(result.body.method, "browser");
      assert.equal(result.body.status, "pending");
      assert.deepEqual(Object.keys(result.body).sort(), ["auth_url", "expires_at", "id", "method", "status"]);
      const auth = new URL(result.body.auth_url);
      assert.equal(auth.origin, "https://auth.openai.com");
      assert.ok(auth.searchParams.get("state"));
      const redirect = new URL(auth.searchParams.get("redirect_uri")!);
      assert.equal(redirect.protocol, "http:");
      assert.equal(redirect.hostname, "127.0.0.1");
      assert.ok(Number(redirect.port) > 0);
      assert.equal(redirect.pathname, "/auth/callback");
      const conflict = await request("POST", "/login", '{"method":"device"}');
      assert.equal(conflict.status, 409);
      assert.equal(conflict.body.error.code, "login_in_progress");
      assert.equal(requests, 0);
      const id = result.body.id;
      snapshot(await request("GET", `/login/${id}`), "browser", "pending");
      snapshot(await request("DELETE", `/login/${id}`), "browser", "cancelled");
      snapshot(await request("DELETE", `/login/${id}`), "browser", "cancelled");
      snapshot(await request("GET", `/login/${id}`), "browser", "cancelled");
    }
    const pendingDevice = request("POST", "/login", '{"method":"device"}');
    let timeout: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        started,
        pendingDevice.then((result) => { throw new Error(`Device start completed before transport: ${result.status} ${JSON.stringify(result.body)}`); }),
        new Promise<never>((_, reject) => { timeout = setTimeout(() => reject(new Error("Device transport did not start")), 2000); }),
      ]);
    } finally { if (timeout) clearTimeout(timeout); }
    const conflict = await request("POST", "/login", "{}");
    assert.equal(conflict.status, 409);
    assert.equal(conflict.body.error.code, "login_in_progress");
    releaseInitial(Response.json({ device_auth_id: "fake-device-id", user_code: "fake-user-code", interval: 5 }));
    const result = await pendingDevice;
    assert.equal(result.status, 202);
    assert.equal(result.body.method, "device");
    assert.equal(result.body.status, "pending");
    assert.deepEqual(Object.keys(result.body).sort(), ["expires_at", "id", "interval", "method", "status", "user_code", "verification_url"]);
    assert.equal(result.body.user_code, "fake-user-code");
    assert.equal(result.body.interval, 5);
    assert.equal(result.body.verification_url, "https://auth.openai.com/codex/device");
    const id = result.body.id;
    snapshot(await request("GET", `/login/${id}`), "device", "pending");
    snapshot(await request("DELETE", `/login/${id}`), "device", "cancelled");
    snapshot(await request("GET", `/login/${id}`), "device", "cancelled");
    const missing = await request("GET", "/login/not-a-real-id");
    assert.equal(missing.status, 404);
    assert.equal(missing.body.error.code, "login_not_found");
    assert.equal(fs.existsSync(file), false);
  } finally {
    releaseInitial(Response.json({ device_auth_id: "fake-device-id", user_code: "fake-user-code", interval: 5 }));
    manager.close();
    server.closeAllConnections();
    await new Promise<void>((resolve, reject) => server.close((err) => err ? reject(err) : resolve()));
    if (oldHome === undefined) delete process.env.HOME; else process.env.HOME = oldHome;
    if (oldAdmin === undefined) delete process.env.CODEX_AS_API_ADMIN_KEY; else process.env.CODEX_AS_API_ADMIN_KEY = oldAdmin;
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
