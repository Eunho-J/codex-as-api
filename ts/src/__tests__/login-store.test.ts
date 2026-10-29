import { test } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { commitLoginCredentials, loadTokenData, localAuthStatus, refreshToken } from "../auth.js";

function jwt(payload: object): string {
  return `e30.${Buffer.from(JSON.stringify(payload)).toString("base64url")}.fake`;
}
const claims = { "https://api.openai.com/auth": { chatgpt_account_id: "fake-account" }, exp: 1000000000 };
const credentials = { access_token: jwt(claims), id_token: jwt(claims), refresh_token: "fake-refresh" };

test("new login validates before replacing store and marks local status", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "login-store-"));
  const file = path.join(dir, "private", "auth.json");
  try {
    assert.equal(localAuthStatus(file), "unauthenticated");
    commitLoginCredentials(file, credentials);
    assert.equal(localAuthStatus(file), "authenticated");
    assert.equal(fs.statSync(file).mode & 0o777, 0o600);
    const before = fs.readFileSync(file);
    assert.throws(() => commitLoginCredentials(file, { ...credentials, id_token: "bad" }));
    assert.deepEqual(fs.readFileSync(file), before);
    fs.writeFileSync(file, "not json");
    assert.equal(localAuthStatus(file), "reauthentication_required");
  } finally { fs.rmSync(dir, { recursive: true, force: true }); }
});

test("same-account identical new grant fences an older in-flight refresh", async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "login-fence-"));
  const file = path.join(dir, "auth.json");
  const originalFetch = globalThis.fetch;
  let release!: (value: Response) => void;
  let started!: () => void;
  const entered = new Promise<void>((resolve) => { started = resolve; });
  globalThis.fetch = (() => {
    started();
    return new Promise<Response>((resolve) => { release = resolve; });
  }) as typeof fetch;
  try {
    const old = commitLoginCredentials(file, credentials);
    const flight = refreshToken(file, old);
    await entered;
    const fresh = commitLoginCredentials(file, credentials);
    assert.notEqual(fresh.login_generation, old.login_generation);
    release(new Response(JSON.stringify({ access_token: jwt({ ...claims, exp: 2000000000 }), refresh_token: "old-rotated" }), { status: 200 }));
    const result = await flight;
    assert.equal(result.login_generation, fresh.login_generation);
    assert.equal(loadTokenData(file).refresh_token, "fake-refresh");
  } finally {
    globalThis.fetch = originalFetch;
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
