import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { commitLoginCredentials, loadTokenData, localAuthStatus, resolveAuthPath } from "../auth.js";

function jwt(account: string): string {
  const payload = Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: account } })).toString("base64url");
  return `e30.${payload}.signature`;
}
function grant(account = "fake-account") {
  return { access_token: jwt(account), id_token: jwt(account), refresh_token: "fake-refresh" };
}
function tempStore(): { dir: string; file: string } {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "standalone-auth-test-"));
  return { dir, file: path.join(dir, "private", "auth.json") };
}

describe("standalone credential store contract", () => {
  it("uses explicit path, then project path, then project home; never CODEX_HOME", () => {
    const names = ["CODEX_AS_API_AUTH_PATH", "CODEX_AS_API_HOME", "CODEX_HOME"] as const;
    const old = names.map((name) => process.env[name]);
    const { dir } = tempStore();
    try {
      process.env.CODEX_HOME = path.join(dir, "legacy");
      process.env.CODEX_AS_API_HOME = path.join(dir, "project");
      process.env.CODEX_AS_API_AUTH_PATH = path.join(dir, "configured.json");
      assert.equal(resolveAuthPath(path.join(dir, "explicit.json")), path.join(dir, "explicit.json"));
      assert.equal(resolveAuthPath(), path.join(dir, "configured.json"));
      delete process.env.CODEX_AS_API_AUTH_PATH;
      assert.equal(resolveAuthPath(), path.join(dir, "project", "auth.json"));
      fs.mkdirSync(process.env.CODEX_HOME, { recursive: true });
      fs.writeFileSync(path.join(process.env.CODEX_HOME, "auth.json"), JSON.stringify({ tokens: grant() }));
      assert.equal(localAuthStatus(), "unauthenticated");
    } finally {
      names.forEach((name, i) => { const value = old[i]; if (value === undefined) delete process.env[name]; else process.env[name] = value; });
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });
  it("reports local status and commits validated credentials privately", () => {
    const { dir, file } = tempStore();
    try {
      assert.equal(localAuthStatus(file), "unauthenticated");
      fs.mkdirSync(path.dirname(file));
      fs.writeFileSync(file, "not json");
      assert.equal(localAuthStatus(file), "reauthentication_required");
      commitLoginCredentials(file, grant());
      assert.equal(localAuthStatus(file), "authenticated");
      assert.equal(loadTokenData(file).account_id, "fake-account");
      if (process.platform !== "win32") assert.equal(fs.statSync(file).mode & 0o777, 0o600);
      assert.deepEqual(fs.readdirSync(path.dirname(file)), ["auth.json"]);
    } finally { fs.rmSync(dir, { recursive: true, force: true }); }
  });
  it("does not replace a valid login when a new grant fails validation", () => {
    const { dir, file } = tempStore();
    try {
      commitLoginCredentials(file, grant());
      const before = fs.readFileSync(file);
      assert.throws(() => commitLoginCredentials(file, { ...grant(), refresh_token: "" }));
      assert.deepEqual(fs.readFileSync(file), before);
      assert.equal(localAuthStatus(file), "authenticated");
    } finally { fs.rmSync(dir, { recursive: true, force: true }); }
  });
});
