import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import * as crypto from "node:crypto";
import {
  resolveAuthPath,
  redactText,
  loadTokenData,
  isAuthLocallyAvailable,
  refreshToken,
  ChatGPTOAuthError,
  ChatGPTOAuthMissingError,
} from "../auth.js";

function makeJwt(
  payload: Record<string, unknown> = {},
): string {
  const header = Buffer.from(
    JSON.stringify({ alg: "RS256", typ: "JWT" }),
  ).toString("base64url");
  const body = Buffer.from(JSON.stringify(payload)).toString(
    "base64url",
  );
  return `${header}.${body}.fakesig`;
}

function writeAuthJson(
  dir: string,
  data: Record<string, unknown>,
): string {
  const filePath = path.join(dir, "auth.json");
  fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(filePath, JSON.stringify(data));
  return filePath;
}

describe("resolveAuthPath", () => {
  it("returns default path when no arg", () => {
    const result = resolveAuthPath();
    assert.ok(result.endsWith(".codex/auth.json"));
  });

  it("uses explicit path", () => {
    const result = resolveAuthPath("/tmp/custom/auth.json");
    assert.equal(result, "/tmp/custom/auth.json");
  });

  it("expands tilde", () => {
    const result = resolveAuthPath("~/mydir/auth.json");
    assert.equal(
      result,
      path.join(os.homedir(), "mydir/auth.json"),
    );
  });
});

describe("redactText", () => {
  it("replaces secret values", () => {
    const result = redactText(
      "token is abc123 here",
      "abc123",
    );
    assert.equal(result, "token is *** here");
  });

  it("replaces multiple values", () => {
    const result = redactText("foo bar baz", "foo", "baz");
    assert.equal(result, "*** bar ***");
  });

  it("handles null/undefined values", () => {
    const result = redactText("hello world", null, undefined);
    assert.equal(result, "hello world");
  });

  it("replaces longest match first", () => {
    const result = redactText("abcdef", "abc", "abcdef");
    assert.equal(result, "***");
  });

  it("handles empty string values", () => {
    const result = redactText("hello", "");
    assert.equal(result, "hello");
  });
});

describe("loadTokenData", () => {
  it("throws on missing file", () => {
    assert.throws(
      () => loadTokenData("/nonexistent/path/auth.json"),
      ChatGPTOAuthMissingError,
    );
  });

  it("throws on invalid JSON", () => {
    const dir = fs.mkdtempSync(
      path.join(os.tmpdir(), "auth-test-"),
    );
    const filePath = path.join(dir, "auth.json");
    fs.writeFileSync(filePath, "not json{");
    assert.throws(
      () => loadTokenData(filePath),
      ChatGPTOAuthError,
    );
    fs.rmSync(dir, { recursive: true });
  });

  it("throws on missing tokens", () => {
    const dir = fs.mkdtempSync(
      path.join(os.tmpdir(), "auth-test-"),
    );
    const filePath = writeAuthJson(dir, {
      auth_mode: "chatgpt",
      tokens: {},
    });
    assert.throws(
      () => loadTokenData(filePath),
      ChatGPTOAuthError,
    );
    fs.rmSync(dir, { recursive: true });
  });

  it("loads latest root token fields", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-test-"));
    const idToken = makeJwt({
      "https://api.openai.com/auth": {
        chatgpt_account_id: "acct-root",
        chatgpt_plan_type: "plus",
        chatgpt_user_id: "user-root",
      },
    });
    const accessToken = makeJwt({ exp: 9999999999 });
    const filePath = writeAuthJson(dir, {
      access_token: accessToken,
      refresh_token: "refresh-root",
      id_token: idToken,
      personal_access_token: "pat-present-but-not-primary",
      agent_identity: { id: "agent" },
    });

    const data = loadTokenData(filePath);

    assert.equal(data.access_token, accessToken);
    assert.equal(data.refresh_token, "refresh-root");
    assert.equal(data.account_id, "acct-root");
    fs.rmSync(dir, { recursive: true });
  });

  it("refresh persists tokens object for latest root token files", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-test-"));
    const oldToken = makeJwt({
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-old" },
      exp: 9999999999,
    });
    const newToken = makeJwt({
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-new" },
      exp: 9999999999,
    });
    const filePath = writeAuthJson(dir, {
      access_token: oldToken,
      refresh_token: "refresh-old",
      id_token: oldToken,
    });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({
      access_token: newToken,
      refresh_token: "refresh-new",
      id_token: newToken,
    }), { status: 200, headers: { "Content-Type": "application/json" } });
    try {
      const data = await refreshToken(filePath);
      const stored = JSON.parse(fs.readFileSync(filePath, "utf-8")) as Record<string, Record<string, string>>;

      assert.equal(data.account_id, "acct-new");
      assert.equal(stored.tokens.access_token, newToken);
      assert.equal(stored.tokens.refresh_token, "refresh-new");
      assert.equal(stored.tokens.id_token, newToken);
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("reports unsupported PAT-only auth", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-test-"));
    const filePath = writeAuthJson(dir, { personal_access_token: "pat-only" });

    assert.throws(() => loadTokenData(filePath), /personal_access_token-only auth is not supported/);
    fs.rmSync(dir, { recursive: true });
  });

  it("reports unsupported agent-only auth", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-test-"));
    const filePath = writeAuthJson(dir, { agent_identity: { id: "agent-only" } });

    assert.throws(() => loadTokenData(filePath), /agent_identity-only auth is not supported/);
    fs.rmSync(dir, { recursive: true });
  });

  it("reports unsupported Bedrock-only auth", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-test-"));
    const filePath = writeAuthJson(dir, { bedrock_api_key: "bedrock-only" });

    assert.throws(() => loadTokenData(filePath), /bedrock_api_key-only auth is not supported/);
    fs.rmSync(dir, { recursive: true });
  });

  it("loads valid token data", () => {
    const dir = fs.mkdtempSync(
      path.join(os.tmpdir(), "auth-test-"),
    );
    const exp = Math.floor(Date.now() / 1000) + 3600;
    const idToken = makeJwt({
      "https://api.openai.com/auth": {
        chatgpt_account_id: "acct-123",
        chatgpt_plan_type: "plus",
        chatgpt_user_id: "user-456",
      },
    });
    const accessToken = makeJwt({ exp });
    const filePath = writeAuthJson(dir, {
      auth_mode: "chatgpt",
      tokens: {
        access_token: accessToken,
        refresh_token: "refresh-tok",
        id_token: idToken,
      },
    });
    const data = loadTokenData(filePath);
    assert.equal(data.account_id, "acct-123");
    assert.equal(data.plan_type, "plus");
    assert.equal(data.user_id, "user-456");
    assert.equal(data.access_token, accessToken);
    assert.equal(data.refresh_token, "refresh-tok");
    assert.equal(data.fedramp, false);
    assert.ok(data.access_expires_at instanceof Date);
    fs.rmSync(dir, { recursive: true });
  });

  it("throws on invalid auth_mode", () => {
    const dir = fs.mkdtempSync(
      path.join(os.tmpdir(), "auth-test-"),
    );
    const filePath = writeAuthJson(dir, {
      auth_mode: "google",
      tokens: {
        access_token: makeJwt(),
        refresh_token: "r",
        id_token: makeJwt(),
      },
    });
    assert.throws(
      () => loadTokenData(filePath),
      ChatGPTOAuthError,
    );
    fs.rmSync(dir, { recursive: true });
  });

  it("detects fedramp flag", () => {
    const dir = fs.mkdtempSync(
      path.join(os.tmpdir(), "auth-test-"),
    );
    const idToken = makeJwt({
      "https://api.openai.com/auth": {
        chatgpt_account_id: "acct-fed",
        chatgpt_account_is_fedramp: true,
      },
    });
    const filePath = writeAuthJson(dir, {
      auth_mode: "chatgpt",
      tokens: {
        access_token: makeJwt({ exp: 9999999999 }),
        refresh_token: "r",
        id_token: idToken,
      },
    });
    const data = loadTokenData(filePath);
    assert.equal(data.fedramp, true);
    fs.rmSync(dir, { recursive: true });
  });
});

describe("isAuthLocallyAvailable", () => {
  it("returns false for missing file", () => {
    assert.equal(
      isAuthLocallyAvailable("/nonexistent/auth.json"),
      false,
    );
  });

  it("returns true for valid file", () => {
    const dir = fs.mkdtempSync(
      path.join(os.tmpdir(), "auth-test-"),
    );
    const idToken = makeJwt({
      "https://api.openai.com/auth": {
        chatgpt_account_id: "acct-x",
      },
    });
    writeAuthJson(dir, {
      auth_mode: "chatgpt",
      tokens: {
        access_token: makeJwt({ exp: 9999999999 }),
        refresh_token: "r",
        id_token: idToken,
      },
    });
    assert.equal(
      isAuthLocallyAvailable(path.join(dir, "auth.json")),
      true,
    );
    fs.rmSync(dir, { recursive: true });
  });
});
