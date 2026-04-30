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
