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
  refreshAfterUnauthorized,
  tokenForRequest,
  ChatGPTOAuthError,
  ChatGPTOAuthMissingError,
  ChatGPTOAuthRefreshError,
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

  it("refresh preserves the latest root token layout", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-test-"));
    const oldToken = makeJwt({
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-old" },
      exp: 9999999999,
    });
    const newToken = makeJwt({
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-old" },
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
      const stored = JSON.parse(fs.readFileSync(filePath, "utf-8")) as Record<string, unknown>;

      assert.equal(data.account_id, "acct-old");
      assert.equal(stored.access_token, newToken);
      assert.equal(stored.refresh_token, "refresh-new");
      assert.equal(stored.id_token, newToken);
      assert.equal(Object.hasOwn(stored, "tokens"), false);
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

describe("OAuth refresh coordination", () => {
  it("refreshes within five minutes, coalesces callers, and preserves partial fields", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-refresh-test-"));
    const accountClaims = {
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-refresh" },
    };
    const oldAccess = makeJwt({ exp: Math.floor(Date.now() / 1000) + 240 });
    const idToken = makeJwt(accountClaims);
    const newAccess = makeJwt({
      exp: Math.floor(Date.now() / 1000) + 3600,
      ...accountClaims,
    });
    const filePath = writeAuthJson(dir, {
      tokens: {
        access_token: oldAccess,
        refresh_token: "refresh-preserved",
        id_token: idToken,
      },
    });
    const previousFetch = globalThis.fetch;
    let calls = 0;
    globalThis.fetch = async () => {
      calls += 1;
      await new Promise((resolve) => setTimeout(resolve, 25));
      return new Response(JSON.stringify({ access_token: newAccess }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    };
    try {
      const [first, second] = await Promise.all([
        tokenForRequest(filePath),
        tokenForRequest(filePath),
      ]);
      assert.equal(calls, 1);
      assert.equal(first.access_token, newAccess);
      assert.equal(second.access_token, newAccess);
      assert.equal(first.refresh_token, "refresh-preserved");
      assert.equal(first.id_token, idToken);
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("reuses a matching-account auth file update after unauthorized", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-reload-test-"));
    const accountClaims = {
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-shared" },
    };
    const filePath = writeAuthJson(dir, {
      tokens: {
        access_token: makeJwt({ exp: 9_999_999_999 }),
        refresh_token: "refresh-old",
        id_token: makeJwt(accountClaims),
      },
    });
    const observed = loadTokenData(filePath);
    const newAccess = makeJwt({ exp: 9_999_999_999, ...accountClaims });
    fs.writeFileSync(filePath, JSON.stringify({
      tokens: {
        access_token: newAccess,
        refresh_token: "refresh-from-file",
        id_token: makeJwt(accountClaims),
      },
    }));
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => {
      throw new Error("matching file update must avoid another token rotation");
    };
    try {
      const result = await refreshAfterUnauthorized(observed);
      assert.equal(result.access_token, newAccess);
      assert.equal(result.refresh_token, "refresh-from-file");
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("rejects an account switch before refresh transport", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-account-switch-test-"));
    const credentials = (account: string, version: string) => {
      const claims = { "https://api.openai.com/auth": { chatgpt_account_id: account } };
      return {
        access_token: makeJwt({ exp: 9_999_999_999, version, ...claims }),
        refresh_token: `refresh-${account}`,
        id_token: makeJwt(claims),
      };
    };
    const filePath = writeAuthJson(dir, { tokens: credentials("acct-old", "old") });
    const observed = loadTokenData(filePath);
    fs.writeFileSync(filePath, JSON.stringify({ tokens: credentials("acct-new", "new") }));
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => {
      throw new Error("account switch must fail before refresh transport");
    };
    try {
      await assert.rejects(refreshAfterUnauthorized(observed), /account changed/);
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("refreshes with latest same-account refresh and ID credentials", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-latest-credentials-test-"));
    const claims = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-shared" } };
    const access = makeJwt({ exp: 9_999_999_999, ...claims });
    const filePath = writeAuthJson(dir, { tokens: {
      access_token: access,
      refresh_token: "refresh-old",
      id_token: makeJwt({ ...claims, version: "old" }),
    } });
    const observed = loadTokenData(filePath);
    fs.writeFileSync(filePath, JSON.stringify({ tokens: {
      access_token: access,
      refresh_token: "refresh-latest",
      id_token: makeJwt({ ...claims, version: "latest" }),
    } }));
    const refreshedAccess = makeJwt({ exp: 9_999_999_999, version: "refreshed", ...claims });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async (_input, init) => {
      assert.equal(JSON.parse(String(init?.body)).refresh_token, "refresh-latest");
      return new Response(JSON.stringify({ access_token: refreshedAccess }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    };
    try {
      assert.equal((await refreshAfterUnauthorized(observed)).access_token, refreshedAccess);
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("compare-and-set reuses a concurrent same-account access update", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-cas-access-test-"));
    const claims = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-shared" } };
    const current = {
      access_token: makeJwt({ exp: 1, version: "old", ...claims }),
      refresh_token: "refresh-old",
      id_token: makeJwt(claims),
    };
    const filePath = writeAuthJson(dir, { tokens: current });
    const externalAccess = makeJwt({ exp: 9_999_999_999, version: "external", ...claims });
    const responseAccess = makeJwt({ exp: 9_999_999_999, version: "response", ...claims });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => ({
      ok: true,
      status: 200,
      async json() {
        fs.writeFileSync(filePath, JSON.stringify({ tokens: { ...current, access_token: externalAccess } }));
        return { access_token: responseAccess };
      },
    }) as Response;
    try {
      const result = await refreshToken(filePath);
      assert.equal(result.access_token, externalAccess);
      assert.equal(loadTokenData(filePath).access_token, externalAccess);
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("compare-and-set rejects another credential change", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-cas-other-test-"));
    const claims = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-shared" } };
    const current = {
      access_token: makeJwt({ exp: 1, ...claims }),
      refresh_token: "refresh-old",
      id_token: makeJwt(claims),
    };
    const filePath = writeAuthJson(dir, { tokens: current });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => ({
      ok: true,
      status: 200,
      async json() {
        fs.writeFileSync(filePath, JSON.stringify({ tokens: { ...current, refresh_token: "refresh-raced" } }));
        return { access_token: makeJwt({ exp: 9_999_999_999, ...claims }) };
      },
    }) as Response;
    try {
      await assert.rejects(refreshToken(filePath), /changed while token refresh was in flight/);
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("rejects loaded and refreshed account claim mismatches", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-account-claim-test-"));
    const oldClaims = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-old" } };
    const newClaims = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-new" } };
    const filePath = writeAuthJson(dir, { tokens: {
      access_token: makeJwt({ exp: 1, ...newClaims }),
      refresh_token: "refresh-old",
      id_token: makeJwt(oldClaims),
    } });
    assert.throws(() => loadTokenData(filePath), /account ids do not match/);
    fs.writeFileSync(filePath, JSON.stringify({ tokens: {
      access_token: makeJwt({ exp: 1, ...oldClaims }),
      refresh_token: "refresh-old",
      id_token: makeJwt(oldClaims),
    } }));
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({
      access_token: makeJwt({ exp: 9_999_999_999, ...newClaims }),
    }), { status: 200, headers: { "Content-Type": "application/json" } });
    try {
      await assert.rejects(refreshToken(filePath), /does not match current account/);
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("redacts account IDs from refresh failures", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-redaction-test-"));
    const claims = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-secret" } };
    const filePath = writeAuthJson(dir, { tokens: {
      access_token: makeJwt({ exp: 1, ...claims }),
      refresh_token: "refresh-old",
      id_token: makeJwt(claims),
    } });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response("account=acct-secret", { status: 400 });
    try {
      await assert.rejects(
        refreshToken(filePath),
        (error: unknown) => error instanceof ChatGPTOAuthRefreshError
          && !error.message.includes("acct-secret")
          && error.message.includes("account=***"),
      );
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true });
    }
  });
});
