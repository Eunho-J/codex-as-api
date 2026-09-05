import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import fsDefault from "node:fs";
import { syncBuiltinESMExports } from "node:module";
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
  validateAuthEnvironment,
  ChatGPTOAuthError,
  ChatGPTOAuthMissingError,
  ChatGPTOAuthProtocolError,
  ChatGPTOAuthRefreshError,
  ChatGPTOAuthUpstreamError,
} from "../auth.js";

const REFRESH_URL_OVERRIDE_ENV = "CODEX_REFRESH_TOKEN_URL_OVERRIDE";

function withRefreshUrl(value: string | undefined, callback: () => void): void {
  const previous = process.env[REFRESH_URL_OVERRIDE_ENV];
  if (value === undefined) delete process.env[REFRESH_URL_OVERRIDE_ENV];
  else process.env[REFRESH_URL_OVERRIDE_ENV] = value;
  try {
    callback();
  } finally {
    if (previous === undefined) delete process.env[REFRESH_URL_OVERRIDE_ENV];
    else process.env[REFRESH_URL_OVERRIDE_ENV] = previous;
  }
}

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

function replaceBuiltinFsMethod(name: string, replacement: unknown): () => void {
  const mutableFs = fsDefault as unknown as Record<string, unknown>;
  const original = mutableFs[name];
  mutableFs[name] = replacement;
  syncBuiltinESMExports();
  return () => {
    mutableFs[name] = original;
    syncBuiltinESMExports();
  };
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

describe("validateAuthEnvironment", () => {
  it("accepts the default endpoint and HTTP(S) overrides with paths and ports", () => {
    for (const endpoint of [
      undefined,
      "http://127.0.0.1:18081/oauth/token",
      "https://auth.example.test/custom/path",
      "https://auth.example.test/custom%20path",
    ]) {
      withRefreshUrl(endpoint, () => validateAuthEnvironment());
    }
  });

  it("rejects ambiguous or credential-bearing refresh endpoints", () => {
    for (const endpoint of [
      "",
      " https://auth.example.test/oauth/token",
      "https://auth.example.test/oauth/token ",
      "/oauth/token",
      "https:auth.example.test/oauth/token",
      "https:/auth.example.test/oauth/token",
      "https:///auth.example.test/oauth/token",
      "https://auth.example.test/oauth token",
      "https://auth.example.test/oauth\ttoken",
      "https://auth.example.test/oauth\r\ntoken",
      "https://auth.example.test/oauth\u00a0token",
      "https://auth.example.test/%",
      "https://auth.example.test/%zz",
      "https://auth.example.test/%0G",
      "ftp://auth.example.test/oauth/token",
      "https://user:secret@auth.example.test/oauth/token",
      "https://auth.example.test/oauth/token?redirect=elsewhere",
      "https://auth.example.test/oauth/token#fragment",
    ]) {
      withRefreshUrl(endpoint, () => {
        assert.throws(validateAuthEnvironment, ChatGPTOAuthRefreshError);
      });
    }
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

  it("terminates when a secret is part of the redaction marker", () => {
    assert.equal(redactText("*** token *", "*"), " token ");
    assert.equal(redactText("*** token ***", "***"), " token ");
    const boundarySafe = redactText("ab", "a*", "b");
    assert.equal(boundarySafe.includes("a*"), false);
    assert.equal(boundarySafe.includes("b"), false);
  });
});

describe("loadTokenData", () => {
  it("throws on missing file", () => {
    const secretPath = "/nonexistent/path/auth.json";
    assert.throws(
      () => loadTokenData(secretPath),
      (error: unknown) => error instanceof ChatGPTOAuthMissingError
        && error.message === "ChatGPT OAuth auth file not found"
        && !error.message.includes(secretPath),
    );
  });

  it("classifies non-ENOENT auth reads as unavailable without exposing the path", () => {
    const secretPath = "/private/secret/auth.json";
    const restoreRead = replaceBuiltinFsMethod("readFileSync", () => {
      const error = new Error(`EACCES: permission denied, open '${secretPath}'`) as NodeJS.ErrnoException;
      error.code = "EACCES";
      throw error;
    });
    try {
      assert.throws(
        () => loadTokenData(secretPath),
        (error: unknown) => error instanceof ChatGPTOAuthMissingError
          && error.message === "ChatGPT OAuth auth file is unavailable"
          && !error.message.includes(secretPath),
      );
    } finally {
      restoreRead();
    }
  });

  it("throws on invalid JSON", () => {
    const dir = fs.mkdtempSync(
      path.join(os.tmpdir(), "auth-test-"),
    );
    const filePath = path.join(dir, "auth.json");
    fs.writeFileSync(filePath, "not json{");
    assert.throws(
      () => loadTokenData(filePath),
      (error: unknown) => error instanceof ChatGPTOAuthError
        && error.message === "ChatGPT OAuth auth file is invalid JSON"
        && !error.message.includes(filePath),
    );
    fs.rmSync(dir, { recursive: true });
  });

  it("rejects invalid UTF-8 auth data without exposing bytes or paths", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-utf8-test-"));
    const filePath = path.join(dir, "auth.json");
    fs.writeFileSync(filePath, Buffer.from([0x7b, 0xff, 0x7d]));
    assert.throws(
      () => loadTokenData(filePath),
      (error: unknown) => error instanceof ChatGPTOAuthError
        && error.message === "ChatGPT OAuth auth file is not valid UTF-8"
        && !error.message.includes(filePath)
        && !error.message.includes("�"),
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

  it("rejects unofficial root-level token fields", () => {
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

    assert.throws(
      () => loadTokenData(filePath),
      (error) => error instanceof ChatGPTOAuthError
        && error.message.includes("file-backed ChatGPT OAuth tokens are required"),
    );
    fs.rmSync(dir, { recursive: true });
  });

  it("rejects root credential fields mixed with managed nested tokens", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-mixed-test-"));
    const claims = {
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-mixed" },
    };
    const tokens = {
      access_token: makeJwt({ exp: 9_999_999_999, ...claims }),
      refresh_token: "refresh-nested",
      id_token: makeJwt(claims),
    };
    try {
      for (const [name, value] of [
        ["access_token", "root-access"],
        ["refresh_token", "root-refresh"],
        ["id_token", "root-id"],
        ["chatgptAuthTokens", { accessToken: "external" }],
      ] as const) {
        const filePath = writeAuthJson(dir, { auth_mode: "chatgpt", tokens, [name]: value });
        assert.throws(
          () => loadTokenData(filePath),
          (error) => error instanceof ChatGPTOAuthError
            && error.message === "ChatGPT OAuth auth file mixes unsupported root credential fields with managed tokens",
        );
      }
    } finally {
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("rejects non-finite, unsafe-integer, and malformed-Unicode auth JSON", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-strict-json-test-"));
    const filePath = path.join(dir, "auth.json");
    try {
      for (const raw of [
        '{"tokens":{},"extra":1e400}',
        '{"tokens":{},"extra":9007199254740993}',
        '{"tokens":{},"extra":"\\ud800"}',
        '{"tokens":{},"\\udfff":true}',
      ]) {
        fs.writeFileSync(filePath, raw);
        assert.throws(
          () => loadTokenData(filePath),
          (error) => error instanceof ChatGPTOAuthError
            && error.message === "ChatGPT OAuth auth file is invalid JSON",
        );
      }
    } finally {
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("rejects an explicit null tokens field instead of falling back to root tokens", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-test-"));
    const claims = {
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-root" },
    };
    const filePath = writeAuthJson(dir, {
      tokens: null,
      access_token: makeJwt({ exp: 9_999_999_999, ...claims }),
      refresh_token: "refresh-root",
      id_token: makeJwt(claims),
    });

    assert.throws(
      () => loadTokenData(filePath),
      /auth file tokens must be an object/,
    );
    fs.rmSync(dir, { recursive: true });
  });

  it("refuses to write refreshed credentials into an unofficial root token layout", async () => {
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
      tokens: {
        access_token: oldToken,
        refresh_token: "refresh-old",
        id_token: oldToken,
      },
    });
    const observed = loadTokenData(filePath);
    const rootLayout = JSON.stringify({
      access_token: oldToken,
      refresh_token: "refresh-old",
      id_token: oldToken,
    });
    const originalRead = fsDefault.readFileSync as unknown as (...args: unknown[]) => unknown;
    let reads = 0;
    const restoreRead = replaceBuiltinFsMethod("readFileSync", (...args: unknown[]) => {
      reads++;
      if (reads === 3) return Buffer.from(rootLayout);
      return originalRead(...args);
    });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({
      access_token: newToken,
      refresh_token: "refresh-new",
      id_token: newToken,
    }), { status: 200, headers: { "Content-Type": "application/json" } });
    try {
      await assert.rejects(
        refreshToken(filePath, observed),
        (error) => error instanceof Error
          && error.message === "ChatGPT OAuth auth file tokens must be an object",
      );
      const stored = JSON.parse(fs.readFileSync(filePath, "utf-8")) as {
        tokens: Record<string, unknown>;
      };
      assert.equal(stored.tokens.access_token, oldToken);
      assert.equal(stored.tokens.refresh_token, "refresh-old");
      assert.equal(stored.tokens.id_token, oldToken);
    } finally {
      globalThis.fetch = previousFetch;
      restoreRead();
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

  it("rejects malformed access and ID JWTs without reflecting their values", () => {
    const validAccess = makeJwt({ exp: 9_999_999_999 });
    const validId = makeJwt({
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-jwt" },
    });
    const [header, payload, signature] = validAccess.split(".") as [string, string, string];
    const invalidUtf8 = Buffer.from([0xff]).toString("base64url");
    const malformed = [
      "access_token=JWT_SENTINEL",
      `${header}.${payload}`,
      `${header}..${signature}`,
      `${header}.${payload}.`,
      `${header}.${payload}.${signature}.extra`,
      `${header}.%zz.${signature}`,
      `${header}.${invalidUtf8}.${signature}`,
    ];

    for (const [field, value] of malformed.flatMap((token) => [
      ["access_token", token] as const,
      ["id_token", token] as const,
    ])) {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-jwt-test-"));
      const filePath = writeAuthJson(dir, {
        tokens: {
          access_token: field === "access_token" ? value : validAccess,
          refresh_token: "refresh-jwt",
          id_token: field === "id_token" ? value : validId,
        },
      });
      assert.throws(
        () => loadTokenData(filePath),
        (error) => error instanceof ChatGPTOAuthError
          && !error.message.includes(value),
      );
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("allows a structurally valid access JWT without an exp claim", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-jwt-test-"));
    const filePath = writeAuthJson(dir, {
      tokens: {
        access_token: makeJwt({}),
        refresh_token: "refresh-jwt",
        id_token: makeJwt({
          "https://api.openai.com/auth": { chatgpt_account_id: "acct-no-exp" },
        }),
      },
    });

    assert.equal(loadTokenData(filePath).access_expires_at, null);
    fs.rmSync(dir, { recursive: true });
  });

  it("accepts only managed or absent auth_mode values", () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-mode-test-"));
    const idToken = makeJwt({
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-mode" },
    });
    const tokens = {
      access_token: makeJwt({ exp: 9_999_999_999 }),
      refresh_token: "refresh-mode",
      id_token: idToken,
    };
    try {
      for (const mode of [undefined, null, "chatgpt"] as const) {
        const filePath = writeAuthJson(dir, {
          ...(mode === undefined ? {} : { auth_mode: mode }),
          tokens,
        });
        assert.equal(loadTokenData(filePath).account_id, "acct-mode");
      }
      for (const mode of [
        "chatgptAuthTokens",
        "Chatgpt",
        "ChatgptAuthTokens",
        "chatgpt_auth_tokens",
      ]) {
        const filePath = writeAuthJson(dir, { auth_mode: mode, tokens });
        assert.throws(
          () => loadTokenData(filePath),
          (error) => error instanceof ChatGPTOAuthError
            && error.message === "ChatGPT OAuth auth_mode is unsupported",
        );
      }
      const externalPath = writeAuthJson(dir, {
        auth_mode: "chatgptAuthTokens",
        tokens: { ...tokens, refresh_token: "" },
      });
      assert.throws(
        () => loadTokenData(externalPath),
        (error) => error instanceof ChatGPTOAuthError
          && error.message === "ChatGPT OAuth auth_mode is unsupported",
      );
    } finally {
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("rejects external auth before managed refresh transport or persistence", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-mode-test-"));
    const filePath = writeAuthJson(dir, {
      auth_mode: "chatgptAuthTokens",
      tokens: {
        access_token: makeJwt({ exp: 1 }),
        refresh_token: "",
        id_token: makeJwt({
          "https://api.openai.com/auth": { chatgpt_account_id: "acct-external" },
        }),
      },
    });
    const original = fs.readFileSync(filePath, "utf-8");
    const previousFetch = globalThis.fetch;
    let refreshCalls = 0;
    globalThis.fetch = async () => {
      refreshCalls++;
      throw new Error("external auth must not use managed refresh transport");
    };
    try {
      await assert.rejects(
        refreshToken(filePath),
        (error) => error instanceof ChatGPTOAuthError
          && error.message === "ChatGPT OAuth auth_mode is unsupported",
      );
      assert.equal(refreshCalls, 0);
      assert.equal(fs.readFileSync(filePath, "utf-8"), original);
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("rejects header-unsafe access tokens and account IDs without echoing secrets", () => {
    const unsafeValues = [
      "LEAKME\r\nInjected: yes",
      "LEAKME\0suffix",
      "LEAKME with-space",
      "LEAKME-\u00e9",
    ];
    for (const [field, unsafe] of unsafeValues.flatMap((value) => [
      ["access_token", value] as const,
      ["account_id", value] as const,
    ])) {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-header-test-"));
      const accountId = field === "account_id" ? unsafe : "acct-safe";
      const filePath = writeAuthJson(dir, {
        tokens: {
          access_token: field === "access_token"
            ? unsafe
            : makeJwt({ exp: 9_999_999_999 }),
          refresh_token: "refresh-safe",
          id_token: makeJwt({
            "https://api.openai.com/auth": { chatgpt_account_id: accountId },
          }),
        },
      });

      assert.throws(
        () => loadTokenData(filePath),
        (error) => error instanceof ChatGPTOAuthError
          && error.message.includes("invalid for an HTTP header")
          && !error.message.includes("LEAKME"),
      );
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("throws on invalid auth_mode", () => {
    const dir = fs.mkdtempSync(
      path.join(os.tmpdir(), "auth-test-"),
    );
    const credentialSentinel = "access_token=AUTH_MODE_SENTINEL";
    const filePath = writeAuthJson(dir, {
      auth_mode: credentialSentinel,
      tokens: {
        access_token: makeJwt(),
        refresh_token: "r",
        id_token: makeJwt(),
      },
    });
    assert.throws(
      () => loadTokenData(filePath),
      (error) => error instanceof ChatGPTOAuthError
        && error.message === "ChatGPT OAuth auth_mode is unsupported"
        && !error.message.includes(credentialSentinel),
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

  it("rejects malformed JWT claim types instead of treating them as absent", () => {
    const cases = [
      {
        access: makeJwt({ exp: "9999999999" }),
        id: makeJwt({
          "https://api.openai.com/auth": { chatgpt_account_id: "acct-invalid" },
        }),
      },
      {
        access: makeJwt({
          exp: 9_999_999_999,
          "https://api.openai.com/auth": "acct-invalid",
        }),
        id: makeJwt({
          "https://api.openai.com/auth": { chatgpt_account_id: "acct-invalid" },
        }),
      },
      {
        access: makeJwt({
          exp: 9_999_999_999,
          "https://api.openai.com/auth": null,
        }),
        id: makeJwt({
          "https://api.openai.com/auth": { chatgpt_account_id: "acct-invalid" },
        }),
      },
    ];

    for (const tokens of cases) {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-test-"));
      const filePath = writeAuthJson(dir, {
        auth_mode: "chatgpt",
        tokens: {
          access_token: tokens.access,
          refresh_token: "r",
          id_token: tokens.id,
        },
      });
      assert.throws(() => loadTokenData(filePath), ChatGPTOAuthError);
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("rejects malformed or conflicting FedRAMP claims instead of coercing them", () => {
    const cases = [
      {
        id: { chatgpt_account_id: "acct-fed", chatgpt_account_is_fedramp: "false" },
        access: { chatgpt_account_id: "acct-fed" },
      },
      {
        id: { chatgpt_account_id: "acct-fed", chatgpt_account_is_fedramp: null },
        access: { chatgpt_account_id: "acct-fed" },
      },
      {
        id: { chatgpt_account_id: "acct-fed", chatgpt_account_is_fedramp: true },
        access: { chatgpt_account_id: "acct-fed", chatgpt_account_is_fedramp: false },
      },
    ];
    for (const claims of cases) {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-test-"));
      const filePath = writeAuthJson(dir, {
        auth_mode: "chatgpt",
        tokens: {
          access_token: makeJwt({
            exp: 9999999999,
            "https://api.openai.com/auth": claims.access,
          }),
          refresh_token: "r",
          id_token: makeJwt({ "https://api.openai.com/auth": claims.id }),
        },
      });
      assert.throws(() => loadTokenData(filePath), ChatGPTOAuthError);
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("rejects malformed or conflicting plan and user claims instead of dropping them", () => {
    const cases = [
      {
        id: { chatgpt_account_id: "acct-meta", chatgpt_plan_type: null },
        access: { chatgpt_account_id: "acct-meta", chatgpt_plan_type: "plus" },
      },
      {
        id: { chatgpt_account_id: "acct-meta", chatgpt_user_id: { id: "bad" } },
        access: { chatgpt_account_id: "acct-meta", chatgpt_user_id: "user-1" },
      },
      {
        id: { chatgpt_account_id: "acct-meta", chatgpt_plan_type: "plus" },
        access: { chatgpt_account_id: "acct-meta", chatgpt_plan_type: "pro" },
      },
      {
        id: { chatgpt_account_id: "acct-meta", chatgpt_user_id: "user-1" },
        access: { chatgpt_account_id: "acct-meta", user_id: "user-2" },
      },
    ];
    for (const claims of cases) {
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-test-"));
      const filePath = writeAuthJson(dir, {
        auth_mode: "chatgpt",
        tokens: {
          access_token: makeJwt({
            exp: 9_999_999_999,
            "https://api.openai.com/auth": claims.access,
          }),
          refresh_token: "r",
          id_token: makeJwt({ "https://api.openai.com/auth": claims.id }),
        },
      });
      assert.throws(() => loadTokenData(filePath), ChatGPTOAuthError);
      fs.rmSync(dir, { recursive: true });
    }
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
  it("does not follow refresh redirects or persist a redirected response", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-refresh-redirect-test-"));
    const claims = {
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-redirect" },
    };
    const original = {
      access_token: makeJwt({ exp: 1, ...claims }),
      refresh_token: "refresh-must-not-leak",
      id_token: makeJwt(claims),
    };
    const filePath = writeAuthJson(dir, { tokens: original });
    const previousFetch = globalThis.fetch;
    let calls = 0;
    globalThis.fetch = async (_input, init) => {
      calls++;
      assert.equal(init?.redirect, "manual");
      return new Response("redirect refused", {
        status: 307,
        headers: { Location: "https://attacker.example/steal" },
      });
    };
    try {
      await assert.rejects(
        refreshToken(filePath),
        (error: unknown) => error instanceof ChatGPTOAuthUpstreamError
          && error.status === 307
          && error.message.includes("HTTP 307")
          && !error.message.includes(original.refresh_token),
      );
      assert.equal(calls, 1);
      const stored = JSON.parse(fs.readFileSync(filePath, "utf-8")) as {
        tokens: Record<string, unknown>;
      };
      assert.deepEqual(stored.tokens, original);
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("does not expose reflected credential prefixes from malformed refresh JSON", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-refresh-json-test-"));
    const claims = {
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-json" },
    };
    const accessToken = makeJwt({ exp: 1, ...claims });
    const filePath = writeAuthJson(dir, {
      tokens: {
        access_token: accessToken,
        refresh_token: "refresh-json-secret",
        id_token: makeJwt(claims),
      },
    });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(`${accessToken}-NOT-JSON`, {
      status: 200,
      headers: { "content-type": "application/json" },
    });
    try {
      await assert.rejects(
        refreshToken(filePath),
        (error) => error instanceof ChatGPTOAuthProtocolError
          && error.message === "ChatGPT OAuth token refresh returned invalid JSON"
          && !error.message.includes(accessToken.slice(0, 8)),
      );
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

  it("rejects invalid UTF-8 in a successful refresh response", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-refresh-utf8-test-"));
    const claims = {
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-utf8" },
    };
    const filePath = writeAuthJson(dir, {
      tokens: {
        access_token: makeJwt({ exp: 1, ...claims }),
        refresh_token: "refresh-utf8-secret",
        id_token: makeJwt(claims),
      },
    });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(
      Uint8Array.from([0x7b, 0xff, 0x7d]).buffer,
      { status: 200, headers: { "content-type": "application/json" } },
    );
    try {
      await assert.rejects(
        refreshToken(filePath),
        (error) => error instanceof ChatGPTOAuthProtocolError
          && error.message === "ChatGPT OAuth token refresh returned invalid JSON"
          && !error.message.includes("refresh-utf8-secret")
          && !error.message.includes("�"),
      );
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true, force: true });
    }
  });

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

  it("does not return an in-flight refresh across observed accounts", async () => {
    const dirA = fs.mkdtempSync(path.join(os.tmpdir(), "auth-flight-a-test-"));
    const dirB = fs.mkdtempSync(path.join(os.tmpdir(), "auth-flight-b-test-"));
    const claimsA = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-flight-a" } };
    const claimsB = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-flight-b" } };
    const fileA = writeAuthJson(dirA, { tokens: {
      access_token: makeJwt({ exp: 1, ...claimsA }),
      refresh_token: "refresh-a",
      id_token: makeJwt(claimsA),
    } });
    const fileB = writeAuthJson(dirB, { tokens: {
      access_token: makeJwt({ exp: 1, ...claimsB }),
      refresh_token: "refresh-b",
      id_token: makeJwt(claimsB),
    } });
    const observedA = loadTokenData(fileA);
    const observedB = { ...loadTokenData(fileB), auth_path: fileA };
    const refreshedA = makeJwt({ exp: 9_999_999_999, ...claimsA });
    let releaseFetch!: () => void;
    const fetchGate = new Promise<void>((resolve) => {
      releaseFetch = resolve;
    });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => {
      await fetchGate;
      return new Response(JSON.stringify({ access_token: refreshedA }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    };
    try {
      const first = refreshToken(fileA, observedA);
      const crossAccountJoin = refreshToken(fileA, observedB);
      const crossAccountRejection = assert.rejects(crossAccountJoin, /account changed/);
      releaseFetch();
      assert.equal((await first).access_token, refreshedA);
      await crossAccountRejection;
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dirA, { recursive: true });
      fs.rmSync(dirB, { recursive: true });
    }
  });

  it("rejects a header-unsafe refreshed access token before persisting it", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-refresh-header-test-"));
    const claims = {
      "https://api.openai.com/auth": { chatgpt_account_id: "acct-safe" },
    };
    const original = {
      access_token: makeJwt({ exp: 1, ...claims }),
      refresh_token: "refresh-safe",
      id_token: makeJwt(claims),
    };
    const filePath = writeAuthJson(dir, { tokens: original });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({
      access_token: "LEAKME\r\nInjected: yes",
    }), { status: 200, headers: { "Content-Type": "application/json" } });
    try {
      await assert.rejects(
        refreshToken(filePath),
        (error) => error instanceof ChatGPTOAuthProtocolError
          && error.message.includes("invalid for an HTTP header")
          && !error.message.includes("LEAKME"),
      );
      const stored = JSON.parse(fs.readFileSync(filePath, "utf-8")) as {
        tokens: Record<string, unknown>;
      };
      assert.deepEqual(stored.tokens, original);
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
    globalThis.fetch = async () => {
      fs.writeFileSync(filePath, JSON.stringify({
        tokens: { ...current, access_token: externalAccess },
      }));
      return new Response(JSON.stringify({ access_token: responseAccess }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    };
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
    globalThis.fetch = async () => {
      fs.writeFileSync(filePath, JSON.stringify({
        tokens: { ...current, refresh_token: "refresh-raced" },
      }));
      return new Response(JSON.stringify({
        access_token: makeJwt({ exp: 9_999_999_999, ...claims }),
      }), {
        status: 200,
        headers: { "content-type": "application/json" },
      });
    };
    try {
      await assert.rejects(refreshToken(filePath), /changed while token refresh was in flight/);
    } finally {
      globalThis.fetch = previousFetch;
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("does not overwrite an account switch in the exact document selected for persistence", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-cas-account-test-"));
    const claimsA = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-cas-a" } };
    const claimsB = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-cas-b" } };
    const credentialsA = {
      access_token: makeJwt({ exp: 1, ...claimsA }),
      refresh_token: "refresh-cas-a",
      id_token: makeJwt(claimsA),
    };
    const documentA = { tokens: credentialsA, last_refresh: "before" };
    const documentB = { tokens: {
      access_token: makeJwt({ exp: 9_999_999_999, ...claimsB }),
      refresh_token: "refresh-cas-b",
      id_token: makeJwt(claimsB),
    }, last_refresh: "account-b" };
    const filePath = writeAuthJson(dir, documentA);
    const observed = loadTokenData(filePath);
    const originalRead = fsDefault.readFileSync as unknown as (...args: unknown[]) => unknown;
    let reads = 0;
    const restoreRead = replaceBuiltinFsMethod("readFileSync", (...args: unknown[]) => {
      reads++;
      if (reads === 3) return Buffer.from(JSON.stringify(documentB));
      return originalRead(...args);
    });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({
      access_token: makeJwt({ exp: 9_999_999_999, ...claimsA }),
    }), { status: 200, headers: { "content-type": "application/json" } });
    try {
      await assert.rejects(refreshToken(filePath, observed), /account changed/);
    } finally {
      globalThis.fetch = previousFetch;
      restoreRead();
    }
    assert.deepEqual(JSON.parse(fs.readFileSync(filePath, "utf-8")), documentA);
    fs.rmSync(dir, { recursive: true });
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

  it("does not expose the auth path when the post-refresh re-read fails", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-reread-test-"));
    const claims = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-reread" } };
    const filePath = writeAuthJson(dir, { tokens: {
      access_token: makeJwt({ exp: 1, ...claims }),
      refresh_token: "refresh-old",
      id_token: makeJwt(claims),
    } });
    const observed = loadTokenData(filePath);
    const originalRead = fsDefault.readFileSync as unknown as (...args: unknown[]) => unknown;
    let reads = 0;
    const restoreRead = replaceBuiltinFsMethod("readFileSync", (...args: unknown[]) => {
      reads++;
      if (reads === 3) {
        const error = new Error(`EACCES: permission denied, open '${filePath}'`) as NodeJS.ErrnoException;
        error.code = "EACCES";
        error.path = filePath;
        throw error;
      }
      return originalRead(...args);
    });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({
      access_token: makeJwt({ exp: 9_999_999_999, ...claims }),
    }), { status: 200, headers: { "Content-Type": "application/json" } });
    try {
      await assert.rejects(
        refreshToken(filePath, observed),
        (error) => error instanceof Error
          && error.message === "failed to re-read ChatGPT OAuth auth file"
          && !error.message.includes(filePath),
      );
    } finally {
      globalThis.fetch = previousFetch;
      restoreRead();
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("closes the temporary auth file descriptor when writing fails", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-write-close-test-"));
    const claims = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-write" } };
    const filePath = writeAuthJson(dir, { tokens: {
      access_token: makeJwt({ exp: 1, ...claims }),
      refresh_token: "refresh-old",
      id_token: makeJwt(claims),
    } });
    const writeFailure = new Error("simulated auth write failure");
    const originalClose = fsDefault.closeSync;
    let closed = false;
    const restoreClose = replaceBuiltinFsMethod("closeSync", (fd: number) => {
      closed = true;
      originalClose(fd);
    });
    const restoreWrite = replaceBuiltinFsMethod("writeFileSync", () => {
      throw writeFailure;
    });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({
      access_token: makeJwt({ exp: 9_999_999_999, ...claims }),
    }), { status: 200, headers: { "Content-Type": "application/json" } });
    try {
      await assert.rejects(refreshToken(filePath), (error: unknown) => error === writeFailure);
      assert.equal(closed, true);
      assert.deepEqual(
        fs.readdirSync(dir).filter((name) => name.includes(".tmp-")),
        [],
      );
    } finally {
      globalThis.fetch = previousFetch;
      restoreWrite();
      restoreClose();
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("preserves the primary auth write failure when temporary cleanup also fails", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-write-cleanup-test-"));
    const claims = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-cleanup" } };
    const filePath = writeAuthJson(dir, { tokens: {
      access_token: makeJwt({ exp: 1, ...claims }),
      refresh_token: "refresh-old",
      id_token: makeJwt(claims),
    } });
    const writeFailure = new Error("simulated auth write failure");
    const cleanupFailure = new Error("simulated auth cleanup failure");
    const restoreWrite = replaceBuiltinFsMethod("writeFileSync", () => {
      throw writeFailure;
    });
    const restoreUnlink = replaceBuiltinFsMethod("unlinkSync", () => {
      throw cleanupFailure;
    });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({
      access_token: makeJwt({ exp: 9_999_999_999, ...claims }),
    }), { status: 200, headers: { "Content-Type": "application/json" } });
    try {
      await assert.rejects(
        refreshToken(filePath),
        (error: unknown) => error instanceof AggregateError
          && error.errors[0] === writeFailure
          && error.errors[1] === cleanupFailure,
      );
    } finally {
      globalThis.fetch = previousFetch;
      restoreUnlink();
      restoreWrite();
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("preserves a directory fsync failure when closing the directory also fails", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-dir-sync-test-"));
    const claims = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-dir-sync" } };
    const filePath = writeAuthJson(dir, { tokens: {
      access_token: makeJwt({ exp: 1, ...claims }),
      refresh_token: "refresh-old",
      id_token: makeJwt(claims),
    } });
    const syncFailure = Object.assign(
      new Error("simulated unsupported auth directory fsync failure"),
      { code: "EINVAL" },
    );
    const closeFailure = new Error("simulated auth directory close failure");
    const originalOpen = fsDefault.openSync;
    const originalFsync = fsDefault.fsyncSync;
    const originalClose = fsDefault.closeSync;
    let directoryFd: number | undefined;
    const restoreOpen = replaceBuiltinFsMethod("openSync", (...args: Parameters<typeof fsDefault.openSync>) => {
      const fd = originalOpen(...args);
      if (args[0] === dir && args[1] === "r") directoryFd = fd;
      return fd;
    });
    const restoreFsync = replaceBuiltinFsMethod("fsyncSync", (fd: number) => {
      if (fd === directoryFd) throw syncFailure;
      originalFsync(fd);
    });
    const restoreClose = replaceBuiltinFsMethod("closeSync", (fd: number) => {
      originalClose(fd);
      if (fd === directoryFd) throw closeFailure;
    });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({
      access_token: makeJwt({ exp: 9_999_999_999, ...claims }),
    }), { status: 200, headers: { "Content-Type": "application/json" } });
    try {
      await assert.rejects(
        refreshToken(filePath),
        (error: unknown) => error instanceof AggregateError
          && error.errors[0] === syncFailure
          && error.errors[1] === closeFailure,
      );
      assert.notEqual(directoryFd, undefined);
    } finally {
      globalThis.fetch = previousFetch;
      restoreClose();
      restoreFsync();
      restoreOpen();
      fs.rmSync(dir, { recursive: true });
    }
  });

  it("skips unsupported directory fsync explicitly on Windows", async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "auth-windows-sync-test-"));
    const claims = { "https://api.openai.com/auth": { chatgpt_account_id: "acct-windows-sync" } };
    const filePath = writeAuthJson(dir, { tokens: {
      access_token: makeJwt({ exp: 1, ...claims }),
      refresh_token: "refresh-old",
      id_token: makeJwt(claims),
    } });
    const originalOpen = fsDefault.openSync;
    let openedDirectory = false;
    const restoreOpen = replaceBuiltinFsMethod("openSync", (...args: Parameters<typeof fsDefault.openSync>) => {
      if (args[0] === dir && args[1] === "r") {
        openedDirectory = true;
        throw new Error("Windows directory fsync must not be attempted");
      }
      return originalOpen(...args);
    });
    const platformDescriptor = Object.getOwnPropertyDescriptor(process, "platform");
    Object.defineProperty(process, "platform", { ...platformDescriptor, value: "win32" });
    const previousFetch = globalThis.fetch;
    globalThis.fetch = async () => new Response(JSON.stringify({
      access_token: makeJwt({ exp: 9_999_999_999, ...claims }),
    }), { status: 200, headers: { "Content-Type": "application/json" } });
    try {
      assert.equal((await refreshToken(filePath)).account_id, "acct-windows-sync");
      assert.equal(openedDirectory, false);
    } finally {
      globalThis.fetch = previousFetch;
      if (platformDescriptor != null) {
        Object.defineProperty(process, "platform", platformDescriptor);
      }
      restoreOpen();
      fs.rmSync(dir, { recursive: true });
    }
  });
});
