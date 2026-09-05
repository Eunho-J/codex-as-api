import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import * as crypto from "node:crypto";
import * as fs from "node:fs";
import { createServer, request as httpRequest, type IncomingHttpHeaders, type Server } from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import upstreamContract from "../../../config/codex-upstream-contract.json";
import { createApp, resolveServerHost, resolveServerPort } from "../server.js";
import {
  ChatGPTOAuthCatalogUnavailableError,
  ChatGPTOAuthInvalidRequestError,
  ChatGPTOAuthMissingError,
  ChatGPTOAuthModelNotFoundError,
  ChatGPTOAuthProtocolError,
  ChatGPTOAuthRefreshError,
  ChatGPTOAuthUnavailableError,
  ChatGPTOAuthUpstreamError,
} from "../auth.js";
import {
  ChatGPTOAuthProvider,
  MODEL_CATALOG_TIMEOUT_MS,
  messagesToResponseItems,
  type PreparedModel,
} from "../provider.js";
import type { CodexConfig } from "../codex-config.js";
import { MessageRole } from "../messages.js";
import type { ToolSchema } from "../messages.js";
import {
  DEFAULT_MODEL_CATALOG_TTL_MS,
  ModelCatalogCache,
  modelCatalogCacheKey,
  modelFromSnapshot,
  parseModelCatalog,
  type ModelCapability,
  type ModelCatalogSnapshot,
} from "../model-capabilities.js";

const TEST_CONFIG: CodexConfig = {
  codexHome: "/test/codex-home",
  configPath: "/test/codex-home/config.toml",
};

function rawModel(
  slug: string,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    slug,
    display_name: slug,
    description: null,
    default_reasoning_level: "low",
    supported_reasoning_levels: [
      { effort: "none", description: "None" },
      { effort: "low", description: "Low" },
      { effort: "medium", description: "Medium" },
      { effort: "high", description: "High" },
      { effort: "xhigh", description: "Extra high" },
      { effort: "max", description: "Maximum" },
    ],
    multi_agent_reasoning_effort: "max",
    priority: 10,
    visibility: "list",
    supported_in_api: true,
    service_tiers: [
      { id: "priority", name: "Priority", description: "Priority tier" },
    ],
    default_service_tier: null,
    support_verbosity: true,
    default_verbosity: "low",
    supports_image_detail_original: true,
    context_window: 272_000,
    max_context_window: 272_000,
    auto_compact_token_limit: null,
    input_modalities: ["text", "image"],
    use_responses_lite: true,
    ...overrides,
  };
}

const TEST_CATALOG_VALUE = {
  models: [
    rawModel("gpt-5.5", { priority: 20, use_responses_lite: false }),
    rawModel("gpt-5.6-sol"),
    rawModel("gpt-5.6-terra", { priority: 30 }),
    rawModel("catalog-only", {
      priority: 40,
      supported_in_api: false,
      visibility: "list",
      comp_hash: "compatibility-family",
    }),
  ],
};

function testCatalogSnapshot(): ModelCatalogSnapshot {
  return parseModelCatalog(TEST_CATALOG_VALUE, {
    key: modelCatalogCacheKey("test-account", "https://catalog.test", "0.153.3"),
    etag: '"test-catalog"',
    fetchedAt: 1_000,
    expiresAt: 301_000,
  });
}

function useTestCatalog(provider: ChatGPTOAuthProvider): ChatGPTOAuthProvider {
  const snapshot = testCatalogSnapshot();
  provider.prepareModel = async (requested?: string): Promise<PreparedModel> => {
    const slug = requested ?? snapshot.defaultModel?.slug;
    if (slug == null) {
      throw new ChatGPTOAuthCatalogUnavailableError(
        "authenticated model catalog has no visible default model",
      );
    }
    const capability = modelFromSnapshot(snapshot, slug);
    if (capability == null) throw new ChatGPTOAuthModelNotFoundError(slug);
    return { slug, accountId: "test-account", capability, snapshot };
  };
  return provider;
}

function adaptMockProvider(
  provider: Record<string, unknown>,
  configuredModel: string | undefined,
): Record<string, unknown> {
  const adapted = provider;
  const snapshot = testCatalogSnapshot();
  if (typeof adapted.catalogSnapshot !== "function") {
    adapted.catalogSnapshot = async () => snapshot;
  }
  if (typeof adapted.prepareModel !== "function") {
    adapted.prepareModel = async (requested?: string): Promise<PreparedModel> => {
      const selected = requested ?? configuredModel ?? snapshot.defaultModel?.slug;
      if (selected == null) {
        throw new ChatGPTOAuthCatalogUnavailableError("authenticated model catalog has no visible default model");
      }
      const slug = selected;
      const capability = modelFromSnapshot(snapshot, slug);
      if (capability == null) throw new ChatGPTOAuthModelNotFoundError(slug);
      return { slug, accountId: "test-account", capability, snapshot };
    };
  }
  if (typeof adapted.prepareAnthropicModel !== "function") {
    adapted.prepareAnthropicModel = async (clientModel?: string): Promise<PreparedModel> => {
      if (clientModel == null || !clientModel.startsWith("claude-")) {
        throw new ChatGPTOAuthInvalidRequestError(
          "Anthropic requests require an explicit claude-* model facade",
        );
      }
      if (configuredModel == null) {
        throw new ChatGPTOAuthInvalidRequestError(
          "Claude facade backend model is not configured",
        );
      }
      return await (adapted.prepareModel as (model: string) => Promise<PreparedModel>)(configuredModel);
    };
  }
  if (typeof adapted.createChatStream !== "function" && typeof adapted.chatStream === "function") {
    const chatStream = adapted.chatStream as (...args: unknown[]) => AsyncIterable<Record<string, unknown>>;
    adapted.createChatStream = async (...args: unknown[]) => chatStream.apply(adapted, args);
  }
  return adapted;
}

function hasNestedKey(value: unknown, key: string): boolean {
  if (Array.isArray(value)) return value.some((item) => hasNestedKey(item, key));
  if (typeof value !== "object" || value === null) return false;
  const record = value as Record<string, unknown>;
  return Object.hasOwn(record, key)
    || Object.values(record).some((item) => hasNestedKey(item, key));
}

async function withServer(
  provider: ChatGPTOAuthProvider | Record<string, unknown>,
  fn: (baseUrl: string) => Promise<void>,
  opts: { model?: string | null; codexConfig?: CodexConfig; authPath?: string } = {},
): Promise<void> {
  const configuredModel = opts.model === null ? undefined : opts.model ?? "gpt-5.5";
  const appProvider = provider instanceof ChatGPTOAuthProvider
    ? provider
    : adaptMockProvider(provider, configuredModel);
  const app = createApp(opts.model === null
    ? {
        provider: appProvider as never,
        codexConfig: opts.codexConfig ?? TEST_CONFIG,
        authPath: opts.authPath,
      }
    : {
        provider: appProvider as never,
        model: configuredModel,
        codexConfig: opts.codexConfig ?? TEST_CONFIG,
        authPath: opts.authPath,
      });
  const server = await new Promise<Server>((resolve) => {
    const listening = app.listen(0, "127.0.0.1", () => resolve(listening));
  });
  try {
    const address = server.address();
    assert.ok(address && typeof address === "object");
    await fn(`http://127.0.0.1:${address.port}`);
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    });
  }
}

function makeJwt(payload: Record<string, unknown>): string {
  const header = Buffer.from(JSON.stringify({ alg: "HS256", typ: "JWT" })).toString("base64url");
  const body = Buffer.from(JSON.stringify(payload)).toString("base64url");
  return `${header}.${body}.sig`;
}

function writeAuthFile(): { authPath: string; directory: string } {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-server-"));
  const authPath = path.join(directory, "auth.json");
  fs.writeFileSync(authPath, JSON.stringify({
    tokens: {
      access_token: makeJwt({ exp: 9_999_999_999 }),
      refresh_token: "refresh-token",
      id_token: makeJwt({
        exp: 9_999_999_999,
        "https://api.openai.com/auth": {
          chatgpt_account_id: "account-123",
          chatgpt_plan_type: "plus",
          chatgpt_user_id: "user-123",
        },
      }),
    },
  }));
  return { authPath, directory };
}

interface RecordedRequest {
  method: string;
  path: string;
  headers: IncomingHttpHeaders;
  body: Record<string, unknown>;
}

interface RecordedCatalogRequest {
  method: string;
  path: string;
  headers: IncomingHttpHeaders;
}

async function withRecordingUpstream(
  fn: (
    baseUrl: string,
    requests: RecordedRequest[],
    catalogRequests: RecordedCatalogRequest[],
  ) => Promise<void>,
): Promise<void> {
  const requests: RecordedRequest[] = [];
  const catalogRequests: RecordedCatalogRequest[] = [];
  const compactOutput = [
    { type: "additional_tools", role: "developer", tools: [] },
    {
      type: "message",
      role: "developer",
      content: [{ type: "input_text", text: "compact-only instructions" }],
    },
    {
      type: "message",
      role: "user",
      content: [{ type: "input_text", text: "<environment_context>stale</environment_context>" }],
    },
    { type: "reasoning", id: "reasoning-1", summary: [] },
    { type: "function_call", call_id: "call-1", name: "lookup", arguments: "{}" },
    {
      type: "message",
      role: "assistant",
      content: [{ type: "output_text", text: "summary" }],
    },
    {
      type: "agent_message",
      author: "agent",
      recipient: "user",
      content: [{ type: "input_text", text: "agent summary" }],
    },
    {
      type: "message",
      role: "user",
      content: [{ type: "input_text", text: "compacted" }],
    },
    { type: "compaction_summary", encrypted_content: "legacy" },
    { type: "context_compaction" },
  ];
  const server = createServer((req, res) => {
    if (
      req.method === upstreamContract.models_request.method
      && req.url?.startsWith(`${upstreamContract.models_request.path}?`)
    ) {
      catalogRequests.push({
        method: req.method,
        path: req.url,
        headers: req.headers,
      });
      res.writeHead(200, {
        "content-type": "application/json",
        [upstreamContract.models_request.etag_header]: '"test-catalog"',
      });
      res.end(JSON.stringify(TEST_CATALOG_VALUE));
      return;
    }
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
      requests.push({ method: req.method ?? "", path: req.url ?? "", headers: req.headers, body });
      if (req.url === "/responses/compact") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({
          output: compactOutput,
        }));
        return;
      }
      const tools = Array.isArray(body.tools)
        ? body.tools as Record<string, unknown>[]
        : [];
      const outputItem = tools.some((tool) => tool.type === "image_generation")
          ? {
            type: "image_generation_call",
            id: "image-1",
            status: "completed",
            result: "data:image/png;base64,RESULT",
          }
        : {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "ok" }],
          };
      res.writeHead(200, {
        "content-type": "text/event-stream",
        [upstreamContract.models_request.responses_etag_header]: '"test-catalog"',
      });
      res.write(`data: ${JSON.stringify({
        type: "response.output_item.done",
        item: outputItem,
      })}\n\n`);
      res.end(`data: ${JSON.stringify({
        type: "response.completed",
        response: {
          id: "response-1",
          end_turn: true,
          output: [],
          usage: {
            input_tokens: 1,
            output_tokens: 1,
            total_tokens: 2,
            input_tokens_details: {
              cached_tokens: 1,
              cache_write_tokens: 3,
            },
          },
        },
      })}\n\n`);
    });
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  try {
    const address = server.address();
    assert.ok(address && typeof address === "object");
    await fn(`http://127.0.0.1:${address.port}`, requests, catalogRequests);
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    });
  }
}

describe("server error handling", () => {
  it("rejects malformed strict JSON request bodies before route handling", async () => {
    let providerCalls = 0;
    await withServer({
      chat: async () => {
        providerCalls++;
        throw new Error("provider must not be called");
      },
    }, async (baseUrl) => {
      const validPrefix = Buffer.from('{"model":"gpt-5.5","messages":[{"role":"user","content":"');
      const validSuffix = Buffer.from('"}]}');
      const bodies = [
        Buffer.concat([validPrefix, Buffer.from([0xff]), validSuffix]),
        Buffer.from('{"model":"gpt-5.5","messages":[],"temperature":1e400}'),
        Buffer.from('{"model":"gpt-5.5","messages":[{"role":"user","content":"\\ud800"}]}'),
        Buffer.from('{"model":"gpt-5.5","messages":[],"metadata":{"limit":9007199254740993}}'),
      ];
      for (const body of bodies) {
        const response = await fetch(`${baseUrl}/v1/chat/completions`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body,
        });
        const error = await response.json() as {
          error: { type: string; code: string };
        };

        assert.equal(response.status, 400);
        assert.deepEqual(error, {
          error: {
            message: "request body must be valid strict JSON",
            type: "invalid_request_error",
            code: "invalid_request_error",
          },
        });
      }
      assert.equal(providerCalls, 0);
    });
  });

  it("rejects empty, role-mismatched, and named Chat message content before provider calls", async () => {
    let providerCalls = 0;
    await withServer({
      chat: async () => {
        providerCalls++;
        throw new Error("provider must not be called");
      },
    }, async (baseUrl) => {
      const invalidMessages = [
        [{ role: "user", content: [{ type: "output_text", text: "wrong" }] }],
        [{ role: "system", content: [{ type: "output_text", text: "wrong" }] }],
        [{ role: "tool", tool_call_id: "call-1", content: [{ type: "output_text", text: "wrong" }] }],
        [{ role: "assistant", content: [{ type: "input_text", text: "wrong" }] }],
        [{ role: "tool", tool_call_id: "call-1", name: "lookup", content: "result" }],
      ];
      for (const messages of invalidMessages) {
        const response = await fetch(`${baseUrl}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ model: "gpt-5.5", messages }),
        });
        assert.equal(response.status, 400);
      }
      assert.equal(providerCalls, 0);
    });
  });

  it("uses Anthropic error envelopes for every Messages route including parser errors", async () => {
    let providerCalls = 0;
    await withServer({
      chat: async () => {
        providerCalls++;
        throw new Error("provider must not be called");
      },
    }, async (baseUrl) => {
      for (const route of [
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/messages/compact",
      ]) {
        for (const body of [
          JSON.stringify({ unsupported: true }),
          "{",
        ]) {
          const response = await fetch(`${baseUrl}${route}`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body,
          });
          assert.equal(response.status, 400);
          const error = await response.json() as {
            type: string;
            error: { type: string };
          };
          assert.equal(error.type, "error");
          assert.equal(error.error.type, "invalid_request_error");
        }
      }
      assert.equal(providerCalls, 0);
    });
  });

  it("requires JSON media types and accepts vendor JSON with parameters", async () => {
    await withServer({}, async (baseUrl) => {
      for (const [route, contentType] of [
        ["/v1/chat/completions", undefined],
        ["/v1/chat/completions", "text/plain"],
        ["/v1/messages", "application-json"],
      ] as const) {
        const headers: Record<string, string> = contentType == null
          ? {}
          : { "content-type": contentType };
        const response = await fetch(`${baseUrl}${route}`, {
          method: "POST",
          headers,
          body: "{}",
        });
        assert.equal(response.status, 415);
        const body = await response.json() as Record<string, unknown>;
        if (route.startsWith("/v1/messages")) {
          assert.equal(body.type, "error");
        } else {
          assert.equal((body.error as Record<string, unknown>).type, "invalid_request_error");
        }
      }

      for (const contentType of [
        "application/vnd.codex+json",
        "Application/JSON; Charset=UTF-8",
      ]) {
        const response = await fetch(`${baseUrl}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": contentType },
          body: "{}",
        });
        assert.equal(response.status, 400);
      }
    });
  });

  it("rejects legacy provider finish reasons on both public facades", async () => {
    const legacyReason = "legacy-secret-reason";
    const nonStreamingProvider = {
      async chat() {
        return {
          content: "done",
          tool_calls: [],
          finish_reason: legacyReason,
          usage: null,
          reasoning_content: null,
          raw: { events: [] },
          response_id: "response-legacy",
        };
      },
    };
    await withServer(nonStreamingProvider, async (baseUrl) => {
      for (const [route, body] of [
        ["/v1/chat/completions", { messages: [{ role: "user", content: "hello" }] }],
        ["/v1/messages", { model: "claude-sonnet-4-6", max_tokens: 64, messages: [{ role: "user", content: "hello" }] }],
      ] as const) {
        const response = await fetch(`${baseUrl}${route}`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "test-session",
          },
          body: JSON.stringify(body),
        });
        assert.equal(response.status, 502);
        const text = await response.text();
        assert.equal(text.includes(legacyReason), false);
      }
    });

    const streamingProvider = {
      async *chatStream() {
        yield { type: "content", text: "done" };
        yield {
          type: "finish",
          response_id: "response-legacy",
          finish_reason: legacyReason,
        };
      },
    };
    await withServer(streamingProvider, async (baseUrl) => {
      for (const [route, body] of [
        ["/v1/chat/completions", { stream: true, messages: [{ role: "user", content: "hello" }] }],
        ["/v1/messages", { model: "claude-sonnet-4-6", max_tokens: 64, stream: true, messages: [{ role: "user", content: "hello" }] }],
      ] as const) {
        const response = await fetch(`${baseUrl}${route}`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "test-session",
          },
          body: JSON.stringify(body),
        });
        assert.equal(response.status, 200);
        const text = await response.text();
        assert.equal(text.includes(legacyReason), false);
        assert.equal(text.includes('"finish_reason":"length"'), false);
        assert.equal(text.includes('"stop_reason":"max_tokens"'), false);
      }
    });
  });

  it("masks authentication details before and after streaming headers", async () => {
    const secret = "SECRET_TOKEN_AT_/private/auth.json";
    const unavailable = () => new ChatGPTOAuthRefreshError(secret);
    const nonStreamingProvider = {
      async chat() {
        throw unavailable();
      },
    };

    await withServer(nonStreamingProvider, async (baseUrl) => {
      for (const [route, body] of [
        ["/v1/chat/completions", { messages: [{ role: "user", content: "hello" }] }],
        ["/v1/messages", { model: "claude-sonnet-4-6", max_tokens: 64, messages: [{ role: "user", content: "hello" }] }],
      ] as const) {
        const response = await fetch(`${baseUrl}${route}`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "test-session",
          },
          body: JSON.stringify(body),
        });
        assert.equal(response.status, 401);
        const text = await response.text();
        assert.match(text, /rerun codex login/);
        assert.equal(text.includes(secret), false);
      }
    });

    const streamingProvider = {
      async *chatStream() {
        yield { type: "content", text: "partial" };
        throw unavailable();
      },
    };
    await withServer(streamingProvider, async (baseUrl) => {
      for (const [route, body] of [
        ["/v1/chat/completions", { stream: true, messages: [{ role: "user", content: "hello" }] }],
        ["/v1/messages", { model: "claude-sonnet-4-6", max_tokens: 64, stream: true, messages: [{ role: "user", content: "hello" }] }],
      ] as const) {
        const response = await fetch(`${baseUrl}${route}`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "test-session",
          },
          body: JSON.stringify(body),
        });
        assert.equal(response.status, 200);
        const text = await response.text();
        assert.match(text, /rerun codex login/);
        assert.equal(text.includes(secret), false);
      }
    });
  });

  it("masks unexpected server exceptions before and after streaming headers", async () => {
    const secret = "access_token=SERVER_SECRET";
    const nonStreamingProvider = {
      async chat() {
        throw new Error(secret);
      },
    };
    await withServer(nonStreamingProvider, async (baseUrl) => {
      for (const [route, body] of [
        ["/v1/chat/completions", { messages: [{ role: "user", content: "hello" }] }],
        ["/v1/messages", { model: "claude-sonnet-4-6", max_tokens: 64, messages: [{ role: "user", content: "hello" }] }],
      ] as const) {
        const response = await fetch(`${baseUrl}${route}`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "test-session",
          },
          body: JSON.stringify(body),
        });
        assert.equal(response.status, 500);
        const text = await response.text();
        assert.match(text, /internal server error/);
        assert.equal(text.includes(secret), false);
      }
    });

    const streamingProvider = {
      async *chatStream() {
        yield { type: "content", text: "partial" };
        throw new Error(secret);
      },
    };
    await withServer(streamingProvider, async (baseUrl) => {
      for (const [route, body] of [
        ["/v1/chat/completions", { stream: true, messages: [{ role: "user", content: "hello" }] }],
        ["/v1/messages", { model: "claude-sonnet-4-6", max_tokens: 64, stream: true, messages: [{ role: "user", content: "hello" }] }],
      ] as const) {
        const response = await fetch(`${baseUrl}${route}`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "test-session",
          },
          body: JSON.stringify(body),
        });
        assert.equal(response.status, 200);
        const text = await response.text();
        assert.match(text, /internal server error/);
        assert.equal(text.includes(secret), false);
      }
    });
  });

  it("masks upstream-controlled error details at JSON and SSE boundaries", async () => {
    const secret = "access_token=UPSTREAM_REFLECTION_SECRET";
    const cases: Array<{
      makeError: () => Error;
      publicMessage: string;
      status: number;
    }> = [
      {
        makeError: () => new ChatGPTOAuthUpstreamError(529, secret),
        publicMessage: "upstream request failed",
        status: 529,
      },
      {
        makeError: () => new ChatGPTOAuthUnavailableError(secret),
        publicMessage: "upstream request failed",
        status: 502,
      },
      {
        makeError: () => new ChatGPTOAuthProtocolError(secret),
        publicMessage: "upstream protocol validation failed",
        status: 502,
      },
      {
        makeError: () => new ChatGPTOAuthCatalogUnavailableError(secret),
        publicMessage: "authenticated model catalog is unavailable",
        status: 503,
      },
    ];

    for (const testCase of cases) {
      const nonStreamingProvider = {
        async chat() {
          throw testCase.makeError();
        },
      };
      await withServer(nonStreamingProvider, async (baseUrl) => {
        for (const [route, body] of [
          ["/v1/chat/completions", { messages: [{ role: "user", content: "hello" }] }],
          ["/v1/messages", { model: "claude-sonnet-4-6", max_tokens: 64, messages: [{ role: "user", content: "hello" }] }],
        ] as const) {
          const response = await fetch(`${baseUrl}${route}`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-claude-code-session-id": "test-session",
            },
            body: JSON.stringify(body),
          });
          assert.equal(response.status, testCase.status);
          const text = await response.text();
          assert.match(text, new RegExp(testCase.publicMessage));
          assert.equal(text.includes(secret), false);
        }
      });

      const streamingProvider = {
        async *chatStream() {
          yield { type: "content", text: "partial" };
          throw testCase.makeError();
        },
      };
      await withServer(streamingProvider, async (baseUrl) => {
        for (const [route, body] of [
          ["/v1/chat/completions", { stream: true, messages: [{ role: "user", content: "hello" }] }],
          ["/v1/messages", { model: "claude-sonnet-4-6", max_tokens: 64, stream: true, messages: [{ role: "user", content: "hello" }] }],
        ] as const) {
          const response = await fetch(`${baseUrl}${route}`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-claude-code-session-id": "test-session",
            },
            body: JSON.stringify(body),
          });
          assert.equal(response.status, 200);
          const text = await response.text();
          assert.match(text, new RegExp(testCase.publicMessage));
          assert.equal(text.includes(secret), false);
        }
      });
    }
  });

  it("maps transport failures to 502 upstream_error", async () => {
    const provider = {
      async chat() {
        throw new ChatGPTOAuthUnavailableError("upstream transport unavailable");
      },
    };
    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "test-session",
        },
        body: JSON.stringify({
          model: "gpt-5.5",
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 502);
      const body = await response.json() as { error: { type: string } };
      assert.equal(body.error.type, "upstream_error");
    });
  });

  it("preserves structured upstream statuses across OpenAI and Anthropic routes", async () => {
    for (const status of [401, 429, 529]) {
      const provider = {
        async chat() {
          throw new ChatGPTOAuthUpstreamError(status, "upstream status without parseable digits");
        },
      };
      await withServer(provider, async (baseUrl) => {
        const openai = await fetch(`${baseUrl}/v1/chat/completions`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "test-session",
          },
          body: JSON.stringify({
            model: "gpt-5.5",
            messages: [{ role: "user", content: "hello" }],
          }),
        });
        const anthropic = await fetch(`${baseUrl}/v1/messages`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "test-session",
          },
          body: JSON.stringify({
            model: "claude-sonnet-4-6",
            max_tokens: 64,
            messages: [{ role: "user", content: "hello" }],
          }),
        });
        assert.equal(openai.status, status);
        assert.equal(anthropic.status, status);
        const openaiBody = await openai.json() as { error: { type: string } };
        assert.equal(openaiBody.error.type, "upstream_error");
        const body = await anthropic.json() as { error: { type: string } };
        assert.equal(body.error.type, {
          401: "authentication_error",
          429: "rate_limit_error",
          529: "overloaded_error",
        }[status]);
      });
    }
  });

  it("returns a typed 400 for non-empty stop before transport", async () => {
    const provider = new ChatGPTOAuthProvider({ model: "gpt-5.5" });
    let transportCalls = 0;
    (provider as unknown as {
      postSSE(path: string, payload: Record<string, unknown>): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      transportCalls += 1;
      yield { type: "response.completed", response: { id: "unexpected" } };
    };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "gpt-5.5",
          stop: "END",
          messages: [
            { role: "system", content: "system" },
            { role: "user", content: "hello" },
          ],
        }),
      });

      assert.equal(response.status, 400);
      assert.equal(response.headers.get("content-type"), "application/json; charset=utf-8");
      const body = await response.json() as { error: { type: string; message: string } };
      assert.equal(body.error.type, "invalid_request_error");
      assert.match(body.error.message, /stop is not supported/);
    });
    assert.equal(transportCalls, 0);
  });

  it("rejects invalid or unsupported request efforts before opening an OpenAI stream", async () => {
    let providerCalls = 0;
    const provider = {
      async createChatStream() {
        providerCalls += 1;
        throw new Error("provider must not be called");
      },
    };

    await withServer(provider, async (baseUrl) => {
      for (const reasoningEffort of ["", "unsupported-request-effort"]) {
        const response = await fetch(`${baseUrl}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            stream: true,
            reasoning_effort: reasoningEffort,
            messages: [{ role: "user", content: "hello" }],
          }),
        });
        assert.equal(response.status, 400);
        assert.equal(response.headers.get("content-type"), "application/json; charset=utf-8");
        const body = await response.json() as { error: { type: string } };
        assert.equal(body.error.type, "invalid_request_error");
      }
    });
    assert.equal(providerCalls, 0);
  });

  it("reports configured reasoning failures as server errors before opening a stream", async () => {
    for (const configuredEffort of ["", " padded-effort", "unsupported-config-effort"]) {
      let providerCalls = 0;
      const provider = {
        async createChatStream() {
          providerCalls += 1;
          throw new Error("provider must not be called");
        },
      };

      await withServer(provider, async (baseUrl) => {
        const response = await fetch(`${baseUrl}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            stream: true,
            messages: [{ role: "user", content: "hello" }],
          }),
        });
        assert.equal(response.status, 500);
        assert.equal(response.headers.get("content-type"), "application/json; charset=utf-8");
        assert.deepEqual(await response.json(), {
          error: {
            message: "internal server error",
            type: "server_error",
            code: "server_error",
          },
        });
      }, {
        codexConfig: { ...TEST_CONFIG, modelReasoningEffort: configuredEffort },
      });
      assert.equal(providerCalls, 0);
    }
  });

  it("rejects an invalid Responses Lite mode or type before opening an OpenAI stream", async () => {
    const provider = useTestCatalog(new ChatGPTOAuthProvider({ model: "gpt-5.6-sol" }));
    await withServer(provider, async (baseUrl) => {
      for (const responsesLite of ["bogus", 42]) {
        const response = await fetch(`${baseUrl}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "gpt-5.6-sol",
            stream: true,
            responses_lite: responsesLite,
            messages: [
              { role: "system", content: "system" },
              { role: "user", content: "hello" },
            ],
          }),
        });

        assert.equal(response.status, 400);
        assert.equal(response.headers.get("content-type"), "application/json; charset=utf-8");
        const error = await response.json() as {
          error: { message: string; type: string; code: string };
        };
        assert.equal(error.error.type, "invalid_request_error");
        assert.equal(error.error.code, "invalid_request_error");
      }
    }, { model: "gpt-5.6-sol" });
  });

  it("maps an unsupported Lite tool choice to a structured 400 before streaming", async () => {
    const provider = useTestCatalog(new ChatGPTOAuthProvider({ model: "gpt-5.6-sol" }));
    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "gpt-5.6-sol",
          stream: true,
          responses_lite: true,
          tool_choice: "required",
          messages: [
            { role: "system", content: "system" },
            { role: "user", content: "hello" },
          ],
        }),
      });

      assert.equal(response.status, 400);
      assert.equal(response.headers.get("content-type"), "application/json; charset=utf-8");
      const error = await response.json() as {
        error: { message: string; type: string; code: string };
      };
      assert.equal(error.error.message, "Responses Lite requires tool_choice to be the exact string auto");
      assert.equal(error.error.type, "invalid_request_error");
      assert.equal(error.error.code, "invalid_request_error");
    }, { model: "gpt-5.6-sol" });
  });

  it("ends OpenAI streams with an SSE error instead of sending JSON after headers", async () => {
    const provider = {
      async *chatStream() {
        yield { type: "content", text: "partial" };
        throw new ChatGPTOAuthInvalidRequestError(
          "OpenAI protocol response failed: Your input exceeds the context window of this model.",
        );
      },
    };

    await withServer(provider, async (baseUrl) => {
      const res = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "gpt-5.5",
          stream: true,
          messages: [{ role: "user", content: "hello" }],
        }),
      });

      assert.equal(res.status, 200);
      const body = await res.text();
      assert.match(body, /partial/);
      assert.match(body, /exceeds the context window/);
      assert.doesNotMatch(body, /data: \[DONE\]/);
    });
  });

  it("ends Anthropic streams with an SSE error instead of sending JSON after headers", async () => {
    const provider = {
      async *chatStream() {
        yield { type: "content", text: "partial" };
        throw new ChatGPTOAuthInvalidRequestError(
          "OpenAI protocol response failed: Your input exceeds the context window of this model.",
        );
      },
    };

    await withServer(provider, async (baseUrl) => {
      const res = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "test-session",
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          max_tokens: 64,
          stream: true,
          messages: [{ role: "user", content: "hello" }],
        }),
      });

      assert.equal(res.status, 200);
      const body = await res.text();
      assert.match(body, /event: message_start/);
      assert.match(body, /partial/);
      assert.match(body, /event: error/);
      assert.match(body, /exceeds the context window/);
    });
  });

  it("delivers Anthropic content before the provider stream completes", async () => {
    let releaseFinish!: () => void;
    const finishGate = new Promise<void>((resolve) => {
      releaseFinish = resolve;
    });
    let providerCompleted = false;
    const provider = {
      chatStream() {
        return (async function* () {
          yield { type: "content", text: "early" };
          await finishGate;
          providerCompleted = true;
          yield {
            type: "finish",
            finish_reason: "stop",
            usage: {
              input_tokens: 1,
              output_tokens: 1,
              total_tokens: 2,
              input_tokens_details: { cached_tokens: 0 },
            },
          };
        })();
      },
    };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "test-session",
        },
        body: JSON.stringify({
          model: "claude-fable-5",
          max_tokens: 64,
          stream: true,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 200);
      assert.ok(response.body);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const events: Record<string, unknown>[] = [];
      let pending = "";
      const parseCompleteBlocks = () => {
        for (;;) {
          const boundary = pending.indexOf("\n\n");
          if (boundary < 0) return;
          const block = pending.slice(0, boundary);
          pending = pending.slice(boundary + 2);
          const dataLine = block.split("\n").find((line) => line.startsWith("data: "));
          if (dataLine !== undefined) {
            events.push(JSON.parse(dataLine.slice(6)) as Record<string, unknown>);
          }
        }
      };
      const hasEarlyDelta = () => events.some((event) => {
        if (event.type !== "content_block_delta") return false;
        const delta = event.delta;
        return typeof delta === "object"
          && delta !== null
          && (delta as Record<string, unknown>).type === "text_delta"
          && (delta as Record<string, unknown>).text === "early";
      });

      try {
        while (!hasEarlyDelta()) {
          const chunk = await reader.read();
          assert.equal(chunk.done, false);
          pending += decoder.decode(chunk.value, { stream: true });
          parseCompleteBlocks();
        }
        assert.equal(providerCompleted, false);
      } finally {
        releaseFinish();
      }

      for (;;) {
        const chunk = await reader.read();
        if (chunk.done) break;
        pending += decoder.decode(chunk.value, { stream: true });
        parseCompleteBlocks();
      }
      pending += decoder.decode();
      parseCompleteBlocks();
      assert.equal(providerCompleted, true);
      assert.ok(events.some((event) => event.type === "message_stop"));
    });
  });

  it("rejects Anthropic hosted WebSearch on all routes and transport modes before provider work", async () => {
    const previous = process.env.CODEX_AS_API_RESPONSES_LITE;
    try {
      for (const mode of ["auto", "on", "off"]) {
        process.env.CODEX_AS_API_RESPONSES_LITE = mode;
        const provider = new ChatGPTOAuthProvider({ model: "gpt-5.6-sol" });
        await withServer(provider, async (baseUrl) => {
          for (const route of [
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/messages/compact",
          ]) {
            const response = await fetch(`${baseUrl}${route}`, {
              method: "POST",
              headers: {
                "content-type": "application/json",
                "x-claude-code-session-id": "web-search-session",
              },
              body: JSON.stringify({
                model: "claude-sonnet-4-5",
                max_tokens: 1024,
                ...(route === "/v1/messages" ? { stream: true } : {}),
                system: "You are helpful.",
                tools: [{ type: "web_search_20250305", name: "web_search" }],
                messages: [{ role: "user", content: "hello" }],
              }),
            });

            assert.equal(response.status, 400);
            assert.equal(response.headers.get("content-type"), "application/json; charset=utf-8");
            const body = await response.json() as { error: { type: string; message: string } };
            assert.equal(body.error.type, "invalid_request_error");
            assert.match(body.error.message, /cannot be represented losslessly/);
          }
        }, { model: "gpt-5.6-sol" });
      }
    } finally {
      if (previous == null) delete process.env.CODEX_AS_API_RESPONSES_LITE;
      else process.env.CODEX_AS_API_RESPONSES_LITE = previous;
    }
  });

  it("maps Anthropic context-window failures to 400 invalid_request_error", async () => {
    const provider = {
      async chat() {
        throw new ChatGPTOAuthInvalidRequestError(
          "OpenAI protocol response failed: Your input exceeds the context window of this model.",
        );
      },
    };

    await withServer(provider, async (baseUrl) => {
      const res = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "test-session",
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          max_tokens: 64,
          messages: [{ role: "user", content: "hello" }],
        }),
      });

      assert.equal(res.status, 400);
      const body = await res.json() as { error: { type: string; message: string } };
      assert.equal(body.error.type, "invalid_request_error");
      assert.match(body.error.message, /exceeds the context window/);
    });
  });
});

describe("pinned Codex transport contract", () => {
  it("matches a recorded Lite Responses request", async () => {
    const auth = writeAuthFile();
    try {
      await withRecordingUpstream(async (upstreamUrl, requests) => {
        const provider = new ChatGPTOAuthProvider({
          model: "gpt-5.6-sol",
          baseUrl: upstreamUrl,
          authJsonPath: auth.authPath,
        });

        await provider.chat([
          { role: MessageRole.SYSTEM, content: "You are helpful." },
          { role: MessageRole.USER, content: "Hello" },
        ], {
          model: "gpt-5.6-sol",
          reasoningEffort: "low",
          responsesLite: true,
          parallelToolCalls: false,
        });

        assert.equal(requests.length, 1);
        const recorded = requests[0];
        const requestContract = upstreamContract.responses_request;
        const liteContract = upstreamContract.responses_lite;
        const originatorContract = upstreamContract.headers.originator;
        const reasoning = recorded.body.reasoning as Record<string, unknown>;

        assert.equal(recorded.method, requestContract.method);
        assert.equal(recorded.path, requestContract.path);
        assert.equal(recorded.headers.accept, requestContract.streaming_accept);
        assert.equal(recorded.headers[originatorContract.name], originatorContract.value);
        assert.equal(
          recorded.headers[liteContract.header.name],
          liteContract.header.value,
        );
        assert.equal(reasoning.context, liteContract.reasoning_context);
        assert.equal(recorded.body.parallel_tool_calls, liteContract.parallel_tool_calls);
        assert.ok(
          (recorded.body.include as unknown[]).includes(
            requestContract.reasoning_encrypted_content_include,
          ),
        );
      });
    } finally {
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });
});

describe("OpenAI stream translation", () => {
  it("rejects missing final finish reasons in streaming and non-streaming responses", async () => {
    const streamingProvider = {
      chatStream() {
        return (async function* () {
          yield {
            type: "finish",
            response_id: "response-empty-stream",
            finish_reason: null,
          };
        })();
      },
    };
    await withServer(streamingProvider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          stream: true,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 200);
      const text = await response.text();
      assert.equal(text.includes('"error"'), true);
      assert.equal(text.includes("data: [DONE]"), false);
    });

    const nonStreamingProvider = {
      async chat() {
        return {
          content: "",
          tool_calls: [],
          finish_reason: null,
          usage: null,
          reasoning_content: null,
          raw: { events: [] },
          response_id: "response-empty",
        };
      },
    };
    await withServer(nonStreamingProvider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] }),
      });
      assert.equal(response.status, 502, await response.clone().text());
    });
  });

  it("adds stable tool indexes and maps Responses usage fields", async () => {
    const provider = {
      chatStream() {
        return (async function* () {
          yield { type: "tool_call", id: "call-a", name: "first", arguments: '{"value":1}' };
          yield { type: "tool_call", id: "call-b", name: "second", arguments: '{"value":2}' };
          yield {
            type: "finish",
            response_id: "response-1",
            finish_reason: "tool_calls",
            usage: {
              input_tokens: 10,
              output_tokens: 4,
              total_tokens: 14,
              input_tokens_details: { cached_tokens: 0, cache_write_tokens: 0 },
            },
          };
        })();
      },
    };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          stream: true,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 200);
      const chunks = (await response.text())
        .split("\n\n")
        .filter((block) => block.startsWith("data: {") )
        .map((block) => JSON.parse(block.slice(6)) as Record<string, unknown>);
      const toolCalls = chunks.flatMap((chunk) => {
        const choices = chunk.choices;
        if (!Array.isArray(choices) || choices.length === 0) return [];
        const delta = (choices[0] as Record<string, unknown>).delta;
        if (typeof delta !== "object" || delta === null) return [];
        const calls = (delta as Record<string, unknown>).tool_calls;
        return Array.isArray(calls) ? calls as Record<string, unknown>[] : [];
      });
      assert.deepEqual(toolCalls.map((call) => [call.index, call.id]), [
        [0, "call-a"],
        [1, "call-b"],
      ]);
      const usageChunk = chunks.find((chunk) => Array.isArray(chunk.choices)
        && chunk.choices.length === 0
        && typeof chunk.usage === "object");
      assert.deepEqual(usageChunk?.usage, {
        prompt_tokens: 10,
        completion_tokens: 4,
        total_tokens: 14,
        prompt_tokens_details: {
          cached_tokens: 0,
          cache_write_tokens: 0,
        },
      });
    });
  });

  it("fails the OpenAI stream instead of merging duplicate tool call ids", async () => {
    const provider = {
      chatStream() {
        return (async function* () {
          yield { type: "tool_call", id: "duplicate-call", name: "first", arguments: "{}" };
          yield { type: "tool_call", id: "duplicate-call", name: "second", arguments: "{}" };
        })();
      },
    };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          stream: true,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 200);
      const body = await response.text();
      const chunks = body
        .split("\n\n")
        .filter((block) => block.startsWith("data: {") )
        .map((block) => JSON.parse(block.slice(6)) as Record<string, unknown>);
      const toolCalls = chunks.flatMap((chunk) => {
        const choices = chunk.choices;
        if (!Array.isArray(choices) || choices.length === 0) return [];
        const delta = (choices[0] as Record<string, unknown>).delta;
        if (typeof delta !== "object" || delta === null) return [];
        const calls = (delta as Record<string, unknown>).tool_calls;
        return Array.isArray(calls) ? calls as Record<string, unknown>[] : [];
      });
      assert.deepEqual(toolCalls.map((call) => [call.index, call.id]), [[0, "duplicate-call"]]);
      assert.equal(body.includes('"error"'), true);
      assert.equal(body.includes("data: [DONE]"), false);
    });
  });

  it("preserves absent usage without synthesizing token counts", async () => {
    const streamingProvider = {
      chatStream() {
        return (async function* () {
          yield { type: "content", text: "hi" };
          yield {
            type: "finish",
            response_id: "response-no-usage",
            finish_reason: "stop",
          };
        })();
      },
    };
    await withServer(streamingProvider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          stream: true,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 200);
      const chunks = (await response.text())
        .split("\n\n")
        .filter((block) => block.startsWith("data: {") )
        .map((block) => JSON.parse(block.slice(6)) as Record<string, unknown>);
      assert.equal(chunks.some((chunk) => Object.hasOwn(chunk, "usage")), false);
      const terminal = chunks.find((chunk) => chunk.response_id === "response-no-usage");
      assert.ok(terminal);
      const terminalChoices = terminal.choices as Record<string, unknown>[];
      assert.equal(terminalChoices[0].finish_reason, "stop");
    });

    const nonStreamingProvider = {
      async chat() {
        return {
          content: "hi",
          tool_calls: [],
          finish_reason: "stop",
          usage: null,
          reasoning_content: null,
          raw: null,
          response_id: "response-no-usage",
        };
      },
    };
    await withServer(nonStreamingProvider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 200, await response.clone().text());
      const body = await response.json() as Record<string, unknown>;
      assert.equal(Object.hasOwn(body, "usage"), false);
      const choices = body.choices as Record<string, unknown>[];
      assert.equal(choices[0].finish_reason, "stop");
      assert.equal(choices[0].logprobs, null);
      const message = choices[0].message as Record<string, unknown>;
      assert.equal(message.refusal, null);
    });
  });
});

describe("server model defaults", () => {
  it("does not expose auth paths through ordinary model errors", async () => {
    const directory = fs.mkdtempSync(path.join(os.tmpdir(), "secret-auth-path-"));
    const missingPath = path.join(directory, "missing-auth.json");
    const invalidPath = path.join(directory, "invalid-auth.json");
    fs.writeFileSync(invalidPath, "not json{");
    try {
      for (const authJsonPath of [missingPath, invalidPath]) {
        const provider = new ChatGPTOAuthProvider({ authJsonPath });
        await withServer(provider, async (baseUrl) => {
          const response = await fetch(`${baseUrl}/v1/models`);
          assert.equal(response.status, 401);
          const text = await response.text();
          assert.match(text, /rerun codex login/);
          assert.equal(text.includes(directory), false);
          assert.equal(text.includes(authJsonPath), false);
        }, { authPath: authJsonPath });
      }
    } finally {
      fs.rmSync(directory, { recursive: true });
    }
  });

  it("sanitizes authentication details in health diagnostics", async () => {
    await withServer({
      async catalogSnapshot() {
        throw new ChatGPTOAuthMissingError("ChatGPT OAuth auth file not found: /secret/auth.json");
      },
    }, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/health`);
      assert.equal(response.status, 503);
      const body = await response.json() as Record<string, unknown>;
      assert.equal(body.auth_available, false);
      assert.deepEqual(body.error, {
        type: "authentication_error",
        message: "ChatGPT OAuth credentials are unavailable",
      });
      assert.equal(JSON.stringify(body).includes("/secret/auth.json"), false);
    });
  });

  it("reports an unknown configured model as not ready without a fallback", async () => {
    await withServer({}, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/health`);
      assert.equal(response.status, 503);
      const body = await response.json() as Record<string, unknown>;
      assert.equal(body.status, "error");
      assert.equal(body.catalog_status, "fresh");
      assert.equal(body.model, null);
      assert.deepEqual(body.error, {
        type: "model_not_found",
        message: "configured model is unavailable in the authenticated catalog",
      });
      assert.deepEqual(Object.keys(body).sort(), [
        "auth_available",
        "auto_compact_token_limit",
        "catalog_expires_at",
        "catalog_fetched_at",
        "catalog_status",
        "context_window",
        "error",
        "model",
        "reasoning_effort",
        "status",
      ]);
    }, { model: "provider future+model" });
  });

  it("reports only safe live-catalog readiness diagnostics", async () => {
    await withServer({}, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/health`);
      assert.equal(response.status, 200);
      const body = await response.json() as Record<string, unknown>;
      assert.equal(body.status, "ok");
      assert.equal(body.auth_available, true);
      assert.equal(body.catalog_status, "fresh");
      assert.equal(Object.hasOwn(body, "catalog_etag"), false);
      assert.equal(body.model, "gpt-5.6-sol");
      assert.equal(body.reasoning_effort, "low");
      assert.equal(body.context_window, 258_400);
      assert.equal(body.auto_compact_token_limit, 244_800);
      assert.deepEqual(Object.keys(body).sort(), [
        "auth_available",
        "auto_compact_token_limit",
        "catalog_expires_at",
        "catalog_fetched_at",
        "catalog_status",
        "context_window",
        "model",
        "reasoning_effort",
        "status",
      ]);
      assert.equal(Object.hasOwn(body, "account_id"), false);
      assert.equal(Object.hasOwn(body, "token"), false);
    }, {
      model: "gpt-5.6-sol",
    });
  });

  it("keeps a model without a reasoning-effort default healthy and sends its summary default", async () => {
    const base = testCatalogSnapshot();
    const capability: ModelCapability = Object.freeze({
      ...base.defaultModel!,
      defaultReasoningEffort: undefined,
    });
    const snapshot: ModelCatalogSnapshot = Object.freeze({
      ...base,
      models: Object.freeze([capability]),
      defaultModel: capability,
    });
    const provider = new ChatGPTOAuthProvider({ model: capability.slug });
    (provider as unknown as {
      catalogSnapshot(): Promise<ModelCatalogSnapshot>;
      prepareModel(): Promise<PreparedModel>;
      postSSE(
        path: string,
        payload: Record<string, unknown>,
      ): AsyncGenerator<Record<string, unknown>>;
    }).catalogSnapshot = async () => snapshot;
    (provider as unknown as {
      prepareModel(): Promise<PreparedModel>;
    }).prepareModel = async () => ({
      slug: capability.slug,
      accountId: "test-account",
      capability,
      snapshot,
    });
    let capturedPayload: Record<string, unknown> | undefined;
    (provider as unknown as {
      postSSE(
        path: string,
        payload: Record<string, unknown>,
      ): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* (_path, payload) {
      capturedPayload = payload;
      yield {
        type: "response.output_item.done",
        item: {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "ok" }],
        },
      };
      yield {
        type: "response.completed",
        response: { id: "response-no-reasoning", end_turn: true, output: [] },
      };
    };

    await withServer(provider, async (baseUrl) => {
      const healthResponse = await fetch(`${baseUrl}/health`);
      assert.equal(healthResponse.status, 200);
      const health = await healthResponse.json() as Record<string, unknown>;
      assert.equal(health.reasoning_effort, null);

      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: capability.slug,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 200, await response.clone().text());
      assert.deepEqual(capturedPayload?.reasoning, {
        summary: "auto",
        context: "all_turns",
      });
    }, { model: capability.slug });
  });

  it("never exposes an upstream catalog ETag through health", async () => {
    const credentialSentinel = "access_token=HEALTH_ETAG_SENTINEL";
    const base = testCatalogSnapshot();
    const snapshot: ModelCatalogSnapshot = Object.freeze({
      ...base,
      etag: credentialSentinel,
    });
    const capability = snapshot.defaultModel!;
    const provider = {
      async catalogSnapshot() {
        return snapshot;
      },
      async prepareModel(): Promise<PreparedModel> {
        return {
          slug: capability.slug,
          accountId: "test-account",
          capability,
          snapshot,
        };
      },
    };
    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/health`);
      assert.equal(response.status, 200);
      const body = await response.json() as Record<string, unknown>;
      assert.equal(Object.hasOwn(body, "catalog_etag"), false);
      assert.equal(JSON.stringify(body).includes(credentialSentinel), false);
    }, { model: capability.slug });
  });

  it("reports catalog_unavailable when selected live limits cannot support readiness", async () => {
    const base = testCatalogSnapshot();
    for (const overrides of [
      { contextWindow: 0 },
      { contextWindow: undefined, maxContextWindow: 0 },
    ]) {
      const capability: ModelCapability = { ...base.defaultModel!, ...overrides };
      const provider = {
        async catalogSnapshot() {
          return base;
        },
        async prepareModel(): Promise<PreparedModel> {
          return {
            slug: capability.slug,
            accountId: "test-account",
            capability,
            snapshot: base,
          };
        },
      };
      await withServer(provider, async (baseUrl) => {
        const response = await fetch(`${baseUrl}/health`);
        assert.equal(response.status, 503);
        const body = await response.json() as Record<string, unknown>;
        assert.equal(body.status, "error");
        assert.equal(body.catalog_status, "fresh");
        assert.equal(body.model, null);
        assert.equal(body.reasoning_effort, null);
        assert.equal(body.context_window, null);
        assert.equal(body.auto_compact_token_limit, null);
        assert.deepEqual(body.error, {
          type: "catalog_unavailable",
          message: "authenticated model catalog is unavailable",
        });
      }, { model: capability.slug });
    }
  });

  it("preserves a negative live compaction metadata limit", async () => {
    const base = testCatalogSnapshot();
    const capability: ModelCapability = { ...base.defaultModel!, autoCompactTokenLimit: -1 };
    const provider = {
      async catalogSnapshot() {
        return base;
      },
      async prepareModel(): Promise<PreparedModel> {
        return { slug: capability.slug, accountId: "test-account", capability, snapshot: base };
      },
    };
    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/health`);
      assert.equal(response.status, 200, await response.clone().text());
      assert.equal((await response.json() as Record<string, unknown>).auto_compact_token_limit, -1);
    }, { model: capability.slug });
  });

  it("rejects a zero effective context window", async () => {
    const base = testCatalogSnapshot();
    const capability = {
      ...base.defaultModel!,
      contextWindow: 1,
      autoCompactTokenLimit: undefined,
    };
    const provider = {
      async catalogSnapshot() {
        return base;
      },
      async prepareModel(): Promise<PreparedModel> {
        return {
          slug: capability.slug,
          accountId: "test-account",
          capability,
          snapshot: base,
        };
      },
    };
    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/health`);
      assert.equal(response.status, 503, await response.clone().text());
      const body = await response.json() as { error: { type: string } };
      assert.equal(body.error.type, "catalog_unavailable");
    }, { model: capability.slug });
  });

  it("preserves an explicit live zero compaction limit", async () => {
    const base = testCatalogSnapshot();
    const capability = {
      ...base.defaultModel!,
      contextWindow: 100,
      autoCompactTokenLimit: 0,
    };
    const provider = {
      async catalogSnapshot() {
        return base;
      },
      async prepareModel(): Promise<PreparedModel> {
        return {
          slug: capability.slug,
          accountId: "test-account",
          capability,
          snapshot: base,
        };
      },
    };
    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/health`);
      assert.equal(response.status, 200, await response.clone().text());
      const body = await response.json() as Record<string, unknown>;
      assert.equal(body.context_window, 95);
      assert.equal(body.auto_compact_token_limit, 0);
    }, { model: capability.slug });
  });

  it("rejects empty or whitespace-padded CODEX_AS_API_MODEL values at startup", () => {
    const previous = process.env.CODEX_AS_API_MODEL;
    try {
      for (const model of ["", " gpt-5.6-sol", "gpt-5.6-sol "]) {
        process.env.CODEX_AS_API_MODEL = model;
        assert.throws(
          () => createApp({ provider: {} as never, codexConfig: TEST_CONFIG }),
          ChatGPTOAuthInvalidRequestError,
        );
      }
    } finally {
      if (previous == null) delete process.env.CODEX_AS_API_MODEL;
      else process.env.CODEX_AS_API_MODEL = previous;
    }
  });

  it("rejects the deprecated Codex CLI version override at startup", () => {
    const previous = process.env.CODEX_AS_API_CODEX_CLI_VERSION;
    process.env.CODEX_AS_API_CODEX_CLI_VERSION = "9.8.7";
    try {
      assert.throws(
        () => createApp({ provider: {} as never, codexConfig: TEST_CONFIG }),
        (error) => error instanceof ChatGPTOAuthInvalidRequestError
          && error.message === "CODEX_AS_API_CODEX_CLI_VERSION is not supported; the wire contract is pinned to 0.153.3",
      );
    } finally {
      if (previous == null) delete process.env.CODEX_AS_API_CODEX_CLI_VERSION;
      else process.env.CODEX_AS_API_CODEX_CLI_VERSION = previous;
    }
  });

  it("rejects an unsafe OAuth refresh endpoint override at startup", () => {
    const previous = process.env.CODEX_REFRESH_TOKEN_URL_OVERRIDE;
    process.env.CODEX_REFRESH_TOKEN_URL_OVERRIDE = "https://user:secret@auth.example.test/token";
    try {
      assert.throws(
        () => createApp({ provider: {} as never, codexConfig: TEST_CONFIG }),
        ChatGPTOAuthRefreshError,
      );
    } finally {
      if (previous == null) delete process.env.CODEX_REFRESH_TOKEN_URL_OVERRIDE;
      else process.env.CODEX_REFRESH_TOKEN_URL_OVERRIDE = previous;
    }
  });

  it("rejects invalid explicit host, port, path, and model configuration", () => {
    assert.throws(() => resolveServerHost("   "), ChatGPTOAuthInvalidRequestError);
    assert.equal(resolveServerHost("0.0.0.0"), "0.0.0.0");
    for (const value of ["", " ", "18080junk", "1.5", "NaN", "0", "65536"]) {
      assert.throws(() => resolveServerPort(value), ChatGPTOAuthInvalidRequestError);
    }
    assert.equal(resolveServerPort("1"), 1);
    assert.equal(resolveServerPort("65535"), 65_535);

    for (const options of [
      { authPath: "" },
      { model: " " },
      { model: " gpt-5.6-sol" },
      { model: "gpt-5.6-sol " },
      { codexConfig: { ...TEST_CONFIG, codexHome: " " } },
      { codexConfig: { ...TEST_CONFIG, configPath: " " } },
      { codexConfig: { ...TEST_CONFIG, model: " " } },
      { codexConfig: { ...TEST_CONFIG, model: " gpt-5.6-sol" } },
    ]) {
      assert.throws(
        () => createApp({ provider: {} as never, ...options }),
        ChatGPTOAuthInvalidRequestError,
      );
    }

    for (const name of ["CODEX_HOME", "CODEX_AS_API_AUTH_PATH"] as const) {
      const previous = process.env[name];
      process.env[name] = " ";
      try {
        assert.throws(
          () => createApp({ provider: {} as never, codexConfig: TEST_CONFIG }),
          ChatGPTOAuthInvalidRequestError,
        );
      } finally {
        if (previous == null) delete process.env[name];
        else process.env[name] = previous;
      }
    }

    for (const [name, value] of [
      ["CODEX_AS_API_RESPONSES_LITE", "sometimes"],
      ["CODEX_AS_API_CODEX_METADATA", "sometimes"],
      ["CODEX_AS_API_CODEX_METADATA", ""],
    ] as const) {
      const previous = process.env[name];
      process.env[name] = value;
      try {
        assert.throws(
          () => createApp({ provider: {} as never, codexConfig: TEST_CONFIG }),
          ChatGPTOAuthInvalidRequestError,
        );
      } finally {
        if (previous == null) delete process.env[name];
        else process.env[name] = previous;
      }
    }
  });
});

describe("live model catalog routes", () => {
  it("exposes an empty live catalog while default readiness fails", async () => {
    const empty = parseModelCatalog({ models: [] }, {
      key: modelCatalogCacheKey("test-account", "https://catalog.test", "0.153.3"),
      etag: null,
      fetchedAt: 1_000,
      expiresAt: 301_000,
    });
    await withServer({
      catalogSnapshot: async () => empty,
      prepareModel: async () => {
        throw new ChatGPTOAuthCatalogUnavailableError("catalog has no default model");
      },
    }, async (baseUrl) => {
      const models = await fetch(`${baseUrl}/v1/models`);
      const health = await fetch(`${baseUrl}/health`);

      assert.equal(models.status, 200);
      assert.deepEqual(await models.json(), { object: "list", data: [] });
      assert.equal(health.status, 503);
      const body = await health.json() as { error: { type: string } };
      assert.equal(body.error.type, "catalog_unavailable");
    });
  });

  it("exposes every live catalog row without inventing timestamps or private prompts", async () => {
    await withServer({}, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/models`);
      assert.equal(response.status, 200);
      assert.equal(response.headers.get("etag"), null);
      const body = await response.json() as {
        object: string;
        data: Record<string, unknown>[];
      };
      assert.equal(body.object, "list");
      assert.deepEqual(body.data.map((model) => model.id), [
        "gpt-5.5",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "catalog-only",
      ]);
      const catalogOnly = body.data.find((model) => model.id === "catalog-only");
      assert.equal(catalogOnly?.supported_in_api, false);
      assert.equal(catalogOnly?.multi_agent_reasoning_effort, "max");
      assert.equal(catalogOnly?.supports_reasoning_summary_parameter, true);
      assert.equal(catalogOnly?.default_reasoning_summary, "auto");
      assert.equal(catalogOnly?.comp_hash, "compatibility-family");
      assert.equal(Object.hasOwn(catalogOnly ?? {}, "supports_parallel_tool_calls"), false);
      for (const model of body.data) {
        assert.equal(Object.hasOwn(model, "created"), false);
        assert.equal(Object.hasOwn(model, "base_instructions"), false);
        assert.equal(Object.hasOwn(model, "model_messages"), false);
      }
    });
  });

  it("classifies a successful malformed catalog response as catalog unavailable", async () => {
    const auth = writeAuthFile();
    const upstream = createServer((req, res) => {
      if (req.method === "GET" && req.url?.startsWith("/models?")) {
        res.writeHead(200, { "content-type": "application/json" });
        res.end("{not-valid-json");
        return;
      }
      res.writeHead(500);
      res.end();
    });
    await new Promise<void>((resolve) => upstream.listen(0, "127.0.0.1", resolve));
    try {
      const address = upstream.address();
      assert.ok(address && typeof address === "object");
      const provider = new ChatGPTOAuthProvider({
        baseUrl: `http://127.0.0.1:${address.port}`,
        authJsonPath: auth.authPath,
      });
      await withServer(provider, async (baseUrl) => {
        const modelsResponse = await fetch(`${baseUrl}/v1/models`);
        assert.equal(modelsResponse.status, 503);
        const models = await modelsResponse.json() as {
          error: { type: string; message: string };
        };
        assert.deepEqual(models.error, {
          type: "catalog_unavailable",
          code: "catalog_unavailable",
          message: "authenticated model catalog is unavailable",
        });

        const healthResponse = await fetch(`${baseUrl}/health`);
        assert.equal(healthResponse.status, 503);
        const health = await healthResponse.json() as {
          catalog_status: string;
          error: { type: string; message: string };
        };
        assert.equal(health.catalog_status, "unavailable");
        assert.deepEqual(health.error, {
          type: "catalog_unavailable",
          message: "authenticated model catalog is unavailable",
        });
      }, { model: "gpt-5.6-sol", authPath: auth.authPath });
    } finally {
      await new Promise<void>((resolve, reject) => {
        upstream.close((err) => (err ? reject(err) : resolve()));
      });
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });

  it("classifies invalid UTF-8 catalog bytes as catalog unavailable", async () => {
    const auth = writeAuthFile();
    const upstream = createServer((req, res) => {
      if (req.method === "GET" && req.url?.startsWith("/models?")) {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(Buffer.from([0x7b, 0xff, 0x7d]));
        return;
      }
      res.writeHead(500);
      res.end();
    });
    await new Promise<void>((resolve) => upstream.listen(0, "127.0.0.1", resolve));
    try {
      const address = upstream.address();
      assert.ok(address && typeof address === "object");
      const provider = new ChatGPTOAuthProvider({
        baseUrl: `http://127.0.0.1:${address.port}`,
        authJsonPath: auth.authPath,
      });
      await withServer(provider, async (baseUrl) => {
        const response = await fetch(`${baseUrl}/v1/models`);
        assert.equal(response.status, 503);
        const body = await response.json() as { error: { type: string } };
        assert.equal(body.error.type, "catalog_unavailable");
      }, { model: "gpt-5.6-sol", authPath: auth.authPath });
    } finally {
      await new Promise<void>((resolve, reject) => {
        upstream.close((err) => (err ? reject(err) : resolve()));
      });
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });

  it("classifies malformed and duplicate catalog schemas as catalog unavailable", async () => {
    const duplicateSentinel = "access_token=CATALOG_DUPLICATE_SENTINEL";
    for (const catalogValue of [
      { models: "not-an-array" },
      { models: [rawModel(duplicateSentinel), rawModel(duplicateSentinel)] },
    ]) {
      const provider = {
        async catalogSnapshot() {
          return parseModelCatalog(catalogValue, {
            key: modelCatalogCacheKey("test-account", "https://catalog.test", "0.153.3"),
            etag: null,
            fetchedAt: 1_000,
            expiresAt: 301_000,
          });
        },
      };
      await withServer(provider, async (baseUrl) => {
        for (const path of ["/v1/models", "/health"]) {
          const response = await fetch(`${baseUrl}${path}`);
          assert.equal(response.status, 503);
          const text = await response.text();
          assert.equal(text.includes(duplicateSentinel), false);
          assert.equal(
            (JSON.parse(text) as { error: { type: string } }).error.type,
            "catalog_unavailable",
          );
        }
      });
    }
  });

  it("classifies an in-flight catalog invalidation as catalog unavailable", async () => {
    let monotonicNow = 0;
    const cache = new ModelCatalogCache(1, () => 1_000, () => monotonicNow);
    const key = modelCatalogCacheKey("test-account", "https://catalog.test", "0.153.3");
    await cache.get(key, async () => ({
      value: { models: [rawModel("seed-model")] },
      etag: '"old"',
    }));
    monotonicNow = 2;
    let markRefreshStarted: (() => void) | undefined;
    let releaseRefresh: (() => void) | undefined;
    const refreshStarted = new Promise<void>((resolve) => { markRefreshStarted = resolve; });
    const refreshGate = new Promise<void>((resolve) => { releaseRefresh = resolve; });
    const provider = {
      async catalogSnapshot() {
        return cache.get(key, async () => {
          markRefreshStarted?.();
          await refreshGate;
          return {
            value: { models: [rawModel("stale-model")] },
            etag: '"old"',
          };
        });
      },
    };

    await withServer(provider, async (baseUrl) => {
      const responsePromise = fetch(`${baseUrl}/v1/models`);
      await refreshStarted;
      cache.invalidateOnEtagMismatch(key, '"new"');
      releaseRefresh?.();
      const response = await responsePromise;
      assert.equal(response.status, 503);
      const body = await response.json() as { error: { type: string } };
      assert.equal(body.error.type, "catalog_unavailable");
    });
  });

  it("keeps invalid UTF-8 compact responses classified as upstream protocol errors", async () => {
    const auth = writeAuthFile();
    const upstream = createServer((req, res) => {
      if (req.method === "GET" && req.url?.startsWith("/models?")) {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify(TEST_CATALOG_VALUE));
        return;
      }
      if (req.method === "POST" && req.url === "/responses/compact") {
        req.resume();
        res.writeHead(200, { "content-type": "application/json" });
        res.end(Buffer.from([0x7b, 0xff, 0x7d]));
        return;
      }
      res.writeHead(500);
      res.end();
    });
    await new Promise<void>((resolve) => upstream.listen(0, "127.0.0.1", resolve));
    try {
      const address = upstream.address();
      assert.ok(address && typeof address === "object");
      const provider = new ChatGPTOAuthProvider({
        baseUrl: `http://127.0.0.1:${address.port}`,
        authJsonPath: auth.authPath,
      });
      await withServer(provider, async (baseUrl) => {
        const response = await fetch(`${baseUrl}/v1/compact`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "gpt-5.6-sol",
            messages: [{ role: "user", content: "history" }],
          }),
        });
        assert.equal(response.status, 502);
        const body = await response.json() as { error: { type: string } };
        assert.equal(body.error.type, "upstream_protocol_error");
      }, { model: "gpt-5.6-sol", authPath: auth.authPath });
    } finally {
      await new Promise<void>((resolve, reject) => {
        upstream.close((err) => (err ? reject(err) : resolve()));
      });
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });

  it("shares one fresh authenticated snapshot across catalog and model routes", async () => {
    const auth = writeAuthFile();
    try {
      await withRecordingUpstream(async (upstreamUrl, _requests, catalogRequests) => {
        const provider = new ChatGPTOAuthProvider({
          model: "gpt-5.6-sol",
          baseUrl: upstreamUrl,
          authJsonPath: auth.authPath,
        });
        await withServer(provider, async (baseUrl) => {
          const models = await fetch(`${baseUrl}/v1/models`);
          assert.equal(models.status, 200);
          const health = await fetch(`${baseUrl}/health`);
          assert.equal(health.status, 200);
          const modelSentinel = "access_token=MODEL_REQUEST_SENTINEL";
          const unknown = await fetch(`${baseUrl}/v1/chat/completions`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: modelSentinel,
              messages: [{ role: "user", content: "hello" }],
            }),
          });
          assert.equal(unknown.status, 404);
          const error = await unknown.json() as {
            error: { type: string; code: string; message: string };
          };
          assert.equal(error.error.type, "model_not_found");
          assert.equal(error.error.code, "model_not_found");
          assert.equal(error.error.message.includes(modelSentinel), false);
        }, { model: "gpt-5.6-sol", authPath: auth.authPath });
        assert.equal(catalogRequests.length, 1);
        const request = catalogRequests[0];
        const contract = upstreamContract.models_request;
        const snapshot = await provider.catalogSnapshot();
        const requestUrl = new URL(request.path, upstreamUrl);
        const keyValues = JSON.parse(snapshot.key) as string[];
        assert.equal(request.method, contract.method);
        assert.equal(requestUrl.pathname, contract.path);
        assert.equal(
          requestUrl.searchParams.get(contract.client_version_query),
          upstreamContract.upstream.version,
        );
        assert.equal(snapshot.etag, '"test-catalog"');
        assert.deepEqual(
          Object.fromEntries(contract.cache_scope.map((field, index) => [field, keyValues[index]])),
          {
            account_id: "account-123",
            base_url: upstreamUrl,
            client_version: upstreamContract.upstream.version,
          },
        );
        assert.equal(contract.allow_stale_on_refresh_error, false);
        assert.equal(request.headers["chatgpt-account-id"], "account-123");
        assert.equal(
          DEFAULT_MODEL_CATALOG_TTL_MS,
          contract.cache_ttl_seconds * 1_000,
        );
        assert.equal(
          MODEL_CATALOG_TIMEOUT_MS,
          contract.request_timeout_seconds * 1_000,
        );
      });
    } finally {
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });

  it("returns catalog failures before committing streaming headers", async () => {
    let streamCalls = 0;
    const provider = {
      async prepareModel() {
        throw new ChatGPTOAuthCatalogUnavailableError("catalog unavailable");
      },
      async createChatStream() {
        streamCalls++;
        throw new Error("stream must not be opened");
      },
    };
    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          stream: true,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 503);
      assert.equal(response.headers.get("content-type"), "application/json; charset=utf-8");
      const error = await response.json() as {
        error: { type: string; code: string };
      };
      assert.equal(error.error.type, "catalog_unavailable");
      assert.equal(error.error.code, "catalog_unavailable");
    });
    assert.equal(streamCalls, 0);
  });

  it("rejects malformed input before provider transport and malformed output as 502", async () => {
    let chatCalls = 0;
    const provider = {
      async chat() {
        chatCalls++;
        return {
          content: "untrusted",
          tool_calls: [],
          finish_reason: "stop",
          usage: null,
          reasoning_content: null,
          raw: null,
        };
      },
    };
    await withServer(provider, async (baseUrl) => {
      const invalid = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          messages: [{ role: "future-role", content: "hello" }],
        }),
      });
      assert.equal(invalid.status, 400);
      assert.equal(chatCalls, 0);

      for (const toolCall of [
        {
          type: "function",
          id: "call-1",
          function: { name: "lookup", arguments: { query: "wrapped" } },
        },
        {
          type: "function",
          function: { name: "lookup", arguments: "{}" },
        },
      ]) {
        const invalidToolCall = await fetch(`${baseUrl}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            messages: [{
              role: "assistant",
              content: null,
              tool_calls: [toolCall],
            }],
          }),
        });
        assert.equal(invalidToolCall.status, 400);
      }
      assert.equal(chatCalls, 0);

      const conflictingImageDetail = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          messages: [{
            role: "user",
            content: [{
              type: "image_url",
              image_url: { url: "data:image/png;base64,AAAA", detail: "low" },
              detail: "high",
            }],
          }],
        }),
      });
      assert.equal(conflictingImageDetail.status, 400);
      assert.equal(chatCalls, 0);

      for (const body of [
        { messages: [{ role: "user", content: "hello", name: "ignored" }] },
        {
          messages: [{
            role: "user",
            content: "hello",
            tool_calls: [{
              id: "call-1",
              type: "function",
              function: { name: "lookup", arguments: "{}" },
            }],
          }],
        },
        {
          messages: [{
            role: "assistant",
            content: null,
            tool_calls: [{
              id: "call-1",
              call_id: "legacy-call",
              type: "function",
              function: { name: "lookup", arguments: "{}" },
            }],
          }],
        },
        {
          messages: [{ role: "user", content: [{ type: "text", text: "hello", ignored: true }] }],
        },
        {
          messages: [{ role: "user", content: "hello" }],
          tools: [{ name: "lookup", parameters: { type: "object" } }],
        },
        {
          messages: [{ role: "user", content: "hello" }],
          tool_choice: { type: "function", function: { name: "lookup", ignored: true } },
        },
        {
          messages: [{ role: "user", content: "hello" }],
          text: { future_control: true },
        },
      ]) {
        const response = await fetch(`${baseUrl}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
        assert.equal(response.status, 400, await response.clone().text());
      }
      assert.equal(chatCalls, 0);

      const malformed = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(malformed.status, 502);
      const error = await malformed.json() as {
        error: { type: string; code: string };
      };
      assert.equal(error.error.type, "upstream_protocol_error");
      assert.equal(error.error.code, "upstream_protocol_error");
    });
    assert.equal(chatCalls, 1);
  });

  it("treats an explicit null prompt cache breakpoint as omitted", async () => {
    let receivedMessages: Array<{ structured_content?: Record<string, unknown>[] }> = [];
    const provider = {
      async chat(messages: Array<{ structured_content?: Record<string, unknown>[] }>) {
        receivedMessages = messages;
        return {
          content: "done",
          tool_calls: [],
          finish_reason: "stop",
          usage: null,
          reasoning_content: null,
          raw: null,
          response_id: "response-null-breakpoint",
        };
      },
    };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          messages: [{
            role: "user",
            content: [{
              type: "text",
              text: "hello",
              prompt_cache_breakpoint: null,
            }],
          }],
        }),
      });

      assert.equal(response.status, 200, await response.clone().text());
      assert.equal(
        Object.hasOwn(receivedMessages[0].structured_content?.[0] ?? {}, "prompt_cache_breakpoint"),
        false,
      );
    });
  });

  it("maps Chat Completions input_audio to a private audio URL", async () => {
    let receivedMessages: Array<{ structured_content?: Record<string, unknown>[] }> = [];
    const provider = {
      async chat(messages: Array<{ structured_content?: Record<string, unknown>[] }>) {
        receivedMessages = messages;
        return {
          content: "done",
          tool_calls: [],
          finish_reason: "stop",
          usage: null,
          reasoning_content: null,
          raw: null,
          response_id: "response-audio",
        };
      },
    };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          messages: [{
            role: "user",
            content: [{
              type: "input_audio",
              input_audio: { data: "AAAA", format: "mp3" },
            }],
          }],
        }),
      });

      assert.equal(response.status, 200, await response.clone().text());
      assert.deepEqual(receivedMessages[0].structured_content, [{
        type: "input_audio",
        audio_url: "data:audio/mp3;base64,AAAA",
      }]);
    });
  });

  it("rejects missing or empty message lists and empty image inspection before provider transport", async () => {
    let providerCalls = 0;
    const provider = {
      async chat() {
        providerCalls++;
        throw new Error("provider must not be called");
      },
      async compactMessages() {
        providerCalls++;
        throw new Error("provider must not be called");
      },
      async inspectImages() {
        providerCalls++;
        throw new Error("provider must not be called");
      },
    };
    const messageRoutes = [
      "/v1/chat/completions",
      "/v1/compact",
      "/v1/messages/count_tokens",
      "/v1/messages",
      "/v1/messages/compact",
    ];

    await withServer(provider, async (baseUrl) => {
      for (const route of messageRoutes) {
        for (const body of [{ model: "gpt-5.5" }, { model: "gpt-5.5", messages: [] }]) {
          const response = await fetch(`${baseUrl}${route}`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(body),
          });
          assert.equal(response.status, 400, `${route}: ${await response.clone().text()}`);
        }
      }

      const inspect = await fetch(`${baseUrl}/v1/inspect`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt: "inspect", images: [] }),
      });
      assert.equal(inspect.status, 400);

      const inspectUnknownImageField = await fetch(`${baseUrl}/v1/inspect`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          prompt: "inspect",
          images: [{ image_url: "data:image/png;base64,AAAA", future_control: true }],
        }),
      });
      assert.equal(inspectUnknownImageField.status, 400);
    });
    assert.equal(providerCalls, 0);
  });

  it("rejects unknown top-level request fields on every compatibility route", async () => {
    let providerCalls = 0;
    const provider = {
      async chat() {
        providerCalls++;
        throw new Error("provider must not be called");
      },
      async generateImage() {
        providerCalls++;
        throw new Error("provider must not be called");
      },
      async inspectImages() {
        providerCalls++;
        throw new Error("provider must not be called");
      },
      async compactMessages() {
        providerCalls++;
        throw new Error("provider must not be called");
      },
    };
    const requests = [
      ["/v1/chat/completions", { messages: [{ role: "user", content: "hello" }] }],
      ["/v1/images/generations", { prompt: "draw" }],
      ["/v1/inspect", { prompt: "inspect" }],
      ["/v1/compact", { messages: [{ role: "user", content: "history" }] }],
      ["/v1/messages/count_tokens", {
        model: "claude-sonnet-4-6",
        messages: [{ role: "user", content: "hello" }],
      }],
      ["/v1/messages", {
        model: "claude-sonnet-4-6",
        messages: [{ role: "user", content: "hello" }],
      }],
      ["/v1/messages/compact", {
        model: "claude-sonnet-4-6",
        messages: [{ role: "user", content: "history" }],
      }],
    ] as const;

    await withServer(provider, async (baseUrl) => {
      for (const [route, baseBody] of requests) {
        const response = await fetch(`${baseUrl}${route}`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "test-session",
          },
          body: JSON.stringify({ ...baseBody, unknown_future_field: true }),
        });
        assert.equal(response.status, 400, `${route}: ${await response.clone().text()}`);
      }
    });
    assert.equal(providerCalls, 0);
  });

  it("rejects image tools but accepts the explicit auto size", async () => {
    let providerCalls = 0;
    const provider = {
      async generateImage(_prompt: string, opts: { size?: string }) {
        providerCalls += 1;
        assert.equal(opts.size, "auto");
        return [{ result: "data:image/png;base64,AA" }];
      },
    };

    await withServer(provider, async (baseUrl) => {
      const unsupportedTools = await fetch(`${baseUrl}/v1/images/generations`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt: "draw", tools: [] }),
      });
      assert.equal(unsupportedTools.status, 400);
      assert.equal(providerCalls, 0);

      const automaticSize = await fetch(`${baseUrl}/v1/images/generations`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ prompt: "draw", size: "auto" }),
      });
      assert.equal(automaticSize.status, 200, await automaticSize.clone().text());
      assert.equal(providerCalls, 1);
    });
  });

  it("requires exact boolean memgen headers and rejects body/header conflicts", async () => {
    let chatCalls = 0;
    const provider = {
      async chat() {
        chatCalls++;
        throw new Error("provider must not be called");
      },
    };
    await withServer(provider, async (baseUrl) => {
      for (const request of [
        {
          headers: { "x-openai-memgen-request": "yes" },
          body: { messages: [{ role: "user", content: "hello" }] },
        },
        {
          headers: { "x-openai-memgen-request": "true" },
          body: {
            memgen_request: false,
            messages: [{ role: "user", content: "hello" }],
          },
        },
      ]) {
        const response = await fetch(`${baseUrl}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json", ...request.headers },
          body: JSON.stringify(request.body),
        });
        assert.equal(response.status, 400);
      }
    });
    assert.equal(chatCalls, 0);
  });

  it("rejects unsafe subagent values from request bodies and explicit headers", async () => {
    let chatCalls = 0;
    const provider = {
      async chat() {
        chatCalls++;
        throw new Error("provider must not be called");
      },
    };
    const routeBodies = [
      {
        route: "/v1/chat/completions",
        body: { messages: [{ role: "user", content: "hello" }] },
      },
      {
        route: "/v1/messages",
        body: { model: "claude-sonnet-4-6", messages: [{ role: "user", content: "hello" }] },
      },
    ];

    await withServer(provider, async (baseUrl) => {
      for (const { route, body } of routeBodies) {
        for (const request of [
          {
            headers: {} as Record<string, string>,
            body: { ...body, subagent: "LEAK ME" },
          },
          {
            headers: { "x-openai-subagent": "LEAK ME" } as Record<string, string>,
            body,
          },
        ]) {
          const response = await fetch(`${baseUrl}${route}`, {
            method: "POST",
            headers: { "content-type": "application/json", ...request.headers },
            body: JSON.stringify(request.body),
          });
          assert.equal(response.status, 400);
          assert.equal((await response.text()).includes("LEAK ME"), false);
        }
      }
    });
    assert.equal(chatCalls, 0);
  });
});

describe("Anthropic final response contract", () => {
  it("returns 502 when final usage or finish reason is absent", async () => {
    for (const missing of ["usage", "finish_reason"] as const) {
      const provider = {
        async chat() {
          return {
            content: "ok",
            tool_calls: [],
            finish_reason: missing === "finish_reason" ? null : "stop",
            usage: missing === "usage"
              ? null
              : { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
            reasoning_content: null,
            raw: null,
            response_id: "response-anthropic-contract",
          };
        },
      };
      await withServer(provider, async (baseUrl) => {
        const response = await fetch(`${baseUrl}/v1/messages`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "final-contract-session",
          },
          body: JSON.stringify({
            model: "gpt-5.5",
            max_tokens: 64,
            messages: [{ role: "user", content: "hello" }],
          }),
        });
        assert.equal(response.status, 502, await response.clone().text());
        const body = await response.json() as { error: { type: string } };
        assert.equal(body.error.type, "api_error");
      });
    }
  });

  it("reports final contract failures in-band without message_stop", async () => {
    for (const event of [
      { type: "finish", finish_reason: null, usage: { input_tokens: 1, output_tokens: 1, total_tokens: 2 } },
      { type: "finish", finish_reason: "stop" },
      { type: "web_search_call", id: "search-1", input: { query: "q" }, content: [] },
    ]) {
      const provider = {
        chatStream() {
          return (async function* () {
            yield event;
          })();
        },
      };
      await withServer(provider, async (baseUrl) => {
        const response = await fetch(`${baseUrl}/v1/messages`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "final-contract-session",
          },
          body: JSON.stringify({
            model: "gpt-5.5",
            max_tokens: 64,
            stream: true,
            messages: [{ role: "user", content: "hello" }],
          }),
        });
        assert.equal(response.status, 200);
        const text = await response.text();
        assert.match(text, /event: message_start/);
        assert.match(text, /event: error/);
        assert.equal(text.includes("event: message_stop"), false);
      });
    }
  });
});

describe("Anthropic compatibility helper routes", () => {
  it("streams a representable Claude Code tool when disabled thinking overrides ambient effort", async () => {
    const previous = process.env.CODEX_AS_API_RESPONSES_LITE;
    process.env.CODEX_AS_API_RESPONSES_LITE = "off";
    const auth = writeAuthFile();
    try {
      await withRecordingUpstream(async (upstreamUrl, requests) => {
        const provider = new ChatGPTOAuthProvider({
          model: "gpt-5.6-sol",
          baseUrl: upstreamUrl,
          authJsonPath: auth.authPath,
        });
        await withServer(provider, async (baseUrl) => {
          const response = await fetch(`${baseUrl}/v1/messages`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-claude-code-session-id": "web-search-session",
            },
            body: JSON.stringify({
              model: "claude-sonnet-4-5",
              max_tokens: 64,
              stream: true,
              system: "You are helpful.",
              tools: [{
                name: "lookup",
                input_schema: {
                  type: "object",
                  properties: { query: { type: "string" } },
                  required: ["query"],
                },
              }],
              thinking: { type: "disabled" },
              output_config: { effort: "high" },
              messages: [{ role: "user", content: "Search the web." }],
            }),
          });

          assert.equal(response.status, 200, await response.clone().text());
          assert.equal(response.headers.get("content-type"), "text/event-stream");
          const events = (await response.text())
            .trim()
            .split("\n\n")
            .map((block) => {
              const lines = block.split("\n");
              const eventLine = lines.find((line) => line.startsWith("event: "));
              const dataLine = lines.find((line) => line.startsWith("data: "));
              assert.ok(eventLine);
              assert.ok(dataLine);
              return {
                event: eventLine.slice(7),
                data: JSON.parse(dataLine.slice(6)) as Record<string, unknown>,
              };
            });
          assert.deepEqual(events.map(({ event, data }) => [event, data.type]), [
            ["message_start", "message_start"],
            ["content_block_start", "content_block_start"],
            ["content_block_delta", "content_block_delta"],
            ["content_block_stop", "content_block_stop"],
            ["message_delta", "message_delta"],
            ["message_stop", "message_stop"],
          ]);
          const messageDelta = events[4].data;
          assert.deepEqual(messageDelta.delta, {
            container: null,
            stop_reason: "end_turn",
            stop_sequence: null,
          });
          assert.equal(messageDelta.context_management, null);

          assert.equal(requests.length, 1);
          assert.equal(requests[0].path, "/responses");
          const upstream = requests[0].body;
          assert.equal((upstream.reasoning as Record<string, unknown>).effort, "none");
          assert.equal((upstream.tools as Record<string, unknown>[])[0].type, "function");
          assert.equal((upstream.tools as Record<string, unknown>[])[0].name, "lookup");
        }, {
          model: "gpt-5.6-sol",
          authPath: auth.authPath,
          codexConfig: {
            ...TEST_CONFIG,
            model: "gpt-5.6-sol",
            modelReasoningEffort: "high",
          },
        });
      });
    } finally {
      fs.rmSync(auth.directory, { recursive: true, force: true });
      if (previous == null) delete process.env.CODEX_AS_API_RESPONSES_LITE;
      else process.env.CODEX_AS_API_RESPONSES_LITE = previous;
    }
  });

  it("wires the Claude Code 2.1.220 request shape and cache hints to a known GPT model", async () => {
    const auth = writeAuthFile();
    try {
      await withRecordingUpstream(async (upstreamUrl, requests) => {
        const provider = new ChatGPTOAuthProvider({
          model: "gpt-5.6-sol",
          baseUrl: upstreamUrl,
          authJsonPath: auth.authPath,
        });
        await withServer(provider, async (baseUrl) => {
          const response = await fetch(`${baseUrl}/v1/messages?beta=true`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-claude-code-session-id": "claude-code-session-fixture",
            },
            body: JSON.stringify({
              model: "claude-sonnet-4-6",
              cache_control: { type: "ephemeral", ttl: "5m" },
              messages: [{
                role: "user",
                content: [{
                  type: "text",
                  text: "Reply OK",
                  cache_control: { type: "ephemeral" },
                }],
              }],
              system: [{
                type: "text",
                text: "You are a Claude agent.",
                cache_control: { type: "ephemeral", ttl: "1h" },
              }],
              tools: [{
                name: "lookup",
                description: "Lookup a value",
                input_schema: { type: "object", properties: {} },
                strict: true,
                cache_control: { type: "ephemeral" },
              }],
              max_tokens: 32_000,
              thinking: { type: "adaptive" },
              context_management: {
                edits: [{ type: "clear_thinking_20251015", keep: "all" }],
              },
              output_config: { effort: "max" },
              speed: "fast",
              stream: true,
            }),
          });

          assert.equal(response.status, 200, await response.clone().text());
          const events = (await response.text())
            .split("\n\n")
            .filter((block) => block.length > 0)
            .map((block) => {
              const dataLine = block.split("\n").find((line) => line.startsWith("data: "));
              return dataLine == null
                ? null
                : JSON.parse(dataLine.slice(6)) as Record<string, unknown>;
            })
            .filter((event): event is Record<string, unknown> => event !== null);
          const messageStart = events.find((event) => event.type === "message_start");
          const responseMessage = messageStart?.message as Record<string, unknown> | undefined;
          assert.equal(responseMessage?.model, "claude-sonnet-4-6");

          assert.equal(requests.length, 1);
          const upstream = requests[0].body;
          assert.equal(upstream.model, "gpt-5.6-sol");
          assert.deepEqual(upstream.reasoning, {
            effort: "max",
            summary: "auto",
            context: "all_turns",
          });
          assert.equal(upstream.service_tier, "priority");
          assert.equal(
            upstream.prompt_cache_key,
            crypto
              .createHash("sha256")
              .update(
                "codex-as-api:claude-code-session:claude-code-session-fixture",
                "utf8",
              )
              .digest("hex"),
          );
          const lookupTool = {
            type: "function",
            name: "lookup",
            description: "Lookup a value",
            parameters: { type: "object", properties: {} },
            strict: true,
          };
          assert.deepEqual(upstream.input, [
            {
              type: "additional_tools",
              role: "developer",
              tools: [lookupTool],
            },
            {
              type: "message",
              role: "developer",
              content: [{
                type: "input_text",
                text: "You are a Claude agent.",
              }],
            },
            {
              type: "message",
              role: "user",
              content: [{ type: "input_text", text: "Reply OK" }],
            },
          ]);
          assert.equal(hasNestedKey(upstream, "cache_control"), false);
          assert.equal(Object.hasOwn(upstream, "client_metadata"), false);
          assert.equal(Object.hasOwn(upstream, "output_config"), false);
          assert.equal(Object.hasOwn(upstream, "context_management"), false);
          assert.equal(Object.hasOwn(upstream, "speed"), false);
        }, { model: "gpt-5.6-sol", authPath: auth.authPath });
      });
    } finally {
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });

  it("derives stable Claude cache affinity while explicit keys take precedence", async () => {
    const options: Record<string, unknown>[] = [];
    const provider = {
      async chat(_messages: unknown, opts: Record<string, unknown>) {
        options.push(opts);
        return {
          content: "done",
          tool_calls: [],
          finish_reason: "stop",
          usage: {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            cached_tokens: 0,
          },
          reasoning_content: null,
          raw: null,
        };
      },
    };

    await withServer(provider, async (baseUrl) => {
      const send = async (
        sessionId?: string,
        promptCacheKey?: string,
      ): Promise<Response> => fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          ...(sessionId == null
            ? {}
            : { "x-claude-code-session-id": sessionId }),
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          max_tokens: 64,
          messages: [{ role: "user", content: "hello" }],
          ...(promptCacheKey == null
            ? {}
            : { prompt_cache_key: promptCacheKey }),
        }),
      });

      for (const response of [
        await send("session-a"),
        await send("session-a"),
        await send("session-b"),
        await send("session-a", "explicit-cache-key"),
      ]) {
        assert.equal(response.status, 200, await response.clone().text());
      }
      const withoutSession = await send();
      assert.equal(withoutSession.status, 400);
      const invalidBlankSession = await send(" ", "explicit-cache-key-with-blank-session");
      assert.equal(invalidBlankSession.status, 400);
    });

    const hash = (sessionId: string): string => crypto
      .createHash("sha256")
      .update(`codex-as-api:claude-code-session:${sessionId}`, "utf8")
      .digest("hex");
    assert.equal(options[0].promptCacheKey, hash("session-a"));
    assert.equal(options[1].promptCacheKey, options[0].promptCacheKey);
    assert.equal(options[2].promptCacheKey, hash("session-b"));
    assert.notEqual(options[2].promptCacheKey, options[0].promptCacheKey);
    assert.equal(options[3].promptCacheKey, "explicit-cache-key");
    for (const opts of options) {
      assert.equal(opts.codexMetadata, false);
      assert.equal(Object.hasOwn(opts, "clientMetadata"), false);
      assert.equal(Object.hasOwn(opts, "previousResponseId"), false);
    }
  });

  it("rejects duplicate Claude Code session header lines before provider transport", async () => {
    let chatCalls = 0;
    const provider = {
      async chat() {
        chatCalls += 1;
        throw new Error("provider must not be called");
      },
    };

    await withServer(provider, async (baseUrl) => {
      const target = new URL("/v1/messages", baseUrl);
      const payload = JSON.stringify({
        model: "claude-sonnet-4-5",
        messages: [{ role: "user", content: "hello" }],
      });
      const response = await new Promise<{ status: number; body: string }>((resolve, reject) => {
        const request = httpRequest({
          hostname: target.hostname,
          port: target.port,
          path: target.pathname,
          method: "POST",
          headers: [
            "host",
            target.host,
            "content-type",
            "application/json",
            "content-length",
            String(Buffer.byteLength(payload)),
            "x-claude-code-session-id",
            "session-a",
            "x-claude-code-session-id",
            "session-b",
          ],
        }, (incoming) => {
          const chunks: Buffer[] = [];
          incoming.on("data", (chunk: Buffer) => chunks.push(chunk));
          incoming.on("end", () => resolve({
            status: incoming.statusCode ?? 0,
            body: Buffer.concat(chunks).toString("utf8"),
          }));
        });
        request.on("error", reject);
        request.end(payload);
      });

      assert.equal(response.status, 400, response.body);
      assert.deepEqual(JSON.parse(response.body), {
        type: "error",
        error: {
          type: "invalid_request_error",
          message: "x-claude-code-session-id must be provided at most once",
        },
      });
      assert.equal(chatCalls, 0);
    });
  });

  it("rejects malformed Claude cache controls before provider transport", async () => {
    let chatCalls = 0;
    const provider = {
      async chat() {
        chatCalls += 1;
        throw new Error("provider must not be called");
      },
    };
    const invalidControls = [
      { cache_control: { type: "persistent" } },
      { cache_control: { type: "ephemeral", ttl: "30m" } },
      { cache_control: { type: "ephemeral", extra: true } },
      {
        system: [{
          type: "text",
          text: "system",
          cache_control: "ephemeral",
        }],
      },
      {
        messages: [{
          role: "user",
          content: "hello",
          cache_control: { type: "persistent" },
        }],
      },
      {
        messages: [{
          role: "user",
          content: [{
            type: "text",
            text: "hello",
            cache_control: { type: "ephemeral", ttl: null },
          }],
        }],
      },
      {
        tools: [{
          name: "lookup",
          input_schema: { type: "object" },
          cache_control: [],
        }],
      },
    ];

    await withServer(provider, async (baseUrl) => {
      for (const fields of invalidControls) {
        const response = await fetch(`${baseUrl}/v1/messages`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "claude-sonnet-4-5",
            max_tokens: 128,
            system: "system",
            messages: [{ role: "user", content: "hello" }],
            ...fields,
          }),
        });
        assert.equal(response.status, 400);
        const error = await response.json() as {
          type: string;
          error: { type: string; message: string };
        };
        assert.equal(error.type, "error");
        assert.equal(error.error.type, "invalid_request_error");
      }
    });
    assert.equal(chatCalls, 0);
  });

  it("rejects previous_response_id on stateless Claude messages", async () => {
    let chatCalls = 0;
    const provider = {
      async chat() {
        chatCalls += 1;
        throw new Error("provider must not be called");
      },
    };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "test-session",
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          max_tokens: 128,
          previous_response_id: "response-not-a-claude-session",
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 400);
      const error = await response.json() as {
        type: string;
        error: { type: string; message: string };
      };
      assert.equal(error.type, "error");
      assert.equal(error.error.type, "invalid_request_error");
    });
    assert.equal(chatCalls, 0);
  });

  it("routes a Claude facade only to the explicitly configured live backend", async () => {
    let providerModel: string | undefined;
    const provider = {
      async chat(_messages: unknown, opts: { model?: string }) {
        providerModel = opts.model;
        return {
          content: "done",
          tool_calls: [],
          finish_reason: "stop",
          usage: {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            cached_tokens: 0,
          },
          reasoning_content: null,
          raw: null,
        };
      },
    };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "test-session",
        },
        body: JSON.stringify({
          model: "claude-fable-5",
          max_tokens: 64,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 200);
      const body = await response.json() as Record<string, unknown>;
      assert.equal(body.model, "claude-fable-5");
      assert.equal(providerModel, "gpt-5.5");
    }, { model: "gpt-5.5" });
  });

  it("accepts direct live catalog slugs on every Anthropic-compatible route", async () => {
    const routedModels: string[] = [];
    const provider = {
      async chat(_messages: unknown, opts: { model?: string }) {
        routedModels.push(opts.model ?? "");
        return {
          content: "done",
          tool_calls: [],
          finish_reason: "stop",
          usage: {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            cached_tokens: 0,
          },
          reasoning_content: null,
          raw: null,
        };
      },
      async compactMessages(_messages: unknown, opts: { model?: string }) {
        routedModels.push(opts.model ?? "");
        return "checkpoint";
      },
    };

    await withServer(provider, async (baseUrl) => {
      const messages = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "test-session",
        },
        body: JSON.stringify({
          model: "gpt-5.6-sol",
          max_tokens: 64,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(messages.status, 200, await messages.clone().text());
      assert.equal((await messages.json() as { model: string }).model, "gpt-5.6-sol");

      const count = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "gpt-5.6-sol",
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(count.status, 200, await count.clone().text());

      const compact = await fetch(`${baseUrl}/v1/messages/compact`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "test-session",
        },
        body: JSON.stringify({
          model: "gpt-5.6-sol",
          max_tokens: 64,
          messages: [{ role: "user", content: "history" }],
        }),
      });
      assert.equal(compact.status, 200, await compact.clone().text());
      assert.deepEqual(await compact.json(), { checkpoint: "checkpoint" });
    }, { model: null });

    assert.deepEqual(routedModels, ["gpt-5.6-sol", "gpt-5.6-sol"]);
  });

  it("rejects a Claude facade when no backend is explicitly configured", async () => {
    let chatCalls = 0;
    const provider = {
      async chat() {
        chatCalls++;
        throw new Error("provider must not be called");
      },
    };
    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-6",
          max_tokens: 64,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 400);
      const body = await response.json() as { error: { type: string } };
      assert.equal(body.error.type, "invalid_request_error");
    }, { model: null });
    assert.equal(chatCalls, 0);
  });

  it("maps Anthropic speed and rejects conflicting service tiers", async () => {
    const serviceTiers: Array<string | undefined> = [];
    const provider = {
      async chat(_messages: unknown, opts: { serviceTier?: string }) {
        serviceTiers.push(opts.serviceTier);
        return {
          content: "done",
          tool_calls: [],
          finish_reason: "stop",
          usage: {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            cached_tokens: 0,
          },
          reasoning_content: null,
          raw: null,
        };
      },
    };

    await withServer(provider, async (baseUrl) => {
      for (const fields of [
        { speed: "fast" },
        { speed: "standard" },
        { speed: "fast", service_tier: "priority" },
      ]) {
        const response = await fetch(`${baseUrl}/v1/messages`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "test-session",
          },
          body: JSON.stringify({
            model: "claude-sonnet-4-6",
            max_tokens: 64,
            messages: [{ role: "user", content: "hello" }],
            ...fields,
          }),
        });
        assert.equal(response.status, 200);
      }
      assert.deepEqual(serviceTiers, ["fast", "default", "fast"]);

      for (const fields of [
        { speed: "fast", service_tier: "default" },
        { speed: "warp" },
      ]) {
        const response = await fetch(`${baseUrl}/v1/messages`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "claude-sonnet-4-6",
            max_tokens: 64,
            messages: [{ role: "user", content: "hello" }],
            ...fields,
          }),
        });
        assert.equal(response.status, 400);
        const body = await response.json() as {
          type: string;
          error: { type: string };
        };
        assert.equal(body.type, "error");
        assert.equal(body.error.type, "invalid_request_error");
      }
      assert.deepEqual(serviceTiers, ["fast", "default", "fast"]);
    }, { model: "gpt-5.5" });
  });

  it("maps Anthropic parallel control through the real provider", async () => {
    const auth = writeAuthFile();
    try {
      await withRecordingUpstream(async (upstreamUrl, requests) => {
        const provider = new ChatGPTOAuthProvider({
          model: "gpt-5.5",
          baseUrl: upstreamUrl,
          authJsonPath: auth.authPath,
        });
        await withServer(provider, async (baseUrl) => {
          for (const [disableParallel, expected] of [[true, false], [false, true]] as const) {
            const response = await fetch(`${baseUrl}/v1/messages`, {
              method: "POST",
              headers: {
                "content-type": "application/json",
                "x-claude-code-session-id": "test-session",
              },
              body: JSON.stringify({
                model: "gpt-5.5",
                max_tokens: 64,
                system: "system",
                messages: [{ role: "user", content: "hello" }],
                responses_lite: false,
                tool_choice: {
                  type: "auto",
                  disable_parallel_tool_use: disableParallel,
                },
              }),
            });
            assert.equal(response.status, 200, await response.clone().text());
            assert.equal(requests.at(-1)?.body.parallel_tool_calls, expected);
          }

          const liteResponse = await fetch(`${baseUrl}/v1/messages`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-claude-code-session-id": "test-session",
            },
            body: JSON.stringify({
              model: "gpt-5.6-sol",
              max_tokens: 64,
              system: "system",
              messages: [{ role: "user", content: "hello" }],
              responses_lite: true,
              tool_choice: {
                type: "auto",
                disable_parallel_tool_use: false,
              },
            }),
          });
          assert.equal(liteResponse.status, 400);
          assert.equal(requests.length, 2);
        }, { model: "gpt-5.5", authPath: auth.authPath });
      });
    } finally {
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });

  it("accepts only the exact no-op context_management shape", async () => {
    let chatCalls = 0;
    const provider = {
      async chat() {
        chatCalls += 1;
        return {
          content: "done",
          tool_calls: [],
          finish_reason: "stop",
          usage: {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            cached_tokens: 0,
          },
          reasoning_content: null,
          raw: null,
        };
      },
    };
    const accepted = {
      edits: [{ type: "clear_thinking_20251015", keep: "all" }],
    };

    await withServer(provider, async (baseUrl) => {
      for (const route of ["/v1/messages/count_tokens", "/v1/messages"] as const) {
        const response = await fetch(`${baseUrl}${route}`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            ...(route === "/v1/messages"
              ? { "x-claude-code-session-id": "test-session" }
              : {}),
          },
          body: JSON.stringify({
            model: "claude-fable-5",
            messages: [{ role: "user", content: "hello" }],
            context_management: accepted,
            ...(route === "/v1/messages" ? { max_tokens: 64 } : {}),
          }),
        });
        assert.equal(response.status, 200);
      }
      assert.equal(chatCalls, 1);

      const invalidValues = [
        { edits: [{ type: "clear_thinking_20251015", keep: "recent" }] },
        {
          edits: [{ type: "clear_thinking_20251015", keep: "all" }],
          extra: true,
        },
        {
          edits: [{
            type: "clear_tool_uses_20250919",
            trigger: { type: "input_tokens", value: 30_000 },
          }],
        },
      ];
      for (const contextManagement of invalidValues) {
        for (const route of ["/v1/messages/count_tokens", "/v1/messages"] as const) {
          const response = await fetch(`${baseUrl}${route}`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-claude-code-session-id": "test-session",
            },
            body: JSON.stringify({
              model: "claude-fable-5",
              messages: [{ role: "user", content: "hello" }],
              context_management: contextManagement,
              ...(route === "/v1/messages" ? { max_tokens: 64 } : {}),
            }),
          });
          assert.equal(response.status, 400);
          const body = await response.json() as {
            type: string;
            error: { type: string };
          };
          assert.equal(body.type, "error");
          assert.equal(body.error.type, "invalid_request_error");
        }
      }
      assert.equal(chatCalls, 1);
    });
  });

  it("rejects unrepresentable Claude beta controls before provider transport", async () => {
    let chatCalls = 0;
    const provider = {
      async chat() {
        chatCalls += 1;
        throw new Error("provider must not be called");
      },
    };
    const unsupported = [
      { output_config: { task_budget: { type: "tokens", total: 20_000 } } },
      { output_config: { effort: "" } },
      { output_config: { effort: "ultra" } },
      { output_config: { unknown_control: true } },
      { text: { format: { type: "text" } } },
      { reasoning_effort: "low", output_config: { effort: "high" } },
      { tools: [{ name: "lookup", input_schema: {}, strict: "true" }] },
      { tools: [{ name: "lookup", input_schema: {}, defer_loading: true }] },
      { tools: [{ name: "lookup", input_schema: {}, defer_loading: false }] },
      { tools: [{ name: "lookup", input_schema: {}, eager_input_streaming: true }] },
      { tools: [{ name: "lookup", input_schema: {}, eager_input_streaming: false }] },
      { tools: [{ name: "lookup", input_schema: {}, future_control: true }] },
      { messages: [{ role: "user", content: "hello", name: "ignored" }] },
      {
        messages: [{
          role: "user",
          content: [{ type: "text", text: "hello", future_control: true }],
        }],
      },
      {
        messages: [{
          role: "user",
          content: [{
            type: "image",
            source: { type: "base64", media_type: "image/png", data: "" },
          }],
        }],
      },
      {
        messages: [{
          role: "user",
          content: [{
            type: "tool_result",
            tool_use_id: "call-image",
            content: [{ type: "image", source: { type: "url", url: "" } }],
          }],
        }],
      },
    ];

    await withServer(provider, async (baseUrl) => {
      for (const controls of unsupported) {
        const response = await fetch(`${baseUrl}/v1/messages`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "claude-sonnet-4-6",
            messages: [{ role: "user", content: "hello" }],
            ...controls,
          }),
        });
        assert.equal(response.status, 400);
        const body = await response.json() as {
          type: string;
          error: { type: string };
        };
        assert.equal(body.type, "error");
        assert.equal(body.error.type, "invalid_request_error");
      }
      assert.equal(chatCalls, 0);
    });
  });

  it("rejects lossy output formats and image sources on every Anthropic route", async () => {
    let providerCalls = 0;
    const provider = {
      async chat() {
        providerCalls += 1;
        throw new Error("provider must not be called");
      },
      async compactMessages() {
        providerCalls += 1;
        throw new Error("provider must not be called");
      },
    };
    const base = {
      model: "claude-sonnet-4-6",
      system: "system",
      messages: [{ role: "user", content: "hello" }],
    };
    const invalidBodies = [
      { ...base, output_format: "json" },
      { ...base, output_config: { format: "json" } },
      { ...base, output_format: { type: "future" } },
      {
        ...base,
        output_format: { type: "json_object", schema: { type: "object" } },
      },
      {
        ...base,
        output_config: {
          format: {
            type: "json_schema",
            schema: { type: "object" },
            extra: true,
          },
        },
      },
      {
        ...base,
        output_format: {
          type: "json_schema",
          schema: { type: "object" },
          name: "",
        },
      },
      {
        ...base,
        output_format: {
          type: "json_schema",
          schema: { type: "object" },
          name: "my schema!",
        },
      },
      {
        ...base,
        output_format: {
          type: "json_schema",
          schema: { type: "object" },
          description: 42,
        },
      },
      {
        ...base,
        output_format: {
          type: "json_schema",
          schema: { type: "object" },
          strict: "true",
        },
      },
      {
        ...base,
        output_format: { type: "json_object" },
        output_config: {
          format: { type: "json_schema", schema: { type: "object" } },
        },
      },
      {
        ...base,
        messages: [{
          role: "user",
          content: [{
            type: "image",
            source: { type: "file", file_id: "file-1" },
          }],
        }],
      },
      {
        ...base,
        messages: [{
          role: "user",
          content: [{
            type: "image",
            source: { type: "base64", media_type: 42, data: "AAAA" },
          }],
        }],
      },
    ];

    await withServer(provider, async (baseUrl) => {
      for (const route of [
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/v1/messages/compact",
      ]) {
        for (const body of invalidBodies) {
          const response = await fetch(`${baseUrl}${route}`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify(body),
          });
          assert.equal(response.status, 400);
        }
      }
      assert.equal(providerCalls, 0);
    });
  });

  it("returns a single-pass count_tokens estimate without calling provider", async () => {
    const provider = {
      async countTokens() {
        throw new Error("count_tokens must not call the Codex backend");
      },
    };
    const tools = [{
      name: "lookup",
      description: "Search docs",
      input_schema: { type: "object", properties: { query: { type: "string" } } },
    }];

    await withServer(provider, async (baseUrl) => {
      const res = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          system: "You are helpful.",
          tools,
          messages: [{ role: "user", content: "hello" }],
        }),
      });

      assert.equal(res.status, 200);
      const body = await res.json() as {
        input_tokens: number;
        context_window: number;
        auto_compact_token_limit: number;
      };
      assert.equal(body.input_tokens, 48);
      assert.ok(body.context_window >= body.auto_compact_token_limit);

      for (const unsupported of [
        { multi_agent: { enabled: true } },
        { programmatic_tool_calling: { enabled: true } },
      ]) {
        const invalid = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "claude-sonnet-4-5",
            messages: [{ role: "user", content: "hello" }],
            ...unsupported,
          }),
        });
        assert.equal(invalid.status, 400);
        const error = await invalid.json() as {
          type: string;
          error: { type: string };
        };
        assert.equal(error.type, "error");
        assert.equal(error.error.type, "invalid_request_error");
      }
    });
  });

  it("reports count_tokens limits for the effective Anthropic backend model", async () => {
    await withServer({}, async (baseUrl) => {
      const count = async (model: string) => {
        const response = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model,
            messages: [{ role: "user", content: "hello" }],
          }),
        });
        assert.equal(response.status, 200);
        return await response.json() as {
          context_window: number;
          auto_compact_token_limit: number;
        };
      };

      const claudeBackend = await count("claude-fable-5");
      assert.equal(claudeBackend.context_window, 258_400);
      assert.equal(claudeBackend.auto_compact_token_limit, 244_800);
    }, { model: "gpt-5.5" });
  });

  it("uses max_context_window when the live context_window is absent", async () => {
    const raw = rawModel("gpt-max-only", { max_context_window: 120_000 });
    delete raw.context_window;
    const snapshot = parseModelCatalog({ models: [raw] }, {
      key: modelCatalogCacheKey("test-account", "https://catalog.test", "0.153.3"),
      etag: '"max-only"',
      fetchedAt: 1_000,
      expiresAt: 301_000,
    });
    const capability = snapshot.models[0];
    const provider = {
      async catalogSnapshot() {
        return snapshot;
      },
      async prepareModel() {
        return {
          slug: capability.slug,
          accountId: "test-account",
          capability,
          snapshot,
        };
      },
    };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "gpt-max-only",
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 200);
      const body = await response.json() as {
        context_window: number;
        auto_compact_token_limit: number;
      };
      assert.equal(body.context_window, 114_000);
      assert.equal(body.auto_compact_token_limit, 108_000);
    }, { model: null });
  });

  it("derives the compaction limit exactly at the JSON safe-integer boundary", async () => {
    const contextWindow = Number.MAX_SAFE_INTEGER;
    const raw = rawModel("gpt-safe-integer-context", {
      context_window: contextWindow,
      max_context_window: contextWindow,
      auto_compact_token_limit: undefined,
    });
    const snapshot = parseModelCatalog({ models: [raw] }, {
      key: modelCatalogCacheKey("test-account", "https://catalog.test", "0.153.3"),
      etag: '"safe-integer-context"',
      fetchedAt: 1_000,
      expiresAt: 301_000,
    });
    const capability = snapshot.models[0];
    const provider = {
      async catalogSnapshot() {
        return snapshot;
      },
      async prepareModel() {
        return {
          slug: capability.slug,
          accountId: "test-account",
          capability,
          snapshot,
        };
      },
    };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: capability.slug,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 200, await response.clone().text());
      const body = await response.json() as {
        context_window: number;
        auto_compact_token_limit: number;
      };
      assert.equal(
        body.context_window,
        Math.trunc((contextWindow * capability.effectiveContextWindowPercent) / 100),
      );
      assert.equal(body.auto_compact_token_limit, 8_106_479_329_266_891);
    }, { model: null });
  });

  it("fails health and count when the saturated effective context is not JSON-safe", async () => {
    const raw = rawModel("gpt-effective-context-overflow", {
      context_window: Number.MAX_SAFE_INTEGER,
      max_context_window: Number.MAX_SAFE_INTEGER,
      effective_context_window_percent: Number.MAX_SAFE_INTEGER,
    });
    const snapshot = parseModelCatalog({ models: [raw] }, {
      key: modelCatalogCacheKey("test-account", "https://catalog.test", "0.153.3"),
      etag: '"effective-context-overflow"',
      fetchedAt: 1_000,
      expiresAt: 301_000,
    });
    const capability = snapshot.models[0];
    const provider = {
      async catalogSnapshot() {
        return snapshot;
      },
      async prepareModel() {
        return {
          slug: capability.slug,
          accountId: "test-account",
          capability,
          snapshot,
        };
      },
    };

    await withServer(provider, async (baseUrl) => {
      const health = await fetch(`${baseUrl}/health`);
      assert.equal(health.status, 503);
      const healthBody = await health.json() as { error: { type: string } };
      assert.equal(healthBody.error.type, "catalog_unavailable");

      const count = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: capability.slug,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(count.status, 503);
      const countBody = await count.json() as { error: { type: string } };
      assert.equal(countBody.error.type, "api_error");
    }, { model: capability.slug });
  });

  it("uses configured context limits and clamps only to a live maximum", async () => {
    const cases = [
      {
        slug: "gpt-context-only",
        contextWindow: 120_000,
        maxContextWindow: undefined,
        configuredContextWindow: 200_000,
        expectedContextWindow: 200_000,
      },
      {
        slug: "gpt-configured-expansion",
        contextWindow: 120_000,
        maxContextWindow: 200_000,
        configuredContextWindow: 150_000,
        expectedContextWindow: 150_000,
      },
      {
        slug: "gpt-configured-clamp",
        contextWindow: 120_000,
        maxContextWindow: 200_000,
        configuredContextWindow: 250_000,
        expectedContextWindow: 200_000,
      },
    ];

    for (const testCase of cases) {
      const raw = rawModel(testCase.slug, {
        context_window: testCase.contextWindow,
        max_context_window: testCase.maxContextWindow,
      });
      const snapshot = parseModelCatalog({ models: [raw] }, {
        key: modelCatalogCacheKey("test-account", "https://catalog.test", "0.153.3"),
        etag: `"${testCase.slug}"`,
        fetchedAt: 1_000,
        expiresAt: 301_000,
      });
      const capability = snapshot.models[0];
      const provider = {
        async catalogSnapshot() {
          return snapshot;
        },
        async prepareModel() {
          return {
            slug: capability.slug,
            accountId: "test-account",
            capability,
            snapshot,
          };
        },
      };

      await withServer(provider, async (baseUrl) => {
        const response = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: testCase.slug,
            messages: [{ role: "user", content: "hello" }],
          }),
        });
        assert.equal(response.status, 200);
        const body = await response.json() as {
          context_window: number;
          auto_compact_token_limit: number;
        };
        assert.equal(body.context_window, Math.trunc(testCase.expectedContextWindow * 0.95));
        assert.equal(
          body.auto_compact_token_limit,
          Math.floor(testCase.expectedContextWindow * 0.9),
        );
      }, {
        model: null,
        codexConfig: {
          ...TEST_CONFIG,
          modelContextWindow: testCase.configuredContextWindow,
        },
      });
    }
  });

  it("fails count_tokens when a selected live limit is non-positive at its use boundary", async () => {
    const cases = [
      {
        slug: "gpt-zero-context",
        overrides: { context_window: 0, max_context_window: 100 },
        codexConfig: TEST_CONFIG,
        expectedStatus: 503,
      },
      {
        slug: "gpt-zero-max-context",
        overrides: { context_window: undefined, max_context_window: 0 },
        codexConfig: TEST_CONFIG,
        expectedStatus: 503,
      },
      {
        slug: "gpt-negative-auto-compact",
        overrides: { context_window: 100, auto_compact_token_limit: -1 },
        codexConfig: TEST_CONFIG,
        expectedStatus: 200,
      },
      {
        slug: "gpt-zero-config-bound",
        overrides: { context_window: 100, max_context_window: 0 },
        codexConfig: { ...TEST_CONFIG, modelContextWindow: 50 },
        expectedStatus: 503,
      },
    ];

    for (const testCase of cases) {
      const raw = rawModel(testCase.slug, testCase.overrides);
      const snapshot = parseModelCatalog({ models: [raw] }, {
        key: modelCatalogCacheKey("test-account", "https://catalog.test", "0.153.3"),
        etag: `"${testCase.slug}"`,
        fetchedAt: 1_000,
        expiresAt: 301_000,
      });
      const capability = snapshot.models[0];
      const provider = {
        async catalogSnapshot() {
          return snapshot;
        },
        async prepareModel() {
          return {
            slug: capability.slug,
            accountId: "test-account",
            capability,
            snapshot,
          };
        },
      };

      await withServer(provider, async (baseUrl) => {
        const response = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: testCase.slug,
            messages: [{ role: "user", content: "hello" }],
          }),
        });
        assert.equal(response.status, testCase.expectedStatus);
        const body = await response.json() as Record<string, unknown>;
        if (testCase.expectedStatus === 200) {
          assert.equal(body.auto_compact_token_limit, -1);
        } else {
          const error = body.error as { type: string; message: string };
          assert.equal(error.type, "api_error");
          assert.equal(error.message, "authenticated model catalog is unavailable");
        }
      }, {
        model: testCase.slug,
        codexConfig: testCase.codexConfig,
      });
    }
  });

  it("keeps health and count metadata nullable when context is unavailable", async () => {
    const raw = rawModel("gpt-no-context");
    delete raw.context_window;
    delete raw.max_context_window;
    const snapshot = parseModelCatalog({ models: [raw] }, {
      key: modelCatalogCacheKey("test-account", "https://catalog.test", "0.153.3"),
      etag: '"no-context"',
      fetchedAt: 1_000,
      expiresAt: 301_000,
    });
    const capability = snapshot.models[0];
    const provider = {
      async catalogSnapshot() {
        return snapshot;
      },
      async prepareModel() {
        return {
          slug: capability.slug,
          accountId: "test-account",
          capability,
          snapshot,
        };
      },
    };
    await withServer(provider, async (baseUrl) => {
      const countResponse = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "gpt-no-context",
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(countResponse.status, 200);
      const count = await countResponse.json() as Record<string, unknown>;
      assert.equal(count.context_window, null);
      assert.equal(count.auto_compact_token_limit, null);

      const healthResponse = await fetch(`${baseUrl}/health`);
      assert.equal(healthResponse.status, 200);
      const health = await healthResponse.json() as Record<string, unknown>;
      assert.equal(health.catalog_status, "fresh");
      assert.equal(health.model, "gpt-no-context");
      assert.equal(health.context_window, null);
      assert.equal(health.auto_compact_token_limit, null);
    }, { model: "gpt-no-context" });
  });

  it("uses explicit configured limits without live context metadata", async () => {
    const raw = rawModel("gpt-configured-no-context");
    delete raw.context_window;
    delete raw.max_context_window;
    const snapshot = parseModelCatalog({ models: [raw] }, {
      key: modelCatalogCacheKey("test-account", "https://catalog.test", "0.153.3"),
      etag: '"configured-no-context"',
      fetchedAt: 1_000,
      expiresAt: 301_000,
    });
    const capability = snapshot.models[0];
    const provider = {
      async catalogSnapshot() {
        return snapshot;
      },
      async prepareModel() {
        return {
          slug: capability.slug,
          accountId: "test-account",
          capability,
          snapshot,
        };
      },
    };
    const codexConfig = {
      ...TEST_CONFIG,
      modelContextWindow: 200_000,
      modelAutoCompactTokenLimit: 100_000,
    };

    await withServer(provider, async (baseUrl) => {
      const countResponse = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: capability.slug,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(countResponse.status, 200);
      const count = await countResponse.json() as Record<string, unknown>;
      assert.equal(count.context_window, 190_000);
      assert.equal(count.auto_compact_token_limit, 100_000);

      const healthResponse = await fetch(`${baseUrl}/health`);
      assert.equal(healthResponse.status, 200);
      const health = await healthResponse.json() as Record<string, unknown>;
      assert.equal(health.model, capability.slug);
      assert.equal(health.context_window, 190_000);
      assert.equal(health.auto_compact_token_limit, 100_000);
    }, { model: capability.slug, codexConfig });
  });

  it("rejects unsupported count_tokens metadata instead of silently ignoring it", async () => {
    const content = "abcd".repeat(1_000);
    await withServer({}, async (baseUrl) => {
      const count = async (extra: Record<string, unknown>) => {
        const res = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "claude-sonnet-4-5",
            messages: [{ role: "user", content }],
            ...extra,
          }),
        });
        assert.equal(res.status, 200);
        return (await res.json() as { input_tokens: number }).input_tokens;
      };

      const plain = await count({});
      const invalid = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          messages: [{ role: "user", content }],
          metadata: { diagnostic: "x".repeat(4_000) },
        }),
      });
      assert.equal(invalid.status, 400);
      const invalidStream = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          messages: [{ role: "user", content }],
          stream: false,
        }),
      });
      assert.equal(invalidStream.status, 400);
      const invalidMaxTokens = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          max_tokens: 1024,
          messages: [{ role: "user", content }],
        }),
      });
      assert.equal(invalidMaxTokens.status, 400);
      const invalidStop = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          stop_sequences: ["STOP"],
          messages: [{ role: "user", content }],
        }),
      });
      assert.equal(invalidStop.status, 400);
      assert.equal(plain, 1_012);
    });
  });

  it("rejects OpenAI-only null controls on Anthropic routes before provider transport", async () => {
    let providerCalls = 0;
    const provider = {
      async prepareModel() {
        providerCalls += 1;
        throw new Error("provider must not be called");
      },
      async chat() {
        providerCalls += 1;
        throw new Error("provider must not be called");
      },
    };
    const controls = [
      "frequency_penalty",
      "presence_penalty",
      "seed",
      "logprobs",
      "top_logprobs",
      "logit_bias",
      "n",
      "response_format",
      "stream_options",
      "modalities",
      "audio",
      "prediction",
      "web_search_options",
      "store",
      "user",
      "quality",
      "style",
      "background",
      "moderation",
      "output_compression",
    ];

    await withServer(provider, async (baseUrl) => {
      for (const route of ["/v1/messages", "/v1/messages/count_tokens"]) {
        for (const control of controls) {
          const response = await fetch(`${baseUrl}${route}`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.5",
              messages: [{ role: "user", content: "hello" }],
              ...(route === "/v1/messages" ? { max_tokens: 1024 } : {}),
              [control]: null,
            }),
          });
          assert.equal(response.status, 400, `${route} must reject ${control}: null`);
        }
      }
    });
    assert.equal(providerCalls, 0);
  });

  it("counts multilingual UTF-8 text with o200k_base", async () => {
    await withServer({}, async (baseUrl) => {
      const content = "hello 안녕 👋";
      const res = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          messages: [{ role: "user", content }],
        }),
      });
      assert.equal(res.status, 200);
      const body = await res.json() as { input_tokens: number };
      assert.equal(body.input_tokens, 17);
    });
  });

  it("counts image input once without tokenizing base64 payload bytes", async () => {
    await withServer({}, async (baseUrl) => {
      const count = async (source: Record<string, unknown>) => {
        const res = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "claude-sonnet-4-5",
            messages: [{
              role: "user",
              content: [{ type: "image", source }],
            }],
          }),
        });
        assert.equal(res.status, 200);
        return (await res.json() as { input_tokens: number }).input_tokens;
      };

      const url = await count({ type: "url", url: "https://example.com/image.png" });
      const base64 = await count({
        type: "base64",
        media_type: "image/png",
        data: "AAAA".repeat(4_000),
      });
      assert.equal(url, 8_512);
      assert.equal(base64, url);
    });
  });

  it("routes Anthropic compact requests through the explicit live backend", async () => {
    const auth = writeAuthFile();
    try {
      await withRecordingUpstream(async (upstreamUrl, requests) => {
        const provider = new ChatGPTOAuthProvider({
          model: "gpt-5.5",
          baseUrl: upstreamUrl,
          authJsonPath: auth.authPath,
        });
        await withServer(provider, async (baseUrl) => {
          const common = {
            max_tokens: 64,
            messages: [{ role: "user", content: "history" }],
            context_management: {
              edits: [{ type: "clear_thinking_20251015", keep: "all" }],
            },
          };
          const known = await fetch(`${baseUrl}/v1/messages/compact`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-claude-code-session-id": "test-session",
            },
            body: JSON.stringify({
              ...common,
              model: "claude-sonnet-4-6",
              thinking: { type: "adaptive" },
              output_config: {
                effort: "high",
                format: {
                  type: "json_schema",
                  name: "compact_schema",
                  description: "Compaction checkpoint",
                  strict: true,
                  schema: {
                    type: "object",
                    properties: { checkpoint: { type: "string" } },
                    required: ["checkpoint"],
                  },
                },
              },
              speed: "fast",
            }),
          });
          assert.equal(known.status, 200);

          const secondFacade = await fetch(`${baseUrl}/v1/messages/compact`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-claude-code-session-id": "test-session",
            },
            body: JSON.stringify({
              ...common,
              model: "claude-fable-5",
              speed: "standard",
            }),
          });
          assert.equal(secondFacade.status, 200);

          const conflict = await fetch(`${baseUrl}/v1/messages/compact`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-claude-code-session-id": "test-session",
            },
            body: JSON.stringify({
              ...common,
              model: "claude-sonnet-4-6",
              reasoning_effort: "low",
              output_config: { effort: "high" },
            }),
          });
          assert.equal(conflict.status, 400);
          const conflictBody = await conflict.json() as {
            error: { type: string };
          };
          assert.equal(conflictBody.error.type, "invalid_request_error");

          assert.equal(requests.length, 2);
          assert.deepEqual(requests.map((request) => request.path), [
            "/responses/compact",
            "/responses/compact",
          ]);
          assert.equal(requests[0].body.model, "gpt-5.5");
          assert.equal(requests[0].body.service_tier, "priority");
          const text = requests[0].body.text as Record<string, unknown>;
          assert.deepEqual(text.format, {
            type: "json_schema",
            name: "compact_schema",
            description: "Compaction checkpoint",
            strict: true,
            schema: {
              type: "object",
              properties: { checkpoint: { type: "string" } },
              required: ["checkpoint"],
            },
          });
          assert.equal(requests[1].body.model, "gpt-5.5");
          assert.equal(Object.hasOwn(requests[1].body, "service_tier"), false);

          for (const fields of [
            { speed: "warp" },
            { speed: "fast", service_tier: "default" },
          ]) {
            const invalid = await fetch(`${baseUrl}/v1/messages/compact`, {
              method: "POST",
              headers: { "content-type": "application/json" },
              body: JSON.stringify({
                ...common,
                model: "claude-sonnet-4-6",
                ...fields,
              }),
            });
            assert.equal(invalid.status, 400);
            const error = await invalid.json() as {
              error: { type: string };
            };
            assert.equal(error.error.type, "invalid_request_error");
          }
          assert.equal(requests.length, 2);
        }, { model: "gpt-5.5", authPath: auth.authPath });
      });
    } finally {
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });


  it("accepts Anthropic shaped compact requests on /v1/messages/compact", async () => {
    let compactCalls = 0;
    const provider = {
      async compactMessages(
        messages: Array<{ content: string }>,
        opts: {
          model?: string;
          reasoningEffort?: string;
          responsesLite?: boolean;
          tools?: ToolSchema[];
          promptCacheKey?: string;
          serviceTier?: string;
          text?: Record<string, unknown>;
        },
      ) {
        compactCalls += 1;
        assert.equal(opts.model, "gpt-5.5");
        assert.equal(opts.reasoningEffort, "high");
        assert.equal(opts.responsesLite, false);
        assert.equal(opts.promptCacheKey, "anthropic-compact-cache");
        assert.equal(opts.serviceTier, "priority");
        assert.deepEqual(opts.text, {
          format: { type: "text" },
          verbosity: "medium",
        });
        assert.deepEqual(opts.tools, [{
          name: "lookup",
          parameters: { type: "object" },
          strict: false,
        }]);
        assert.deepEqual(messages.map((m) => m.content), ["sys", "hello"]);
        return "checkpoint";
      },
    };

    await withServer(provider, async (baseUrl) => {
      const res = await fetch(`${baseUrl}/v1/messages/compact`, {
        method: "POST",
        headers: {
          "content-type": "application/json",
          "x-claude-code-session-id": "compact-session",
        },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          max_tokens: 2048,
          system: "sys",
          thinking: { type: "enabled", budget_tokens: 1024 },
          responses_lite: false,
          prompt_cache_key: "anthropic-compact-cache",
          service_tier: "priority",
          text: { format: { type: "text" } },
          verbosity: "medium",
          tools: [{
            name: "lookup",
            input_schema: { type: "object" },
            strict: false,
          }],
          tool_choice: {
            type: "auto",
            disable_parallel_tool_use: true,
          },
          messages: [{ role: "user", content: "hello" }],
        }),
      });

      assert.equal(res.status, 200);
      assert.deepEqual(await res.json(), { checkpoint: "checkpoint" });

      const unsupported = await fetch(`${baseUrl}/v1/messages/compact`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          messages: [{ role: "user", content: "hello" }],
          tool_choice: {
            type: "auto",
            disable_parallel_tool_use: false,
          },
        }),
      });
      assert.equal(unsupported.status, 400);
      assert.equal(compactCalls, 1);
    });
  });

  it("rejects non-auto Anthropic compact tool choices before provider transport", async () => {
    let compactCalls = 0;
    const provider = {
      async compactMessages() {
        compactCalls += 1;
        throw new Error("provider must not be called");
      },
    };
    await withServer(provider, async (baseUrl) => {
      for (const toolChoice of [
        { type: "any" },
        { type: "tool", name: "lookup" },
        { type: "none" },
      ]) {
        const response = await fetch(`${baseUrl}/v1/messages/compact`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "compact-session",
          },
          body: JSON.stringify({
            model: "gpt-5.5",
            max_tokens: 1024,
            messages: [{ role: "user", content: "hello" }],
            tool_choice: toolChoice,
          }),
        });
        assert.equal(response.status, 400, await response.clone().text());
      }
      assert.equal(compactCalls, 0);
    });
  });
});

describe("Responses Lite route overrides", () => {
  it("forwards explicit classic mode through every provider-backed route", async () => {
    const seen: string[] = [];
    const assertClassic = (route: string, opts: { responsesLite?: boolean }) => {
      assert.equal(opts.responsesLite, false);
      seen.push(route);
    };
    const provider = {
      async generateImage(_prompt: string, opts: { responsesLite?: boolean }) {
        assertClassic("image", opts);
        return [{ result: "data:image/png;base64,AA" }];
      },
      async inspectImages(_prompt: string, opts: { responsesLite?: boolean }) {
        assertClassic("inspect", opts);
        return "inspected";
      },
      async compactMessages(_messages: unknown, opts: { responsesLite?: boolean }) {
        assertClassic("compact", opts);
        return "checkpoint";
      },
      async chat(_messages: unknown, opts: { responsesLite?: boolean }) {
        assertClassic("anthropic", opts);
        return {
          content: "done",
          tool_calls: [],
          finish_reason: "stop",
          usage: {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            cached_tokens: 0,
          },
          reasoning_content: null,
          raw: null,
        };
      },
    };

    await withServer(provider, async (baseUrl) => {
      const requests = [
        fetch(`${baseUrl}/v1/images/generations`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ prompt: "draw", responses_lite: false }),
        }),
        fetch(`${baseUrl}/v1/inspect`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            prompt: "inspect",
            images: [{ image_url: "data:image/png;base64,AAAA" }],
            responses_lite: false,
          }),
        }),
        fetch(`${baseUrl}/v1/compact`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            messages: [{ role: "user", content: "history" }],
            responses_lite: false,
          }),
        }),
        fetch(`${baseUrl}/v1/messages`, {
          method: "POST",
          headers: {
            "content-type": "application/json",
            "x-claude-code-session-id": "test-session",
          },
          body: JSON.stringify({
            model: "claude-sonnet-4-5",
            max_tokens: 64,
            system: "system",
            messages: [{ role: "user", content: "hello" }],
            responses_lite: false,
          }),
        }),
      ];
      const responses = await Promise.all(requests);
      assert.deepEqual(responses.map((response) => response.status), [200, 200, 200, 200]);
      assert.deepEqual(seen.sort(), ["anthropic", "compact", "image", "inspect"]);
    }, { model: "gpt-5.6-sol" });
  });

  it("rejects invalid route overrides before provider transport", async () => {
    const provider = new ChatGPTOAuthProvider({ model: "gpt-5.6-sol" });
    await withServer(provider, async (baseUrl) => {
      const requests = [
        ["/v1/images/generations", { prompt: "draw", responses_lite: 42 }],
        ["/v1/inspect", { prompt: "inspect", images: [], responses_lite: 42 }],
        ["/v1/compact", {
          messages: [{ role: "user", content: "history" }],
          responses_lite: 42,
        }],
        ["/v1/messages", {
          model: "claude-sonnet-4-5",
          max_tokens: 64,
          stream: true,
          system: "system",
          messages: [{ role: "user", content: "hello" }],
          responses_lite: 42,
        }],
      ] as const;
      for (const [route, body] of requests) {
        const response = await fetch(`${baseUrl}${route}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
        assert.equal(response.status, 400);
        assert.equal(response.headers.get("content-type"), "application/json; charset=utf-8");
      }
    }, { model: "gpt-5.6-sol" });
  });
});

describe("GPT-5.6 request extensions", () => {
  it("wires supported request fields through the real HTTP pipeline", async () => {
    const auth = writeAuthFile();
    try {
      await withRecordingUpstream(async (upstreamUrl, requests) => {
        const provider = new ChatGPTOAuthProvider({
          model: "gpt-5.6-sol",
          baseUrl: upstreamUrl,
          authJsonPath: auth.authPath,
        });
        await withServer(provider, async (baseUrl) => {
          const chatResponse = await fetch(`${baseUrl}/v1/chat/completions`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.6-sol",
              responses_lite: false,
              reasoning: { context: "current_turn" },
              reasoning_effort: "medium",
              service_tier: "fast",
              verbosity: "high",
              text: { format: { type: "text" } },
              prompt_cache_options: null,
              tools: [{
                type: "function",
                function: {
                  name: "lookup",
                  description: "Lookup",
                  parameters: {
                    type: "object",
                    properties: {
                      prompt_cache_breakpoint: { type: "string" },
                    },
                  },
                  strict: true,
                },
              }],
              tool_choice: {
                type: "function",
                function: { name: "lookup" },
              },
              messages: [
                { role: "system", content: "You are helpful." },
                {
                  role: "user",
                  content: [
                    {
                      type: "text",
                      text: "Inspect",
                    },
                    {
                      type: "image_url",
                      image_url: {
                        url: "data:image/png;base64,AAAA",
                        detail: "original",
                      },
                    },
                  ],
                },
              ],
            }),
          });
          assert.equal(chatResponse.status, 200, await chatResponse.clone().text());
          const chat = await chatResponse.json() as Record<string, unknown>;
          assert.equal(chat.response_id, "response-1");
          assert.deepEqual(chat.usage, {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            prompt_tokens_details: {
              cached_tokens: 1,
              cache_write_tokens: 3,
            },
          });

          const chatRequest = requests[0].body;
          assert.equal(chatRequest.model, "gpt-5.6-sol");
          assert.deepEqual(chatRequest.reasoning, {
            effort: "medium",
            context: "current_turn",
            summary: "auto",
          });
          assert.equal(chatRequest.service_tier, "priority");
          assert.deepEqual(chatRequest.tool_choice, { type: "function", name: "lookup" });
          assert.equal(Object.hasOwn(chatRequest, "safety_identifier"), false);
          assert.equal(Object.hasOwn(chatRequest, "prompt_cache_options"), false);
          assert.deepEqual(chatRequest.text, {
            format: { type: "text" },
            verbosity: "high",
          });
          assert.deepEqual(
            (chatRequest.tools as Record<string, unknown>[])[0].parameters,
            {
              type: "object",
              properties: {
                prompt_cache_breakpoint: { type: "string" },
              },
            },
          );
          assert.equal(
            (chatRequest.tools as Record<string, unknown>[])[0].strict,
            true,
          );
          for (const field of [
            "allowed_callers",
            "output_schema",
            "defer_loading",
            "eager_input_streaming",
          ]) {
            assert.equal(
              Object.hasOwn((chatRequest.tools as Record<string, unknown>[])[0], field),
              false,
            );
          }
          assert.deepEqual(chatRequest.input, [{
            type: "message",
            role: "user",
            content: [
              {
                type: "input_text",
                text: "Inspect",
              },
              {
                type: "input_image",
                image_url: "data:image/png;base64,AAAA",
                detail: "original",
              },
            ],
          }]);

          const streamResponse = await fetch(`${baseUrl}/v1/chat/completions`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.6-sol",
              responses_lite: false,
              stream: true,
              messages: [
                { role: "system", content: "You are helpful." },
                { role: "user", content: "Hello" },
              ],
            }),
          });
          assert.equal(streamResponse.status, 200);
          const chunks = (await streamResponse.text())
            .split("\n\n")
            .filter((block) => block.startsWith("data: {"))
            .map((block) => JSON.parse(block.slice(6)) as Record<string, unknown>);
          const terminal = chunks.filter((chunk) => chunk.response_id === "response-1").at(-1);
          assert.ok(terminal);
          assert.equal(terminal.response_id, "response-1");

          const compactResponse = await fetch(`${baseUrl}/v1/compact`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              responses_lite: false,
              prompt_cache_key: "compact-cache-key",
              service_tier: "fast",
              reasoning: { effort: "medium" },
              prompt_cache_options: null,
              text: { format: { type: "text" } },
              verbosity: "high",
              messages: [
                { role: "system", content: "Instructions" },
                { role: "user", content: "History" },
              ],
            }),
          });
          assert.equal(compactResponse.status, 200);
          assert.equal(requests[2].body.prompt_cache_key, "compact-cache-key");
          assert.equal(Object.hasOwn(requests[2].body, "previous_response_id"), false);
          assert.equal(Object.hasOwn(requests[2].body, "prompt_cache_options"), false);
          assert.equal(requests[2].body.service_tier, "priority");
          assert.deepEqual(requests[2].body.text, {
            format: { type: "text" },
            verbosity: "high",
          });
          assert.deepEqual(requests[2].body.reasoning, {
            effort: "medium",
            summary: "auto",
          });

          const imageResponse = await fetch(`${baseUrl}/v1/images/generations`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.6-sol",
              prompt: "Draw",
              responses_lite: false,
              reasoning_effort: "medium",
              reference_images: [{
                image_url: "data:image/png;base64,AAAA",
                detail: "high",
              }],
            }),
          });
          assert.equal(imageResponse.status, 200, await imageResponse.clone().text());
          const imageRequest = requests[3].body;
          assert.deepEqual(imageRequest.reasoning, {
            effort: "medium",
            summary: "auto",
          });
          assert.equal(Object.hasOwn(imageRequest, "safety_identifier"), false);
          const imageInput = imageRequest.input as Record<string, unknown>[];
          assert.deepEqual((imageInput[0].content as Record<string, unknown>[])[1], {
            type: "input_image",
            image_url: "data:image/png;base64,AAAA",
            detail: "high",
          });

          const inspectResponse = await fetch(`${baseUrl}/v1/inspect`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              prompt: "Inspect",
              responses_lite: false,
              reasoning: { context: "all_turns" },
              images: [{
                image_url: "data:image/png;base64,BBBB",
                detail: "original",
              }],
            }),
          });
          assert.equal(inspectResponse.status, 200);
          const inspectRequest = requests[4].body;
          assert.deepEqual(inspectRequest.reasoning, {
            effort: "low",
            summary: "auto",
            context: "all_turns",
          });
          assert.equal(Object.hasOwn(inspectRequest, "safety_identifier"), false);

          const anthropicResponse = await fetch(`${baseUrl}/v1/messages`, {
            method: "POST",
            headers: {
              "content-type": "application/json",
              "x-claude-code-session-id": "test-session",
            },
            body: JSON.stringify({
              model: "claude-sonnet-4-6",
              max_tokens: 64,
              system: "You are helpful.",
              messages: [{ role: "user", content: "Hello" }],
              responses_lite: false,
              reasoning: { effort: "high" },
            }),
          });
          assert.equal(anthropicResponse.status, 200);
          const anthropicRequest = requests[5].body;
          assert.deepEqual(anthropicRequest.reasoning, {
            effort: "high",
            summary: "auto",
          });
          assert.equal(Object.hasOwn(anthropicRequest, "safety_identifier"), false);
          assert.equal(Object.hasOwn(anthropicRequest, "prompt_cache_options"), false);

          const requestCount = requests.length;
          const unsupportedTier = await fetch(`${baseUrl}/v1/chat/completions`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.6-sol",
              responses_lite: false,
              service_tier: "flex",
              messages: [
                { role: "system", content: "You are helpful." },
                { role: "user", content: "Do not send" },
              ],
            }),
          });
          assert.equal(unsupportedTier.status, 400);
          const unsupportedTierBody = await unsupportedTier.json() as {
            error: { type: string };
          };
          assert.equal(unsupportedTierBody.error.type, "invalid_request_error");
          assert.equal(requests.length, requestCount);
        }, {
          model: "gpt-5.6-sol",
          codexConfig: TEST_CONFIG,
          authPath: auth.authPath,
        });
      });
    } finally {
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });

  it("resolves a returned response ID into local full-history replay", async () => {
    const auth = writeAuthFile();
    try {
      await withRecordingUpstream(async (upstreamUrl, requests) => {
        const provider = new ChatGPTOAuthProvider({
          model: "gpt-5.5",
          baseUrl: upstreamUrl,
          authJsonPath: auth.authPath,
        });
        await withServer(provider, async (baseUrl) => {
          const firstResponse = await fetch(`${baseUrl}/v1/chat/completions`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.5",
              responses_lite: false,
              messages: [
                { role: "system", content: "You are helpful." },
                { role: "user", content: "First turn" },
              ],
            }),
          });
          assert.equal(firstResponse.status, 200);
          const first = await firstResponse.json() as { response_id?: string };
          assert.equal(typeof first.response_id, "string");

          const secondResponse = await fetch(`${baseUrl}/v1/chat/completions`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.5",
              responses_lite: false,
              previous_response_id: first.response_id,
              messages: [
                { role: "system", content: "You are helpful." },
                { role: "user", content: "Second turn" },
              ],
            }),
          });
          assert.equal(secondResponse.status, 200);

          assert.equal(requests.length, 2);
          assert.equal(Object.hasOwn(requests[0].body, "previous_response_id"), false);
          assert.equal(Object.hasOwn(requests[1].body, "previous_response_id"), false);
          assert.deepEqual(requests[1].body.input, [
            {
              type: "message",
              role: "user",
              content: [{ type: "input_text", text: "First turn" }],
            },
            {
              type: "message",
              role: "assistant",
              content: [{ type: "output_text", text: "ok" }],
            },
            {
              type: "message",
              role: "user",
              content: [{ type: "input_text", text: "Second turn" }],
            },
          ]);

          const unknown = await fetch(`${baseUrl}/v1/chat/completions`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.5",
              responses_lite: false,
              previous_response_id: "response-does-not-exist",
              messages: [
                { role: "system", content: "You are helpful." },
                { role: "user", content: "Must not reach upstream" },
              ],
            }),
          });
          assert.equal(unknown.status, 400);
          assert.equal(requests.length, 2);
        }, { model: "gpt-5.5" });
      });
    } finally {
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });

  it("rejects compact-only unsupported fields instead of silently dropping them", async () => {
    let compactCalls = 0;
    const provider = {
      async compactMessages() {
        compactCalls += 1;
        throw new Error("provider must not be called");
      },
    };
    await withServer(provider, async (baseUrl) => {
      for (const unsupported of [
        { safety_identifier: "stable-user" },
        { include: ["reasoning.encrypted_content"] },
        { prompt_cache_retention: "24h" },
        { prompt_cache_options: { mode: "implicit", ttl: "30m" } },
        { reasoning: { mode: "standard" } },
        { reasoning: { mode: "pro" } },
        { reasoning: { context: "all_turns" } },
        { previous_response_id: "" },
        { previous_response_id: "   " },
      ]) {
        const response = await fetch(`${baseUrl}/v1/compact`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            messages: [{ role: "system", content: "Instructions" }],
            ...unsupported,
          }),
        });
        assert.equal(response.status, 400);
        const body = await response.json() as { error: { type: string } };
        assert.equal(body.error.type, "invalid_request_error");
      }
      assert.equal(compactCalls, 0);
    });
  });

  it("rejects explicit null safety_identifier on every accepting facade before provider transport", async () => {
    let providerCalls = 0;
    const provider = {
      async prepareModel() {
        providerCalls += 1;
        throw new Error("provider must not be called");
      },
      async compactMessages() {
        providerCalls += 1;
        throw new Error("provider must not be called");
      },
      async chat() {
        providerCalls += 1;
        throw new Error("provider must not be called");
      },
    };
    await withServer(provider, async (baseUrl) => {
      const requests: Array<[string, Record<string, unknown>]> = [
        ["/v1/images/generations", { prompt: "draw", safety_identifier: null }],
        [
          "/v1/inspect",
          {
            prompt: "inspect",
            images: [{ image_url: "data:image/png;base64,AAAA" }],
            safety_identifier: null,
          },
        ],
        [
          "/v1/compact",
          {
            messages: [{ role: "user", content: "hello" }],
            safety_identifier: null,
          },
        ],
        [
          "/v1/messages",
          {
            model: "claude-sonnet-4-5",
            max_tokens: 128,
            messages: [{ role: "user", content: "hello" }],
            safety_identifier: null,
          },
        ],
      ];
      for (const [route, body] of requests) {
        const response = await fetch(`${baseUrl}${route}`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
        assert.equal(response.status, 400, `${route}: ${await response.text()}`);
      }
      assert.equal(providerCalls, 0);
    });
  });

  it("rejects Anthropic-only fields on the OpenAI compact route", async () => {
    let compactCalls = 0;
    const provider = {
      async compactMessages() {
        compactCalls += 1;
        throw new Error("provider must not be called");
      },
    };
    await withServer(provider, async (baseUrl) => {
      for (const unsupported of [
        { system: "instructions" },
        { tool_choice: { type: "auto" } },
        { thinking: { type: "enabled", budget_tokens: 1024 } },
        { stop_sequences: ["stop"] },
        { max_tokens: 1024 },
        { output_format: { type: "json_schema", schema: { type: "object" } } },
        { output_config: { effort: "high" } },
        { context_management: { edits: [] } },
        { speed: "fast" },
      ]) {
        const response = await fetch(`${baseUrl}/v1/compact`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            messages: [{ role: "user", content: "history" }],
            ...unsupported,
          }),
        });
        assert.equal(response.status, 400, await response.clone().text());
        const body = await response.json() as { error: { type: string } };
        assert.equal(body.error.type, "invalid_request_error");
      }
      assert.equal(compactCalls, 0);
    });
  });

  it("does not misclassify prompt_cache_breakpoint keys in opaque request data", async () => {
    const options: Record<string, unknown>[] = [];
    const provider = {
      async chat(_messages: unknown, opts: Record<string, unknown>) {
        options.push(opts);
        return {
          content: "done",
          tool_calls: [],
          finish_reason: "stop",
          usage: {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            cached_tokens: 0,
          },
          reasoning_content: null,
          raw: null,
          response_id: "response-opaque-data",
        };
      },
    };
    const schema = {
      type: "object",
      properties: {
        prompt_cache_breakpoint: { type: "string" },
      },
    };
    const clientMetadata = { opaque: "client-owned" };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "gpt-5.6-sol",
          messages: [{ role: "user", content: "hello" }],
          text: {
            format: {
              type: "json_schema",
              name: "result",
              schema,
            },
          },
          client_metadata: clientMetadata,
        }),
      });
      assert.equal(response.status, 200, await response.clone().text());

      const invalid = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "gpt-5.6-sol",
          messages: [{ role: "user", content: "hello" }],
          client_metadata: { opaque: { prompt_cache_breakpoint: { client_owned: true } } },
        }),
      });
      assert.equal(invalid.status, 400, await invalid.clone().text());
    });

    assert.deepEqual(options[0].text, {
      format: { type: "json_schema", name: "result", schema },
    });
    assert.deepEqual(options[0].clientMetadata, clientMetadata);
    assert.equal(options.length, 1);
  });

  it("returns structured 400 errors for unsupported and conflicting request fields", async () => {
    let providerCalls = 0;
    const provider = {
      async chat() {
        providerCalls += 1;
        throw new Error("provider must not be called");
      },
      async compactMessages() {
        providerCalls += 1;
        throw new Error("provider must not be called");
      },
    };
    await withServer(provider, async (baseUrl) => {
      const requests = [
        {
          model: " gpt-5.6-sol",
          messages: [{ role: "user", content: "hello" }],
        },
        {
          model: "gpt-5.6-sol ",
          messages: [{ role: "user", content: "hello" }],
        },
        {
          model: "gpt-5.6-sol",
          reasoning_effort: " low",
          messages: [{ role: "user", content: "hello" }],
        },
        {
          model: "gpt-5.6-sol",
          reasoning: { effort: "low " },
          messages: [{ role: "user", content: "hello" }],
        },
        {
          model: "gpt-5.6-sol",
          reasoning: { mode: "pro" },
          messages: [{ role: "user", content: "hello" }],
        },
        {
          model: "gpt-5.6-sol",
          safety_identifier: "stable-user",
          messages: [{ role: "user", content: "hello" }],
        },
        {
          model: "gpt-5.6-sol",
          prompt_cache_options: { mode: "implicit", ttl: "30m" },
          messages: [{ role: "user", content: "hello" }],
        },
        {
          model: "gpt-5.6-sol",
          multi_agent: { enabled: true },
          messages: [{ role: "user", content: "hello" }],
        },
        {
          model: "gpt-5.6-sol",
          tools: [{ type: "programmatic_tool_calling" }],
          messages: [{ role: "user", content: "hello" }],
        },
        {
          model: "gpt-5.6-sol",
          tools: [{
            type: "function",
            function: {
              name: "lookup",
              parameters: { type: "object" },
              allowed_callers: ["programmatic"],
            },
          }],
          messages: [{ role: "user", content: "hello" }],
        },
        {
          model: "gpt-5.6-sol",
          tools: [{
            type: "function",
            function: {
              name: "lookup",
              parameters: { type: "object" },
              output_schema: { type: "object" },
            },
          }],
          messages: [{ role: "user", content: "hello" }],
        },
        {
          model: "gpt-5.6-sol",
          reasoning_effort: "low",
          reasoning: { effort: "high" },
          messages: [{ role: "user", content: "hello" }],
        },
        {
          model: "gpt-5.6-sol",
          verbosity: "low",
          text: { verbosity: "high" },
          messages: [{ role: "user", content: "hello" }],
        },
        {
          model: "gpt-5.6-sol",
          prompt_cache_options: { mode: "explicit" },
          messages: [{
            role: "user",
            content: [{
              type: "text",
              text: "hello",
              prompt_cache_breakpoint: { mode: "implicit" },
            }],
          }],
        },
        {
          model: "gpt-5.6-sol",
          messages: [{
            role: "user",
            content: [{ type: "input_audio", input_audio: { data: "AA" } }],
          }],
        },
        {
          model: "gpt-5.6-sol",
          prompt_cache_options: { mode: "explicit" },
          messages: [{
            role: "system",
            content: [{
              type: "text",
              text: "instructions",
              prompt_cache_breakpoint: { mode: "explicit" },
            }],
          }],
        },
        {
          model: "gpt-5.6-sol",
          messages: [{
            role: "assistant",
            content: [{
              type: "text",
              text: "prior answer",
              prompt_cache_breakpoint: { mode: "explicit" },
            }],
          }],
        },
      ];
      for (const body of requests) {
        const response = await fetch(`${baseUrl}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
        assert.equal(response.status, 400);
        const error = await response.json() as { error: { type: string } };
        assert.equal(error.error.type, "invalid_request_error");
      }

      for (const body of [
        {
          reasoning_effort: " low",
          messages: [{ role: "user", content: "history" }],
        },
        {
          reasoning: { mode: "pro" },
          messages: [{ role: "user", content: "history" }],
        },
        {
          prompt_cache_key: "",
          messages: [{ role: "user", content: "history" }],
        },
        {
          verbosity: "low",
          text: { verbosity: "high" },
          messages: [{ role: "user", content: "history" }],
        },
      ]) {
        const compactResponse = await fetch(`${baseUrl}/v1/compact`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
        assert.equal(compactResponse.status, 400);
      }
      assert.equal(providerCalls, 0);
    }, { model: "gpt-5.6-sol" });
  });
});

describe("GPT-5.6 HTTP pipeline", () => {
  it("sends finalized chat and compact Lite requests through the real provider", async () => {
    const auth = writeAuthFile();
    const previousLite = process.env.CODEX_AS_API_RESPONSES_LITE;
    process.env.CODEX_AS_API_RESPONSES_LITE = "auto";
    const codexConfig: CodexConfig = {
      codexHome: auth.directory,
      configPath: path.join(auth.directory, "config.toml"),
      model: "gpt-5.6-sol",
      modelReasoningEffort: "ultra",
    };

    try {
      await withRecordingUpstream(async (upstreamUrl, requests) => {
        const provider = new ChatGPTOAuthProvider({
          model: "gpt-5.6-sol",
          baseUrl: upstreamUrl,
          authJsonPath: auth.authPath,
        });

        await withServer(provider, async (baseUrl) => {
          const healthResponse = await fetch(`${baseUrl}/health`);
          assert.equal(healthResponse.status, 200);
          const health = await healthResponse.json() as Record<string, unknown>;
          assert.equal(health.model, "gpt-5.6-sol");
          assert.equal(health.catalog_status, "fresh");
          assert.equal(health.auth_available, true);
          assert.equal(health.reasoning_effort, "max");

          const chatResponse = await fetch(`${baseUrl}/v1/chat/completions`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.6-sol",
              messages: [
                { role: "system", content: "You are helpful." },
                { role: "user", content: "Hello" },
              ],
            }),
          });
          assert.equal(chatResponse.status, 200);
          const chatResult = await chatResponse.json() as {
            choices: { message: { content: string } }[];
            response_id?: string;
            usage?: Record<string, unknown>;
          };
          assert.equal(chatResult.choices[0].message.content, "ok");
          assert.equal(chatResult.response_id, "response-1");
          assert.deepEqual(chatResult.usage, {
            prompt_tokens: 1,
            completion_tokens: 1,
            total_tokens: 2,
            prompt_tokens_details: {
              cached_tokens: 1,
              cache_write_tokens: 3,
            },
          });

          assert.equal(requests.length, 1);
          const chatRequest = requests[0];
          assert.equal(chatRequest.path, "/responses");
          assert.equal(chatRequest.headers["x-openai-internal-codex-responses-lite"], "true");
          assert.equal(chatRequest.body.model, "gpt-5.6-sol");
          assert.deepEqual(chatRequest.body.reasoning, {
            effort: "max",
            summary: "auto",
            context: "all_turns",
          });
          assert.deepEqual(chatRequest.body.include, ["reasoning.encrypted_content"]);
          assert.deepEqual(chatRequest.body.text, { verbosity: "low" });
          assert.equal(chatRequest.body.tool_choice, "auto");
          assert.equal(chatRequest.body.parallel_tool_calls, false);
          assert.equal(Object.hasOwn(chatRequest.body, "instructions"), false);
          assert.equal(Object.hasOwn(chatRequest.body, "tools"), false);
          assert.deepEqual(chatRequest.body.input, [
            { type: "additional_tools", role: "developer", tools: [] },
            {
              type: "message",
              role: "developer",
              content: [{ type: "input_text", text: "You are helpful." }],
            },
            {
              type: "message",
              role: "user",
              content: [{ type: "input_text", text: "Hello" }],
            },
          ]);

          const compactResponse = await fetch(`${baseUrl}/v1/compact`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              messages: [
                { role: "system", content: "Old prompt must not survive compaction." },
                { role: "user", content: "History" },
              ],
              tools: [{
                type: "function",
                function: {
                  name: "lookup",
                  parameters: { type: "object" },
                  strict: false,
                },
              }],
            }),
          });
          assert.equal(compactResponse.status, 200);
          const compactResult = await compactResponse.json() as { checkpoint: string };
          assert.deepEqual(
            messagesToResponseItems([{
              role: MessageRole.SYSTEM,
              content: compactResult.checkpoint,
            }]),
            [
              {
                type: "message",
                role: "assistant",
                content: [{ type: "output_text", text: "summary" }],
              },
              {
                type: "agent_message",
                author: "agent",
                recipient: "user",
                content: [{ type: "input_text", text: "agent summary" }],
              },
              {
                type: "message",
                role: "user",
                content: [{ type: "input_text", text: "compacted" }],
              },
              { type: "compaction_summary", encrypted_content: "legacy" },
              { type: "context_compaction" },
            ],
          );

          assert.equal(requests.length, 2);
          const compactRequest = requests[1];
          assert.equal(compactRequest.path, "/responses/compact");
          assert.equal(compactRequest.headers["x-openai-internal-codex-responses-lite"], "true");
          assert.equal(compactRequest.body.model, "gpt-5.6-sol");
          assert.deepEqual(compactRequest.body.reasoning, {
            effort: "max",
            summary: "auto",
            context: "all_turns",
          });
          assert.deepEqual(compactRequest.body.text, { verbosity: "low" });
          assert.equal(compactRequest.body.parallel_tool_calls, false);
          assert.equal(Object.hasOwn(compactRequest.body, "include"), false);
          assert.equal(Object.hasOwn(compactRequest.body, "instructions"), false);
          assert.equal(Object.hasOwn(compactRequest.body, "tools"), false);
          assert.equal(Object.hasOwn(compactRequest.body, "tool_choice"), false);
          assert.deepEqual(compactRequest.body.input, [
            {
              type: "additional_tools",
              role: "developer",
              tools: [{
                type: "function",
                name: "lookup",
                parameters: { type: "object" },
                strict: false,
              }],
            },
            {
              type: "message",
              role: "developer",
              content: [{
                type: "input_text",
                text: "Old prompt must not survive compaction.",
              }],
            },
            {
              type: "message",
              role: "user",
              content: [{ type: "input_text", text: "History" }],
            },
          ]);

          const continuationResponse = await fetch(`${baseUrl}/v1/chat/completions`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.6-sol",
              messages: [
                { role: "system", content: "Current prompt." },
                { role: "system", content: compactResult.checkpoint },
                { role: "user", content: "Continue" },
              ],
            }),
          });
          assert.equal(continuationResponse.status, 200);
          assert.equal(requests.length, 3);
          const continuationRequest = requests[2];
          assert.equal(continuationRequest.path, "/responses");
          assert.deepEqual(continuationRequest.body.input, [
            { type: "additional_tools", role: "developer", tools: [] },
            {
              type: "message",
              role: "developer",
              content: [{ type: "input_text", text: "Current prompt." }],
            },
            {
              type: "message",
              role: "assistant",
              content: [{ type: "output_text", text: "summary" }],
            },
            {
              type: "agent_message",
              author: "agent",
              recipient: "user",
              content: [{ type: "input_text", text: "agent summary" }],
            },
            {
              type: "message",
              role: "user",
              content: [{ type: "input_text", text: "compacted" }],
            },
            { type: "compaction_summary", encrypted_content: "legacy" },
            { type: "context_compaction" },
            {
              type: "message",
              role: "user",
              content: [{ type: "input_text", text: "Continue" }],
            },
          ]);
          assert.equal(
            JSON.stringify(continuationRequest.body).includes("Old prompt must not survive"),
            false,
          );

          const inspectResponse = await fetch(`${baseUrl}/v1/inspect`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.6-terra",
              prompt: "Inspect this image.",
              images: [{ image_url: "data:image/png;base64,AAAA" }],
            }),
          });
          assert.equal(inspectResponse.status, 200);
          assert.deepEqual(await inspectResponse.json(), { content: "ok" });

          assert.equal(requests.length, 4);
          const inspectRequest = requests[3];
          assert.equal(inspectRequest.path, "/responses");
          assert.equal(inspectRequest.headers["x-openai-internal-codex-responses-lite"], "true");
          assert.equal(inspectRequest.body.model, "gpt-5.6-terra");
          assert.equal(inspectRequest.body.tool_choice, "auto");
          assert.deepEqual(inspectRequest.body.reasoning, {
            effort: "max",
            summary: "auto",
            context: "all_turns",
          });
        }, {
          model: "gpt-5.6-sol",
          codexConfig,
          authPath: auth.authPath,
        });
      });
    } finally {
      if (previousLite == null) delete process.env.CODEX_AS_API_RESPONSES_LITE;
      else process.env.CODEX_AS_API_RESPONSES_LITE = previousLite;
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });
});
