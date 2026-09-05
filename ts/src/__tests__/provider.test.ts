import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { MessageRole } from "../messages.js";
import type { Message, ToolCall, ToolSchema } from "../messages.js";
import {
  decodeSSEBlock,
  splitInstructionsAndInput,
  messagesToResponseItems,
  messageItem,
  toolSchemaToResponseDict,
  setReasoningPayload,
  toolCallFromResponseItem,
  webSearchEventFromResponseItem,
  textFromResponseItems,
  validateImageContentItems,
  imageGenerationFromItem,
  usageFromResponse,
  REMOTE_COMPACTION_MARKER,
  ChatGPTOAuthProvider,
  codexCliHeadersForVersion,
  resolveCodexCliVersion,
} from "../provider.js";
import type { PreparedModel, StreamEvent } from "../provider.js";
import type { ModelCapability, ModelCatalogSnapshot } from "../model-capabilities.js";
import { modelCatalogCacheKey } from "../model-capabilities.js";
import {
  ChatGPTOAuthError,
  ChatGPTOAuthCatalogUnavailableError,
  ChatGPTOAuthInvalidRequestError,
  ChatGPTOAuthModelNotFoundError,
  ChatGPTOAuthProtocolError,
  ChatGPTOAuthUnavailableError,
  ChatGPTOAuthUpstreamError,
} from "../auth.js";

function providerMessages(): Message[] {
  return [
    { role: MessageRole.SYSTEM, content: "You are helpful." },
    { role: MessageRole.USER, content: "Hello" },
  ];
}

function providerMessagesWithImageDetail(
  detail: "auto" | "low" | "high" | "original",
): Message[] {
  return [
    { role: MessageRole.SYSTEM, content: "You are helpful." },
    {
      role: MessageRole.USER,
      content: "Inspect this image.",
      structured_content: [
        { type: "text", text: "Inspect this image." },
        {
          type: "image_url",
          image_url: "data:image/png;base64,AAAA",
          detail,
        },
      ],
    },
  ];
}

function makeJwt(payload: Record<string, unknown>): string {
  const header = Buffer.from(JSON.stringify({ alg: "HS256", typ: "JWT" }))
    .toString("base64url");
  const body = Buffer.from(JSON.stringify(payload)).toString("base64url");
  return `${header}.${body}.sig`;
}

function writeAuthFile(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-provider-"));
  const filePath = path.join(dir, "auth.json");
  fs.writeFileSync(
    filePath,
    JSON.stringify({
      tokens: {
        access_token: makeJwt({ exp: 9999999999 }),
        refresh_token: "refresh-token",
        id_token: makeJwt({
          exp: 9999999999,
          "https://api.openai.com/auth": {
            chatgpt_account_id: "acc-123",
            chatgpt_plan_type: "plus",
            chatgpt_user_id: "user-abc",
          },
        }),
      },
    }),
  );
  return filePath;
}

function unreadUnauthorizedResponse(cancelFailure?: string): {
  response: Response;
  state: { cancelCalls: number; textCalls: number };
} {
  const state = { cancelCalls: 0, textCalls: 0 };
  const response = new Response("unread unauthorized body", { status: 401 });
  Object.defineProperty(response, "text", {
    value: async () => {
      state.textCalls++;
      throw new Error("the first 401 body must not be read");
    },
  });
  assert.ok(response.body);
  Object.defineProperty(response.body, "cancel", {
    value: () => {
      state.cancelCalls++;
      return cancelFailure == null
        ? Promise.resolve()
        : Promise.reject(new Error(cancelFailure));
    },
  });
  return { response, state };
}

function testCapability(slug = "gpt-5.5"): ModelCapability {
  const hasPriorityTier = ["gpt-5.5", "gpt-5.6-sol"].includes(slug);
  return Object.freeze({
    slug,
    displayName: slug,
    description: null,
    defaultReasoningEffort: slug === "gpt-5.6-sol" ? "low" : "medium",
    supportedReasoningEfforts: Object.freeze([
      "none", "minimal", "low", "medium", "high", "xhigh", "max",
    ].map((effort) => Object.freeze({ effort, description: effort }))),
    priority: 1,
    visibility: "list",
    supportedInApi: true,
    useResponsesLite: slug.startsWith("gpt-5.6"),
    supportsImageDetailOriginal: !slug.includes("spark") && slug !== "gpt-5.2",
    supportVerbosity: !slug.includes("spark"),
    defaultVerbosity: slug.includes("spark") ? null : "low",
    supportsReasoningSummaryParameter: true,
    defaultReasoningSummary: "auto",
    contextWindow: 272_000,
    maxContextWindow: 272_000,
    effectiveContextWindowPercent: 95,
    serviceTiers: hasPriorityTier
      ? Object.freeze([{ id: "priority", name: "Priority", description: "Priority" }])
      : Object.freeze([]),
    defaultServiceTier: null,
    inputModalities: Object.freeze(["text", "image"] as const),
  });
}

function preparedForTest(slug = "gpt-5.5"): PreparedModel {
  const capability = testCapability(slug);
  const snapshot: ModelCatalogSnapshot = Object.freeze({
    key: modelCatalogCacheKey("acc-123", "https://example.test", "0.153.3"),
    etag: '"test"',
    fetchedAt: 1,
    expiresAt: 300_001,
    models: Object.freeze([capability]),
    defaultModel: capability,
  });
  return Object.freeze({ slug, accountId: "acc-123", capability, snapshot });
}

function testUsage(): Record<string, unknown> {
  return {
    input_tokens: 1,
    output_tokens: 1,
    total_tokens: 2,
    input_tokens_details: { cached_tokens: 0 },
  };
}

function testProvider(
  opts: ConstructorParameters<typeof ChatGPTOAuthProvider>[0] = {},
): ChatGPTOAuthProvider {
  const provider = new ChatGPTOAuthProvider(opts);
  const originalPayload = (provider as unknown as {
    responsesPayload(messages: Message[], options: Record<string, unknown>, prepared: PreparedModel): Record<string, unknown>;
  }).responsesPayload.bind(provider);
  (provider as unknown as { responsesPayload: unknown }).responsesPayload = (
    messages: Message[],
    options: Record<string, unknown>,
    prepared?: PreparedModel,
  ) => originalPayload(
    messages,
    options,
    prepared ?? preparedForTest(typeof options.model === "string" ? options.model : opts.model),
  );
  provider.prepareModel = async (requested?: string) => preparedForTest(requested ?? opts.model);
  provider.prepareAnthropicModel = async (requested?: string) => preparedForTest(requested ?? opts.model);
  return provider;
}

describe("ChatGPTOAuthProvider model resolution", () => {
  it("accepts any non-blank exact live catalog model id", async () => {
    const slug = "access_token=VALID_MODEL_SENTINEL";
    const capability = testCapability(slug);
    const snapshot: ModelCatalogSnapshot = Object.freeze({
      key: modelCatalogCacheKey("acc-123", "https://example.test", "0.153.3"),
      etag: '"test"',
      fetchedAt: 1,
      expiresAt: 300_001,
      models: Object.freeze([capability]),
      defaultModel: capability,
    });
    const provider = new ChatGPTOAuthProvider({ model: slug });

    assert.equal((await provider.prepareModel(undefined, snapshot)).slug, slug);
    assert.equal((await provider.prepareModel(slug, snapshot)).slug, slug);
    assert.equal((await provider.prepareModel(null as unknown as undefined, snapshot)).slug, slug);
    const missing = "access_token=MISSING_MODEL_SENTINEL";
    await assert.rejects(provider.prepareModel(missing, snapshot), (error) => (
      error instanceof ChatGPTOAuthModelNotFoundError
      && !error.message.includes(missing)
      && error.message === "requested model is not available in the authenticated upstream catalog"
    ));
  });

  it("rejects non-string, blank, or whitespace-padded configured model values", async () => {
    for (const model of ["", " \t ", " gpt-5.5", "gpt-5.5 ", 42, {}]) {
      assert.throws(
        () => new ChatGPTOAuthProvider({ model } as never),
        ChatGPTOAuthInvalidRequestError,
      );
    }
    const provider = new ChatGPTOAuthProvider();
    const snapshot = preparedForTest().snapshot;
    for (const model of [" gpt-5.5", "gpt-5.5 "]) {
      await assert.rejects(
        provider.prepareModel(model, snapshot),
        ChatGPTOAuthInvalidRequestError,
      );
    }
  });

  it("fails implicit hidden-only and removed configured models as catalog availability errors", async () => {
    const hidden = Object.freeze({ ...testCapability("hidden-model"), visibility: "hide" });
    const snapshot: ModelCatalogSnapshot = Object.freeze({
      key: modelCatalogCacheKey("acc-123", "https://example.test", "0.153.3"),
      etag: null,
      fetchedAt: 1,
      expiresAt: 300_001,
      models: Object.freeze([hidden]),
      defaultModel: null,
    });
    const unconfigured = new ChatGPTOAuthProvider();
    assert.equal((await unconfigured.prepareModel("hidden-model", snapshot)).slug, "hidden-model");
    await assert.rejects(
      unconfigured.prepareModel(undefined, snapshot),
      ChatGPTOAuthCatalogUnavailableError,
    );

    const configured = new ChatGPTOAuthProvider({ model: "removed-model" });
    await assert.rejects(
      configured.prepareModel(undefined, snapshot),
      ChatGPTOAuthCatalogUnavailableError,
    );
    await assert.rejects(
      unconfigured.prepareModel("removed-model", snapshot),
      ChatGPTOAuthModelNotFoundError,
    );
    await assert.rejects(
      configured.prepareAnthropicModel("claude-sonnet-4-6", snapshot),
      ChatGPTOAuthCatalogUnavailableError,
    );
  });

  it("preserves opaque live slugs but rejects an unusable implicit default", async () => {
    const capability = Object.freeze({ ...testCapability(" "), visibility: "list" as const });
    const snapshot: ModelCatalogSnapshot = Object.freeze({
      key: modelCatalogCacheKey("acc-123", "https://example.test", "0.153.3"),
      etag: null,
      fetchedAt: 1,
      expiresAt: 300_001,
      models: Object.freeze([capability]),
      defaultModel: capability,
    });

    await assert.rejects(
      new ChatGPTOAuthProvider().prepareModel(undefined, snapshot),
      ChatGPTOAuthCatalogUnavailableError,
    );
    await assert.rejects(
      new ChatGPTOAuthProvider().prepareModel(" ", snapshot),
      ChatGPTOAuthInvalidRequestError,
    );
  });
});

describe("ChatGPTOAuthProvider payload", () => {
  it("does not follow authenticated Responses redirects", async () => {
    const authPath = writeAuthFile();
    const originalFetch = globalThis.fetch;
    const requests: RequestInit[] = [];
    try {
      globalThis.fetch = async (_input, init) => {
        requests.push(init ?? {});
        return new Response("redirect refused", {
          status: 308,
          headers: { Location: "https://attacker.example/steal" },
        });
      };
      const provider = testProvider({ authJsonPath: authPath });
      for (const operation of [
        () => provider.chat(providerMessages(), { model: "gpt-5.5" }),
        () => provider.compactMessages(providerMessages(), {
          model: "gpt-5.5",
          responsesLite: false,
        }),
      ]) {
        await assert.rejects(
          operation,
          (error: unknown) => error instanceof ChatGPTOAuthUpstreamError
            && error.status === 308,
        );
      }
      assert.equal(requests.length, 2);
      assert.equal(requests.every((request) => request.redirect === "manual"), true);
    } finally {
      globalThis.fetch = originalFetch;
      fs.rmSync(path.dirname(authPath), { recursive: true, force: true });
    }
  });

  it("refreshes a JSON request 401 without reading its response body", async () => {
    const authPath = writeAuthFile();
    const originalFetch = globalThis.fetch;
    const unauthorized = unreadUnauthorizedResponse();
    const refreshedAccess = makeJwt({ exp: 9_999_999_999 });
    const refreshedId = makeJwt({
      exp: 9_999_999_999,
      "https://api.openai.com/auth": { chatgpt_account_id: "acc-123" },
    });
    let responseCalls = 0;
    try {
      globalThis.fetch = async (input) => {
        const url = String(input);
        if (url === "https://auth.openai.com/oauth/token") {
          return new Response(JSON.stringify({
            access_token: refreshedAccess,
            refresh_token: "refresh-new",
            id_token: refreshedId,
          }), { status: 200, headers: { "content-type": "application/json" } });
        }
        responseCalls++;
        return responseCalls === 1
          ? unauthorized.response
          : new Response('{"output":[]}', {
            status: 200,
            headers: { "content-type": "application/json" },
          });
      };

      const provider = testProvider({ authJsonPath: authPath, baseUrl: "https://catalog.test" });
      const checkpoint = await provider.compactMessages(providerMessages(), {
        model: "gpt-5.5",
        responsesLite: false,
      });
      assert.equal(typeof checkpoint, "string");
      assert.equal(responseCalls, 2);
      assert.deepEqual(unauthorized.state, { cancelCalls: 1, textCalls: 0 });
    } finally {
      globalThis.fetch = originalFetch;
      fs.rmSync(path.dirname(authPath), { recursive: true, force: true });
    }
  });

  it("refreshes an SSE request 401 without reading its response body", async () => {
    const authPath = writeAuthFile();
    const originalFetch = globalThis.fetch;
    const unauthorized = unreadUnauthorizedResponse();
    const refreshedAccess = makeJwt({ exp: 9_999_999_999 });
    const refreshedId = makeJwt({
      exp: 9_999_999_999,
      "https://api.openai.com/auth": { chatgpt_account_id: "acc-123" },
    });
    let responseCalls = 0;
    try {
      globalThis.fetch = async (input) => {
        const url = String(input);
        if (url === "https://auth.openai.com/oauth/token") {
          return new Response(JSON.stringify({
            access_token: refreshedAccess,
            refresh_token: "refresh-new",
            id_token: refreshedId,
          }), { status: 200, headers: { "content-type": "application/json" } });
        }
        responseCalls++;
        if (responseCalls === 1) return unauthorized.response;
        return new Response([
          'data: {"type":"response.output_item.done","item":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"ok"}]}}\n\n',
          'data: {"type":"response.completed","response":{"id":"response-1","end_turn":true}}\n\n',
        ].join(""), {
          status: 200,
          headers: { "content-type": "text/event-stream" },
        });
      };

      const provider = testProvider({ authJsonPath: authPath, baseUrl: "https://catalog.test" });
      const response = await provider.chat(providerMessages(), { model: "gpt-5.5" });
      assert.equal(response.content, "ok");
      assert.equal(responseCalls, 2);
      assert.deepEqual(unauthorized.state, { cancelCalls: 1, textCalls: 0 });
    } finally {
      globalThis.fetch = originalFetch;
      fs.rmSync(path.dirname(authPath), { recursive: true, force: true });
    }
  });

  it("surfaces and redacts 401 response cancellation failures before refresh", async () => {
    for (const transport of ["json", "sse"] as const) {
      const authPath = writeAuthFile();
      const originalFetch = globalThis.fetch;
      const secret = "refresh-token";
      const unauthorized = unreadUnauthorizedResponse(`cancel exposed ${secret}`);
      let responseCalls = 0;
      let refreshCalls = 0;
      try {
        globalThis.fetch = async (input) => {
          if (String(input) === "https://auth.openai.com/oauth/token") {
            refreshCalls++;
            throw new Error("refresh must not start after cancellation failure");
          }
          responseCalls++;
          return unauthorized.response;
        };

        const provider = testProvider({ authJsonPath: authPath, baseUrl: "https://catalog.test" });
        const operation = transport === "json"
          ? provider.compactMessages(providerMessages(), {
            model: "gpt-5.5",
            responsesLite: false,
          })
          : provider.chat(providerMessages(), { model: "gpt-5.5" });
        await assert.rejects(operation, (error: unknown) => (
          error instanceof ChatGPTOAuthUnavailableError
          && error.message.includes("response cancellation failed")
          && !error.message.includes(secret)
        ));
        assert.equal(responseCalls, 1);
        assert.equal(refreshCalls, 0);
        assert.deepEqual(unauthorized.state, { cancelCalls: 1, textCalls: 0 });
      } finally {
        globalThis.fetch = originalFetch;
        fs.rmSync(path.dirname(authPath), { recursive: true, force: true });
      }
    }
  });

  it("rejects unsafe subagent header values before transport without leaking them", async () => {
    const provider = testProvider({});
    let transportCalls = 0;
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      transportCalls++;
      yield { type: "response.completed", response: { id: "unexpected" } };
    };

    for (const subagent of [
      "",
      "LEAK ME",
      "LEAK\tME",
      "LEAK\r\nME",
      "LEAK\u0000ME",
      "LEAK\u007fME",
      "LEAK-ME-한글",
    ]) {
      await assert.rejects(
        () => provider.createChatStream(providerMessages(), { subagent }),
        (error) => {
          assert.ok(error instanceof ChatGPTOAuthInvalidRequestError);
          if (subagent.length > 0) {
            assert.equal(error.message.includes(subagent), false);
          }
          return true;
        },
      );
    }
    assert.equal(transportCalls, 0);
  });

  it("forwards a safe subagent token unchanged", async () => {
    const provider = testProvider({});
    let observedHeaders: Record<string, string> | undefined;
    (provider as unknown as {
      postSSE(
        path: string,
        payload: Record<string, unknown>,
        headers: Record<string, string>,
      ): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* (_path, _payload, headers) {
      observedHeaders = { ...headers };
      yield {
        type: "response.completed",
        response: { id: "response-safe-subagent", output: [], usage: testUsage() },
      };
    };

    for await (const _event of provider.chatStream(providerMessages(), {
      subagent: "reviewer/v1:fast",
    })) {
      // Consume the stream so the transport boundary is exercised.
    }
    assert.equal(observedHeaders?.["x-openai-subagent"], "reviewer/v1:fast");
  });

  it("gates only the input modalities present in the request", () => {
    const provider = testProvider({});
    const prepared = preparedForTest("gpt-5.5");
    const imageOnly: PreparedModel = {
      ...prepared,
      capability: {
        ...prepared.capability,
        inputModalities: Object.freeze(["image"]),
      },
    };
    const responsesPayload = (provider as unknown as {
      responsesPayload(
        messages: Message[],
        opts: { model: string },
        prepared: PreparedModel,
      ): Record<string, unknown>;
    }).responsesPayload.bind(provider);

    const imagePayload = responsesPayload([{
      role: MessageRole.USER,
      content: "",
      structured_content: [{
        type: "image_url",
        image_url: "data:image/png;base64,AAAA",
      }],
    }], { model: "image-only" }, imageOnly);
    assert.equal(
      ((imagePayload.input as Record<string, unknown>[])[0].content as Record<string, unknown>[])[0].type,
      "input_image",
    );
    assert.throws(
      () => responsesPayload(providerMessages(), { model: "image-only" }, imageOnly),
      ChatGPTOAuthInvalidRequestError,
    );

    const audioOnly: PreparedModel = {
      ...prepared,
      capability: {
        ...prepared.capability,
        inputModalities: Object.freeze(["audio"]),
      },
    };
    const audioPayload = responsesPayload([{
      role: MessageRole.USER,
      content: "",
      structured_content: [{
        type: "input_audio",
        audio_url: "data:audio/wav;base64,AAAA",
      }],
    }], { model: "audio-only" }, audioOnly);
    assert.equal(
      ((audioPayload.input as Record<string, unknown>[])[0].content as Record<string, unknown>[])[0].type,
      "input_audio",
    );
  });

  it("requires live image capability for hosted image generation", async () => {
    const provider = testProvider({});
    const prepared = preparedForTest("gpt-5.5");
    const textOnly: PreparedModel = {
      ...prepared,
      capability: {
        ...prepared.capability,
        inputModalities: Object.freeze(["text"]),
      },
    };

    await assert.rejects(
      () => provider.generateImage("Draw", { preparedModel: textOnly }),
      ChatGPTOAuthInvalidRequestError,
    );
  });

  it("rejects maxTokens because the private transport has no wire field", () => {
    const provider = testProvider({});
    const responsesPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; maxTokens: number }): Record<string, unknown>;
    }).responsesPayload.bind(provider);
    assert.throws(
      () => responsesPayload(providerMessages(), { model: "gpt-5.5", maxTokens: 1024 }),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError,
    );
  });

  it("omits absent stop values from the private request", async () => {
    for (const stop of [undefined, null] as const) {
      const provider = testProvider({});
      const payloads: Record<string, unknown>[] = [];
      (provider as unknown as {
        postSSE(path: string, payload: Record<string, unknown>): AsyncGenerator<Record<string, unknown>>;
      }).postSSE = async function* (_path, payload) {
        payloads.push(structuredClone(payload));
        yield {
          type: "response.output_item.done",
          item: {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "" }],
          },
        };
        yield {
          type: "response.completed",
          response: { id: "resp-empty-stop", output: [], usage: testUsage() },
        };
      };

      for await (const _event of provider.chatStream(providerMessages(), {
        model: "gpt-5.5",
        stop: stop as unknown as string | string[] | undefined,
      })) {
        // Consume the real provider stream so transport invocation is observable.
      }

      assert.equal(payloads.length, 1);
      assert.equal(Object.hasOwn(payloads[0], "stop"), false);
    }
  });

  it("rejects any explicit stop before the private transport starts", async () => {
    const provider = testProvider({});
    let transportCalls = 0;
    (provider as unknown as {
      postSSE(path: string, payload: Record<string, unknown>): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      transportCalls += 1;
      yield { type: "response.completed", response: { id: "unexpected" } };
    };

    await assert.rejects(
      () => provider.createChatStream(providerMessages(), {
        model: "gpt-5.5",
        stop: ["END"],
      }),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError,
    );
    assert.equal(transportCalls, 0);
  });

  it("rejects original image detail for models without verified support", () => {
    const provider = testProvider({});
    const responsesPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: Record<string, unknown>): Record<string, unknown>;
    }).responsesPayload.bind(provider);

    for (const model of ["gpt-5.2", "gpt-5.3-codex-spark"]) {
      assert.throws(
        () => responsesPayload(providerMessagesWithImageDetail("original"), {
          model,
          responsesLite: false,
        }),
        (error) => error instanceof ChatGPTOAuthInvalidRequestError,
      );
    }
  });

  it("does not mistake opaque tool schemas for image input and rejects non-string metadata", () => {
    const provider = testProvider({});
    const responsesPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: Record<string, unknown>): Record<string, unknown>;
    }).responsesPayload.bind(provider);
    const opaqueImageShape = { type: "input_image", detail: "original" };
    const tool: ToolSchema = {
      name: "inspect_shape",
      description: "Inspect a client shape",
      parameters: {
        type: "object",
        properties: {
          shape: { type: "object", default: opaqueImageShape },
        },
      },
    };

    for (const responsesLite of [false, true]) {
      assert.throws(
        () => responsesPayload(providerMessages(), {
          model: responsesLite ? "gpt-5.6-sol" : "gpt-5.2",
          tools: [tool],
          responsesLite,
          clientMetadata: { opaque: opaqueImageShape },
        }),
        ChatGPTOAuthInvalidRequestError,
      );
    }

    const classic = responsesPayload(providerMessages(), {
      model: "gpt-5.2",
      tools: [tool],
      responsesLite: false,
      clientMetadata: { opaque: "client-owned" },
    });
    assert.deepEqual((classic.tools as Record<string, unknown>[])[0].parameters, tool.parameters);
    assert.equal((classic.client_metadata as Record<string, unknown>).opaque, "client-owned");
  });

  it("keeps auto, low, and high image detail for GPT-5.2", () => {
    const provider = testProvider({});
    const responsesPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: Record<string, unknown>): Record<string, unknown>;
    }).responsesPayload.bind(provider);

    for (const detail of ["auto", "low", "high"] as const) {
      const payload = responsesPayload(providerMessagesWithImageDetail(detail), {
        model: "gpt-5.2",
        responsesLite: false,
      });
      const input = payload.input as Record<string, unknown>[];
      const content = input[0].content as Record<string, unknown>[];
      assert.equal(content[1].detail, detail);
    }
  });

  it("rejects unsupported original detail on inspection and generation references before transport", async () => {
    const provider = testProvider({});
    let transportStarted = false;
    (provider as unknown as {
      postSSE(path: string, payload: Record<string, unknown>): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      transportStarted = true;
      yield { type: "response.completed", response: { id: "unexpected" } };
    };
    const images = [{
      image_url: "data:image/png;base64,AAAA",
      detail: "original" as const,
    }];

    await assert.rejects(
      provider.inspectImages("Inspect this", { model: "gpt-5.2", images }),
      (error) => error instanceof ChatGPTOAuthError,
    );
    await assert.rejects(
      provider.generateImage("Draw this", { model: "gpt-5.2", referenceImages: images }),
      (error) => error instanceof ChatGPTOAuthError,
    );
    assert.equal(transportStarted, false);
  });

  it("rejects an explicit empty image size instead of treating it as omitted", async () => {
    const provider = testProvider({});
    let transportStarted = false;
    (provider as unknown as {
      postSSE(path: string, payload: Record<string, unknown>): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      transportStarted = true;
      yield { type: "response.completed", response: { id: "unexpected" } };
    };

    await assert.rejects(
      provider.generateImage("Draw this", { model: "gpt-5.5", size: "" }),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError,
    );
    assert.equal(transportStarted, false);
  });

  it("rejects an empty chat prompt cache key", () => {
    const provider = testProvider({});

    assert.throws(
      () => (provider as unknown as {
        responsesPayload(
          messages: Message[],
          opts: { model: string; promptCacheKey: string },
        ): Record<string, unknown>;
      }).responsesPayload(providerMessages(), {
        model: "gpt-5.6-sol",
        promptCacheKey: "",
      }),
      /prompt_cache_key must be a non-empty string/,
    );
  });

  it("includes web_search hosted tool sources in Responses payload", () => {
    const provider = testProvider({});
    const webSearchTool: ToolSchema = {
      name: "web_search",
      description: "Web search",
      parameters: {
        __codex_as_api_tool_type: "web_search",
        openai_tool: { type: "web_search", external_web_access: true },
      },
    };
    const payload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; tools: ToolSchema[]; toolChoice: Record<string, unknown>; responsesLite: boolean }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      tools: [webSearchTool],
      toolChoice: { type: "web_search" },
      responsesLite: false,
    });

    assert.deepEqual(payload.tools, [{ type: "web_search", external_web_access: true }]);
    assert.deepEqual(payload.tool_choice, { type: "web_search" });
    assert.deepEqual(payload.include, [
      "web_search_call.action.sources",
      "reasoning.encrypted_content",
    ]);
  });

  it("adds encrypted reasoning include when reasoning effort is present", () => {
    const provider = testProvider({});
    const payload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; reasoningEffort: string; responsesLite: boolean }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      reasoningEffort: "high",
      responsesLite: false,
    });

    assert.deepEqual(payload.reasoning, { effort: "high", summary: "auto" });
    assert.deepEqual(payload.include, ["reasoning.encrypted_content"]);
  });

  it("forces Responses Lite payload shape", () => {
    const provider = testProvider({});
    const tool: ToolSchema = { name: "lookup", description: "Lookup", parameters: { type: "object" } };
    const payload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; tools: ToolSchema[]; responsesLite: boolean }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.5", tools: [tool], responsesLite: true });

    assert.equal(Object.hasOwn(payload, "tools"), false);
    assert.equal(Object.hasOwn(payload, "instructions"), false);
    assert.equal(payload.parallel_tool_calls, false);
    assert.equal(payload.tool_choice, "auto");
    assert.deepEqual(payload.input, [
      {
        type: "additional_tools",
        role: "developer",
        tools: [{ type: "function", name: "lookup", description: "Lookup", parameters: { type: "object" }, strict: false }],
      },
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
  });

  it("transports a user-only chat without inventing empty instructions", async () => {
    for (const responsesLite of [false, true]) {
      const provider = testProvider({});
      let captured: Record<string, unknown> | undefined;
      (provider as unknown as {
        postSSE(path: string, payload: Record<string, unknown>): AsyncGenerator<Record<string, unknown>>;
      }).postSSE = async function* (_path, payload) {
        captured = payload;
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
          response: { id: "response-user-only", output: [], usage: testUsage() },
        };
      };

      const response = await provider.chat(
        [{ role: MessageRole.USER, content: "Hello" }],
        { model: "gpt-5.5", responsesLite },
      );

      assert.equal(response.content, "ok");
      assert.equal(Object.hasOwn(captured ?? {}, "instructions"), false);
      const input = captured?.input as Record<string, unknown>[];
      assert.deepEqual(
        input.map((item) => [item.type, item.role]),
        responsesLite
          ? [["additional_tools", "developer"], ["message", "user"]]
          : [["message", "user"]],
      );
    }
  });

  it("treats null image detail in retained history as omitted in Responses Lite", () => {
    const provider = testProvider({});
    const retained = [{
      type: "message",
      role: "user",
      content: [{
        type: "input_image",
        image_url: "data:image/png;base64,AAAA",
        detail: null,
      }],
    }];
    const messages: Message[] = [
      { role: MessageRole.SYSTEM, content: "You are helpful." },
      {
        role: MessageRole.SYSTEM,
        content: `${REMOTE_COMPACTION_MARKER}\n${JSON.stringify(retained)}`,
      },
    ];
    const payload = (provider as unknown as {
      responsesPayload(
        messages: Message[],
        opts: { model: string; responsesLite: boolean },
      ): Record<string, unknown>;
    }).responsesPayload(messages, { model: "gpt-5.6-sol", responsesLite: true });

    assert.deepEqual((payload.input as Record<string, unknown>[]).slice(2), retained);
  });

  it("does not treat opaque retained metadata as a prompt cache breakpoint", () => {
    const provider = testProvider({});
    const retained = [{
      type: "message",
      role: "user",
      content: [{
        type: "input_text",
        text: "hi",
        metadata: { prompt_cache_breakpoint: "opaque client data" },
      }],
    }];
    const messages: Message[] = [
      { role: MessageRole.SYSTEM, content: "You are helpful." },
      {
        role: MessageRole.SYSTEM,
        content: `${REMOTE_COMPACTION_MARKER}\n${JSON.stringify(retained)}`,
      },
    ];
    const payload = (provider as unknown as {
      responsesPayload(
        messages: Message[],
        opts: { model: string; responsesLite: boolean },
      ): Record<string, unknown>;
    }).responsesPayload(messages, { model: "gpt-5.5", responsesLite: false });

    assert.deepEqual(payload.input, retained);
  });

  it("uses GPT-5.6 catalog defaults and the Lite reasoning context", () => {
    const provider = testProvider({});
    const payload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; responsesLite: string }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.6-sol", responsesLite: "auto" });

    assert.equal(payload.model, "gpt-5.6-sol");
    assert.deepEqual(payload.reasoning, { effort: "low", summary: "auto", context: "all_turns" });
    assert.deepEqual(payload.text, { verbosity: "low" });
    assert.equal((payload.input as Record<string, unknown>[])[0].type, "additional_tools");
  });

  it("rejects non-auto tool choice in Responses Lite", () => {
    const provider = testProvider({});
    assert.throws(
      () => (provider as unknown as {
        responsesPayload(messages: Message[], opts: {
          model: string;
          responsesLite: boolean;
          toolChoice: Record<string, unknown>;
        }): Record<string, unknown>;
      }).responsesPayload(providerMessages(), {
        model: "gpt-5.6-sol",
        responsesLite: true,
        toolChoice: { type: "function", name: "lookup" },
      }),
      /Responses Lite requires tool_choice to be the exact string auto/,
    );
  });

  it("does not normalize an explicit empty Lite tool choice to auto", () => {
    const provider = testProvider({});
    assert.throws(
      () => (provider as unknown as {
        responsesPayload(messages: Message[], opts: {
          model: string;
          responsesLite: boolean;
          toolChoice: string;
        }): Record<string, unknown>;
      }).responsesPayload(providerMessages(), {
        model: "gpt-5.6-sol",
        responsesLite: true,
        toolChoice: "",
      }),
      /Responses Lite requires tool_choice to be the exact string auto/,
    );
  });

  it("rejects hosted tools in Lite mode and allows classic mode when disabled", () => {
    const previous = process.env.CODEX_AS_API_RESPONSES_LITE;
    const provider = testProvider({});
    const webSearchTool: ToolSchema = {
      name: "web_search",
      description: "Web search",
      parameters: {
        __codex_as_api_tool_type: "web_search",
        openai_tool: { type: "web_search", external_web_access: true },
      },
    };
    const call = () => (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; tools: ToolSchema[] }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.6-sol", tools: [webSearchTool] });

    try {
      process.env.CODEX_AS_API_RESPONSES_LITE = "auto";
      assert.throws(call, (error) => error instanceof ChatGPTOAuthError);

      process.env.CODEX_AS_API_RESPONSES_LITE = "off";
      const payload = call();
      assert.deepEqual(payload.tools, [{ type: "web_search", external_web_access: true }]);
      assert.equal(Object.hasOwn(payload, "instructions"), true);
    } finally {
      if (previous == null) delete process.env.CODEX_AS_API_RESPONSES_LITE;
      else process.env.CODEX_AS_API_RESPONSES_LITE = previous;
    }
  });

  it("fails image generation before transport when Lite lacks a standalone executor", async () => {
    const previous = process.env.CODEX_AS_API_RESPONSES_LITE;
    process.env.CODEX_AS_API_RESPONSES_LITE = "auto";
    try {
      const provider = testProvider({});
      await assert.rejects(
        provider.generateImage("draw a circle", { model: "gpt-5.6-sol" }),
        (error) => error instanceof ChatGPTOAuthError,
      );
    } finally {
      if (previous == null) delete process.env.CODEX_AS_API_RESPONSES_LITE;
      else process.env.CODEX_AS_API_RESPONSES_LITE = previous;
    }
  });

  it("allows explicit classic mode for GPT-5.6 image generation", async () => {
    const provider = testProvider({});
    const imageItem = {
      type: "image_generation_call",
      id: "image-1",
      status: "completed",
      result: "data:image/png;base64,AA",
    };
    (provider as unknown as {
      postSSE(path: string, payload: Record<string, unknown>): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* (_path, payload) {
      assert.equal(Object.hasOwn(payload, "tools"), true);
      yield {
        type: "response.output_item.done",
        item: imageItem,
      };
      yield {
        type: "response.completed",
        response: { id: "response-image", output: [], usage: testUsage() },
      };
    };

    const images = await provider.generateImage("draw", {
      model: "gpt-5.6-sol",
      responsesLite: false,
    });
    assert.equal(images[0].result, "data:image/png;base64,AA");
  });

  it("uses the shared capability table and canonicalizes fast service tier", () => {
    const provider = testProvider({});
    const payload = (provider as unknown as {
      responsesPayload(
        messages: Message[],
        opts: { model: string; responsesLite: string; serviceTier: string },
      ): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.5", responsesLite: "auto", serviceTier: "fast" });
    const responsesPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; serviceTier: string }): Record<string, unknown>;
    }).responsesPayload.bind(provider);

    assert.equal(Object.hasOwn(payload, "tools"), true);
    assert.deepEqual(payload.text, { verbosity: "low" });
    assert.equal(payload.service_tier, "priority");
    assert.throws(
      () => responsesPayload(providerMessages(), { model: "unknown-model", serviceTier: "priority" }),
      (error) => error instanceof ChatGPTOAuthError,
    );
    const defaultPayload = responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      serviceTier: "default",
    });
    assert.equal(Object.hasOwn(defaultPayload, "service_tier"), false);
  });

  it("parallel tool calls depend only on the transport mode", () => {
    const provider = testProvider({});
    const payload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; parallelToolCalls: boolean; responsesLite: boolean }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.5", parallelToolCalls: true, responsesLite: false });
    assert.equal(payload.parallel_tool_calls, true);
    const sparkPayload = (provider as unknown as {
      responsesPayload(messages: Message[], options: Record<string, unknown>): Record<string, unknown>;
    }).responsesPayload(providerMessages(), {
      model: "gpt-5.3-codex-spark",
      parallelToolCalls: true,
      responsesLite: false,
    });
    assert.equal(sparkPayload.parallel_tool_calls, true);
    assert.throws(
      () => (provider as unknown as {
        responsesPayload(messages: Message[], options: Record<string, unknown>): Record<string, unknown>;
      }).responsesPayload(providerMessages(), {
        model: "gpt-5.5",
        parallelToolCalls: true,
        responsesLite: true,
      }),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError,
    );
  });

  it("keeps compact tools top-level in explicit classic mode", async () => {
    const provider = testProvider({});
    let captured: Record<string, unknown> | undefined;
    (provider as unknown as {
      postJSON(path: string, payload: Record<string, unknown>): Promise<Record<string, unknown>>;
    }).postJSON = async (_path, payload) => {
      captured = payload;
      return { output: [] };
    };
    const tool: ToolSchema = {
      name: "lookup",
      description: "Lookup",
      parameters: { type: "object" },
    };

    await provider.compactMessages(providerMessages(), {
      model: "gpt-5.6-sol",
      responsesLite: false,
      tools: [tool],
      promptCacheKey: "compact-cache-key",
      serviceTier: "priority",
      text: {
        verbosity: "high",
        format: { type: "text" },
      },
    });
    assert.deepEqual(captured?.tools, [{
      type: "function",
      name: "lookup",
      description: "Lookup",
      parameters: { type: "object" },
      strict: false,
    }]);
    assert.equal(captured?.instructions, "You are helpful.");
    assert.equal(captured?.prompt_cache_key, "compact-cache-key");
    assert.equal(captured?.service_tier, "priority");
    assert.deepEqual(captured?.text, {
      verbosity: "high",
      format: { type: "text" },
    });
  });

  it("classifies malformed upstream compact output as a protocol error", async () => {
    for (const malformed of [
      { type: "additional_tools", role: "developer", tools: [42] },
      { type: "reasoning", summary: "bad" },
      { type: "function_call", call_id: "call-1", name: "lookup", arguments: 42 },
      { type: "future_compaction_item" },
      { type: "message", role: "system", content: [] },
    ]) {
      const provider = testProvider({});
      (provider as unknown as {
        postJSON(path: string, payload: Record<string, unknown>): Promise<Record<string, unknown>>;
      }).postJSON = async () => ({ output: [malformed] });
      await assert.rejects(
        provider.compactMessages(providerMessages(), {
          model: "gpt-5.6-sol",
          responsesLite: false,
        }),
        (error) => error instanceof ChatGPTOAuthProtocolError,
      );
    }
  });

  it("rejects an empty compact prompt cache key", async () => {
    const provider = testProvider({});
    await assert.rejects(provider.compactMessages(providerMessages(), {
      model: "gpt-5.6-sol",
      responsesLite: false,
      promptCacheKey: "",
    }), (error) => error instanceof ChatGPTOAuthError);
  });

  it("rejects unsupported compact service tiers through the model capability catalog", async () => {
    const provider = testProvider({});
    let captured: Record<string, unknown> | undefined;
    (provider as unknown as {
      postJSON(path: string, payload: Record<string, unknown>): Promise<Record<string, unknown>>;
    }).postJSON = async (_path, payload) => {
      captured = payload;
      return { output: [] };
    };
    await assert.rejects(provider.compactMessages(providerMessages(), {
      model: "gpt-5.6-sol",
      responsesLite: false,
      serviceTier: "flex",
    }), (error) => error instanceof ChatGPTOAuthError);
    assert.equal(captured, undefined);
  });

  it("omits compact instructions when no base system instruction exists", async () => {
    for (const responsesLite of [false, true]) {
      const provider = testProvider({});
      let captured: Record<string, unknown> | undefined;
      (provider as unknown as {
        postJSON(path: string, payload: Record<string, unknown>): Promise<Record<string, unknown>>;
      }).postJSON = async (_path, payload) => {
        captured = payload;
        return { output: [] };
      };

      await provider.compactMessages(
        [{ role: MessageRole.USER, content: "history" }],
        { model: "gpt-5.6-sol", responsesLite },
      );
      assert.equal(Object.hasOwn(captured ?? {}, "instructions"), false);
      if (responsesLite) {
        assert.deepEqual((captured?.input as Record<string, unknown>[]).map((item) => item.type), [
          "additional_tools",
          "message",
        ]);
      }
    }
  });

  it("rejects hosted compact tools in Lite mode before transport", async () => {
    const provider = testProvider({});
    const webSearchTool: ToolSchema = {
      name: "web_search",
      description: "Web search",
      parameters: {
        __codex_as_api_tool_type: "web_search",
        openai_tool: { type: "web_search", external_web_access: true },
      },
    };
    await assert.rejects(
      provider.compactMessages(providerMessages(), {
        model: "gpt-5.6-sol",
        responsesLite: true,
        tools: [webSearchTool],
      }),
      (error) => error instanceof ChatGPTOAuthError,
    );
  });

  it("rejects reasoning mode and unsupported private request fields", () => {
    const provider = testProvider({});
    const responsesPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: Record<string, unknown>): Record<string, unknown>;
    }).responsesPayload.bind(provider);
    for (const opts of [
      { reasoning: { mode: "standard", context: "current_turn" } },
      { reasoning: { mode: "pro" } },
      { safetyIdentifier: "user_7ccdef" },
      { promptCacheOptions: { mode: "implicit", ttl: "30m" } },
      { promptCacheOptions: { mode: "explicit" } },
    ]) {
      assert.throws(
        () => responsesPayload(providerMessages(), {
          model: "gpt-5.6-sol",
          responsesLite: false,
          ...opts,
        }),
        (error) => error instanceof ChatGPTOAuthError,
      );
    }
  });

  it("rejects conflicting reasoning effort and invalid GPT-5.6-only fields", () => {
    const provider = testProvider({});
    const responsesPayload = (opts: Record<string, unknown>) => (
      provider as unknown as {
        responsesPayload(messages: Message[], options: Record<string, unknown>): Record<string, unknown>;
      }
    ).responsesPayload(providerMessages(), opts);

    assert.throws(() => responsesPayload({
      model: "gpt-5.6-sol",
      reasoningEffort: "high",
      reasoning: { effort: "low" },
    }), (error) => error instanceof ChatGPTOAuthError);
    assert.throws(() => responsesPayload({
      model: "gpt-5.5",
      reasoning: { mode: "pro" },
    }), (error) => error instanceof ChatGPTOAuthError);
    assert.throws(() => responsesPayload({
      model: "gpt-5.5",
      promptCacheOptions: { mode: "explicit" },
    }), (error) => error instanceof ChatGPTOAuthError);
    for (const opts of [
      { model: "gpt-5.6-sol", safetyIdentifier: "x".repeat(65) },
      { model: "gpt-5.6-sol", promptCacheOptions: { mode: "future" } },
      { model: "gpt-5.6-sol", promptCacheOptions: { mode: null } },
      { model: "gpt-5.6-sol", promptCacheOptions: { ttl: "1h" } },
      { model: "gpt-5.6-sol", promptCacheOptions: { ttl: null } },
      { model: "gpt-5.6-sol", reasoning: { mode: "future" } },
      { model: "gpt-5.6-sol", reasoning: { context: "future" } },
    ]) {
      assert.throws(
        () => responsesPayload(opts),
        (error) => error instanceof ChatGPTOAuthError,
      );
    }
    const breakpointMessages: Message[] = [
      { role: MessageRole.SYSTEM, content: "instructions" },
      {
        role: MessageRole.USER,
        content: "cache",
        structured_content: [{
          type: "text",
          text: "cache",
          prompt_cache_breakpoint: { mode: "explicit" },
        }],
      },
    ];
    const providerWithPayload = provider as unknown as {
      responsesPayload(messages: Message[], options: Record<string, unknown>): Record<string, unknown>;
    };
    for (const model of ["gpt-5.6-sol", "gpt-5.5"]) {
      assert.throws(() => providerWithPayload.responsesPayload(breakpointMessages, {
        model,
        responsesLite: false,
      }), (error) => error instanceof ChatGPTOAuthError);
    }
  });

  it("fails loudly for non-all_turns Lite context and applies the private compact default", async () => {
    const provider = testProvider({});
    assert.throws(() => (
      provider as unknown as {
        responsesPayload(messages: Message[], opts: Record<string, unknown>): Record<string, unknown>;
      }
    ).responsesPayload(providerMessages(), {
      model: "gpt-5.6-sol",
      responsesLite: true,
      reasoning: { context: "current_turn" },
    }), (error) => error instanceof ChatGPTOAuthError);

    let captured: Record<string, unknown> | undefined;
    (provider as unknown as {
      postJSON(path: string, payload: Record<string, unknown>): Promise<Record<string, unknown>>;
    }).postJSON = async (_path, payload) => {
      captured = payload;
      return { output: [] };
    };
    await provider.compactMessages(providerMessages(), {
      model: "gpt-5.6-sol",
      responsesLite: true,
    });
    assert.deepEqual(captured?.reasoning, { effort: "low", summary: "auto", context: "all_turns" });
    assert.equal(Object.hasOwn(captured ?? {}, "previous_response_id"), false);
    assert.equal(Object.hasOwn(captured ?? {}, "prompt_cache_options"), false);
    await assert.rejects(
      provider.compactMessages(providerMessages(), {
        model: "gpt-5.6-sol",
        previousResponseId: "",
      }),
      (error) => error instanceof ChatGPTOAuthError,
    );
  });

  it("rejects cache breakpoints in Lite messages", () => {
    const provider = testProvider({});
    assert.throws(() => (provider as unknown as {
      responsesPayload(messages: Message[], opts: Record<string, unknown>): Record<string, unknown>;
    }).responsesPayload([
      { role: MessageRole.SYSTEM, content: "You are helpful." },
      {
        role: MessageRole.USER,
        content: "look",
        structured_content: [{
          type: "image_url",
          image_url: "data:image/png;base64,AAAA",
          detail: "original",
          prompt_cache_breakpoint: { mode: "explicit" },
        }],
      },
    ], {
      model: "gpt-5.6-sol",
      responsesLite: true,
    }), (error) => error instanceof ChatGPTOAuthError);
  });

  it("preserves supported image detail for classic image requests", async () => {
    const provider = testProvider({});
    let captured: Record<string, unknown> | undefined;
    const imageItem = {
      type: "image_generation_call",
      id: "image-1",
      status: "completed",
      result: "data:image/png;base64,AA",
    };
    (provider as unknown as {
      postSSE(path: string, payload: Record<string, unknown>): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* (_path, payload) {
      captured = payload;
      yield {
        type: "response.output_item.done",
        item: imageItem,
      };
      yield {
        type: "response.completed",
        response: { id: "response-image", output: [], usage: testUsage() },
      };
    };

    await provider.generateImage("draw", {
      model: "gpt-5.6-sol",
      responsesLite: false,
      referenceImages: [{
        image_url: "data:image/png;base64,AAAA",
        detail: "original",
      }],
      reasoningEffort: "medium",
    });
    const input = captured?.input as Record<string, unknown>[];
    const content = input[0].content as Record<string, unknown>[];
    assert.deepEqual(content[1], {
      type: "input_image",
      image_url: "data:image/png;base64,AAAA",
      detail: "original",
    });
    assert.deepEqual(captured?.reasoning, { effort: "medium", summary: "auto" });
    assert.equal(Object.hasOwn(captured ?? {}, "safety_identifier"), false);
  });

  it("uses explicit Codex session identity and preserves metadata lifetimes", () => {
    const provider = testProvider({ authJsonPath: writeAuthFile() });
    const responsesPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: Record<string, unknown>): Record<string, unknown>;
    }).responsesPayload.bind(provider);
    const first = responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      clientMetadata: {
        app: "kept",
        session_id: "session-root",
        turn_id: "user-value",
      },
      codexMetadata: true,
    });
    const second = responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      clientMetadata: { session_id: "session-root" },
      codexMetadata: true,
    });
    const child = responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      clientMetadata: {
        session_id: "session-root",
        thread_id: "thread-child",
      },
      codexMetadata: true,
    });
    const firstMetadata = first.client_metadata as Record<string, string>;
    const secondMetadata = second.client_metadata as Record<string, string>;
    const childMetadata = child.client_metadata as Record<string, string>;
    const turnMetadata = JSON.parse(
      firstMetadata["x-codex-turn-metadata"],
    ) as Record<string, string>;

    assert.equal(firstMetadata.app, "kept");
    assert.equal(firstMetadata.session_id, "session-root");
    assert.equal(firstMetadata.thread_id, "session-root");
    assert.equal(childMetadata.session_id, "session-root");
    assert.equal(childMetadata.thread_id, "thread-child");
    assert.notEqual(firstMetadata.turn_id, "user-value");
    assert.notEqual(firstMetadata.turn_id, secondMetadata.turn_id);
    assert.equal(
      firstMetadata["x-codex-installation-id"],
      secondMetadata["x-codex-installation-id"],
    );
    assert.equal(
      firstMetadata["x-codex-window-id"],
      secondMetadata["x-codex-window-id"],
    );
    assert.deepEqual(
      {
        session_id: turnMetadata.session_id,
        thread_id: turnMetadata.thread_id,
        turn_id: turnMetadata.turn_id,
        source: turnMetadata.source,
      },
      {
        session_id: "session-root",
        thread_id: "session-root",
        turn_id: firstMetadata.turn_id,
        source: "codex-as-api",
      },
    );
    assert.equal(first.prompt_cache_key, "session-root");
    assert.equal(child.prompt_cache_key, "session-root");
  });

  it("derives installation identity from the CODEX_HOME auth path", () => {
    const previousCodexHome = process.env.CODEX_HOME;
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-metadata-home-"));
    process.env.CODEX_HOME = codexHome;
    try {
      const options = {
        model: "gpt-5.5",
        clientMetadata: { session_id: "session-root" },
        codexMetadata: true,
      };
      const implicitProvider = testProvider();
      const explicitProvider = testProvider({ authJsonPath: path.join(codexHome, "auth.json") });
      const responsesPayload = (provider: ChatGPTOAuthProvider) => (
        provider as unknown as {
          responsesPayload(messages: Message[], opts: Record<string, unknown>): Record<string, unknown>;
        }
      ).responsesPayload(providerMessages(), options);

      const implicit = responsesPayload(implicitProvider).client_metadata as Record<string, string>;
      const explicit = responsesPayload(explicitProvider).client_metadata as Record<string, string>;
      assert.equal(
        implicit["x-codex-installation-id"],
        explicit["x-codex-installation-id"],
      );
    } finally {
      if (previousCodexHome === undefined) delete process.env.CODEX_HOME;
      else process.env.CODEX_HOME = previousCodexHome;
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("requires session identity when Codex metadata is enabled", () => {
    const provider = testProvider({ authJsonPath: writeAuthFile() });
    const responsesPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: Record<string, unknown>): Record<string, unknown>;
    }).responsesPayload.bind(provider);

    assert.throws(
      () => responsesPayload(providerMessages(), {
        model: "gpt-5.5",
        clientMetadata: { thread_id: "orphan-thread" },
        codexMetadata: true,
      }),
      ChatGPTOAuthInvalidRequestError,
    );
  });

  it("prefers an explicit prompt cache key, then a valid session identity, then omission", () => {
    const provider = testProvider({});
    const responsesPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: Record<string, unknown>): Record<string, unknown>;
    }).responsesPayload.bind(provider);
    const explicit = responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      promptCacheKey: "explicit-cache",
      clientMetadata: { session_id: "session-cache" },
      codexMetadata: false,
    });
    const derived = responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      clientMetadata: { session_id: "session-cache" },
      codexMetadata: false,
    });
    const absent = responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      codexMetadata: false,
    });
    assert.equal(explicit.prompt_cache_key, "explicit-cache");
    assert.equal(derived.prompt_cache_key, "session-cache");
    assert.equal(Object.hasOwn(absent, "prompt_cache_key"), false);
    for (const clientMetadata of [
      { session_id: "   " },
      { session_id: 42 },
      { thread_id: "" },
    ]) {
      assert.throws(
        () => responsesPayload(providerMessages(), {
          model: "gpt-5.5",
          promptCacheKey: "explicit-cache",
          clientMetadata,
          codexMetadata: false,
        }),
        ChatGPTOAuthInvalidRequestError,
      );
    }
  });
});

describe("Responses stream completion", () => {
  it("cancels and unlocks an upstream body when completion arrives before EOF", async () => {
    const authPath = writeAuthFile();
    const originalFetch = globalThis.fetch;
    let cancelCount = 0;
    let responseBody: ReadableStream<Uint8Array> | undefined;
    try {
      const encoded = new TextEncoder().encode([
        {
          type: "response.output_item.done",
          item: {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "done" }],
          },
        },
        {
          type: "response.completed",
          response: { id: "response-cancel", end_turn: true, output: [], usage: testUsage() },
        },
      ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join(""));
      responseBody = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(encoded);
        },
        cancel() {
          cancelCount++;
        },
      });
      globalThis.fetch = async () => new Response(responseBody, {
        status: 200,
        headers: { "content-type": "text/event-stream" },
      });

      const response = await testProvider({ authJsonPath: authPath }).chat(
        providerMessages(),
        { model: "gpt-5.5" },
      );

      assert.equal(response.content, "done");
      assert.equal(cancelCount, 1);
      assert.equal(responseBody.locked, false);
    } finally {
      globalThis.fetch = originalFetch;
      fs.rmSync(path.dirname(authPath), { recursive: true, force: true });
    }
  });

  it("renders text and tool output from authoritative output_item.done items", async () => {
    const provider = testProvider();
    const output = [
      {
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "completion text" }],
      },
      {
        type: "function_call",
        id: "item-1",
        call_id: "call-1",
        name: "lookup",
        arguments: '{"query":"one"}',
      },
    ];
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      for (const item of output) {
        yield { type: "response.output_item.done", item };
      }
      yield {
        type: "response.completed",
        response: { id: "response-completion-only", usage: testUsage(), output: [] },
      };
    };

    const events: Record<string, unknown>[] = [];
    for await (const event of provider.chatStream(providerMessages(), {
      model: "gpt-5.6-sol",
    })) {
      events.push(event);
    }

    assert.deepEqual(
      events.map((event) => event.type),
      ["tool_call", "content", "finish"],
    );
    assert.deepEqual(events[0], {
      type: "tool_call",
      id: "call-1",
      name: "lookup",
      arguments: '{"query":"one"}',
    });
    assert.deepEqual(events[1], { type: "content", text: "completion text" });
    assert.equal(events[2].finish_reason, "tool_calls");
    assert.equal(events[2].response_id, "response-completion-only");
  });

  it("rejects duplicate function call ids while streaming", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield {
        type: "response.output_item.done",
        item: { type: "function_call", call_id: "duplicate-call", name: "first", arguments: "{}" },
      };
      yield {
        type: "response.output_item.done",
        item: { type: "function_call", call_id: "duplicate-call", name: "second", arguments: "{}" },
      };
    };

    await assert.rejects(async () => {
      for await (const _event of provider.chatStream(providerMessages(), { model: "gpt-5.6-sol" })) {
        // Consume the real normalized stream through the duplicate boundary.
      }
    }, /duplicate call_id/);
  });

  it("rejects duplicate function call ids in non-streaming chat", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield {
        type: "response.output_item.done",
        item: { type: "function_call", call_id: "duplicate-call", name: "first", arguments: "{}" },
      };
      yield {
        type: "response.output_item.done",
        item: { type: "function_call", call_id: "duplicate-call", name: "second", arguments: "{}" },
      };
    };

    await assert.rejects(
      provider.chat(providerMessages(), { model: "gpt-5.6-sol" }),
      /duplicate call_id/,
    );
  });

  it("rejects custom_tool_call output while streaming", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield {
        type: "response.output_item.done",
        item: {
          type: "custom_tool_call",
          call_id: "custom-call",
          name: "shell",
          input: '{"command":"pwd"}',
        },
      };
    };

    await assert.rejects(async () => {
      for await (const _event of provider.chatStream(providerMessages(), { model: "gpt-5.6-sol" })) {
        // Consume the stream through output-item validation.
      }
    }, /custom_tool_call/);
  });

  it("rejects custom_tool_call output in non-streaming chat", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield {
        type: "response.output_item.done",
        item: {
          type: "custom_tool_call",
          call_id: "custom-call",
          name: "shell",
          input: '{"command":"pwd"}',
        },
      };
    };

    await assert.rejects(
      provider.chat(providerMessages(), { model: "gpt-5.6-sol" }),
      /custom_tool_call/,
    );
  });

  it("ignores additive completed.output without output_item.done events", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield {
        type: "response.completed",
        response: {
          id: "response-mismatch",
          usage: testUsage(),
          output: [{
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "uncommitted" }],
          }],
        },
      };
    };

    const response = await provider.chat(providerMessages(), { model: "gpt-5.6-sol" });
    assert.equal(response.content, "");
    assert.deepEqual(response.tool_calls, []);
  });

  it("treats completed empty responses without end_turn as terminal", async () => {
    for (const output of [undefined, []]) {
      const provider = testProvider();
      (provider as unknown as {
        postSSE(): AsyncGenerator<Record<string, unknown>>;
      }).postSSE = async function* () {
        yield {
          type: "response.completed",
          response: {
            id: output == null ? "response-empty-absent" : "response-empty-array",
            usage: testUsage(),
            ...(output == null ? {} : { output }),
          },
        };
      };

      const response = await provider.chat(providerMessages(), { model: "gpt-5.6-sol" });
      assert.equal(response.content, "");
      assert.deepEqual(response.tool_calls, []);
      assert.equal(response.finish_reason, "stop");
    }
  });

  it("accepts an assistant output item with an empty content array", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield {
        type: "response.output_item.done",
        item: { type: "message", role: "assistant", content: [] },
      };
      yield {
        type: "response.completed",
        response: { id: "response-empty-message", usage: testUsage(), output: [] },
      };
    };

    const response = await provider.chat(providerMessages(), { model: "gpt-5.6-sol" });
    assert.equal(response.content, "");
    assert.deepEqual(response.tool_calls, []);
    assert.equal(response.finish_reason, "stop");
  });

  it("treats only explicit false response.completed end_turn as non-terminal", async () => {
    for (const [endTurn, expected] of [
      [undefined, "stop"],
      [false, null],
      [true, "stop"],
    ] as const) {
      const provider = testProvider();
      (provider as unknown as {
        postSSE(): AsyncGenerator<Record<string, unknown>>;
      }).postSSE = async function* () {
        yield {
          type: "response.output_item.done",
          item: {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "done" }],
          },
        };
        yield {
          type: "response.completed",
          response: {
            id: `response-end-turn-${String(endTurn)}`,
            usage: testUsage(),
            output: [],
            ...(endTurn === undefined ? {} : { end_turn: endTurn }),
          },
        };
      };

      const response = await provider.chat(providerMessages(), { model: "gpt-5.6-sol" });
      assert.equal(response.finish_reason, expected);
    }
  });

  it("rejects completed text that differs from streamed text", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield { type: "response.output_text.delta", delta: "partial" };
      yield {
        type: "response.output_item.done",
        item: {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "different" }],
        },
      };
      yield {
        type: "response.completed",
        response: {
          id: "response-text-mismatch",
          usage: testUsage(),
          output: [],
        },
      };
    };

    await assert.rejects(async () => {
      for await (const _event of provider.chatStream(providerMessages(), {
        model: "gpt-5.6-sol",
      })) {
        // Drain the stream so the terminal consistency check runs.
      }
    }, ChatGPTOAuthProtocolError);
  });

  it("rejects streamed reasoning families that differ from done items", async () => {
    for (const [deltaEvent, reasoningItem] of [
      [
        { type: "response.reasoning_summary_text.delta", delta: "streamed", summary_index: 0 },
        { type: "reasoning", summary: [{ type: "summary_text", text: "done" }] },
      ],
      [
        { type: "response.reasoning_text.delta", delta: "streamed", content_index: 0 },
        {
          type: "reasoning",
          summary: [],
          content: [{ type: "reasoning_text", text: "done" }],
        },
      ],
    ] as const) {
      const provider = testProvider();
      (provider as unknown as {
        postSSE(): AsyncGenerator<Record<string, unknown>>;
      }).postSSE = async function* () {
        yield deltaEvent;
        yield { type: "response.output_item.done", item: reasoningItem };
        yield { type: "response.completed", response: { id: "reasoning-mismatch" } };
      };

      await assert.rejects(
        provider.chat(providerMessages(), { model: "gpt-5.6-sol" }),
        (error) => error instanceof ChatGPTOAuthProtocolError,
      );
    }
  });

  it("rejects unsupported output_item.done items instead of silently dropping them", async () => {
    const provider = testProvider();
    const unsupported = { type: "future_output", value: true };
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield { type: "response.output_item.done", item: unsupported };
      yield {
        type: "response.completed",
        response: {
          id: "response-unsupported",
          usage: testUsage(),
          output: [],
        },
      };
    };

    await assert.rejects(
      provider.chat(providerMessages(), { model: "gpt-5.6-sol" }),
      ChatGPTOAuthProtocolError,
    );
    await assert.rejects(
      provider.chat(providerMessages(), {
        model: "gpt-5.6-sol",
        previousResponseId: "response-unsupported",
      }),
      ChatGPTOAuthInvalidRequestError,
    );
  });

  it("uses tool_calls for client function calls regardless of end_turn", async () => {
    const toolCall = {
      type: "function_call",
      id: "item-1",
      call_id: "call-1",
      name: "lookup",
      arguments: '{"query":"one"}',
    };
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield { type: "response.output_item.done", item: toolCall };
      yield {
        type: "response.completed",
        response: { id: "response-1", end_turn: false, usage: testUsage(), output: [] },
      };
    };

    const events: Record<string, unknown>[] = [];
    for await (const event of provider.chatStream(providerMessages(), { model: "gpt-5.6-sol" })) {
      events.push(event);
    }

    const emittedTool = events.find((event) => event.type === "tool_call");
    const finish = events.find((event) => event.type === "finish");
    assert.equal(emittedTool?.id, toolCall.call_id);
    assert.equal(finish?.finish_reason, "tool_calls");
    assert.equal(finish?.response_id, "response-1");
  });

  it("fails when the upstream SSE stream ends before response.completed", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield { type: "response.output_text.delta", delta: "partial" };
    };

    await assert.rejects(async () => {
      for await (const _event of provider.chatStream(providerMessages(), {
        model: "gpt-5.6-sol",
      })) {
        // Drain the provider stream so the terminal protocol check runs.
      }
    }, /ended before response\.completed/);
  });

  it("accepts official partial added items and preserves reasoning indexes", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield { type: "codex.response.metadata", metadata: "opaque" };
      yield { type: "response.metadata", metadata: null };
      yield { type: "responsesapi.websocket_timing", future: ["opaque"] };
      yield { type: "response.created", response: { id: "response-1" } };
      yield { type: "response.output_text.delta", delta: "" };
      yield {
        type: "response.output_item.added",
        item: {
          type: "function_call",
          call_id: "call-1",
          name: "apply_patch",
          arguments: "",
        },
      };
      yield { type: "response.reasoning_summary_part.added", summary_index: 2 };
      yield {
        type: "response.reasoning_summary_text.delta",
        delta: "",
        summary_index: 2,
      };
      yield {
        type: "response.reasoning_summary_text.delta",
        delta: "summary",
        summary_index: 2,
      };
      yield {
        type: "response.reasoning_summary_text.done",
        item_id: "",
        text: "",
        summary_index: 2,
      };
      yield {
        type: "response.reasoning_text.delta",
        delta: "",
        content_index: 4,
      };
      yield {
        type: "response.reasoning_text.delta",
        delta: "raw",
        content_index: 4,
      };
      yield {
        type: "response.output_item.done",
        item: {
          type: "reasoning",
          summary: [{ type: "summary_text", text: "summary" }],
          content: [{ type: "text", text: "raw" }],
        },
      };
      yield {
        type: "response.output_item.done",
        item: {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "done" }],
        },
      };
      yield {
        type: "response.completed",
        response: { id: "response-1", usage: testUsage(), output: [] },
      };
    };

    const events: StreamEvent[] = [];
    for await (const event of provider.chatStream(providerMessages(), { model: "gpt-5.6-sol" })) {
      events.push(event);
    }

    assert.deepEqual(
      events.filter((event) => [
        "reasoning_section_break",
        "reasoning_delta",
        "reasoning_raw_delta",
      ].includes(event.type)),
      [
        { type: "reasoning_section_break", summary_index: 2 },
        { type: "reasoning_delta", text: "summary", summary_index: 2 },
        { type: "reasoning_raw_delta", text: "raw", content_index: 4 },
      ],
    );
  });

  it("validates pinned optional ResponseItem fields without rejecting additive fields", async () => {
    const validItems: Record<string, unknown>[] = [
      {
        type: "message",
        id: "",
        role: "assistant",
        content: [],
        phase: "commentary",
        internal_chat_message_metadata_passthrough: {
          turn_id: null,
          create_time: 1.5,
          content_item_kinds: { future: true },
        },
        future: true,
      },
      {
        type: "reasoning",
        summary: [],
        content: [{ type: "text", text: "raw", future: true }],
        encrypted_content: null,
        future: true,
      },
      {
        type: "function_call",
        name: "",
        call_id: "",
        arguments: "not-json",
        namespace: null,
        encrypted_function_args: ["a", ""],
        future: true,
      },
      {
        type: "web_search_call",
        id: "",
        status: null,
        action: { type: "search", query: "q", sources: [] },
        future: true,
      },
      {
        type: "image_generation_call",
        id: "",
        status: "",
        result: "",
        revised_prompt: null,
        future: true,
      },
    ];

    for (const item of validItems) {
      const provider = testProvider();
      (provider as unknown as {
        postSSE(): AsyncGenerator<Record<string, unknown>>;
      }).postSSE = async function* () {
        yield { type: "response.output_item.added", item };
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
          response: { id: "response-optional", end_turn: true, usage: testUsage() },
        };
      };
      for await (const _event of provider.chatStream(providerMessages(), {
        model: "gpt-5.5",
        responsesLite: false,
      })) {
        // Drain the stream so every validator runs.
      }
    }

    const invalidItems: Record<string, unknown>[] = [
      { type: "message", id: 1, role: "assistant", content: [] },
      {
        type: "message",
        role: "assistant",
        content: [],
        internal_chat_message_metadata_passthrough: [],
      },
      {
        type: "message",
        role: "assistant",
        content: [],
        internal_chat_message_metadata_passthrough: { turn_id: 1 },
      },
      {
        type: "message",
        role: "assistant",
        content: [],
        internal_chat_message_metadata_passthrough: { create_time: "now" },
      },
      { type: "message", role: "assistant", content: [], phase: "future" },
      {
        type: "function_call",
        name: "f",
        call_id: "c",
        arguments: "{}",
        namespace: 1,
      },
      {
        type: "function_call",
        name: "f",
        call_id: "c",
        arguments: "{}",
        encrypted_function_args: [1],
      },
      { type: "custom_tool_call", name: "f", call_id: "c", input: "", status: 1 },
      { type: "web_search_call", status: 1 },
      { type: "image_generation_call", status: "", result: "", revised_prompt: 1 },
    ];

    for (const item of invalidItems) {
      const provider = testProvider();
      (provider as unknown as {
        postSSE(): AsyncGenerator<Record<string, unknown>>;
      }).postSSE = async function* () {
        yield { type: "response.output_item.added", item };
      };
      await assert.rejects(async () => {
        for await (const _event of provider.chatStream(providerMessages(), {
          model: "gpt-5.5",
          responsesLite: false,
        })) {
          // Drain the stream so validation errors surface.
        }
      }, (error) => error instanceof ChatGPTOAuthProtocolError);
    }
  });

  it("rejects unsupported or malformed consumed events instead of silently ignoring them", async () => {
    const credentialSentinel = "access_token=UPSTREAM_TYPE_SENTINEL";
    for (const event of [
      { type: "response.file_search_call.in_progress" },
      { type: "response.file_search_call.searching" },
      { type: "response.file_search_call.completed" },
      { type: "response.code_interpreter_call.in_progress" },
      { type: "response.code_interpreter_call.interpreting" },
      { type: "response.code_interpreter_call_code.delta" },
      { type: "response.code_interpreter_call_code.done" },
      { type: "response.code_interpreter_call.completed" },
      { type: "response.mcp_call.in_progress" },
      { type: "response.mcp_call_arguments.delta" },
      { type: "response.mcp_call_arguments.done" },
      { type: "response.mcp_call.completed" },
      { type: "response.mcp_call.failed" },
      { type: "response.mcp_list_tools.in_progress" },
      { type: "response.mcp_list_tools.completed" },
      { type: "response.mcp_list_tools.failed" },
      { type: "response.shell_call_command.added" },
      { type: "response.shell_call_command.delta" },
      { type: "response.shell_call_command.done" },
      { type: "response.shell_call_output_content.delta" },
      { type: "response.shell_call_output_content.done" },
      { type: "response.audio.delta" },
      { type: "response.audio.done" },
      { type: "response.audio.transcript.delta" },
      { type: "response.audio.transcript.done" },
      { type: "response.refusal.delta" },
      { type: "response.refusal.done" },
      { type: "response.output_text.annotation.added" },
      { type: "response.custom_tool_call_input.delta", delta: "{}", call_id: "call-1" },
      { type: "response.custom_tool_call_input.done", input: "{}", call_id: "call-1" },
      { type: "response.output_text.delta", delta: null },
      { type: "response.created" },
      {
        type: "response.output_item.added",
        item: { type: "computer_call" },
      },
      {
        type: "response.output_item.added",
        item: { type: "custom_tool_call", call_id: "call-1", name: "apply_patch" },
      },
      {
        type: "response.output_item.added",
        item: {
          type: "reasoning",
          summary: [],
          content: [{ type: "text", text: "legacy" }],
        },
      },
      { type: "response.content_part.added" },
      {
        type: "response.content_part.added",
        part: { type: "refusal", refusal: "blocked" },
      },
      {
        type: "response.content_part.done",
        part: { type: "future_content", value: "opaque" },
      },
      {
        type: "response.content_part.done",
        part: { type: "output_text", annotations: [] },
      },
      {
        type: "response.content_part.done",
        part: { type: "output_text", text: "ok", annotations: {} },
      },
      {
        type: "response.content_part.done",
        part: { type: "output_text", text: "ok", logprobs: {} },
      },
      { type: "response.custom_tool_call_input.delta", delta: "{}" },
      { type: "response.reasoning_summary_text.delta", delta: "summary" },
      {
        type: "response.reasoning_summary_text.done",
        text: "summary",
        summary_index: 0,
      },
      { type: "response.reasoning_text.delta", delta: "raw", summary_index: 0 },
      { type: "response.reasoning_summary_part.added", part_index: 0 },
      {
        type: "response.output_item.done",
        item: { type: credentialSentinel, text: "legacy" },
      },
      {
        type: "response.output_item.done",
        item: {
          type: "message",
          role: "assistant",
          content: [{ type: credentialSentinel, text: "legacy" }],
        },
      },
      {
        type: "response.output_item.done",
        item: {
          type: "message",
          role: "assistant",
          content: [{ type: "text", text: "legacy" }],
        },
      },
    ]) {
      const provider = testProvider();
      (provider as unknown as {
        postSSE(): AsyncGenerator<Record<string, unknown>>;
      }).postSSE = async function* () {
        yield event;
      };
      await assert.rejects(
        provider.chat(providerMessages(), { model: "gpt-5.6-sol" }),
        (error) => error instanceof ChatGPTOAuthProtocolError
          && !error.message.includes(credentialSentinel),
      );
    }
  });

  it("ignores unknown event types before a valid completion", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield { type: "", opaque: "ignored" };
      yield { type: "response.future_telemetry", opaque: true };
      yield {
        type: "response.content_part.added",
        part: { type: "output_text", text: "", annotations: [], logprobs: [] },
      };
      yield {
        type: "response.content_part.done",
        part: { type: "output_text", text: "done", annotations: [] },
      };
      yield {
        type: "response.output_item.done",
        item: {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "done" }],
        },
      };
      yield { type: "response.completed", response: { id: "response-1" } };
    };

    const response = await provider.chat(providerMessages(), { model: "gpt-5.6-sol" });
    assert.equal(response.content, "done");
    assert.equal(response.response_id, "response-1");
  });

  it("surfaces upstream error events immediately", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield { type: "error", error: { message: "quota denied" } };
      yield {
        type: "response.output_item.done",
        item: {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "" }],
        },
      };
    };
    await assert.rejects(
      provider.chat(providerMessages(), { model: "gpt-5.6-sol" }),
      (error) => error instanceof ChatGPTOAuthUpstreamError
        && error.status === 502
        && error.message === "OpenAI protocol response error: quota denied",
    );
  });

  it("redacts credential values from parsed and malformed upstream SSE events", async () => {
    const authPath = writeAuthFile();
    const authData = JSON.parse(fs.readFileSync(authPath, "utf8")) as {
      tokens: { access_token: string; refresh_token: string; id_token: string };
    };
    const secrets = [
      authData.tokens.access_token,
      authData.tokens.refresh_token,
      authData.tokens.id_token,
      "acc-123",
    ];
    const reflected = secrets.join(" ");
    const originalFetch = globalThis.fetch;
    let responseBody = "";
    try {
      globalThis.fetch = async () => new Response(
        responseBody,
        { status: 200, headers: { "content-type": "text/event-stream" } },
      );
      for (const body of [
        `data: ${JSON.stringify({
          type: "error",
          error: { message: reflected },
        })}\n\n`,
        `data: ${JSON.stringify({
          type: "response.failed",
          response: { error: { message: reflected } },
        })}\n\n`,
        `data: ${JSON.stringify({
          type: "response.incomplete",
          response: { incomplete_details: { reason: reflected } },
        })}\n\n`,
        `data: ${authData.tokens.access_token}-NOT-JSON\n\n`,
        `data: ${JSON.stringify({ type: authData.tokens.access_token })}\n\n`,
      ]) {
        responseBody = body;
        const provider = testProvider({ authJsonPath: authPath });
        await assert.rejects(
          provider.chat(providerMessages(), { model: "gpt-5.5" }),
          (error) => {
            assert.ok(error instanceof ChatGPTOAuthError);
            for (const secret of secrets) {
              assert.equal(error.message.includes(secret), false);
              assert.equal(error.message.includes(secret.slice(0, 8)), false);
            }
            return true;
          },
        );
      }

      responseBody = [
        {
          type: "response.output_text.delta",
          delta: authData.tokens.access_token,
        },
        {
          type: "response.reasoning_summary_text.delta",
          delta: authData.tokens.refresh_token,
          summary_index: 0,
        },
        {
          type: "response.output_item.done",
          item: {
            type: "reasoning",
            summary: [{ type: "summary_text", text: authData.tokens.refresh_token }],
          },
        },
        {
          type: "response.output_item.done",
          item: {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: authData.tokens.access_token }],
          },
        },
        {
          type: "response.output_item.done",
          item: {
            type: "function_call",
            call_id: "call-normal-secret",
            name: "echo",
            arguments: JSON.stringify({ value: authData.tokens.id_token }),
          },
        },
        {
          type: "response.completed",
          response: { id: "response-normal-content", end_turn: true, output: [] },
        },
      ].map((event) => `data: ${JSON.stringify(event)}\n\n`).join("");
      const response = await testProvider({ authJsonPath: authPath }).chat(
        providerMessages(),
        { model: "gpt-5.5" },
      );
      assert.equal(response.content, authData.tokens.access_token);
      assert.equal(response.reasoning_content, authData.tokens.refresh_token);
      assert.deepEqual(response.tool_calls, [{
        id: "call-normal-secret",
        name: "echo",
        arguments: JSON.stringify({ value: authData.tokens.id_token }),
      }]);
    } finally {
      globalThis.fetch = originalFetch;
      fs.rmSync(path.dirname(authPath), { recursive: true, force: true });
    }
  });

  it("does not reflect unsupported normalized provider event types", async () => {
    const credentialSentinel = "access_token=NORMALIZED_PROVIDER_SECRET";
    const provider = testProvider();
    provider.chatStream = async function* () {
      yield { type: credentialSentinel };
    };
    await assert.rejects(
      provider.chat(providerMessages(), { model: "gpt-5.5" }),
      (error) => error instanceof ChatGPTOAuthProtocolError
        && !error.message.includes(credentialSentinel),
    );
  });

  it("returns immediately after response.completed once a done item was received", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield {
        type: "response.output_item.done",
        item: {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "" }],
        },
      };
      yield {
        type: "response.completed",
        response: { id: "response-1", usage: testUsage(), output: [] },
      };
      yield { type: "response.failed", error: { message: "must be ignored" } };
    };

    const events: Record<string, unknown>[] = [];
    for await (const event of provider.chatStream(providerMessages(), {
      model: "gpt-5.6-sol",
    })) {
      events.push(event);
    }
    assert.deepEqual(events.map((event) => event.type), ["finish"]);
    assert.equal(events[0].response_id, "response-1");
  });

  it("allows absent completion usage and rejects malformed present usage", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
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
        response: { id: "response-no-usage", output: [] },
      };
    };
    const response = await provider.chat(providerMessages(), {
      model: "gpt-5.6-sol",
    });
    assert.equal(response.usage, null);

    const malformed = testProvider();
    (malformed as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
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
        response: { id: "response-bad-usage", output: [], usage: { input_tokens: 1 } },
      };
    };
    await assert.rejects(
      malformed.chat(providerMessages(), { model: "gpt-5.6-sol" }),
      ChatGPTOAuthProtocolError,
    );
  });

  it("rejects malformed response.completed payloads", async () => {
    for (const response of [undefined, null, {}, { id: "" }, { id: "response-1", end_turn: "yes" }]) {
      const provider = testProvider();
      (provider as unknown as {
        postSSE(): AsyncGenerator<Record<string, unknown>>;
      }).postSSE = async function* () {
        yield { type: "response.completed", response };
      };
      await assert.rejects(async () => {
        for await (const _event of provider.chatStream(providerMessages(), {
          model: "gpt-5.6-sol",
        })) {
          // Drain the provider stream so completion validation runs.
        }
      }, (error) => error instanceof ChatGPTOAuthError);
    }
  });

  it("requires response.completed in the image and inspect output collector", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield {
        type: "response.output_item.done",
        item: {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "" }],
        },
      };
    };

    await assert.rejects(
      (provider as unknown as {
        collectResponseOutputItems(payload: Record<string, unknown>, prepared: PreparedModel): Promise<Record<string, unknown>[]>;
      }).collectResponseOutputItems({}, preparedForTest()),
      /ended before response\.completed/,
    );
  });

  it("rejects malformed output_item.done in the image and inspect collector", async () => {
    for (const item of [
      null,
      [],
      "not-an-object",
      { type: "future_item" },
      {
        type: "message",
        role: "assistant",
        content: [
          { type: "output_text", text: "valid" },
          { type: "future_part" },
        ],
      },
      {
        type: "reasoning",
        summary: [],
        encrypted_content: { malformed: true },
      },
    ]) {
      const provider = testProvider();
      (provider as unknown as {
        postSSE(): AsyncGenerator<Record<string, unknown>>;
      }).postSSE = async function* () {
        yield { type: "response.output_item.done", item };
        yield { type: "response.completed", response: { id: "response-1" } };
      };

      await assert.rejects(
        (provider as unknown as {
          collectResponseOutputItems(payload: Record<string, unknown>, prepared: PreparedModel): Promise<Record<string, unknown>[]>;
        }).collectResponseOutputItems({}, preparedForTest()),
        (error) => error instanceof ChatGPTOAuthError,
      );
    }
  });

  it("validates response.completed identity in the image and inspect output collector", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield { type: "response.completed", response: { id: "" } };
    };

    await assert.rejects(
      (provider as unknown as {
        collectResponseOutputItems(payload: Record<string, unknown>, prepared: PreparedModel): Promise<Record<string, unknown>[]>;
      }).collectResponseOutputItems({}, preparedForTest()),
      (error) => error instanceof ChatGPTOAuthError,
    );
  });

  it("returns authoritative done output and does not read events after response.completed", async () => {
    const provider = testProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      const item = { type: "message", role: "assistant", content: [] };
      yield { type: "response.output_item.done", item };
      yield {
        type: "response.completed",
        response: {
          id: "response-1",
          output: [],
          usage: testUsage(),
        },
      };
      yield { type: "response.failed", error: { message: "must be ignored" } };
    };

    const output = await (provider as unknown as {
      collectResponseOutputItems(payload: Record<string, unknown>, prepared: PreparedModel): Promise<Record<string, unknown>[]>;
    }).collectResponseOutputItems({}, preparedForTest());
    assert.deepEqual(output, [{ type: "message", role: "assistant", content: [] }]);
  });
});

describe("local previous_response_id history", () => {
  it("requires compaction only when both model compatibility hashes are known and differ", async () => {
    const provider = testProvider();
    let transportCalls = 0;
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      transportCalls += 1;
      const output = [{
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "ok" }],
      }];
      yield { type: "response.output_item.done", item: output[0] };
      yield {
        type: "response.completed",
        response: { id: `response-${transportCalls}`, output: [], usage: testUsage(), end_turn: true },
      };
    };
    const preparedWithHash = (compHash?: string): PreparedModel => {
      const base = preparedForTest();
      const capability = Object.freeze({
        ...base.capability,
        ...(compHash === undefined ? {} : { compHash }),
      });
      return Object.freeze({
        ...base,
        capability,
        snapshot: Object.freeze({
          ...base.snapshot,
          models: Object.freeze([capability]),
          defaultModel: capability,
        }),
      });
    };

    await provider.chat(providerMessages(), { preparedModel: preparedWithHash("family-a") });
    await provider.chat(providerMessages(), {
      previousResponseId: "response-1",
      preparedModel: preparedWithHash("family-a"),
    });
    await assert.rejects(
      provider.chat(providerMessages(), {
        previousResponseId: "response-1",
        preparedModel: preparedWithHash("family-b"),
      }),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError
        && error.message.includes("requires compaction"),
    );
    await provider.chat(providerMessages(), {
      previousResponseId: "response-1",
      preparedModel: preparedWithHash(),
    });
    assert.equal(transportCalls, 3);
  });

  it("replays exact output_item.done history and supports concurrent branches without forwarding the ID", async () => {
    const provider = testProvider();
    const payloads: Record<string, unknown>[] = [];
    let responseNumber = 0;
    const firstOutput: Record<string, unknown>[] = [
      {
        type: "reasoning",
        id: "reasoning-1",
        encrypted_content: "encrypted-original",
        summary: [],
      },
      {
        type: "function_call",
        id: "function-1",
        call_id: "call-1",
        name: "lookup",
        arguments: '{"query":"one"}',
      },
    ];
    (provider as unknown as {
      postSSE(
        path: string,
        payload: Record<string, unknown>,
      ): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* (_path, payload) {
      payloads.push(payload);
      const id = `response-${++responseNumber}`;
      const output = id === "response-1"
        ? firstOutput
        : [{
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: id }],
          }];
      for (const item of output) {
        yield { type: "response.output_item.done", item };
      }
      await Promise.resolve();
      yield {
        type: "response.completed",
        response: { id, output: [], usage: testUsage() },
      };
    };

    const first = await provider.chat(providerMessages(), {
      model: "gpt-5.5",
      responsesLite: false,
    });
    assert.equal(first.response_id, "response-1");

    // Mutating transport-owned objects after completion must not corrupt the
    // immutable local history used by later branches.
    firstOutput[0].encrypted_content = "mutated-after-commit";
    const firstWireInput = payloads[0].input as Record<string, unknown>[];
    ((firstWireInput[0].content as Record<string, unknown>[])[0]).text = "mutated-input";

    const branch = (output: string) => provider.chat([
      { role: MessageRole.SYSTEM, content: "You are helpful." },
      {
        role: MessageRole.TOOL,
        content: output,
        tool_call_id: "call-1",
      },
    ], {
      model: "gpt-5.5",
      responsesLite: false,
      previousResponseId: "response-1",
    });
    const [left, right] = await Promise.all([
      branch('{"result":"left"}'),
      branch('{"result":"right"}'),
    ]);
    assert.notEqual(left.response_id, right.response_id);

    const branchInputs = payloads.slice(1).map(
      (payload) => payload.input as Record<string, unknown>[],
    );
    for (const input of branchInputs) {
      assert.deepEqual(input.slice(0, 3), [
        {
          type: "message",
          role: "user",
          content: [{ type: "input_text", text: "Hello" }],
        },
        {
          type: "reasoning",
          id: "reasoning-1",
          encrypted_content: "encrypted-original",
          summary: [],
        },
        {
          type: "function_call",
          id: "function-1",
          call_id: "call-1",
          name: "lookup",
          arguments: '{"query":"one"}',
        },
      ]);
      assert.equal(Object.hasOwn(payloads[branchInputs.indexOf(input) + 1], "previous_response_id"), false);
    }
    assert.deepEqual(
      branchInputs.map((input) => (input[3] as Record<string, unknown>).output).sort(),
      ['{"result":"left"}', '{"result":"right"}'],
    );
  });

  it("adds exactly one current Lite developer prefix while replaying semantic history", async () => {
    const provider = testProvider({ model: "gpt-5.6-sol" });
    const payloads: Record<string, unknown>[] = [];
    let responseNumber = 0;
    (provider as unknown as {
      postSSE(
        path: string,
        payload: Record<string, unknown>,
      ): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* (_path, payload) {
      payloads.push(payload);
      const id = `lite-response-${++responseNumber}`;
      const output = [{
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: id }],
      }];
      yield { type: "response.output_item.done", item: output[0] };
      yield { type: "response.completed", response: { id, output: [], usage: testUsage() } };
    };

    let firstResponseId: string | undefined;
    for await (const event of provider.chatStream(providerMessages(), {
      model: "gpt-5.6-sol",
      responsesLite: true,
    })) {
      if (event.type === "finish" && typeof event.response_id === "string") {
        firstResponseId = event.response_id;
      }
    }
    assert.equal(firstResponseId, "lite-response-1");
    await provider.chat([
      { role: MessageRole.SYSTEM, content: "You are helpful." },
      { role: MessageRole.USER, content: "Second" },
    ], {
      model: "gpt-5.6-sol",
      responsesLite: true,
      previousResponseId: firstResponseId,
    });

    const input = payloads[1].input as Record<string, unknown>[];
    assert.equal(input.filter((item) => item.type === "additional_tools").length, 1);
    assert.equal(input.filter((item) => item.role === "developer").length, 2);
    assert.deepEqual(input.slice(2), [
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "Hello" }],
      },
      {
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "lite-response-1" }],
      },
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "Second" }],
      },
    ]);
    assert.equal(Object.hasOwn(payloads[1], "previous_response_id"), false);
  });

  it("resolves known history for compact and fails unknown IDs before transport", async () => {
    const provider = testProvider();
    let postJsonCalls = 0;
    let compactPayload: Record<string, unknown> | undefined;
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      const output = [{
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "first answer" }],
      }];
      yield { type: "response.output_item.done", item: output[0] };
      yield {
        type: "response.completed",
        response: { id: "compact-parent", output: [], usage: testUsage() },
      };
    };
    (provider as unknown as {
      postJSON(
        path: string,
        payload: Record<string, unknown>,
      ): Promise<Record<string, unknown>>;
    }).postJSON = async (_path, payload) => {
      postJsonCalls += 1;
      compactPayload = payload;
      return { output: [] };
    };

    await provider.chat(providerMessages(), {
      model: "gpt-5.5",
      responsesLite: false,
    });
    await provider.compactMessages([
      { role: MessageRole.SYSTEM, content: "Compact instructions" },
      { role: MessageRole.USER, content: "Compact this" },
    ], {
      model: "gpt-5.5",
      responsesLite: false,
      previousResponseId: "compact-parent",
    });
    assert.deepEqual(compactPayload?.input, [
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "Hello" }],
      },
      {
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "first answer" }],
      },
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "Compact this" }],
      },
    ]);
    assert.equal(Object.hasOwn(compactPayload ?? {}, "previous_response_id"), false);

    const credentialSentinel = "access_token=UNKNOWN_RESPONSE_SENTINEL";
    await assert.rejects(
      provider.compactMessages(providerMessages(), {
        model: "gpt-5.5",
        previousResponseId: credentialSentinel,
      }),
      (error) => error instanceof ChatGPTOAuthError
        && !error.message.includes(credentialSentinel),
    );
    assert.equal(postJsonCalls, 1);
  });

  it("keeps 256 chains and evicts the least-recently-used response", async () => {
    const provider = testProvider();
    let transportCalls = 0;
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      const id = `bounded-response-${++transportCalls}`;
      yield {
        type: "response.output_item.done",
        item: {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "" }],
        },
      };
      yield {
        type: "response.completed",
        response: { id, output: [], usage: testUsage() },
      };
    };

    for (let i = 0; i < 256; i += 1) {
      await provider.chat(providerMessages(), {
        model: "gpt-5.5",
        responsesLite: false,
      });
    }
    await provider.chat(providerMessages(), {
      model: "gpt-5.5",
      responsesLite: false,
      previousResponseId: "bounded-response-1",
    });
    await provider.chat(providerMessages(), {
      model: "gpt-5.5",
      responsesLite: false,
      previousResponseId: "bounded-response-1",
    });
    await assert.rejects(
      provider.chat(providerMessages(), {
        model: "gpt-5.5",
        responsesLite: false,
        previousResponseId: "bounded-response-2",
      }),
      (error) => error instanceof ChatGPTOAuthError,
    );
    assert.equal(transportCalls, 258);
  });

  it("does not commit a malformed output_item.done event", async () => {
    const provider = testProvider();
    let transportCalls = 0;
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      transportCalls += 1;
      yield {
        type: "response.output_item.done",
        item: [],
      };
      yield { type: "response.completed", response: { id: "malformed-response" } };
    };

    await assert.rejects(
      provider.chat(providerMessages(), { model: "gpt-5.5", responsesLite: false }),
      (error) => error instanceof ChatGPTOAuthError,
    );
    await assert.rejects(
      provider.chat(providerMessages(), {
        model: "gpt-5.5",
        responsesLite: false,
        previousResponseId: "malformed-response",
      }),
      (error) => error instanceof ChatGPTOAuthError,
    );
    assert.equal(transportCalls, 1);
  });

  it("never resolves previous_response_id history across OAuth accounts", async () => {
    const provider = testProvider();
    let transportCalls = 0;
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      transportCalls += 1;
      const output = [{
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "account A answer" }],
      }];
      yield { type: "response.output_item.done", item: output[0] };
      yield {
        type: "response.completed",
        response: { id: "account-a-response", output: [], usage: testUsage() },
      };
    };

    const accountA = preparedForTest();
    await provider.chat(providerMessages(), {
      model: "gpt-5.5",
      responsesLite: false,
      preparedModel: accountA,
    });

    const accountB = Object.freeze({
      ...accountA,
      accountId: "account-b",
      snapshot: Object.freeze({
        ...accountA.snapshot,
        key: modelCatalogCacheKey("account-b", "https://example.test", "0.153.3"),
      }),
    });
    await assert.rejects(
      provider.chat(providerMessages(), {
        model: "gpt-5.5",
        responsesLite: false,
        previousResponseId: "account-a-response",
        preparedModel: accountB,
      }),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError,
    );
    assert.equal(transportCalls, 1);
  });

  it("uses collision-free account and response identities for local history", async () => {
    const provider = testProvider();
    let transportCalls = 0;
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      transportCalls += 1;
      yield {
        type: "response.output_item.done",
        item: {
          type: "message",
          role: "assistant",
          content: [{ type: "output_text", text: "account one" }],
        },
      };
      yield {
        type: "response.completed",
        response: { id: "response-c", output: [], usage: testUsage() },
      };
    };
    const base = preparedForTest();
    const forAccount = (accountId: string): PreparedModel => Object.freeze({
      ...base,
      accountId,
      snapshot: Object.freeze({
        ...base.snapshot,
        key: modelCatalogCacheKey(accountId, "https://example.test", "0.153.3"),
      }),
    });

    await provider.chat(providerMessages(), {
      model: "gpt-5.5",
      responsesLite: false,
      preparedModel: forAccount("account-a\0response-b"),
    });
    await assert.rejects(
      provider.chat(providerMessages(), {
        model: "gpt-5.5",
        responsesLite: false,
        previousResponseId: "response-b\0response-c",
        preparedModel: forAccount("account-a"),
      }),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError,
    );
    assert.equal(transportCalls, 1);
  });
});

describe("Codex CLI request headers", () => {
  it("formats official originator and the pinned User-Agent version", () => {
    const headers = codexCliHeadersForVersion();

    assert.equal(headers.originator, "codex_cli_rs");
    assert.match(headers["User-Agent"], /^codex_cli_rs\/0\.153\.3 \(.+\) codex-as-api\/0\.7\.0$/);
  });

  it("rejects invalid and arbitrary compatibility versions", () => {
    for (const version of ["not-a-version", "", "   ", " 0.153.3", "0.153.3 "]) {
      assert.throws(
        () => codexCliHeadersForVersion(version),
        /semantic version/,
      );
    }
    assert.throws(
      () => codexCliHeadersForVersion("1.2.3"),
      ChatGPTOAuthInvalidRequestError,
    );
  });

  it("returns the pinned upstream contract version", () => {
    assert.equal(resolveCodexCliVersion(), "0.153.3");
  });

  it("adds Codex CLI headers to ChatGPT OAuth requests", () => {
    const provider = testProvider({ authJsonPath: writeAuthFile() });
    const headers = (provider as unknown as { getHeaders(): Record<string, string> }).getHeaders();

    assert.equal(headers.originator, "codex_cli_rs");
    assert.equal(headers["User-Agent"].startsWith("codex_cli_rs/0.153.3 "), true);
    assert.equal(headers.Authorization.startsWith("Bearer "), true);
  });
});

describe("decodeSSEBlock", () => {
  it("parses data lines", () => {
    const lines = ['data: {"type":"test","value":1}'];
    const result = decodeSSEBlock(lines);
    assert.deepEqual(result, { type: "test", value: 1 });
  });

  it("returns null for [DONE]", () => {
    assert.equal(decodeSSEBlock(["data: [DONE]"]), null);
  });

  it("returns null for no data lines", () => {
    assert.equal(decodeSSEBlock(["event: ping"]), null);
  });

  it("joins multiple data lines", () => {
    const lines = ['data: {"type":"test",', 'data: "a":"b"}'];
    const result = decodeSSEBlock(lines);
    assert.deepEqual(result, { type: "test", a: "b" });
  });

  it("rejects scalar and array JSON SSE events", () => {
    for (const data of ["42", "[]"]) {
      assert.throws(
        () => decodeSSEBlock([`data: ${data}`]),
        (error) => error instanceof ChatGPTOAuthError,
      );
    }
  });
});

describe("splitInstructionsAndInput", () => {
  it("separates system messages as instructions", () => {
    const messages: Message[] = [
      {
        role: MessageRole.SYSTEM,
        content: "You are helpful.",
      },
      { role: MessageRole.USER, content: "Hello" },
    ];
    const [instructions, items] =
      splitInstructionsAndInput(messages);
    assert.equal(instructions, "You are helpful.");
    assert.equal(items.length, 1);
    assert.equal(items[0].role, "user");
  });

  it("combines multiple system messages", () => {
    const messages: Message[] = [
      { role: MessageRole.SYSTEM, content: "Rule 1" },
      { role: MessageRole.SYSTEM, content: "Rule 2" },
      { role: MessageRole.USER, content: "Hi" },
    ];
    const [instructions] =
      splitInstructionsAndInput(messages);
    assert.equal(instructions, "Rule 1\n\nRule 2");
  });

  it("keeps compaction marker as input", () => {
    const compacted =
      REMOTE_COMPACTION_MARKER +
      '\n[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]';
    const messages: Message[] = [
      { role: MessageRole.SYSTEM, content: compacted },
    ];
    const [instructions, items] =
      splitInstructionsAndInput(messages);
    assert.equal(instructions, "");
    assert.equal(items.length, 1);
  });
});

describe("messagesToResponseItems", () => {
  it("converts user message", () => {
    const messages: Message[] = [
      { role: MessageRole.USER, content: "Hello" },
    ];
    const items = messagesToResponseItems(messages);
    assert.equal(items.length, 1);
    assert.equal(items[0].type, "message");
    assert.equal(items[0].role, "user");
    const content = items[0].content as Record<
      string,
      unknown
    >[];
    assert.equal(content[0].type, "input_text");
    assert.equal(content[0].text, "Hello");
  });

  it("converts assistant message", () => {
    const messages: Message[] = [
      { role: MessageRole.ASSISTANT, content: "Hi there" },
    ];
    const items = messagesToResponseItems(messages);
    const content = items[0].content as Record<
      string,
      unknown
    >[];
    assert.equal(content[0].type, "output_text");
  });

  it("rejects prompt cache breakpoints in structured content", () => {
    assert.throws(() => messagesToResponseItems([{
      role: MessageRole.USER,
      content: "cache me",
      structured_content: [
        {
          type: "text",
          text: "cache me",
          prompt_cache_breakpoint: { mode: "explicit" },
        },
        {
          type: "image_url",
          image_url: "data:image/png;base64,AAAA",
          detail: "original",
          prompt_cache_breakpoint: { mode: "explicit" },
        },
      ],
    }]), (error) => error instanceof ChatGPTOAuthError);
  });

  it("omits a null prompt cache breakpoint from structured content", () => {
    const items = messagesToResponseItems([{
      role: MessageRole.USER,
      content: "hello",
      structured_content: [{
        type: "text",
        text: "hello",
        prompt_cache_breakpoint: null,
      }] as unknown as NonNullable<Message["structured_content"]>,
    }]);

    assert.deepEqual(items[0].content, [
      { type: "input_text", text: "hello" },
    ]);
  });

  it("rejects a system-message prompt cache breakpoint", () => {
    assert.throws(() => splitInstructionsAndInput([{
      role: MessageRole.SYSTEM,
      content: "instructions",
      structured_content: [{
        type: "text",
        text: "instructions",
        prompt_cache_breakpoint: { mode: "explicit" },
      }],
    }]), (error) => error instanceof ChatGPTOAuthError);
  });

  it("rejects an assistant-message prompt cache breakpoint", () => {
    assert.throws(() => messagesToResponseItems([{
      role: MessageRole.ASSISTANT,
      content: "prior answer",
      structured_content: [{
        type: "text",
        text: "prior answer",
        prompt_cache_breakpoint: { mode: "explicit" },
      }],
    }]), (error) => error instanceof ChatGPTOAuthError);
  });

  it("converts tool message", () => {
    const messages: Message[] = [
      {
        role: MessageRole.TOOL,
        content: '{"result": 42}',
        tool_call_id: "call-1",
      },
    ];
    const items = messagesToResponseItems(messages);
    assert.equal(items[0].type, "function_call_output");
    assert.equal(items[0].call_id, "call-1");
    assert.equal(items[0].output, '{"result": 42}');
  });

  it("converts assistant with tool calls", () => {
    const tc: ToolCall = {
      id: "tc-1",
      name: "get_weather",
      arguments: '{"city":"Seoul"}',
    };
    const messages: Message[] = [
      {
        role: MessageRole.ASSISTANT,
        content: "Let me check",
        tool_calls: [tc],
      },
    ];
    const items = messagesToResponseItems(messages);
    assert.equal(items.length, 2);
    assert.equal(items[0].type, "message");
    assert.equal(items[1].type, "function_call");
    assert.equal(items[1].name, "get_weather");
    assert.equal(items[1].call_id, "tc-1");
  });

  it("installs only durable replacement-history items from a compaction marker", () => {
    const inner = [
      { type: "additional_tools", role: "developer", tools: [] },
      {
        type: "message",
        role: "developer",
        content: [{ type: "input_text", text: "stale instructions" }],
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
        content: [{ type: "input_text", text: "hi" }],
      },
      { type: "compaction_summary", encrypted_content: "legacy" },
      { type: "context_compaction" },
    ];
    const messages: Message[] = [
      {
        role: MessageRole.SYSTEM,
        content:
          REMOTE_COMPACTION_MARKER +
          "\n" +
          JSON.stringify(inner),
      },
    ];
    const items = messagesToResponseItems(messages);
    assert.deepEqual(items.map((item) => [item.type, item.role]), [
      ["message", "assistant"],
      ["agent_message", undefined],
      ["message", "user"],
      ["compaction_summary", undefined],
      ["context_compaction", undefined],
    ]);
  });

  it("rejects array entries inside a compaction marker", () => {
    assert.throws(
      () => messagesToResponseItems([{
        role: MessageRole.SYSTEM,
        content: `${REMOTE_COMPACTION_MARKER}\n[[]]`,
      }]),
      /remote compaction marker item 0 must be an object/,
    );
  });

  it("rejects malformed message entries inside a compaction marker", () => {
    const malformedMessages = [
      { type: "message", role: "user" },
      { type: "message", role: "user", content: "text" },
      { type: "message", role: "user", content: [[]] },
      { type: "message", role: "user", content: [null] },
      { type: "message", content: [] },
      { type: "message", role: "user", content: [{}] },
      { type: "message", role: "user", content: [{ type: "input_text", text: 42 }] },
      { type: "message", role: "assistant", content: [{ type: "output_text" }] },
      { type: "message", role: "user", content: [{ type: "input_image", image_url: 42 }] },
      {
        type: "message",
        role: "user",
        content: [{ type: "input_image", image_url: "data:image/png;base64,AA", detail: "bogus" }],
      },
      { type: "message", role: "user", content: [{ type: "unknown", text: "bad" }] },
    ];
    for (const malformed of malformedMessages) {
      assert.throws(
        () => messagesToResponseItems([{
          role: MessageRole.SYSTEM,
          content: `${REMOTE_COMPACTION_MARKER}\n${JSON.stringify([malformed])}`,
        }]),
        (error) => error instanceof ChatGPTOAuthError,
      );
    }
  });

  it("rejects malformed retained non-message entries inside a compaction marker", () => {
    const malformedItems = [
      {},
      { type: "agent_message", recipient: "user", content: [] },
      { type: "agent_message", author: "agent", recipient: "user", content: "text" },
      {
        type: "agent_message",
        author: "agent",
        recipient: "user",
        content: [{ type: "input_text", text: 42 }],
      },
      { type: "compaction" },
      { type: "compaction_summary", encrypted_content: 42 },
      { type: "context_compaction", encrypted_content: 42 },
    ];
    for (const malformed of malformedItems) {
      assert.throws(
        () => messagesToResponseItems([{
          role: MessageRole.SYSTEM,
          content: `${REMOTE_COMPACTION_MARKER}\n${JSON.stringify([malformed])}`,
        }]),
        (error) => error instanceof ChatGPTOAuthError,
      );
    }
  });

  it("validates common ResponseItem fields on every accepted compact variant", () => {
    const valid = [
      {
        type: "message",
        id: "",
        role: "assistant",
        content: [],
        phase: "final_answer",
        internal_chat_message_metadata_passthrough: {
          turn_id: null,
          create_time: 1.5,
          content_item_kinds: "default-on-error",
        },
        future: true,
      },
      { type: "agent_message", id: null, author: "agent", recipient: "parent", content: [] },
      { type: "compaction", id: "", encrypted_content: "opaque" },
      { type: "context_compaction", id: null },
    ];
    const marker = `${REMOTE_COMPACTION_MARKER}\n${JSON.stringify(valid)}`;
    assert.deepEqual(
      messagesToResponseItems([{ role: MessageRole.SYSTEM, content: marker }]),
      valid,
    );

    for (const item of [
      { type: "message", role: "assistant", content: [], phase: "future" },
      { type: "agent_message", id: 42, author: "agent", recipient: "parent", content: [] },
      { type: "additional_tools", role: "developer", tools: [], id: 42 },
      { type: "compaction", encrypted_content: "opaque", id: 42 },
      { type: "context_compaction", internal_chat_message_metadata_passthrough: "bad" },
    ]) {
      assert.throws(
        () => messagesToResponseItems([{
          role: MessageRole.SYSTEM,
          content: `${REMOTE_COMPACTION_MARKER}\n${JSON.stringify([item])}`,
        }]),
        (error) => error instanceof ChatGPTOAuthInvalidRequestError,
      );
    }
  });

  it("rejects unknown compacted item types and message roles instead of dropping them", () => {
    for (const item of [
      { type: "future_compaction_item", value: true },
      { type: "message", role: "future", content: [] },
      { type: "message", role: "system", content: [] },
      { type: "function_call", call_id: "call-1", name: "lookup", arguments: 42 },
    ]) {
      assert.throws(
        () => messagesToResponseItems([{
          role: MessageRole.SYSTEM,
          content: `${REMOTE_COMPACTION_MARKER}\n${JSON.stringify([item])}`,
        }]),
        (error) => error instanceof ChatGPTOAuthInvalidRequestError,
      );
    }
  });

  it("accepts null compact option fields and supported image detail", () => {
    const retained = [
      {
        type: "message",
        role: "user",
        content: [{
          type: "input_image",
          image_url: "data:image/png;base64,AA",
          detail: "original",
        }],
      },
      { type: "context_compaction", encrypted_content: null },
    ];
    const items = messagesToResponseItems([{
      role: MessageRole.SYSTEM,
      content: `${REMOTE_COMPACTION_MARKER}\n${JSON.stringify(retained)}`,
    }]);
    assert.deepEqual(items, retained);
  });
});

describe("messageItem", () => {
  it("creates user input_text item", () => {
    const item = messageItem("user", "hello");
    assert.equal(item.type, "message");
    assert.equal(item.role, "user");
    const content = item.content as Record<string, unknown>[];
    assert.equal(content[0].type, "input_text");
    assert.equal(content[0].text, "hello");
  });

  it("creates assistant output_text item", () => {
    const item = messageItem("assistant", "response");
    const content = item.content as Record<string, unknown>[];
    assert.equal(content[0].type, "output_text");
  });

  it("handles empty content", () => {
    const item = messageItem("user", "");
    const content = item.content as Record<string, unknown>[];
    assert.equal(content[0].text, "");
  });
});

describe("toolSchemaToResponseDict", () => {
  it("converts tool schema", () => {
    const tool: ToolSchema = {
      name: "get_weather",
      description: "Get weather",
      parameters: {
        type: "object",
        properties: { city: { type: "string" } },
      },
    };
    const result = toolSchemaToResponseDict(tool);
    assert.equal(result.type, "function");
    assert.equal(result.name, "get_weather");
    assert.equal(result.strict, false);

    const strictResult = toolSchemaToResponseDict({ ...tool, strict: true });
    assert.equal(strictResult.strict, true);

    const noDescription = toolSchemaToResponseDict({
      name: "no_description",
      parameters: { type: "object" },
      strict: false,
    });
    assert.equal(Object.hasOwn(noDescription, "description"), false);
    assert.equal(noDescription.strict, false);

    const emptyDescription = toolSchemaToResponseDict({
      ...tool,
      description: "",
    });
    assert.equal(emptyDescription.description, "");
  });

  it("converts internal web_search schema to hosted tool", () => {
    const tool: ToolSchema = {
      name: "web_search",
      description: "Web search",
      parameters: {
        __codex_as_api_tool_type: "web_search",
        openai_tool: {
          type: "web_search",
          external_web_access: true,
          filters: { allowed_domains: ["example.com"] },
        },
      },
    };
    assert.deepEqual(toolSchemaToResponseDict(tool), {
      type: "web_search",
      external_web_access: true,
      filters: { allowed_domains: ["example.com"] },
    });
  });
});

describe("webSearchEventFromResponseItem", () => {
  it("extracts sources from web_search_call action", () => {
    const result = webSearchEventFromResponseItem({
      type: "web_search_call",
      id: "ws_1",
      action: {
        type: "search",
        query: "hello",
        queries: ["hello"],
        sources: [
          { url: "https://example.com", title: "Example", page_age: "today" },
          { url: "https://example.com", title: "Duplicate", page_age: "" },
        ],
      },
    });
    assert.ok(result);
    assert.equal(result.id, "ws_1");
    assert.deepEqual(result.input, { query: "hello" });
    assert.deepEqual(result.content, [
      {
        type: "web_search_result",
        url: "https://example.com",
        title: "Example",
        page_age: "today",
      },
      {
        type: "web_search_result",
        url: "https://example.com",
        title: "Duplicate",
        page_age: "",
      },
    ]);
  });

  it("validates query types and source metadata even when query is present", () => {
    for (const action of [
      { query: "hello", sources: [] },
      { type: "search", query: "hello", queries: "hello", sources: [] },
      { type: "search", query: "", queries: ["first", "second"], sources: [] },
      { type: "search", query: "first", queries: ["second"], sources: [] },
      { type: "search", sources: [] },
      { type: "search", queries: [], sources: [] },
      { type: "open_page", url: "https://example.com", sources: [] },
      { type: "find_in_page", pattern: "needle", sources: [] },
      { type: "future", sources: [] },
      { type: "search", query: "hello", sources: [{ url: "https://example.com", page_age: 1 }] },
    ]) {
      assert.throws(
        () => webSearchEventFromResponseItem({
          type: "web_search_call",
          id: "ws_1",
          action,
        }),
        ChatGPTOAuthProtocolError,
      );
    }
  });

  it("rejects web_search_call output without requested action sources", () => {
    for (const action of [
      { type: "search", queries: ["q"] },
      { type: "search", queries: ["q"], sources: null },
    ]) {
      assert.throws(
        () => webSearchEventFromResponseItem({
          type: "web_search_call",
          id: "ws_1",
          action,
        }),
        ChatGPTOAuthProtocolError,
      );
    }
  });

  it("matches pinned nullable and empty web-search detail semantics", () => {
    for (const [action, expected] of [
      [{ query: "direct", queries: null, sources: [] }, "direct"],
      [{ query: null, queries: ["fallback"], sources: [] }, "fallback"],
      [{ query: "", sources: [] }, ""],
      [{ query: "same", queries: ["same"], sources: [] }, "same"],
    ] as const) {
      const result = webSearchEventFromResponseItem({
        type: "web_search_call",
        id: "",
        action: { type: "search", ...action },
      });
      assert.ok(result);
      assert.equal(result.id, "");
      assert.deepEqual(result.input, { query: expected });
    }

    for (const id of [undefined, null]) {
      assert.throws(() => webSearchEventFromResponseItem({
        type: "web_search_call",
        id,
        action: { type: "search", query: "q", sources: [] },
      }), ChatGPTOAuthProtocolError);
    }
  });
});

describe("setReasoningPayload", () => {
  it("sets valid effort", () => {
    const payload: Record<string, unknown> = {};
    setReasoningPayload(payload, "high", undefined, testCapability());
    assert.deepEqual(payload.reasoning, { effort: "high" });
    assert.deepEqual(payload.include, ["reasoning.encrypted_content"]);
  });

  it("rejects unsupported effort values", () => {
    const payload: Record<string, unknown> = {};
    const effortSentinel = "access_token=EFFORT_SENTINEL";
    const modelSentinel = "access_token=MODEL_SENTINEL";
    assert.throws(
      () => setReasoningPayload(payload, effortSentinel, undefined, testCapability(modelSentinel)),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError
        && !error.message.includes(effortSentinel)
        && !error.message.includes(modelSentinel),
    );
  });

  it("does nothing for undefined", () => {
    const payload: Record<string, unknown> = {};
    setReasoningPayload(payload, undefined);
    assert.equal(payload.reasoning, undefined);
  });

  it("throws on empty or whitespace-padded effort values", () => {
    for (const effort of ["", " low", "low "]) {
      const payload: Record<string, unknown> = {};
      assert.throws(
        () => setReasoningPayload(payload, effort, undefined, testCapability()),
        (error) => error instanceof ChatGPTOAuthInvalidRequestError,
      );
    }
    for (const effort of [" low", "low "]) {
      const payload: Record<string, unknown> = {};
      assert.throws(
        () => setReasoningPayload(payload, undefined, { effort }),
        (error) => error instanceof ChatGPTOAuthInvalidRequestError,
      );
    }
  });

  it("rejects a malformed existing reasoning payload instead of replacing it", () => {
    for (const reasoning of [null, undefined, "medium", [], 1]) {
      const payload: Record<string, unknown> = { reasoning };
      assert.throws(
        () => setReasoningPayload(payload, "high", undefined, testCapability()),
        (error) => error instanceof ChatGPTOAuthInvalidRequestError,
      );
      assert.equal(payload.reasoning, reasoning);
    }
  });

  it("rejects a malformed existing include payload instead of replacing it", () => {
    for (const include of [null, undefined, "reasoning.encrypted_content", {}]) {
      const payload: Record<string, unknown> = { include };
      assert.throws(
        () => setReasoningPayload(payload, "high", undefined, testCapability()),
        (error) => error instanceof ChatGPTOAuthInvalidRequestError,
      );
      assert.equal(payload.include, include);
      assert.equal(Object.hasOwn(payload, "reasoning"), false);
    }
  });

  it("maps ultra using the live model capability", () => {
    const ultraPayload: Record<string, unknown> = {};
    setReasoningPayload(ultraPayload, "ultra", undefined, testCapability());
    assert.deepEqual(ultraPayload.reasoning, { effort: "max" });
  });

  it("maps persistent reasoning to the private disabled wire value", () => {
    const capability = Object.freeze({
      ...testCapability(),
      defaultReasoningEffort: "persistent",
      supportedReasoningEfforts: Object.freeze([
        Object.freeze({ effort: "persistent", description: "persistent" }),
      ]),
    });
    const explicitPayload: Record<string, unknown> = {};
    setReasoningPayload(explicitPayload, "persistent", undefined, capability);
    assert.deepEqual(explicitPayload.reasoning, { effort: "disabled" });

    const defaultPayload: Record<string, unknown> = {};
    setReasoningPayload(defaultPayload, undefined, undefined, capability);
    assert.equal(Object.hasOwn(defaultPayload, "reasoning"), false);
  });

  it("applies live reasoning summary controls to the final private payload", () => {
    const responsesPayload = (testProvider() as unknown as {
      responsesPayload(
        messages: Message[],
        options: Record<string, unknown>,
        prepared: PreparedModel,
      ): Record<string, unknown>;
    }).responsesPayload;
    for (const [supported, summary, expected] of [
      [true, "detailed", "detailed"],
      [true, "none", undefined],
      [false, "concise", undefined],
    ] as const) {
      const base = preparedForTest("gpt-5.5");
      const capability = Object.freeze({
        ...base.capability,
        defaultReasoningEffort: undefined,
        supportedReasoningEfforts: Object.freeze([]),
        supportsReasoningSummaryParameter: supported,
        defaultReasoningSummary: summary,
      });
      const payload = responsesPayload(
        providerMessages(),
        { model: "gpt-5.5", responsesLite: false },
        { ...base, capability },
      );
      assert.equal((payload.reasoning as Record<string, unknown> | undefined)?.summary, expected);
      assert.equal(
        (payload.include as unknown[]).includes("reasoning.encrypted_content"),
        true,
      );
    }
  });

  it("does not forward ultra from an invalid multi-agent override", () => {
    const base = testCapability();
    const capability = Object.freeze({
      ...base,
      supportedReasoningEfforts: Object.freeze([
        { effort: "xhigh", description: "xhigh" },
        { effort: "ultra", description: "ultra" },
      ]),
      multiAgentReasoningEffort: "ultra",
    });
    const payload: Record<string, unknown> = {};

    setReasoningPayload(payload, "ultra", undefined, capability);

    assert.deepEqual(payload.reasoning, { effort: "xhigh" });
  });

  it("accepts all valid values", () => {
    for (const effort of [
      "none",
      "minimal",
      "low",
      "medium",
      "high",
      "xhigh",
      "max",
    ]) {
      const payload: Record<string, unknown> = {};
      setReasoningPayload(payload, effort, undefined, testCapability());
      assert.deepEqual(payload.reasoning, { effort });
      assert.deepEqual(payload.include, ["reasoning.encrypted_content"]);
    }
  });

  it("rejects reasoning mode because the transport has no wire field", () => {
    const payload: Record<string, unknown> = {
      model: "gpt-5.6-sol",
      reasoning: { summary: "auto" },
    };
    assert.throws(() => setReasoningPayload(
      payload,
      "medium",
      { mode: "standard", context: "all_turns" },
      testCapability("gpt-5.6-sol"),
    ), (error) => error instanceof ChatGPTOAuthInvalidRequestError);
  });
});

describe("toolCallFromResponseItem", () => {
  it("parses function_call item", () => {
    const item = {
      type: "function_call",
      name: "get_weather",
      call_id: "call-1",
      arguments: '{"city":"Seoul"}',
    };
    const result = toolCallFromResponseItem(item);
    assert.ok(result);
    assert.equal(result.name, "get_weather");
    assert.equal(result.id, "call-1");
    assert.equal(result.arguments, '{"city":"Seoul"}');
  });

  it("rejects custom_tool_call items", () => {
    const item = {
      type: "custom_tool_call",
      name: "my_tool",
      call_id: "ct-1",
      input: '{"x":1}',
    };
    assert.throws(
      () => toolCallFromResponseItem(item),
      (error) => error instanceof ChatGPTOAuthProtocolError
        && error.message.includes("custom_tool_call"),
    );
  });

  it("returns null for non-tool items", () => {
    assert.equal(
      toolCallFromResponseItem({ type: "message" }),
      null,
    );
  });

  it("rejects a missing name", () => {
    assert.throws(
      () => toolCallFromResponseItem({
        type: "function_call",
        name: "",
      }),
      (error) => error instanceof ChatGPTOAuthProtocolError,
    );
  });

  it("rejects malformed canonical argument fields and missing call IDs", () => {
    const item = {
      type: "function_call",
      name: "tool",
      call_id: "c1",
      arguments: { key: "value" },
    };
    assert.throws(
      () => toolCallFromResponseItem(item),
      ChatGPTOAuthProtocolError,
    );
    assert.throws(
      () => toolCallFromResponseItem({
        type: "function_call",
        name: "tool",
        id: "item-id-is-not-a-call-id",
        arguments: "{}",
      }),
      ChatGPTOAuthProtocolError,
    );
  });

  it("ignores an additive input field on function calls", () => {
    assert.deepEqual(
      toolCallFromResponseItem({
        type: "function_call",
        name: "tool",
        call_id: "c1",
        arguments: '{"canonical":true}',
        input: "future metadata",
      }),
      { id: "c1", name: "tool", arguments: '{"canonical":true}' },
    );
  });

  it("preserves raw function-call arguments without parsing", () => {
    for (const argumentsValue of ["not-json", "[]", "  {\"b\":2, \"a\":1}  "]) {
      assert.equal(
        toolCallFromResponseItem({
          type: "function_call",
          name: "tool",
          call_id: "c1",
          arguments: argumentsValue,
        })?.arguments,
        argumentsValue,
      );
    }
  });
});

describe("textFromResponseItems", () => {
  it("extracts from message items", () => {
    const items = [
      {
        type: "message",
        role: "assistant",
        content: [{ type: "output_text", text: "content" }],
      },
    ];
    assert.equal(textFromResponseItems(items), "content");
  });

  it("skips non-text items", () => {
    const items = [
      { type: "function_call", name: "tool" },
    ];
    assert.equal(textFromResponseItems(items), "");
  });

  it("rejects stale top-level text aliases", () => {
    for (const type of ["output_text", "text"]) {
      assert.throws(
        () => textFromResponseItems([{ type, text: "simple" }]),
        ChatGPTOAuthProtocolError,
      );
    }
  });

  it("returns empty for no text", () => {
    assert.equal(
      textFromResponseItems([{ type: "image_generation_call" }]),
      "",
    );
  });

  it("rejects unsupported output items and message content parts", () => {
    assert.throws(
      () => textFromResponseItems([{ type: "future_item" }]),
      ChatGPTOAuthProtocolError,
    );
    assert.throws(
      () => textFromResponseItems([{
        type: "message",
        role: "assistant",
        content: [
          { type: "output_text", text: "valid" },
          { type: "future_part", value: true },
        ],
      }]),
      ChatGPTOAuthProtocolError,
    );
  });

  it("rejects string content parts in message", () => {
    const items = [
      { type: "message", content: ["hello", " world"] },
    ];
    assert.throws(
      () => textFromResponseItems(items),
      (error) => error instanceof ChatGPTOAuthProtocolError,
    );
  });

  it("rejects the stale message content text alias", () => {
    assert.throws(
      () => textFromResponseItems([{
        type: "message",
        content: [{ type: "text", text: "legacy" }],
      }]),
      ChatGPTOAuthProtocolError,
    );
  });
});

describe("validateImageContentItems", () => {
  it("validates data URLs", () => {
    const result = validateImageContentItems([
      {
        image_url: "data:image/png;base64,abc",
        detail: "original",
      },
    ]);
    assert.deepEqual(result, [{
      type: "input_image",
      image_url: "data:image/png;base64,abc",
      detail: "original",
    }]);
  });

  it("rejects non-data URLs", () => {
    assert.throws(
      () =>
        validateImageContentItems([
          { image_url: "https://example.com/img.png" },
        ]),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError,
    );
  });

  it("rejects empty image_url", () => {
    assert.throws(
      () => validateImageContentItems([{ image_url: "" }]),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError,
    );
  });

  it("rejects invalid detail and cache breakpoint values", () => {
    assert.throws(() => validateImageContentItems([{
      image_url: "data:image/png;base64,abc",
      detail: "full" as never,
    }]), (error) => error instanceof ChatGPTOAuthError);
    assert.throws(() => validateImageContentItems([{
      image_url: "data:image/png;base64,abc",
      prompt_cache_breakpoint: { mode: "implicit" } as never,
    }]), (error) => error instanceof ChatGPTOAuthError);
  });
});

describe("imageGenerationFromItem", () => {
  it("extracts image_generation_call", () => {
    const item = {
      type: "image_generation_call",
      id: "img-1",
      result: "data:image/png;base64,abc",
      status: "completed",
      revised_prompt: "a cat",
    };
    const result = imageGenerationFromItem(item);
    assert.ok(result);
    assert.equal(result.id, "img-1");
    assert.equal(
      result.result,
      "data:image/png;base64,abc",
    );
    assert.equal(result.revised_prompt, "a cat");
  });

  it("preserves optional image generation fields without fabricating them", () => {
    for (const optionalFields of [
      {},
      { id: null, revised_prompt: null },
    ]) {
      const result = imageGenerationFromItem({
        type: "image_generation_call",
        status: "completed",
        result: "data:image/png;base64,abc",
        ...optionalFields,
      });
      assert.deepEqual(result, {
        status: "completed",
        result: "data:image/png;base64,abc",
      });
    }
  });

  it("returns null for non-image items", () => {
    assert.equal(
      imageGenerationFromItem({ type: "message" }),
      null,
    );
  });

  it("preserves empty plain string fields", () => {
    assert.deepEqual(imageGenerationFromItem({
      type: "image_generation_call",
      id: "",
      status: "",
      result: "",
    }), {
      id: "",
      status: "",
      result: "",
    });
  });
});

describe("usageFromResponse", () => {
  it("parses Responses API format", () => {
    const value = {
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
      input_tokens_details: {
        cached_tokens: 20,
        cache_write_tokens: 30,
      },
    };
    const result = usageFromResponse(value);
    assert.ok(result);
    assert.equal(result.prompt_tokens, 100);
    assert.equal(result.completion_tokens, 50);
    assert.equal(result.total_tokens, 150);
    assert.equal(result.cache_write_tokens, 30);
    assert.equal(result.cached_tokens, 20);
  });

  it("rejects Chat Completions aliases on the private Responses wire", () => {
    const value = {
      prompt_tokens: 80,
      completion_tokens: 40,
      total_tokens: 120,
      prompt_tokens_details: { cached_tokens: 10 },
    };
    assert.equal(usageFromResponse(value), null);
  });

  it("accepts absent or null input token details without synthesizing cache counts", () => {
    for (const details of [undefined, null]) {
      const result = usageFromResponse({
        input_tokens: 8,
        output_tokens: 5,
        total_tokens: 13,
        ...(details === undefined ? {} : { input_tokens_details: details }),
      });
      assert.ok(result);
      assert.equal(Object.hasOwn(result, "cached_tokens"), false);
      assert.equal(Object.hasOwn(result, "cache_write_tokens"), false);
    }
  });

  it("strictly validates input token details when present", () => {
    for (const inputTokensDetails of [
      {},
      { cached_tokens: null },
      { cached_tokens: 1, cache_write_tokens: "2" },
      [],
    ]) {
      assert.equal(usageFromResponse({
        input_tokens: 8,
        output_tokens: 5,
        total_tokens: 13,
        input_tokens_details: inputTokensDetails,
      }), null);
    }
  });

  it("returns null for null input", () => {
    assert.equal(usageFromResponse(null), null);
  });

  it("returns null for missing tokens", () => {
    assert.equal(
      usageFromResponse({ input_tokens: 10 }),
      null,
    );
  });

  it("rejects missing total_tokens", () => {
    assert.equal(usageFromResponse({
      input_tokens: 10,
      output_tokens: 5,
    }), null);
  });

  it("rejects cached_input_tokens aliases", () => {
    assert.equal(usageFromResponse({
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
      cached_input_tokens: 30,
    }), null);
  });

  it("rejects cache_read_input_tokens aliases", () => {
    assert.equal(usageFromResponse({
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
      cache_read_input_tokens: 25,
    }), null);
  });

  it("rejects cache_creation_input_tokens aliases", () => {
    assert.equal(usageFromResponse({
      input_tokens: 100,
      output_tokens: 50,
      total_tokens: 150,
      cache_creation_input_tokens: 25,
    }), null);
  });

  it("rejects Responses usage mixed with Chat Completions aliases", () => {
    assert.equal(usageFromResponse({
      ...testUsage(),
      prompt_tokens: 2,
    }), null);
    assert.equal(usageFromResponse({
      ...testUsage(),
      prompt_tokens_details: { cached_tokens: 1 },
    }), null);
    assert.equal(usageFromResponse({
      ...testUsage(),
      prompt_tokens: null,
    }), null);
  });
});
