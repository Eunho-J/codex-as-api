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
import {
  ChatGPTOAuthError,
  ChatGPTOAuthInvalidRequestError,
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

describe("ChatGPTOAuthProvider payload", () => {
  it("omits max_output_tokens even when maxTokens is set", () => {
    const provider = new ChatGPTOAuthProvider({});
    const payload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; maxTokens: number }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.5", maxTokens: 1024 });

    assert.equal(Object.hasOwn(payload, "max_output_tokens"), false);
  });

  it("omits null and empty stop values from the private request", async () => {
    for (const stop of [undefined, null, "", [], [""], ["", ""]] as const) {
      const provider = new ChatGPTOAuthProvider({});
      const payloads: Record<string, unknown>[] = [];
      (provider as unknown as {
        postSSE(path: string, payload: Record<string, unknown>): AsyncGenerator<Record<string, unknown>>;
      }).postSSE = async function* (_path, payload) {
        payloads.push(structuredClone(payload));
        yield {
          type: "response.completed",
          response: { id: "resp-empty-stop", output: [], usage: {} },
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

  it("rejects a non-empty stop before the private transport starts", () => {
    const provider = new ChatGPTOAuthProvider({});
    let transportCalls = 0;
    (provider as unknown as {
      postSSE(path: string, payload: Record<string, unknown>): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      transportCalls += 1;
      yield { type: "response.completed", response: { id: "unexpected" } };
    };

    assert.throws(
      () => provider.chatStream(providerMessages(), {
        model: "gpt-5.5",
        stop: ["END"],
      }),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError,
    );
    assert.equal(transportCalls, 0);
  });

  it("rejects original image detail for models without verified support", () => {
    const provider = new ChatGPTOAuthProvider({});
    const responsesPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: Record<string, unknown>): Record<string, unknown>;
    }).responsesPayload.bind(provider);

    for (const model of ["gpt-5.2", "gpt-5.3-codex", "gpt-5.3-codex-spark", "future-model"]) {
      assert.throws(
        () => responsesPayload(providerMessagesWithImageDetail("original"), {
          model,
          responsesLite: false,
        }),
        (error) => error instanceof ChatGPTOAuthError,
      );
    }
  });

  it("keeps auto, low, and high image detail for GPT-5.2", () => {
    const provider = new ChatGPTOAuthProvider({});
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
    const provider = new ChatGPTOAuthProvider({});
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

  it("rejects an empty chat prompt cache key", () => {
    const provider = new ChatGPTOAuthProvider({});

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
    const provider = new ChatGPTOAuthProvider({});
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
    const provider = new ChatGPTOAuthProvider({});
    const payload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; reasoningEffort: string; responsesLite: boolean }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      reasoningEffort: "high",
      responsesLite: false,
    });

    assert.deepEqual(payload.reasoning, { effort: "high" });
    assert.deepEqual(payload.include, ["reasoning.encrypted_content"]);
  });

  it("forces Responses Lite payload shape", () => {
    const provider = new ChatGPTOAuthProvider({});
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

  it("uses GPT-5.6 catalog defaults and the Lite reasoning context", () => {
    const provider = new ChatGPTOAuthProvider({});
    const payload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; responsesLite: string }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.6-sol", responsesLite: "auto" });

    assert.equal(payload.model, "gpt-5.6-sol");
    assert.deepEqual(payload.reasoning, { effort: "low", context: "all_turns" });
    assert.deepEqual(payload.text, { verbosity: "low" });
    assert.equal((payload.input as Record<string, unknown>[])[0].type, "additional_tools");
  });

  it("resolves the public GPT-5.6 alias to Sol on the outbound wire", () => {
    const provider = new ChatGPTOAuthProvider({});
    const payload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: Record<string, unknown>): Record<string, unknown>;
    }).responsesPayload(providerMessages(), {
      model: "gpt-5.6",
      responsesLite: false,
    });
    assert.equal(payload.model, "gpt-5.6-sol");
    assert.deepEqual(payload.reasoning, { effort: "low" });
    assert.deepEqual(payload.text, { verbosity: "low" });
  });

  it("rejects non-auto tool choice in Responses Lite", () => {
    const provider = new ChatGPTOAuthProvider({});
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
    const provider = new ChatGPTOAuthProvider({});
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
    const provider = new ChatGPTOAuthProvider({});
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
      const provider = new ChatGPTOAuthProvider({});
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
    const provider = new ChatGPTOAuthProvider({});
    (provider as unknown as {
      postSSE(path: string, payload: Record<string, unknown>): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* (_path, payload) {
      assert.equal(Object.hasOwn(payload, "tools"), true);
      yield {
        type: "response.output_item.done",
        item: { type: "image_generation_call", result: "data:image/png;base64,AA" },
      };
      yield { type: "response.completed", response: { id: "response-image" } };
    };

    const images = await provider.generateImage("draw", {
      model: "gpt-5.6-sol",
      responsesLite: false,
    });
    assert.equal(images[0].result, "data:image/png;base64,AA");
  });

  it("uses the shared capability table and canonicalizes fast service tier", () => {
    const provider = new ChatGPTOAuthProvider({});
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

  it("parallel tool calls use the shared capability table", () => {
    const provider = new ChatGPTOAuthProvider({});
    const payload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; parallelToolCalls: boolean; responsesLite: boolean }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.5", parallelToolCalls: true, responsesLite: false });
    const sparkPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; parallelToolCalls: boolean; responsesLite: boolean }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.3-codex-spark", parallelToolCalls: true, responsesLite: false });
    const litePayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; parallelToolCalls: boolean; responsesLite: boolean }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.5", parallelToolCalls: true, responsesLite: true });

    assert.equal(payload.parallel_tool_calls, true);
    assert.equal(sparkPayload.parallel_tool_calls, false);
    assert.equal(litePayload.parallel_tool_calls, false);
  });

  it("keeps compact tools top-level in explicit classic mode", async () => {
    const provider = new ChatGPTOAuthProvider({});
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

  it("rejects an empty compact prompt cache key", async () => {
    const provider = new ChatGPTOAuthProvider({});
    await assert.rejects(provider.compactMessages(providerMessages(), {
      model: "gpt-5.6-sol",
      responsesLite: false,
      promptCacheKey: "",
    }), (error) => error instanceof ChatGPTOAuthError);
  });

  it("rejects unsupported compact service tiers through the model capability catalog", async () => {
    const provider = new ChatGPTOAuthProvider({});
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
      const provider = new ChatGPTOAuthProvider({});
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
    const provider = new ChatGPTOAuthProvider({});
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

  it("omits standard reasoning mode and rejects unsupported private request fields", () => {
    const provider = new ChatGPTOAuthProvider({});
    const responsesPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: Record<string, unknown>): Record<string, unknown>;
    }).responsesPayload.bind(provider);
    const payload = responsesPayload(providerMessages(), {
      model: "gpt-5.6-sol",
      responsesLite: false,
      reasoning: { mode: "standard", context: "current_turn" },
    });

    assert.deepEqual(payload.reasoning, {
      effort: "medium",
      context: "current_turn",
    });
    assert.equal(Object.hasOwn(payload.reasoning as object, "mode"), false);
    for (const opts of [
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
    const provider = new ChatGPTOAuthProvider({});
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
    const provider = new ChatGPTOAuthProvider({});
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
    assert.deepEqual(captured?.reasoning, { effort: "low", context: "all_turns" });
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
    const provider = new ChatGPTOAuthProvider({});
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

  it("preserves image detail and omits standard reasoning mode for classic image requests", async () => {
    const provider = new ChatGPTOAuthProvider({});
    let captured: Record<string, unknown> | undefined;
    (provider as unknown as {
      postSSE(path: string, payload: Record<string, unknown>): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* (_path, payload) {
      captured = payload;
      yield {
        type: "response.output_item.done",
        item: { type: "image_generation_call", result: "data:image/png;base64,AA" },
      };
      yield { type: "response.completed", response: { id: "response-image" } };
    };

    await provider.generateImage("draw", {
      model: "gpt-5.6-sol",
      responsesLite: false,
      referenceImages: [{
        image_url: "data:image/png;base64,AAAA",
        detail: "original",
      }],
      reasoning: { mode: "standard" },
    });
    const input = captured?.input as Record<string, unknown>[];
    const content = input[0].content as Record<string, unknown>[];
    assert.deepEqual(content[1], {
      type: "input_image",
      image_url: "data:image/png;base64,AAAA",
      detail: "original",
    });
    assert.deepEqual(captured?.reasoning, { effort: "medium" });
    assert.equal(Object.hasOwn(captured ?? {}, "safety_identifier"), false);
  });

  it("uses explicit Codex session identity and preserves metadata lifetimes", () => {
    const provider = new ChatGPTOAuthProvider({ authJsonPath: writeAuthFile() });
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

  it("requires session identity when Codex metadata is enabled", () => {
    const provider = new ChatGPTOAuthProvider({ authJsonPath: writeAuthFile() });
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

  it("prefers an explicit prompt cache key, then session identity, then omission", () => {
    const provider = new ChatGPTOAuthProvider({});
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
    const blankSession = responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      clientMetadata: { session_id: "   " },
      codexMetadata: false,
    });
    const explicitWithBlankSession = responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      promptCacheKey: "explicit-cache",
      clientMetadata: { session_id: "   " },
      codexMetadata: false,
    });

    assert.equal(explicit.prompt_cache_key, "explicit-cache");
    assert.equal(derived.prompt_cache_key, "session-cache");
    assert.equal(Object.hasOwn(absent, "prompt_cache_key"), false);
    assert.equal(Object.hasOwn(blankSession, "prompt_cache_key"), false);
    assert.equal(explicitWithBlankSession.prompt_cache_key, "explicit-cache");
  });
});

describe("Responses stream completion", () => {
  it("renders output_item.done text and tool output when completed output is empty", async () => {
    const provider = new ChatGPTOAuthProvider();
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
        response: { id: "response-completion-only", usage: {}, output: [] },
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
      arguments: { query: "one" },
    });
    assert.deepEqual(events[1], { type: "content", text: "completion text" });
    assert.equal(events[2].finish_reason, "tool_calls");
    assert.equal(events[2].response_id, "response-completion-only");
  });

  it("uses tool_calls as the terminal reason for an emitted tool call", async () => {
    const provider = new ChatGPTOAuthProvider();
    const toolCall = {
      type: "function_call",
      id: "item-1",
      call_id: "call-1",
      name: "lookup",
      arguments: '{"query":"one"}',
    };
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield { type: "response.output_item.done", item: toolCall };
      yield {
        type: "response.completed",
        response: { id: "response-1", usage: {}, output: [toolCall] },
      };
    };

    const events: Record<string, unknown>[] = [];
    for await (const event of provider.chatStream(providerMessages(), { model: "gpt-5.6-sol" })) {
      events.push(event);
    }

    const emittedTool = events.find((event) => event.type === "tool_call");
    const finish = events.find((event) => event.type === "finish");
    assert.deepEqual([emittedTool?.id, emittedTool?.arguments], ["call-1", { query: "one" }]);
    assert.equal(finish?.finish_reason, "tool_calls");
    assert.equal(finish?.response_id, "response-1");
  });

  it("fails when the upstream SSE stream ends before response.completed", async () => {
    const provider = new ChatGPTOAuthProvider();
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

  it("returns immediately after response.completed", async () => {
    const provider = new ChatGPTOAuthProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield {
        type: "response.completed",
        response: { id: "response-1", usage: {}, output: [] },
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

  it("rejects malformed response.completed payloads", async () => {
    for (const response of [undefined, null, {}, { id: "" }]) {
      const provider = new ChatGPTOAuthProvider();
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
    const provider = new ChatGPTOAuthProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield {
        type: "response.output_item.done",
        item: { type: "message", role: "assistant", content: [] },
      };
    };

    await assert.rejects(
      (provider as unknown as {
        collectResponseOutputItems(payload: Record<string, unknown>): Promise<Record<string, unknown>[]>;
      }).collectResponseOutputItems({}),
      /ended before response\.completed/,
    );
  });

  it("rejects malformed output_item.done in the image and inspect collector", async () => {
    for (const item of [null, [], "not-an-object"]) {
      const provider = new ChatGPTOAuthProvider();
      (provider as unknown as {
        postSSE(): AsyncGenerator<Record<string, unknown>>;
      }).postSSE = async function* () {
        yield { type: "response.output_item.done", item };
        yield { type: "response.completed", response: { id: "response-1" } };
      };

      await assert.rejects(
        (provider as unknown as {
          collectResponseOutputItems(payload: Record<string, unknown>): Promise<Record<string, unknown>[]>;
        }).collectResponseOutputItems({}),
        (error) => error instanceof ChatGPTOAuthError,
      );
    }
  });

  it("validates response.completed identity in the image and inspect output collector", async () => {
    const provider = new ChatGPTOAuthProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield { type: "response.completed", response: { id: "" } };
    };

    await assert.rejects(
      (provider as unknown as {
        collectResponseOutputItems(payload: Record<string, unknown>): Promise<Record<string, unknown>[]>;
      }).collectResponseOutputItems({}),
      (error) => error instanceof ChatGPTOAuthError,
    );
  });

  it("does not recover output items or read failures after response.completed", async () => {
    const provider = new ChatGPTOAuthProvider();
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      yield {
        type: "response.completed",
        response: {
          id: "response-1",
          output: [{ type: "message", role: "assistant", content: [] }],
        },
      };
      yield { type: "response.failed", error: { message: "must be ignored" } };
    };

    const output = await (provider as unknown as {
      collectResponseOutputItems(payload: Record<string, unknown>): Promise<Record<string, unknown>[]>;
    }).collectResponseOutputItems({});
    assert.deepEqual(output, []);
  });
});

describe("local previous_response_id history", () => {
  it("replays exact output_item.done history and supports concurrent branches without forwarding the ID", async () => {
    const provider = new ChatGPTOAuthProvider();
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
        response: { id, output, usage: {} },
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
    const provider = new ChatGPTOAuthProvider({ model: "gpt-5.6-sol" });
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
      yield { type: "response.completed", response: { id, output, usage: {} } };
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
    const provider = new ChatGPTOAuthProvider();
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
        response: { id: "compact-parent", output, usage: {} },
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

    await assert.rejects(
      provider.compactMessages(providerMessages(), {
        model: "gpt-5.5",
        previousResponseId: "unknown-response",
      }),
      (error) => error instanceof ChatGPTOAuthError,
    );
    assert.equal(postJsonCalls, 1);
  });

  it("keeps 256 chains and evicts the least-recently-used response", async () => {
    const provider = new ChatGPTOAuthProvider();
    let transportCalls = 0;
    (provider as unknown as {
      postSSE(): AsyncGenerator<Record<string, unknown>>;
    }).postSSE = async function* () {
      const id = `bounded-response-${++transportCalls}`;
      yield {
        type: "response.completed",
        response: { id, output: [], usage: {} },
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
    const provider = new ChatGPTOAuthProvider();
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
});

describe("Codex CLI request headers", () => {
  it("formats official originator and versioned User-Agent headers", () => {
    const headers = codexCliHeadersForVersion("1.2.3\n");

    assert.equal(headers.originator, "codex_cli_rs");
    assert.match(headers["User-Agent"], /^codex_cli_rs\/1\.2\.3 \(.+\) codex-as-api\/0\.6\.5$/);
  });

  it("rejects an invalid compatibility version", () => {
    assert.throws(
      () => codexCliHeadersForVersion("not-a-version"),
      /semantic version/,
    );
  });

  it("defaults to the pinned upstream contract version", () => {
    const previous = process.env.CODEX_AS_API_CODEX_CLI_VERSION;
    delete process.env.CODEX_AS_API_CODEX_CLI_VERSION;
    try {
      assert.equal(resolveCodexCliVersion(), "0.147.0");
    } finally {
      if (previous != null) process.env.CODEX_AS_API_CODEX_CLI_VERSION = previous;
    }
  });

  it("adds Codex CLI headers to ChatGPT OAuth requests", () => {
    const previous = process.env.CODEX_AS_API_CODEX_CLI_VERSION;
    process.env.CODEX_AS_API_CODEX_CLI_VERSION = "9.8.7";
    try {
      const provider = new ChatGPTOAuthProvider({ authJsonPath: writeAuthFile() });
      const headers = (provider as unknown as { getHeaders(): Record<string, string> }).getHeaders();

      assert.equal(headers.originator, "codex_cli_rs");
      assert.match(headers["User-Agent"], /^codex_cli_rs\/9\.8\.7 \(.+\) codex-as-api\/0\.6\.5$/);
      assert.match(headers.Authorization, /^Bearer /);
    } finally {
      if (previous == null) {
        delete process.env.CODEX_AS_API_CODEX_CLI_VERSION;
      } else {
        process.env.CODEX_AS_API_CODEX_CLI_VERSION = previous;
      }
    }
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
    const lines = ['data: {"a":', 'data: "b"}'];
    const result = decodeSSEBlock(lines);
    assert.deepEqual(result, { a: "b" });
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
      arguments: { city: "Seoul" },
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
        sources: [{ url: "https://example.com", title: "Example", page_age: "today" }],
      },
    });
    assert.ok(result);
    assert.equal(result.id, "srvtoolu_ws_1");
    assert.deepEqual(result.input, { query: "hello" });
    assert.deepEqual(result.content, [{
      type: "web_search_result",
      url: "https://example.com",
      title: "Example",
      page_age: "today",
    }]);
  });

  it("falls back to url_citation annotations", () => {
    const result = webSearchEventFromResponseItem(
      { type: "web_search_call", id: "ws_1", action: { type: "search", queries: ["q"] } },
      [
        { type: "web_search_call", id: "ws_1" },
        {
          type: "message",
          content: [{
            type: "output_text",
            text: "answer",
            annotations: [{ type: "url_citation", url: "https://a.test", title: "A" }],
          }],
        },
      ],
    );
    assert.ok(result);
    assert.deepEqual(result.content, [{
      type: "web_search_result",
      url: "https://a.test",
      title: "A",
    }]);
  });
});

describe("setReasoningPayload", () => {
  it("sets valid effort", () => {
    const payload: Record<string, unknown> = {};
    setReasoningPayload(payload, "high");
    assert.deepEqual(payload.reasoning, { effort: "high" });
    assert.deepEqual(payload.include, ["reasoning.encrypted_content"]);
  });

  it("preserves case for custom values", () => {
    const payload: Record<string, unknown> = {};
    setReasoningPayload(payload, "HIGH");
    assert.deepEqual(payload.reasoning, { effort: "HIGH" });
    assert.deepEqual(payload.include, ["reasoning.encrypted_content"]);
  });

  it("does nothing for undefined", () => {
    const payload: Record<string, unknown> = {};
    setReasoningPayload(payload, undefined);
    assert.equal(payload.reasoning, undefined);
  });

  it("throws on an empty value", () => {
    const payload: Record<string, unknown> = {};
    assert.throws(
      () => setReasoningPayload(payload, ""),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError,
    );
  });

  it("maps ultra to max and preserves custom effort values", () => {
    const ultraPayload: Record<string, unknown> = {};
    setReasoningPayload(ultraPayload, "ultra");
    assert.deepEqual(ultraPayload.reasoning, { effort: "max" });

    const customCasePayload: Record<string, unknown> = {};
    setReasoningPayload(customCasePayload, "ULTRA");
    assert.deepEqual(customCasePayload.reasoning, { effort: "ULTRA" });

    const customPayload: Record<string, unknown> = {};
    setReasoningPayload(customPayload, "provider-Future");
    assert.deepEqual(customPayload.reasoning, { effort: "provider-Future" });
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
      setReasoningPayload(payload, effort);
      assert.deepEqual(payload.reasoning, { effort });
      assert.deepEqual(payload.include, ["reasoning.encrypted_content"]);
    }
  });

  it("omits standard mode, preserves context, and rejects pro mode", () => {
    const payload: Record<string, unknown> = {
      model: "gpt-5.6-sol",
      reasoning: { summary: "auto" },
    };
    setReasoningPayload(
      payload,
      "medium",
      { mode: "standard", context: "all_turns" },
      "gpt-5.6-sol",
    );
    assert.deepEqual(payload.reasoning, {
      summary: "auto",
      effort: "medium",
      context: "all_turns",
    });
    assert.throws(() => setReasoningPayload(
      { model: "gpt-5.6-sol" },
      "medium",
      { mode: "pro" },
      "gpt-5.6-sol",
    ), (error) => error instanceof ChatGPTOAuthError);
    const existingStandard: Record<string, unknown> = {
      model: "gpt-5.6-sol",
      reasoning: { mode: "standard" },
    };
    setReasoningPayload(existingStandard, undefined, undefined, "gpt-5.6-sol");
    assert.equal(Object.hasOwn(existingStandard, "reasoning"), false);
    assert.throws(() => setReasoningPayload(
      { model: "gpt-5.6-sol", reasoning: { mode: "pro" } },
      undefined,
      undefined,
      "gpt-5.6-sol",
    ), (error) => error instanceof ChatGPTOAuthError);
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
    assert.deepEqual(result.arguments, { city: "Seoul" });
  });

  it("parses custom_tool_call item", () => {
    const item = {
      type: "custom_tool_call",
      name: "my_tool",
      id: "ct-1",
      input: '{"x":1}',
    };
    const result = toolCallFromResponseItem(item);
    assert.ok(result);
    assert.equal(result.name, "my_tool");
    assert.deepEqual(result.arguments, { x: 1 });
  });

  it("returns null for non-tool items", () => {
    assert.equal(
      toolCallFromResponseItem({ type: "message" }),
      null,
    );
  });

  it("returns null for missing name", () => {
    assert.equal(
      toolCallFromResponseItem({
        type: "function_call",
        name: "",
      }),
      null,
    );
  });

  it("handles dict arguments", () => {
    const item = {
      type: "function_call",
      name: "tool",
      call_id: "c1",
      arguments: { key: "value" },
    };
    const result = toolCallFromResponseItem(item);
    assert.ok(result);
    assert.deepEqual(result.arguments, { key: "value" });
  });

  it("handles malformed JSON arguments", () => {
    const item = {
      type: "function_call",
      name: "tool",
      call_id: "c1",
      arguments: "not-json",
    };
    const result = toolCallFromResponseItem(item);
    assert.ok(result);
    assert.deepEqual(result.arguments, {
      input: "not-json",
    });
  });
});

describe("textFromResponseItems", () => {
  it("extracts from output_text items", () => {
    const items = [
      { type: "output_text", text: "hello" },
      { type: "output_text", text: " world" },
    ];
    assert.equal(textFromResponseItems(items), "hello world");
  });

  it("extracts from message items", () => {
    const items = [
      {
        type: "message",
        content: [{ type: "output_text", text: "content" }],
      },
    ];
    assert.equal(textFromResponseItems(items), "content");
  });

  it("skips non-text items", () => {
    const items = [
      { type: "function_call", name: "tool" },
      { type: "output_text", text: "result" },
    ];
    assert.equal(textFromResponseItems(items), "result");
  });

  it("handles text type", () => {
    const items = [{ type: "text", text: "simple" }];
    assert.equal(textFromResponseItems(items), "simple");
  });

  it("returns empty for no text", () => {
    assert.equal(
      textFromResponseItems([{ type: "image" }]),
      "",
    );
  });

  it("handles string content parts in message", () => {
    const items = [
      { type: "message", content: ["hello", " world"] },
    ];
    assert.equal(
      textFromResponseItems(items),
      "hello world",
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
      { name: "ChatGPTOAuthError" },
    );
  });

  it("rejects empty image_url", () => {
    assert.throws(
      () => validateImageContentItems([{ image_url: "" }]),
      { name: "ChatGPTOAuthError" },
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

  it("returns null for non-image items", () => {
    assert.equal(
      imageGenerationFromItem({ type: "message" }),
      null,
    );
  });

  it("throws on empty result", () => {
    assert.throws(
      () =>
        imageGenerationFromItem({
          type: "image_generation_call",
          result: "",
        }),
      { name: "ChatGPTOAuthError" },
    );
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

  it("parses Chat Completions format", () => {
    const value = {
      prompt_tokens: 80,
      completion_tokens: 40,
      total_tokens: 120,
      prompt_tokens_details: { cached_tokens: 10 },
    };
    const result = usageFromResponse(value);
    assert.ok(result);
    assert.equal(result.prompt_tokens, 80);
    assert.equal(result.completion_tokens, 40);
    assert.equal(result.cached_tokens, 10);
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

  it("calculates total_tokens if missing", () => {
    const result = usageFromResponse({
      input_tokens: 10,
      output_tokens: 5,
    });
    assert.ok(result);
    assert.equal(result.total_tokens, 15);
  });

  it("reads cached_input_tokens fallback", () => {
    const result = usageFromResponse({
      input_tokens: 100,
      output_tokens: 50,
      cached_input_tokens: 30,
    });
    assert.ok(result);
    assert.equal(result.cached_tokens, 30);
  });

  it("reads cache_read_input_tokens fallback", () => {
    const result = usageFromResponse({
      input_tokens: 100,
      output_tokens: 50,
      cache_read_input_tokens: 25,
    });
    assert.ok(result);
    assert.equal(result.cached_tokens, 25);
  });
});
