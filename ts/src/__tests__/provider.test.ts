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
} from "../provider.js";
import { ChatGPTOAuthError } from "../auth.js";

function providerMessages(): Message[] {
  return [
    { role: MessageRole.SYSTEM, content: "You are helpful." },
    { role: MessageRole.USER, content: "Hello" },
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
      responsesPayload(messages: Message[], opts: { model: string; tools: ToolSchema[]; toolChoice: Record<string, unknown> }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      tools: [webSearchTool],
      toolChoice: { type: "web_search" },
    });

    assert.deepEqual(payload.tools, [{ type: "web_search", external_web_access: true }]);
    assert.deepEqual(payload.tool_choice, { type: "web_search" });
    assert.deepEqual(payload.include, ["web_search_call.action.sources"]);
  });

  it("adds encrypted reasoning include when reasoning effort is present", () => {
    const provider = new ChatGPTOAuthProvider({});
    const payload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; reasoningEffort: string }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.5", reasoningEffort: "high" });

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
    assert.equal(payload.parallel_tool_calls, false);
    assert.deepEqual((payload.input as Record<string, unknown>[])[0], { type: "developer_message", content: "You are helpful." });
    assert.deepEqual((payload.input as Record<string, unknown>[])[1], {
      type: "developer_message",
      additional_tools: [{ type: "function", name: "lookup", description: "Lookup", parameters: { type: "object" }, strict: false }],
    });
  });

  it("Responses Lite auto uses the shared capability table", () => {
    const provider = new ChatGPTOAuthProvider({});
    const payload = (provider as unknown as {
      responsesPayload(
        messages: Message[],
        opts: { model: string; responsesLite: string; serviceTier: string },
      ): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.5", responsesLite: "auto", serviceTier: "priority" });
    const unknownPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; serviceTier: string }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "unknown-model", serviceTier: "priority" });

    assert.equal(Object.hasOwn(payload, "tools"), true);
    assert.deepEqual(payload.text, { verbosity: "low" });
    assert.equal(payload.service_tier, "priority");
    assert.equal(Object.hasOwn(unknownPayload, "service_tier"), false);
  });

  it("parallel tool calls use the shared capability table", () => {
    const provider = new ChatGPTOAuthProvider({});
    const payload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; parallelToolCalls: boolean }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.5", parallelToolCalls: true });
    const sparkPayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; parallelToolCalls: boolean }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.3-codex-spark", parallelToolCalls: true });
    const litePayload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; parallelToolCalls: boolean; responsesLite: boolean }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), { model: "gpt-5.5", parallelToolCalls: true, responsesLite: true });

    assert.equal(payload.parallel_tool_calls, true);
    assert.equal(sparkPayload.parallel_tool_calls, false);
    assert.equal(litePayload.parallel_tool_calls, false);
  });

  it("codex metadata mode overlays reserved keys", () => {
    const provider = new ChatGPTOAuthProvider({ authJsonPath: writeAuthFile() });
    const payload = (provider as unknown as {
      responsesPayload(messages: Message[], opts: { model: string; clientMetadata: Record<string, string>; codexMetadata: boolean }): Record<string, unknown>;
    }).responsesPayload(providerMessages(), {
      model: "gpt-5.5",
      clientMetadata: { app: "kept", turn_id: "user-value" },
      codexMetadata: true,
    });
    const metadata = payload.client_metadata as Record<string, string>;

    assert.equal(metadata.app, "kept");
    assert.notEqual(metadata.turn_id, "user-value");
    assert.equal(JSON.parse(metadata["x-codex-turn-metadata"]).source, "codex-as-api");
  });
});

describe("Codex CLI request headers", () => {
  it("formats official originator and versioned User-Agent headers", () => {
    const headers = codexCliHeadersForVersion("1.2.3\n");

    assert.equal(headers.originator, "codex_cli_rs");
    assert.match(headers["User-Agent"], /^codex_cli_rs\/1\.2\.3 \(.+\) codex-as-api$/);
  });

  it("omits User-Agent when the npm version cannot be validated", () => {
    const headers = codexCliHeadersForVersion("not-a-version");

    assert.equal(headers.originator, "codex_cli_rs");
    assert.equal(Object.hasOwn(headers, "User-Agent"), false);
  });

  it("adds Codex CLI headers to ChatGPT OAuth requests", () => {
    const previous = process.env.CODEX_AS_API_CODEX_CLI_VERSION;
    process.env.CODEX_AS_API_CODEX_CLI_VERSION = "9.8.7";
    try {
      const provider = new ChatGPTOAuthProvider({ authJsonPath: writeAuthFile() });
      const headers = (provider as unknown as { getHeaders(): Record<string, string> }).getHeaders();

      assert.equal(headers.originator, "codex_cli_rs");
      assert.match(headers["User-Agent"], /^codex_cli_rs\/9\.8\.7 \(.+\) codex-as-api$/);
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

  it("returns null for non-object JSON", () => {
    assert.equal(decodeSSEBlock(["data: 42"]), null);
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
      '\n[{"type":"message","role":"user","content":"hi"}]';
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

  it("expands compaction marker", () => {
    const inner = [
      {
        type: "message",
        role: "user",
        content: [{ type: "input_text", text: "hi" }],
      },
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
    assert.equal(items.length, 1);
    assert.equal(items[0].type, "message");
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

  it("normalizes case", () => {
    const payload: Record<string, unknown> = {};
    setReasoningPayload(payload, "HIGH");
    assert.deepEqual(payload.reasoning, { effort: "high" });
    assert.deepEqual(payload.include, ["reasoning.encrypted_content"]);
  });

  it("does nothing for undefined", () => {
    const payload: Record<string, unknown> = {};
    setReasoningPayload(payload, undefined);
    assert.equal(payload.reasoning, undefined);
  });

  it("throws on invalid value", () => {
    const payload: Record<string, unknown> = {};
    assert.throws(() => setReasoningPayload(payload, "ultra"), {
      name: "ChatGPTOAuthError",
    });
  });

  it("accepts all valid values", () => {
    for (const effort of [
      "none",
      "minimal",
      "low",
      "medium",
      "high",
      "xhigh",
    ]) {
      const payload: Record<string, unknown> = {};
      setReasoningPayload(payload, effort);
      assert.deepEqual(payload.reasoning, { effort });
      assert.deepEqual(payload.include, ["reasoning.encrypted_content"]);
    }
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
      { image_url: "data:image/png;base64,abc" },
    ]);
    assert.equal(result.length, 1);
    assert.equal(result[0].type, "input_image");
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
      input_tokens_details: { cached_tokens: 20 },
    };
    const result = usageFromResponse(value);
    assert.ok(result);
    assert.equal(result.prompt_tokens, 100);
    assert.equal(result.completion_tokens, 50);
    assert.equal(result.total_tokens, 150);
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
