import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import {
  anthropicRequestToInternal,
  internalResponseToAnthropic,
  anthropicStreamAdapter,
  formatAnthropicError,
} from "../anthropic-adapter.js";
import { MessageRole } from "../messages.js";
import type { AssistantResponse, ToolCall, Usage } from "../messages.js";

function makeResponse(overrides: Partial<AssistantResponse> = {}): AssistantResponse {
  return {
    content: "",
    tool_calls: [],
    finish_reason: "stop",
    usage: null,
    reasoning_content: null,
    raw: null,
    ...overrides,
  };
}

function makeUsage(overrides: Partial<Usage> = {}): Usage {
  return {
    prompt_tokens: 0,
    completion_tokens: 0,
    total_tokens: 0,
    cached_tokens: 0,
    ...overrides,
  };
}

async function collectStreamEvents(
  events: Record<string, unknown>[],
): Promise<Record<string, unknown>[]> {
  async function* makeIter(): AsyncIterable<Record<string, unknown>> {
    for (const e of events) yield e;
  }
  const result: Record<string, unknown>[] = [];
  for await (const sseStr of anthropicStreamAdapter(makeIter(), "test-model", "msg_test")) {
    for (const line of sseStr.trim().split("\n")) {
      if (line.startsWith("data: ")) {
        result.push(JSON.parse(line.slice(6)));
      }
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// Request conversion tests
// ---------------------------------------------------------------------------

describe("anthropicRequestToInternal", () => {
  it("system string", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      system: "You are helpful.",
    });
    assert.equal(messages[0].role, MessageRole.SYSTEM);
    assert.equal(messages[0].content, "You are helpful.");
    assert.equal(messages[1].role, MessageRole.USER);
  });

  it("system content blocks", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      system: [
        { type: "text", text: "Rule 1" },
        { type: "text", text: "Rule 2" },
      ],
    });
    assert.equal(messages[0].content, "Rule 1\n\nRule 2");
  });

  it("no system", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
    });
    assert.equal(messages[0].role, MessageRole.USER);
  });

  it("user text message", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "Hello" }],
    });
    assert.equal(messages.length, 1);
    assert.equal(messages[0].role, MessageRole.USER);
    assert.equal(messages[0].content, "Hello");
  });

  it("user content blocks with tool_result", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{
        role: "user",
        content: [
          { type: "text", text: "Here is the result:" },
          { type: "tool_result", tool_use_id: "call-1", content: "42" },
        ],
      }],
    });
    assert.equal(messages.length, 2);
    assert.equal(messages[0].role, MessageRole.USER);
    assert.equal(messages[0].content, "Here is the result:");
    assert.equal(messages[1].role, MessageRole.TOOL);
    assert.equal(messages[1].content, "42");
    assert.equal(messages[1].tool_call_id, "call-1");
  });

  it("user tool_result only", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "call-1", content: "result1" },
          { type: "tool_result", tool_use_id: "call-2", content: "result2" },
        ],
      }],
    });
    assert.equal(messages.length, 2);
    assert.ok(messages.every((m) => m.role === MessageRole.TOOL));
  });

  it("maps direct and tool-result URL images and preserves tool errors", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{
        role: "user",
        content: [
          { type: "text", text: "inspect" },
          {
            type: "image",
            source: { type: "url", url: "https://example.com/direct.png" },
          },
          {
            type: "tool_result",
            tool_use_id: "call-image",
            is_error: true,
            content: [
              { type: "text", text: "tool failed" },
              {
                type: "image",
                source: { type: "url", url: "https://example.com/tool.png" },
              },
            ],
          },
        ],
      }],
    });

    assert.deepEqual(messages, [
      {
        role: MessageRole.USER,
        content: "inspect",
        images: ["https://example.com/direct.png"],
      },
      {
        role: MessageRole.TOOL,
        content: "[tool_error]\ntool failed",
        tool_call_id: "call-image",
        name: "call-image",
      },
      {
        role: MessageRole.USER,
        content: "",
        images: ["https://example.com/tool.png"],
      },
    ]);
  });

  it("rejects known image sources with missing payloads", () => {
    const malformedBlocks = [
      { type: "image" },
      { type: "image", source: null },
      { type: "image", source: { type: "file", file_id: "file-1" } },
      { type: "image", source: { type: "base64", media_type: 42, data: "AAAA" } },
      { type: "image", source: { type: "base64", media_type: "image/png", data: "" } },
      { type: "image", source: { type: "base64", media_type: "image/png" } },
      { type: "image", source: { type: "url", url: "" } },
      { type: "image", source: { type: "url" } },
    ];
    for (const block of malformedBlocks) {
      assert.throws(() => anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: [block] }],
      }));
      assert.throws(() => anthropicRequestToInternal({
        model: "test",
        messages: [{
          role: "user",
          content: [{
            type: "tool_result",
            tool_use_id: "call-image",
            content: [block],
          }],
        }],
      }));
    }

    assert.throws(
      () => anthropicRequestToInternal({
        model: "test",
        messages: [{
          role: "user",
          content: [{
            type: "image",
            source: { type: "base64", media_type: 42, data: "AAAA" },
          }],
        }],
      }),
      /Anthropic base64 image source requires a string media_type/,
    );
  });

  it("assistant text", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "assistant", content: "Hello!" }],
    });
    assert.equal(messages[0].role, MessageRole.ASSISTANT);
    assert.equal(messages[0].content, "Hello!");
  });

  it("assistant tool_use blocks", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{
        role: "assistant",
        content: [
          { type: "text", text: "Let me check." },
          { type: "tool_use", id: "tc-1", name: "get_weather", input: { city: "Seoul" } },
        ],
      }],
    });
    assert.equal(messages[0].content, "Let me check.");
    assert.equal(messages[0].tool_calls?.length, 1);
    assert.equal(messages[0].tool_calls?.[0].name, "get_weather");
    assert.deepEqual(messages[0].tool_calls?.[0].arguments, { city: "Seoul" });
  });

  it("assistant thinking block", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{
        role: "assistant",
        content: [
          { type: "thinking", thinking: "Let me think...", signature: "sig-abc" },
          { type: "text", text: "The answer is 42." },
        ],
      }],
    });
    assert.equal(messages[0].reasoning_content, "Let me think...");
    assert.equal(messages[0].content, "The answer is 42.");
  });

  it("preserves assistant server web-search history as context text", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{
        role: "assistant",
        content: [
          { type: "server_tool_use", id: "srv_1", name: "web_search", input: { query: "codex" } },
          {
            type: "web_search_tool_result",
            tool_use_id: "srv_1",
            content: [{ title: "Codex", url: "https://example.com", page_age: "1d" }],
          },
          { type: "text", text: "Summary" },
        ],
      }],
    });
    assert.match(messages[0].content, /server_tool_use: web_search/);
    assert.match(messages[0].content, /https:\/\/example.com/);
    assert.match(messages[0].content, /Summary/);
  });

  it("preserves non-text tool_result blocks as text context", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: "call-1",
          content: [
            { type: "search_result", title: "Docs", url: "https://docs.example", content: "body" },
            { type: "document", title: "Spec", source: { type: "text", data: "document body" } },
          ],
        }],
      }],
    });
    assert.equal(messages[0].role, MessageRole.TOOL);
    assert.match(messages[0].content, /Docs/);
    assert.match(messages[0].content, /document body/);
  });

  it("tools conversion", () => {
    const { tools } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      tools: [{
        name: "get_weather",
        description: "Get weather",
        input_schema: { type: "object", properties: { city: { type: "string" } } },
      }],
    });
    assert.ok(tools !== null);
    assert.equal(tools!.length, 1);
    assert.equal(tools![0].name, "get_weather");
    assert.deepEqual(tools![0].parameters, { type: "object", properties: { city: { type: "string" } } });
  });

  it("rejects programmatic tool-only fields", () => {
    const base = {
      model: "test",
      messages: [],
    };
    assert.throws(() => anthropicRequestToInternal({
      ...base,
      tools: [{ type: "programmatic_tool_calling" }],
    }));
    assert.throws(() => anthropicRequestToInternal({
      ...base,
      tools: [{
        name: "lookup",
        input_schema: { type: "object" },
        allowed_callers: ["programmatic"],
      }],
    }));
    assert.throws(() => anthropicRequestToInternal({
      ...base,
      tools: [{
        name: "lookup",
        input_schema: { type: "object" },
        output_schema: { type: "object" },
      }],
    }));
  });

  it("rejects enabled beta tool semantics and accepts explicit no-ops", () => {
    const base = {
      model: "test",
      messages: [] as Record<string, unknown>[],
    };
    for (const field of ["strict", "defer_loading", "eager_input_streaming"] as const) {
      assert.throws(() => anthropicRequestToInternal({
        ...base,
        tools: [{
          name: "lookup",
          input_schema: { type: "object" },
          [field]: true,
        }],
      }));
    }

    const { tools } = anthropicRequestToInternal({
      ...base,
      tools: [{
        name: "lookup",
        input_schema: { type: "object" },
        strict: false,
        defer_loading: null,
        eager_input_streaming: false,
      }],
    });
    assert.deepEqual(tools, [{
      name: "lookup",
      description: "",
      parameters: { type: "object" },
    }]);
  });

  it("tool_choice auto", () => {
    const { toolChoice } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      toolChoice: { type: "auto" },
    });
    assert.equal(toolChoice, "auto");
  });

  it("tool_choice any", () => {
    const { toolChoice } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      toolChoice: { type: "any" },
    });
    assert.equal(toolChoice, "required");
  });

  it("tool_choice specific", () => {
    const { toolChoice } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      toolChoice: { type: "tool", name: "get_weather" },
    });
    assert.deepEqual(toolChoice, { type: "function", name: "get_weather" });
  });

  it("converts Anthropic web_search server tool", () => {
    const { tools } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      tools: [{
        type: "web_search_20260209",
        name: "web_search",
        allowed_domains: ["example.com"],
        max_uses: 8,
        user_location: { type: "approximate", country: "US" },
      }],
    });
    assert.ok(tools);
    assert.equal(tools[0].name, "web_search");
    assert.equal(tools[0].parameters.__codex_as_api_tool_type, "web_search");
    const openaiTool = tools[0].parameters.openai_tool as Record<string, unknown>;
    assert.equal(openaiTool.type, "web_search");
    assert.equal(openaiTool.external_web_access, true);
    assert.deepEqual(openaiTool.filters, { allowed_domains: ["example.com"] });
    assert.deepEqual(openaiTool.user_location, { type: "approximate", country: "US" });
  });

  it("converts unsuffixed Anthropic web_search server tool", () => {
    const { tools } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      tools: [{ type: "web_search", name: "web_search" }],
    });
    assert.equal(tools?.[0].parameters.__codex_as_api_tool_type, "web_search");
  });

  it("rejects unsupported Anthropic web_search blocked_domains", () => {
    assert.throws(() => anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      tools: [{
        type: "web_search_20250305",
        name: "web_search",
        blocked_domains: ["example.com"],
      }],
    }), /blocked_domains/);
  });

  it("converts web_search tool_choice to hosted tool choice", () => {
    const { toolChoice } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      toolChoice: { type: "tool", name: "web_search" },
    });
    assert.deepEqual(toolChoice, { type: "web_search" });
  });

  it("tool_choice none", () => {
    const { toolChoice } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      toolChoice: { type: "none" },
    });
    assert.equal(toolChoice, "none");
  });

  it("thinking enabled", () => {
    const { reasoningEffort } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      thinking: { type: "enabled", budget_tokens: 4096 },
    });
    assert.equal(reasoningEffort, "high");
  });

  it("thinking adaptive", () => {
    const { reasoningEffort } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      thinking: { type: "adaptive" },
    });
    assert.equal(reasoningEffort, "medium");
  });

  it("thinking disabled", () => {
    const { reasoningEffort } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      thinking: { type: "disabled" },
    });
    assert.equal(reasoningEffort, "none");
  });

  it("uses output_config effort ahead of adaptive or enabled thinking", () => {
    for (const thinkingType of ["adaptive", "enabled"] as const) {
      const { reasoningEffort } = anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        thinking: { type: thinkingType, display: "omitted" },
        outputConfig: { effort: "max" },
      });
      assert.equal(reasoningEffort, "max");
    }
  });

  it("uses call-level disabled thinking ahead of output_config effort", () => {
    for (const effort of ["low", "medium", "high", "xhigh", "max"]) {
      const { reasoningEffort } = anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        thinking: { type: "disabled" },
        outputConfig: { effort },
      });
      assert.equal(reasoningEffort, "none");
    }
  });

  it("validates output_config effort before disabled-thinking precedence", () => {
    for (const effort of ["", 42, "ultra"]) {
      assert.throws(() => anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        thinking: { type: "disabled" },
        outputConfig: { effort },
      }));
    }
  });

  it("rejects invalid output_config effort and non-null task budgets", () => {
    for (const outputConfig of [
      { effort: "" },
      { effort: 42 },
      { effort: "ultra" },
      { task_budget: { type: "tokens", total: 20_000 } },
      { unknown_control: true },
    ]) {
      assert.throws(() => anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        outputConfig,
      }));
    }
  });

  it("keeps nested output_config format conversion alongside effort", () => {
    const { reasoningEffort, text } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      thinking: { type: "adaptive" },
      outputConfig: {
        effort: "high",
        format: { type: "json_object" },
      },
      outputFormat: { type: "json_object" },
    });
    assert.equal(reasoningEffort, "high");
    assert.deepEqual(text, { format: { type: "json_object" } });
  });

  it("rejects malformed, unsupported, and conflicting output formats", () => {
    const invalid = [
      { outputFormat: "json" },
      { outputConfig: { format: "json" } },
      { outputFormat: {} },
      { outputFormat: { type: "future" } },
      { outputFormat: { type: "json_schema", schema: [] } },
      {
        outputFormat: {
          type: "json_schema",
          schema: { type: "object" },
          extra: true,
        },
      },
      {
        outputConfig: {
          format: { type: "json_object", schema: { type: "object" } },
        },
      },
      {
        outputFormat: {
          type: "json_schema",
          schema: { type: "object" },
          name: "",
        },
      },
      {
        outputFormat: {
          type: "json_schema",
          schema: { type: "object" },
          description: 42,
        },
      },
      {
        outputFormat: {
          type: "json_schema",
          schema: { type: "object" },
          strict: "true",
        },
      },
      {
        outputFormat: { type: "json_object" },
        outputConfig: {
          format: { type: "json_schema", schema: { type: "object" } },
        },
      },
    ];
    for (const controls of invalid) {
      assert.throws(() => anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        ...controls,
      }));
    }
  });

  it("maps Anthropic output_format to OpenAI Responses text.format", () => {
    const { text } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      outputFormat: {
        type: "json_schema",
        name: "my schema!",
        description: "Answer envelope",
        schema: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
        strict: false,
      },
    });
    assert.deepEqual(text, {
      format: {
        type: "json_schema",
        name: "my_schema_",
        description: "Answer envelope",
        schema: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
        strict: false,
      },
    });
  });

  it("stop sequences", () => {
    const { stop } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      stopSequences: ["STOP", "END"],
    });
    assert.deepEqual(stop, ["STOP", "END"]);
  });

  it("tool_result with content blocks", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{
        role: "user",
        content: [{
          type: "tool_result",
          tool_use_id: "call-1",
          content: [
            { type: "text", text: "result line 1" },
            { type: "text", text: "result line 2" },
          ],
        }],
      }],
    });
    assert.equal(messages[0].role, MessageRole.TOOL);
    assert.equal(messages[0].content, "result line 1result line 2");
  });
});

// ---------------------------------------------------------------------------
// Non-streaming response conversion
// ---------------------------------------------------------------------------

describe("internalResponseToAnthropic", () => {
  it("text response", () => {
    const resp = makeResponse({
      content: "Hello!",
      finish_reason: "stop",
      usage: makeUsage({ prompt_tokens: 10, completion_tokens: 5 }),
    });
    const result = internalResponseToAnthropic(resp, "test-model", "msg_123");
    assert.equal(result.id, "msg_123");
    assert.equal(result.type, "message");
    assert.equal(result.role, "assistant");
    assert.equal(result.stop_reason, "end_turn");
    const content = result.content as Record<string, unknown>[];
    assert.equal(content.length, 1);
    assert.equal(content[0].type, "text");
    assert.equal(content[0].text, "Hello!");
    const usage = result.usage as Record<string, unknown>;
    assert.equal(usage.input_tokens, 10);
    assert.equal(usage.output_tokens, 5);
  });

  it("tool_use response", () => {
    const tc: ToolCall = { id: "tc-1", name: "get_weather", arguments: { city: "Seoul" } };
    const resp = makeResponse({
      tool_calls: [tc],
      finish_reason: "stop",
      usage: makeUsage({ prompt_tokens: 20, completion_tokens: 10 }),
    });
    const result = internalResponseToAnthropic(resp, "test-model", "msg_123");
    assert.equal(result.stop_reason, "tool_use");
    const content = result.content as Record<string, unknown>[];
    assert.equal(content.length, 1);
    assert.equal(content[0].type, "tool_use");
    assert.equal(content[0].name, "get_weather");
  });

  it("reasoning response", () => {
    const resp = makeResponse({
      content: "42",
      reasoning_content: "Let me think about this...",
      finish_reason: "stop",
      usage: makeUsage({ prompt_tokens: 10, completion_tokens: 5 }),
    });
    const result = internalResponseToAnthropic(resp, "test-model", "msg_123");
    const content = result.content as Record<string, unknown>[];
    assert.equal(content.length, 2);
    assert.equal(content[0].type, "thinking");
    assert.equal(content[0].thinking, "Let me think about this...");
    assert.equal(content[1].type, "text");
  });

  it("empty response", () => {
    const resp = makeResponse({ content: "", finish_reason: "stop" });
    const result = internalResponseToAnthropic(resp, "test-model", "msg_123");
    const content = result.content as Record<string, unknown>[];
    assert.equal(content.length, 1);
    assert.equal(content[0].type, "text");
    assert.equal(content[0].text, "");
  });

  it("cached tokens in usage", () => {
    const resp = makeResponse({
      content: "hi",
      finish_reason: "stop",
      usage: makeUsage({
        prompt_tokens: 100,
        completion_tokens: 10,
        cached_tokens: 50,
        cache_write_tokens: 25,
      }),
    });
    const result = internalResponseToAnthropic(resp, "m", "msg_1");
    const usage = result.usage as Record<string, unknown>;
    assert.equal(usage.cache_read_input_tokens, 50);
    assert.equal(usage.cache_creation_input_tokens, 25);
  });

  it("adds web_search server tool blocks before text", () => {
    const resp = makeResponse({
      content: "Final answer",
      raw: {
        events: [{
          type: "web_search_call",
          id: "srvtoolu_ws1",
          input: { query: "latest news" },
          content: [{ type: "web_search_result", url: "https://example.com", title: "Example" }],
        }],
      },
    });
    const result = internalResponseToAnthropic(resp, "m", "msg_1");
    const content = result.content as Record<string, unknown>[];
    assert.deepEqual(content.map((c) => c.type), ["server_tool_use", "web_search_tool_result", "text"]);
    assert.equal(content[0].name, "web_search");
    assert.equal(content[1].tool_use_id, "srvtoolu_ws1");
    assert.deepEqual((content[1].content as unknown[])[0], {
      type: "web_search_result",
      url: "https://example.com",
      title: "Example",
    });
    assert.deepEqual((result.usage as Record<string, unknown>).server_tool_use, { web_search_requests: 1 });
  });
});

// ---------------------------------------------------------------------------
// Streaming adapter
// ---------------------------------------------------------------------------

describe("anthropicStreamAdapter", () => {
  it("text only stream", async () => {
    const events = [
      { type: "content", text: "Hello" },
      { type: "content", text: " world" },
      { type: "finish", finish_reason: "stop", usage: { output_tokens: 5 } },
    ];
    const result = await collectStreamEvents(events);
    const types = result.map((e) => e.type);
    assert.equal(types[0], "message_start");
    assert.ok(types.includes("content_block_start"));
    assert.ok(types.includes("content_block_delta"));
    assert.ok(types.includes("content_block_stop"));
    assert.ok(types.includes("message_delta"));
    assert.equal(types[types.length - 1], "message_stop");

    const textDeltas = result.filter(
      (e) => e.type === "content_block_delta" &&
        (e.delta as Record<string, unknown>).type === "text_delta",
    );
    assert.equal(textDeltas.length, 2);
    assert.equal((textDeltas[0].delta as Record<string, unknown>).text, "Hello");
    assert.equal((textDeltas[1].delta as Record<string, unknown>).text, " world");
  });

  it("thinking then text", async () => {
    const events = [
      { type: "reasoning_delta", text: "thinking..." },
      { type: "content", text: "result" },
      { type: "finish", finish_reason: "stop" },
    ];
    const result = await collectStreamEvents(events);
    const blockStarts = result.filter((e) => e.type === "content_block_start");
    assert.equal(blockStarts.length, 2);
    assert.equal((blockStarts[0].content_block as Record<string, unknown>).type, "thinking");
    assert.equal((blockStarts[1].content_block as Record<string, unknown>).type, "text");
  });

  it("tool call stream", async () => {
    const events = [
      { type: "tool_call", id: "tc-1", name: "get_weather", arguments: { city: "Seoul" } },
      { type: "finish", finish_reason: "tool_calls" },
    ];
    const result = await collectStreamEvents(events);
    const blockStarts = result.filter((e) => e.type === "content_block_start");
    assert.equal(blockStarts.length, 1);
    assert.equal((blockStarts[0].content_block as Record<string, unknown>).type, "tool_use");
    assert.equal((blockStarts[0].content_block as Record<string, unknown>).name, "get_weather");

    const jsonDeltas = result.filter(
      (e) => e.type === "content_block_delta" &&
        (e.delta as Record<string, unknown>).type === "input_json_delta",
    );
    assert.equal(jsonDeltas.length, 1);
    assert.deepEqual(
      JSON.parse((jsonDeltas[0].delta as Record<string, unknown>).partial_json as string),
      { city: "Seoul" },
    );
    const messageDelta = result.find((event) => event.type === "message_delta");
    assert.equal(
      ((messageDelta?.delta as Record<string, unknown>) ?? {}).stop_reason,
      "tool_use",
    );
  });

  it("text then tool call", async () => {
    const events = [
      { type: "content", text: "Let me check." },
      { type: "tool_call", id: "tc-1", name: "search", arguments: { q: "test" } },
      { type: "finish", finish_reason: "stop" },
    ];
    const result = await collectStreamEvents(events);
    const blockStarts = result.filter((e) => e.type === "content_block_start");
    assert.equal(blockStarts.length, 2);
    assert.equal((blockStarts[0].content_block as Record<string, unknown>).type, "text");
    assert.equal((blockStarts[1].content_block as Record<string, unknown>).type, "tool_use");
  });

  it("empty stream", async () => {
    const events = [{ type: "finish", finish_reason: "stop" }];
    const result = await collectStreamEvents(events);
    const blockStarts = result.filter((e) => e.type === "content_block_start");
    assert.equal(blockStarts.length, 1);
    assert.equal((blockStarts[0].content_block as Record<string, unknown>).type, "text");
  });

  it("message_delta stop reason", async () => {
    const events = [
      { type: "content", text: "hi" },
      { type: "finish", finish_reason: "stop", usage: { output_tokens: 3 } },
    ];
    const result = await collectStreamEvents(events);
    const msgDelta = result.find((e) => e.type === "message_delta")!;
    assert.equal((msgDelta.delta as Record<string, unknown>).stop_reason, "end_turn");
    assert.equal((msgDelta.usage as Record<string, unknown>).output_tokens, 3);
  });

  it("routes real cumulative usage into message_delta", async () => {
    const events = [
      { type: "content", text: "hi" },
      {
        type: "finish",
        finish_reason: "stop",
        usage: {
          input_tokens: 123,
          output_tokens: 7,
          cache_creation_input_tokens: 11,
          cache_read_input_tokens: 13,
          cache_creation: { ephemeral_5m_input_tokens: 11, ephemeral_1h_input_tokens: 0 },
          server_tool_use: { web_search_requests: 2 },
        },
      },
    ];
    const result = await collectStreamEvents(events);
    const msgStart = result.find((e) => e.type === "message_start")!;
    assert.deepEqual((msgStart.message as Record<string, unknown>).usage, {
      input_tokens: 0,
      output_tokens: 0,
    });

    const msgDelta = result.find((e) => e.type === "message_delta")!;
    assert.deepEqual(msgDelta.usage, {
      input_tokens: 123,
      output_tokens: 7,
      cache_creation_input_tokens: 11,
      cache_read_input_tokens: 13,
      cache_creation: { ephemeral_5m_input_tokens: 11, ephemeral_1h_input_tokens: 0 },
      server_tool_use: { web_search_requests: 2 },
    });
  });

  it("multiple tool calls", async () => {
    const events = [
      { type: "tool_call", id: "tc-1", name: "tool_a", arguments: { a: 1 } },
      { type: "tool_call", id: "tc-2", name: "tool_b", arguments: { b: 2 } },
      { type: "finish", finish_reason: "stop" },
    ];
    const result = await collectStreamEvents(events);
    const blockStarts = result.filter((e) => e.type === "content_block_start");
    assert.equal(blockStarts.length, 2);
    assert.equal((blockStarts[0].content_block as Record<string, unknown>).name, "tool_a");
    assert.equal((blockStarts[1].content_block as Record<string, unknown>).name, "tool_b");
    assert.equal(blockStarts[0].index, 0);
    assert.equal(blockStarts[1].index, 1);
  });

  it("web_search_call stream emits server tool result before text", async () => {
    const events = [
      {
        type: "web_search_call",
        id: "srvtoolu_ws1",
        input: { query: "current time" },
        content: [{ type: "web_search_result", url: "https://example.com", title: "Example" }],
      },
      { type: "content", text: "It is noon." },
      { type: "finish", finish_reason: "stop" },
    ];
    const result = await collectStreamEvents(events);
    const blockStarts = result.filter((e) => e.type === "content_block_start");
    assert.deepEqual(blockStarts.map((e) => (e.content_block as Record<string, unknown>).type), [
      "server_tool_use",
      "web_search_tool_result",
      "text",
    ]);
    const msgDelta = result.find((e) => e.type === "message_delta")!;
    assert.deepEqual((msgDelta.usage as Record<string, unknown>).server_tool_use, {
      web_search_requests: 1,
    });
  });
});

// ---------------------------------------------------------------------------
// Error formatting
// ---------------------------------------------------------------------------

describe("formatAnthropicError", () => {
  it("auth error", () => {
    const result = formatAnthropicError(401, "bad key");
    assert.equal(result.type, "error");
    const error = result.error as Record<string, unknown>;
    assert.equal(error.type, "authentication_error");
    assert.equal(error.message, "bad key");
  });

  it("server error", () => {
    const result = formatAnthropicError(500, "internal");
    assert.equal((result.error as Record<string, unknown>).type, "api_error");
  });

  it("rate limit", () => {
    const result = formatAnthropicError(429, "slow down");
    assert.equal((result.error as Record<string, unknown>).type, "rate_limit_error");
  });
});
