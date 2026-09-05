import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import {
  anthropicRequestToInternal,
  internalResponseToAnthropic,
  anthropicStreamAdapter,
  formatAnthropicError,
} from "../anthropic-adapter.js";
import {
  ChatGPTOAuthInvalidRequestError,
  ChatGPTOAuthProtocolError,
} from "../auth.js";
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
  const promptTokens = overrides.prompt_tokens ?? 0;
  const completionTokens = overrides.completion_tokens ?? 0;
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: overrides.total_tokens ?? promptTokens + completionTokens,
    cached_tokens: 0,
    ...overrides,
  };
}

function makeStreamUsage(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    input_tokens: 0,
    output_tokens: 0,
    total_tokens: 0,
    input_tokens_details: { cached_tokens: 0 },
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

  it("requires tool_result tool_use_id to be a string", () => {
    for (const block of [
      { type: "tool_result", content: "result" },
      { type: "tool_result", tool_use_id: null, content: "result" },
      { type: "tool_result", tool_use_id: 1, content: "result" },
    ]) {
      assert.throws(() => anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: [block] }],
      }), /string tool_use_id/);
    }
  });

  it("maps omitted tool_result content to empty and rejects explicit null typed fields", () => {
    const { messages } = anthropicRequestToInternal({
      model: "test",
      messages: [{
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "call-empty" }],
      }],
    });
    assert.equal(messages[0].content, "");

    for (const block of [
      { type: "tool_result", tool_use_id: "call-null", content: null },
      { type: "tool_result", tool_use_id: "call-null", content: "ok", is_error: null },
    ]) {
      assert.throws(() => anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: [block] }],
      }), ChatGPTOAuthInvalidRequestError);
    }
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
      { type: "image", source: { type: "base64", media_type: "image/svg+xml", data: "PHN2Zz4=" } },
      { type: "image", source: { type: "base64", media_type: "text/plain", data: "dGV4dA==" } },
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
      ChatGPTOAuthInvalidRequestError,
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
          {
            type: "tool_use",
            id: "tc-1",
            name: "get_weather",
            input: { city: "Seoul" },
            caller: { type: "direct" },
          },
        ],
      }],
    });
    assert.equal(messages[0].content, "Let me check.");
    assert.equal(messages[0].tool_calls?.length, 1);
    assert.equal(messages[0].tool_calls?.[0].name, "get_weather");
    assert.equal(messages[0].tool_calls?.[0].arguments, '{"city":"Seoul"}');
  });

  it("rejects non-direct or malformed tool_use caller history", () => {
    for (const caller of [
      null,
      "direct",
      {},
      { type: "code_execution_20260120", tool_id: "srv_1" },
      { type: "direct", extra: true },
    ]) {
      assert.throws(() => anthropicRequestToInternal({
        model: "test",
        messages: [{
          role: "assistant",
          content: [{
            type: "tool_use",
            id: "tc-1",
            name: "get_weather",
            input: { city: "Seoul" },
            caller,
          }],
        }],
      }), ChatGPTOAuthInvalidRequestError);
    }
  });

  it("rejects assistant thinking history that cannot be represented", () => {
    assert.throws(() => anthropicRequestToInternal({
      model: "test",
      messages: [{
        role: "assistant",
        content: [
          { type: "thinking", thinking: "Let me think...", signature: "sig-abc" },
          { type: "text", text: "The answer is 42." },
        ],
      }],
    }), ChatGPTOAuthInvalidRequestError);
  });

  it("rejects assistant server web-search history that cannot be represented", () => {
    assert.throws(() => anthropicRequestToInternal({
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
    }), ChatGPTOAuthInvalidRequestError);
  });

  it("rejects non-text tool_result blocks that cannot be represented", () => {
    assert.throws(() => anthropicRequestToInternal({
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
    }), ChatGPTOAuthInvalidRequestError);
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

    assert.throws(() => anthropicRequestToInternal({
      ...base,
      tools: [{
        name: "lookup",
        input_schema: { type: "object" },
        allowed_callers: null,
      }],
    }));
  });

  it("forwards function strict and rejects unsupported enabled beta semantics", () => {
    const base = {
      model: "test",
      messages: [] as Record<string, unknown>[],
    };
    for (const field of ["defer_loading", "eager_input_streaming"] as const) {
      for (const value of [false, true]) {
        assert.throws(() => anthropicRequestToInternal({
          ...base,
          tools: [{
            name: "lookup",
            input_schema: { type: "object" },
            [field]: value,
          }],
        }));
      }
    }

    const { tools } = anthropicRequestToInternal({
      ...base,
      tools: [{
        name: "lookup",
        description: "",
        input_schema: { type: "object" },
        strict: true,
        eager_input_streaming: null,
      }],
    });
    assert.deepEqual(tools, [{
      name: "lookup",
      description: "",
      parameters: { type: "object" },
      strict: true,
    }]);

    assert.throws(() => anthropicRequestToInternal({
      ...base,
      tools: [{ name: "lookup", input_schema: {}, strict: "true" }],
    }), ChatGPTOAuthInvalidRequestError);
    for (const invalidTool of [
      { name: "lookup", input_schema: {}, strict: null },
      { name: "lookup", input_schema: {}, defer_loading: null },
      { name: "lookup", input_schema: {}, allowed_callers: null },
    ]) {
      assert.throws(() => anthropicRequestToInternal({
        ...base,
        tools: [invalidTool],
      }), ChatGPTOAuthInvalidRequestError);
    }
  });

  it("accepts omitted, custom, and null custom tool discriminators", () => {
    for (const tool of [
      { name: "lookup", input_schema: {} },
      { type: "custom", name: "lookup", input_schema: {} },
      { type: null, name: "lookup", input_schema: {} },
    ]) {
      const { tools } = anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        tools: [tool],
      });
      assert.equal(tools?.[0].name, "lookup");
    }
    for (const type of ["future", 1, false, {}]) {
      assert.throws(() => anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        tools: [{ type, name: "lookup", input_schema: {} }],
      }), ChatGPTOAuthInvalidRequestError);
    }
  });

  it("tool_choice auto", () => {
    const { toolChoice, parallelToolCalls } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      toolChoice: { type: "auto" },
    });
    assert.equal(toolChoice, "auto");
    assert.equal(parallelToolCalls, undefined);
  });

  it("maps disable_parallel_tool_use to the provider parallel flag", () => {
    for (const [disableParallel, expected] of [[true, false], [false, true]] as const) {
      const { parallelToolCalls } = anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        toolChoice: {
          type: "auto",
          disable_parallel_tool_use: disableParallel,
        },
      });
      assert.equal(parallelToolCalls, expected);
    }
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

  it("rejects Anthropic hosted web_search tools", () => {
    for (const tool of [
      { type: "web_search", name: "web_search" },
      { type: "web_search_20250305", name: "web_search", blocked_domains: ["example.com"] },
      {
        type: "web_search_20260209",
        name: "web_search",
        allowed_domains: ["example.com"],
        max_uses: 8,
        strict: false,
        user_location: { type: "approximate", country: "US" },
      },
    ]) {
      assert.throws(() => anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        tools: [tool],
      }), /cannot be represented losslessly/);
    }
  });

  it("rejects unknown Anthropic web_search tool versions", () => {
    for (const type of ["web_search_20240101", "web_search_future"]) {
      assert.throws(() => anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        tools: [{ type, name: "web_search" }],
      }), ChatGPTOAuthInvalidRequestError);
    }
  });

  it("rejects web_search tool_choice", () => {
    assert.throws(() => anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      toolChoice: { type: "tool", name: "web_search" },
    }), /cannot be represented losslessly/);
  });

  it("tool_choice none", () => {
    const { toolChoice } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      toolChoice: { type: "none" },
    });
    assert.equal(toolChoice, "none");
    assert.throws(() => anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      toolChoice: { type: "none", disable_parallel_tool_use: false },
    }), ChatGPTOAuthInvalidRequestError);
  });

  it("thinking enabled", () => {
    const { reasoningEffort } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      thinking: { type: "enabled", budget_tokens: 4096 },
    });
    assert.equal(reasoningEffort, "high");
    const withDisplay = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      maxTokens: 4096,
      thinking: { type: "enabled", budget_tokens: 1024, display: "omitted" },
    });
    assert.equal(withDisplay.reasoningEffort, "high");
  });

  it("thinking adaptive", () => {
    const { reasoningEffort } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      thinking: { type: "adaptive", display: "omitted" },
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
        thinking: thinkingType === "enabled"
          ? { type: thinkingType, budget_tokens: 4096 }
          : { type: thinkingType },
        outputConfig: { effort: "max" },
      });
      assert.equal(reasoningEffort, "max");
    }
  });

  it("rejects malformed thinking controls instead of repairing them", () => {
    for (const thinking of [
      { type: "enabled" },
      { type: "enabled", budget_tokens: 0 },
      { type: "enabled", budget_tokens: 1023 },
      { type: "enabled", budget_tokens: 1024, display: "summarized" },
      { type: "adaptive", budget_tokens: 4096 },
      { type: "adaptive", display: "visible" },
      { type: "disabled", budget_tokens: 4096 },
    ]) {
      assert.throws(() => anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        thinking,
      }), ChatGPTOAuthInvalidRequestError);
    }
  });

  it("requires enabled thinking budget below max_tokens", () => {
    for (const budgetTokens of [2048, 4096]) {
      assert.throws(() => anthropicRequestToInternal({
        model: "test",
        messages: [{ role: "user", content: "hi" }],
        maxTokens: 2048,
        thinking: { type: "enabled", budget_tokens: budgetTokens },
      }), ChatGPTOAuthInvalidRequestError);
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
          name: "my schema!",
        },
      },
      {
        outputFormat: {
          type: "json_schema",
          schema: { type: "object" },
          name: "schéma",
        },
      },
      {
        outputFormat: {
          type: "json_schema",
          schema: { type: "object" },
          name: "x".repeat(65),
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
          description: null,
        },
      },
      {
        outputFormat: {
          type: "json_schema",
          schema: { type: "object" },
          name: null,
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
          format: { type: "json_schema", name: "nested", schema: { type: "object" } },
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
        name: "my_schema",
        description: "Answer envelope",
        schema: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
        strict: false,
      },
    });
    assert.deepEqual(text, {
      format: {
        type: "json_schema",
        name: "my_schema",
        description: "Answer envelope",
        schema: { type: "object", properties: { answer: { type: "string" } }, required: ["answer"] },
        strict: false,
      },
    });
  });

  it("uses the pinned Codex schema name when Anthropic omits name", () => {
    const { text } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      outputFormat: { type: "json_schema", schema: { type: "object" } },
    });
    assert.deepEqual(text, {
      format: {
        type: "json_schema",
        name: "codex_output_schema",
        schema: { type: "object" },
      },
    });
  });

  it("omits nullable json_schema strict", () => {
    const { text } = anthropicRequestToInternal({
      model: "test",
      messages: [{ role: "user", content: "hi" }],
      outputFormat: {
        type: "json_schema",
        schema: { type: "object" },
        strict: null,
      },
    });
    assert.deepEqual(text, {
      format: { type: "json_schema", name: "codex_output_schema", schema: { type: "object" } },
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

  it("rejects nested fields and duplicate identities that would otherwise be dropped", () => {
    const invalidRequests = [
      { messages: [{ role: "user", content: "hi", name: "ignored" }] },
      { messages: [{ role: "user", content: [{ type: "text", text: "hi", future: true }] }] },
      {
        messages: [{
          role: "user",
          content: [{
            type: "image",
            source: { type: "url", url: "https://example.com/a.png", future: true },
          }],
        }],
      },
      {
        messages: [{
          role: "assistant",
          content: [
            { type: "tool_use", id: "call-1", name: "lookup", input: {} },
            { type: "tool_use", id: "call-1", name: "lookup", input: {} },
          ],
        }],
      },
      {
        messages: [{ role: "user", content: "hi" }],
        tools: [
          { name: "lookup", input_schema: {} },
          { name: "lookup", input_schema: {} },
        ],
      },
      {
        messages: [{ role: "user", content: "hi" }],
        toolChoice: { type: "auto", disable_parallel_tool_use: null },
      },
    ];
    for (const request of invalidRequests) {
      assert.throws(
        () => anthropicRequestToInternal({ model: "test", ...request }),
        ChatGPTOAuthInvalidRequestError,
      );
    }
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
    assert.equal(content[0].citations, null);
    assert.equal(result.container, null);
    assert.equal(result.context_management, null);
    const usage = result.usage as Record<string, unknown>;
    assert.deepEqual(usage, {
      cache_creation: null,
      cache_creation_input_tokens: null,
      cache_read_input_tokens: 0,
      inference_geo: null,
      input_tokens: 10,
      iterations: null,
      output_tokens: 5,
      server_tool_use: null,
      service_tier: null,
      speed: null,
    });
  });

  it("rejects a response without authoritative usage", () => {
    assert.throws(() => internalResponseToAnthropic(
      makeResponse({ content: "Hello", usage: null }),
      "test-model",
      "msg_no_usage",
    ), /authoritative usage/);
  });

  it("rejects a null upstream finish reason", () => {
    assert.throws(() => internalResponseToAnthropic(
      makeResponse({ content: "pending", finish_reason: null, usage: makeUsage() }),
      "test-model",
      "msg_pending",
    ), /non-null finish_reason/);
  });

  it("does not synthesize usage from raw event extensions", () => {
    assert.throws(() => internalResponseToAnthropic(
      makeResponse({
        content: "Hello",
        usage: null,
        raw: {
          events: [{
            type: "finish",
            usage: { server_tool_use: { web_search_requests: 1, web_fetch_requests: 0 } },
          }],
        },
      }),
      "test-model",
      "msg_no_usage",
    ), /authoritative usage/);
  });

  it("rejects malformed present raw events instead of treating them as absent", () => {
    for (const raw of [
      {},
      { events: null },
      { events: [null] },
      { events: [{ type: "finish", usage: "unknown" }] },
    ]) {
      assert.throws(
        () => internalResponseToAnthropic(
          makeResponse({ content: "Hello", usage: makeUsage(), raw }),
          "test-model",
          "msg_bad_raw",
        ),
        ChatGPTOAuthProtocolError,
      );
    }
  });

  it("tool_use response", () => {
    const tc: ToolCall = { id: "tc-1", name: "get_weather", arguments: '{"city":"Seoul"}' };
    const resp = makeResponse({
      tool_calls: [tc],
      finish_reason: "tool_calls",
      usage: makeUsage({ prompt_tokens: 20, completion_tokens: 10 }),
    });
    const result = internalResponseToAnthropic(resp, "test-model", "msg_123");
    assert.equal(result.stop_reason, "tool_use");
    const content = result.content as Record<string, unknown>[];
    assert.equal(content.length, 1);
    assert.equal(content[0].type, "tool_use");
    assert.equal(content[0].name, "get_weather");
    assert.deepEqual(content[0].caller, { type: "direct" });

    const replayed = anthropicRequestToInternal({
      model: "test-model",
      messages: [{ role: "assistant", content }],
    });
    assert.deepEqual(replayed.messages[0].tool_calls, resp.tool_calls);
  });

  it("round-trips an empty tool_use id through the following tool_result", () => {
    const tc: ToolCall = { id: "", name: "lookup", arguments: "{}" };
    const resp = makeResponse({
      tool_calls: [tc],
      finish_reason: "tool_calls",
      usage: makeUsage({ prompt_tokens: 1, completion_tokens: 1 }),
    });
    const result = internalResponseToAnthropic(resp, "test-model", "msg_empty_call");
    const content = result.content as Record<string, unknown>[];

    const replayed = anthropicRequestToInternal({
      model: "test-model",
      messages: [
        { role: "assistant", content },
        { role: "user", content: [{ type: "tool_result", tool_use_id: "", content: "done" }] },
      ],
    });

    assert.deepEqual(replayed.messages[0].tool_calls, [tc]);
    assert.equal(replayed.messages[1].role, MessageRole.TOOL);
    assert.equal(replayed.messages[1].tool_call_id, "");
    assert.equal(replayed.messages[1].content, "done");
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
    assert.equal(content.length, 1);
    assert.equal(content[0].type, "text");
    assert.equal(content[0].text, "42");
    assert.equal(result.codex_reasoning, "Let me think about this...");
    assert.equal(Object.hasOwn(content[0], "thinking"), false);
    assert.equal(Object.hasOwn(content[0], "signature"), false);
  });

  it("represents an empty response with an empty content array", () => {
    const resp = makeResponse({
      content: "",
      finish_reason: "stop",
      usage: makeUsage(),
    });
    const result = internalResponseToAnthropic(resp, "test-model", "msg_123");
    assert.deepEqual(result.content, []);
    assert.equal(result.stop_reason, "end_turn");
    const replayed = anthropicRequestToInternal({
      model: "test-model",
      messages: [{ role: "assistant", content: result.content }],
    });
    assert.equal(replayed.messages.length, 1);
    assert.equal(replayed.messages[0].role, MessageRole.ASSISTANT);
    assert.equal(replayed.messages[0].content, "");
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

  it("uses null cache counters when internal usage omits them", () => {
    const result = internalResponseToAnthropic(makeResponse({
      content: "hi",
      usage: {
        prompt_tokens: 2,
        completion_tokens: 1,
        total_tokens: 3,
      },
    }), "m", "msg_1");
    const usage = result.usage as Record<string, unknown>;
    assert.equal(usage.cache_read_input_tokens, null);
    assert.equal(usage.cache_creation_input_tokens, null);
  });

  it("rejects web_search provider output", () => {
    const resp = makeResponse({
      content: "Final answer",
      usage: makeUsage(),
      raw: {
        events: [{
          type: "web_search_call",
          id: "srvtoolu_ws1",
          input: { query: "latest news" },
          content: [{ type: "web_search_result", url: "https://example.com", title: "Example" }],
        }],
      },
    });
    assert.throws(
      () => internalResponseToAnthropic(resp, "m", "msg_1"),
      /cannot be represented losslessly/,
    );
  });

  it("rejects malformed actual non-stream server tool usage", () => {
    for (const serverToolUse of [
      { web_search_requests: "one" },
      { web_search_requests: 1, web_fetch_requests: 0, future_counter: 1 },
      { web_search_requests: 1 },
    ]) {
      const resp = makeResponse({
        content: "Final answer",
        usage: makeUsage(),
        raw: {
          events: [{
            type: "finish",
            usage: { server_tool_use: serverToolUse },
          }],
        },
      });
      assert.throws(
        () => internalResponseToAnthropic(resp, "m", "msg_1"),
        ChatGPTOAuthProtocolError,
      );
    }
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
      { type: "finish", finish_reason: "stop", usage: makeStreamUsage({ output_tokens: 5, total_tokens: 5 }) },
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
      { type: "finish", finish_reason: "stop", usage: makeStreamUsage() },
    ];
    const result = await collectStreamEvents(events);
    const blockStarts = result.filter((e) => e.type === "content_block_start");
    assert.equal(blockStarts.length, 1);
    assert.equal((blockStarts[0].content_block as Record<string, unknown>).type, "text");
    const reasoningEvents = result.filter((e) => e.type === "codex_reasoning_delta");
    assert.deepEqual(reasoningEvents, [{ type: "codex_reasoning_delta", delta: "thinking..." }]);
    assert.equal(result.some((event) => event.type === "signature_delta"), false);
  });

  it("represents a reasoning-only stream without content blocks", async () => {
    const events = [
      { type: "reasoning_delta", text: "thinking..." },
      { type: "finish", finish_reason: "stop", usage: makeStreamUsage() },
    ];
    const result = await collectStreamEvents(events);

    assert.equal(result.some((event) => event.type === "content_block_start"), false);
    assert.deepEqual(
      result.filter((event) => event.type === "codex_reasoning_delta"),
      [{ type: "codex_reasoning_delta", delta: "thinking..." }],
    );
    assert.equal(result.at(-1)?.type, "message_stop");
  });

  it("tool call stream", async () => {
    const events = [
      { type: "tool_call", id: "tc-1", name: "get_weather", arguments: '{"city":"Seoul"}' },
      { type: "finish", finish_reason: "tool_calls", usage: makeStreamUsage() },
    ];
    const result = await collectStreamEvents(events);
    const blockStarts = result.filter((e) => e.type === "content_block_start");
    assert.equal(blockStarts.length, 1);
    assert.equal((blockStarts[0].content_block as Record<string, unknown>).type, "tool_use");
    assert.equal((blockStarts[0].content_block as Record<string, unknown>).name, "get_weather");
    assert.deepEqual(
      (blockStarts[0].content_block as Record<string, unknown>).caller,
      { type: "direct" },
    );

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
      { type: "tool_call", id: "tc-1", name: "search", arguments: '{"q":"test"}' },
      { type: "finish", finish_reason: "tool_calls", usage: makeStreamUsage() },
    ];
    const result = await collectStreamEvents(events);
    const blockStarts = result.filter((e) => e.type === "content_block_start");
    assert.equal(blockStarts.length, 2);
    assert.equal((blockStarts[0].content_block as Record<string, unknown>).type, "text");
    assert.equal((blockStarts[1].content_block as Record<string, unknown>).type, "tool_use");
  });

  it("represents a finish-only empty stream without content blocks", async () => {
    const result = await collectStreamEvents([
      { type: "finish", finish_reason: "stop", usage: makeStreamUsage() },
    ]);

    assert.equal(result.some((event) => event.type === "content_block_start"), false);
    assert.deepEqual(
      result.map((event) => event.type),
      ["message_start", "message_delta", "message_stop"],
    );
  });

  it("preserves empty content and reasoning deltas", async () => {
    const result = await collectStreamEvents([
      { type: "reasoning_delta", text: "" },
      { type: "content", text: "" },
      { type: "finish", finish_reason: "stop", usage: makeStreamUsage() },
    ]);
    const reasoning = result.find((event) => event.type === "codex_reasoning_delta")!;
    assert.equal(reasoning.delta, "");
    const textDelta = result.find((event) => event.type === "content_block_delta")!;
    assert.deepEqual(textDelta.delta, { type: "text_delta", text: "" });
  });

  it("rejects legacy provider finish reasons without reflecting them", async () => {
    const legacyReason = "legacy-secret-reason";
    await assert.rejects(
      () => collectStreamEvents([
        { type: "finish", finish_reason: legacyReason, usage: makeStreamUsage() },
      ]),
      (error: unknown) => error instanceof ChatGPTOAuthProtocolError
        && !error.message.includes(legacyReason),
    );
    assert.throws(
      () => internalResponseToAnthropic(
        makeResponse({ finish_reason: legacyReason as never }),
        "test-model",
        "msg_legacy",
      ),
      (error: unknown) => error instanceof ChatGPTOAuthProtocolError
        && !error.message.includes(legacyReason),
    );
  });

  it("rejects unsupported provider events instead of silently dropping them", async () => {
    const credentialSentinel = "access_token=NORMALIZED_PROVIDER_SECRET";
    for (const events of [
      [{ type: credentialSentinel, value: true }],
      [{
        type: "finish",
        finish_reason: "stop",
        usage: {
          ...makeStreamUsage(),
          cache_creation: { [credentialSentinel]: 1 },
        },
      }],
    ]) {
      await assert.rejects(
        () => collectStreamEvents(events),
        (error) => error instanceof ChatGPTOAuthProtocolError
          && !error.message.includes(credentialSentinel),
      );
    }
  });

  it("message_delta stop reason", async () => {
    const events = [
      { type: "content", text: "hi" },
      { type: "finish", finish_reason: "stop", usage: makeStreamUsage({ output_tokens: 3, total_tokens: 3 }) },
    ];
    const result = await collectStreamEvents(events);
    const msgDelta = result.find((e) => e.type === "message_delta")!;
    assert.deepEqual(msgDelta.delta, {
      container: null,
      stop_reason: "end_turn",
      stop_sequence: null,
    });
    assert.deepEqual(msgDelta.usage, {
      cache_creation_input_tokens: null,
      cache_read_input_tokens: 0,
      input_tokens: 0,
      iterations: null,
      output_tokens: 3,
      server_tool_use: null,
    });
    assert.equal(msgDelta.context_management, null);
  });

  it("rejects a null stream finish reason", async () => {
    await assert.rejects(() => collectStreamEvents([
      { type: "content", text: "pending" },
      { type: "finish", finish_reason: null, usage: makeStreamUsage() },
    ]), /non-null finish_reason/);
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
          total_tokens: 130,
          input_tokens_details: {
            cached_tokens: 13,
            cache_write_tokens: 11,
          },
          cache_creation: { ephemeral_5m_input_tokens: 11, ephemeral_1h_input_tokens: 0 },
          server_tool_use: { web_search_requests: 2, web_fetch_requests: 1 },
        },
      },
    ];
    const result = await collectStreamEvents(events);
    const msgStart = result.find((e) => e.type === "message_start")!;
    assert.equal(
      Object.hasOwn(msgStart.message as Record<string, unknown>, "usage"),
      false,
    );
    assert.equal((msgStart.message as Record<string, unknown>).container, null);
    assert.equal((msgStart.message as Record<string, unknown>).context_management, null);

    const msgDelta = result.find((e) => e.type === "message_delta")!;
    assert.deepEqual(msgDelta.usage, {
      cache_creation_input_tokens: 11,
      cache_read_input_tokens: 13,
      input_tokens: 123,
      iterations: null,
      output_tokens: 7,
      server_tool_use: { web_search_requests: 2, web_fetch_requests: 1 },
    });
  });

  it("rejects absent stream usage and uses null unknown cache counters", async () => {
    await assert.rejects(() => collectStreamEvents([
      { type: "content", text: "hi" },
      { type: "finish", finish_reason: "stop" },
    ]), /authoritative usage/);

    const withoutDetails = await collectStreamEvents([
      { type: "content", text: "hi" },
      {
        type: "finish",
        finish_reason: "stop",
        usage: { input_tokens: 2, output_tokens: 1, total_tokens: 3 },
      },
    ]);
    const usage = withoutDetails.find((event) => event.type === "message_delta")!
      .usage as Record<string, unknown>;
    assert.deepEqual(usage, {
      cache_creation_input_tokens: null,
      cache_read_input_tokens: null,
      input_tokens: 2,
      iterations: null,
      output_tokens: 1,
      server_tool_use: null,
    });
  });

  it("rejects malformed actual usage instead of copying it into Anthropic events", async () => {
    for (const usage of [
      makeStreamUsage({ total_tokens: 99 }),
      { ...makeStreamUsage(), prompt_tokens: 0 },
      { ...makeStreamUsage(), completion_tokens: 0 },
      { ...makeStreamUsage(), prompt_tokens_details: { cached_tokens: 0 } },
      { ...makeStreamUsage(), server_tool_use: { web_search_requests: "one", web_fetch_requests: 0 } },
      { ...makeStreamUsage(), server_tool_use: { web_search_requests: 1, web_fetch_requests: 0, future_counter: 1 } },
      { ...makeStreamUsage(), server_tool_use: { web_search_requests: 1 } },
      { ...makeStreamUsage(), cache_creation: "invalid" },
      { ...makeStreamUsage(), cache_creation: { future_counter: 1 } },
      { ...makeStreamUsage(), service_tier: 1 },
    ]) {
      await assert.rejects(
        () => collectStreamEvents([
          { type: "content", text: "hi" },
          { type: "finish", finish_reason: "stop", usage },
        ]),
        ChatGPTOAuthProtocolError,
      );
    }
  });

  it("multiple tool calls", async () => {
    const events = [
      { type: "tool_call", id: "tc-1", name: "tool_a", arguments: '{"a":1}' },
      { type: "tool_call", id: "tc-2", name: "tool_b", arguments: '{"b":2}' },
      { type: "finish", finish_reason: "tool_calls", usage: makeStreamUsage() },
    ];
    const result = await collectStreamEvents(events);
    const blockStarts = result.filter((e) => e.type === "content_block_start");
    assert.equal(blockStarts.length, 2);
    assert.equal((blockStarts[0].content_block as Record<string, unknown>).name, "tool_a");
    assert.equal((blockStarts[1].content_block as Record<string, unknown>).name, "tool_b");
    assert.equal(blockStarts[0].index, 0);
    assert.equal(blockStarts[1].index, 1);
  });

  it("rejects web_search_call stream output", async () => {
    const events = [
      {
        type: "web_search_call",
        id: "srvtoolu_ws1",
        input: { query: "current time" },
        content: [{ type: "web_search_result", url: "https://example.com", title: "Example" }],
      },
      { type: "content", text: "It is noon." },
      { type: "finish", finish_reason: "stop", usage: makeStreamUsage() },
    ];
    await assert.rejects(
      () => collectStreamEvents(events),
      /cannot be represented losslessly/,
    );
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
