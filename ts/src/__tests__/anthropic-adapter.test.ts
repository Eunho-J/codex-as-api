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
    assert.deepEqual(toolChoice, { type: "function", function: { name: "get_weather" } });
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
    assert.equal(reasoningEffort, null);
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
      usage: makeUsage({ prompt_tokens: 100, completion_tokens: 10, cached_tokens: 50 }),
    });
    const result = internalResponseToAnthropic(resp, "m", "msg_1");
    const usage = result.usage as Record<string, unknown>;
    assert.equal(usage.cache_read_input_tokens, 50);
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
      { type: "finish", finish_reason: "stop" },
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
