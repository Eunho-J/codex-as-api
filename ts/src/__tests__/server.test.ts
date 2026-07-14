import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import { createServer, type IncomingHttpHeaders, type Server } from "node:http";
import * as os from "node:os";
import * as path from "node:path";
import { createApp } from "../server.js";
import { ChatGPTOAuthError } from "../auth.js";
import { ChatGPTOAuthProvider, messagesToResponseItems } from "../provider.js";
import type { CodexConfig } from "../codex-config.js";
import { MessageRole } from "../messages.js";
import type { ToolSchema } from "../messages.js";

const TEST_CONFIG: CodexConfig = {
  codexHome: "/test/codex-home",
  configPath: "/test/codex-home/config.toml",
};

async function withServer(
  provider: ChatGPTOAuthProvider | Record<string, unknown>,
  fn: (baseUrl: string) => Promise<void>,
  opts: { model?: string | null; codexConfig?: CodexConfig; authPath?: string } = {},
): Promise<void> {
  const app = createApp(opts.model === null
    ? {
        provider: provider as never,
        codexConfig: opts.codexConfig ?? TEST_CONFIG,
        authPath: opts.authPath,
      }
    : {
        provider: provider as never,
        model: opts.model ?? "gpt-5.5",
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
  path: string;
  headers: IncomingHttpHeaders;
  body: Record<string, unknown>;
}

async function withRecordingUpstream(
  fn: (baseUrl: string, requests: RecordedRequest[]) => Promise<void>,
): Promise<void> {
  const requests: RecordedRequest[] = [];
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
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      const body = JSON.parse(Buffer.concat(chunks).toString("utf8")) as Record<string, unknown>;
      requests.push({ path: req.url ?? "", headers: req.headers, body });
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
            result: "data:image/png;base64,RESULT",
          }
        : {
            type: "message",
            role: "assistant",
            content: [{ type: "output_text", text: "ok" }],
          };
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.write(`data: ${JSON.stringify({
        type: "response.output_item.done",
        item: outputItem,
      })}\n\n`);
      res.end(`data: ${JSON.stringify({
        type: "response.completed",
        response: {
          id: "response-1",
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
    await fn(`http://127.0.0.1:${address.port}`, requests);
  } finally {
    await new Promise<void>((resolve, reject) => {
      server.close((err) => (err ? reject(err) : resolve()));
    });
  }
}

describe("server error handling", () => {
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
      assert.equal(body.error.type, "chatgpt_oauth_error");
      assert.match(body.error.message, /stop is not supported/);
    });
    assert.equal(transportCalls, 0);
  });

  it("rejects an empty effort before opening an OpenAI stream", async () => {
    const provider = {
      async *chatStream() {
        throw new Error("provider must not be called");
      },
    };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          stream: true,
          reasoning_effort: "",
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 400);
      assert.equal(response.headers.get("content-type"), "application/json; charset=utf-8");
      const body = await response.json() as { error: { type: string } };
      assert.equal(body.error.type, "chatgpt_oauth_error");
    });
  });

  it("rejects an invalid Responses Lite mode or type before opening an OpenAI stream", async () => {
    const provider = new ChatGPTOAuthProvider({ model: "gpt-5.6-sol" });
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
        assert.deepEqual(await response.json(), {
          error: {
            message: "ChatGPTOAuthError: responses_lite must be one of: off, on, auto",
            type: "chatgpt_oauth_error",
          },
        });
      }
    }, { model: "gpt-5.6-sol" });
  });

  it("maps an unsupported Lite tool choice to a structured 400 before streaming", async () => {
    const provider = new ChatGPTOAuthProvider({ model: "gpt-5.6-sol" });
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
      assert.deepEqual(await response.json(), {
        error: {
          message: "ChatGPTOAuthError: Responses Lite requires tool_choice to be the exact string auto",
          type: "chatgpt_oauth_error",
        },
      });
    }, { model: "gpt-5.6-sol" });
  });

  it("ends OpenAI streams with an SSE error instead of sending JSON after headers", async () => {
    const provider = {
      async *chatStream() {
        yield { type: "content", text: "partial" };
        throw new ChatGPTOAuthError(
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
      assert.match(body, /data: \[DONE\]/);
    });
  });

  it("ends Anthropic streams with an SSE error instead of sending JSON after headers", async () => {
    const provider = {
      async *chatStream() {
        yield { type: "content", text: "partial" };
        throw new ChatGPTOAuthError(
          "OpenAI protocol response failed: Your input exceeds the context window of this model.",
        );
      },
    };

    await withServer(provider, async (baseUrl) => {
      const res = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          max_tokens: 1024,
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
            usage: { input_tokens: 1, output_tokens: 1 },
          };
        })();
      },
    };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-fable-5",
          max_tokens: 1024,
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

  it("rejects Anthropic hosted WebSearch in auto and on Lite modes before SSE headers", async () => {
    const previous = process.env.CODEX_AS_API_RESPONSES_LITE;
    try {
      for (const mode of ["auto", "on"]) {
        process.env.CODEX_AS_API_RESPONSES_LITE = mode;
        const provider = new ChatGPTOAuthProvider({ model: "gpt-5.6-sol" });
        await withServer(provider, async (baseUrl) => {
          const response = await fetch(`${baseUrl}/v1/messages`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "claude-sonnet-4-5",
              max_tokens: 1024,
              stream: true,
              system: "You are helpful.",
              tools: [{ type: "web_search_20250305", name: "web_search", max_uses: 1 }],
              messages: [{ role: "user", content: "hello" }],
            }),
          });

          assert.equal(response.status, 400);
          assert.equal(response.headers.get("content-type"), "application/json; charset=utf-8");
          const body = await response.json() as { error: { type: string } };
          assert.equal(body.error.type, "invalid_request_error");
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
        throw new ChatGPTOAuthError(
          "OpenAI protocol response failed: Your input exceeds the context window of this model.",
        );
      },
    };

    await withServer(provider, async (baseUrl) => {
      const res = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          max_tokens: 1024,
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

describe("OpenAI stream translation", () => {
  it("adds stable tool indexes and maps Responses usage fields", async () => {
    const provider = {
      chatStream() {
        return (async function* () {
          yield { type: "tool_call", id: "call-a", name: "first", arguments: { value: 1 } };
          yield { type: "tool_call", id: "call-b", name: "second", arguments: { value: 2 } };
          yield {
            type: "finish",
            finish_reason: "tool_calls",
            usage: { input_tokens: 10, output_tokens: 4, total_tokens: 14 },
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
});

describe("server model defaults", () => {
  it("uses the conservative context window for an unknown model", async () => {
    await withServer({}, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/health`);
      assert.equal(response.status, 200);
      const body = await response.json() as Record<string, unknown>;
      assert.equal(body.model, "provider/future-model");
      assert.equal(body.reasoning_effort, null);
      assert.equal(body.context_window, 200_000);
      assert.equal(body.auto_compact_token_limit, 160_000);
    }, { model: "provider/future-model" });
  });

  it("clamps an unknown model compact override to the resolved fallback context", async () => {
    await withServer({}, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/health`);
      assert.equal(response.status, 200);
      const body = await response.json() as Record<string, unknown>;
      assert.equal(body.context_window, 200_000);
      assert.equal(body.auto_compact_token_limit, 180_000);
    }, {
      model: "provider/future-model",
      codexConfig: { ...TEST_CONFIG, modelAutoCompactTokenLimit: 500_000 },
    });
  });

  it("clamps GPT-5.6 context and compact overrides to official model limits", async () => {
    await withServer({}, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/health`);
      assert.equal(response.status, 200);
      const body = await response.json() as Record<string, unknown>;
      assert.equal(body.context_window, 372_000);
      assert.equal(body.auto_compact_token_limit, 334_800);
    }, {
      model: "gpt-5.6-sol",
      codexConfig: {
        ...TEST_CONFIG,
        modelContextWindow: 500_000,
        modelAutoCompactTokenLimit: 450_000,
      },
    });
  });

  it("does not report healthy when configured reasoning effort is empty", async () => {
    await withServer({}, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/health`);
      assert.equal(response.status, 400);
      const body = await response.json() as { error: { message: string } };
      assert.match(body.error.message, /reasoning_effort must be a non-empty string/);
    }, {
      model: "gpt-5.6-sol",
      codexConfig: { ...TEST_CONFIG, modelReasoningEffort: "" },
    });
  });

  it("treats an empty CODEX_AS_API_MODEL as absent", async () => {
    const previous = process.env.CODEX_AS_API_MODEL;
    process.env.CODEX_AS_API_MODEL = "";
    try {
      await withServer({}, async (baseUrl) => {
        const response = await fetch(`${baseUrl}/health`);
        assert.equal(response.status, 200);
        const body = await response.json() as Record<string, unknown>;
        assert.equal(body.model, "gpt-5.6-sol");
        assert.equal(body.reasoning_effort, "low");
        assert.equal(body.context_window, 372_000);
      }, {
        model: null,
        codexConfig: { ...TEST_CONFIG, model: "gpt-5.6-sol" },
      });
    } finally {
      if (previous == null) delete process.env.CODEX_AS_API_MODEL;
      else process.env.CODEX_AS_API_MODEL = previous;
    }
  });
});

describe("Anthropic compatibility helper routes", () => {
  it("streams Claude Code WebSearch when disabled thinking overrides ambient effort", async () => {
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
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "claude-sonnet-4-5",
              max_tokens: 64,
              stream: true,
              system: "You are helpful.",
              tools: [{ type: "web_search_20250305", name: "web_search", max_uses: 1 }],
              thinking: { type: "disabled" },
              output_config: { effort: "high" },
              messages: [{ role: "user", content: "Search the web." }],
            }),
          });

          assert.equal(response.status, 200);
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
            stop_reason: "end_turn",
            stop_sequence: null,
          });

          assert.equal(requests.length, 1);
          assert.equal(requests[0].path, "/responses");
          const upstream = requests[0].body;
          assert.deepEqual(upstream.reasoning, { effort: "none" });
          assert.deepEqual(upstream.tools, [{
            type: "web_search",
            external_web_access: true,
          }]);
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

  it("wires the Claude Code 2.1.209 request shape to a known GPT model", async () => {
    const auth = writeAuthFile();
    try {
      await withRecordingUpstream(async (upstreamUrl, requests) => {
        const provider = new ChatGPTOAuthProvider({
          model: "gpt-5.5",
          baseUrl: upstreamUrl,
          authJsonPath: auth.authPath,
        });
        await withServer(provider, async (baseUrl) => {
          const response = await fetch(`${baseUrl}/v1/messages?beta=true`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.6",
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
                cache_control: { type: "ephemeral" },
              }],
              tools: [],
              metadata: { user_id: "claude-code-2.1.209" },
              max_tokens: 32_000,
              thinking: { type: "adaptive", display: "omitted" },
              context_management: {
                edits: [{ type: "clear_thinking_20251015", keep: "all" }],
              },
              output_config: { effort: "max" },
              speed: "fast",
              stream: true,
            }),
          });

          assert.equal(response.status, 200);
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
          assert.equal(responseMessage?.model, "gpt-5.6");

          assert.equal(requests.length, 1);
          const upstream = requests[0].body;
          assert.equal(upstream.model, "gpt-5.6-sol");
          assert.deepEqual(upstream.reasoning, {
            effort: "max",
            context: "all_turns",
          });
          assert.equal(upstream.service_tier, "priority");
          assert.equal(Object.hasOwn(upstream, "output_config"), false);
          assert.equal(Object.hasOwn(upstream, "context_management"), false);
          assert.equal(Object.hasOwn(upstream, "speed"), false);
        }, { model: "gpt-5.5", authPath: auth.authPath });
      });
    } finally {
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });

  it("uses the configured GPT fallback for built-in Claude model names", async () => {
    let providerModel: string | undefined;
    const provider = {
      async chat(_messages: unknown, opts: { model?: string }) {
        providerModel = opts.model;
        return {
          content: "done",
          tool_calls: [],
          finish_reason: "stop",
          usage: null,
          reasoning_content: null,
          raw: null,
        };
      },
    };

    await withServer(provider, async (baseUrl) => {
      const response = await fetch(`${baseUrl}/v1/messages`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-fable-5",
          max_tokens: 1024,
          messages: [{ role: "user", content: "hello" }],
        }),
      });
      assert.equal(response.status, 200);
      const body = await response.json() as Record<string, unknown>;
      assert.equal(body.model, "claude-fable-5");
      assert.equal(providerModel, "gpt-5.5");
    }, { model: "gpt-5.5" });
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
          usage: null,
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
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "gpt-5.6-sol",
            max_tokens: 1024,
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
            model: "gpt-5.6-sol",
            max_tokens: 1024,
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

  it("accepts only the exact no-op context_management shape", async () => {
    let chatCalls = 0;
    const provider = {
      async chat() {
        chatCalls += 1;
        return {
          content: "done",
          tool_calls: [],
          finish_reason: "stop",
          usage: null,
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
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "claude-fable-5",
            max_tokens: 1024,
            messages: [{ role: "user", content: "hello" }],
            context_management: accepted,
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
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "claude-fable-5",
              max_tokens: 1024,
              messages: [{ role: "user", content: "hello" }],
              context_management: contextManagement,
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
      { reasoning_effort: "low", output_config: { effort: "high" } },
      { tools: [{ name: "lookup", input_schema: {}, strict: true }] },
      { tools: [{ name: "lookup", input_schema: {}, defer_loading: true }] },
      { tools: [{ name: "lookup", input_schema: {}, eager_input_streaming: true }] },
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
            model: "gpt-5.6-sol",
            max_tokens: 1024,
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
      model: "gpt-5.6-sol",
      max_tokens: 1024,
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
          max_tokens: 1024,
          system: "You are helpful.",
          multi_agent: null,
          programmatic_tool_calling: null,
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
            max_tokens: 1024,
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

      const gpt = await count("gpt-5.6");
      assert.equal(gpt.context_window, 372_000);
      assert.equal(gpt.auto_compact_token_limit, 334_800);

      const claudeFallback = await count("claude-fable-5");
      assert.equal(claudeFallback.context_window, 272_000);
      assert.equal(claudeFallback.auto_compact_token_limit, 244_800);
    }, { model: "gpt-5.5" });
  });

  it("does not double-count the raw count_tokens payload", async () => {
    const content = "abcd".repeat(1_000);
    await withServer({}, async (baseUrl) => {
      const count = async (extra: Record<string, unknown>) => {
        const res = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "claude-sonnet-4-5",
            max_tokens: 1024,
            messages: [{ role: "user", content }],
            ...extra,
          }),
        });
        assert.equal(res.status, 200);
        return (await res.json() as { input_tokens: number }).input_tokens;
      };

      const plain = await count({});
      const withNonModelFields = await count({
        stream: false,
        metadata: { diagnostic: "x".repeat(4_000) },
      });
      assert.equal(plain, 1_012);
      assert.equal(withNonModelFields, plain);
    });
  });

  it("counts multilingual UTF-8 text with o200k_base", async () => {
    await withServer({}, async (baseUrl) => {
      const content = "hello 안녕 👋";
      const res = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          max_tokens: 1024,
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

  it("routes Anthropic compact requests by exact bundled model ID", async () => {
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
            max_tokens: 1024,
            messages: [{ role: "user", content: "history" }],
            context_management: {
              edits: [{ type: "clear_thinking_20251015", keep: "all" }],
            },
          };
          const known = await fetch(`${baseUrl}/v1/messages/compact`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              ...common,
              model: "gpt-5.6",
              thinking: { type: "adaptive", display: "omitted" },
              output_config: {
                effort: "high",
                format: {
                  type: "json_schema",
                  name: "compact schema!",
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

          const fallback = await fetch(`${baseUrl}/v1/messages/compact`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              ...common,
              model: "claude-fable-5",
              speed: "standard",
            }),
          });
          assert.equal(fallback.status, 200);

          const conflict = await fetch(`${baseUrl}/v1/messages/compact`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              ...common,
              model: "gpt-5.6",
              reasoning_effort: "low",
              output_config: { effort: "high" },
            }),
          });
          assert.equal(conflict.status, 400);
          const conflictBody = await conflict.json() as {
            error: { type: string };
          };
          assert.equal(conflictBody.error.type, "chatgpt_oauth_error");

          assert.equal(requests.length, 2);
          assert.deepEqual(requests.map((request) => request.path), [
            "/responses/compact",
            "/responses/compact",
          ]);
          assert.equal(requests[0].body.model, "gpt-5.6-sol");
          assert.equal(requests[0].body.service_tier, "priority");
          const text = requests[0].body.text as Record<string, unknown>;
          assert.deepEqual(text.format, {
            type: "json_schema",
            name: "compact_schema_",
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
                model: "gpt-5.6",
                ...fields,
              }),
            });
            assert.equal(invalid.status, 400);
            const error = await invalid.json() as {
              error: { type: string };
            };
            assert.equal(error.error.type, "chatgpt_oauth_error");
          }
          assert.equal(requests.length, 2);
        }, { model: "gpt-5.5", authPath: auth.authPath });
      });
    } finally {
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });


  it("accepts Anthropic shaped compact requests on /v1/messages/compact", async () => {
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
          description: "Lookup",
          parameters: { type: "object" },
        }]);
        assert.deepEqual(messages.map((m) => m.content), ["sys", "hello"]);
        return "checkpoint";
      },
    };

    await withServer(provider, async (baseUrl) => {
      const res = await fetch(`${baseUrl}/v1/messages/compact`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          max_tokens: 1024,
          system: "sys",
          thinking: { type: "enabled", budget_tokens: 1024 },
          responses_lite: false,
          prompt_cache_key: "anthropic-compact-cache",
          service_tier: "priority",
          text: { format: { type: "text" } },
          verbosity: "medium",
          tools: [{
            name: "lookup",
            description: "Lookup",
            input_schema: { type: "object" },
          }],
          messages: [{ role: "user", content: "hello" }],
        }),
      });

      assert.equal(res.status, 200);
      assert.deepEqual(await res.json(), { checkpoint: "checkpoint" });
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
          usage: null,
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
          body: JSON.stringify({ prompt: "inspect", images: [], responses_lite: false }),
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
          headers: { "content-type": "application/json" },
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
              model: "gpt-5.6",
              responses_lite: false,
              reasoning: { mode: "standard", context: "current_turn" },
              service_tier: "fast",
              verbosity: "high",
              text: { format: { type: "text" } },
              safety_identifier: null,
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
                },
              }],
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
          assert.equal(chatResponse.status, 200);
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
          });
          assert.equal(chatRequest.service_tier, "priority");
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
              safety_identifier: null,
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
          assert.deepEqual(requests[2].body.reasoning, { effort: "medium" });

          const imageResponse = await fetch(`${baseUrl}/v1/images/generations`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.6-sol",
              prompt: "Draw",
              responses_lite: false,
              reasoning: { mode: "standard" },
              reference_images: [{
                image_url: "data:image/png;base64,AAAA",
                detail: "high",
              }],
            }),
          });
          assert.equal(imageResponse.status, 200);
          const imageRequest = requests[3].body;
          assert.deepEqual(imageRequest.reasoning, {
            effort: "medium",
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
            context: "all_turns",
          });
          assert.equal(Object.hasOwn(inspectRequest, "safety_identifier"), false);

          const anthropicResponse = await fetch(`${baseUrl}/v1/messages`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              system: "You are helpful.",
              messages: [{ role: "user", content: "Hello" }],
              max_tokens: 64,
              responses_lite: false,
              reasoning: { effort: "high", mode: "standard" },
            }),
          });
          assert.equal(anthropicResponse.status, 200);
          const anthropicRequest = requests[5].body;
          assert.deepEqual(anthropicRequest.reasoning, {
            effort: "high",
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
          assert.equal(unsupportedTierBody.error.type, "chatgpt_oauth_error");
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
        assert.equal(body.error.type, "chatgpt_oauth_error");
      }
      assert.equal(compactCalls, 0);
    });
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
        assert.equal(error.error.type, "chatgpt_oauth_error");
      }

      for (const body of [
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
    const previousVersion = process.env.CODEX_AS_API_CODEX_CLI_VERSION;
    process.env.CODEX_AS_API_RESPONSES_LITE = "auto";
    process.env.CODEX_AS_API_CODEX_CLI_VERSION = "0.144.0";
    const codexConfig: CodexConfig = {
      codexHome: auth.directory,
      configPath: path.join(auth.directory, "config.toml"),
      model: "gpt-5.6-sol",
      modelReasoningEffort: "high",
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
          assert.equal(health.reasoning_effort, "high");
          assert.equal(health.context_window, 372_000);
          assert.equal(health.auto_compact_token_limit, 334_800);

          const chatResponse = await fetch(`${baseUrl}/v1/chat/completions`, {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({
              model: "gpt-5.6-sol",
              reasoning_effort: "ultra",
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
          assert.deepEqual(chatRequest.body.reasoning, { effort: "max", context: "all_turns" });
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
                  description: "Lookup",
                  parameters: { type: "object" },
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
            effort: "high",
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
                description: "Lookup",
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
          assert.equal(inspectRequest.body.model, "gpt-5.6-sol");
          assert.equal(inspectRequest.body.tool_choice, "auto");
          assert.deepEqual(inspectRequest.body.reasoning, { effort: "high", context: "all_turns" });
        }, {
          model: "gpt-5.6-sol",
          codexConfig,
          authPath: auth.authPath,
        });
      });
    } finally {
      if (previousLite == null) delete process.env.CODEX_AS_API_RESPONSES_LITE;
      else process.env.CODEX_AS_API_RESPONSES_LITE = previousLite;
      if (previousVersion == null) delete process.env.CODEX_AS_API_CODEX_CLI_VERSION;
      else process.env.CODEX_AS_API_CODEX_CLI_VERSION = previousVersion;
      fs.rmSync(auth.directory, { recursive: true, force: true });
    }
  });
});
