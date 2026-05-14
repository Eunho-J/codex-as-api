import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import type { Server } from "node:http";
import { createApp } from "../server.js";
import { ChatGPTOAuthError } from "../auth.js";

async function withServer(
  provider: Record<string, unknown>,
  fn: (baseUrl: string) => Promise<void>,
): Promise<void> {
  const app = createApp({ provider: provider as never });
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

describe("server error handling", () => {
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

describe("Anthropic compatibility helper routes", () => {
  it("returns a count_tokens estimate with context metadata", async () => {
    await withServer({}, async (baseUrl) => {
      const res = await fetch(`${baseUrl}/v1/messages/count_tokens`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          model: "claude-sonnet-4-5",
          max_tokens: 1024,
          system: "You are helpful.",
          messages: [{ role: "user", content: "hello" }],
        }),
      });

      assert.equal(res.status, 200);
      const body = await res.json() as {
        input_tokens: number;
        context_window: number;
        auto_compact_token_limit: number;
      };
      assert.ok(body.input_tokens > 0);
      assert.ok(body.context_window >= body.auto_compact_token_limit);
    });
  });

  it("accepts Anthropic shaped compact requests on /v1/messages/compact", async () => {
    const provider = {
      async compactMessages(messages: Array<{ content: string }>, opts: { model?: string; reasoningEffort?: string }) {
        assert.equal(opts.model, "gpt-5.5");
        assert.equal(opts.reasoningEffort, "high");
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
          messages: [{ role: "user", content: "hello" }],
        }),
      });

      assert.equal(res.status, 200);
      assert.deepEqual(await res.json(), { checkpoint: "checkpoint" });
    });
  });
});
