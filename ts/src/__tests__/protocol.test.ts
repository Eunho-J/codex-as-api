import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import {
  normalizeStreamContent,
  responseFailureMessage,
  reasoningFromResponseItems,
} from "../protocol.js";
import { ChatGPTOAuthProtocolError } from "../auth.js";

describe("normalizeStreamContent", () => {
  it("returns empty string for null", () => {
    assert.equal(normalizeStreamContent(null), "");
  });

  it("returns empty string for undefined", () => {
    assert.equal(normalizeStreamContent(undefined), "");
  });

  it("passes through strings", () => {
    assert.equal(normalizeStreamContent("hello"), "hello");
  });

  it("joins array text parts", () => {
    const content = [
      { text: "hello " },
      { text: "world" },
    ];
    assert.equal(normalizeStreamContent(content), "hello world");
  });

  it("rejects non-text array items", () => {
    const content = [
      { text: "hello" },
      { image: "data:..." },
      { text: " world" },
    ];
    assert.throws(
      () => normalizeStreamContent(content),
      (error) => error instanceof ChatGPTOAuthProtocolError,
    );
  });

  it("rejects other types", () => {
    assert.throws(
      () => normalizeStreamContent(42),
      (error) => error instanceof ChatGPTOAuthProtocolError,
    );
  });

  it("handles empty array", () => {
    assert.equal(normalizeStreamContent([]), "");
  });
});

describe("responseFailureMessage", () => {
  it("extracts error message", () => {
    const event = {
      error: { message: "rate limited", code: "429" },
    };
    const msg = responseFailureMessage(event, "failed");
    assert.ok(msg.includes("rate limited"));
    assert.ok(msg.includes("failed"));
  });

  it("extracts error from nested response", () => {
    const event = {
      response: { error: { message: "server error" } },
    };
    const msg = responseFailureMessage(event, "incomplete");
    assert.ok(msg.includes("server error"));
    assert.ok(msg.includes("incomplete"));
  });

  it("handles string error", () => {
    const event = { error: "something broke" };
    const msg = responseFailureMessage(event, "failed");
    assert.ok(msg.includes("something broke"));
  });

  it("extracts incomplete_details reason", () => {
    const event = {
      incomplete_details: { reason: "max_tokens" },
    };
    const msg = responseFailureMessage(event, "incomplete");
    assert.ok(msg.includes("max_tokens"));
  });

  it("falls back to JSON serialization", () => {
    const event = { some: "data" };
    const msg = responseFailureMessage(event, "failed");
    assert.ok(msg.includes("some"));
    assert.ok(msg.includes("data"));
  });

  it("combines error and incomplete_details", () => {
    const event = {
      error: { message: "err" },
      incomplete_details: { reason: "length" },
    };
    const msg = responseFailureMessage(event, "failed");
    assert.ok(msg.includes("err"));
    assert.ok(msg.includes("length"));
  });
});

describe("reasoningFromResponseItems", () => {
  it("returns empty string for no reasoning items", () => {
    const items = [{ type: "message", content: "hello" }];
    assert.equal(reasoningFromResponseItems(items), "");
  });

  it("extracts official tagged summary parts", () => {
    const items = [
      {
        type: "reasoning",
        summary: [{ type: "summary_text", text: "thought about it" }],
      },
    ];
    assert.equal(
      reasoningFromResponseItems(items),
      "thought about it",
    );
  });

  it("extracts both pinned tagged reasoning content variants", () => {
    for (const type of ["reasoning_text", "text"]) {
      const items = [
        {
          type: "reasoning",
          summary: [],
          content: [{ type, text: "deep thought" }],
        },
      ];
      assert.equal(
        reasoningFromResponseItems(items),
        "deep thought",
      );
    }
  });

  it("extracts from summary array with text objects", () => {
    const items = [
      {
        type: "reasoning",
        summary: [
          { type: "summary_text", text: "step 1" },
          { type: "summary_text", text: " step 2" },
        ],
      },
    ];
    assert.equal(
      reasoningFromResponseItems(items),
      "step 1 step 2",
    );
  });

  it("concatenates multiple reasoning items", () => {
    const items = [
      {
        type: "reasoning",
        summary: [{ type: "summary_text", text: "first" }],
      },
      { type: "message", content: "ignore" },
      {
        type: "reasoning",
        summary: [],
        content: [{ type: "reasoning_text", text: "second" }],
      },
    ];
    assert.equal(
      reasoningFromResponseItems(items),
      "firstsecond",
    );
  });

  it("skips empty values", () => {
    const items = [
      {
        type: "reasoning",
        summary: [{ type: "summary_text", text: "" }],
      },
      {
        type: "reasoning",
        summary: [],
        content: [{ type: "reasoning_text", text: "real" }],
      },
    ];
    assert.equal(reasoningFromResponseItems(items), "real");
  });

  it("rejects stale untagged reasoning shapes", () => {
    for (const item of [
      { type: "reasoning", summary: "direct" },
      { type: "reasoning", content: "direct" },
      { type: "reasoning", summary: ["bare"] },
      { type: "reasoning", summary: [{ text: "untyped" }] },
      { type: "reasoning", summary: [{ type: "reasoning_text", text: "wrong" }] },
      { type: "reasoning", content: [{ type: "summary_text", text: "wrong" }] },
    ]) {
      assert.throws(
        () => reasoningFromResponseItems([item]),
        ChatGPTOAuthProtocolError,
      );
    }
  });

  it("requires the upstream reasoning summary array", () => {
    for (const summary of [undefined, null, "", {}]) {
      assert.throws(
        () => reasoningFromResponseItems([{
          type: "reasoning",
          summary,
          content: null,
        }]),
        (error) => error instanceof ChatGPTOAuthProtocolError,
      );
    }
  });
});
