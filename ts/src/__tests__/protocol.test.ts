import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import {
  normalizeStreamContent,
  responseFailureMessage,
  reasoningFromResponseItems,
} from "../protocol.js";

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

  it("skips non-text array items", () => {
    const content = [
      { text: "hello" },
      { image: "data:..." },
      { text: " world" },
    ];
    assert.equal(normalizeStreamContent(content), "hello world");
  });

  it("stringifies other types", () => {
    assert.equal(normalizeStreamContent(42), "42");
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

  it("extracts summary string", () => {
    const items = [
      { type: "reasoning", summary: "thought about it" },
    ];
    assert.equal(
      reasoningFromResponseItems(items),
      "thought about it",
    );
  });

  it("extracts content string", () => {
    const items = [
      { type: "reasoning", content: "deep thought" },
    ];
    assert.equal(
      reasoningFromResponseItems(items),
      "deep thought",
    );
  });

  it("extracts from summary array with text objects", () => {
    const items = [
      {
        type: "reasoning",
        summary: [
          { text: "step 1" },
          { text: " step 2" },
        ],
      },
    ];
    assert.equal(
      reasoningFromResponseItems(items),
      "step 1 step 2",
    );
  });

  it("extracts from summary array with strings", () => {
    const items = [
      {
        type: "reasoning",
        summary: ["part a", "part b"],
      },
    ];
    assert.equal(
      reasoningFromResponseItems(items),
      "part apart b",
    );
  });

  it("concatenates multiple reasoning items", () => {
    const items = [
      { type: "reasoning", summary: "first" },
      { type: "message", content: "ignore" },
      { type: "reasoning", content: "second" },
    ];
    assert.equal(
      reasoningFromResponseItems(items),
      "firstsecond",
    );
  });

  it("skips empty values", () => {
    const items = [
      { type: "reasoning", summary: "" },
      { type: "reasoning", content: "real" },
    ];
    assert.equal(reasoningFromResponseItems(items), "real");
  });
});
