import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";
import {
  countO200kOrdinaryTokens,
  encodeO200kOrdinary,
} from "../o200k-tokenizer.js";

type ReferenceCase = { text: string; tokens: number[] };

const referenceCases = JSON.parse(readFileSync(fileURLToPath(new URL(
  "../../../tests/fixtures/o200k_base_encode_ordinary.json",
  import.meta.url,
)), "utf8")) as ReferenceCase[];

describe("o200k_base ordinary tokenizer", () => {
  it("matches the tiktoken 0.13.0 multilingual and code reference vectors", () => {
    for (const reference of referenceCases) {
      const actual = encodeO200kOrdinary(reference.text);
      assert.deepEqual(actual, reference.tokens, JSON.stringify(reference.text));
      assert.equal(countO200kOrdinaryTokens(reference.text), reference.tokens.length);
    }
  });

  it("treats special-token-shaped literals as ordinary text", () => {
    const reference = referenceCases.find((item) => item.text.includes("<|endoftext|>"));
    assert.ok(reference);
    assert.deepEqual(encodeO200kOrdinary(reference.text), reference.tokens);
    assert.ok(!reference.tokens.includes(199_999));
    assert.ok(!reference.tokens.includes(200_018));
  });

  it("uses the large-piece BPE path without quadratic behavior", () => {
    encodeO200kOrdinary("warm up");
    const text = "abcd".repeat(1_000);
    const startedAt = performance.now();
    const tokens = encodeO200kOrdinary(text);
    const elapsedMs = performance.now() - startedAt;

    assert.equal(Buffer.byteLength(text, "utf8"), 4_000);
    assert.equal(tokens.length, 1_000);
    assert.ok(elapsedMs < 1_000, `4,000-byte BPE took ${elapsedMs.toFixed(1)} ms`);
  });
});
