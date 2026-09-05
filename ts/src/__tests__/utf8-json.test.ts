import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import { parseJsonStrict } from "../utf8-json.js";

describe("parseJsonStrict", () => {
  it("accepts finite numbers, safe integers, and paired Unicode surrogates", () => {
    assert.deepEqual(
      parseJsonStrict('{"minimum":-9007199254740991,"ratio":0.125,"emoji":"\\ud83d\\ude80"}'),
      { minimum: Number.MIN_SAFE_INTEGER, ratio: 0.125, emoji: "🚀" },
    );
  });

  it("rejects numeric values JavaScript cannot preserve", () => {
    for (const raw of [
      '{"value":1e400}',
      '{"value":9007199254740992}',
      '{"value":9007199254740993}',
      '{"value":-9007199254740992}',
    ]) {
      assert.throws(() => parseJsonStrict(raw), SyntaxError);
    }
  });

  it("rejects unpaired surrogates in values and object keys", () => {
    for (const raw of [
      '{"value":"\\ud800"}',
      '{"value":"\\udfff"}',
      '{"\\ud800":true}',
    ]) {
      assert.throws(() => parseJsonStrict(raw), SyntaxError);
    }
  });
});
