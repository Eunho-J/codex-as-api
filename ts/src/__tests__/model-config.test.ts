import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { TomlError } from "smol-toml";
import { loadCodexConfig } from "../codex-config.js";
import { capabilityForModel } from "../model-capabilities.js";

describe("Codex config", () => {
  it("treats only a missing config file as absent", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      fs.mkdirSync(path.join(codexHome, "config.toml"));
      assert.throws(() => loadCodexConfig(codexHome));
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("rejects invalid UTF-8 config bytes", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      fs.writeFileSync(path.join(codexHome, "config.toml"), Buffer.from([0xff]));
      assert.throws(
        () => loadCodexConfig(codexHome),
        (error) => error instanceof TypeError,
      );
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("loads model reasoning and token settings from the real config file", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      fs.writeFileSync(
        path.join(codexHome, "config.toml"),
        [
          'model = "gpt-5.6-sol"',
          'model_reasoning_effort = "ultra"',
          "model_context_window = 372_000",
          "model_auto_compact_token_limit = 334_800",
        ].join("\n"),
      );

      const config = loadCodexConfig(codexHome);
      assert.deepEqual(config, {
        codexHome,
        configPath: path.join(codexHome, "config.toml"),
        model: "gpt-5.6-sol",
        modelReasoningEffort: "ultra",
        modelContextWindow: 372_000,
        modelAutoCompactTokenLimit: 334_800,
      });
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("rejects an explicitly empty configured effort", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      fs.writeFileSync(path.join(codexHome, "config.toml"), 'model_reasoning_effort = ""\n');
      assert.throws(
        () => loadCodexConfig(codexHome),
        /model_reasoning_effort must be a non-empty TOML string/,
      );
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("decodes TOML Unicode escapes in the root reasoning effort", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      fs.writeFileSync(
        path.join(codexHome, "config.toml"),
        'model_reasoning_effort = "\\u0075ltra"\n',
      );
      assert.equal(loadCodexConfig(codexHome).modelReasoningEffort, "ultra");
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("fails loudly when root reasoning effort is not a TOML string", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      fs.writeFileSync(path.join(codexHome, "config.toml"), "model_reasoning_effort = 42\n");
      assert.throws(
        () => loadCodexConfig(codexHome),
        /model_reasoning_effort must be a TOML string/,
      );
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("rejects malformed TOML in an unrelated root value", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      fs.writeFileSync(
        path.join(codexHome, "config.toml"),
        'model = "gpt-5.6-sol"\nunrelated = [1,, 2]\n',
      );
      assert.throws(
        () => loadCodexConfig(codexHome),
        (error) => error instanceof TomlError,
      );
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("rejects a malformed unrelated TOML table", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      fs.writeFileSync(
        path.join(codexHome, "config.toml"),
        'model = "gpt-5.6-sol"\n[profiles.expensive\nreasoning = "high"\n',
      );
      assert.throws(
        () => loadCodexConfig(codexHome),
        (error) => error instanceof TomlError,
      );
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("requires selected token settings to be positive TOML integers", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      fs.writeFileSync(path.join(codexHome, "config.toml"), "model_context_window = 372000.0\n");
      assert.throws(
        () => loadCodexConfig(codexHome),
        /model_context_window must be a positive TOML integer/,
      );
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("ignores reasoning effort from an inactive profile table", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      fs.writeFileSync(
        path.join(codexHome, "config.toml"),
        'model = "gpt-5.6-sol"\n\n[profiles.expensive]\nmodel_reasoning_effort = "ultra"\n',
      );
      const config = loadCodexConfig(codexHome);
      assert.equal(config.model, "gpt-5.6-sol");
      assert.equal(config.modelReasoningEffort, undefined);
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });
});

describe("model capability catalog", () => {
  it("loads GPT-5.6 Lite, context, and default effort fields", () => {
    const expected = [
      ["gpt-5.6-sol", "low"],
      ["gpt-5.6-terra", "medium"],
      ["gpt-5.6-luna", "medium"],
    ] as const;

    for (const [model, effort] of expected) {
      const capability = capabilityForModel(model);
      assert.equal(capability.useResponsesLite, true);
      assert.equal(capability.supportsParallelToolCalls, true);
      assert.equal(capability.defaultReasoningEffort, effort);
      assert.equal(capability.contextWindow, 372_000);
      assert.equal(capability.maxContextWindow, 372_000);
    }
  });

  it("keeps unknown model capability fields absent", () => {
    const capability = capabilityForModel("provider/future-model");
    assert.equal(capability.defaultReasoningEffort, undefined);
    assert.equal(capability.contextWindow, undefined);
    assert.equal(capability.supportsImageDetailOriginal, false);
  });

  it("loads original image detail support conservatively", () => {
    for (const model of [
      "gpt-5.6",
      "gpt-5.6-sol",
      "gpt-5.6-terra",
      "gpt-5.6-luna",
      "gpt-5.5",
      "gpt-5.4",
      "gpt-5.4-mini",
    ]) {
      assert.equal(capabilityForModel(model).supportsImageDetailOriginal, true);
    }
    for (const model of [
      "gpt-5.2",
      "gpt-5.3-codex",
      "gpt-5.3-codex-spark",
      "provider/future-model",
    ]) {
      assert.equal(capabilityForModel(model).supportsImageDetailOriginal, false);
    }
  });

  it("loads current existing-model context and effort defaults", () => {
    const expected = [
      ["gpt-5.5", 272_000],
      ["gpt-5.4", 1_000_000],
      ["gpt-5.4-mini", 272_000],
      ["gpt-5.2", 272_000],
    ] as const;

    for (const [model, maximum] of expected) {
      const capability = capabilityForModel(model);
      assert.equal(capability.defaultReasoningEffort, "medium");
      assert.equal(capability.contextWindow, 272_000);
      assert.equal(capability.maxContextWindow, maximum);
    }
  });
});
