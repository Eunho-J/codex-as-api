import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { TomlError } from "smol-toml";
import { loadCodexConfig } from "../codex-config.js";

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
          "model_context_window = 272_000",
          "model_auto_compact_token_limit = 244_800",
        ].join("\n"),
      );

      const config = loadCodexConfig(codexHome);
      assert.deepEqual(config, {
        codexHome,
        configPath: path.join(codexHome, "config.toml"),
        model: "gpt-5.6-sol",
        modelReasoningEffort: "ultra",
        modelContextWindow: 272_000,
        modelAutoCompactTokenLimit: 244_800,
      });
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("rejects an explicitly empty configured effort", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      for (const value of ["", "   "]) {
        fs.writeFileSync(
          path.join(codexHome, "config.toml"),
          `model_reasoning_effort = ${JSON.stringify(value)}\n`,
        );
        assert.throws(
          () => loadCodexConfig(codexHome),
          /model_reasoning_effort must be a non-empty TOML string/,
        );
      }
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("rejects surrounding whitespace in configured model identifiers and efforts", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      for (const [field, value] of [
        ["model", " gpt-5.6-sol"],
        ["model", "gpt-5.6-sol "],
        ["model_reasoning_effort", " ultra"],
        ["model_reasoning_effort", "ultra "],
      ]) {
        fs.writeFileSync(
          path.join(codexHome, "config.toml"),
          `${field} = ${JSON.stringify(value)}\n`,
        );
        assert.throws(
          () => loadCodexConfig(codexHome),
          /must not contain surrounding whitespace/,
        );
      }
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
      fs.writeFileSync(path.join(codexHome, "config.toml"), "model_context_window = 272000.0\n");
      assert.throws(
        () => loadCodexConfig(codexHome),
        /model_context_window must be a positive TOML integer/,
      );
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("accepts zero as an immediate configured auto-compaction limit", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      fs.writeFileSync(
        path.join(codexHome, "config.toml"),
        "model_auto_compact_token_limit = 0\n",
      );
      assert.equal(loadCodexConfig(codexHome).modelAutoCompactTokenLimit, 0);
    } finally {
      fs.rmSync(codexHome, { recursive: true, force: true });
    }
  });

  it("preserves negative configured auto-compaction limits", () => {
    const codexHome = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-config-"));
    try {
      fs.writeFileSync(
        path.join(codexHome, "config.toml"),
        "model_auto_compact_token_limit = -1\n",
      );
      assert.equal(loadCodexConfig(codexHome).modelAutoCompactTokenLimit, -1);
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
