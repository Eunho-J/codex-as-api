import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import upstreamContract from "../../../config/codex-upstream-contract.json";

import {
  ChatGPTOAuthCatalogUnavailableError,
  ChatGPTOAuthInvalidRequestError,
  ChatGPTOAuthModelNotFoundError,
  ChatGPTOAuthProtocolError,
  ChatGPTOAuthRefreshError,
  ChatGPTOAuthUpstreamError,
} from "../auth.js";
import {
  accountIdFromModelCatalogCacheKey,
  applyModelCapabilityFields,
  ModelCatalogCache,
  modelCatalogCacheKey,
  modelFromSnapshot,
  parseModelCatalog,
  shouldEnableParallelToolCalls,
} from "../model-capabilities.js";
import { MessageRole } from "../messages.js";
import { ChatGPTOAuthProvider } from "../provider.js";

function rawModel(
  slug: string,
  overrides: Record<string, unknown> = {},
): Record<string, unknown> {
  return {
    slug,
    display_name: slug,
    description: null,
    default_reasoning_level: "low",
    supported_reasoning_levels: [
      { effort: "low", description: "Low" },
      { effort: "max", description: "Maximum" },
    ],
    multi_agent_reasoning_effort: "max",
    priority: 10,
    visibility: "list",
    supported_in_api: true,
    service_tiers: [
      { id: "priority", name: "Priority", description: "Priority tier" },
    ],
    default_service_tier: null,
    support_verbosity: true,
    default_verbosity: "low",
    supports_image_detail_original: false,
    context_window: 272_000,
    max_context_window: 272_000,
    auto_compact_token_limit: null,
    input_modalities: ["text", "image"],
    use_responses_lite: false,
    supports_reasoning_summaries: true,
    available_in_plans: ["plus", "pro"],
    prefer_websockets: false,
    requires_sandboxed_review: false,
    minimal_client_version: "0.153.3",
    ...overrides,
  };
}

describe("live verbosity capability", () => {
  it("gates only a non-null verbosity value and preserves other text options", () => {
    const credentialSentinel = "access_token=MODEL_CAPABILITY_SENTINEL";
    const snapshot = parseModelCatalog({
      models: [rawModel(credentialSentinel, {
        support_verbosity: false,
        default_verbosity: null,
      })],
    }, metadata());
    const capability = snapshot.models[0];
    const payload: Record<string, unknown> = {};

    applyModelCapabilityFields(payload, capability, {
      format: { type: "json_schema" },
      verbosity: null,
    });
    assert.deepEqual(payload.text, { format: { type: "json_schema" } });
    assert.throws(
      () => applyModelCapabilityFields({}, capability, { verbosity: "high" }),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError
        && !error.message.includes(credentialSentinel),
    );
  });

  it("rejects malformed verbosity even when the live model supports it", () => {
    const capability = parseModelCatalog({
      models: [rawModel("verbosity-model")],
    }, metadata()).models[0];

    for (const verbosity of ["future", 1, false, {}]) {
      assert.throws(
        () => applyModelCapabilityFields({}, capability, { verbosity }),
        ChatGPTOAuthInvalidRequestError,
      );
    }
  });
});

function metadata(key = modelCatalogCacheKey("account-a", "https://example.test", "0.153.3")) {
  return { key, etag: '"etag-a"', fetchedAt: 100, expiresAt: 400 };
}

function makeJwt(payload: Record<string, unknown>): string {
  const header = Buffer.from(JSON.stringify({ alg: "HS256", typ: "JWT" })).toString("base64url");
  const body = Buffer.from(JSON.stringify(payload)).toString("base64url");
  return `${header}.${body}.sig`;
}

function writeAuthFile(accessToken = makeJwt({ exp: 9_999_999_999 })): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "codex-as-api-model-catalog-"));
  const authPath = path.join(dir, "auth.json");
  fs.writeFileSync(authPath, JSON.stringify({
    tokens: {
      access_token: accessToken,
      refresh_token: "refresh-old",
      id_token: makeJwt({
        exp: 9_999_999_999,
        "https://api.openai.com/auth": { chatgpt_account_id: "account-a" },
      }),
    },
  }));
  return authPath;
}

function unreadUnauthorizedResponse(cancelFailure?: string): {
  response: Response;
  state: { cancelCalls: number; textCalls: number };
} {
  const state = { cancelCalls: 0, textCalls: 0 };
  const response = new Response("unread unauthorized body", { status: 401 });
  Object.defineProperty(response, "text", {
    value: async () => {
      state.textCalls++;
      throw new Error("the first 401 body must not be read");
    },
  });
  assert.ok(response.body);
  Object.defineProperty(response.body, "cancel", {
    value: () => {
      state.cancelCalls++;
      return cancelFailure == null
        ? Promise.resolve()
        : Promise.reject(new Error(cancelFailure));
    },
  });
  return { response, state };
}

describe("authenticated model catalog parser", () => {
  it("uses stable priority ordering for the live default and preserves all rows", () => {
    const hidden = rawModel("hidden-first", { priority: 0, visibility: "hide" });
    const later = rawModel("visible-later", { priority: 20, supported_in_api: false });
    const selected = rawModel("visible-selected", { priority: 5 });
    const snapshot = parseModelCatalog({ models: [hidden, later, selected] }, metadata());

    assert.equal(snapshot.defaultModel?.slug, "visible-selected");
    assert.deepEqual(snapshot.models.map((model) => model.slug), [
      "hidden-first",
      "visible-later",
      "visible-selected",
    ]);
    assert.equal(snapshot.models[1].supportedInApi, false);
    assert.equal(snapshot.models[2].effectiveContextWindowPercent, 95);
    assert.deepEqual(snapshot.models[2].inputModalities, ["text", "image"]);
    assert.equal(snapshot.models[2].supportsReasoningSummaryParameter, true);
    assert.equal(snapshot.models[2].defaultReasoningSummary, "auto");
    assert.deepEqual(snapshot.models[2].serviceTiers, [
      { id: "priority", name: "Priority", description: "Priority tier" },
    ]);
    assert.equal(Object.isFrozen(snapshot), true);
    assert.equal(Object.isFrozen(snapshot.models), true);
  });

  it("treats an empty optional ETag as absent", () => {
    const snapshot = parseModelCatalog(
      { models: [rawModel("etag-model")] },
      { ...metadata(), etag: "   " },
    );
    assert.equal(snapshot.etag, null);
  });

  it("keeps hidden rows addressable without inventing an implicit default", () => {
    const snapshot = parseModelCatalog({
      models: [
        rawModel("hidden-first", { visibility: "hide" }),
        rawModel("hidden-second", { visibility: "none" }),
      ],
    }, metadata());

    assert.equal(snapshot.defaultModel, null);
    assert.equal(modelFromSnapshot(snapshot, "hidden-first")?.slug, "hidden-first");
  });

  it("preserves empty cosmetic catalog strings without rejecting the live catalog", () => {
    const model = parseModelCatalog({
      models: [rawModel("cosmetic-empty", {
        display_name: "",
        description: "",
        supported_reasoning_levels: [{ effort: "low", description: "" }],
        multi_agent_reasoning_effort: "low",
        service_tiers: [{ id: "priority", name: "", description: "" }],
      })],
    }, metadata()).models[0];

    assert.equal(model.displayName, "");
    assert.equal(model.description, "");
    assert.equal(model.supportedReasoningEfforts[0].description, "");
    assert.equal(model.serviceTiers[0].name, "");
    assert.equal(model.serviceTiers[0].description, "");
  });

  it("rejects malformed and duplicate-slug catalogs atomically", () => {
    assert.throws(
      () => parseModelCatalog([rawModel("legacy-array")], metadata()),
      (error) => error instanceof ChatGPTOAuthCatalogUnavailableError,
    );
    assert.throws(
      () => parseModelCatalog({ models: [rawModel("same"), rawModel("same")] }, metadata()),
      (error) => error instanceof ChatGPTOAuthCatalogUnavailableError,
    );
    for (const malformed of [
      { default_verbosity: "verbose" },
      { default_reasoning_level: "" },
      { multi_agent_reasoning_effort: "" },
      { supported_reasoning_levels: [{ effort: "", description: "Empty" }] },
      { auto_compact_token_limit: "90000" },
      { priority: 2_147_483_648 },
      { priority: -2_147_483_649 },
      { context_window: Number.MAX_SAFE_INTEGER + 1 },
      { effective_context_window_percent: 1.5 },
    ]) {
      assert.throws(
        () => parseModelCatalog({ models: [rawModel("bad-live-field", malformed)] }, metadata()),
        (error) => error instanceof ChatGPTOAuthCatalogUnavailableError,
      );
    }
  });

  it("preserves empty catalogs, opaque slugs, comp_hash, and nonempty custom reasoning values", () => {
    const empty = parseModelCatalog({ models: [] }, metadata());
    const custom = parseModelCatalog({
      models: [rawModel(" ", {
        comp_hash: " compatibility family ",
        default_reasoning_level: " ",
        multi_agent_reasoning_effort: " custom ",
        supported_reasoning_levels: [{ effort: " ", description: "Custom" }],
      })],
    }, metadata()).models[0];

    assert.deepEqual(empty.models, []);
    assert.equal(empty.defaultModel, null);
    assert.equal(custom.slug, " ");
    assert.equal(custom.compHash, " compatibility family ");
    assert.equal(custom.defaultReasoningEffort, " ");
    assert.equal(custom.multiAgentReasoningEffort, " custom ");
    assert.equal(custom.supportedReasoningEfforts[0].effort, " ");
  });

  it("preserves catalog defaults without inventing cross-field constraints", () => {
    const unadvertised = parseModelCatalog({
      models: [rawModel("unadvertised", {
        default_reasoning_level: "future",
        supported_reasoning_levels: [{ effort: "low", description: "Low" }],
      })],
    }, metadata()).models[0];
    const ultraOnly = parseModelCatalog({
      models: [rawModel("ultra-only", {
        default_reasoning_level: "ultra",
        supported_reasoning_levels: [{ effort: "ultra", description: "Ultra" }],
        multi_agent_reasoning_effort: "ultra",
      })],
    }, metadata()).models[0];

    assert.equal(unadvertised.defaultReasoningEffort, "future");
    assert.equal(ultraOnly.defaultReasoningEffort, "ultra");
    assert.deepEqual(ultraOnly.supportedReasoningEfforts.map(({ effort }) => effort), ["ultra"]);
  });

  it("does not reflect duplicate slugs while preserving valid catalog identifiers", () => {
    const credentialSentinel = "access_token=CATALOG_SENTINEL";
    assert.throws(
      () => parseModelCatalog({
        models: [rawModel(credentialSentinel), rawModel(credentialSentinel)],
      }, metadata()),
      (error) => error instanceof ChatGPTOAuthCatalogUnavailableError
        && !error.message.includes(credentialSentinel),
    );
    assert.equal(
      parseModelCatalog({ models: [rawModel(credentialSentinel)] }, metadata())
        .models[0].slug,
      credentialSentinel,
    );
  });

  it("preserves official ModelInfo rows without invented semantic constraints", () => {
    const model = parseModelCatalog({
      models: [rawModel("lossless-model-info", {
        default_reasoning_level: "future-default",
        supported_reasoning_levels: [
          { effort: "future-default", description: "future" },
          { effort: "low", description: "first" },
          { effort: "low", description: "second" },
        ],
        multi_agent_reasoning_effort: "future-multi-agent",
        service_tiers: [
          { id: "priority", name: "first", description: "first" },
          { id: "priority", name: "second", description: "second" },
        ],
        default_service_tier: "future-tier",
        context_window: 0,
        max_context_window: -1,
        auto_compact_token_limit: -2,
        effective_context_window_percent: 101,
      })],
    }, metadata()).models[0];

    assert.equal(model.defaultReasoningEffort, "future-default");
    assert.deepEqual(model.supportedReasoningEfforts, [
      { effort: "future-default", description: "future" },
      { effort: "low", description: "first" },
      { effort: "low", description: "second" },
    ]);
    assert.equal(model.multiAgentReasoningEffort, "future-multi-agent");
    assert.deepEqual(model.serviceTiers, [
      { id: "priority", name: "first", description: "first" },
      { id: "priority", name: "second", description: "second" },
    ]);
    assert.equal(model.defaultServiceTier, "future-tier");
    assert.equal(model.contextWindow, 0);
    assert.equal(model.maxContextWindow, -1);
    assert.equal(model.autoCompactTokenLimit, -2);
    assert.equal(model.effectiveContextWindowPercent, 101);
  });

  it("preserves unconstrained service tier strings", () => {
    const model = parseModelCatalog({
      models: [rawModel("service-strings", {
        service_tiers: [{ id: "", name: "", description: "" }],
        default_service_tier: "",
      })],
    }, metadata()).models[0];

    assert.equal(model.serviceTiers[0].id, "");
    assert.equal(model.defaultServiceTier, "");
  });

  it("uses only the official upstream defaults for omitted optional model fields", () => {
    const value = rawModel("minimal", {
      supported_reasoning_levels: [],
    });
    for (const field of [
      "description",
      "default_reasoning_level",
      "default_verbosity",
      "context_window",
      "max_context_window",
      "auto_compact_token_limit",
      "default_service_tier",
      "multi_agent_reasoning_effort",
      "service_tiers",
      "use_responses_lite",
      "supports_image_detail_original",
      "input_modalities",
      "supports_reasoning_summary_parameter",
      "default_reasoning_summary",
    ]) {
      delete value[field];
    }

    const model = parseModelCatalog({ models: [value] }, metadata()).models[0];
    assert.equal(model.description, null);
    assert.equal(model.defaultReasoningEffort, undefined);
    assert.equal(model.defaultVerbosity, null);
    assert.equal(model.contextWindow, undefined);
    assert.equal(model.maxContextWindow, undefined);
    assert.equal(model.autoCompactTokenLimit, undefined);
    assert.equal(model.defaultServiceTier, null);
    assert.equal(model.multiAgentReasoningEffort, undefined);
    assert.deepEqual(model.supportedReasoningEfforts, []);
    assert.deepEqual(model.serviceTiers, []);
    assert.equal(model.useResponsesLite, false);
    assert.equal(model.supportsImageDetailOriginal, false);
    assert.deepEqual(model.inputModalities, ["text", "image"]);
    assert.equal(model.supportsReasoningSummaryParameter, true);
    assert.equal(model.defaultReasoningSummary, "auto");
    const noModalities = parseModelCatalog({
      models: [rawModel("no-modalities", { input_modalities: [] })],
    }, metadata()).models[0];
    assert.deepEqual(noModalities.inputModalities, []);
    assert.equal(shouldEnableParallelToolCalls({ requested: true, responsesLite: false }), true);
  });

  it("preserves and validates reasoning summary controls", () => {
    const model = parseModelCatalog({
      models: [rawModel("summary-controls", {
        supports_reasoning_summary_parameter: false,
        default_reasoning_summary: "detailed",
      })],
    }, metadata()).models[0];
    assert.equal(model.supportsReasoningSummaryParameter, false);
    assert.equal(model.defaultReasoningSummary, "detailed");

    for (const malformed of [
      { supports_reasoning_summary_parameter: null },
      { supports_reasoning_summary_parameter: "true" },
      { default_reasoning_summary: null },
      { default_reasoning_summary: "future" },
    ]) {
      assert.throws(
        () => parseModelCatalog({ models: [rawModel("bad-summary", malformed)] }, metadata()),
        (error) => error instanceof ChatGPTOAuthCatalogUnavailableError,
      );
    }
  });

  it("does not reflect model or service-tier values in capability diagnostics", () => {
    const modelSentinel = "access_token=MODEL_DIAGNOSTIC_SENTINEL";
    const requestSentinel = "access_token=SERVICE_TIER_SENTINEL";
    const capability = parseModelCatalog({
      models: [rawModel(modelSentinel, { service_tiers: [] })],
    }, metadata()).models[0];

    assert.throws(
      () => applyModelCapabilityFields({}, capability, undefined, requestSentinel),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError
        && !error.message.includes(modelSentinel)
        && !error.message.includes(requestSentinel),
    );
  });
});

describe("fresh-only model catalog cache", () => {
  it("uses collision-free tuple keys and preserves the exact account id", () => {
    const accountId = "account-a\0delegated";
    const first = modelCatalogCacheKey(accountId, "base", "version");
    const second = modelCatalogCacheKey("account-a", "delegated\0base", "version");

    assert.notEqual(first, second);
    assert.equal(accountIdFromModelCatalogCacheKey(first), accountId);
    assert.throws(
      () => accountIdFromModelCatalogCacheKey("account-a\0base\0version"),
      (error) => error instanceof ChatGPTOAuthProtocolError,
    );
  });

  it("shares an in-flight fetch and reuses only a fresh snapshot", async () => {
    let now = 1_000;
    let calls = 0;
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const cache = new ModelCatalogCache(100, () => now, () => now);
    const fetchCatalog = async () => {
      calls++;
      await gate;
      return { value: { models: [rawModel("live")] }, etag: '"one"' };
    };

    const first = cache.get("key", fetchCatalog);
    const second = cache.get("key", fetchCatalog);
    release?.();
    assert.equal(await first, await second);
    assert.equal(calls, 1);

    await cache.get("key", fetchCatalog);
    assert.equal(calls, 1);
    now = 1_101;
    await cache.get("key", fetchCatalog);
    assert.equal(calls, 2);
  });

  it("keys snapshots by account/base/version and never serves stale data on refresh failure", async () => {
    let now = 10;
    let calls = 0;
    const cache = new ModelCatalogCache(10, () => now, () => now);
    const load = async () => {
      calls++;
      return { value: { models: [rawModel(`model-${calls}`)] }, etag: `"${calls}"` };
    };
    const scope = {
      account_id: "account-a",
      base_url: "https://one.test",
      client_version: upstreamContract.upstream.version,
    };
    assert.deepEqual(upstreamContract.models_request.cache_scope, Object.keys(scope));
    const keyA = modelCatalogCacheKey(scope.account_id, scope.base_url, scope.client_version);
    const keyB = modelCatalogCacheKey("account-b", scope.base_url, scope.client_version);
    const keyC = modelCatalogCacheKey(scope.account_id, "https://two.test", scope.client_version);
    const keyD = modelCatalogCacheKey(scope.account_id, scope.base_url, "0.154.0");

    await cache.get(keyA, load);
    await cache.get(keyB, load);
    await cache.get(keyC, load);
    await cache.get(keyD, load);
    assert.equal(calls, 4);

    now = 21;
    const refreshError = new Error("refresh failed");
    assert.equal(upstreamContract.models_request.allow_stale_on_refresh_error, false);
    await assert.rejects(
      () => cache.get(keyA, async () => { throw refreshError; }),
      (error) => error === refreshError,
    );
    assert.equal(calls, 4);
  });

  it("invalidates only the matching key after an X-Models-Etag mismatch", async () => {
    let callsA = 0;
    let callsB = 0;
    const cache = new ModelCatalogCache(1_000, () => 1);
    const keyA = modelCatalogCacheKey("account-a", "base", "version");
    const keyB = modelCatalogCacheKey("account-b", "base", "version");
    await cache.get(keyA, async () => ({ value: { models: [rawModel("a")] }, etag: '"a1"' }));
    await cache.get(keyB, async () => ({ value: { models: [rawModel("b")] }, etag: '"b1"' }));

    cache.invalidateOnEtagMismatch(keyA, '"a2"');
    await cache.get(keyA, async () => {
      callsA++;
      return { value: { models: [rawModel("a")] }, etag: '"a2"' };
    });
    await cache.get(keyB, async () => {
      callsB++;
      return { value: { models: [rawModel("b")] }, etag: '"b2"' };
    });
    assert.equal(callsA, 1);
    assert.equal(callsB, 0);
  });

  it("applies repeated observations of one changed ETag only once during refresh", async () => {
    let releaseRefresh: (() => void) | undefined;
    const refreshGate = new Promise<void>((resolve) => { releaseRefresh = resolve; });
    const cache = new ModelCatalogCache(1_000, () => 1, () => 1);
    const key = modelCatalogCacheKey("account-a", "base", "version");
    await cache.get(key, async () => ({
      value: { models: [rawModel("initial")] },
      etag: '"old"',
    }));

    cache.invalidateOnEtagMismatch(key, '"new"');
    const refresh = cache.get(key, async () => {
      await refreshGate;
      return { value: { models: [rawModel("fresh")] }, etag: '"new"' };
    });
    cache.invalidateOnEtagMismatch(key, '"new"');
    cache.invalidateOnEtagMismatch(key, '"new"');
    releaseRefresh?.();

    assert.equal((await refresh).models[0].slug, "fresh");
  });

  it("does not invalidate an initial in-flight load before a snapshot exists", async () => {
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const cache = new ModelCatalogCache(1_000, () => 1, () => 1);
    const key = modelCatalogCacheKey("account-a", "base", "version");
    const initial = cache.get(key, async () => {
      await gate;
      return { value: { models: [rawModel("initial")] }, etag: '"e1"' };
    });

    cache.invalidateOnEtagMismatch(key, '"e1"');
    release?.();
    assert.equal((await initial).models[0].slug, "initial");
  });

  it("does not fence an expired refresh when the observed ETag still matches", async () => {
    let now = 1;
    let release: (() => void) | undefined;
    const gate = new Promise<void>((resolve) => { release = resolve; });
    const cache = new ModelCatalogCache(10, () => now, () => now);
    const key = modelCatalogCacheKey("account-a", "base", "version");
    await cache.get(key, async () => ({
      value: { models: [rawModel("initial")] },
      etag: '"e1"',
    }));

    now = 12;
    const refresh = cache.get(key, async () => {
      await gate;
      return { value: { models: [rawModel("refreshed")] }, etag: '"e1"' };
    });
    cache.invalidateOnEtagMismatch(key, '"e1"');
    release?.();
    assert.equal((await refresh).models[0].slug, "refreshed");
  });

  it("expires by monotonic age even when the wall clock rolls backward", async () => {
    let wallNow = 10_000;
    let monotonicNow = 100;
    let calls = 0;
    const cache = new ModelCatalogCache(
      100,
      () => wallNow,
      () => monotonicNow,
    );
    const load = async () => {
      calls++;
      return { value: { models: [rawModel(`model-${calls}`)] }, etag: `"${calls}"` };
    };

    const first = await cache.get("key", load);
    assert.equal(first.fetchedAt, 10_000);
    assert.equal(first.expiresAt, 10_100);

    wallNow = 1_000;
    monotonicNow = 201;
    const second = await cache.get("key", load);
    assert.equal(calls, 2);
    assert.equal(second.models[0].slug, "model-2");
    assert.equal(second.fetchedAt, 1_000);
    assert.equal(second.expiresAt, 1_100);
  });

  it("does not publish an in-flight snapshot invalidated by a newer Responses ETag", async () => {
    let now = 1;
    let releaseRefresh: (() => void) | undefined;
    const refreshGate = new Promise<void>((resolve) => { releaseRefresh = resolve; });
    const cache = new ModelCatalogCache(10, () => now, () => now);
    const key = modelCatalogCacheKey("account-a", "base", "version");
    await cache.get(key, async () => ({
      value: { models: [rawModel("initial")] },
      etag: '"old"',
    }));

    now = 12;
    const staleRefresh = cache.get(key, async () => {
      await refreshGate;
      return { value: { models: [rawModel("stale-refresh")] }, etag: '"old"' };
    });
    cache.invalidateOnEtagMismatch(key, '"new"');
    releaseRefresh?.();
    await assert.rejects(staleRefresh, ChatGPTOAuthCatalogUnavailableError);

    const fresh = await cache.get(key, async () => ({
      value: { models: [rawModel("fresh-refresh")] },
      etag: '"new"',
    }));
    assert.equal(fresh.models[0].slug, "fresh-refresh");
  });

  it("ignores blank X-Models-Etag values", async () => {
    let refreshes = 0;
    const cache = new ModelCatalogCache(1_000, () => 1);
    const key = modelCatalogCacheKey("account-a", "base", "version");
    await cache.get(key, async () => ({
      value: { models: [rawModel("a")] },
      etag: '"a1"',
    }));

    cache.invalidateOnEtagMismatch(key, " \t ");
    await cache.get(key, async () => {
      refreshes++;
      return { value: { models: [rawModel("a")] }, etag: '"a2"' };
    });
    assert.equal(refreshes, 0);
  });
});

describe("ChatGPTOAuthProvider model discovery", () => {
  it("passes a control-character account id through prepareModel without truncation", async () => {
    const accountId = "account-a\0delegated";
    const snapshot = parseModelCatalog({ models: [rawModel("live-model")] }, metadata(
      modelCatalogCacheKey(accountId, "https://catalog.test", "0.153.3"),
    ));
    const prepared = await new ChatGPTOAuthProvider().prepareModel("live-model", snapshot);

    assert.equal(prepared.accountId, accountId);
  });

  it("rejects unsafe or ambiguous upstream base URLs", () => {
    for (const baseUrl of [
      " ",
      " https://catalog.test/codex",
      "https://catalog.test/codex ",
      "not-a-url",
      "https:catalog.test/codex",
      "https:/catalog.test/codex",
      "https:///catalog.test/codex",
      "https://catalog.test/co dex",
      "https://catalog.test/co\tdex",
      "https://catalog.test/co\r\ndex",
      "https://catalog.test/co\u0000dex",
      "https://catalog.test/co\u00a0dex",
      "https://catalog.test/%",
      "https://catalog.test/%zz",
      "https://catalog.test/%0G",
      "ftp://catalog.test",
      "https://user@catalog.test",
      "https://catalog.test?x=1",
      "https://catalog.test#fragment",
    ]) {
      assert.throws(
        () => new ChatGPTOAuthProvider({ baseUrl }),
        (error) => error instanceof ChatGPTOAuthInvalidRequestError,
      );
    }
    assert.doesNotThrow(
      () => new ChatGPTOAuthProvider({ baseUrl: "http://127.0.0.1:18081/backend-api/codex/" }),
    );
    assert.doesNotThrow(
      () => new ChatGPTOAuthProvider({ baseUrl: "https://catalog.test/backend-api/co%20dex" }),
    );
  });

  it("sends the pinned authenticated catalog request and refreshes a 401 exactly once", async () => {
    const oldAccessToken = makeJwt({ exp: 9_999_999_999 });
    const authPath = writeAuthFile(oldAccessToken);
    const originalFetch = globalThis.fetch;
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const refreshedIdToken = makeJwt({
      exp: 9_999_999_999,
      "https://api.openai.com/auth": { chatgpt_account_id: "account-a" },
    });
    const refreshedAccessToken = makeJwt({
      exp: 9_999_999_999,
      "https://api.openai.com/auth": { chatgpt_account_id: "account-a" },
    });
    const unauthorized = unreadUnauthorizedResponse();
    try {
      globalThis.fetch = async (input, init) => {
        const url = String(input);
        requests.push({ url, init });
        if (url === "https://auth.openai.com/oauth/token") {
          return new Response(JSON.stringify({
            access_token: refreshedAccessToken,
            refresh_token: "refresh-new",
            id_token: refreshedIdToken,
          }), { status: 200, headers: { "Content-Type": "application/json" } });
        }
        const catalogUrl = `https://catalog.test${upstreamContract.models_request.path}`;
        if (requests.filter((request) => request.url.startsWith(catalogUrl)).length === 1) {
          return unauthorized.response;
        }
        return new Response(JSON.stringify({ models: [rawModel("live-model")] }), {
          status: 200,
          headers: {
            "Content-Type": "application/json",
            [upstreamContract.models_request.etag_header]: '"catalog"',
          },
        });
      };

      const provider = new ChatGPTOAuthProvider({
        authJsonPath: authPath,
        baseUrl: "https://catalog.test",
      });
      const snapshot = await provider.catalogSnapshot();
      assert.equal(snapshot.models[0].slug, "live-model");
      const catalogRequests = requests.filter((request) => request.url.startsWith(
        `https://catalog.test${upstreamContract.models_request.path}`,
      ));
      assert.equal(catalogRequests.length, 2);
      const firstCatalogUrl = new URL(catalogRequests[0].url);
      assert.equal(firstCatalogUrl.pathname, upstreamContract.models_request.path);
      assert.equal(
        firstCatalogUrl.searchParams.get(upstreamContract.models_request.client_version_query),
        upstreamContract.upstream.version,
      );
      assert.equal(new Headers(catalogRequests[0].init?.headers).get("originator"), "codex_cli_rs");
      assert.equal(new Headers(catalogRequests[0].init?.headers).get("ChatGPT-Account-Id"), "account-a");
      assert.equal(new Headers(catalogRequests[0].init?.headers).get("Authorization"), `Bearer ${oldAccessToken}`);
      assert.equal(
        (new Headers(catalogRequests[0].init?.headers).get("User-Agent") ?? "").startsWith("codex_cli_rs/0.153.3 "),
        true,
      );
      assert.equal(new Headers(catalogRequests[1].init?.headers).get("Authorization"), `Bearer ${refreshedAccessToken}`);
      assert.deepEqual(unauthorized.state, { cancelCalls: 1, textCalls: 0 });
    } finally {
      globalThis.fetch = originalFetch;
      fs.rmSync(path.dirname(authPath), { recursive: true, force: true });
    }
  });

  it("surfaces and redacts catalog response cancellation failure before refresh", async () => {
    const authPath = writeAuthFile();
    const originalFetch = globalThis.fetch;
    const secret = "refresh-old";
    const unauthorized = unreadUnauthorizedResponse(`cancel exposed ${secret}`);
    let catalogCalls = 0;
    let refreshCalls = 0;
    try {
      globalThis.fetch = async (input) => {
        if (String(input) === "https://auth.openai.com/oauth/token") {
          refreshCalls++;
          throw new Error("refresh must not start after cancellation failure");
        }
        catalogCalls++;
        return unauthorized.response;
      };

      const provider = new ChatGPTOAuthProvider({
        authJsonPath: authPath,
        baseUrl: "https://cancel-failure.test",
      });
      await assert.rejects(provider.catalogSnapshot(), (error: unknown) => (
        error instanceof ChatGPTOAuthCatalogUnavailableError
        && error.message.includes("response cancellation failed")
        && !error.message.includes(secret)
      ));
      assert.equal(catalogCalls, 1);
      assert.equal(refreshCalls, 0);
      assert.deepEqual(unauthorized.state, { cancelCalls: 1, textCalls: 0 });
    } finally {
      globalThis.fetch = originalFetch;
      fs.rmSync(path.dirname(authPath), { recursive: true, force: true });
    }
  });

  it("never commits a refreshed account catalog under the initial account key", async () => {
    const authPath = writeAuthFile();
    const originalFetch = globalThis.fetch;
    const accountBIdToken = makeJwt({
      exp: 9_999_999_999,
      "https://api.openai.com/auth": { chatgpt_account_id: "account-b" },
    });
    const accountBAccessToken = makeJwt({
      exp: 9_999_999_999,
      "https://api.openai.com/auth": { chatgpt_account_id: "account-b" },
    });
    let catalogRequests = 0;
    let rejectAccountChange = true;
    try {
      globalThis.fetch = async (input) => {
        const url = String(input);
        if (url === "https://auth.openai.com/oauth/token") {
          return new Response(JSON.stringify({
            access_token: accountBAccessToken,
            refresh_token: "refresh-account-b",
            id_token: accountBIdToken,
          }), { status: 200, headers: { "Content-Type": "application/json" } });
        }
        if (!url.startsWith("https://catalog.test/models")) {
          throw new Error("unexpected request");
        }
        catalogRequests++;
        if (rejectAccountChange && catalogRequests === 1) {
          return new Response("unauthorized", { status: 401 });
        }
        return new Response(JSON.stringify({
          models: [rawModel(rejectAccountChange ? "account-b-model" : "account-a-model")],
        }), {
          status: 200,
          headers: { "Content-Type": "application/json", ETag: '"catalog"' },
        });
      };

      const provider = new ChatGPTOAuthProvider({
        authJsonPath: authPath,
        baseUrl: "https://catalog.test",
      });
      await assert.rejects(
        () => provider.catalogSnapshot(),
        ChatGPTOAuthRefreshError,
      );
      assert.equal(catalogRequests, 1);

      rejectAccountChange = false;
      const recovered = await provider.catalogSnapshot();
      assert.equal(catalogRequests, 2);
      assert.equal(recovered.models[0].slug, "account-a-model");
      assert.equal(
        accountIdFromModelCatalogCacheKey(recovered.key),
        "account-a",
      );
    } finally {
      globalThis.fetch = originalFetch;
      fs.rmSync(path.dirname(authPath), { recursive: true, force: true });
    }
  });

  it("preserves an upstream catalog HTTP status instead of hiding it as generic unavailability", async () => {
    const authPath = writeAuthFile();
    const originalFetch = globalThis.fetch;
    try {
      globalThis.fetch = async () => new Response("account-a rate limited", { status: 429 });
      const provider = new ChatGPTOAuthProvider({
        authJsonPath: authPath,
        baseUrl: "https://catalog.test",
      });

      await assert.rejects(
        () => provider.catalogSnapshot(),
        (error) => error instanceof ChatGPTOAuthUpstreamError
          && error.status === 429
          && !error.message.includes("account-a")
          && error.message.includes("*** rate limited"),
      );
    } finally {
      globalThis.fetch = originalFetch;
      fs.rmSync(path.dirname(authPath), { recursive: true, force: true });
    }
  });

  it("does not follow authenticated model catalog redirects", async () => {
    const authPath = writeAuthFile();
    const originalFetch = globalThis.fetch;
    let calls = 0;
    try {
      globalThis.fetch = async (_input, init) => {
        calls++;
        assert.equal(init?.redirect, "manual");
        return new Response("redirect refused", {
          status: 307,
          headers: { Location: "https://attacker.example/steal" },
        });
      };
      const provider = new ChatGPTOAuthProvider({
        authJsonPath: authPath,
        baseUrl: "https://catalog.test",
      });
      await assert.rejects(
        () => provider.catalogSnapshot(),
        (error: unknown) => error instanceof ChatGPTOAuthUpstreamError
          && error.status === 307,
      );
      assert.equal(calls, 1);
    } finally {
      globalThis.fetch = originalFetch;
      fs.rmSync(path.dirname(authPath), { recursive: true, force: true });
    }
  });

  it("refreshes the next snapshot after an upstream X-Models-Etag mismatch", async () => {
    const authPath = writeAuthFile();
    const originalFetch = globalThis.fetch;
    let catalogRequests = 0;
    try {
      globalThis.fetch = async (input) => {
        const url = new URL(String(input));
        if (url.pathname === upstreamContract.models_request.path) {
          catalogRequests++;
          return new Response(JSON.stringify({ models: [rawModel("live-model")] }), {
            status: 200,
            headers: {
              "Content-Type": "application/json",
              [upstreamContract.models_request.etag_header]: catalogRequests === 1
                ? '"catalog-v1"'
                : '"catalog-v2"',
            },
          });
        }
        if (url.pathname === "/responses/compact") {
          return new Response(JSON.stringify({ output: [] }), {
            status: 200,
            headers: {
              "Content-Type": "application/json",
              [upstreamContract.models_request.responses_etag_header]: '"catalog-v2"',
            },
          });
        }
        throw new Error(`unexpected URL ${url}`);
      };

      const provider = new ChatGPTOAuthProvider({
        authJsonPath: authPath,
        baseUrl: "https://catalog.test",
      });
      const first = await provider.catalogSnapshot();
      const prepared = await provider.prepareModel("live-model", first);
      await provider.compactMessages([
        { role: MessageRole.USER, content: "history" },
      ], {
        model: "live-model",
        responsesLite: false,
        preparedModel: prepared,
      });
      const second = await provider.catalogSnapshot();

      assert.equal(catalogRequests, 2);
      assert.equal(first.etag, '"catalog-v1"');
      assert.equal(second.etag, '"catalog-v2"');
    } finally {
      globalThis.fetch = originalFetch;
      fs.rmSync(path.dirname(authPath), { recursive: true, force: true });
    }
  });

  it("validates exact live membership and explicit Claude facade backends", async () => {
    const snapshot = parseModelCatalog({ models: [
      rawModel("gpt-5.6-sol"),
      rawModel("configured-backend", { priority: 20 }),
    ] }, metadata());
    const provider = new ChatGPTOAuthProvider({ model: "configured-backend" });

    assert.equal((await provider.prepareModel("gpt-5.6-sol", snapshot)).slug, "gpt-5.6-sol");
    await assert.rejects(
      () => provider.prepareModel("gpt-5.6", snapshot),
      (error) => error instanceof ChatGPTOAuthModelNotFoundError,
    );
    await assert.rejects(
      () => provider.prepareModel("gpt-5.6-terra", snapshot),
      (error) => error instanceof ChatGPTOAuthModelNotFoundError,
    );
    assert.equal(
      (await provider.prepareAnthropicModel("claude-sonnet-4-5", snapshot)).slug,
      "configured-backend",
    );
    await assert.rejects(
      () => provider.prepareAnthropicModel("gpt-5.6-sol", snapshot),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError,
    );
    await assert.rejects(
      () => new ChatGPTOAuthProvider().prepareAnthropicModel("claude-sonnet-4-5", snapshot),
      (error) => error instanceof ChatGPTOAuthInvalidRequestError,
    );
  });
});
