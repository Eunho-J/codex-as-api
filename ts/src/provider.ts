import * as os from "node:os";
import packageMetadata from "../package.json";
import upstreamContract from "../../config/codex-upstream-contract.json";
import {
  ChatGPTOAuthError,
  ChatGPTOAuthCatalogUnavailableError,
  ChatGPTOAuthInvalidRequestError,
  ChatGPTOAuthModelNotFoundError,
  ChatGPTOAuthProtocolError,
  ChatGPTOAuthRefreshError,
  ChatGPTOAuthUnavailableError,
  ChatGPTOAuthUpstreamError,
  type ChatGPTTokenData,
  loadTokenData,
  redactText,
  refreshAfterUnauthorized,
  tokenForRequest,
} from "./auth.js";
import type {
  AssistantResponse,
  FinishReason,
  Message,
  MessageContentPart,
  ToolCall,
  ToolSchema,
  Usage,
} from "./messages.js";
import { MessageRole } from "./messages.js";
import {
  normalizeFinishReason,
  reasoningFromResponseItems,
  reasoningPartsFromResponseItems,
  responseFailureMessage,
} from "./protocol.js";
import {
  LITE_HEADER_NAME,
  LITE_HEADER_VALUE,
  accountIdFromModelCatalogCacheKey,
  applyModelCapabilityFields,
  buildCodexClientMetadata,
  ModelCatalogCache,
  modelCatalogCacheKey,
  modelFromSnapshot,
  resolveCodexMetadataEnabled,
  shouldEnableParallelToolCalls,
  useResponsesLite,
} from "./model-capabilities.js";
import type {
  ModelCapability,
  ModelCatalogSnapshot,
} from "./model-capabilities.js";
import { parseJsonResponseStrict, parseJsonStrict } from "./utf8-json.js";

export const CHATGPT_OAUTH_DEFAULT_BASE_URL =
  "https://chatgpt.com/backend-api/codex";
const DEFAULT_REQUEST_TIMEOUT_MS = 300_000;
export const MODEL_CATALOG_TIMEOUT_MS = 5_000;
const REMOTE_COMPACTION_MARKER = "[Remote Responses compacted history]";
const CODEX_CLI_ORIGINATOR = "codex_cli_rs";
const CODEX_COMPATIBILITY_VERSION = requireCodexCliVersion(
  upstreamContract.upstream.version,
  "bundled Codex upstream contract version",
);
const CODEX_AS_API_VERSION = requireCodexCliVersion(
  packageMetadata.version,
  "codex-as-api package version",
);
const RESPONSES_LITE_PAYLOAD = Symbol("responses-lite-payload");
const REASONING_MODES = new Set(["standard", "pro"]);
const REASONING_CONTEXTS = new Set(["auto", "current_turn", "all_turns"]);
const IMAGE_DETAILS = new Set(["auto", "low", "high", "original"]);
const RESPONSE_EVENT_TYPES = new Set([
  "error",
  "codex.response.metadata",
  "response.created",
  "response.in_progress",
  "response.metadata",
  "response.queued",
  "response.output_item.added",
  "response.output_item.done",
  "response.content_part.added",
  "response.content_part.done",
  "response.output_text.delta",
  "response.output_text.done",
  "response.function_call_arguments.delta",
  "response.function_call_arguments.done",
  "response.reasoning_summary_part.added",
  "response.reasoning_summary_part.done",
  "response.reasoning_summary_text.delta",
  "response.reasoning_summary_text.done",
  "response.reasoning_text.delta",
  "response.reasoning_text.done",
  "response.web_search_call.in_progress",
  "response.web_search_call.searching",
  "response.web_search_call.completed",
  "response.image_generation_call.in_progress",
  "response.image_generation_call.generating",
  "response.image_generation_call.partial_image",
  "response.image_generation_call.completed",
  "response.failed",
  "response.incomplete",
  "response.completed",
  "responsesapi.websocket_timing",
]);
const UNSUPPORTED_RESPONSE_EVENT_TYPES = new Set([
  "response.file_search_call.in_progress",
  "response.file_search_call.searching",
  "response.file_search_call.completed",
  "response.code_interpreter_call.in_progress",
  "response.code_interpreter_call.interpreting",
  "response.code_interpreter_call_code.delta",
  "response.code_interpreter_call_code.done",
  "response.code_interpreter_call.completed",
  "response.mcp_call.in_progress",
  "response.mcp_call_arguments.delta",
  "response.mcp_call_arguments.done",
  "response.mcp_call.completed",
  "response.mcp_call.failed",
  "response.mcp_list_tools.in_progress",
  "response.mcp_list_tools.completed",
  "response.mcp_list_tools.failed",
  "response.shell_call_command.added",
  "response.shell_call_command.delta",
  "response.shell_call_command.done",
  "response.shell_call_output_content.delta",
  "response.shell_call_output_content.done",
  "response.audio.delta",
  "response.audio.done",
  "response.audio.transcript.delta",
  "response.audio.transcript.done",
  "response.refusal.delta",
  "response.refusal.done",
  "response.output_text.annotation.added",
  "response.custom_tool_call_input.delta",
  "response.custom_tool_call_input.done",
]);
const RESPONSE_CHAIN_CAPACITY = 256;

function callerAbortController(callerSignal?: AbortSignal): {
  controller: AbortController;
  dispose: () => void;
} {
  const controller = new AbortController();
  const abort = () => controller.abort(callerSignal?.reason);
  if (callerSignal?.aborted) {
    abort();
  } else {
    callerSignal?.addEventListener("abort", abort, { once: true });
  }
  return {
    controller,
    dispose: () => callerSignal?.removeEventListener("abort", abort),
  };
}

async function readWithIdleTimeout<T>(
  reader: ReadableStreamDefaultReader<T>,
  controller: AbortController,
  timeoutMs: number,
): Promise<ReadableStreamReadResult<T>> {
  const timeout = setTimeout(() => {
    controller.abort(new Error("ChatGPT OAuth SSE stream exceeded its idle timeout"));
  }, timeoutMs);
  timeout.unref?.();
  try {
    return await reader.read();
  } finally {
    clearTimeout(timeout);
  }
}

function abortAfter(
  controller: AbortController,
  timeoutMs: number,
  message: string,
): () => void {
  const timeout = setTimeout(() => controller.abort(new Error(message)), timeoutMs);
  timeout.unref?.();
  return () => clearTimeout(timeout);
}

interface ResponseChain {
  input: Record<string, unknown>[];
  output: Record<string, unknown>[];
  compHash?: string;
}

class ResponseChainStore {
  private readonly chains = new Map<string, ResponseChain>();

  resolve(
    accountId: string,
    responseId: string,
    currentCompHash?: string,
  ): Record<string, unknown>[] {
    const key = JSON.stringify([accountId, responseId]);
    const chain = this.chains.get(key);
    if (chain == null) {
      throw new ChatGPTOAuthInvalidRequestError(
        "previous_response_id is unknown or has been evicted from the local response history",
      );
    }

    // Map operations are synchronous, so concurrent requests cannot observe a
    // partially updated LRU entry. Returning a clone also lets multiple
    // branches reuse the same completed response without sharing mutations.
    if (
      chain.compHash !== undefined
      && currentCompHash !== undefined
      && chain.compHash !== currentCompHash
    ) {
      throw new ChatGPTOAuthInvalidRequestError(
        "previous_response_id requires compaction because the model compatibility hash changed",
      );
    }
    this.chains.delete(key);
    this.chains.set(key, chain);
    return cloneResponseItems([...chain.input, ...chain.output]);
  }

  commit(
    accountId: string,
    responseId: string,
    input: Record<string, unknown>[],
    output: Record<string, unknown>[],
    compHash?: string,
  ): void {
    const key = JSON.stringify([accountId, responseId]);
    const chain = {
      input: cloneResponseItems(input),
      output: cloneResponseItems(output),
      ...(compHash === undefined ? {} : { compHash }),
    };
    this.chains.delete(key);
    this.chains.set(key, chain);
    while (this.chains.size > RESPONSE_CHAIN_CAPACITY) {
      const oldest = this.chains.keys().next().value as string | undefined;
      if (oldest == null) break;
      this.chains.delete(oldest);
    }
  }
}

function cloneResponseItems(
  items: Record<string, unknown>[],
): Record<string, unknown>[] {
  return structuredClone(items);
}

export function resolveCodexCliVersion(): string {
  return CODEX_COMPATIBILITY_VERSION;
}

function normalizeCodexCliVersion(value: string | undefined): string | undefined {
  if (value == null || value.length === 0 || value !== value.trim()) return undefined;
  const version = value;
  return /^[0-9]+(?:\.[0-9]+){1,3}(?:[-+][0-9A-Za-z.-]+)?$/.test(version)
    ? version
    : undefined;
}

function requireCodexCliVersion(value: string | undefined, source: string): string {
  const version = normalizeCodexCliVersion(value);
  if (version == null) {
    throw new Error(`${source} must be a semantic version`);
  }
  return version;
}

export function codexCliHeadersForVersion(
  version: string | undefined = CODEX_COMPATIBILITY_VERSION,
): Record<string, string> {
  const normalized = requireCodexCliVersion(
    version,
    "Codex compatibility version",
  );
  if (normalized !== CODEX_COMPATIBILITY_VERSION) {
    throw new ChatGPTOAuthInvalidRequestError(
      `Codex compatibility version is pinned to ${CODEX_COMPATIBILITY_VERSION}`,
    );
  }
  return {
    originator: CODEX_CLI_ORIGINATOR,
    "User-Agent": sanitizeHeaderValue(
      `${CODEX_CLI_ORIGINATOR}/${normalized} (${codexOsInfo()}) codex-as-api/${CODEX_AS_API_VERSION}`,
    ),
  };
}

function codexCliHeaders(): Record<string, string> {
  return codexCliHeadersForVersion();
}

function codexOsInfo(): string {
  return `${codexOsName()} ${os.release()}; ${os.arch()}`;
}

function codexOsName(): string {
  const platform = os.platform();
  switch (platform) {
    case "darwin":
      return "Mac OS";
    case "win32":
      return "Windows";
    case "linux":
      return "Linux";
    default:
      return platform;
  }
}

function sanitizeHeaderValue(value: string): string {
  return value.replace(/[^\x20-\x7E]/g, "_");
}

export interface ChatOptions {
  model?: string;
  tools?: ToolSchema[];
  toolChoice?: string | Record<string, unknown>;
  temperature?: number;
  reasoningEffort?: string;
  reasoning?: ReasoningOptions;
  maxTokens?: number;
  stop?: string | string[];
  promptCacheKey?: string;
  promptCacheOptions?: PromptCacheOptions;
  safetyIdentifier?: string;
  subagent?: string;
  memgenRequest?: boolean;
  previousResponseId?: string;
  serviceTier?: string;
  text?: Record<string, unknown>;
  clientMetadata?: Record<string, unknown>;
  codexMetadata?: boolean;
  responsesLite?: boolean | string;
  parallelToolCalls?: boolean;
  ignoreMaxTokens?: boolean;
  preparedModel?: PreparedModel;
  signal?: AbortSignal;
}

export interface PreparedModel {
  readonly slug: string;
  readonly accountId: string;
  readonly capability: ModelCapability;
  readonly snapshot: ModelCatalogSnapshot;
}

export interface ReasoningOptions {
  effort?: string;
  mode?: "standard" | "pro";
  context?: "auto" | "current_turn" | "all_turns";
}

export interface PromptCacheOptions {
  mode?: "implicit" | "explicit";
  ttl?: "30m";
}

export interface ImageReference {
  image_url: string;
  detail?: "auto" | "low" | "high" | "original";
  prompt_cache_breakpoint?: { mode: "explicit" };
}

export interface StreamEvent {
  type: string;
  [key: string]: unknown;
}

export class ChatGPTOAuthProvider {
  readonly name = "chatgpt_oauth";
  readonly supportsPromptCacheKey = true;

  private model: string | undefined;
  private baseUrl: string;
  private authJsonPath: string | undefined;
  private timeout: number;
  private readonly modelCatalog: ModelCatalogCache;
  private readonly responseChains = new ResponseChainStore();
  private readonly semanticInputs = new WeakMap<
    Record<string, unknown>,
    Record<string, unknown>[]
  >();

  constructor(
    opts: {
      model?: string;
      baseUrl?: string;
      authJsonPath?: string;
      timeout?: number;
      modelCatalogTtlMs?: number;
      now?: () => number;
      monotonicNow?: () => number;
    } = {},
  ) {
    if (
      opts.model != null
      && (typeof opts.model !== "string" || opts.model.trim().length === 0)
    ) {
      throw new ChatGPTOAuthInvalidRequestError("model must be a non-empty string when provided");
    }
    if (opts.model != null && opts.model !== opts.model.trim()) {
      throw new ChatGPTOAuthInvalidRequestError(
        "model must not contain surrounding whitespace",
      );
    }
    if (opts.authJsonPath != null && opts.authJsonPath.trim().length === 0) {
      throw new ChatGPTOAuthInvalidRequestError("authJsonPath must not be blank");
    }
    this.model = opts.model;
    this.baseUrl = normalizeBaseUrl(
      opts.baseUrl ?? CHATGPT_OAUTH_DEFAULT_BASE_URL,
    );
    this.authJsonPath = opts.authJsonPath;
    this.timeout = opts.timeout ?? DEFAULT_REQUEST_TIMEOUT_MS;
    if (!Number.isFinite(this.timeout) || this.timeout <= 0) {
      throw new ChatGPTOAuthInvalidRequestError("timeout must be a positive finite number");
    }
    this.modelCatalog = new ModelCatalogCache(
      opts.modelCatalogTtlMs,
      opts.now,
      opts.monotonicNow,
    );
  }

  async catalogSnapshot(): Promise<ModelCatalogSnapshot> {
    const token = await tokenForRequest(this.authJsonPath);
    const clientVersion = resolveCodexCliVersion();
    const key = modelCatalogCacheKey(token.account_id, this.baseUrl, clientVersion);
    try {
      return await this.modelCatalog.get(key, () => this.fetchModelCatalog(token, clientVersion));
    } catch (err) {
      if (err instanceof ChatGPTOAuthProtocolError) {
        throw new ChatGPTOAuthCatalogUnavailableError(err.message);
      }
      throw err;
    }
  }

  async prepareModel(
    requestedModel?: string,
    snapshot?: ModelCatalogSnapshot,
  ): Promise<PreparedModel> {
    if (
      requestedModel != null
      && (typeof requestedModel !== "string" || requestedModel.trim().length === 0)
    ) {
      throw new ChatGPTOAuthInvalidRequestError("model must be a non-empty string when provided");
    }
    if (requestedModel != null && requestedModel !== requestedModel.trim()) {
      throw new ChatGPTOAuthInvalidRequestError(
        "model must not contain surrounding whitespace",
      );
    }
    const catalog = snapshot ?? await this.catalogSnapshot();
    const accountId = accountIdFromModelCatalogCacheKey(catalog.key);
    const selected = requestedModel ?? this.model ?? catalog.defaultModel?.slug;
    if (selected == null) {
      throw new ChatGPTOAuthCatalogUnavailableError(
        "upstream model catalog has no model with list visibility for default selection",
      );
    }
    if (
      requestedModel == null
      && (selected.trim().length === 0 || selected !== selected.trim())
    ) {
      throw new ChatGPTOAuthCatalogUnavailableError(
        this.model == null
          ? "default model publishes an unusable slug"
          : "configured model is unusable",
      );
    }
    const slug = selected;
    const capability = modelFromSnapshot(catalog, slug);
    if (capability == null) {
      if (requestedModel == null && this.model != null) {
        throw new ChatGPTOAuthCatalogUnavailableError(
          "configured model is unavailable in the authenticated catalog",
        );
      }
      throw new ChatGPTOAuthModelNotFoundError(selected);
    }
    return Object.freeze({ slug, accountId, capability, snapshot: catalog });
  }

  async prepareAnthropicModel(
    clientModel?: string,
    snapshot?: ModelCatalogSnapshot,
  ): Promise<PreparedModel> {
    if (typeof clientModel !== "string" || clientModel.trim().length === 0) {
      throw new ChatGPTOAuthInvalidRequestError("Anthropic requests require an explicit claude-* model facade");
    }
    if (clientModel !== clientModel.trim()) {
      throw new ChatGPTOAuthInvalidRequestError(
        "Anthropic model must not contain surrounding whitespace",
      );
    }
    if (!clientModel.startsWith("claude-")) {
      throw new ChatGPTOAuthInvalidRequestError("Anthropic model must use an explicit claude-* facade");
    }
    if (this.model == null) {
      throw new ChatGPTOAuthInvalidRequestError("Claude facade backend model is not configured");
    }
    try {
      return await this.prepareModel(this.model, snapshot);
    } catch (err) {
      if (err instanceof ChatGPTOAuthModelNotFoundError) {
        throw new ChatGPTOAuthCatalogUnavailableError(
          "configured model is unavailable in the authenticated catalog",
        );
      }
      throw err;
    }
  }

  async chat(
    messages: Message[],
    opts: ChatOptions = {},
  ): Promise<AssistantResponse> {
    const contentParts: string[] = [];
    let reasoningParts: string[] = [];
    const toolCalls: ToolCall[] = [];
    let finishReason: FinishReason = null;
    let sawFinish = false;
    const rawEvents: Record<string, unknown>[] = [];
    let usage: Usage | null = null;
    let responseId: string | null = null;
    const toolCallIds = new Set<string>();

    for await (const event of this.chatStream(messages, opts)) {
      rawEvents.push({ ...event });
      if (event.type === "content") {
        if (typeof event.text !== "string") {
          throw new ChatGPTOAuthProtocolError("content event text must be a string");
        }
        contentParts.push(event.text);
      } else if (
        event.type === "reasoning_delta" ||
        event.type === "reasoning_raw_delta"
      ) {
        if (typeof event.text !== "string") {
          throw new ChatGPTOAuthProtocolError("reasoning event text must be a string");
        }
        reasoningParts.push(event.text);
      } else if (event.type === "tool_call") {
        if (typeof event.id !== "string") {
          throw new ChatGPTOAuthProtocolError("tool_call event requires a string id");
        }
        if (typeof event.name !== "string") {
          throw new ChatGPTOAuthProtocolError("tool_call event requires a string name");
        }
        if (typeof event.arguments !== "string") {
          throw new ChatGPTOAuthProtocolError("tool_call event arguments must be a string");
        }
        if (toolCallIds.has(event.id)) {
          throw new ChatGPTOAuthProtocolError(
            `provider response contains duplicate call_id ${JSON.stringify(event.id)}`,
          );
        }
        toolCallIds.add(event.id);
        toolCalls.push({
          id: event.id,
          name: event.name,
          arguments: event.arguments,
        });
      } else if (event.type === "finish") {
        finishReason = normalizeFinishReason(event.finish_reason);
        sawFinish = true;
        if (typeof event.reasoning_content === "string") {
          reasoningParts = [event.reasoning_content];
        }
        if (event.usage == null) {
          usage = null;
        } else {
          usage = usageFromResponse(event.usage);
          if (usage == null) {
            throw new ChatGPTOAuthProtocolError("finish event usage is malformed");
          }
        }
        if (typeof event.response_id !== "string" || event.response_id.length === 0) {
          throw new ChatGPTOAuthProtocolError("finish event requires response_id");
        }
        responseId = event.response_id;
      } else if (event.type === "reasoning_section_break") {
        continue;
      } else if (event.type === "web_search_call") {
        if (typeof event.id !== "string") {
          throw new ChatGPTOAuthProtocolError("web_search_call event requires a string id");
        }
        if (typeof event.input !== "object" || event.input === null || Array.isArray(event.input)) {
          throw new ChatGPTOAuthProtocolError("web_search_call event input must be an object");
        }
        if (!Array.isArray(event.content)) {
          throw new ChatGPTOAuthProtocolError("web_search_call event content must be an array");
        }
      } else {
        throw new ChatGPTOAuthProtocolError(
          "provider emitted an unsupported event type",
        );
      }
    }

    if (responseId == null || !sawFinish) {
      throw new ChatGPTOAuthProtocolError("provider stream ended before a valid finish event");
    }

    return {
      content: contentParts.join(""),
      tool_calls: toolCalls,
      finish_reason: finishReason,
      usage,
      reasoning_content: reasoningParts.join("") || null,
      raw: { events: compactRawEvents(rawEvents) },
      response_id: responseId,
    };
  }

  async *chatStream(
    messages: Message[],
    opts: ChatOptions = {},
  ): AsyncGenerator<StreamEvent> {
    const stream = await this.createChatStream(messages, opts);
    yield* stream;
  }

  async createChatStream(
    messages: Message[],
    opts: ChatOptions = {},
  ): Promise<AsyncGenerator<StreamEvent>> {
    const subagent = opts.subagent == null
      ? undefined
      : requireSubagentHeaderValue(opts.subagent);
    const prepared = opts.preparedModel ?? await this.prepareModel(opts.model);
    const payload = this.responsesPayload(messages, opts, prepared);
    const extraHeaders: Record<string, string> = {};
    if (subagent != null) {
      extraHeaders["x-openai-subagent"] = subagent;
    }
    if (opts.memgenRequest != null) {
      extraHeaders["x-openai-memgen-request"] = opts.memgenRequest
        ? "true"
        : "false";
    }

    return this.streamChatPayload(payload, extraHeaders, prepared, opts.signal);
  }

  private async *streamChatPayload(
    payload: Record<string, unknown>,
    extraHeaders: Record<string, string>,
    prepared: PreparedModel,
    signal?: AbortSignal,
  ): AsyncGenerator<StreamEvent> {
    const semanticInput = this.semanticInputs.get(payload);
    if (semanticInput == null) {
      throw new ChatGPTOAuthProtocolError(
        "internal error: missing semantic input for response history",
      );
    }
    const finalOutput: Record<string, unknown>[] = [];
    const reasoningSummaryParts: string[] = [];
    const reasoningRawParts: string[] = [];
    const streamedTextParts: string[] = [];
    const yieldedWebSearchIds = new Set<string>();
    const yieldedToolCallIds = new Set<string>();
    let sawTextDelta = false;
    let sawReasoningSummaryDelta = false;
    let sawReasoningRawDelta = false;
    let sawToolCall = false;

    for await (const event of this.postSSE(
      "/responses",
      payload,
      extraHeaders,
      prepared.accountId,
      prepared.snapshot.key,
      signal,
    )) {
      if (!validateResponseEvent(event)) {
        continue;
      }
      const typ = event.type;

      if (typ === "response.output_text.delta") {
        const delta = event.delta;
        if (typeof delta !== "string") {
          throw new ChatGPTOAuthProtocolError("response.output_text.delta delta must be a string");
        }
        if (delta) {
          sawTextDelta = true;
          streamedTextParts.push(delta);
          yield { type: "content", text: delta };
        }
      } else if (typ === "response.output_item.done") {
        const item = event.item;
        if (typeof item !== "object" || item === null || Array.isArray(item)) {
          throw new ChatGPTOAuthProtocolError(
            "response.output_item.done must contain an object item",
          );
        }
        const itemDict = item as Record<string, unknown>;
        validateChatResponseItem(itemDict);
        finalOutput.push(itemDict);
        const tool = toolCallFromResponseItem(itemDict);
        if (tool) {
          if (yieldedToolCallIds.has(tool.id)) {
            throw new ChatGPTOAuthProtocolError(
              `provider response contains duplicate call_id ${JSON.stringify(tool.id)}`,
            );
          }
          sawToolCall = true;
          yieldedToolCallIds.add(tool.id);
          yield {
            type: "tool_call",
            id: tool.id,
            name: tool.name,
            arguments: tool.arguments,
          };
        }
        const webSearch = webSearchEventFromResponseItem(itemDict);
        if (webSearch) {
          yieldedWebSearchIds.add(String(webSearch.id));
          yield webSearch;
        }
      } else if (typ === "response.reasoning_summary_part.added") {
        yield {
          type: "reasoning_section_break",
          summary_index: event.summary_index,
        };
      } else if (typ === "response.reasoning_summary_text.delta") {
        const delta = event.delta;
        if (typeof delta !== "string") {
          throw new ChatGPTOAuthProtocolError("response.reasoning_summary_text.delta delta must be a string");
        }
        sawReasoningSummaryDelta = true;
        reasoningSummaryParts.push(delta);
        if (delta) {
          yield {
            type: "reasoning_delta",
            text: delta,
            summary_index: event.summary_index,
          };
        }
      } else if (typ === "response.reasoning_text.delta") {
        const delta = event.delta;
        if (typeof delta !== "string") {
          throw new ChatGPTOAuthProtocolError("response.reasoning_text.delta delta must be a string");
        }
        sawReasoningRawDelta = true;
        reasoningRawParts.push(delta);
        if (delta) {
          yield {
            type: "reasoning_raw_delta",
            text: delta,
            content_index: event.content_index,
          };
        }
      } else if (typ === "error") {
        throw new ChatGPTOAuthUpstreamError(502, responseFailureMessage(event, "error"));
      } else if (typ === "response.failed") {
        throw new ChatGPTOAuthUpstreamError(502, responseFailureMessage(event, "failed"));
      } else if (typ === "response.incomplete") {
        throw new ChatGPTOAuthUpstreamError(
          502,
          responseFailureMessage(event, "incomplete"),
        );
      } else if (typ === "response.completed") {
        const response = completedResponseFromEvent(event);
        const usageData = response.usage;
        if (usageData != null && usageFromResponse(usageData) == null) {
          throw new ChatGPTOAuthProtocolError(
            "response.completed response.usage is malformed",
          );
        }
        const completedOutput = finalOutput;
        const completedText = textFromResponseItems(completedOutput);
        if (sawTextDelta && streamedTextParts.join("") !== completedText) {
          throw new ChatGPTOAuthProtocolError(
            "response.completed output text does not match streamed output text",
          );
        }
        const completedReasoning = reasoningPartsFromResponseItems(completedOutput);
        if (
          sawReasoningSummaryDelta
          && reasoningSummaryParts.join("") !== completedReasoning.summary
        ) {
          throw new ChatGPTOAuthProtocolError(
            "response.completed reasoning summary does not match streamed reasoning summary",
          );
        }
        if (
          sawReasoningRawDelta
          && reasoningRawParts.join("") !== completedReasoning.content
        ) {
          throw new ChatGPTOAuthProtocolError(
            "response.completed reasoning content does not match streamed reasoning content",
          );
        }
        this.responseChains.commit(
          prepared.accountId,
          response.id as string,
          semanticInput,
          completedOutput,
          prepared.capability.compHash,
        );
        for (const item of completedOutput) {
          const tool = toolCallFromResponseItem(item);
          if (tool && !yieldedToolCallIds.has(tool.id)) {
            sawToolCall = true;
            yieldedToolCallIds.add(tool.id);
            yield {
              type: "tool_call",
              id: tool.id,
              name: tool.name,
              arguments: tool.arguments,
            };
          }
          const webSearch = webSearchEventFromResponseItem(item);
          if (webSearch && !yieldedWebSearchIds.has(String(webSearch.id))) {
            yieldedWebSearchIds.add(String(webSearch.id));
            yield webSearch;
          }
        }
        if (!sawTextDelta) {
          if (completedText) {
            sawTextDelta = true;
            yield { type: "content", text: completedText };
          }
        }
        if (!sawReasoningSummaryDelta && completedReasoning.summary) {
          reasoningSummaryParts.push(completedReasoning.summary);
          yield { type: "reasoning_delta", text: completedReasoning.summary };
        }
        if (!sawReasoningRawDelta && completedReasoning.content) {
          reasoningRawParts.push(completedReasoning.content);
          yield { type: "reasoning_raw_delta", text: completedReasoning.content };
        }
        yield {
          type: "finish",
          finish_reason: sawToolCall
            ? "tool_calls"
            : response.end_turn === false
              ? null
              : "stop",
          ...(usageData == null ? {} : { usage: usageData }),
          reasoning_content:
            (reasoningSummaryParts.join("") + reasoningRawParts.join("")) || null,
          response_id: response.id,
        };
        return;
      }
    }
    throw new ChatGPTOAuthProtocolError(
      "ChatGPT OAuth response stream ended before response.completed",
    );
  }

  async generateImage(
    prompt: string,
    opts: {
      model?: string;
      referenceImages?: ImageReference[];
      size?: string;
      reasoningEffort?: string;
      reasoning?: ReasoningOptions;
      safetyIdentifier?: string;
      promptCacheOptions?: PromptCacheOptions;
      text?: Record<string, unknown>;
      responsesLite?: boolean | string;
      preparedModel?: PreparedModel;
    } = {},
  ): Promise<Record<string, unknown>[]> {
    if (!prompt || prompt.trim() === "") {
      throw new ChatGPTOAuthInvalidRequestError("image generation prompt is required");
    }
    const content: Record<string, unknown>[] = [
      { type: "input_text", text: prompt },
    ];
    content.push(
      ...validateImageContentItems(opts.referenceImages ?? []),
    );
    if (opts.size != null && opts.size !== "auto") {
      throw new ChatGPTOAuthInvalidRequestError(
        "image size is not supported by the private Codex OAuth transport",
      );
    }
    const prepared = opts.preparedModel ?? await this.prepareModel(opts.model);
    if (!prepared.capability.inputModalities.includes("image")) {
      throw new ChatGPTOAuthInvalidRequestError(
        "image generation is not supported by the selected model",
      );
    }
    const requestModel = prepared.slug;
    const payload: Record<string, unknown> = {
      model: wireModel(requestModel),
      instructions:
        "Use the image_generation tool to create the requested image. " +
        "Return the generated image through an image_generation_call result.",
      input: [{ type: "message", role: "user", content }],
      tools: [{ type: "image_generation", output_format: "png" }],
      tool_choice: "auto",
      parallel_tool_calls: false,
      stream: true,
      store: false,
      include: [],
    };
    setReasoningPayload(
      payload,
      effectiveReasoningEffort(prepared.capability, opts.reasoningEffort, opts.reasoning),
      opts.reasoning,
      prepared.capability,
    );
    rejectUnsupportedPrivateRequestFields(
      payload,
      opts.safetyIdentifier,
      opts.promptCacheOptions,
    );
    validatePromptCacheBreakpoints(payload);
    finalizeResponsesPayload(payload, {
      endpoint: "responses",
      capability: prepared.capability,
      responsesLite: opts.responsesLite,
      text: opts.text,
      tools: payload.tools as Record<string, unknown>[],
    });
    const outputItems = await this.collectResponseOutputItems(payload, prepared);
    const generated: Record<string, unknown>[] = [];
    for (const item of outputItems) {
      if (item.type === "reasoning") {
        reasoningFromResponseItems([item]);
        continue;
      }
      const image = imageGenerationFromItem(item);
      if (image == null) {
        throw new ChatGPTOAuthProtocolError(
          "image generation response contains an unsupported output item",
        );
      }
      generated.push(image);
    }
    if (!generated.length) {
      throw new ChatGPTOAuthProtocolError(
        "image generation response returned no image_generation_call",
      );
    }
    return generated;
  }

  async inspectImages(
    prompt: string,
    opts: {
      model?: string;
      images: ImageReference[];
      reasoningEffort?: string;
      reasoning?: ReasoningOptions;
      safetyIdentifier?: string;
      promptCacheOptions?: PromptCacheOptions;
      text?: Record<string, unknown>;
      responsesLite?: boolean | string;
      preparedModel?: PreparedModel;
    },
  ): Promise<string> {
    if (!prompt || prompt.trim() === "") {
      throw new ChatGPTOAuthInvalidRequestError("image inspection prompt is required");
    }
    const content: Record<string, unknown>[] = [
      { type: "input_text", text: prompt },
    ];
    content.push(...validateImageContentItems(opts.images));
    const prepared = opts.preparedModel ?? await this.prepareModel(opts.model);
    const requestModel = prepared.slug;
    const payload: Record<string, unknown> = {
      model: wireModel(requestModel),
      instructions:
        "Inspect the attached image(s) and answer the user's review prompt directly.",
      input: [{ type: "message", role: "user", content }],
      tools: [],
      tool_choice: "auto",
      parallel_tool_calls: false,
      stream: true,
      store: false,
      include: [],
    };
    setReasoningPayload(
      payload,
      effectiveReasoningEffort(prepared.capability, opts.reasoningEffort, opts.reasoning),
      opts.reasoning,
      prepared.capability,
    );
    rejectUnsupportedPrivateRequestFields(
      payload,
      opts.safetyIdentifier,
      opts.promptCacheOptions,
    );
    validatePromptCacheBreakpoints(payload);
    finalizeResponsesPayload(payload, {
      endpoint: "responses",
      capability: prepared.capability,
      responsesLite: opts.responsesLite,
      text: opts.text,
      tools: payload.tools as Record<string, unknown>[],
    });
    const outputItems = await this.collectResponseOutputItems(payload, prepared);
    for (const item of outputItems) {
      if (item.type === "reasoning") {
        reasoningFromResponseItems([item]);
      } else {
        validateAssistantMessageOutputItem(item);
      }
    }
    const text = textFromResponseItems(outputItems).trim();
    if (!text) {
      throw new ChatGPTOAuthProtocolError(
        "image inspection response returned empty content",
      );
    }
    return text;
  }


  async compactMessages(
    messages: Message[],
    opts: {
      model?: string;
      reasoningEffort?: string;
      responsesLite?: boolean | string;
      tools?: ToolSchema[];
      promptCacheOptions?: PromptCacheOptions;
      promptCacheKey?: string;
      previousResponseId?: string;
      serviceTier?: string;
      text?: Record<string, unknown>;
      preparedModel?: PreparedModel;
    } = {},
  ): Promise<string> {
    const prepared = opts.preparedModel ?? await this.prepareModel(opts.model);
    const requestModel = prepared.slug;
    const [baseInstructions, compactInput] = splitInstructionsAndInput(messages);
    const semanticInput = this.resolveSemanticInput(
      compactInput,
      opts.previousResponseId,
      prepared.accountId,
      prepared.capability.compHash,
    );
    const toolsPayload = opts.tools?.map(toolSchemaToResponseDict) ?? [];
    const payload: Record<string, unknown> = {
      model: wireModel(requestModel),
      input: semanticInput,
      tools: toolsPayload,
      parallel_tool_calls: false,
    };
    if (baseInstructions) payload.instructions = baseInstructions;
    if (opts.promptCacheKey != null) {
      if (
        typeof opts.promptCacheKey !== "string"
        || opts.promptCacheKey.length === 0
      ) {
        throw new ChatGPTOAuthInvalidRequestError(
          "prompt_cache_key must be a non-empty string when provided",
        );
      }
      payload.prompt_cache_key = opts.promptCacheKey;
    }
    setReasoningPayload(
      payload,
      opts.reasoningEffort ?? prepared.capability.defaultReasoningEffort,
      undefined,
      prepared.capability,
    );
    rejectUnsupportedPrivateRequestFields(
      payload,
      undefined,
      opts.promptCacheOptions,
    );
    validatePromptCacheBreakpoints(payload);
    finalizeResponsesPayload(payload, {
      endpoint: "compact",
      capability: prepared.capability,
      responsesLite: opts.responsesLite,
      serviceTier: opts.serviceTier,
      text: opts.text,
      tools: toolsPayload,
    });
    const data = await this.postJSON(
      "/responses/compact",
      payload,
      prepared.accountId,
      prepared.snapshot.key,
    );
    const output = data.output;
    if (!Array.isArray(output)) {
      throw new ChatGPTOAuthProtocolError(
        "remote compact response missing output array",
      );
    }
    let compacted: Record<string, unknown>[];
    try {
      compacted = filterCompactedHistoryItems(output);
    } catch (err) {
      if (err instanceof ChatGPTOAuthInvalidRequestError) {
        throw new ChatGPTOAuthProtocolError(err.message);
      }
      throw err;
    }
    return REMOTE_COMPACTION_MARKER + "\n" + JSON.stringify(compacted);
  }

  private async collectResponseOutputItems(
    payload: Record<string, unknown>,
    prepared: PreparedModel,
  ): Promise<Record<string, unknown>[]> {
    const outputItems: Record<string, unknown>[] = [];

    for await (const event of this.postSSE(
      "/responses",
      payload,
      {},
      prepared.accountId,
      prepared.snapshot.key,
    )) {
      if (!validateResponseEvent(event)) {
        continue;
      }
      const typ = event.type;
      if (typ === "response.output_item.done") {
        const item = event.item;
        if (typeof item !== "object" || item === null || Array.isArray(item)) {
          throw new ChatGPTOAuthProtocolError(
            "response.output_item.done must contain an object item",
          );
        }
        outputItems.push(item as Record<string, unknown>);
      } else if (typ === "error") {
        throw new ChatGPTOAuthUpstreamError(
          502,
          responseFailureMessage(event, "error"),
        );
      } else if (typ === "response.failed") {
        throw new ChatGPTOAuthUpstreamError(
          502,
          responseFailureMessage(event, "failed"),
        );
      } else if (typ === "response.incomplete") {
        throw new ChatGPTOAuthUpstreamError(
          502,
          responseFailureMessage(event, "incomplete"),
        );
      } else if (typ === "response.completed") {
        const response = completedResponseFromEvent(event);
        if (response.usage != null && usageFromResponse(response.usage) == null) {
          throw new ChatGPTOAuthProtocolError(
            "response.completed response.usage is malformed",
          );
        }
        return outputItems;
      }
    }
    throw new ChatGPTOAuthProtocolError(
      "ChatGPT OAuth response stream ended before response.completed",
    );
  }

  private responsesPayload(
    messages: Message[],
    opts: ChatOptions,
    prepared: PreparedModel,
  ): Record<string, unknown> {
    rejectUnsupportedStop(opts.stop);
    const [instructions, inputItems] =
      splitInstructionsAndInput(messages);
    const requestModel = prepared.slug;
    const semanticInput = this.resolveSemanticInput(
      inputItems,
      opts.previousResponseId,
      prepared.accountId,
      prepared.capability.compHash,
    );
    const toolsPayload = opts.tools
      ? opts.tools.map(toolSchemaToResponseDict)
      : [];
    const lite = useResponsesLite(prepared.capability, opts.responsesLite);
    const payload: Record<string, unknown> = {
      model: wireModel(requestModel),
      input: semanticInput,
      tools: toolsPayload,
      tool_choice: opts.toolChoice ?? "auto",
      parallel_tool_calls: shouldEnableParallelToolCalls({
        requested: opts.parallelToolCalls,
        responsesLite: lite,
      }),
      stream: true,
      store: false,
      include: [],
    };
    if (instructions.length > 0) payload.instructions = instructions;
    if ((payload.tools as Record<string, unknown>[]).some((tool) => tool.type === "web_search")) {
      payload.include = ["web_search_call.action.sources"];
    }
    if (opts.maxTokens != null && opts.ignoreMaxTokens !== true) {
      throw new ChatGPTOAuthInvalidRequestError(
        "max_tokens and max_completion_tokens are not supported by the private Codex OAuth HTTP transport",
      );
    }
    if (opts.temperature != null) {
      throw new ChatGPTOAuthInvalidRequestError(
        "temperature is not supported by the private Codex OAuth HTTP transport",
      );
    }
    let clientMetadata = normalizeClientMetadata(opts.clientMetadata);
    if (resolveCodexMetadataEnabled(opts.codexMetadata)) {
      clientMetadata = buildCodexClientMetadata({
        authJsonPath: this.authJsonPath,
        existing: clientMetadata,
      });
    }
    if (clientMetadata != null) {
      payload.client_metadata = clientMetadata;
    }
    if (opts.promptCacheKey != null) {
      if (
        typeof opts.promptCacheKey !== "string"
        || opts.promptCacheKey.trim().length === 0
      ) {
        throw new ChatGPTOAuthInvalidRequestError(
          "prompt_cache_key must be a non-empty string when provided",
        );
      }
      payload.prompt_cache_key = opts.promptCacheKey;
    } else {
      const sessionId = sessionIdFromClientMetadata(clientMetadata);
      if (sessionId != null) {
        payload.prompt_cache_key = sessionId;
      }
    }
    setReasoningPayload(
      payload,
      effectiveReasoningEffort(prepared.capability, opts.reasoningEffort, opts.reasoning),
      opts.reasoning,
      prepared.capability,
    );
    rejectUnsupportedPrivateRequestFields(
      payload,
      opts.safetyIdentifier,
      opts.promptCacheOptions,
    );
    validatePromptCacheBreakpoints(payload);
    finalizeResponsesPayload(payload, {
      endpoint: "responses",
      capability: prepared.capability,
      responsesLite: lite,
      serviceTier: opts.serviceTier,
      text: opts.text,
      tools: toolsPayload,
    });
    this.semanticInputs.set(payload, cloneResponseItems(semanticInput));
    return payload;
  }

  private resolveSemanticInput(
    input: Record<string, unknown>[],
    previousResponseId: string | undefined,
    accountId: string,
    currentCompHash?: string,
  ): Record<string, unknown>[] {
    if (previousResponseId == null) return cloneResponseItems(input);
    if (
      typeof previousResponseId !== "string"
      || previousResponseId.trim().length === 0
    ) {
      throw new ChatGPTOAuthInvalidRequestError(
        "previous_response_id must be a non-empty string when provided",
      );
    }
    return [
      ...this.responseChains.resolve(accountId, previousResponseId, currentCompHash),
      ...cloneResponseItems(input),
    ];
  }

  private async fetchModelCatalog(
    initialToken: ChatGPTTokenData,
    clientVersion: string,
  ): Promise<{ value: unknown; etag: string | null }> {
    let token = initialToken;
    for (let attempt = 0; attempt < 2; attempt++) {
      const url = new URL(`${this.baseUrl}/models`);
      url.searchParams.set("client_version", clientVersion);
      let response: globalThis.Response;
      try {
        response = await fetch(url, {
          method: "GET",
          redirect: "manual",
          headers: this.getHeaders(token),
          signal: AbortSignal.timeout(MODEL_CATALOG_TIMEOUT_MS),
        });
      } catch (err) {
        throw new ChatGPTOAuthCatalogUnavailableError(
          `upstream model catalog request failed: ${redactText(
            String(err),
            token.access_token,
            token.refresh_token,
            token.id_token,
            token.account_id,
          )}`,
        );
      }
      if (response.status === 401 && attempt === 0) {
        await cancelResponseBody(response, [
          token.access_token,
          token.refresh_token,
          token.id_token,
          token.account_id,
        ], "catalog");
        const refreshed = await refreshAfterUnauthorized(token);
        if (refreshed.account_id !== initialToken.account_id) {
          throw new ChatGPTOAuthRefreshError(
            "ChatGPT OAuth account changed while refreshing the model catalog",
          );
        }
        token = refreshed;
        continue;
      }
      if (!response.ok) {
        const body = await readRedactedResponseBody(response, [
          token.access_token,
          token.refresh_token,
          token.id_token,
          token.account_id,
        ]);
        if (response.status === 401) {
          throw new ChatGPTOAuthUpstreamError(401, `upstream model catalog authentication failed: ${body}`);
        }
        throw new ChatGPTOAuthUpstreamError(
          response.status,
          `upstream model catalog request failed: HTTP ${response.status}: ${body}`,
        );
      }
      let value: unknown;
      try {
        value = await parseJsonResponseStrict(response);
      } catch {
        throw new ChatGPTOAuthCatalogUnavailableError(
          "upstream model catalog is not valid JSON",
        );
      }
      return { value, etag: response.headers.get("etag") };
    }
    throw new ChatGPTOAuthUpstreamError(401, "upstream model catalog authentication failed");
  }

  private getHeaders(token = loadTokenData(this.authJsonPath)): Record<string, string> {
    const headers: Record<string, string> = {
      ...codexCliHeaders(),
      Authorization: `Bearer ${token.access_token}`,
      "ChatGPT-Account-Id": token.account_id,
      "Content-Type": "application/json",
    };
    if (token.fedramp) {
      headers["X-OpenAI-Fedramp"] = "true";
    }
    return headers;
  }

  private async postJSON(
    path: string,
    payload: Record<string, unknown>,
    expectedAccountId: string,
    catalogKey: string,
  ): Promise<Record<string, unknown>> {
    let tokenValues: (string | null)[] = [null];
    for (let attempt = 0; attempt < 2; attempt++) {
      const token = await tokenForRequest(this.authJsonPath);
      requireExpectedAccount(token, expectedAccountId);
      const headers = this.getHeaders(token);
      if (isResponsesLitePayload(payload)) {
        headers[LITE_HEADER_NAME] = LITE_HEADER_VALUE;
      }
      tokenValues = [
        token.access_token,
        token.refresh_token,
        token.id_token,
        token.account_id,
      ];

      const url = this.baseUrl + path;
      let response: Response;
      try {
        response = await fetch(url, {
          method: "POST",
          redirect: "manual",
          headers,
          body: JSON.stringify(payload),
          signal: AbortSignal.timeout(this.timeout),
        });
      } catch (err) {
        throw new ChatGPTOAuthUnavailableError(
          `ChatGPT OAuth request failed: ${redactText(String(err), ...tokenValues)}`,
        );
      }

      if (!response.ok) {
        if (response.status === 401 && attempt === 0) {
          await cancelResponseBody(response, tokenValues, "request");
          await refreshAfterUnauthorized(token);
          continue;
        }
        const redacted = await readRedactedResponseBody(response, tokenValues);
        throw new ChatGPTOAuthUpstreamError(
          response.status,
          `ChatGPT OAuth request failed: HTTP ${response.status}: ${redacted}`,
        );
      }

      this.modelCatalog.invalidateOnEtagMismatch(
        catalogKey,
        response.headers.get("x-models-etag"),
      );

      let data: unknown;
      try {
        data = await parseJsonResponseStrict(response);
      } catch {
        throw new ChatGPTOAuthProtocolError("ChatGPT OAuth response is not valid JSON");
      }
      if (
        typeof data !== "object" ||
        data === null ||
        Array.isArray(data)
      ) {
        throw new ChatGPTOAuthProtocolError(
          "ChatGPT OAuth response must be a JSON object",
        );
      }
      return data as Record<string, unknown>;
    }
    throw new Error("unreachable");
  }

  private async *postSSE(
    path: string,
    payload: Record<string, unknown>,
    extraHeaders: Record<string, string>,
    expectedAccountId: string,
    catalogKey: string,
    callerSignal?: AbortSignal,
  ): AsyncGenerator<Record<string, unknown>> {
    let tokenValues: (string | null)[] = [null];
    for (let attempt = 0; attempt < 2; attempt++) {
      const token = await tokenForRequest(this.authJsonPath);
      requireExpectedAccount(token, expectedAccountId);
      const headers = this.getHeaders(token);
      headers.Accept = "text/event-stream";
      Object.assign(headers, extraHeaders);
      if (isResponsesLitePayload(payload)) {
        headers[LITE_HEADER_NAME] = LITE_HEADER_VALUE;
      }
      tokenValues = [
        token.access_token,
        token.refresh_token,
        token.id_token,
        token.account_id,
      ];

      const url = this.baseUrl + path;
      const requestAbort = callerAbortController(callerSignal);
      const clearHeaderTimeout = abortAfter(
        requestAbort.controller,
        this.timeout,
        "ChatGPT OAuth request exceeded its response header idle timeout",
      );
      let response: globalThis.Response;
      try {
        response = await fetch(url, {
          method: "POST",
          redirect: "manual",
          headers,
          body: JSON.stringify(payload),
          signal: requestAbort.controller.signal,
        });
      } catch (err) {
        clearHeaderTimeout();
        requestAbort.dispose();
        throw new ChatGPTOAuthUnavailableError(
          `ChatGPT OAuth request failed: ${redactText(String(err), ...tokenValues)}`,
        );
      }
      clearHeaderTimeout();

      if (!response.ok) {
        if (response.status === 401 && attempt === 0) {
          try {
            await cancelResponseBody(response, tokenValues, "request");
          } finally {
            requestAbort.dispose();
          }
          await refreshAfterUnauthorized(token);
          continue;
        }
        const clearErrorBodyTimeout = abortAfter(
          requestAbort.controller,
          this.timeout,
          "ChatGPT OAuth error response body exceeded its idle timeout",
        );
        const redacted = await readRedactedResponseBody(response, tokenValues);
        clearErrorBodyTimeout();
        requestAbort.dispose();
        throw new ChatGPTOAuthUpstreamError(
          response.status,
          `ChatGPT OAuth request failed: HTTP ${response.status}: ${redacted}`,
        );
      }

      this.modelCatalog.invalidateOnEtagMismatch(
        catalogKey,
        response.headers.get("x-models-etag"),
      );

      if (response.body == null) {
        requestAbort.dispose();
        throw new ChatGPTOAuthProtocolError("ChatGPT OAuth streaming response has no body");
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder("utf-8", { fatal: true });
      let buffer = "";
      const block: string[] = [];
      let reachedEof = false;
      let streamFailure: unknown;

      try {
        while (true) {
          const { done, value } = await readWithIdleTimeout(
            reader,
            requestAbort.controller,
            this.timeout,
          );
          if (done) {
            reachedEof = true;
            try {
              buffer += decoder.decode();
            } catch (err) {
              throw new ChatGPTOAuthProtocolError(
                `ChatGPT OAuth SSE stream is not valid UTF-8: ${String(err)}`,
              );
            }
            if (buffer.length > 0) {
              for (const rawLine of buffer.split("\n")) {
                const line = rawLine.replace(/\r$/, "");
                if (line === "") {
                  const event = decodeSSEBlock(block);
                  block.length = 0;
                  if (event) yield redactSSEEvent(event, tokenValues);
                } else {
                  block.push(line);
                }
              }
            }
            if (block.length) {
              const event = decodeSSEBlock(block);
              if (event) yield redactSSEEvent(event, tokenValues);
            }
            return;
          }
          try {
            buffer += decoder.decode(value, { stream: true });
          } catch (err) {
            throw new ChatGPTOAuthProtocolError(
              `ChatGPT OAuth SSE stream is not valid UTF-8: ${String(err)}`,
            );
          }
          const lines = buffer.split("\n");
          buffer = lines.pop()!;
          for (const rawLine of lines) {
            const line = rawLine.replace(/\r$/, "");
            if (line === "") {
              const event = decodeSSEBlock(block);
              block.length = 0;
              if (event) yield redactSSEEvent(event, tokenValues);
              continue;
            }
            block.push(line);
          }
        }
      } catch (err) {
        streamFailure = err;
        if (err instanceof ChatGPTOAuthProtocolError) {
          throw new ChatGPTOAuthProtocolError(
            redactText(err.message, ...tokenValues),
          );
        }
        if (err instanceof ChatGPTOAuthError) throw err;
        throw new ChatGPTOAuthUnavailableError(
          `ChatGPT OAuth request failed: ${redactText(String(err), ...tokenValues)}`,
        );
      } finally {
        if (!reachedEof) {
          try {
            await reader.cancel();
          } catch (err) {
            if (streamFailure === undefined) {
              throw new ChatGPTOAuthUnavailableError(
                `ChatGPT OAuth stream cancellation failed: ${redactText(String(err), ...tokenValues)}`,
              );
            }
          }
        }
        try {
          reader.releaseLock();
        } catch (err) {
          if (streamFailure === undefined) {
            throw new ChatGPTOAuthUnavailableError(
              `ChatGPT OAuth stream reader release failed: ${redactText(String(err), ...tokenValues)}`,
            );
          }
        }
        requestAbort.dispose();
      }
      return;
    }
  }
}

function redactSSEEvent(
  event: Record<string, unknown>,
  tokenValues: (string | null | undefined)[],
): Record<string, unknown> {
  if (!["error", "response.failed", "response.incomplete"].includes(String(event.type))) {
    return event;
  }
  return redactStructuredStrings(event, tokenValues) as Record<string, unknown>;
}

function redactStructuredStrings(
  value: unknown,
  tokenValues: (string | null | undefined)[],
): unknown {
  if (typeof value === "string") return redactText(value, ...tokenValues);
  if (Array.isArray(value)) {
    return value.map((item) => redactStructuredStrings(item, tokenValues));
  }
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        redactStructuredStrings(item, tokenValues),
      ]),
    );
  }
  return value;
}

async function readRedactedResponseBody(
  response: globalThis.Response,
  tokenValues: (string | null | undefined)[],
): Promise<string> {
  try {
    return redactText(await response.text(), ...tokenValues);
  } catch (err) {
    return `unable to read upstream error body: ${redactText(String(err), ...tokenValues)}`;
  }
}

async function cancelResponseBody(
  response: globalThis.Response,
  tokenValues: (string | null | undefined)[],
  context: "catalog" | "request",
): Promise<void> {
  if (response.body == null) return;
  try {
    await response.body.cancel();
  } catch (err) {
    const detail = redactText(String(err), ...tokenValues);
    if (context === "catalog") {
      throw new ChatGPTOAuthCatalogUnavailableError(
        `upstream model catalog response cancellation failed: ${detail}`,
      );
    }
    throw new ChatGPTOAuthUnavailableError(
      `ChatGPT OAuth response cancellation failed: ${detail}`,
    );
  }
}

function normalizeBaseUrl(value: string): string {
  if (typeof value !== "string" || value.length === 0) {
    throw new ChatGPTOAuthInvalidRequestError("baseUrl must not be blank");
  }
  if (value.trim() !== value) {
    throw new ChatGPTOAuthInvalidRequestError("baseUrl must not contain surrounding whitespace");
  }
  if (/[\p{White_Space}\p{Cc}]/u.test(value)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "baseUrl must not contain raw whitespace or control characters",
    );
  }
  if (/%(?![0-9A-Fa-f]{2})/.test(value)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "baseUrl must not contain malformed percent encoding",
    );
  }
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new ChatGPTOAuthInvalidRequestError("baseUrl must be a valid HTTP(S) URL");
  }
  const hasExplicitAuthority = value
    .split("://", 2)[1]
    ?.split(/[/?#]/, 1)[0]
    .length;
  if (
    (url.protocol !== "https:" && url.protocol !== "http:")
    || url.hostname.length === 0
    || !hasExplicitAuthority
    || url.username.length > 0
    || url.password.length > 0
    || url.search.length > 0
    || url.hash.length > 0
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "baseUrl must be an HTTP(S) URL without credentials, query, or fragment",
    );
  }
  return value.replace(/\/+$/, "");
}

function normalizeClientMetadata(
  value: Record<string, unknown> | undefined,
): Record<string, string> | undefined {
  if (value == null) return undefined;
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "client_metadata must be an object when provided",
    );
  }
  for (const [field, fieldValue] of Object.entries(value)) {
    if (typeof fieldValue !== "string") {
      throw new ChatGPTOAuthInvalidRequestError(
        `client_metadata.${field} must be a string`,
      );
    }
  }
  for (const field of ["session_id", "thread_id"] as const) {
    const fieldValue = value[field];
    if (
      Object.hasOwn(value, field)
      && typeof fieldValue === "string"
      && fieldValue.trim().length === 0
    ) {
      throw new ChatGPTOAuthInvalidRequestError(
        `client_metadata.${field} must be a non-empty string when provided`,
      );
    }
  }
  return { ...value } as Record<string, string>;
}

function requireExpectedAccount(
  token: ChatGPTTokenData,
  expectedAccountId: string,
): void {
  if (token.account_id !== expectedAccountId) {
    throw new ChatGPTOAuthRefreshError(
      "ChatGPT OAuth account changed after the model catalog snapshot was prepared",
    );
  }
}

function sessionIdFromClientMetadata(
  metadata: Record<string, unknown> | undefined,
): string | undefined {
  if (metadata == null || !Object.hasOwn(metadata, "session_id")) {
    return undefined;
  }
  const sessionId = metadata.session_id;
  return sessionId as string;
}

function rejectUnsupportedStop(stop: string | string[] | undefined): void {
  if (stop == null) return;
  throw new ChatGPTOAuthInvalidRequestError(
    "stop is not supported by the private Codex OAuth HTTP transport",
  );
}

// --- Helper functions (exported for testing) ---

function completedResponseFromEvent(event: Record<string, unknown>): Record<string, unknown> {
  const response = event.response;
  if (typeof response !== "object" || response === null || Array.isArray(response)) {
    throw new ChatGPTOAuthProtocolError(
      "response.completed must contain an object response",
    );
  }
  const responseRecord = response as Record<string, unknown>;
  if (typeof responseRecord.id !== "string" || responseRecord.id.length === 0) {
    throw new ChatGPTOAuthProtocolError(
      "response.completed response.id must be a non-empty string",
    );
  }
  if (
    responseRecord.end_turn !== undefined
    && responseRecord.end_turn !== null
    && typeof responseRecord.end_turn !== "boolean"
  ) {
    throw new ChatGPTOAuthProtocolError(
      "response.completed response.end_turn must be a boolean when provided",
    );
  }
  return responseRecord;
}

export function decodeSSEBlock(
  lines: string[],
): Record<string, unknown> | null {
  const dataLines = lines
    .filter((l) => l.startsWith("data:"))
    .map((l) => l.slice(5).trim());
  if (!dataLines.length) return null;
  const joined = dataLines.join("\n");
  if (joined === "[DONE]") return null;
  let event: unknown;
  try {
    event = parseJsonStrict(joined);
  } catch {
    throw new ChatGPTOAuthProtocolError("ChatGPT OAuth SSE event is not valid JSON");
  }
  if (typeof event !== "object" || event === null || Array.isArray(event)) {
    throw new ChatGPTOAuthProtocolError("ChatGPT OAuth SSE event must be a JSON object");
  }
  if (typeof (event as Record<string, unknown>).type !== "string") {
    throw new ChatGPTOAuthProtocolError("ChatGPT OAuth SSE event type must be a string");
  }
  return event as Record<string, unknown>;
}

function validateResponseEvent(event: Record<string, unknown>): boolean {
  const eventType = event.type;
  if (typeof eventType !== "string") {
    throw new ChatGPTOAuthProtocolError(
      "ChatGPT OAuth SSE event requires a string type",
    );
  }
  if (eventType === "error") {
    return true;
  }
  if (UNSUPPORTED_RESPONSE_EVENT_TYPES.has(eventType)) {
    throw new ChatGPTOAuthProtocolError(
      "ChatGPT OAuth SSE event has an unsupported semantic type",
    );
  }
  if (!RESPONSE_EVENT_TYPES.has(eventType)) {
    return false;
  }
  if (eventType === "response.created") {
    if (!isRecordValue(event.response)) {
      throw new ChatGPTOAuthProtocolError(
        "response.created must contain an object response",
      );
    }
    return true;
  }
  if (eventType === "response.output_item.added") {
    if (!isRecordValue(event.item)) {
      throw new ChatGPTOAuthProtocolError(
        "response.output_item.added must contain an object item",
      );
    }
    validateAddedResponseItem(event.item);
    return true;
  }
  if (eventType === "response.output_item.done") {
    if (!isRecordValue(event.item)) {
      throw new ChatGPTOAuthProtocolError(
        "response.output_item.done must contain an object item",
      );
    }
    validateResponseOutputItem(event.item);
    return true;
  }
  if (
    eventType === "response.content_part.added"
    || eventType === "response.content_part.done"
  ) {
    if (!isRecordValue(event.part)) {
      throw new ChatGPTOAuthProtocolError(
        `${eventType} must contain an object part`,
      );
    }
    if (event.part.type !== "output_text") {
      throw new ChatGPTOAuthProtocolError(
        `${eventType} has an unsupported semantic part type`,
      );
    }
    if (typeof event.part.text !== "string") {
      throw new ChatGPTOAuthProtocolError(
        `${eventType} output_text part requires a text string`,
      );
    }
    if (Object.hasOwn(event.part, "annotations") && !Array.isArray(event.part.annotations)) {
      throw new ChatGPTOAuthProtocolError(
        `${eventType} output_text annotations must be an array`,
      );
    }
    if (Object.hasOwn(event.part, "logprobs") && !Array.isArray(event.part.logprobs)) {
      throw new ChatGPTOAuthProtocolError(
        `${eventType} output_text logprobs must be an array`,
      );
    }
    return true;
  }
  if (eventType === "response.output_text.delta") {
    if (typeof event.delta !== "string") {
      throw new ChatGPTOAuthProtocolError(
        `${eventType} requires a string delta`,
      );
    }
    return true;
  }
  if (eventType === "response.reasoning_summary_text.delta") {
    if (
      typeof event.delta !== "string"
      || !Number.isSafeInteger(event.summary_index)
    ) {
      throw new ChatGPTOAuthProtocolError(
        "response.reasoning_summary_text.delta requires string delta and integer summary_index",
      );
    }
    return true;
  }
  if (eventType === "response.reasoning_summary_text.done") {
    if (
      typeof event.item_id !== "string"
      || typeof event.text !== "string"
      || !Number.isSafeInteger(event.summary_index)
    ) {
      throw new ChatGPTOAuthProtocolError(
        "response.reasoning_summary_text.done requires string item_id/text and integer summary_index",
      );
    }
    return true;
  }
  if (eventType === "response.reasoning_text.delta") {
    if (
      typeof event.delta !== "string"
      || !Number.isSafeInteger(event.content_index)
    ) {
      throw new ChatGPTOAuthProtocolError(
        "response.reasoning_text.delta requires string delta and integer content_index",
      );
    }
    return true;
  }
  if (eventType === "response.reasoning_summary_part.added") {
    if (!Number.isSafeInteger(event.summary_index)) {
      throw new ChatGPTOAuthProtocolError(
        "response.reasoning_summary_part.added requires integer summary_index",
      );
    }
  }
  return true;
}

function validateAddedResponseItem(item: Record<string, unknown>): void {
  const itemType = item.type;
  if (typeof itemType !== "string" || itemType.length === 0) {
    throw new ChatGPTOAuthProtocolError(
      "response.output_item.added item requires a non-empty string type",
    );
  }
  if (itemType === "custom_tool_call") {
    throw new ChatGPTOAuthProtocolError(
      "custom_tool_call is not supported by the public tool contract",
    );
  }
  validateResponseItemOptionalFields(item, itemType);
  if (itemType === "function_call") {
    requireAddedItemStrings(item, itemType, ["name", "arguments", "call_id"]);
    return;
  }
  if (itemType === "web_search_call") {
    if (item.status != null && typeof item.status !== "string") {
      throw new ChatGPTOAuthProtocolError(
        "response.output_item.added web_search_call status must be a string when provided",
      );
    }
    if (item.action != null && !isRecordValue(item.action)) {
      throw new ChatGPTOAuthProtocolError(
        "response.output_item.added web_search_call action must be an object when provided",
      );
    }
    return;
  }
  if (itemType === "message") {
    if (typeof item.role !== "string" || !Array.isArray(item.content)) {
      throw new ChatGPTOAuthProtocolError(
        "response.output_item.added message requires string role and content array",
      );
    }
    for (const [index, rawPart] of item.content.entries()) {
      if (!isRecordValue(rawPart) || typeof rawPart.type !== "string") {
        throw new ChatGPTOAuthProtocolError(
          `response.output_item.added message content ${index} must be a typed object`,
        );
      }
      const valueField = rawPart.type === "input_image"
        ? "image_url"
        : rawPart.type === "input_audio"
          ? "audio_url"
          : "text";
      if (
        !["input_text", "input_image", "input_audio", "output_text"].includes(rawPart.type)
        || typeof rawPart[valueField] !== "string"
      ) {
        throw new ChatGPTOAuthProtocolError(
          `response.output_item.added message content ${index} is not a valid ResponseItem content part`,
        );
      }
    }
    return;
  }
  if (itemType === "reasoning") {
    validateAddedReasoningParts(item.summary, "summary", new Set(["summary_text"]));
    if (item.content != null) {
      validateAddedReasoningParts(
        item.content,
        "content",
        new Set(["reasoning_text", "text"]),
      );
    }
    if (item.encrypted_content != null && typeof item.encrypted_content !== "string") {
      throw new ChatGPTOAuthProtocolError(
        "response.output_item.added reasoning encrypted_content must be a string when provided",
      );
    }
    return;
  }
  if (itemType === "image_generation_call") {
    imageGenerationFromItem(item);
    return;
  }
  throw new ChatGPTOAuthProtocolError(
    "response.output_item.added item has an unsupported type",
  );
}

function requireAddedItemStrings(
  item: Record<string, unknown>,
  itemType: string,
  fields: string[],
): void {
  for (const field of fields) {
    if (typeof item[field] !== "string") {
      throw new ChatGPTOAuthProtocolError(
        `response.output_item.added ${itemType} requires string ${field}`,
      );
    }
  }
}

function validateAddedReasoningParts(
  value: unknown,
  field: string,
  allowedTypes: ReadonlySet<string>,
): void {
  if (!Array.isArray(value)) {
    throw new ChatGPTOAuthProtocolError(
      `response.output_item.added reasoning requires ${field} array`,
    );
  }
  for (const [index, rawPart] of value.entries()) {
    if (
      !isRecordValue(rawPart)
      || typeof rawPart.type !== "string"
      || !allowedTypes.has(rawPart.type)
      || typeof rawPart.text !== "string"
    ) {
      throw new ChatGPTOAuthProtocolError(
        `response.output_item.added reasoning ${field} ${index} is not a valid ResponseItem part`,
      );
    }
  }
}

function isRecordValue(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function splitInstructionsAndInput(
  messages: Message[],
): [string, Record<string, unknown>[]] {
  const instructions: string[] = [];
  const inputMessages: Message[] = [];
  for (const msg of messages) {
    if (
      msg.role === MessageRole.SYSTEM &&
      !msg.content.startsWith(REMOTE_COMPACTION_MARKER)
    ) {
      if (msg.structured_content?.some((part) => part.prompt_cache_breakpoint != null)) {
        throw new ChatGPTOAuthInvalidRequestError(
          "prompt_cache_breakpoint is not supported by the private Codex OAuth HTTP transport",
        );
      }
      instructions.push(msg.content);
    } else {
      inputMessages.push(msg);
    }
  }
  return [
    instructions.join("\n\n"),
    messagesToResponseItems(inputMessages),
  ];
}

export function messagesToResponseItems(
  messages: Message[],
): Record<string, unknown>[] {
  const items: Record<string, unknown>[] = [];
  for (const message of messages) {
    if (message.structured_content?.some(
      (part) => part.prompt_cache_breakpoint != null,
    )) {
      throw new ChatGPTOAuthInvalidRequestError(
        "prompt_cache_breakpoint is not supported by the private Codex OAuth HTTP transport",
      );
    }
    if (
      message.role === MessageRole.SYSTEM &&
      message.content.startsWith(REMOTE_COMPACTION_MARKER)
    ) {
      const raw = message.content
        .slice(REMOTE_COMPACTION_MARKER.length)
        .trim();
      let parsed: unknown;
      try {
        parsed = parseJsonStrict(raw);
      } catch (err) {
        throw new ChatGPTOAuthInvalidRequestError(
          `remote compaction marker must contain valid JSON: ${String(err)}`,
        );
      }
      if (!Array.isArray(parsed)) {
        throw new ChatGPTOAuthInvalidRequestError(
          "remote compaction marker must contain a response item array",
        );
      }
      items.push(...filterCompactedHistoryItems(parsed, "remote compaction marker"));
      continue;
    }

    if (message.role === MessageRole.TOOL) {
      if (typeof message.tool_call_id !== "string") {
        throw new ChatGPTOAuthInvalidRequestError("tool message requires a string tool_call_id");
      }
      items.push({
        type: "function_call_output",
        call_id: message.tool_call_id,
        output: message.content,
      });
      continue;
    }

    if (
      message.role === MessageRole.ASSISTANT &&
      message.tool_calls?.length
    ) {
      if (message.content || message.structured_content != null) {
        items.push(messageItem(
          "assistant",
          message.content,
          undefined,
          message.structured_content,
        ));
      }
      for (const tc of message.tool_calls) {
        if (typeof tc.id !== "string") {
          throw new ChatGPTOAuthInvalidRequestError("assistant tool call requires a string id");
        }
        if (typeof tc.name !== "string") {
          throw new ChatGPTOAuthInvalidRequestError("assistant tool call requires a string name");
        }
        if (typeof tc.arguments !== "string") {
          throw new ChatGPTOAuthInvalidRequestError("assistant tool call arguments must be a string");
        }
        items.push({
          type: "function_call",
          call_id: tc.id,
          name: tc.name,
          arguments: tc.arguments,
        });
      }
      continue;
    }

    if (
      message.role !== MessageRole.USER
      && message.role !== MessageRole.DEVELOPER
      && message.role !== MessageRole.ASSISTANT
    ) {
      throw new ChatGPTOAuthInvalidRequestError(
        `unsupported internal message role ${JSON.stringify(message.role)}`,
      );
    }
    const role = message.role;
    items.push(messageItem(
      role,
      message.content,
      message.images,
      message.structured_content,
    ));
  }
  return items;
}

export function messageItem(
  role: string,
  content: string,
  images?: string[],
  structuredContent?: MessageContentPart[],
): Record<string, unknown> {
  if (role !== "user" && role !== "developer" && role !== "assistant") {
    throw new ChatGPTOAuthInvalidRequestError("message role must be user, developer, or assistant");
  }
  const typ = role === "assistant" ? "output_text" : "input_text";
  const contentItems: Record<string, unknown>[] = [];
  if (structuredContent != null) {
    for (const part of structuredContent) {
      validatePromptCacheBreakpointValue(
        part.prompt_cache_breakpoint,
        "structured message content",
      );
      if (part.prompt_cache_breakpoint != null) {
        throw new ChatGPTOAuthInvalidRequestError(
          "prompt_cache_breakpoint is not supported by the private Codex OAuth HTTP transport",
        );
      }
      if (part.type === "text") {
        const item: Record<string, unknown> = { type: typ, text: part.text };
        contentItems.push(item);
      } else if (part.type === "image_url") {
        const item: Record<string, unknown> = {
          type: "input_image",
          image_url: part.image_url,
        };
        if (part.detail != null) item.detail = part.detail;
        contentItems.push(item);
      } else {
        contentItems.push({ type: "input_audio", audio_url: part.audio_url });
      }
    }
  } else {
    contentItems.push({ type: typ, text: content });
  }
  if (images && structuredContent == null) {
    for (const imageUrl of images) {
      contentItems.push({ type: "input_image", image_url: imageUrl });
    }
  }
  return {
    type: "message",
    role,
    content: contentItems,
  };
}

export function toolSchemaToResponseDict(
  tool: ToolSchema,
): Record<string, unknown> {
  if (Object.hasOwn(tool, "allowed_callers")) {
    throw new ChatGPTOAuthInvalidRequestError(
      "programmatic tool allowed_callers is not supported",
    );
  }
  if (Object.hasOwn(tool, "output_schema")) {
    throw new ChatGPTOAuthInvalidRequestError(
      "programmatic tool output_schema is not supported",
    );
  }
  if (tool.parameters.__codex_as_api_tool_type === "web_search") {
    const openaiTool = tool.parameters.openai_tool;
    if (
      typeof openaiTool === "object" &&
      openaiTool !== null &&
      !Array.isArray(openaiTool)
    ) {
      return { ...(openaiTool as Record<string, unknown>) };
    }
    throw new ChatGPTOAuthInvalidRequestError(
      "web_search tool metadata must contain an OpenAI tool object",
    );
  }
  const result: Record<string, unknown> = {
    type: "function",
    name: tool.name,
    parameters: tool.parameters,
    strict: tool.strict ?? false,
  };
  if (tool.description !== undefined) result.description = tool.description;
  return result;
}

function finalizeResponsesPayload(
  payload: Record<string, unknown>,
  opts: {
    endpoint: "responses" | "compact";
    capability: ModelCapability;
    responsesLite?: boolean | string;
    serviceTier?: string;
    text?: Record<string, unknown>;
    tools: Record<string, unknown>[];
  },
): void {
  for (const modality of requiredInputModalities(payload)) {
    if (!opts.capability.inputModalities.includes(modality)) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${modality} input is not supported by the selected model`,
      );
    }
  }
  if (
    !opts.capability.supportsImageDetailOriginal
    && hasOriginalImageDetail(payload)
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "image detail original is not supported by the selected model",
    );
  }
  applyModelCapabilityFields(payload, opts.capability, opts.text, opts.serviceTier);
  const configuredSummary = opts.capability.supportsReasoningSummaryParameter
    && opts.capability.defaultReasoningSummary !== "none"
    ? opts.capability.defaultReasoningSummary
    : undefined;
  if (configuredSummary != null) {
    if (payload.reasoning == null) payload.reasoning = {};
    if (!isRecordValue(payload.reasoning)) {
      throw new ChatGPTOAuthProtocolError("internal Responses reasoning must be an object");
    }
    payload.reasoning.summary = configuredSummary;
  } else if (isRecordValue(payload.reasoning)) {
    delete payload.reasoning.summary;
  }
  if (opts.endpoint === "responses") {
    if (Object.hasOwn(payload, "include") && !Array.isArray(payload.include)) {
      throw new ChatGPTOAuthInvalidRequestError("include must be an array");
    }
    const include = Array.isArray(payload.include) ? payload.include : [];
    if (!include.includes("reasoning.encrypted_content")) {
      include.push("reasoning.encrypted_content");
    }
    payload.include = include;
  }
  if (opts.endpoint === "compact") {
    delete payload.include;
  }
  if (!useResponsesLite(opts.capability, opts.responsesLite)) return;

  if (hasAnyImageDetail(payload)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "explicit image detail cannot be represented by Responses Lite",
    );
  }

  const unsupportedTool = opts.tools.find(
    (tool) => tool.type === "web_search"
      || tool.type === "image_generation"
      || tool.type === "programmatic_tool_calling",
  );
  if (unsupportedTool) {
    throw new ChatGPTOAuthInvalidRequestError(
      `Responses Lite cannot use hosted ${String(unsupportedTool.type)} without a standalone executor`,
    );
  }

  if (payload.instructions != null && typeof payload.instructions !== "string") {
    throw new ChatGPTOAuthProtocolError("internal Responses instructions must be a string");
  }
  const instructions = payload.instructions ?? "";
  delete payload.instructions;
  delete payload.tools;
  if (opts.endpoint === "responses") {
    if (payload.tool_choice !== "auto") {
      throw new ChatGPTOAuthInvalidRequestError(
        "Responses Lite requires tool_choice to be the exact string auto",
      );
    }
    payload.tool_choice = "auto";
  }
  payload.parallel_tool_calls = false;
  if (!Array.isArray(payload.input)) {
    throw new ChatGPTOAuthProtocolError("internal Responses input must be an array");
  }
  const input = payload.input;
  const developerItems: Record<string, unknown>[] = [
    { type: "additional_tools", role: "developer", tools: opts.tools },
  ];
  if (instructions.length > 0) {
    developerItems.push({
      type: "message",
      role: "developer",
      content: [{ type: "input_text", text: instructions }],
    });
  }
  payload.input = [...developerItems, ...input];
  if (payload.reasoning == null) payload.reasoning = {};
  if (!isRecordValue(payload.reasoning)) {
    throw new ChatGPTOAuthProtocolError("internal Responses reasoning must be an object");
  }
  const reasoning = payload.reasoning;
  if (opts.endpoint === "responses") {
    if (reasoning.context != null && reasoning.context !== "all_turns") {
      throw new ChatGPTOAuthInvalidRequestError(
        "Responses Lite reasoning.context must be all_turns when explicitly provided",
      );
    }
  } else {
    delete reasoning.mode;
  }
  reasoning.context = "all_turns";
  (payload as Record<PropertyKey, unknown>)[RESPONSES_LITE_PAYLOAD] = true;
}

function hasOriginalImageDetail(payload: Record<string, unknown>): boolean {
  return responseInputImageParts(payload).some((part) => part.detail === "original");
}

function hasInputImage(payload: Record<string, unknown>): boolean {
  return responseInputImageParts(payload).length > 0;
}

function requiredInputModalities(
  payload: Record<string, unknown>,
): Array<"text" | "image" | "audio"> {
  const required = new Set<"text" | "image" | "audio">();
  const collectParts = (parts: unknown): void => {
    if (!Array.isArray(parts)) return;
    for (const part of parts) {
      if (!isRecordValue(part)) continue;
      if (part.type === "input_text" || part.type === "output_text") required.add("text");
      else if (part.type === "input_image") required.add("image");
      else if (part.type === "input_audio") required.add("audio");
    }
  };
  if (typeof payload.instructions === "string" && payload.instructions.length > 0) {
    required.add("text");
  }
  if (!Array.isArray(payload.input)) return [...required];
  for (const item of payload.input) {
    if (!isRecordValue(item)) continue;
    if (item.type === "message") {
      collectParts(item.content);
    } else if (item.type === "function_call_output" || item.type === "custom_tool_call_output") {
      if (typeof item.output === "string") required.add("text");
      else collectParts(item.output);
    }
  }
  return [...required];
}

function hasAnyImageDetail(payload: Record<string, unknown>): boolean {
  return responseInputImageParts(payload).some((part) => part.detail != null);
}

function responseInputImageParts(
  payload: Record<string, unknown>,
): Record<string, unknown>[] {
  if (!Array.isArray(payload.input)) return [];
  const images: Record<string, unknown>[] = [];
  for (const item of payload.input) {
    if (!isRecordValue(item) || item.type !== "message" || !Array.isArray(item.content)) {
      continue;
    }
    for (const part of item.content) {
      if (isRecordValue(part) && part.type === "input_image") images.push(part);
    }
  }
  return images;
}

function isResponsesLitePayload(payload: Record<string, unknown>): boolean {
  return (payload as Record<PropertyKey, unknown>)[RESPONSES_LITE_PAYLOAD] === true;
}

function compactRawEvents(events: Record<string, unknown>[]): Record<string, unknown>[] {
  const keep = events.filter((event) => event.type === "web_search_call");
  for (const event of events.slice(-20)) {
    if (!keep.includes(event)) keep.push(event);
  }
  return keep;
}

function filterCompactedHistoryItems(
  items: unknown[],
  source = "remote compact output",
): Record<string, unknown>[] {
  const compacted: Record<string, unknown>[] = [];
  for (const [index, item] of items.entries()) {
    if (typeof item !== "object" || item === null || Array.isArray(item)) {
      throw new ChatGPTOAuthInvalidRequestError(`${source} item ${index} must be an object`);
    }
    const record = item as Record<string, unknown>;
    validateCompactedHistoryItem(record, source, index);
    if (shouldKeepCompactedHistoryItem(record)) {
      compacted.push(record);
    }
  }
  return compacted;
}

function validateCompactedHistoryItem(
  item: Record<string, unknown>,
  source: string,
  index: number,
): void {
  if (typeof item.type !== "string") {
    throw new ChatGPTOAuthInvalidRequestError(`${source} item ${index} must have a string type`);
  }
  try {
    validateResponseItemOptionalFields(item, item.type);
  } catch (err) {
    if (err instanceof ChatGPTOAuthProtocolError) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${source} ${item.type} item ${index} is invalid: ${err.message}`,
      );
    }
    throw err;
  }
  if (item.type === "message") {
    if (
      typeof item.role !== "string"
      || !["user", "assistant", "developer"].includes(item.role)
    ) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${source} message item ${index} role must be one of: user, assistant, developer`,
      );
    }
    if (!Array.isArray(item.content)) {
      throw new ChatGPTOAuthInvalidRequestError(`${source} message item ${index} must have an array content field`);
    }
    validateMessageContentItems(item.content, source, index);
    return;
  }
  if (item.type === "agent_message") {
    if (typeof item.author !== "string" || typeof item.recipient !== "string") {
      throw new ChatGPTOAuthInvalidRequestError(
        `${source} agent_message item ${index} must have string author and recipient fields`,
      );
    }
    if (!Array.isArray(item.content)) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${source} agent_message item ${index} must have an array content field`,
      );
    }
    for (const [contentIndex, contentItem] of item.content.entries()) {
      if (typeof contentItem !== "object" || contentItem === null || Array.isArray(contentItem)) {
        throw new ChatGPTOAuthInvalidRequestError(
          `${source} agent_message item ${index} content item ${contentIndex} must be an object`,
        );
      }
      const part = contentItem as Record<string, unknown>;
      const validInputText = part.type === "input_text" && typeof part.text === "string";
      const validEncrypted = part.type === "encrypted_content"
        && typeof part.encrypted_content === "string";
      if (!validInputText && !validEncrypted) {
        throw new ChatGPTOAuthInvalidRequestError(
          `${source} agent_message item ${index} content item ${contentIndex} is invalid`,
        );
      }
    }
    return;
  }
  if (item.type === "compaction" || item.type === "compaction_summary") {
    if (typeof item.encrypted_content !== "string") {
      throw new ChatGPTOAuthInvalidRequestError(
        `${source} ${String(item.type)} item ${index} must have string encrypted_content`,
      );
    }
    return;
  }
  if (
    item.type === "context_compaction"
    && Object.hasOwn(item, "encrypted_content")
    && item.encrypted_content != null
    && typeof item.encrypted_content !== "string"
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      `${source} context_compaction item ${index} encrypted_content must be a string`,
    );
  }
  if (item.type === "context_compaction") return;
  if (item.type === "additional_tools") {
    if (item.role !== "developer" || !Array.isArray(item.tools)) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${source} additional_tools item ${index} must have developer role and an array tools field`,
      );
    }
    if (item.tools.some((tool) => typeof tool !== "object" || tool === null || Array.isArray(tool))) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${source} additional_tools item ${index} tools must contain only objects`,
      );
    }
    return;
  }
  if (item.type === "reasoning") {
    if (
      Object.hasOwn(item, "encrypted_content")
      && item.encrypted_content != null
      && typeof item.encrypted_content !== "string"
    ) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${source} reasoning item ${index} encrypted_content must be a string or null`,
      );
    }
    try {
      reasoningFromResponseItems([item]);
    } catch (err) {
      if (err instanceof ChatGPTOAuthProtocolError) {
        throw new ChatGPTOAuthInvalidRequestError(`${source} reasoning item ${index} is invalid: ${err.message}`);
      }
      throw err;
    }
    return;
  }
  if (item.type === "function_call") {
    if (
      typeof item.call_id !== "string"
      || typeof item.name !== "string"
      || typeof item.arguments !== "string"
    ) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${source} function_call item ${index} requires string call_id/name/arguments`,
      );
    }
    return;
  }
  throw new ChatGPTOAuthInvalidRequestError(
    `${source} item ${index} has an unsupported type`,
  );
}

function validateMessageContentItems(
  content: unknown[],
  source: string,
  index: number,
): void {
  for (const [contentIndex, contentItem] of content.entries()) {
    if (typeof contentItem !== "object" || contentItem === null || Array.isArray(contentItem)) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${source} message item ${index} content item ${contentIndex} must be an object`,
      );
    }
    const part = contentItem as Record<string, unknown>;
    if (part.type === "input_text" || part.type === "output_text") {
      if (typeof part.text !== "string") {
        throw new ChatGPTOAuthInvalidRequestError(
          `${source} message item ${index} content item ${contentIndex} must have string text`,
        );
      }
      validatePromptCacheBreakpointValue(
        part.prompt_cache_breakpoint,
        `${source} message item ${index} content item ${contentIndex}`,
      );
      continue;
    }
    if (part.type === "input_image") {
      if (typeof part.image_url !== "string") {
        throw new ChatGPTOAuthInvalidRequestError(
          `${source} message item ${index} content item ${contentIndex} must have string image_url`,
        );
      }
      if (
        part.detail != null
        && (
          typeof part.detail !== "string"
          || !["auto", "low", "high", "original"].includes(part.detail)
        )
      ) {
        throw new ChatGPTOAuthInvalidRequestError(
          `${source} message item ${index} content item ${contentIndex} has invalid detail`,
        );
      }
      validatePromptCacheBreakpointValue(
        part.prompt_cache_breakpoint,
        `${source} message item ${index} content item ${contentIndex}`,
      );
      continue;
    }
    if (part.type === "input_audio") {
      if (typeof part.audio_url !== "string") {
        throw new ChatGPTOAuthInvalidRequestError(
          `${source} message item ${index} content item ${contentIndex} must have string audio_url`,
        );
      }
      continue;
    }
    throw new ChatGPTOAuthInvalidRequestError(
      `${source} message item ${index} content item ${contentIndex} has an unsupported type`,
    );
  }
}

function validatePromptCacheBreakpointValue(
  value: unknown,
  _source: string,
): void {
  if (value == null) return;
  throw new ChatGPTOAuthInvalidRequestError(
    "prompt_cache_breakpoint is not supported by the private Codex OAuth HTTP transport",
  );
}

function shouldKeepCompactedHistoryItem(item: Record<string, unknown>): boolean {
  if (item.type === "message") {
    if (item.role === "assistant") return true;
    if (item.role !== "user") return false;
    return isRealUserOrHookMessage(item.content);
  }
  return item.type === "agent_message"
    || item.type === "compaction"
    || item.type === "compaction_summary"
    || item.type === "context_compaction";
}

function isRealUserOrHookMessage(content: unknown): boolean {
  if (!Array.isArray(content)) return true;
  const textItems = content.filter(
    (item): item is Record<string, unknown> => typeof item === "object" && item !== null,
  );
  const hasVisibleHook = textItems.some(
    (item) => item.type === "input_text"
      && typeof item.text === "string"
      && isHookPromptText(item.text),
  );
  if (hasVisibleHook && textItems.every((item) =>
    item.type === "input_text"
    && typeof item.text === "string"
    && (isHookPromptText(item.text) || isContextualUserText(item.text)))) {
    return true;
  }
  return !textItems.some(
    (item) => item.type === "input_text"
      && typeof item.text === "string"
      && (isHookPromptText(item.text) || isContextualUserText(item.text)),
  );
}

function isHookPromptText(text: string): boolean {
  const match = text.trim().match(
    /^<hook_prompt\s+[^>]*hook_run_id="([^"]+)"[^>]*>[\s\S]*<\/hook_prompt>$/,
  );
  return match != null && match[1].trim().length > 0;
}

function isContextualUserText(text: string): boolean {
  const trimmed = text.trim();
  const lower = trimmed.toLowerCase();
  const markerPairs = [
    ["# agents.md instructions", "</instructions>"],
    ["<environment_context>", "</environment_context>"],
    ["<skill>", "</skill>"],
    ["<user_shell_command>", "</user_shell_command>"],
    ["<turn_aborted>", "</turn_aborted>"],
    ["<subagent_notification>", "</subagent_notification>"],
    ["<recommended_plugins>", "</recommended_plugins>"],
  ];
  if (markerPairs.some(([start, end]) => lower.startsWith(start) && lower.endsWith(end))) {
    return true;
  }
  const external = trimmed.match(/^<external_([^>]+)>[\s\S]*<\/external_([^>]+)>$/);
  if (external != null && external[1] === external[2]) return true;
  if (/^<codex_internal_context source="[a-z][a-z0-9_]*">[\s\S]*<\/codex_internal_context>$/.test(trimmed)) {
    return true;
  }
  if (lower.startsWith("<goal_context>") && lower.endsWith("</goal_context>")) return true;
  return trimmed.startsWith(
    "Warning: The maximum number of unified exec processes you can keep open is",
  ) || (
    trimmed.startsWith("Warning: apply_patch was requested via ")
    && trimmed.endsWith("Use the apply_patch tool instead of exec_command.")
  ) || trimmed.startsWith(
    "Warning: Your account was flagged for potentially high-risk cyber activity",
  );
}

export function webSearchEventFromResponseItem(
  item: Record<string, unknown>,
): StreamEvent | null {
  if (item.type !== "web_search_call") return null;
  if (typeof item.id !== "string") {
    throw new ChatGPTOAuthProtocolError("web_search_call must contain a string id");
  }
  if (typeof item.action !== "object" || item.action === null || Array.isArray(item.action)) {
    throw new ChatGPTOAuthProtocolError("web_search_call must contain an object action");
  }
  const id = item.id;
  const action = item.action as Record<string, unknown>;
  const query = webSearchQueryFromAction(action);
  const sources = webSearchSourcesFromAction(action);
  return {
    type: "web_search_call",
    id,
    input: { query },
    content: sources,
  };
}

function webSearchQueryFromAction(action: Record<string, unknown>): string {
  const actionType = action.type;
  if (typeof actionType !== "string") {
    throw new ChatGPTOAuthProtocolError("web_search_call action.type must be a string");
  }
  if (actionType !== "search") {
    throw new ChatGPTOAuthProtocolError(
      `web_search_call action type ${actionType} cannot be represented by this facade`,
    );
  }

  const queries = action.queries;
  if (
    queries != null
    && (
      !Array.isArray(queries)
      || queries.some((candidate) => typeof candidate !== "string")
    )
  ) {
    throw new ChatGPTOAuthProtocolError(
      "web_search_call action.queries must be an array of strings",
    );
  }
  const query = action.query;
  if (query != null && typeof query !== "string") {
    throw new ChatGPTOAuthProtocolError("web_search_call action.query must be a string");
  }
  if (Array.isArray(queries) && queries.length > 1) {
    throw new ChatGPTOAuthProtocolError(
      "web_search_call action contains multiple queries that cannot be represented by this facade",
    );
  }
  if (typeof query === "string") {
    if (Array.isArray(queries) && queries.length === 1 && queries[0] !== query) {
      throw new ChatGPTOAuthProtocolError("web_search_call action.query conflicts with action.queries");
    }
    return query;
  }
  if (Array.isArray(queries) && queries.length === 1) return queries[0] as string;
  throw new ChatGPTOAuthProtocolError("web_search_call action must contain a query");
}

function webSearchSourcesFromAction(action: Record<string, unknown>): Record<string, unknown>[] {
  if (!Object.hasOwn(action, "sources")) {
    throw new ChatGPTOAuthProtocolError(
      "web_search_call action.sources is required when sources were requested",
    );
  }
  return normalizeWebSearchSources(action.sources);
}

function normalizeWebSearchSources(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    throw new ChatGPTOAuthProtocolError("web_search_call action.sources must be an array");
  }
  const out: Record<string, unknown>[] = [];
  for (const [index, source] of value.entries()) {
    if (typeof source !== "object" || source === null || Array.isArray(source)) {
      throw new ChatGPTOAuthProtocolError(`web_search source ${index} must be an object`);
    }
    const s = source as Record<string, unknown>;
    if (typeof s.url !== "string" || s.url.length === 0) {
      throw new ChatGPTOAuthProtocolError(`web_search source ${index} requires a non-empty url`);
    }
    if (s.title != null && typeof s.title !== "string") {
      throw new ChatGPTOAuthProtocolError(`web_search source ${index} title must be a string when provided`);
    }
    if (s.page_age != null && typeof s.page_age !== "string") {
      throw new ChatGPTOAuthProtocolError(`web_search source ${index} page_age must be a string when provided`);
    }
    const url = s.url;
    const result: Record<string, unknown> = {
      type: "web_search_result",
      url,
      ...(typeof s.title === "string" ? { title: s.title } : {}),
    };
    if (typeof s.page_age === "string") result.page_age = s.page_age;
    out.push(result);
  }
  return out;
}

function effectiveReasoningEffort(
  capability: ModelCapability,
  reasoningEffort?: string,
  reasoning?: ReasoningOptions,
): string | undefined {
  const nestedEffort = reasoning?.effort;
  if (
    reasoningEffort != null
    && nestedEffort != null
    && reasoningEffort !== nestedEffort
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning_effort conflicts with reasoning.effort",
    );
  }
  return reasoningEffort
    ?? nestedEffort
    ?? capability.defaultReasoningEffort;
}

export function setReasoningPayload(
  payload: Record<string, unknown>,
  reasoningEffort?: string,
  reasoning?: ReasoningOptions,
  capability?: ModelCapability,
): void {
  if (reasoning != null && (typeof reasoning !== "object" || Array.isArray(reasoning))) {
    throw new ChatGPTOAuthInvalidRequestError("reasoning must be an object");
  }
  const nested = reasoning as Record<string, unknown> | undefined;
  if (nested != null) {
    for (const key of Object.keys(nested)) {
      if (!["effort", "mode", "context"].includes(key)) {
        throw new ChatGPTOAuthInvalidRequestError(`reasoning.${key} is not supported`);
      }
    }
  }
  const nestedEffort = nested?.effort;
  if (
    reasoningEffort != null
    && nestedEffort != null
    && reasoningEffort !== nestedEffort
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning_effort conflicts with reasoning.effort",
    );
  }
  const selectedEffort = reasoningEffort ?? nestedEffort;
  if (
    selectedEffort != null
    && (typeof selectedEffort !== "string" || selectedEffort.length === 0)
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning_effort must be a non-empty string when provided",
    );
  }
  if (selectedEffort != null && selectedEffort !== selectedEffort.trim()) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning_effort must not contain surrounding whitespace",
    );
  }
  const mode = nested?.mode;
  if (mode != null && (typeof mode !== "string" || !REASONING_MODES.has(mode))) {
    throw new ChatGPTOAuthInvalidRequestError("reasoning.mode must be one of: standard, pro");
  }
  const context = nested?.context;
  if (
    context != null
    && (typeof context !== "string" || !REASONING_CONTEXTS.has(context))
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning.context must be one of: auto, current_turn, all_turns",
    );
  }
  if (mode != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning.mode is not supported by the private Codex OAuth transport",
    );
  }

  let existing: Record<string, unknown> = {};
  if (Object.hasOwn(payload, "reasoning")) {
    if (
      typeof payload.reasoning !== "object"
      || payload.reasoning === null
      || Array.isArray(payload.reasoning)
    ) {
      throw new ChatGPTOAuthInvalidRequestError("reasoning must be an object");
    }
    existing = payload.reasoning as Record<string, unknown>;
  }
  const existingMode = mode == null ? existing.mode : undefined;
  if (
    existingMode != null
    && (typeof existingMode !== "string" || !REASONING_MODES.has(existingMode))
  ) {
    throw new ChatGPTOAuthInvalidRequestError("reasoning.mode must be one of: standard, pro");
  }
  if (existingMode != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning.mode is not supported by the private Codex OAuth transport",
    );
  }
  const merged: Record<string, unknown> = { ...existing };
  delete merged.mode;
  if (selectedEffort != null) {
    if (capability == null) {
      throw new ChatGPTOAuthInvalidRequestError(
        "model catalog capability is required when reasoning effort is provided",
      );
    }
    merged.effort = wireReasoningEffort(selectedEffort, capability);
  }
  if (context != null) merged.context = context;
  if (Object.keys(merged).length === 0) {
    delete payload.reasoning;
    return;
  }
  if (Object.hasOwn(payload, "include") && !Array.isArray(payload.include)) {
    throw new ChatGPTOAuthInvalidRequestError("include must be an array");
  }
  payload.reasoning = merged;
  const include = Array.isArray(payload.include) ? payload.include : [];
  if (!include.includes("reasoning.encrypted_content")) {
    include.push("reasoning.encrypted_content");
  }
  payload.include = include;
}

export function wireReasoningEffort(effort: string, capability: ModelCapability): string {
  const supported = capability.supportedReasoningEfforts.map((preset) => preset.effort);
  if (effort !== "ultra") {
    if (!supported.includes(effort)) {
      throw new ChatGPTOAuthInvalidRequestError(
        "reasoning effort is not supported by the selected model",
      );
    }
    return effort === "persistent" ? "disabled" : effort;
  }
  if (
    capability.multiAgentReasoningEffort != null
    && capability.multiAgentReasoningEffort !== "ultra"
    && supported.includes(capability.multiAgentReasoningEffort)
  ) {
    return capability.multiAgentReasoningEffort;
  }
  if (supported.includes("max")) return "max";
  const lastSupported = [...supported].reverse().find((candidate) => candidate !== "ultra");
  if (lastSupported != null) return lastSupported;
  throw new ChatGPTOAuthInvalidRequestError(
    "reasoning effort ultra has no live wire mapping for the selected model",
  );
}

function wireModel(model: string): string {
  return model;
}

function rejectUnsupportedPrivateRequestFields(
  _payload: Record<string, unknown>,
  safetyIdentifier?: string,
  promptCacheOptions?: PromptCacheOptions,
): void {
  if (safetyIdentifier != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "safety_identifier is not supported by the private Codex OAuth HTTP transport",
    );
  }

  if (promptCacheOptions != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "prompt_cache_options is not supported by the private Codex OAuth HTTP transport",
    );
  }
}

function validatePromptCacheBreakpoints(
  payload: Record<string, unknown>,
): void {
  if (!Array.isArray(payload.input)) return;
  for (const item of payload.input) {
    if (typeof item !== "object" || item === null || Array.isArray(item)) continue;
    const record = item as Record<string, unknown>;
    if (record.prompt_cache_breakpoint != null) {
      throw new ChatGPTOAuthInvalidRequestError(
        "prompt_cache_breakpoint is not supported by the private Codex OAuth HTTP transport",
      );
    }
    if (!Array.isArray(record.content)) continue;
    for (const contentItem of record.content) {
      if (
        typeof contentItem === "object"
        && contentItem !== null
        && !Array.isArray(contentItem)
        && (contentItem as Record<string, unknown>).prompt_cache_breakpoint != null
      ) {
        throw new ChatGPTOAuthInvalidRequestError(
          "prompt_cache_breakpoint is not supported by the private Codex OAuth HTTP transport",
        );
      }
    }
  }
}

function validateAssistantMessageOutputItem(item: Record<string, unknown>): void {
  if (item.type !== "message") {
    throw new ChatGPTOAuthProtocolError(
      "expected an assistant message output item",
    );
  }
  if (item.role !== "assistant") {
    throw new ChatGPTOAuthProtocolError("message output item role must be assistant");
  }
  if (!Array.isArray(item.content)) {
    throw new ChatGPTOAuthProtocolError("message output item content must be an array");
  }
  for (const [index, part] of item.content.entries()) {
    if (typeof part !== "object" || part === null || Array.isArray(part)) {
      throw new ChatGPTOAuthProtocolError(
        `message output item content ${index} must be an object`,
      );
    }
    const contentPart = part as Record<string, unknown>;
    if (contentPart.type !== "output_text") {
      throw new ChatGPTOAuthProtocolError(
        `message output item content ${index} has an unsupported type`,
      );
    }
    if (typeof contentPart.text !== "string") {
      throw new ChatGPTOAuthProtocolError(
        `message output item content ${index} text must be a string`,
      );
    }
  }
}

function validateResponseOutputItem(item: Record<string, unknown>): void {
  if (item.type === "custom_tool_call") {
    throw new ChatGPTOAuthProtocolError(
      "custom_tool_call is not supported by the public tool contract",
    );
  }
  if (typeof item.type === "string") {
    validateResponseItemOptionalFields(item, item.type);
  }
  switch (item.type) {
    case "function_call":
      toolCallFromResponseItem(item);
      return;
    case "image_generation_call":
      imageGenerationFromItem(item);
      return;
    case "web_search_call":
      webSearchEventFromResponseItem(item);
      return;
    case "reasoning":
      reasoningFromResponseItems([item]);
      if (
        item.encrypted_content !== undefined
        && item.encrypted_content !== null
        && typeof item.encrypted_content !== "string"
      ) {
        throw new ChatGPTOAuthProtocolError(
          "reasoning encrypted_content must be a string or null",
        );
      }
      return;
    case "message":
      validateAssistantMessageOutputItem(item);
      return;
    default:
      throw new ChatGPTOAuthProtocolError(
        "response output item has an unsupported type",
      );
  }
}

function validateResponseItemOptionalFields(
  item: Record<string, unknown>,
  itemType: string,
): void {
  if (item.id != null && typeof item.id !== "string") {
    throw new ChatGPTOAuthProtocolError(`${itemType} id must be a string or null`);
  }

  const metadata = item.internal_chat_message_metadata_passthrough;
  if (metadata != null) {
    if (!isRecordValue(metadata)) {
      throw new ChatGPTOAuthProtocolError(
        `${itemType} internal_chat_message_metadata_passthrough must be an object or null`,
      );
    }
    if (metadata.turn_id != null && typeof metadata.turn_id !== "string") {
      throw new ChatGPTOAuthProtocolError(
        `${itemType} internal_chat_message_metadata_passthrough.turn_id must be a string or null`,
      );
    }
    if (
      metadata.create_time != null
      && (typeof metadata.create_time !== "number" || !Number.isFinite(metadata.create_time))
    ) {
      throw new ChatGPTOAuthProtocolError(
        `${itemType} internal_chat_message_metadata_passthrough.create_time must be a JSON number or null`,
      );
    }
  }

  const requireNullableString = (field: string): void => {
    if (item[field] != null && typeof item[field] !== "string") {
      throw new ChatGPTOAuthProtocolError(`${itemType} ${field} must be a string or null`);
    }
  };

  if (itemType === "message") {
    if (item.phase != null && item.phase !== "commentary" && item.phase !== "final_answer") {
      throw new ChatGPTOAuthProtocolError(
        "message phase must be commentary, final_answer, or null",
      );
    }
  } else if (itemType === "reasoning") {
    requireNullableString("encrypted_content");
  } else if (itemType === "function_call") {
    requireNullableString("namespace");
    if (
      item.encrypted_function_args != null
      && (!Array.isArray(item.encrypted_function_args)
        || item.encrypted_function_args.some((value) => typeof value !== "string"))
    ) {
      throw new ChatGPTOAuthProtocolError(
        "function_call encrypted_function_args must be a string array or null",
      );
    }
  } else if (itemType === "web_search_call") {
    requireNullableString("status");
  } else if (itemType === "image_generation_call") {
    requireNullableString("revised_prompt");
  }
}

function validateChatResponseItem(
  item: Record<string, unknown>,
): void {
  validateResponseOutputItem(item);
  if (item.type === "image_generation_call") {
    throw new ChatGPTOAuthProtocolError(
      "chat response contains an unsupported output item",
    );
  }
}

export function toolCallFromResponseItem(
  item: Record<string, unknown>,
): ToolCall | null {
  if (item.type === "custom_tool_call") {
    throw new ChatGPTOAuthProtocolError(
      "custom_tool_call is not supported by the public tool contract",
    );
  }
  if (item.type !== "function_call") return null;
  const name = item.name;
  if (typeof name !== "string") {
    throw new ChatGPTOAuthProtocolError(`${String(item.type)} must contain a string name`);
  }
  const argumentField = "arguments";
  const rawArgs = item[argumentField];
  if (typeof rawArgs !== "string") {
    throw new ChatGPTOAuthProtocolError(
      `${String(item.type)} ${argumentField} must be a string`,
    );
  }
  if (typeof item.call_id !== "string") {
    throw new ChatGPTOAuthProtocolError(`${String(item.type)} must contain a string call_id`);
  }
  return { id: item.call_id, name, arguments: rawArgs };
}

export function textFromResponseItems(
  items: Record<string, unknown>[],
): string {
  const parts: string[] = [];
  for (const item of items) {
    const itemType = item.type;
    if (itemType === "output_text" || itemType === "text") {
      throw new ChatGPTOAuthProtocolError(
        `${String(itemType)} is not a supported top-level response item`,
      );
    }
    if (itemType !== "message") {
      if (![
        "function_call",
        "image_generation_call",
        "reasoning",
        "web_search_call",
      ].includes(String(itemType))) {
        throw new ChatGPTOAuthProtocolError(
          "response output item has an unsupported type",
        );
      }
      continue;
    }
    if (item.role !== "assistant") {
      throw new ChatGPTOAuthProtocolError(
        "response message item role must be assistant",
      );
    }
    const content = item.content;
    if (!Array.isArray(content)) {
      throw new ChatGPTOAuthProtocolError("message output item content must be an array");
    }
    for (const [index, part] of content.entries()) {
      if (typeof part !== "object" || part === null || Array.isArray(part)) {
        throw new ChatGPTOAuthProtocolError(
          `message output item content ${index} must be an object`,
        );
      }
      const p = part as Record<string, unknown>;
      if (p.type === "text") {
        throw new ChatGPTOAuthProtocolError(
          `message output item content ${index} type text is not supported`,
        );
      }
      if (p.type !== "output_text") {
        throw new ChatGPTOAuthProtocolError(
          `message output item content ${index} has an unsupported type`,
        );
      }
      const text = p.text;
      if (typeof text !== "string") {
        throw new ChatGPTOAuthProtocolError(
          `message output item content ${index} text must be a string`,
        );
      }
      if (text) parts.push(text);
    }
  }
  return parts.join("");
}

export function validateImageContentItems(
  images: ImageReference[],
): Record<string, unknown>[] {
  if (!Array.isArray(images)) {
    throw new ChatGPTOAuthInvalidRequestError("image references must be an array");
  }
  const items: Record<string, unknown>[] = [];
  for (let i = 0; i < images.length; i++) {
    const image = images[i];
    if (typeof image !== "object" || image === null) {
      throw new ChatGPTOAuthInvalidRequestError(
        `image reference ${i} must be an object`,
      );
    }
    const unknownField = Object.keys(image).find(
      (field) => !["image_url", "detail", "prompt_cache_breakpoint"].includes(field),
    );
    if (unknownField != null) {
      throw new ChatGPTOAuthInvalidRequestError(
        `image reference ${i} does not support field ${JSON.stringify(unknownField)}`,
      );
    }
    const imageUrl = image.image_url;
    if (typeof imageUrl !== "string" || !imageUrl.trim()) {
      throw new ChatGPTOAuthInvalidRequestError(
        `image reference ${i} requires image_url`,
      );
    }
    if (!imageUrl.startsWith("data:image/")) {
      throw new ChatGPTOAuthInvalidRequestError(
        `image reference ${i} must be a data:image URL`,
      );
    }
    const item: Record<string, unknown> = {
      type: "input_image",
      image_url: imageUrl,
    };
    if (image.detail != null) {
      if (typeof image.detail !== "string" || !IMAGE_DETAILS.has(image.detail)) {
        throw new ChatGPTOAuthInvalidRequestError(
          `image reference ${i} detail must be one of: auto, low, high, original`,
        );
      }
      item.detail = image.detail;
    }
    if (image.prompt_cache_breakpoint != null) {
      throw new ChatGPTOAuthInvalidRequestError(
        "prompt_cache_breakpoint is not supported by the private Codex OAuth HTTP transport",
      );
    }
    items.push(item);
  }
  return items;
}

export function imageGenerationFromItem(
  item: Record<string, unknown>,
): Record<string, unknown> | null {
  if (item.type !== "image_generation_call") return null;
  const result = item.result;
  if (typeof result !== "string") {
    throw new ChatGPTOAuthProtocolError(
      "image_generation_call requires a string result",
    );
  }
  if (item.id != null && typeof item.id !== "string") {
    throw new ChatGPTOAuthProtocolError(
      "image_generation_call id must be a string or null",
    );
  }
  if (typeof item.status !== "string") {
    throw new ChatGPTOAuthProtocolError("image_generation_call must contain a string status");
  }
  if (item.revised_prompt != null && typeof item.revised_prompt !== "string") {
    throw new ChatGPTOAuthProtocolError("image_generation_call revised_prompt must be a string when provided");
  }
  return {
    ...(item.id == null ? {} : { id: item.id }),
    status: item.status,
    ...(item.revised_prompt == null ? {} : { revised_prompt: item.revised_prompt }),
    result,
  };
}

export function usageFromResponse(value: unknown): Usage | null {
  if (
    typeof value !== "object" ||
    value === null ||
    Array.isArray(value)
  )
    return null;
  const v = value as Record<string, unknown>;
  if (
    [
      "prompt_tokens",
      "completion_tokens",
      "prompt_tokens_details",
      "cached_input_tokens",
      "cache_read_input_tokens",
      "cache_creation_input_tokens",
    ].some((field) => Object.hasOwn(v, field))
  ) return null;
  const prompt = v.input_tokens;
  const completion = v.output_tokens;
  const total = v.total_tokens;
  if (
    !isNonNegativeSafeInteger(prompt)
    || !isNonNegativeSafeInteger(completion)
    || !isNonNegativeSafeInteger(total)
    || total !== prompt + completion
  )
    return null;
  const tokenDetails = v.input_tokens_details;
  const result: Usage = {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens: total,
  };
  if (tokenDetails == null) return result;
  if (typeof tokenDetails !== "object" || Array.isArray(tokenDetails)) return null;
  const d = tokenDetails as Record<string, unknown>;
  if (!isNonNegativeSafeInteger(d.cached_tokens)) return null;
  result.cached_tokens = d.cached_tokens;
  const cacheWriteTokens = d.cache_write_tokens;
  if (cacheWriteTokens != null) {
    if (!isNonNegativeSafeInteger(cacheWriteTokens)) return null;
    result.cache_write_tokens = cacheWriteTokens;
  }
  return result;
}

export function requireSubagentHeaderValue(value: unknown): string {
  if (typeof value !== "string" || !/^[\x21-\x7E]+$/.test(value)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "subagent must be a non-empty visible ASCII token without whitespace",
    );
  }
  return value;
}

function isNonNegativeSafeInteger(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

export { REMOTE_COMPACTION_MARKER };
