import * as crypto from "node:crypto";
import { execFileSync } from "node:child_process";
import * as os from "node:os";
import {
  ChatGPTOAuthError,
  ChatGPTOAuthInvalidRequestError,
  loadTokenData,
  redactText,
  refreshToken,
} from "./auth.js";
import type {
  AssistantResponse,
  Message,
  MessageContentPart,
  ToolCall,
  ToolSchema,
  Usage,
} from "./messages.js";
import { MessageRole } from "./messages.js";
import {
  reasoningFromResponseItems,
  responseFailureMessage,
} from "./protocol.js";
import {
  LITE_HEADER_NAME,
  LITE_HEADER_VALUE,
  applyModelCapabilityFields,
  buildCodexClientMetadata,
  capabilityForModel,
  resolveCodexMetadataEnabled,
  shouldEnableParallelToolCalls,
  stripImageDetailFields,
  useResponsesLite,
} from "./model-capabilities.js";

export const CHATGPT_OAUTH_DEFAULT_BASE_URL =
  "https://chatgpt.com/backend-api/codex";
export const CHATGPT_OAUTH_DEFAULT_MODEL = "gpt-5.5";
const REMOTE_COMPACTION_MARKER = "[Remote Responses compacted history]";
const CODEX_CLI_ORIGINATOR = "codex_cli_rs";
const CODEX_CLI_VERSION_ENV = "CODEX_AS_API_CODEX_CLI_VERSION";
const CODEX_CLI_NPM_PACKAGE = "@openai/codex";
const KNOWN_REASONING_EFFORT_VALUES = new Set([
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
  "max",
  "ultra",
]);
const RESPONSES_LITE_PAYLOAD = Symbol("responses-lite-payload");
const REASONING_MODES = new Set(["standard", "pro"]);
const REASONING_CONTEXTS = new Set(["auto", "current_turn", "all_turns"]);
const IMAGE_DETAILS = new Set(["auto", "low", "high", "original"]);
const RESPONSE_CHAIN_CAPACITY = 256;
let cachedCodexCliVersion: string | null | undefined;

interface ResponseChain {
  input: Record<string, unknown>[];
  output: Record<string, unknown>[];
}

class ResponseChainStore {
  private readonly chains = new Map<string, ResponseChain>();

  resolve(responseId: string): Record<string, unknown>[] {
    const chain = this.chains.get(responseId);
    if (chain == null) {
      throw new ChatGPTOAuthInvalidRequestError(
        `previous_response_id ${JSON.stringify(responseId)} is unknown or has been evicted from the local response history`,
      );
    }

    // Map operations are synchronous, so concurrent requests cannot observe a
    // partially updated LRU entry. Returning a clone also lets multiple
    // branches reuse the same completed response without sharing mutations.
    this.chains.delete(responseId);
    this.chains.set(responseId, chain);
    return cloneResponseItems([...chain.input, ...chain.output]);
  }

  commit(
    responseId: string,
    input: Record<string, unknown>[],
    output: Record<string, unknown>[],
  ): void {
    const chain = {
      input: cloneResponseItems(input),
      output: cloneResponseItems(output),
    };
    this.chains.delete(responseId);
    this.chains.set(responseId, chain);
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

export function primeCodexCliVersionCache(): void {
  void resolveCodexCliVersion();
}

export function resolveCodexCliVersion(): string | undefined {
  const override = normalizeCodexCliVersion(
    process.env[CODEX_CLI_VERSION_ENV],
  );
  if (override) return override;
  if (cachedCodexCliVersion !== undefined) {
    return cachedCodexCliVersion ?? undefined;
  }
  cachedCodexCliVersion = fetchLatestCodexCliVersionFromNpm() ?? null;
  return cachedCodexCliVersion ?? undefined;
}

function fetchLatestCodexCliVersionFromNpm(): string | undefined {
  try {
    const output = execFileSync(
      "npm",
      ["view", CODEX_CLI_NPM_PACKAGE, "version"],
      {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
        timeout: 3000,
      },
    );
    return normalizeCodexCliVersion(output);
  } catch {
    return undefined;
  }
}

function normalizeCodexCliVersion(value: string | undefined): string | undefined {
  const version = value?.trim();
  if (!version) return undefined;
  return /^[0-9]+(?:\.[0-9]+){1,3}(?:[-+][0-9A-Za-z.-]+)?$/.test(version)
    ? version
    : undefined;
}

export function codexCliHeadersForVersion(
  version: string | undefined,
): Record<string, string> {
  const headers: Record<string, string> = {
    originator: CODEX_CLI_ORIGINATOR,
  };
  const normalized = normalizeCodexCliVersion(version);
  if (normalized) {
    headers["User-Agent"] = sanitizeHeaderValue(
      `${CODEX_CLI_ORIGINATOR}/${normalized} (${codexOsInfo()}) codex-as-api`,
    );
  }
  return headers;
}

function codexCliHeaders(): Record<string, string> {
  return codexCliHeadersForVersion(resolveCodexCliVersion());
}

function codexOsInfo(): string {
  return `${codexOsName()} ${os.release() || "unknown"}; ${os.arch() || "unknown"}`;
}

function codexOsName(): string {
  switch (os.platform()) {
    case "darwin":
      return "Mac OS";
    case "win32":
      return "Windows";
    case "linux":
      return "Linux";
    default:
      return os.platform() || "unknown";
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
  clientMetadata?: Record<string, string>;
  codexMetadata?: boolean;
  responsesLite?: boolean | string;
  parallelToolCalls?: boolean;
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

  private model: string;
  private baseUrl: string;
  private authJsonPath: string | undefined;
  private timeout: number | undefined;
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
    } = {},
  ) {
    this.model = opts.model || CHATGPT_OAUTH_DEFAULT_MODEL;
    this.baseUrl = (
      opts.baseUrl || CHATGPT_OAUTH_DEFAULT_BASE_URL
    ).replace(/\/+$/, "");
    this.authJsonPath = opts.authJsonPath;
    this.timeout = opts.timeout;
  }

  async chat(
    messages: Message[],
    opts: ChatOptions = {},
  ): Promise<AssistantResponse> {
    const contentParts: string[] = [];
    let reasoningParts: string[] = [];
    const toolCalls: ToolCall[] = [];
    let finishReason = "stop";
    const rawEvents: Record<string, unknown>[] = [];
    let usage: Usage | null = null;
    let responseId: string | null = null;

    for await (const event of this.chatStream(messages, opts)) {
      rawEvents.push({ ...event });
      if (event.type === "content") {
        contentParts.push(String(event.text ?? ""));
      } else if (
        event.type === "reasoning_delta" ||
        event.type === "reasoning_raw_delta"
      ) {
        reasoningParts.push(String(event.text ?? ""));
      } else if (event.type === "tool_call") {
        toolCalls.push({
          id: String(event.id),
          name: String(event.name),
          arguments: (event.arguments as Record<string, unknown>) || {},
        });
      } else if (event.type === "finish") {
        finishReason = String(event.finish_reason ?? finishReason);
        if (typeof event.reasoning_content === "string") {
          reasoningParts = [event.reasoning_content];
        }
        usage = usageFromResponse(event.usage) ?? usage;
        if (typeof event.response_id === "string") {
          responseId = event.response_id;
        }
      }
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

  chatStream(
    messages: Message[],
    opts: ChatOptions = {},
  ): AsyncGenerator<StreamEvent> {
    const payload = this.responsesPayload(messages, opts);
    const extraHeaders: Record<string, string> = {};
    if (opts.subagent != null) {
      extraHeaders["x-openai-subagent"] = opts.subagent;
    }
    if (opts.memgenRequest != null) {
      extraHeaders["x-openai-memgen-request"] = opts.memgenRequest
        ? "true"
        : "false";
    }

    return this.streamChatPayload(payload, extraHeaders);
  }

  private async *streamChatPayload(
    payload: Record<string, unknown>,
    extraHeaders: Record<string, string>,
  ): AsyncGenerator<StreamEvent> {
    const semanticInput = this.semanticInputs.get(payload);
    if (semanticInput == null) {
      throw new ChatGPTOAuthError(
        "internal error: missing semantic input for response history",
      );
    }
    const finalOutput: Record<string, unknown>[] = [];
    const reasoningParts: string[] = [];
    const yieldedWebSearchIds = new Set<string>();
    let sawTextDelta = false;
    let sawReasoningDelta = false;
    let sawToolCall = false;

    for await (const event of this.postSSE(
      "/responses",
      payload,
      extraHeaders,
    )) {
      const typ = event.type;

      if (typ === "response.output_text.delta") {
        const delta = event.delta;
        if (typeof delta === "string" && delta) {
          sawTextDelta = true;
          yield { type: "content", text: delta };
        }
      } else if (typ === "response.output_item.done") {
        const item = event.item;
        if (typeof item !== "object" || item === null || Array.isArray(item)) {
          throw new ChatGPTOAuthError(
            "response.output_item.done must contain an object item",
          );
        }
        const itemDict = item as Record<string, unknown>;
        finalOutput.push(itemDict);
        const tool = toolCallFromResponseItem(itemDict);
        if (tool) {
          sawToolCall = true;
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
          part_index: event.part_index,
        };
      } else if (typ === "response.reasoning_summary_text.delta") {
        const delta = event.delta;
        if (typeof delta === "string" && delta) {
          sawReasoningDelta = true;
          reasoningParts.push(delta);
          yield {
            type: "reasoning_delta",
            text: delta,
            summary_index: event.summary_index,
          };
        }
      } else if (typ === "response.reasoning_text.delta") {
        const delta = event.delta;
        if (typeof delta === "string" && delta) {
          sawReasoningDelta = true;
          reasoningParts.push(delta);
          yield {
            type: "reasoning_raw_delta",
            text: delta,
            summary_index: event.summary_index,
          };
        }
      } else if (typ === "response.failed") {
        throw new ChatGPTOAuthError(responseFailureMessage(event, "failed"));
      } else if (typ === "response.incomplete") {
        throw new ChatGPTOAuthError(
          responseFailureMessage(event, "incomplete"),
        );
      } else if (typ === "response.completed") {
        const response = completedResponseFromEvent(event);
        this.responseChains.commit(
          String(response.id),
          semanticInput,
          finalOutput,
        );
        const usageData = response.usage;
        for (const item of finalOutput) {
          const webSearch = webSearchEventFromResponseItem(item, finalOutput);
          if (webSearch && !yieldedWebSearchIds.has(String(webSearch.id))) {
            yieldedWebSearchIds.add(String(webSearch.id));
            yield webSearch;
          }
        }
        if (!sawTextDelta) {
          const finalText = textFromResponseItems(finalOutput);
          if (finalText) {
            sawTextDelta = true;
            yield { type: "content", text: finalText };
          }
        }
        if (!sawReasoningDelta) {
          const completedReasoning =
            reasoningFromResponseItems(finalOutput);
          if (completedReasoning) {
            sawReasoningDelta = true;
            reasoningParts.push(completedReasoning);
            yield { type: "reasoning_delta", text: completedReasoning };
          }
        }
        yield {
          type: "finish",
          finish_reason: sawToolCall ? "tool_calls" : "stop",
          usage: usageData,
          reasoning_content: reasoningParts.join("") || null,
          response_id: response.id,
        };
        return;
      }
    }
    throw new ChatGPTOAuthError(
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
    } = {},
  ): Promise<Record<string, unknown>[]> {
    if (!prompt || prompt.trim() === "") {
      throw new ChatGPTOAuthError("image generation prompt is required");
    }
    const content: Record<string, unknown>[] = [
      { type: "input_text", text: prompt },
    ];
    content.push(
      ...validateImageContentItems(opts.referenceImages || []),
    );
    if (opts.size && opts.size !== "auto") {
      content[0].text = `${prompt}\n\nRequested output size/aspect: ${opts.size}`;
    }
    const requestModel = opts.model || this.model;
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
      prompt_cache_key: crypto.randomUUID(),
    };
    setReasoningPayload(
      payload,
      effectiveReasoningEffort(requestModel, opts.reasoningEffort, opts.reasoning),
      opts.reasoning,
      requestModel,
    );
    rejectUnsupportedPrivateRequestFields(
      payload,
      opts.safetyIdentifier,
      opts.promptCacheOptions,
    );
    validatePromptCacheBreakpoints(payload);
    finalizeResponsesPayload(payload, {
      endpoint: "responses",
      model: requestModel,
      responsesLite: opts.responsesLite,
      text: opts.text,
      tools: payload.tools as Record<string, unknown>[],
    });
    const outputItems = await this.collectResponseOutputItems(payload);
    const generated = outputItems
      .map(imageGenerationFromItem)
      .filter(
        (img): img is Record<string, unknown> => img !== null,
      );
    if (!generated.length) {
      throw new ChatGPTOAuthError(
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
    },
  ): Promise<string> {
    if (!prompt || prompt.trim() === "") {
      throw new ChatGPTOAuthError("image inspection prompt is required");
    }
    const content: Record<string, unknown>[] = [
      { type: "input_text", text: prompt },
    ];
    content.push(...validateImageContentItems(opts.images));
    const requestModel = opts.model || this.model;
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
      prompt_cache_key: crypto.randomUUID(),
    };
    setReasoningPayload(
      payload,
      effectiveReasoningEffort(requestModel, opts.reasoningEffort, opts.reasoning),
      opts.reasoning,
      requestModel,
    );
    rejectUnsupportedPrivateRequestFields(
      payload,
      opts.safetyIdentifier,
      opts.promptCacheOptions,
    );
    validatePromptCacheBreakpoints(payload);
    finalizeResponsesPayload(payload, {
      endpoint: "responses",
      model: requestModel,
      responsesLite: opts.responsesLite,
      text: opts.text,
      tools: payload.tools as Record<string, unknown>[],
    });
    const outputItems = await this.collectResponseOutputItems(payload);
    const text = textFromResponseItems(outputItems).trim();
    if (!text) {
      throw new ChatGPTOAuthError(
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
    } = {},
  ): Promise<string> {
    const requestModel = opts.model || this.model;
    const [baseInstructions, compactInput] = splitInstructionsAndInput(messages);
    const semanticInput = this.resolveSemanticInput(
      compactInput,
      opts.previousResponseId,
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
      opts.reasoningEffort ?? capabilityForModel(requestModel).defaultReasoningEffort,
    );
    rejectUnsupportedPrivateRequestFields(
      payload,
      undefined,
      opts.promptCacheOptions,
    );
    validatePromptCacheBreakpoints(payload);
    finalizeResponsesPayload(payload, {
      endpoint: "compact",
      model: requestModel,
      responsesLite: opts.responsesLite,
      serviceTier: opts.serviceTier,
      text: opts.text,
      tools: toolsPayload,
    });
    const data = await this.postJSON("/responses/compact", payload);
    const output = data.output;
    if (!Array.isArray(output)) {
      throw new ChatGPTOAuthError(
        "remote compact response missing output array",
      );
    }
    return REMOTE_COMPACTION_MARKER + "\n" + JSON.stringify(filterCompactedHistoryItems(output));
  }

  private async collectResponseOutputItems(
    payload: Record<string, unknown>,
  ): Promise<Record<string, unknown>[]> {
    const outputItems: Record<string, unknown>[] = [];
    let sawCompleted = false;

    for await (const event of this.postSSE("/responses", payload)) {
      const typ = event.type;
      if (typ === "response.output_item.done") {
        const item = event.item;
        if (typeof item !== "object" || item === null || Array.isArray(item)) {
          throw new ChatGPTOAuthError(
            "response.output_item.done must contain an object item",
          );
        }
        outputItems.push(item as Record<string, unknown>);
      } else if (typ === "response.failed") {
        throw new ChatGPTOAuthError(
          responseFailureMessage(event, "failed"),
        );
      } else if (typ === "response.incomplete") {
        throw new ChatGPTOAuthError(
          responseFailureMessage(event, "incomplete"),
        );
      } else if (typ === "response.completed") {
        completedResponseFromEvent(event);
        sawCompleted = true;
        break;
      }
    }
    if (!sawCompleted) {
      throw new ChatGPTOAuthError(
        "ChatGPT OAuth response stream ended before response.completed",
      );
    }
    return outputItems;
  }

  private responsesPayload(
    messages: Message[],
    opts: ChatOptions,
  ): Record<string, unknown> {
    rejectUnsupportedStop(opts.stop);
    const [instructions, inputItems] =
      splitInstructionsAndInput(messages);
    if (!instructions) {
      throw new ChatGPTOAuthError(
        "ChatGPT OAuth Responses request requires system instructions",
      );
    }
    const requestModel = opts.model || this.model;
    const semanticInput = this.resolveSemanticInput(
      inputItems,
      opts.previousResponseId,
    );
    const toolsPayload = opts.tools
      ? opts.tools.map(toolSchemaToResponseDict)
      : [];
    const lite = useResponsesLite(requestModel, opts.responsesLite);
    const payload: Record<string, unknown> = {
      model: wireModel(requestModel),
      instructions,
      input: semanticInput,
      tools: toolsPayload,
      tool_choice: opts.toolChoice ?? "auto",
      parallel_tool_calls: shouldEnableParallelToolCalls({
        model: requestModel,
        requested: opts.parallelToolCalls,
        responsesLite: lite,
      }),
      stream: true,
      store: false,
      include: [],
    };
    if ((payload.tools as Record<string, unknown>[]).some((tool) => tool.type === "web_search")) {
      payload.include = ["web_search_call.action.sources"];
    }
    void opts.maxTokens; // ChatGPT Codex backend rejects max_output_tokens for this endpoint.
    if (opts.promptCacheKey != null) {
      if (
        typeof opts.promptCacheKey !== "string"
        || opts.promptCacheKey.length === 0
      ) {
        throw new ChatGPTOAuthError(
          "prompt_cache_key must be a non-empty string when provided",
        );
      }
      payload.prompt_cache_key = opts.promptCacheKey;
    }
    if (opts.clientMetadata != null)
      payload.client_metadata = opts.clientMetadata;
    if (resolveCodexMetadataEnabled(opts.codexMetadata)) {
      payload.client_metadata = buildCodexClientMetadata({
        authJsonPath: this.authJsonPath,
        existing: typeof payload.client_metadata === "object" && payload.client_metadata !== null
          ? payload.client_metadata as Record<string, string>
          : undefined,
      });
    }
    setReasoningPayload(
      payload,
      effectiveReasoningEffort(requestModel, opts.reasoningEffort, opts.reasoning),
      opts.reasoning,
      requestModel,
    );
    rejectUnsupportedPrivateRequestFields(
      payload,
      opts.safetyIdentifier,
      opts.promptCacheOptions,
    );
    validatePromptCacheBreakpoints(payload);
    finalizeResponsesPayload(payload, {
      endpoint: "responses",
      model: requestModel,
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
    previousResponseId?: string,
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
      ...this.responseChains.resolve(previousResponseId),
      ...cloneResponseItems(input),
    ];
  }

  private getHeaders(): Record<string, string> {
    const token = loadTokenData(this.authJsonPath);
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
  ): Promise<Record<string, unknown>> {
    let tokenValues: (string | null)[] = [null];
    for (let attempt = 0; attempt < 2; attempt++) {
      const headers = this.getHeaders();
      if (isResponsesLitePayload(payload)) {
        headers[LITE_HEADER_NAME] = LITE_HEADER_VALUE;
      }
      const token = loadTokenData(this.authJsonPath);
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
          headers,
          body: JSON.stringify(payload),
          signal: this.timeout
            ? AbortSignal.timeout(this.timeout)
            : undefined,
        });
      } catch (err) {
        throw new ChatGPTOAuthError(
          `ChatGPT OAuth request failed: ${redactText(String(err), ...tokenValues)}`,
        );
      }

      if (!response.ok) {
        const body = await response.text();
        const redacted = redactText(body, ...tokenValues);
        if (response.status === 401 && attempt === 0) {
          await refreshToken(this.authJsonPath);
          continue;
        }
        throw new ChatGPTOAuthError(
          `ChatGPT OAuth request failed: HTTP ${response.status}: ${redacted}`,
        );
      }

      const data = await response.json();
      if (
        typeof data !== "object" ||
        data === null ||
        Array.isArray(data)
      ) {
        throw new ChatGPTOAuthError(
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
    extraHeaders: Record<string, string> = {},
  ): AsyncGenerator<Record<string, unknown>> {
    let tokenValues: (string | null)[] = [null];
    for (let attempt = 0; attempt < 2; attempt++) {
      const headers = this.getHeaders();
      headers.Accept = "text/event-stream";
      Object.assign(headers, extraHeaders);
      if (isResponsesLitePayload(payload)) {
        headers[LITE_HEADER_NAME] = LITE_HEADER_VALUE;
      }
      const token = loadTokenData(this.authJsonPath);
      tokenValues = [
        token.access_token,
        token.refresh_token,
        token.id_token,
        token.account_id,
      ];

      const url = this.baseUrl + path;
      let response: globalThis.Response;
      try {
        response = await fetch(url, {
          method: "POST",
          headers,
          body: JSON.stringify(payload),
          signal: this.timeout
            ? AbortSignal.timeout(this.timeout)
            : undefined,
        });
      } catch (err) {
        throw new ChatGPTOAuthError(
          `ChatGPT OAuth request failed: ${redactText(String(err), ...tokenValues)}`,
        );
      }

      if (!response.ok) {
        const body = await response.text();
        const redacted = redactText(body, ...tokenValues);
        if (response.status === 401 && attempt === 0) {
          await refreshToken(this.authJsonPath);
          continue;
        }
        throw new ChatGPTOAuthError(
          `ChatGPT OAuth request failed: HTTP ${response.status}: ${redacted}`,
        );
      }

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const block: string[] = [];

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            if (block.length) {
              const event = decodeSSEBlock(block);
              if (event) yield event;
            }
            return;
          }
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop()!;
          for (const rawLine of lines) {
            const line = rawLine.replace(/\r$/, "");
            if (line === "") {
              const event = decodeSSEBlock(block);
              block.length = 0;
              if (event) yield event;
              continue;
            }
            block.push(line);
          }
        }
      } catch (err) {
        if (err instanceof ChatGPTOAuthError) throw err;
        throw new ChatGPTOAuthError(
          `ChatGPT OAuth request failed: ${redactText(String(err), ...tokenValues)}`,
        );
      }
      return;
    }
  }
}

function rejectUnsupportedStop(stop: string | string[] | undefined): void {
  const values = stop == null ? [] : Array.isArray(stop) ? stop : [stop];
  if (!values.some((value) => value.length > 0)) return;
  throw new ChatGPTOAuthInvalidRequestError(
    "stop is not supported by the private Codex OAuth HTTP transport",
  );
}

// --- Helper functions (exported for testing) ---

function completedResponseFromEvent(event: Record<string, unknown>): Record<string, unknown> {
  const response = event.response;
  if (typeof response !== "object" || response === null || Array.isArray(response)) {
    throw new ChatGPTOAuthError(
      "response.completed must contain an object response",
    );
  }
  const responseRecord = response as Record<string, unknown>;
  if (typeof responseRecord.id !== "string" || responseRecord.id.length === 0) {
    throw new ChatGPTOAuthError(
      "response.completed response.id must be a non-empty string",
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
  const event = JSON.parse(joined);
  if (typeof event !== "object" || event === null || Array.isArray(event)) {
    throw new ChatGPTOAuthError("ChatGPT OAuth SSE event must be a JSON object");
  }
  return event;
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
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) {
        throw new ChatGPTOAuthError(
          "remote compaction marker must contain a response item array",
        );
      }
      items.push(...filterCompactedHistoryItems(parsed, "remote compaction marker"));
      continue;
    }

    if (message.role === MessageRole.TOOL) {
      items.push({
        type: "function_call_output",
        call_id:
          message.tool_call_id || message.name || "tool-call",
        output: message.content,
      });
      continue;
    }

    if (
      message.role === MessageRole.ASSISTANT &&
      message.tool_calls?.length
    ) {
      if (message.content || message.structured_content?.length) {
        items.push(messageItem(
          "assistant",
          message.content,
          undefined,
          message.structured_content,
        ));
      }
      for (const tc of message.tool_calls) {
        items.push({
          type: "function_call",
          call_id: tc.id,
          name: tc.name,
          arguments: JSON.stringify(tc.arguments),
        });
      }
      continue;
    }

    const role =
      message.role === MessageRole.ASSISTANT ? "assistant" : "user";
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
      } else {
        const item: Record<string, unknown> = {
          type: "input_image",
          image_url: part.image_url,
        };
        if (part.detail != null) item.detail = part.detail;
        contentItems.push(item);
      }
    }
  } else {
    contentItems.push({ type: typ, text: content || "" });
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
  if (tool.allowed_callers != null) {
    throw new ChatGPTOAuthError(
      "programmatic tool allowed_callers is not supported",
    );
  }
  if (tool.output_schema != null) {
    throw new ChatGPTOAuthError(
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
    return { type: "web_search", external_web_access: true };
  }
  return {
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
    strict: false,
  };
}

function finalizeResponsesPayload(
  payload: Record<string, unknown>,
  opts: {
    endpoint: "responses" | "compact";
    model: string;
    responsesLite?: boolean | string;
    serviceTier?: string;
    text?: Record<string, unknown>;
    tools: Record<string, unknown>[];
  },
): void {
  if (
    !capabilityForModel(opts.model).supportsImageDetailOriginal
    && hasOriginalImageDetail(payload)
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      `image detail "original" is not supported by model ${JSON.stringify(opts.model)}`,
    );
  }
  applyModelCapabilityFields(payload, opts.model, opts.text, opts.serviceTier);
  if (opts.endpoint === "compact") {
    delete payload.include;
  }
  if (!useResponsesLite(opts.model, opts.responsesLite)) return;

  const unsupportedTool = opts.tools.find(
    (tool) => tool.type === "web_search"
      || tool.type === "image_generation"
      || tool.type === "programmatic_tool_calling",
  );
  if (unsupportedTool) {
    throw new ChatGPTOAuthError(
      `Responses Lite cannot use hosted ${String(unsupportedTool.type)} without a standalone executor`,
    );
  }

  const instructions = String(payload.instructions ?? "");
  delete payload.instructions;
  delete payload.tools;
  if (opts.endpoint === "responses") {
    if (payload.tool_choice !== "auto") {
      throw new ChatGPTOAuthError(
        "Responses Lite requires tool_choice to be the exact string auto",
      );
    }
    payload.tool_choice = "auto";
  }
  payload.parallel_tool_calls = false;
  const input = Array.isArray(payload.input) ? payload.input : [];
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
  payload.input = stripImageDetailFields([...developerItems, ...input]);
  const reasoning = typeof payload.reasoning === "object"
    && payload.reasoning !== null
    && !Array.isArray(payload.reasoning)
    ? payload.reasoning as Record<string, unknown>
    : {};
  if (opts.endpoint === "responses") {
    if (reasoning.context != null && reasoning.context !== "all_turns") {
      throw new ChatGPTOAuthError(
        "Responses Lite reasoning.context must be all_turns when explicitly provided",
      );
    }
  } else {
    delete reasoning.mode;
  }
  reasoning.context = "all_turns";
  payload.reasoning = reasoning;
  (payload as Record<PropertyKey, unknown>)[RESPONSES_LITE_PAYLOAD] = true;
}

function hasOriginalImageDetail(value: unknown): boolean {
  if (Array.isArray(value)) return value.some(hasOriginalImageDetail);
  if (typeof value !== "object" || value === null) return false;
  const object = value as Record<string, unknown>;
  if (object.type === "input_image" && object.detail === "original") return true;
  return Object.values(object).some(hasOriginalImageDetail);
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
      throw new ChatGPTOAuthError(`${source} item ${index} must be an object`);
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
    throw new ChatGPTOAuthError(`${source} item ${index} must have a string type`);
  }
  if (item.type === "message") {
    if (typeof item.role !== "string") {
      throw new ChatGPTOAuthError(`${source} message item ${index} must have a string role`);
    }
    if (!Array.isArray(item.content)) {
      throw new ChatGPTOAuthError(`${source} message item ${index} must have an array content field`);
    }
    validateMessageContentItems(item.content, source, index);
    if (containsPromptCacheBreakpoint(item.content)) {
      throw new ChatGPTOAuthError(
        "prompt_cache_breakpoint is not supported by the private Codex OAuth HTTP transport",
      );
    }
    return;
  }
  if (item.type === "agent_message") {
    if (typeof item.author !== "string" || typeof item.recipient !== "string") {
      throw new ChatGPTOAuthError(
        `${source} agent_message item ${index} must have string author and recipient fields`,
      );
    }
    if (!Array.isArray(item.content)) {
      throw new ChatGPTOAuthError(
        `${source} agent_message item ${index} must have an array content field`,
      );
    }
    for (const [contentIndex, contentItem] of item.content.entries()) {
      if (typeof contentItem !== "object" || contentItem === null || Array.isArray(contentItem)) {
        throw new ChatGPTOAuthError(
          `${source} agent_message item ${index} content item ${contentIndex} must be an object`,
        );
      }
      const part = contentItem as Record<string, unknown>;
      const validInputText = part.type === "input_text" && typeof part.text === "string";
      const validEncrypted = part.type === "encrypted_content"
        && typeof part.encrypted_content === "string";
      if (!validInputText && !validEncrypted) {
        throw new ChatGPTOAuthError(
          `${source} agent_message item ${index} content item ${contentIndex} is invalid`,
        );
      }
    }
    return;
  }
  if (item.type === "compaction" || item.type === "compaction_summary") {
    if (typeof item.encrypted_content !== "string") {
      throw new ChatGPTOAuthError(
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
    throw new ChatGPTOAuthError(
      `${source} context_compaction item ${index} encrypted_content must be a string`,
    );
  }
}

function validateMessageContentItems(
  content: unknown[],
  source: string,
  index: number,
): void {
  for (const [contentIndex, contentItem] of content.entries()) {
    if (typeof contentItem !== "object" || contentItem === null || Array.isArray(contentItem)) {
      throw new ChatGPTOAuthError(
        `${source} message item ${index} content item ${contentIndex} must be an object`,
      );
    }
    const part = contentItem as Record<string, unknown>;
    if (part.type === "input_text" || part.type === "output_text") {
      if (typeof part.text !== "string") {
        throw new ChatGPTOAuthError(
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
        throw new ChatGPTOAuthError(
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
        throw new ChatGPTOAuthError(
          `${source} message item ${index} content item ${contentIndex} has invalid detail`,
        );
      }
      validatePromptCacheBreakpointValue(
        part.prompt_cache_breakpoint,
        `${source} message item ${index} content item ${contentIndex}`,
      );
      continue;
    }
    throw new ChatGPTOAuthError(
      `${source} message item ${index} content item ${contentIndex} has an unsupported type`,
    );
  }
}

function validatePromptCacheBreakpointValue(
  value: unknown,
  _source: string,
): void {
  if (value == null) return;
  throw new ChatGPTOAuthError(
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
  allItems: Record<string, unknown>[] = [],
): StreamEvent | null {
  if (item.type !== "web_search_call") return null;
  const rawId = String(item.id ?? item.call_id ?? crypto.randomUUID().replace(/-/g, ""));
  const id = rawId.startsWith("srvtoolu_") ? rawId : `srvtoolu_${rawId.replace(/[^A-Za-z0-9_]/g, "")}`;
  const action = (typeof item.action === "object" && item.action !== null && !Array.isArray(item.action))
    ? item.action as Record<string, unknown>
    : {};
  const query = webSearchQueryFromAction(action);
  const sources = webSearchSourcesFromAction(action);
  if (!sources.length && allItems.length) {
    sources.push(...webSearchSourcesFromAnnotations(allItems));
  }
  return {
    type: "web_search_call",
    id,
    input: { query },
    content: sources,
  };
}

function webSearchQueryFromAction(action: Record<string, unknown>): string {
  const query = action.query;
  if (typeof query === "string") return query;
  const queries = action.queries;
  if (Array.isArray(queries)) {
    const first = queries.find((q): q is string => typeof q === "string" && q.length > 0);
    if (first) return first;
  }
  const url = action.url;
  if (typeof url === "string") return url;
  return "";
}

function webSearchSourcesFromAction(action: Record<string, unknown>): Record<string, unknown>[] {
  return normalizeWebSearchSources(action.sources);
}

function webSearchSourcesFromAnnotations(items: Record<string, unknown>[]): Record<string, unknown>[] {
  const rawSources: Record<string, unknown>[] = [];
  for (const item of items) {
    const content = item.content;
    if (!Array.isArray(content)) continue;
    for (const part of content) {
      if (typeof part !== "object" || part === null) continue;
      const annotations = (part as Record<string, unknown>).annotations;
      if (!Array.isArray(annotations)) continue;
      for (const ann of annotations) {
        if (typeof ann !== "object" || ann === null) continue;
        const a = ann as Record<string, unknown>;
        if (a.type !== "url_citation") continue;
        rawSources.push(a);
      }
    }
  }
  return normalizeWebSearchSources(rawSources);
}

function normalizeWebSearchSources(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) return [];
  const out: Record<string, unknown>[] = [];
  const seen = new Set<string>();
  for (const source of value) {
    if (typeof source !== "object" || source === null) continue;
    const s = source as Record<string, unknown>;
    const url = typeof s.url === "string" ? s.url : "";
    if (!url || seen.has(url)) continue;
    seen.add(url);
    const result: Record<string, unknown> = {
      type: "web_search_result",
      url,
      title: typeof s.title === "string" ? s.title : url,
    };
    if (typeof s.page_age === "string") result.page_age = s.page_age;
    out.push(result);
  }
  return out;
}

function effectiveReasoningEffort(
  model: string,
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
    ?? (reasoning?.mode != null ? "medium" : undefined)
    ?? capabilityForModel(model).defaultReasoningEffort;
}

export function setReasoningPayload(
  payload: Record<string, unknown>,
  reasoningEffort?: string,
  reasoning?: ReasoningOptions,
  model?: string,
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
  const requestModel = model
    ?? (typeof payload.model === "string" ? payload.model : undefined);
  if (mode != null && !isGpt56Model(requestModel)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning.mode is supported only by GPT-5.6 models",
    );
  }
  if (mode === "pro") {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning.mode pro is not supported by the private Codex OAuth transport",
    );
  }

  const existing = typeof payload.reasoning === "object"
    && payload.reasoning !== null
    && !Array.isArray(payload.reasoning)
    ? payload.reasoning as Record<string, unknown>
    : {};
  const existingMode = mode == null ? existing.mode : undefined;
  if (
    existingMode != null
    && (typeof existingMode !== "string" || !REASONING_MODES.has(existingMode))
  ) {
    throw new ChatGPTOAuthInvalidRequestError("reasoning.mode must be one of: standard, pro");
  }
  if (existingMode != null && !isGpt56Model(requestModel)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning.mode is supported only by GPT-5.6 models",
    );
  }
  if (existingMode === "pro") {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning.mode pro is not supported by the private Codex OAuth transport",
    );
  }
  const merged: Record<string, unknown> = { ...existing };
  delete merged.mode;
  if (selectedEffort != null) {
    merged.effort = KNOWN_REASONING_EFFORT_VALUES.has(selectedEffort)
      ? selectedEffort === "ultra" ? "max" : selectedEffort
      : selectedEffort;
  }
  // The private Codex request schema has no mode field. Public "standard"
  // retains its public default-effort behavior but is omitted on the wire.
  if (context != null) merged.context = context;
  if (Object.keys(merged).length === 0) {
    delete payload.reasoning;
    return;
  }
  payload.reasoning = merged;
  const include = Array.isArray(payload.include) ? payload.include : [];
  if (!include.includes("reasoning.encrypted_content")) {
    include.push("reasoning.encrypted_content");
  }
  payload.include = include;
}

function isGpt56Model(model: string | undefined): boolean {
  return typeof model === "string" && /^gpt-5\.6(?:$|-)/.test(model);
}

function wireModel(model: string): string {
  return model === "gpt-5.6" ? "gpt-5.6-sol" : model;
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
  if (!containsPromptCacheBreakpoint(payload.input)) return;
  throw new ChatGPTOAuthInvalidRequestError(
    "prompt_cache_breakpoint is not supported by the private Codex OAuth HTTP transport",
  );
}

function containsPromptCacheBreakpoint(value: unknown): boolean {
  if (Array.isArray(value)) return value.some(containsPromptCacheBreakpoint);
  if (typeof value !== "object" || value === null) return false;
  const record = value as Record<string, unknown>;
  if (record.prompt_cache_breakpoint != null) return true;
  return Object.values(record).some(containsPromptCacheBreakpoint);
}

export function toolCallFromResponseItem(
  item: Record<string, unknown>,
): ToolCall | null {
  if (
    item.type !== "function_call" &&
    item.type !== "custom_tool_call"
  )
    return null;
  const name = item.name;
  if (typeof name !== "string" || !name) return null;
  const rawArgs = item.arguments ?? item.input ?? "{}";
  let args: Record<string, unknown>;
  if (typeof rawArgs === "string") {
    try {
      args = rawArgs ? JSON.parse(rawArgs) : {};
    } catch {
      args = { input: rawArgs };
    }
  } else if (
    typeof rawArgs === "object" &&
    rawArgs !== null &&
    !Array.isArray(rawArgs)
  ) {
    args = rawArgs as Record<string, unknown>;
  } else {
    args = {};
  }
  const callId = String(
    item.call_id ??
      item.id ??
      crypto.randomUUID().replace(/-/g, ""),
  );
  return { id: callId, name, arguments: args };
}

export function textFromResponseItems(
  items: Record<string, unknown>[],
): string {
  const parts: string[] = [];
  for (const item of items) {
    const itemType = item.type;
    if (itemType === "output_text" || itemType === "text") {
      const text = item.text;
      if (typeof text === "string" && text) parts.push(text);
      continue;
    }
    if (itemType !== "message") continue;
    const content = item.content;
    if (!Array.isArray(content)) continue;
    for (const part of content) {
      if (typeof part === "string") {
        if (part) parts.push(part);
        continue;
      }
      if (typeof part !== "object" || part === null) continue;
      const p = part as Record<string, unknown>;
      if (p.type !== "output_text" && p.type !== "text") continue;
      const text = p.text;
      if (typeof text === "string" && text) parts.push(text);
    }
  }
  return parts.join("");
}

export function validateImageContentItems(
  images: ImageReference[],
): Record<string, unknown>[] {
  if (!Array.isArray(images)) {
    throw new ChatGPTOAuthError("image references must be an array");
  }
  const items: Record<string, unknown>[] = [];
  for (let i = 0; i < images.length; i++) {
    const image = images[i];
    if (typeof image !== "object" || image === null) {
      throw new ChatGPTOAuthError(
        `image reference ${i} must be an object`,
      );
    }
    const imageUrl = image.image_url;
    if (typeof imageUrl !== "string" || !imageUrl.trim()) {
      throw new ChatGPTOAuthError(
        `image reference ${i} requires image_url`,
      );
    }
    if (!imageUrl.startsWith("data:image/")) {
      throw new ChatGPTOAuthError(
        `image reference ${i} must be a data:image URL`,
      );
    }
    const item: Record<string, unknown> = {
      type: "input_image",
      image_url: imageUrl,
    };
    if (image.detail != null) {
      if (typeof image.detail !== "string" || !IMAGE_DETAILS.has(image.detail)) {
        throw new ChatGPTOAuthError(
          `image reference ${i} detail must be one of: auto, low, high, original`,
        );
      }
      item.detail = image.detail;
    }
    if (image.prompt_cache_breakpoint !== undefined) {
      throw new ChatGPTOAuthError(
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
  if (typeof result !== "string" || !result.trim()) {
    throw new ChatGPTOAuthError(
      "image_generation_call returned empty result",
    );
  }
  return {
    id: String(
      item.id ?? crypto.randomUUID().replace(/-/g, ""),
    ),
    status: String(item.status ?? "completed"),
    revised_prompt:
      typeof item.revised_prompt === "string"
        ? item.revised_prompt
        : null,
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
  const prompt = v.input_tokens ?? v.prompt_tokens;
  const completion = v.output_tokens ?? v.completion_tokens;
  const total = v.total_tokens;
  if (typeof prompt !== "number" || typeof completion !== "number")
    return null;
  const tokenDetails =
    v.input_tokens_details ?? v.prompt_tokens_details;
  let cachedTokens = 0;
  let cacheWriteTokens = 0;
  if (typeof tokenDetails === "object" && tokenDetails !== null) {
    const d = tokenDetails as Record<string, unknown>;
    if (typeof d.cached_tokens === "number")
      cachedTokens = d.cached_tokens;
    if (typeof d.cache_write_tokens === "number")
      cacheWriteTokens = d.cache_write_tokens;
  }
  if (cachedTokens === 0) {
    if (typeof v.cached_input_tokens === "number") {
      cachedTokens = v.cached_input_tokens;
    } else if (typeof v.cache_read_input_tokens === "number") {
      cachedTokens = v.cache_read_input_tokens;
    }
  }
  if (cacheWriteTokens === 0) {
    const topLevelCacheWrite = v.cache_write_tokens
      ?? v.cache_write_input_tokens
      ?? v.cache_creation_input_tokens;
    if (typeof topLevelCacheWrite === "number") {
      cacheWriteTokens = topLevelCacheWrite;
    }
  }
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens:
      typeof total === "number" ? total : prompt + completion,
    cached_tokens: cachedTokens,
    cache_write_tokens: cacheWriteTokens,
  };
}

export { REMOTE_COMPACTION_MARKER };
