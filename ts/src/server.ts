import * as crypto from "node:crypto";
import { isDeepStrictEqual } from "node:util";
import express, { type NextFunction, type Request, type Response } from "express";
import {
  ChatGPTOAuthError,
  ChatGPTOAuthCatalogUnavailableError,
  ChatGPTOAuthInvalidRequestError,
  ChatGPTOAuthMissingError,
  ChatGPTOAuthModelNotFoundError,
  ChatGPTOAuthProtocolError,
  ChatGPTOAuthRefreshError,
  ChatGPTOAuthUnavailableError,
  ChatGPTOAuthUpstreamError,
  validateAuthEnvironment,
} from "./auth.js";
import type {
  Message,
  MessageContentPart,
  ToolCall,
  ToolSchema,
  Usage,
} from "./messages.js";
import { MessageRole } from "./messages.js";
import {
  ChatGPTOAuthProvider,
  requireSubagentHeaderValue,
  resolveCodexCliVersion,
  usageFromResponse,
  validateImageContentItems,
  wireReasoningEffort,
} from "./provider.js";
import type { ImageReference, ReasoningOptions, StreamEvent } from "./provider.js";
import {
  anthropicRequestToInternal,
  internalResponseToAnthropic,
  anthropicStreamAdapter,
  formatAnthropicError,
} from "./anthropic-adapter.js";
import { loadCodexConfig, type CodexConfig } from "./codex-config.js";
import {
  resolveCodexMetadataEnabled,
  resolveResponsesLiteMode,
  type ModelCapability,
} from "./model-capabilities.js";
import { countO200kOrdinaryTokens } from "./o200k-tokenizer.js";
import { normalizeFinishReason } from "./protocol.js";
import { decodeUtf8Strict, parseJsonStrict } from "./utf8-json.js";

const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 18080;
export const REQUEST_BODY_LIMIT_BYTES = 50 * 1024 * 1024;
const PUBLIC_AUTH_ERROR_MESSAGE =
  "ChatGPT OAuth credentials are unavailable; rerun codex login";

const UNSUPPORTED_GENERATION_FIELDS = [
  "top_p",
  "top_k",
  "frequency_penalty",
  "presence_penalty",
  "seed",
  "logprobs",
  "top_logprobs",
  "logit_bias",
  "n",
  "response_format",
  "stream_options",
  "modalities",
  "audio",
  "prediction",
  "web_search_options",
  "store",
  "metadata",
  "user",
  "quality",
  "style",
  "background",
  "moderation",
  "output_compression",
  "safety_identifier",
  "prompt_cache_options",
  "multi_agent",
  "programmatic_tool_calling",
] as const;

const CHAT_COMPLETION_FIELDS = new Set<string>([
  "audio",
  "frequency_penalty",
  "function_call",
  "functions",
  "logit_bias",
  "logprobs",
  "max_completion_tokens",
  "max_tokens",
  "metadata",
  "modalities",
  "n",
  "prediction",
  "presence_penalty",
  "prompt_cache_retention",
  "reasoning_effort",
  "response_format",
  "seed",
  "service_tier",
  "stop",
  "store",
  "stream",
  "stream_options",
  "temperature",
  "top_logprobs",
  "top_p",
  "user",
  "verbosity",
  "web_search_options",
  "model",
  "messages",
  "tools",
  "tool_choice",
  "prompt_cache_key",
  "previous_response_id",
  "client_metadata",
  "codex_metadata",
  "responses_lite",
  "parallel_tool_calls",
  "subagent",
  "memgen_request",
  "reasoning",
  "text",
  "safety_identifier",
  "prompt_cache_options",
  "multi_agent",
  "programmatic_tool_calling",
]);

const IMAGE_GENERATION_FIELDS = new Set<string>([
  "background",
  "model",
  "moderation",
  "n",
  "output_compression",
  "output_format",
  "partial_images",
  "prompt",
  "quality",
  "response_format",
  "size",
  "stream",
  "style",
  "user",
  "reference_images",
  "responses_lite",
  "reasoning",
  "reasoning_effort",
  "verbosity",
  "safety_identifier",
  "prompt_cache_options",
  "multi_agent",
  "programmatic_tool_calling",
  "tools",
]);

const INSPECT_FIELDS = new Set<string>([
  "model",
  "prompt",
  "images",
  "responses_lite",
  "reasoning",
  "reasoning_effort",
  "verbosity",
  "safety_identifier",
  "prompt_cache_options",
  "multi_agent",
  "programmatic_tool_calling",
  "tools",
]);

const OPENAI_COMPACT_FIELDS = new Set<string>([
  "model",
  "messages",
  "tools",
  "programmatic_tool_calling",
  "safety_identifier",
  "include",
  "prompt_cache_retention",
  "reasoning",
  "multi_agent",
  "reasoning_effort",
  "responses_lite",
  "previous_response_id",
  "service_tier",
  "text",
  "prompt_cache_key",
  "prompt_cache_options",
  "verbosity",
]);

const ANTHROPIC_COMPACT_FIELDS = new Set<string>([
  "model",
  "messages",
  "system",
  "tools",
  "tool_choice",
  "stop_sequences",
  "thinking",
  "output_config",
  "output_format",
  "context_management",
  "max_tokens",
  "programmatic_tool_calling",
  "safety_identifier",
  "include",
  "prompt_cache_retention",
  "reasoning",
  "multi_agent",
  "reasoning_effort",
  "responses_lite",
  "previous_response_id",
  "service_tier",
  "speed",
  "text",
  "prompt_cache_key",
  "prompt_cache_options",
  "verbosity",
]);

const ANTHROPIC_COUNT_FIELDS = new Set<string>([
  "model",
  "messages",
  "system",
  "max_tokens",
  "tools",
  "tool_choice",
  "stop_sequences",
  "thinking",
  "output_format",
  "output_config",
  "context_management",
]);

const ANTHROPIC_MESSAGE_FIELDS = new Set<string>([
  "model",
  "messages",
  "system",
  "max_tokens",
  "tools",
  "tool_choice",
  "stop_sequences",
  "thinking",
  "output_format",
  "output_config",
  "context_management",
  "previous_response_id",
  "responses_lite",
  "stream",
  "prompt_cache_key",
  "subagent",
  "memgen_request",
  "reasoning",
  "reasoning_effort",
  "verbosity",
  "speed",
  "service_tier",
]);
function errorStatus(err: unknown): number {
  if (err instanceof ChatGPTOAuthUpstreamError) {
    return err.status >= 100 && err.status <= 599 ? err.status : 500;
  }
  if (err instanceof ChatGPTOAuthMissingError || err instanceof ChatGPTOAuthRefreshError) return 401;
  if (err instanceof ChatGPTOAuthModelNotFoundError) return 404;
  if (err instanceof ChatGPTOAuthCatalogUnavailableError) return 503;
  if (err instanceof ChatGPTOAuthUnavailableError) return 502;
  if (err instanceof ChatGPTOAuthProtocolError) return 502;
  if (err instanceof ChatGPTOAuthInvalidRequestError) return 400;
  if (err instanceof ChatGPTOAuthError) return 401;
  return 500;
}

function errorType(err: unknown): string {
  if (err instanceof ChatGPTOAuthInvalidRequestError) return "invalid_request_error";
  if (err instanceof ChatGPTOAuthModelNotFoundError) return "model_not_found";
  if (err instanceof ChatGPTOAuthMissingError || err instanceof ChatGPTOAuthRefreshError) return "authentication_error";
  if (err instanceof ChatGPTOAuthCatalogUnavailableError) return "catalog_unavailable";
  if (err instanceof ChatGPTOAuthProtocolError) return "upstream_protocol_error";
  if (err instanceof ChatGPTOAuthUpstreamError) return "upstream_error";
  if (err instanceof ChatGPTOAuthUnavailableError) return "upstream_error";
  if (err instanceof ChatGPTOAuthError) return "authentication_error";
  return "server_error";
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function publicErrorMessage(err: unknown): string {
  if (
    err instanceof ChatGPTOAuthInvalidRequestError
    || err instanceof ChatGPTOAuthModelNotFoundError
  ) {
    return errorMessage(err);
  }
  if (
    err instanceof ChatGPTOAuthMissingError
    || err instanceof ChatGPTOAuthRefreshError
    || (err instanceof ChatGPTOAuthError && err.constructor === ChatGPTOAuthError)
  ) {
    return PUBLIC_AUTH_ERROR_MESSAGE;
  }
  if (err instanceof ChatGPTOAuthCatalogUnavailableError) {
    return "authenticated model catalog is unavailable";
  }
  if (err instanceof ChatGPTOAuthProtocolError) {
    return "upstream protocol validation failed";
  }
  if (
    err instanceof ChatGPTOAuthUpstreamError
    || err instanceof ChatGPTOAuthUnavailableError
  ) {
    return "upstream request failed";
  }
  if (err instanceof ChatGPTOAuthError) return PUBLIC_AUTH_ERROR_MESSAGE;
  return "internal server error";
}

function healthErrorMessage(err: unknown): string {
  if (err instanceof ChatGPTOAuthMissingError || err instanceof ChatGPTOAuthRefreshError) {
    return "ChatGPT OAuth credentials are unavailable";
  }
  if (err instanceof ChatGPTOAuthCatalogUnavailableError) {
    return "authenticated model catalog is unavailable";
  }
  if (err instanceof ChatGPTOAuthModelNotFoundError) {
    return "configured model is unavailable in the authenticated catalog";
  }
  if (err instanceof ChatGPTOAuthInvalidRequestError) {
    return "health configuration is invalid";
  }
  if (err instanceof ChatGPTOAuthProtocolError) {
    return "upstream protocol validation failed";
  }
  if (err instanceof ChatGPTOAuthUpstreamError || err instanceof ChatGPTOAuthUnavailableError) {
    return "upstream request failed";
  }
  return "health preflight failed";
}

function handleError(err: unknown, res: Response): void {
  const status = errorStatus(err);
  const type = errorType(err);
  const body = {
    error: { message: publicErrorMessage(err), type, code: type },
  };

  if (res.headersSent) {
    if (!res.writableEnded) res.end();
    return;
  }

  res.status(status).json(body);
}

function sendRequestTransportError(
  req: Request,
  res: Response,
  status: number,
  message: string,
): void {
  if (req.path.startsWith("/v1/messages")) {
    res.status(status).json(formatAnthropicError(status, message));
    return;
  }
  const type = "invalid_request_error";
  res.status(status).json({ error: { message, type, code: type } });
}

function isAcceptedJsonContentType(value: string | undefined): boolean {
  if (value == null) return false;
  const mediaType = value.split(";", 1)[0]!.trim().toLowerCase();
  return mediaType === "application/json"
    || (mediaType.startsWith("application/") && mediaType.endsWith("+json"));
}

function downstreamAbortController(res: Response): {
  controller: AbortController;
  dispose: () => void;
} {
  const controller = new AbortController();
  const onClose = () => {
    if (!res.writableEnded) {
      controller.abort(new Error("downstream client disconnected"));
    }
  };
  res.once("close", onClose);
  return {
    controller,
    dispose: () => res.off("close", onClose),
  };
}

async function writeWithBackpressure(
  res: Response,
  chunk: string,
  signal: AbortSignal,
): Promise<void> {
  if (signal.aborted || res.destroyed || res.writableEnded) {
    throw new Error("downstream client disconnected");
  }
  if (res.write(chunk)) return;
  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      res.off("drain", onDrain);
      res.off("close", onClose);
      signal.removeEventListener("abort", onAbort);
    };
    const onDrain = () => {
      cleanup();
      resolve();
    };
    const onClose = () => {
      cleanup();
      reject(new Error("downstream client disconnected"));
    };
    const onAbort = () => {
      cleanup();
      reject(new Error("downstream client disconnected"));
    };
    res.once("drain", onDrain);
    res.once("close", onClose);
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

function writeOpenAIStreamError(err: unknown, res: Response): void {
  if (res.writableEnded) return;
  const type = errorType(err);
  res.write(
    `data: ${JSON.stringify({
      error: { message: publicErrorMessage(err), type, code: type },
    })}\n\n`,
  );
  res.end();
}

function handleAnthropicError(err: unknown, res: Response): void {
  const status = errorStatus(err);
  const body = formatAnthropicError(status, publicErrorMessage(err));

  if (res.headersSent) {
    if (!res.writableEnded) {
      res.write(`event: error\ndata: ${JSON.stringify(body)}\n\n`);
      res.end();
    }
    return;
  }

  res.status(status).json(body);
}

export interface CreateAppOptions {
  provider?: ChatGPTOAuthProvider;
  codexConfig?: CodexConfig;
  model?: string;
  authPath?: string;
}

export function resolveServerHost(raw = process.env.CODEX_AS_API_HOST): string {
  if (raw === undefined) return DEFAULT_HOST;
  if (raw.trim().length === 0) {
    throw new ChatGPTOAuthInvalidRequestError("CODEX_AS_API_HOST must not be blank");
  }
  return raw;
}

export function resolveServerPort(raw = process.env.CODEX_AS_API_PORT): number {
  if (raw === undefined) return DEFAULT_PORT;
  if (!/^[0-9]+$/.test(raw)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "CODEX_AS_API_PORT must be an integer from 1 through 65535",
    );
  }
  const port = Number(raw);
  if (!Number.isSafeInteger(port) || port < 1 || port > 65_535) {
    throw new ChatGPTOAuthInvalidRequestError(
      "CODEX_AS_API_PORT must be an integer from 1 through 65535",
    );
  }
  return port;
}

export function createApp(opts?: CreateAppOptions): express.Express {
  resolveServerHost();
  resolveServerPort();
  validateAuthEnvironment();
  resolveResponsesLiteMode();
  resolveCodexMetadataEnabled();
  if (process.env.CODEX_AS_API_CODEX_CLI_VERSION !== undefined) {
    throw new ChatGPTOAuthInvalidRequestError(
      `CODEX_AS_API_CODEX_CLI_VERSION is not supported; the wire contract is pinned to ${resolveCodexCliVersion()}`,
    );
  }
  if (process.env.CODEX_HOME !== undefined && process.env.CODEX_HOME.trim().length === 0) {
    throw new ChatGPTOAuthInvalidRequestError("CODEX_HOME must not be blank");
  }
  if (
    process.env.CODEX_AS_API_AUTH_PATH !== undefined
    && process.env.CODEX_AS_API_AUTH_PATH.trim().length === 0
  ) {
    throw new ChatGPTOAuthInvalidRequestError("CODEX_AS_API_AUTH_PATH must not be blank");
  }
  if (opts?.authPath != null && opts.authPath.trim().length === 0) {
    throw new ChatGPTOAuthInvalidRequestError("authPath must not be blank");
  }
  const codexConfig = opts?.codexConfig ?? loadCodexConfig();
  if (codexConfig.codexHome.trim().length === 0) {
    throw new ChatGPTOAuthInvalidRequestError("Codex config home path must not be blank");
  }
  if (codexConfig.configPath.trim().length === 0) {
    throw new ChatGPTOAuthInvalidRequestError("Codex config path must not be blank");
  }
  const envModelValue = process.env.CODEX_AS_API_MODEL;
  if (envModelValue != null) validateModelValue(envModelValue, "CODEX_AS_API_MODEL");
  if (opts?.model != null) validateModelValue(opts.model, "model");
  if (codexConfig.model != null) validateModelValue(codexConfig.model, "config.toml model");
  const envModel = envModelValue;
  const model = opts?.model ?? envModel ?? codexConfig.model;
  const authPath = opts?.authPath ?? process.env.CODEX_AS_API_AUTH_PATH;
  const provider =
    opts?.provider ??
    new ChatGPTOAuthProvider({
      model,
      authJsonPath: authPath,
    });
  const prepareAnthropicRouteModel = async (
    requestedModel: string | undefined,
  ) => {
    if (requestedModel == null) {
      throw new ChatGPTOAuthInvalidRequestError(
        "Anthropic-compatible routes require an explicit model",
      );
    }
    validateModelValue(requestedModel, "model");
    return requestedModel.startsWith("claude-")
      ? provider.prepareAnthropicModel(requestedModel)
      : provider.prepareModel(requestedModel);
  };

  const app = express();
  app.disable("etag");
  app.use((req: Request, res: Response, next: NextFunction) => {
    if (
      ["POST", "PUT", "PATCH"].includes(req.method)
      && !isAcceptedJsonContentType(req.headers["content-type"])
    ) {
      sendRequestTransportError(
        req,
        res,
        415,
        "request Content-Type must be application/json or application/*+json",
      );
      return;
    }
    next();
  });
  app.use(express.json({
    limit: REQUEST_BODY_LIMIT_BYTES,
    type: ["application/json", "application/*+json"],
    verify: (_req, _res, buffer) => {
      try {
        parseJsonStrict(decodeUtf8Strict(buffer));
      } catch {
        throw new ChatGPTOAuthInvalidRequestError(
          "request body must be valid strict JSON",
        );
      }
    },
  }));

  app.get("/health", async (_req: Request, res: Response) => {
    let authAvailable = false;
    let catalogStatus: "fresh" | "unavailable" = "unavailable";
    let catalogFetchedAt: string | null = null;
    let catalogExpiresAt: string | null = null;
    try {
      const snapshot = await provider.catalogSnapshot();
      authAvailable = true;
      catalogStatus = "fresh";
      catalogFetchedAt = new Date(snapshot.fetchedAt).toISOString();
      catalogExpiresAt = new Date(snapshot.expiresAt).toISOString();
      const prepared = await provider.prepareModel(model, snapshot);
      const reasoningEffort = resolveReasoningEffort(
        undefined,
        codexConfig,
        prepared.capability,
      );
      const contextWindow = getContextWindow(prepared.capability, codexConfig);
      const autoCompactTokenLimit = getAutoCompactTokenLimit(
        prepared.capability,
        codexConfig,
      );
      res.json({
        status: "ok",
        auth_available: true,
        catalog_status: "fresh",
        catalog_fetched_at: catalogFetchedAt,
        catalog_expires_at: catalogExpiresAt,
        model: prepared.slug,
        reasoning_effort: reasoningEffort ?? null,
        context_window: contextWindow ?? null,
        auto_compact_token_limit: autoCompactTokenLimit ?? null,
      });
    } catch (err) {
      const type = errorType(err);
      res.status(503).json({
        status: "error",
        auth_available: authAvailable || type !== "authentication_error",
        catalog_status: catalogStatus,
        catalog_fetched_at: catalogFetchedAt,
        catalog_expires_at: catalogExpiresAt,
        model: null,
        reasoning_effort: null,
        context_window: null,
        auto_compact_token_limit: null,
        error: {
          type,
          message: healthErrorMessage(err),
        },
      });
    }
  });

  app.get("/v1/models", async (_req: Request, res: Response) => {
    try {
      const snapshot = await provider.catalogSnapshot();
      res.json({
        object: "list",
        data: snapshot.models.map((entry) => ({
          id: entry.slug,
          object: "model",
          owned_by: "openai",
          display_name: entry.displayName,
          description: entry.description,
          priority: entry.priority,
          visibility: entry.visibility,
          supported_in_api: entry.supportedInApi,
          default_reasoning_level: entry.defaultReasoningEffort ?? null,
          supported_reasoning_levels: entry.supportedReasoningEfforts,
          multi_agent_reasoning_effort: entry.multiAgentReasoningEffort ?? null,
          supports_reasoning_summary_parameter: entry.supportsReasoningSummaryParameter,
          default_reasoning_summary: entry.defaultReasoningSummary,
          comp_hash: entry.compHash ?? null,
          context_window: entry.contextWindow ?? null,
          max_context_window: entry.maxContextWindow ?? null,
          auto_compact_token_limit: entry.autoCompactTokenLimit ?? null,
          effective_context_window_percent: entry.effectiveContextWindowPercent,
          service_tiers: entry.serviceTiers,
          default_service_tier: entry.defaultServiceTier,
          input_modalities: entry.inputModalities,
          supports_image_detail_original: entry.supportsImageDetailOriginal,
          support_verbosity: entry.supportVerbosity,
          default_verbosity: entry.defaultVerbosity,
          use_responses_lite: entry.useResponsesLite,
        })),
      });
    } catch (err) {
      handleError(err, res);
    }
  });

  app.post(
    "/v1/chat/completions",
    async (req: Request, res: Response) => {
      try {
        const body = requireRequestBody(req.body);
        assertAllowedTopLevelFields(body, CHAT_COMPLETION_FIELDS, "/v1/chat/completions");
        rejectExplicitNullFields(body, [
          "model",
          "function_call",
          "functions",
          "parallel_tool_calls",
          "prompt_cache_key",
          "response_format",
          "safety_identifier",
          "tool_choice",
          "tools",
          "user",
          "web_search_options",
        ]);
        rejectUnsupportedGenerationFeatures(body);
        rejectNonNullUnsupportedFields(body, [
          "function_call",
          "functions",
          "prompt_cache_retention",
        ]);
        const messages = requestMessagesToInternal(
          requiredMessageRecords(body),
        );
        const tools = parseTools(body.tools);
        if (tools?.some((tool) => tool.parameters.__codex_as_api_tool_type === "web_search")) {
          throw new ChatGPTOAuthInvalidRequestError(
            "web_search tools cannot be represented by /v1/chat/completions",
          );
        }
        const stop = normalizeStop(body.stop);
        const maxTokens = mergeOptionalNumbers(
          body.max_completion_tokens,
          body.max_tokens,
          "max_completion_tokens",
          "max_tokens",
        );
        const toolChoice = optionalToolChoice(body.tool_choice);
        const temperature = optionalFiniteNumber(body.temperature, "temperature");
        rejectUnsupportedTransportControls({ maxTokens, stop, temperature });
        const promptCacheKey = resolvePromptCacheKey(body.prompt_cache_key);
        const previousResponseId = resolvePreviousResponseId(body.previous_response_id);
        const serviceTier = optionalString(body.service_tier, "service_tier");
        const clientMetadata = optionalStringRecord(body.client_metadata, "client_metadata");
        const codexMetadata = optionalBoolean(body.codex_metadata, "codex_metadata");
        const responsesLite = optionalBooleanOrString(body.responses_lite, "responses_lite");
        const parallelToolCalls = optionalBoolean(body.parallel_tool_calls, "parallel_tool_calls");
        const stream = optionalBoolean(body.stream, "stream") ?? false;
        const requestedModel = optionalModel(body.model);

        const subagent = mergeBodyAndHeaderString(
          optionalString(body.subagent, "subagent"),
          optionalHeader(req.headers["x-openai-subagent"], "x-openai-subagent"),
          "subagent",
          "x-openai-subagent",
        );
        if (subagent != null) requireSubagentHeaderValue(subagent);
        const memgenRequest = mergeBodyAndHeaderBoolean(
          optionalBoolean(body.memgen_request, "memgen_request"),
          optionalBooleanHeader(
            req.headers["x-openai-memgen-request"],
            "x-openai-memgen-request",
          ),
          "memgen_request",
          "x-openai-memgen-request",
        );
        const preparedModel = await provider.prepareModel(requestedModel ?? model);
        const requestModel = preparedModel.slug;
        const reasoning = resolveReasoning(
          body.reasoning,
          body.reasoning_effort,
          codexConfig,
          preparedModel.capability,
        );

        const chatOpts = {
          model: requestModel,
          tools,
          toolChoice,
          temperature,
          reasoningEffort: reasoning?.effort,
          reasoning,
          maxTokens,
          stop,
          promptCacheKey,
          subagent,
          memgenRequest,
          previousResponseId,
          serviceTier,
          text: resolveTextOptions(body.text, body.verbosity),
          clientMetadata,
          codexMetadata,
          responsesLite,
          parallelToolCalls,
          preparedModel,
        };

        const modelId = requestModel;

        if (stream) {
          // ChatGPTOAuthProvider builds and validates the deterministic request
          // synchronously, before Express commits streaming response headers.
          const downstream = downstreamAbortController(res);
          let responseStream: AsyncGenerator<StreamEvent> | undefined;
          try {
            responseStream = await provider.createChatStream(messages, {
              ...chatOpts,
              signal: downstream.controller.signal,
            });
          res.setHeader("Content-Type", "text/event-stream");
          res.setHeader("Cache-Control", "no-cache");
          res.setHeader("Connection", "keep-alive");

          const requestId = `chatcmpl-${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;
          const created = Math.floor(Date.now() / 1000);

          const preamble = {
            id: requestId,
            object: "chat.completion.chunk",
            created,
            model: modelId,
            choices: [
              {
                index: 0,
                delta: { role: "assistant" },
                finish_reason: null,
              },
            ],
          };
          await writeWithBackpressure(
            res,
            `data: ${JSON.stringify(preamble)}\n\n`,
            downstream.controller.signal,
          );

          let parsedUsage: Usage | null = null;
          let upstreamResponseId: string | null = null;
          let sawFinish = false;
          const toolCallIndices = new Map<string, number>();

          for await (const event of responseStream) {
            if (sawFinish) {
              throw new ChatGPTOAuthProtocolError("provider emitted an event after finish");
            }
            const typ = event.type;
            if (typ === "content") {
              if (typeof event.text !== "string") {
                throw new ChatGPTOAuthProtocolError("provider content event text must be a string");
              }
              const chunk = {
                id: requestId,
                object: "chat.completion.chunk",
                created,
                model: modelId,
                choices: [
                  {
                    index: 0,
                    delta: { content: event.text },
                    finish_reason: null,
                  },
                ],
              };
              await writeWithBackpressure(res, `data: ${JSON.stringify(chunk)}\n\n`, downstream.controller.signal);
            } else if (typ === "reasoning_delta") {
              if (typeof event.text !== "string") {
                throw new ChatGPTOAuthProtocolError("provider reasoning event text must be a string");
              }
              const chunk = {
                id: requestId,
                object: "chat.completion.chunk",
                created,
                model: modelId,
                choices: [
                  {
                    index: 0,
                    delta: { reasoning_content: event.text },
                    finish_reason: null,
                  },
                ],
              };
              await writeWithBackpressure(res, `data: ${JSON.stringify(chunk)}\n\n`, downstream.controller.signal);
            } else if (typ === "reasoning_raw_delta") {
              if (typeof event.text !== "string") {
                throw new ChatGPTOAuthProtocolError("provider reasoning event text must be a string");
              }
              const chunk = {
                id: requestId,
                object: "chat.completion.chunk",
                created,
                model: modelId,
                choices: [
                  {
                    index: 0,
                    delta: { reasoning: event.text },
                    finish_reason: null,
                  },
                ],
              };
              await writeWithBackpressure(res, `data: ${JSON.stringify(chunk)}\n\n`, downstream.controller.signal);
            } else if (typ === "tool_call") {
              if (typeof event.id !== "string") {
                throw new ChatGPTOAuthProtocolError("provider tool_call event requires a string id");
              }
              if (typeof event.name !== "string") {
                throw new ChatGPTOAuthProtocolError("provider tool_call event requires a string name");
              }
              if (typeof event.arguments !== "string") {
                throw new ChatGPTOAuthProtocolError("provider tool_call event arguments must be a string");
              }
              const toolCallId = event.id;
              if (toolCallIndices.has(toolCallId)) {
                throw new ChatGPTOAuthProtocolError(
                  `provider response contains duplicate call_id ${JSON.stringify(toolCallId)}`,
                );
              }
              const toolCallIndex = toolCallIndices.size;
              toolCallIndices.set(toolCallId, toolCallIndex);
              const tc = {
                index: toolCallIndex,
                id: toolCallId,
                type: "function",
                function: {
                  name: event.name,
                  arguments: event.arguments,
                },
              };
              const chunk = {
                id: requestId,
                object: "chat.completion.chunk",
                created,
                model: modelId,
                choices: [
                  {
                    index: 0,
                    delta: { tool_calls: [tc] },
                    finish_reason: null,
                  },
                ],
              };
              await writeWithBackpressure(res, `data: ${JSON.stringify(chunk)}\n\n`, downstream.controller.signal);
            } else if (typ === "finish") {
              if (typeof event.response_id !== "string" || event.response_id.length === 0) {
                throw new ChatGPTOAuthProtocolError("provider finish event requires response_id");
              }
              const finishReason = normalizeFinishReason(event.finish_reason);
              if (finishReason === null) {
                throw new ChatGPTOAuthProtocolError(
                  "provider finish event requires a final finish_reason",
                );
              }
              let usage: Usage | null = null;
              if (event.usage != null) {
                usage = usageFromResponse(event.usage);
                if (usage == null) {
                  throw new ChatGPTOAuthProtocolError("provider finish event usage is malformed");
                }
              }
              upstreamResponseId = event.response_id;
              parsedUsage = usage;
              sawFinish = true;
              const chunk = {
                id: requestId,
                object: "chat.completion.chunk",
                created,
                model: modelId,
                choices: [
                  {
                    index: 0,
                    delta: {},
                    finish_reason: finishReason,
                  },
                ],
                response_id: upstreamResponseId,
              };
              await writeWithBackpressure(res, `data: ${JSON.stringify(chunk)}\n\n`, downstream.controller.signal);
            } else if (typ === "reasoning_section_break") {
              continue;
            } else if (typ === "web_search_call") {
              throw new ChatGPTOAuthProtocolError(
                "provider web_search_call event cannot be represented by /v1/chat/completions",
              );
            } else {
              throw new ChatGPTOAuthProtocolError(
                `provider emitted unsupported event type ${JSON.stringify(typ)}`,
              );
            }
          }

          if (!sawFinish || upstreamResponseId == null) {
            throw new ChatGPTOAuthProtocolError("provider stream ended before finish");
          }
          if (parsedUsage != null) {
            const finishChunk = {
              id: requestId,
              object: "chat.completion.chunk",
              created,
              model: modelId,
              choices: [],
              response_id: upstreamResponseId,
              usage: formatOpenAIUsage(parsedUsage),
            };
            await writeWithBackpressure(
              res,
              `data: ${JSON.stringify(finishChunk)}\n\n`,
              downstream.controller.signal,
            );
          }

          await writeWithBackpressure(res, "data: [DONE]\n\n", downstream.controller.signal);
          res.end();
          } finally {
            downstream.dispose();
            await responseStream?.return(undefined);
          }
        } else {
          const response = await provider.chat(
            messages,
            chatOpts,
          );
          if (typeof response.response_id !== "string" || response.response_id.length === 0) {
            throw new ChatGPTOAuthProtocolError("provider response_id is missing");
          }
          if (
            Array.isArray(response.raw?.events)
            && response.raw.events.some((event) => isRecord(event) && event.type === "web_search_call")
          ) {
            throw new ChatGPTOAuthProtocolError(
              "provider web_search_call output cannot be represented by /v1/chat/completions",
            );
          }

          const choiceMessage: Record<string, unknown> = {
            role: "assistant",
            content: response.content,
            refusal: null,
          };
          const finishReason = normalizeFinishReason(response.finish_reason);
          if (finishReason === null) {
            throw new ChatGPTOAuthProtocolError(
              "provider response requires a final finish_reason",
            );
          }
          if (response.tool_calls.length) {
            choiceMessage.tool_calls = response.tool_calls.map(
              (tc) => ({
                id: tc.id,
                type: "function",
                function: {
                  name: tc.name,
                  arguments: tc.arguments,
                },
              }),
            );
          }
          if (response.reasoning_content) {
            choiceMessage.reasoning_content =
              response.reasoning_content;
          }

          const result: Record<string, unknown> = {
            id: `chatcmpl-${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`,
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: modelId,
            choices: [
              {
                index: 0,
                message: choiceMessage,
                finish_reason: finishReason,
                logprobs: null,
              },
            ],
          };
          result.response_id = response.response_id;
          if (response.usage != null) {
            result.usage = formatOpenAIUsage(response.usage);
          }

          res.json(result);
        }
      } catch (err) {
        if (res.headersSent) {
          writeOpenAIStreamError(err, res);
        } else {
          handleError(err, res);
        }
      }
    },
  );

  app.post(
    "/v1/images/generations",
    async (req: Request, res: Response) => {
      try {
        const body = requireRequestBody(req.body);
        assertAllowedTopLevelFields(body, IMAGE_GENERATION_FIELDS, "/v1/images/generations");
        rejectExplicitNullFields(body, ["user"]);
        rejectUnsupportedGenerationFeatures(body);
        rejectNonNullUnsupportedFields(body, ["output_format", "partial_images", "stream"]);
        if (Object.hasOwn(body, "tools")) {
          throw new ChatGPTOAuthInvalidRequestError(
            "tools are not supported by the image generation endpoint",
          );
        }
        if (typeof body.prompt !== "string" || body.prompt.trim().length === 0) {
          throw new ChatGPTOAuthInvalidRequestError("image generation prompt is required");
        }
        const referenceImages = imageReferences(body.reference_images, "reference_images");
        const size = optionalString(body.size, "size");
        if (size != null && size !== "auto") {
          throw new ChatGPTOAuthInvalidRequestError(
            "image size is not supported by the private Codex OAuth transport",
          );
        }
        const responsesLite = optionalBooleanOrString(body.responses_lite, "responses_lite");
        const preparedModel = await provider.prepareModel(optionalModel(body.model) ?? model);
        const requestModel = preparedModel.slug;
        const reasoning = resolveReasoning(
          body.reasoning,
          body.reasoning_effort,
          codexConfig,
          preparedModel.capability,
        );
        const images = await provider.generateImage(body.prompt, {
          model: requestModel,
          size,
          referenceImages,
          reasoningEffort: reasoning?.effort,
          reasoning,
          text: resolveTextOptions(undefined, body.verbosity),
          responsesLite,
          preparedModel,
        });
        const data = images.map((img) => ({
          url: img.result,
          ...(typeof img.revised_prompt === "string" ? { revised_prompt: img.revised_prompt } : {}),
        }));
        res.json({ created: Math.floor(Date.now() / 1000), data });
      } catch (err) {
        handleError(err, res);
      }
    },
  );

  app.post("/v1/inspect", async (req: Request, res: Response) => {
    try {
      const body = requireRequestBody(req.body);
      assertAllowedTopLevelFields(body, INSPECT_FIELDS, "/v1/inspect");
      rejectUnsupportedGenerationFeatures(body);
      if (typeof body.prompt !== "string" || body.prompt.trim().length === 0) {
        throw new ChatGPTOAuthInvalidRequestError("image inspection prompt is required");
      }
      const images = imageReferences(body.images, "images");
      if (images.length === 0) {
        throw new ChatGPTOAuthInvalidRequestError("image inspection requires at least one image");
      }
      const responsesLite = optionalBooleanOrString(body.responses_lite, "responses_lite");
      const preparedModel = await provider.prepareModel(optionalModel(body.model) ?? model);
      const requestModel = preparedModel.slug;
      const reasoning = resolveReasoning(
        body.reasoning,
        body.reasoning_effort,
        codexConfig,
        preparedModel.capability,
      );
      const result = await provider.inspectImages(
        body.prompt,
        {
          model: requestModel,
          images,
          reasoningEffort: reasoning?.effort,
          reasoning,
          text: resolveTextOptions(undefined, body.verbosity),
          responsesLite,
          preparedModel,
        },
      );
      res.json({ content: result });
    } catch (err) {
      handleError(err, res);
    }
  });

  async function compact(req: Request, res: Response): Promise<void> {
    const isAnthropicCompact = req.path === "/v1/messages/compact";
    try {
      const isClaudeCode = resolveClaudeCodeSessionId(req) != null;
      const rawBody = requireRequestBody(req.body);
      if (isAnthropicCompact) rejectExplicitNullAnthropicFields(rawBody);
      const body = isAnthropicCompact
        ? stripAnthropicCacheControls(rawBody, isClaudeCode)
        : rawBody;
      assertAllowedTopLevelFields(
        body,
        isAnthropicCompact ? ANTHROPIC_COMPACT_FIELDS : OPENAI_COMPACT_FIELDS,
        req.path,
      );
      rejectUnsupportedGenerationFeatures(body, { anthropic: isAnthropicCompact });
      rejectUnsupportedCompactFields(body);
      const compactMaxTokens = optionalPositiveInteger(body.max_tokens, "max_tokens");
      if (compactMaxTokens != null && !isClaudeCode) {
        throw new ChatGPTOAuthInvalidRequestError(
          "max_tokens is accepted without forwarding only for Claude Code requests",
        );
      }
      if (body.stop_sequences != null) {
        throw new ChatGPTOAuthInvalidRequestError(
          "stop_sequences is not supported by the private Codex OAuth transport",
        );
      }
      if (isAnthropicCompact) {
        validateAnthropicContextManagement(body.context_management);
      }
      const requestedModel = optionalModel(body.model);
      const responsesLite = optionalBooleanOrString(body.responses_lite, "responses_lite");
      const directServiceTier = optionalString(body.service_tier, "service_tier");
      const { messages, reasoningEffort, tools, text: outputFormatText } = messagesFromCompactBody(
        body,
        requestedModel ?? model,
        isAnthropicCompact,
      );
      const preparedModel = isAnthropicCompact
        ? await prepareAnthropicRouteModel(requestedModel)
        : await provider.prepareModel(requestedModel ?? model);
      const requestModel = preparedModel.slug;
      const requestedReasoningEffort = mergeAnthropicReasoningEffort(
        body.reasoning_effort,
        reasoningEffort,
      );
      const checkpoint = await provider.compactMessages(messages, {
        model: requestModel,
        reasoningEffort: resolveReasoningEffort(
          compactReasoningEffort(
            body.reasoning,
            requestedReasoningEffort,
          ),
          codexConfig,
          preparedModel.capability,
        ),
        responsesLite,
        tools: tools ?? undefined,
        promptCacheKey: resolvePromptCacheKey(body.prompt_cache_key),
        previousResponseId: resolvePreviousResponseId(body.previous_response_id),
        serviceTier: isAnthropicCompact
          ? resolveAnthropicServiceTier(body)
          : directServiceTier,
        text: mergeAnthropicTextOptions(
          resolveTextOptions(body.text, body.verbosity),
          outputFormatText,
        ),
        preparedModel,
      });
      res.json({ checkpoint });
    } catch (err) {
      if (isAnthropicCompact) {
        handleAnthropicError(err, res);
      } else {
        handleError(err, res);
      }
    }
  }

  app.post("/v1/compact", compact);
  app.post("/v1/messages/compact", compact);

  app.post("/v1/messages/count_tokens", async (req: Request, res: Response) => {
    try {
      const isClaudeCode = resolveClaudeCodeSessionId(req) != null;
      const rawBody = requireRequestBody(req.body);
      rejectAnthropicUnsupportedFieldPresence(rawBody, [
        "temperature",
        "top_p",
        "top_k",
        "metadata",
      ]);
      rejectExplicitNullFields(rawBody, ["max_tokens"]);
      rejectExplicitNullAnthropicFields(rawBody);
      const body = stripAnthropicCacheControls(rawBody, isClaudeCode);
      assertAllowedTopLevelFields(body, ANTHROPIC_COUNT_FIELDS, "/v1/messages/count_tokens");
      rejectUnsupportedGenerationFeatures(body, { anthropic: true });
      validateAnthropicContextManagement(body.context_management);
      const requestedModel = optionalModel(body.model);
      const maxTokens = optionalPositiveInteger(body.max_tokens, "max_tokens");
      if (maxTokens != null && !isClaudeCode) {
        throw new ChatGPTOAuthInvalidRequestError(
          "max_tokens is accepted without forwarding only for Claude Code requests",
        );
      }
      if (body.stop_sequences != null) {
        throw new ChatGPTOAuthInvalidRequestError(
          "stop_sequences is not supported by the private Codex OAuth transport",
        );
      }
      const { messages, tools } = anthropicRequestToInternal({
        model: requestedModel,
        messages: requiredMessageRecords(body),
        system: anthropicSystem(body.system),
        maxTokens,
        tools: optionalRecordArray(body.tools, "tools"),
        toolChoice: optionalRecord(body.tool_choice, "tool_choice"),
        stopSequences: optionalStringArray(body.stop_sequences, "stop_sequences"),
        thinking: optionalRecord(body.thinking, "thinking"),
        outputFormat: anthropicOutputFormatFromBody(body),
        outputConfig: body.output_config,
      });
      const preparedModel = await prepareAnthropicRouteModel(requestedModel);
      const inputTokens = estimateInputTokens(messages, tools);
      const contextWindow = getContextWindow(preparedModel.capability, codexConfig);
      res.json({
        input_tokens: inputTokens,
        context_window: contextWindow ?? null,
        auto_compact_token_limit: contextWindow == null
          ? null
          : (getAutoCompactTokenLimit(preparedModel.capability, codexConfig) ?? null),
      });
    } catch (err) {
      handleAnthropicError(err, res);
    }
  });

  app.post("/v1/messages", async (req: Request, res: Response) => {
    try {
      const claudeCodeSessionId = resolveClaudeCodeSessionId(req);
      const rawBody = requireRequestBody(req.body);
      rejectAnthropicUnsupportedFieldPresence(rawBody, [
        "temperature",
        "top_p",
        "top_k",
        "metadata",
      ]);
      rejectExplicitNullAnthropicFields(rawBody);
      const body = stripAnthropicCacheControls(
        rawBody,
        claudeCodeSessionId != null,
      );
      assertAllowedTopLevelFields(body, ANTHROPIC_MESSAGE_FIELDS, "/v1/messages");
      rejectUnsupportedGenerationFeatures(body, { anthropic: true });
      validateAnthropicContextManagement(body.context_management);
      if (body.previous_response_id != null) {
        throw new ChatGPTOAuthInvalidRequestError(
          "previous_response_id is not supported by /v1/messages; send the full messages history",
        );
      }
      const clientModel = optionalModel(body.model);
      const maxTokens = requiredPositiveInteger(body.max_tokens, "max_tokens");
      const responsesLite = optionalBooleanOrString(body.responses_lite, "responses_lite");
      const stream = optionalBoolean(body.stream, "stream") ?? false;
      const requestId = `msg_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;
      const explicitPromptCacheKey = resolvePromptCacheKey(
        body.prompt_cache_key,
      );
      const promptCacheKey = explicitPromptCacheKey
        ?? (
          claudeCodeSessionId == null
            ? undefined
            : claudeSessionPromptCacheKey(claudeCodeSessionId)
        );

      const subagent = mergeBodyAndHeaderString(
        optionalString(body.subagent, "subagent"),
        optionalHeader(req.headers["x-openai-subagent"], "x-openai-subagent"),
        "subagent",
        "x-openai-subagent",
      );
      if (subagent != null) requireSubagentHeaderValue(subagent);
      const memgenRequest = mergeBodyAndHeaderBoolean(
        optionalBoolean(body.memgen_request, "memgen_request"),
        optionalBooleanHeader(
          req.headers["x-openai-memgen-request"],
          "x-openai-memgen-request",
        ),
        "memgen_request",
        "x-openai-memgen-request",
      );

      const {
        messages,
        tools,
        toolChoice,
        parallelToolCalls,
        stop,
        reasoningEffort,
        text,
      } =
        anthropicRequestToInternal({
          model: clientModel,
          messages: requiredMessageRecords(body),
          system: anthropicSystem(body.system),
          maxTokens,
          tools: optionalRecordArray(body.tools, "tools"),
          toolChoice: optionalRecord(body.tool_choice, "tool_choice"),
          stopSequences: optionalStringArray(body.stop_sequences, "stop_sequences"),
          thinking: optionalRecord(body.thinking, "thinking"),
          outputFormat: anthropicOutputFormatFromBody(body),
          outputConfig: body.output_config,
        });

      if (maxTokens != null && claudeCodeSessionId == null) {
        throw new ChatGPTOAuthInvalidRequestError(
          "max_tokens is accepted without forwarding only for Claude Code requests",
        );
      }
      if (stop != null) {
        throw new ChatGPTOAuthInvalidRequestError(
          "stop_sequences is not supported by the private Codex OAuth transport",
        );
      }

      const preparedModel = await prepareAnthropicRouteModel(clientModel);
      const requestModel = preparedModel.slug;
      const requestedReasoningEffort = mergeAnthropicReasoningEffort(
        body.reasoning_effort,
        reasoningEffort,
      );
      const resolvedReasoning = resolveReasoning(
        body.reasoning,
        requestedReasoningEffort,
        codexConfig,
        preparedModel.capability,
      );
      const chatOpts = {
        model: requestModel,
        tools: tools ?? undefined,
        toolChoice: toolChoice ?? undefined,
        reasoningEffort: resolvedReasoning?.effort,
        reasoning: resolvedReasoning,
        maxTokens,
        stop: stop ?? undefined,
        text: resolveTextOptions(text, body.verbosity),
        promptCacheKey,
        subagent,
        memgenRequest,
        codexMetadata: false,
        responsesLite,
        parallelToolCalls,
        serviceTier: resolveAnthropicServiceTier(body),
        ignoreMaxTokens: claudeCodeSessionId != null,
        preparedModel,
      };

      if (stream) {
        // Keep deterministic request-shape failures as normal Anthropic JSON
        // errors instead of committing an SSE 200 response first.
        const downstream = downstreamAbortController(res);
        let responseStream: AsyncGenerator<StreamEvent> | undefined;
        try {
          responseStream = await provider.createChatStream(messages, {
            ...chatOpts,
            signal: downstream.controller.signal,
          });
          res.setHeader("Content-Type", "text/event-stream");
          res.setHeader("Cache-Control", "no-cache");
          res.setHeader("Connection", "keep-alive");

          for await (const chunk of anthropicStreamAdapter(
            responseStream,
            clientModel!,
            requestId,
          )) {
            await writeWithBackpressure(res, chunk, downstream.controller.signal);
          }
          res.end();
        } finally {
          downstream.dispose();
          await responseStream?.return(undefined);
        }
      } else {
        const response = await provider.chat(messages, chatOpts);
        res.json(internalResponseToAnthropic(response, clientModel!, requestId));
      }
    } catch (err) {
      handleAnthropicError(err, res);
    }
  });

  app.use((err: unknown, req: Request, res: Response, _next: unknown) => {
    if (isRecord(err) && (err.type === "entity.too.large" || err.status === 413)) {
      sendRequestTransportError(req, res, 413, "request body exceeds 50 MiB");
      return;
    }
    const normalized = isRecord(err) && err.type === "entity.parse.failed"
      ? new ChatGPTOAuthInvalidRequestError("request body must be valid JSON")
      : err;
    if (["/v1/messages", "/v1/messages/count_tokens", "/v1/messages/compact"].includes(req.path)) {
      handleAnthropicError(normalized, res);
      return;
    }
    if (isRecord(err) && err.type === "entity.parse.failed") {
      handleError(normalized, res);
      return;
    }
    handleError(normalized, res);
  });

  return app;
}

function compactReasoningEffort(
  requestedReasoning: unknown,
  requestedEffort: unknown,
): unknown {
  if (requestedReasoning == null) return requestedEffort;
  if (typeof requestedReasoning !== "object" || Array.isArray(requestedReasoning)) {
    throw new ChatGPTOAuthInvalidRequestError("reasoning must be an object");
  }
  const raw = requestedReasoning as Record<string, unknown>;
  for (const key of Object.keys(raw)) {
    if (!["effort", "mode", "context"].includes(key)) {
      throw new ChatGPTOAuthInvalidRequestError(`reasoning.${key} is not supported`);
    }
  }
  if (raw.mode != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning.mode is not supported by compact",
    );
  }
  if (raw.context != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning.context is not supported by compact",
    );
  }
  if (
    requestedEffort != null
    && raw.effort != null
    && requestedEffort !== raw.effort
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning_effort conflicts with reasoning.effort",
    );
  }
  return requestedEffort ?? raw.effort;
}

function resolvePromptCacheKey(value: unknown): string | undefined {
  if (value == null) return undefined;
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new ChatGPTOAuthInvalidRequestError(
      "prompt_cache_key must be a non-empty string when provided",
    );
  }
  return value;
}

function resolvePreviousResponseId(value: unknown): string | undefined {
  if (value == null) return undefined;
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new ChatGPTOAuthInvalidRequestError(
      "previous_response_id must be a non-empty string when provided",
    );
  }
  return value;
}

function resolveClaudeCodeSessionId(
  req: Request,
): string | undefined {
  const values: string[] = [];
  for (let index = 0; index < req.rawHeaders.length; index += 2) {
    if (req.rawHeaders[index]?.toLowerCase() === "x-claude-code-session-id") {
      values.push(req.rawHeaders[index + 1] ?? "");
    }
  }
  if (values.length === 0) return undefined;
  if (values.length !== 1) {
    throw new ChatGPTOAuthInvalidRequestError(
      "x-claude-code-session-id must be provided at most once",
    );
  }
  const value = values[0];
  if (value.trim().length === 0) {
    throw new ChatGPTOAuthInvalidRequestError(
      "x-claude-code-session-id must be a non-empty string when provided",
    );
  }
  return value;
}

function claudeSessionPromptCacheKey(sessionId: string): string {
  return crypto
    .createHash("sha256")
    .update(`codex-as-api:claude-code-session:${sessionId}`, "utf8")
    .digest("hex");
}

function stripAnthropicCacheControls<T extends Record<string, unknown>>(
  body: T,
  allow: boolean,
): T {
  const stripped = stripCacheControlFromRecord(body, "request", allow);
  if (Array.isArray(stripped.system)) {
    stripped.system = stripped.system.map((block, index) =>
      stripAnthropicContentCacheControls(block, `system block ${index}`, allow)
    );
  } else if (isRecord(stripped.system)) {
    stripped.system = stripAnthropicContentCacheControls(
      stripped.system,
      "system",
      allow,
    );
  }
  if (Array.isArray(stripped.messages)) {
    stripped.messages = stripped.messages.map((message, index) => {
      if (!isRecord(message)) return message;
      const cleanMessage = { ...message };
      if (Array.isArray(cleanMessage.content)) {
        cleanMessage.content = cleanMessage.content.map((block, blockIndex) =>
          stripAnthropicContentCacheControls(
            block,
            `message ${index} content block ${blockIndex}`,
            allow,
          )
        );
      }
      return cleanMessage;
    });
  }
  if (Array.isArray(stripped.tools)) {
    stripped.tools = stripped.tools.map((tool, index) =>
      isRecord(tool)
        ? stripCacheControlFromRecord(tool, `tool ${index}`, allow)
        : tool
    );
  }
  return stripped as T;
}

function stripAnthropicContentCacheControls(
  value: unknown,
  location: string,
  allow: boolean,
): unknown {
  if (!isRecord(value)) return value;
  const stripped = stripCacheControlFromRecord(value, location, allow);
  if (Array.isArray(stripped.content)) {
    stripped.content = stripped.content.map((block, index) =>
      stripAnthropicContentCacheControls(
        block,
        `${location} nested content block ${index}`,
        allow,
      )
    );
  }
  return stripped;
}

function stripCacheControlFromRecord(
  value: Record<string, unknown>,
  location: string,
  allow: boolean,
): Record<string, unknown> {
  const stripped = { ...value };
  if (!Object.hasOwn(stripped, "cache_control")) return stripped;
  if (stripped.cache_control === null) {
    delete stripped.cache_control;
    return stripped;
  }
  if (!allow) {
    throw new ChatGPTOAuthInvalidRequestError(
      `${location} cache_control is accepted only for Claude Code requests`,
    );
  }
  validateAnthropicCacheControl(stripped.cache_control, location);
  delete stripped.cache_control;
  return stripped;
}

function validateAnthropicCacheControl(
  value: unknown,
  location: string,
): void {
  // Accepted only as a Claude request-shape hint; Codex receives no TTL or
  // breakpoint metadata.
  if (!isRecord(value)) {
    throw new ChatGPTOAuthInvalidRequestError(
      `${location} cache_control must be an object`,
    );
  }
  const unknownKeys = Object.keys(value).filter(
    (key) => key !== "type" && key !== "ttl",
  );
  if (
    value.type !== "ephemeral"
    || unknownKeys.length > 0
    || (
      Object.hasOwn(value, "ttl")
      && value.ttl !== "5m"
      && value.ttl !== "1h"
    )
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      `${location} cache_control must have type ephemeral and optional ttl 5m or 1h`,
    );
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function requireRequestBody(value: unknown): Record<string, unknown> {
  if (!isRecord(value)) {
    throw new ChatGPTOAuthInvalidRequestError("request body must be a JSON object");
  }
  return value;
}

// --- Helpers ---

function getRawContextWindow(
  capability: ModelCapability,
  config: CodexConfig,
): number | undefined {
  let resolved: number;
  if (config.modelContextWindow != null) {
    const liveMaximum = capability.maxContextWindow;
    if (liveMaximum == null) {
      resolved = config.modelContextWindow;
    } else {
      requirePositiveCatalogLimit(capability, "maximum context window", liveMaximum);
      resolved = Math.min(config.modelContextWindow, liveMaximum);
    }
  } else {
    const liveContextWindow = capability.contextWindow ?? capability.maxContextWindow;
    if (liveContextWindow == null) return undefined;
    requirePositiveCatalogLimit(capability, "context window", liveContextWindow);
    resolved = liveContextWindow;
  }
  return resolved;
}

function getContextWindow(
  capability: ModelCapability,
  config: CodexConfig,
): number | undefined {
  const resolved = getRawContextWindow(capability, config);
  if (resolved == null) return undefined;
  const i64Minimum = -(2n ** 63n);
  const i64Maximum = 2n ** 63n - 1n;
  const rawProduct = BigInt(resolved) * BigInt(capability.effectiveContextWindowPercent);
  const saturatedProduct = rawProduct < i64Minimum
    ? i64Minimum
    : rawProduct > i64Maximum
      ? i64Maximum
      : rawProduct;
  const effective = saturatedProduct / 100n;
  if (effective <= 0n || effective > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new ChatGPTOAuthCatalogUnavailableError(
      "selected model publishes an unusable effective context window",
    );
  }
  return Number(effective);
}

function getAutoCompactTokenLimit(
  capability: ModelCapability,
  config: CodexConfig,
): number | undefined {
  const contextWindow = getRawContextWindow(capability, config);
  const maximum = contextWindow == null
    ? undefined
    : automaticCompactionMaximum(contextWindow);
  if (config.modelAutoCompactTokenLimit != null) {
    return maximum == null
      ? config.modelAutoCompactTokenLimit
      : Math.min(config.modelAutoCompactTokenLimit, maximum);
  }
  if (capability.autoCompactTokenLimit != null) {
    return maximum == null
      ? capability.autoCompactTokenLimit
      : Math.min(capability.autoCompactTokenLimit, maximum);
  }
  return maximum;
}

function automaticCompactionMaximum(contextWindow: number): number {
  const quotient = Math.floor(contextWindow / 10);
  const remainder = contextWindow % 10;
  return quotient * 9 + Math.floor((remainder * 9) / 10);
}

function requirePositiveCatalogLimit(
  capability: ModelCapability,
  field: string,
  value: number,
): void {
  if (value <= 0) {
    throw new ChatGPTOAuthCatalogUnavailableError(
      `selected model publishes a non-positive ${field}`,
    );
  }
}

function resolveReasoningEffort(
  requested: unknown,
  config: CodexConfig,
  capability: ModelCapability,
): string | undefined {
  if (requested != null) {
    validateReasoningEffortValue(requested, "reasoning_effort");
    return wireReasoningEffort(requested, capability);
  }

  const configured = configuredReasoningEffort(config);
  if (configured != null) {
    try {
      return wireReasoningEffort(configured, capability);
    } catch (err) {
      if (err instanceof ChatGPTOAuthInvalidRequestError) {
        throw new Error("configured reasoning effort is not supported by the selected model");
      }
      throw err;
    }
  }

  const catalogDefault = capability.defaultReasoningEffort;
  if (catalogDefault == null) return undefined;
  try {
    return wireReasoningEffort(catalogDefault, capability);
  } catch (err) {
    if (err instanceof ChatGPTOAuthInvalidRequestError) {
      throw new ChatGPTOAuthCatalogUnavailableError(
        "selected model publishes an unsupported default reasoning effort",
      );
    }
    throw err;
  }
}

function configuredReasoningEffort(config: CodexConfig): string | undefined {
  const effort: unknown = config.modelReasoningEffort;
  if (effort == null) return undefined;
  if (
    typeof effort !== "string"
    || effort.length === 0
    || effort !== effort.trim()
  ) {
    throw new Error("configured reasoning effort must be a non-empty string");
  }
  return effort;
}

function resolveReasoning(
  requestedReasoning: unknown,
  requestedEffort: unknown,
  config: CodexConfig,
  capability: ModelCapability,
): ReasoningOptions | undefined {
  if (
    requestedReasoning != null
    && (
      typeof requestedReasoning !== "object"
      || Array.isArray(requestedReasoning)
    )
  ) {
    throw new ChatGPTOAuthInvalidRequestError("reasoning must be an object");
  }
  const raw = (requestedReasoning ?? {}) as Record<string, unknown>;
  for (const key of Object.keys(raw)) {
    if (!["effort", "mode", "context"].includes(key)) {
      throw new ChatGPTOAuthInvalidRequestError(`reasoning.${key} is not supported`);
    }
  }
  const nestedEffort = raw.effort;
  if (requestedEffort != null) {
    validateReasoningEffortValue(requestedEffort, "reasoning_effort");
  }
  if (nestedEffort != null) {
    validateReasoningEffortValue(nestedEffort, "reasoning.effort");
  }
  if (
    requestedEffort != null
    && nestedEffort != null
    && requestedEffort !== nestedEffort
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning_effort conflicts with reasoning.effort",
    );
  }
  const explicitEffort = requestedEffort ?? nestedEffort;
  if (explicitEffort != null) {
    validateReasoningEffortValue(explicitEffort, "reasoning_effort");
  }
  const mode = raw.mode;
  if (mode != null && !["standard", "pro"].includes(String(mode))) {
    throw new ChatGPTOAuthInvalidRequestError("reasoning.mode must be one of: standard, pro");
  }
  const context = raw.context;
  if (
    context != null
    && !["auto", "current_turn", "all_turns"].includes(String(context))
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
  const effort = resolveReasoningEffort(explicitEffort, config, capability);
  if (effort == null && mode == null && context == null) return undefined;
  const reasoning: ReasoningOptions = {};
  if (effort != null) reasoning.effort = effort as string;
  if (mode != null) reasoning.mode = mode as ReasoningOptions["mode"];
  if (context != null) reasoning.context = context as ReasoningOptions["context"];
  return reasoning;
}

function resolveTextOptions(
  textValue: unknown,
  verbosityValue: unknown,
): Record<string, unknown> | undefined {
  if (
    textValue != null
    && (typeof textValue !== "object" || Array.isArray(textValue))
  ) {
    throw new ChatGPTOAuthInvalidRequestError("text must be an object when provided");
  }
  const text = textValue == null
    ? {}
    : { ...(textValue as Record<string, unknown>) };
  assertRecordFields(text, new Set(["verbosity", "format"]), "text");
  if (
    text.verbosity != null
    && (
      typeof text.verbosity !== "string"
      || !["low", "medium", "high"].includes(text.verbosity)
    )
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "text.verbosity must be one of: low, medium, high",
    );
  }
  if (
    verbosityValue != null
    && (
      typeof verbosityValue !== "string"
      || !["low", "medium", "high"].includes(verbosityValue)
    )
  ) {
    throw new ChatGPTOAuthInvalidRequestError("verbosity must be one of: low, medium, high");
  }
  if (
    verbosityValue != null
    && text.verbosity != null
    && text.verbosity !== verbosityValue
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "verbosity conflicts with text.verbosity",
    );
  }
  if (verbosityValue != null) text.verbosity = verbosityValue;
  if (text.format != null) validateOpenAITextFormat(text.format);
  return Object.keys(text).length > 0 ? text : undefined;
}

function validateOpenAITextFormat(value: unknown): void {
  if (!isRecord(value)) {
    throw new ChatGPTOAuthInvalidRequestError("text.format must be an object when provided");
  }
  const type = value.type;
  if (type === "text" || type === "json_object") {
    assertRecordFields(value, new Set(["type"]), "text.format");
    return;
  }
  if (type !== "json_schema") {
    throw new ChatGPTOAuthInvalidRequestError(
      "text.format.type must be one of: text, json_object, json_schema",
    );
  }
  assertRecordFields(
    value,
    new Set(["type", "name", "description", "schema", "strict"]),
    "text.format",
  );
  if (
    typeof value.name !== "string"
    || !/^[A-Za-z0-9_-]{1,64}$/.test(value.name)
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "text.format.name must contain only letters, digits, underscores, or hyphens and be at most 64 characters",
    );
  }
  if (!isRecord(value.schema)) {
    throw new ChatGPTOAuthInvalidRequestError("text.format.schema must be an object");
  }
  if (value.description != null && typeof value.description !== "string") {
    throw new ChatGPTOAuthInvalidRequestError("text.format.description must be a string when provided");
  }
  if (value.strict != null && typeof value.strict !== "boolean") {
    throw new ChatGPTOAuthInvalidRequestError("text.format.strict must be a boolean when provided");
  }
}

function messagesFromCompactBody(
  body: Record<string, unknown>,
  model: string | undefined,
  forceAnthropic = false,
): {
  messages: Message[];
  reasoningEffort: string | null;
  tools: ToolSchema[] | null;
  text: Record<string, unknown> | null;
} {
  if (forceAnthropic) {
    const converted = anthropicRequestToInternal({
      model: optionalModel(body.model) ?? model,
      messages: requiredMessageRecords(body),
      system: anthropicSystem(body.system),
      maxTokens: requiredPositiveInteger(body.max_tokens, "max_tokens"),
      tools: optionalRecordArray(body.tools, "tools"),
      toolChoice: optionalRecord(body.tool_choice, "tool_choice"),
      stopSequences: optionalStringArray(body.stop_sequences, "stop_sequences"),
      thinking: optionalRecord(body.thinking, "thinking"),
      outputFormat: anthropicOutputFormatFromBody(body),
      outputConfig: body.output_config,
    });
    if (converted.parallelToolCalls === true) {
      throw new ChatGPTOAuthInvalidRequestError(
        "tool_choice.disable_parallel_tool_use=false cannot be represented by the compact endpoint",
      );
    }
    if (converted.toolChoice != null && converted.toolChoice !== "auto") {
      throw new ChatGPTOAuthInvalidRequestError(
        "compact supports only Anthropic tool_choice.type=auto",
      );
    }
    return {
      messages: converted.messages,
      reasoningEffort: converted.reasoningEffort,
      tools: converted.tools,
      text: converted.text,
    };
  }

  const rawMessages = requiredMessageRecords(body);
  return {
    messages: requestMessagesToInternal(rawMessages),
    reasoningEffort: null,
    tools: parseTools(body.tools) ?? null,
    text: null,
  };
}

function anthropicOutputFormatFromBody(body: Record<string, unknown>): unknown {
  return body.output_format;
}

function mergeAnthropicTextOptions(
  directText: Record<string, unknown> | undefined,
  outputFormatText: Record<string, unknown> | null,
): Record<string, unknown> | undefined {
  if (outputFormatText === null) return directText;
  const merged = { ...(directText ?? {}) };
  for (const [key, value] of Object.entries(outputFormatText)) {
    if (
      Object.hasOwn(merged, key)
      && !isDeepStrictEqual(merged[key], value)
    ) {
      throw new ChatGPTOAuthInvalidRequestError(
        `text.${key} conflicts with Anthropic output format`,
      );
    }
    merged[key] = value;
  }
  return merged;
}

function optionalModel(value: unknown): string | undefined {
  if (value == null) return undefined;
  validateModelValue(value, "model");
  return value;
}

function validateModelValue(value: unknown, field: string): asserts value is string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be a non-empty string`);
  }
  if (value !== value.trim()) {
    throw new ChatGPTOAuthInvalidRequestError(
      `${field} must not contain surrounding whitespace`,
    );
  }
}

function validateReasoningEffortValue(
  value: unknown,
  field: string,
): asserts value is string {
  if (typeof value !== "string" || value.length === 0) {
    throw new ChatGPTOAuthInvalidRequestError(
      `${field} must be a non-empty string when provided`,
    );
  }
  if (value !== value.trim()) {
    throw new ChatGPTOAuthInvalidRequestError(
      `${field} must not contain surrounding whitespace`,
    );
  }
}

function assertAllowedTopLevelFields(
  body: Record<string, unknown>,
  allowed: ReadonlySet<string>,
  endpoint: string,
): void {
  for (const field of Object.keys(body)) {
    if (!allowed.has(field)) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${endpoint} does not support field ${JSON.stringify(field)}`,
      );
    }
  }
}

function rejectExplicitNullFields(
  body: Record<string, unknown>,
  fields: readonly string[],
): void {
  for (const field of fields) {
    if (Object.hasOwn(body, field) && body[field] === null) {
      throw new ChatGPTOAuthInvalidRequestError(`${field} must not be null`);
    }
  }
}

function rejectNonNullUnsupportedFields(
  body: Record<string, unknown>,
  fields: readonly string[],
): void {
  for (const field of fields) {
    if (body[field] != null) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${field} is not supported by the private Codex OAuth transport`,
      );
    }
  }
}

function assertRecordFields(
  value: Record<string, unknown>,
  allowed: ReadonlySet<string>,
  field: string,
): void {
  const unknown = Object.keys(value).find((key) => !allowed.has(key));
  if (unknown != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      `${field} does not support field ${JSON.stringify(unknown)}`,
    );
  }
}

function optionalString(value: unknown, field: string): string | undefined {
  if (value == null) return undefined;
  if (typeof value !== "string" || value.length === 0) {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be a non-empty string when provided`);
  }
  return value;
}

function optionalBoolean(value: unknown, field: string): boolean | undefined {
  if (value == null) return undefined;
  if (typeof value !== "boolean") {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be a boolean when provided`);
  }
  return value;
}

function optionalBooleanOrString(value: unknown, field: string): boolean | string | undefined {
  if (value === undefined) return undefined;
  if (typeof value !== "boolean" && typeof value !== "string") {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be a boolean or string when provided`);
  }
  return value;
}

function optionalFiniteNumber(value: unknown, field: string): number | undefined {
  if (value == null) return undefined;
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be a finite number when provided`);
  }
  return value;
}

function optionalPositiveInteger(value: unknown, field: string): number | undefined {
  const result = optionalFiniteNumber(value, field);
  if (result != null && (!Number.isSafeInteger(result) || result <= 0)) {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be a positive integer when provided`);
  }
  return result;
}

function requiredPositiveInteger(value: unknown, field: string): number {
  const result = optionalPositiveInteger(value, field);
  if (result == null) {
    throw new ChatGPTOAuthInvalidRequestError(`${field} is required`);
  }
  return result;
}

function rejectExplicitNullAnthropicFields(body: Record<string, unknown>): void {
  for (const field of [
    "system",
    "tools",
    "tool_choice",
    "stop_sequences",
    "thinking",
    "output_config",
    "stream",
    "service_tier",
  ]) {
    if (Object.hasOwn(body, field) && body[field] === null) {
      throw new ChatGPTOAuthInvalidRequestError(`${field} must not be null`);
    }
  }
}

function rejectAnthropicUnsupportedFieldPresence(
  body: Record<string, unknown>,
  fields: readonly string[],
): void {
  for (const field of fields) {
    if (Object.hasOwn(body, field)) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${field} is not supported by the Codex OAuth HTTP transport`,
      );
    }
  }
}

function mergeOptionalNumbers(
  primary: unknown,
  secondary: unknown,
  primaryField: string,
  secondaryField: string,
): number | undefined {
  const first = optionalPositiveInteger(primary, primaryField);
  const second = optionalPositiveInteger(secondary, secondaryField);
  if (first != null && second != null && first !== second) {
    throw new ChatGPTOAuthInvalidRequestError(`${primaryField} conflicts with ${secondaryField}`);
  }
  return first ?? second;
}

function optionalRecord(value: unknown, field: string): Record<string, unknown> | undefined {
  if (value == null) return undefined;
  if (!isRecord(value)) {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be an object when provided`);
  }
  return value;
}

function optionalStringRecord(value: unknown, field: string): Record<string, string> | undefined {
  const record = optionalRecord(value, field);
  if (record == null) return undefined;
  for (const [key, item] of Object.entries(record)) {
    if (typeof item !== "string") {
      throw new ChatGPTOAuthInvalidRequestError(`${field}.${key} must be a string`);
    }
  }
  return record as Record<string, string>;
}

function recordArray(value: unknown, field: string): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be an array`);
  }
  return value.map((item, index) => {
    if (!isRecord(item)) {
      throw new ChatGPTOAuthInvalidRequestError(`${field}[${index}] must be an object`);
    }
    return item;
  });
}

function requiredMessageRecords(body: Record<string, unknown>): Record<string, unknown>[] {
  if (!Object.hasOwn(body, "messages")) {
    throw new ChatGPTOAuthInvalidRequestError("messages is required");
  }
  const messages = recordArray(body.messages, "messages");
  if (messages.length === 0) {
    throw new ChatGPTOAuthInvalidRequestError("messages must not be empty");
  }
  return messages;
}

function optionalRecordArray(value: unknown, field: string): Record<string, unknown>[] | undefined {
  return value == null ? undefined : recordArray(value, field);
}

function optionalStringArray(value: unknown, field: string): string[] | undefined {
  if (value == null) return undefined;
  if (!Array.isArray(value) || value.some((item) => typeof item !== "string")) {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be an array of strings when provided`);
  }
  return value as string[];
}

function optionalToolChoice(value: unknown): string | Record<string, unknown> | undefined {
  if (value == null) return undefined;
  if (typeof value === "string") {
    if (!["auto", "none", "required"].includes(value)) {
      throw new ChatGPTOAuthInvalidRequestError(
        "tool_choice must be one of: auto, none, required, or a named function",
      );
    }
    return value;
  }
  if (!isRecord(value)) {
    throw new ChatGPTOAuthInvalidRequestError("tool_choice must be a string or object when provided");
  }
  assertRecordFields(value, new Set(["type", "function"]), "tool_choice");
  if (value.type !== "function" || !isRecord(value.function)) {
    throw new ChatGPTOAuthInvalidRequestError("tool_choice object must select a function");
  }
  assertRecordFields(value.function, new Set(["name"]), "tool_choice.function");
  if (typeof value.function.name !== "string" || value.function.name.length === 0) {
    throw new ChatGPTOAuthInvalidRequestError(
      "tool_choice.function.name must be a non-empty string",
    );
  }
  return { type: "function", name: value.function.name };
}

function anthropicSystem(value: unknown): string | Record<string, unknown>[] | undefined {
  if (value === undefined) return undefined;
  if (typeof value === "string") return value;
  return recordArray(value, "system");
}

function optionalHeader(value: string | string[] | undefined, field: string): string | undefined {
  if (value == null) return undefined;
  if (typeof value !== "string" || value.length === 0) {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be a single non-empty header value`);
  }
  return value;
}

function optionalBooleanHeader(
  value: string | string[] | undefined,
  field: string,
): boolean | undefined {
  const header = optionalHeader(value, field);
  if (header == null) return undefined;
  if (header === "true") return true;
  if (header === "false") return false;
  throw new ChatGPTOAuthInvalidRequestError(`${field} must be exactly true or false`);
}

function mergeBodyAndHeaderString(
  bodyValue: string | undefined,
  headerValue: string | undefined,
  bodyField: string,
  headerField: string,
): string | undefined {
  if (bodyValue != null && headerValue != null && bodyValue !== headerValue) {
    throw new ChatGPTOAuthInvalidRequestError(`${bodyField} conflicts with ${headerField}`);
  }
  return bodyValue ?? headerValue;
}

function mergeBodyAndHeaderBoolean(
  bodyValue: boolean | undefined,
  headerValue: boolean | undefined,
  bodyField: string,
  headerField: string,
): boolean | undefined {
  if (bodyValue != null && headerValue != null && bodyValue !== headerValue) {
    throw new ChatGPTOAuthInvalidRequestError(`${bodyField} conflicts with ${headerField}`);
  }
  return bodyValue ?? headerValue;
}

function imageReferences(value: unknown, field: string): ImageReference[] {
  if (value == null) return [];
  if (!Array.isArray(value)) {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be an array`);
  }
  for (const [index, image] of value.entries()) {
    if (!isRecord(image)) {
      throw new ChatGPTOAuthInvalidRequestError(`${field}[${index}] must be an object`);
    }
    assertRecordFields(
      image,
      new Set(["image_url", "detail", "prompt_cache_breakpoint"]),
      `${field}[${index}]`,
    );
  }
  validateImageContentItems(value as ImageReference[]);
  return value as ImageReference[];
}

function mergeAnthropicReasoningEffort(
  explicitEffort: unknown,
  convertedEffort: string | null,
): unknown {
  if (
    explicitEffort != null
    && convertedEffort != null
    && explicitEffort !== convertedEffort
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning_effort conflicts with Anthropic thinking or output_config",
    );
  }
  return explicitEffort ?? convertedEffort ?? undefined;
}

function validateAnthropicContextManagement(value: unknown): void {
  if (value == null) return;
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "context_management supports only clear_thinking_20251015 with keep set to all",
    );
  }

  const contextManagement = value as Record<string, unknown>;
  const edits = contextManagement.edits;
  const exactOuterShape = Object.keys(contextManagement).length === 1
    && Array.isArray(edits)
    && edits.length === 1;
  const edit = exactOuterShape ? edits[0] : null;
  if (
    typeof edit === "object"
    && edit !== null
    && !Array.isArray(edit)
  ) {
    const record = edit as Record<string, unknown>;
    if (
      Object.keys(record).length === 2
      && record.type === "clear_thinking_20251015"
      && record.keep === "all"
    ) {
      return;
    }
  }
  throw new ChatGPTOAuthInvalidRequestError(
    "context_management supports only clear_thinking_20251015 with keep set to all",
  );
}

function resolveAnthropicServiceTier(
  body: Record<string, unknown>,
): string | undefined {
  const serviceTier = body.service_tier;
  if (
    serviceTier != null
    && (
      typeof serviceTier !== "string"
      || serviceTier.trim().length === 0
    )
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "service_tier must be a non-empty string when provided",
    );
  }

  const speed = body.speed;
  if (speed == null) return serviceTier as string | undefined;
  if (speed !== "fast" && speed !== "standard") {
    throw new ChatGPTOAuthInvalidRequestError(
      "speed must be one of: fast, standard",
    );
  }
  const speedTier = speed === "fast" ? "fast" : "default";
  const equivalentTiers = speed === "fast"
    ? new Set(["fast", "priority"])
    : new Set(["default"]);
  if (serviceTier != null && !equivalentTiers.has(serviceTier)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "speed conflicts with service_tier",
    );
  }
  return speedTier;
}


const BASE_PROMPT_TOKENS = 8;
const MESSAGE_BOUNDARY_TOKENS = 3;
const IMAGE_TOKEN_ESTIMATE = 8_500;

function jsonTokenCount(value: unknown): number {
  const serialized = JSON.stringify(value);
  if (serialized === undefined) {
    throw new ChatGPTOAuthInvalidRequestError(
      "structured token-count input must be JSON serializable",
    );
  }
  return countO200kOrdinaryTokens(serialized);
}

function formatOpenAIUsage(usage: Usage): Record<string, unknown> {
  for (const [field, value] of [
    ["prompt_tokens", usage.prompt_tokens],
    ["completion_tokens", usage.completion_tokens],
    ["total_tokens", usage.total_tokens],
  ] as const) {
    if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
      throw new ChatGPTOAuthProtocolError(
        `provider usage ${field} must be a non-negative integer`,
      );
    }
  }
  if (usage.total_tokens !== usage.prompt_tokens + usage.completion_tokens) {
    throw new ChatGPTOAuthProtocolError(
      "provider usage total_tokens must equal prompt_tokens plus completion_tokens",
    );
  }
  const details: Record<string, number> = {};
  for (const [field, value] of [
    ["cached_tokens", usage.cached_tokens],
    ["cache_write_tokens", usage.cache_write_tokens],
  ] as const) {
    if (value == null) continue;
    if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
      throw new ChatGPTOAuthProtocolError(
        `provider usage ${field} must be a non-negative integer when provided`,
      );
    }
    details[field] = value;
  }
  return {
    prompt_tokens: usage.prompt_tokens,
    completion_tokens: usage.completion_tokens,
    total_tokens: usage.total_tokens,
    ...(Object.keys(details).length === 0
      ? {}
      : { prompt_tokens_details: details }),
  };
}

function estimateInputTokens(
  messages: Message[],
  tools: ToolSchema[] | null = null,
): number {
  let inputTokens = BASE_PROMPT_TOKENS;
  for (const message of messages) {
    inputTokens += MESSAGE_BOUNDARY_TOKENS
      + countO200kOrdinaryTokens(message.role)
      + countO200kOrdinaryTokens(message.content);
    inputTokens += (message.images?.length || 0) * IMAGE_TOKEN_ESTIMATE;
    inputTokens += (
      message.structured_content?.filter((part) => part.type === "image_url").length || 0
    ) * IMAGE_TOKEN_ESTIMATE;
    if (message.tool_calls?.length) {
      inputTokens += jsonTokenCount(message.tool_calls);
    }
    if (message.tool_call_id) {
      inputTokens += countO200kOrdinaryTokens(message.tool_call_id);
    }
    if (message.name) inputTokens += countO200kOrdinaryTokens(message.name);
    if (message.reasoning_content) {
      inputTokens += countO200kOrdinaryTokens(message.reasoning_content);
    }
  }
  if (tools?.length) inputTokens += jsonTokenCount(tools);
  return Math.max(1, inputTokens);
}


function requestMessagesToInternal(
  rawMessages: unknown,
): Message[] {
  if (!Array.isArray(rawMessages)) {
    throw new ChatGPTOAuthInvalidRequestError("messages must be an array");
  }
  const result: Message[] = [];
  for (const [messageIndex, rawMessage] of rawMessages.entries()) {
    if (!isRecord(rawMessage)) {
      throw new ChatGPTOAuthInvalidRequestError(`message ${messageIndex} must be an object`);
    }
    const msg = rawMessage;
    const role = mapRole(msg.role, messageIndex);
    const allowedFields = role === MessageRole.ASSISTANT
      ? new Set(["role", "content", "tool_calls", "audio", "function_call", "refusal"])
      : role === MessageRole.TOOL
        ? new Set(["role", "content", "tool_call_id"])
        : new Set(["role", "content"]);
    assertRecordFields(
      msg,
      allowedFields,
      `message ${messageIndex}`,
    );
    if (role === MessageRole.ASSISTANT) {
      if (Object.hasOwn(msg, "tool_calls") && msg.tool_calls === null) {
        throw new ChatGPTOAuthInvalidRequestError(
          `message ${messageIndex} tool_calls must not be null`,
        );
      }
      for (const field of ["audio", "function_call", "refusal"] as const) {
        if (msg[field] != null) {
          throw new ChatGPTOAuthInvalidRequestError(
            `message ${messageIndex} ${field} is not supported by the private Codex OAuth transport`,
          );
        }
      }
    }
    let { content, structuredContent } = normalizeMessageContent(
      msg.content,
      role,
      messageIndex,
    );
    const toolCalls = msg.tool_calls != null
      ? parseToolCalls(msg.tool_calls, messageIndex)
      : undefined;
    if (toolCalls != null && role !== MessageRole.ASSISTANT) {
      throw new ChatGPTOAuthInvalidRequestError(
        `message ${messageIndex} tool_calls is supported only for assistant messages`,
      );
    }
    const contentPresent = Object.hasOwn(msg, "content");
    const toolCallsPresent = Object.hasOwn(msg, "tool_calls");
    if (
      role === MessageRole.ASSISTANT
      && msg.content == null
      && (contentPresent || toolCallsPresent)
    ) {
      structuredContent = [];
    } else if (
      msg.content == null
    ) {
      throw new ChatGPTOAuthInvalidRequestError(
        `message ${messageIndex} content is required unless assistant tool_calls is specified`,
      );
    }
    if (msg.tool_call_id != null && typeof msg.tool_call_id !== "string") {
      throw new ChatGPTOAuthInvalidRequestError(`message ${messageIndex} tool_call_id must be a string`);
    }
    if (role === MessageRole.TOOL && typeof msg.tool_call_id !== "string") {
      throw new ChatGPTOAuthInvalidRequestError(`tool message ${messageIndex} requires tool_call_id`);
    }
    if (role !== MessageRole.TOOL && msg.tool_call_id != null) {
      throw new ChatGPTOAuthInvalidRequestError(
        `message ${messageIndex} tool_call_id is supported only for tool messages`,
      );
    }
    result.push({
      role,
      content,
      tool_calls: toolCalls,
      tool_call_id:
        typeof msg.tool_call_id === "string"
          ? msg.tool_call_id
          : undefined,
      structured_content: structuredContent,
    });
  }
  return result;
}

function mapRole(role: unknown, messageIndex: number): MessageRole {
  const mapping: Record<string, MessageRole> = {
    system: MessageRole.SYSTEM,
    developer: MessageRole.DEVELOPER,
    user: MessageRole.USER,
    assistant: MessageRole.ASSISTANT,
    tool: MessageRole.TOOL,
  };
  if (typeof role !== "string" || mapping[role] == null) {
    throw new ChatGPTOAuthInvalidRequestError(
      `message ${messageIndex} role must be one of: system, developer, user, assistant, tool`,
    );
  }
  return mapping[role];
}

function normalizeMessageContent(
  content: unknown,
  role: MessageRole,
  messageIndex: number,
): { content: string; structuredContent?: MessageContentPart[] } {
  if (content == null) return { content: "" };
  if (typeof content === "string") return { content };
  if (!Array.isArray(content)) {
    throw new ChatGPTOAuthInvalidRequestError(
      `message ${messageIndex} content must be a string or array`,
    );
  }
  const textParts: string[] = [];
  const structuredContent: MessageContentPart[] = [];
  for (const [contentIndex, rawPart] of content.entries()) {
    if (typeof rawPart !== "object" || rawPart === null || Array.isArray(rawPart)) {
      throw new ChatGPTOAuthInvalidRequestError(
        `message ${messageIndex} content block ${contentIndex} must be an object`,
      );
    }
    const part = rawPart as Record<string, unknown>;
    const breakpoint = normalizePromptCacheBreakpoint(
      part.prompt_cache_breakpoint,
      `message ${messageIndex} content block ${contentIndex}`,
    );
    if (role !== MessageRole.USER && breakpoint != null) {
      throw new ChatGPTOAuthInvalidRequestError(
        "prompt_cache_breakpoint is supported only on user message content",
      );
    }
    if (["text", "input_text", "output_text"].includes(String(part.type))) {
      if (
        (part.type === "output_text" && role !== MessageRole.ASSISTANT)
        || (part.type === "input_text" && role === MessageRole.ASSISTANT)
      ) {
        throw new ChatGPTOAuthInvalidRequestError(
          `message ${messageIndex} content block ${contentIndex} type ${JSON.stringify(part.type)} is not valid for role ${role}`,
        );
      }
      assertRecordFields(
        part,
        new Set(["type", "text", "prompt_cache_breakpoint"]),
        `message ${messageIndex} content block ${contentIndex}`,
      );
      if (typeof part.text !== "string") {
        throw new ChatGPTOAuthInvalidRequestError(
          `message ${messageIndex} content block ${contentIndex} text must be a string`,
        );
      }
      textParts.push(part.text);
      structuredContent.push({
        type: "text",
        text: part.text,
        ...(breakpoint == null ? {} : { prompt_cache_breakpoint: breakpoint }),
      });
      continue;
    }
    if (["image_url", "input_image"].includes(String(part.type))) {
      assertRecordFields(
        part,
        new Set(["type", "image_url", "detail", "prompt_cache_breakpoint"]),
        `message ${messageIndex} content block ${contentIndex}`,
      );
      if (role !== MessageRole.USER) {
        throw new ChatGPTOAuthInvalidRequestError(
          `message ${messageIndex} has unsupported content block image_url for role ${role}`,
        );
      }
      const rawImage = part.image_url;
      const imageUrl = typeof rawImage === "string"
        ? rawImage
        : typeof rawImage === "object" && rawImage !== null && !Array.isArray(rawImage)
          ? (rawImage as Record<string, unknown>).url
          : undefined;
      if (typeof imageUrl !== "string" || imageUrl.trim().length === 0) {
        throw new ChatGPTOAuthInvalidRequestError(
          `message ${messageIndex} content block ${contentIndex} image_url requires url`,
        );
      }
      const imageObject = typeof rawImage === "object" && rawImage !== null
        ? rawImage as Record<string, unknown>
        : {};
      if (isRecord(rawImage)) {
        assertRecordFields(
          rawImage,
          new Set(["url", "detail"]),
          `message ${messageIndex} content block ${contentIndex} image_url`,
        );
        if (Object.hasOwn(rawImage, "detail") && rawImage.detail === null) {
          throw new ChatGPTOAuthInvalidRequestError(
            `message ${messageIndex} content block ${contentIndex} image_url.detail must not be null`,
          );
        }
      }
      if (
        Object.hasOwn(imageObject, "detail")
        && Object.hasOwn(part, "detail")
        && imageObject.detail !== part.detail
      ) {
        throw new ChatGPTOAuthInvalidRequestError(
          `message ${messageIndex} content block ${contentIndex} has conflicting image detail values`,
        );
      }
      const detail = imageObject.detail ?? part.detail;
      if (
        detail != null
        && (
          typeof detail !== "string"
          || !["auto", "low", "high", "original"].includes(detail)
        )
      ) {
        throw new ChatGPTOAuthInvalidRequestError(
          `message ${messageIndex} content block ${contentIndex} image detail must be one of: auto, low, high, original`,
        );
      }
      structuredContent.push({
        type: "image_url",
        image_url: imageUrl,
        ...(detail == null ? {} : { detail: detail as "auto" | "low" | "high" | "original" }),
        ...(breakpoint == null ? {} : { prompt_cache_breakpoint: breakpoint }),
      });
      continue;
    }
    if (part.type === "input_audio") {
      assertRecordFields(
        part,
        new Set(["type", "input_audio", "prompt_cache_breakpoint"]),
        `message ${messageIndex} content block ${contentIndex}`,
      );
      if (role !== MessageRole.USER) {
        throw new ChatGPTOAuthInvalidRequestError(
          `message ${messageIndex} has unsupported content block input_audio for role ${role}`,
        );
      }
      if (!isRecord(part.input_audio)) {
        throw new ChatGPTOAuthInvalidRequestError(
          `message ${messageIndex} content block ${contentIndex} input_audio must be an object`,
        );
      }
      assertRecordFields(
        part.input_audio,
        new Set(["data", "format"]),
        `message ${messageIndex} content block ${contentIndex} input_audio`,
      );
      const data = part.input_audio.data;
      const format = part.input_audio.format;
      if (typeof data !== "string") {
        throw new ChatGPTOAuthInvalidRequestError(
          `message ${messageIndex} content block ${contentIndex} input_audio.data must be a string`,
        );
      }
      if (format !== "wav" && format !== "mp3") {
        throw new ChatGPTOAuthInvalidRequestError(
          `message ${messageIndex} content block ${contentIndex} input_audio.format must be wav or mp3`,
        );
      }
      structuredContent.push({
        type: "input_audio",
        audio_url: `data:audio/${format};base64,${data}`,
        ...(breakpoint == null ? {} : { prompt_cache_breakpoint: breakpoint }),
      });
      continue;
    }
    throw new ChatGPTOAuthInvalidRequestError(
      `message ${messageIndex} has unsupported content block ${String(part.type ?? "unknown")}`,
    );
  }
  return {
    content: textParts.join(""),
    structuredContent,
  };
}

function normalizePromptCacheBreakpoint(
  value: unknown,
  source: string,
): { mode: "explicit" } | undefined {
  if (value == null) return undefined;
  if (
    typeof value !== "object"
    || Array.isArray(value)
    || (value as Record<string, unknown>).mode !== "explicit"
    || Object.keys(value as Record<string, unknown>).some((key) => key !== "mode")
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      `${source} prompt_cache_breakpoint must have mode explicit`,
    );
  }
  return { mode: "explicit" };
}

function parseToolCalls(
  raw: unknown,
  messageIndex: number,
): ToolCall[] {
  if (!Array.isArray(raw)) {
    throw new ChatGPTOAuthInvalidRequestError(`message ${messageIndex} tool_calls must be an array`);
  }
  const calls: ToolCall[] = [];
  for (const [callIndex, item] of raw.entries()) {
    if (!isRecord(item)) {
      throw new ChatGPTOAuthInvalidRequestError(`message ${messageIndex} tool_call ${callIndex} must be an object`);
    }
    assertRecordFields(
      item,
      new Set(["id", "type", "function"]),
      `message ${messageIndex} tool_call ${callIndex}`,
    );
    if (item.type !== "function") {
      throw new ChatGPTOAuthInvalidRequestError(
        `message ${messageIndex} tool_call ${callIndex} type must be function`,
      );
    }
    const callId = item.id;
    if (typeof callId !== "string") {
      throw new ChatGPTOAuthInvalidRequestError(`message ${messageIndex} tool_call ${callIndex} requires a string id`);
    }
    if (!isRecord(item.function)) {
      throw new ChatGPTOAuthInvalidRequestError(`message ${messageIndex} tool_call ${callIndex} function must be an object`);
    }
    const func = item.function;
    assertRecordFields(
      func,
      new Set(["name", "arguments"]),
      `message ${messageIndex} tool_call ${callIndex} function`,
    );
    const name = func.name;
    if (typeof name !== "string") {
      throw new ChatGPTOAuthInvalidRequestError(`message ${messageIndex} tool_call ${callIndex} requires a function name`);
    }
    const rawArgs = func.arguments;
    if (typeof rawArgs !== "string") {
      throw new ChatGPTOAuthInvalidRequestError(
        `message ${messageIndex} tool_call ${callIndex} arguments must be a string`,
      );
    }
    calls.push({ id: callId, name, arguments: rawArgs });
  }
  return calls;
}

function parseTools(raw: unknown): ToolSchema[] | undefined {
  if (raw == null) return undefined;
  if (!Array.isArray(raw)) throw new ChatGPTOAuthInvalidRequestError("tools must be an array");
  if (!raw.length) return undefined;
  const schemas: ToolSchema[] = [];
  for (const [index, item] of raw.entries()) {
    if (!isRecord(item)) throw new ChatGPTOAuthInvalidRequestError(`tool ${index} must be an object`);
    assertRecordFields(
      item,
      new Set([
        "type",
        "function",
        "allowed_callers",
        "output_schema",
        "defer_loading",
        "eager_input_streaming",
      ]),
      `tool ${index}`,
    );
    if (item.type !== "function") {
      throw new ChatGPTOAuthInvalidRequestError(`tool ${index} type must be function`);
    }
    if (!isRecord(item.function)) throw new ChatGPTOAuthInvalidRequestError(`tool ${index} function must be an object`);
    const func = item.function;
    assertRecordFields(
      func,
      new Set([
        "name",
        "description",
        "parameters",
        "strict",
        "allowed_callers",
        "output_schema",
        "defer_loading",
        "eager_input_streaming",
      ]),
      `tool ${index} function`,
    );
    for (const field of ["allowed_callers", "output_schema", "defer_loading", "eager_input_streaming"] as const) {
      if (Object.hasOwn(item, field) || Object.hasOwn(func, field)) {
        throw new ChatGPTOAuthInvalidRequestError(`tool ${index} ${field} is not supported`);
      }
    }
    const name = func.name;
    if (typeof name !== "string" || name.length === 0) {
      throw new ChatGPTOAuthInvalidRequestError(`tool ${index} name must be a non-empty string`);
    }
    if (Object.hasOwn(func, "description") && func.description === null) {
      throw new ChatGPTOAuthInvalidRequestError(`tool ${index} description must not be null`);
    }
    if (func.description != null && typeof func.description !== "string") {
      throw new ChatGPTOAuthInvalidRequestError(`tool ${index} description must be a string`);
    }
    if (Object.hasOwn(func, "parameters") && func.parameters === null) {
      throw new ChatGPTOAuthInvalidRequestError(`tool ${index} parameters must not be null`);
    }
    const parameters = func.parameters === undefined ? {} : func.parameters;
    if (!isRecord(parameters)) {
      throw new ChatGPTOAuthInvalidRequestError(`tool ${index} parameters must be an object`);
    }
    if (func.strict != null && typeof func.strict !== "boolean") {
      throw new ChatGPTOAuthInvalidRequestError(`tool ${index} strict must be a boolean`);
    }
    schemas.push({
      name,
      parameters,
      ...(typeof func.description === "string"
        ? { description: func.description }
        : {}),
      ...(func.strict == null ? {} : { strict: func.strict }),
    });
  }
  return schemas;
}

function rejectUnsupportedGenerationFeatures(
  body: Record<string, unknown>,
  opts: { anthropic?: boolean } = {},
): void {
  for (const field of UNSUPPORTED_GENERATION_FIELDS) {
    if (
      field === "safety_identifier"
      || field === "prompt_cache_options"
      || field === "multi_agent"
      || field === "programmatic_tool_calling"
    ) {
      continue;
    }
    if (body[field] != null) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${field} is not supported by the private Codex OAuth transport`,
      );
    }
  }
  if (Object.hasOwn(body, "safety_identifier")) {
    throw new ChatGPTOAuthInvalidRequestError(
      "safety_identifier is not supported by the private Codex OAuth HTTP transport",
    );
  }
  if (body.prompt_cache_options != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "prompt_cache_options is not supported by the private Codex OAuth HTTP transport",
    );
  }
  if (
    Object.hasOwn(body, "multi_agent")
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "multi_agent is not supported by this compatibility API",
    );
  }
  if (
    Object.hasOwn(body, "programmatic_tool_calling")
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "programmatic_tool_calling is not supported by this compatibility API",
    );
  }
  if (!Array.isArray(body.tools)) return;
  for (const [index, rawTool] of body.tools.entries()) {
    if (typeof rawTool !== "object" || rawTool === null || Array.isArray(rawTool)) {
      throw new ChatGPTOAuthInvalidRequestError(`tool ${index} must be an object`);
    }
    const tool = rawTool as Record<string, unknown>;
    if (tool.type === "programmatic_tool_calling") {
      throw new ChatGPTOAuthInvalidRequestError(
        "programmatic_tool_calling tools are not supported by this compatibility API",
      );
    }
    const func = typeof tool.function === "object"
      && tool.function !== null
      && !Array.isArray(tool.function)
      ? tool.function as Record<string, unknown>
      : tool;
    if (Object.hasOwn(tool, "allowed_callers") || Object.hasOwn(func, "allowed_callers")) {
      throw new ChatGPTOAuthInvalidRequestError(
        "programmatic tool allowed_callers is not supported",
      );
    }
    if (Object.hasOwn(tool, "output_schema") || Object.hasOwn(func, "output_schema")) {
      throw new ChatGPTOAuthInvalidRequestError(
        "programmatic tool output_schema is not supported",
      );
    }
    for (const field of ["defer_loading", "eager_input_streaming"] as const) {
      for (const owner of [tool, func]) {
        const value = owner[field];
        if (
          Object.hasOwn(owner, field)
          && !(opts.anthropic === true && field === "eager_input_streaming" && value === null)
        ) {
          throw new ChatGPTOAuthInvalidRequestError(
            `programmatic tool ${field} is not supported`,
          );
        }
      }
    }
  }
}

function rejectUnsupportedTransportControls(opts: {
  maxTokens?: number;
  stop?: string[];
  temperature?: number;
}): void {
  if (opts.maxTokens != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "max_tokens and max_completion_tokens are not supported by the private Codex OAuth transport",
    );
  }
  if (opts.stop != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "stop is not supported by the private Codex OAuth transport",
    );
  }
  if (opts.temperature != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "temperature is not supported by the private Codex OAuth transport",
    );
  }
}

function rejectUnsupportedCompactFields(body: Record<string, unknown>): void {
  for (const field of [
    "safety_identifier",
    "include",
    "prompt_cache_retention",
  ]) {
    if (body[field] != null) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${field} is not supported by the compact compatibility endpoint`,
      );
    }
  }
}

function normalizeStop(stop: unknown): string[] | undefined {
  if (stop == null) return undefined;
  if (typeof stop === "string") return [stop];
  if (Array.isArray(stop) && stop.every((value) => typeof value === "string")) {
    return stop as string[];
  }
  throw new ChatGPTOAuthInvalidRequestError("stop must be a string or an array of strings");
}

export function main(): void {
  const app = createApp();
  const host = resolveServerHost();
  const port = resolveServerPort();
  app.listen(port, host, () => {
    console.log(`codex-as-api listening on ${host}:${port}`);
  });
}
