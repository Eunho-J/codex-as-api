import * as crypto from "node:crypto";
import express, { type Request, type Response } from "express";
import {
  ChatGPTOAuthError,
  ChatGPTOAuthInvalidRequestError,
  ChatGPTOAuthMissingError,
  isAuthLocallyAvailable,
} from "./auth.js";
import type {
  Message,
  MessageContentPart,
  ToolCall,
  ToolSchema,
} from "./messages.js";
import { MessageRole } from "./messages.js";
import {
  ChatGPTOAuthProvider,
  primeCodexCliVersionCache,
  usageFromResponse,
} from "./provider.js";
import type { PromptCacheOptions, ReasoningOptions } from "./provider.js";
import {
  anthropicRequestToInternal,
  internalResponseToAnthropic,
  anthropicStreamAdapter,
  formatAnthropicError,
} from "./anthropic-adapter.js";
import { loadCodexConfig, type CodexConfig } from "./codex-config.js";
import { capabilityForModel } from "./model-capabilities.js";

const HOST = process.env.CODEX_AS_API_HOST || "127.0.0.1";
const PORT = parseInt(process.env.CODEX_AS_API_PORT || "18080", 10);
const DEFAULT_MODEL = "gpt-5.5";
const DEFAULT_CONTEXT_WINDOW = 200_000;

function errorStatus(err: unknown): number {
  if (err instanceof ChatGPTOAuthMissingError) return 401;
  if (err instanceof ChatGPTOAuthInvalidRequestError) return 400;
  if (
    isContextWindowError(err)
    || isReasoningEffortValidationError(err)
    || isResponsesLiteValidationError(err)
    || isRequestShapeValidationError(err)
  ) return 400;
  return 500;
}

function errorType(err: unknown): string {
  if (err instanceof ChatGPTOAuthMissingError || err instanceof ChatGPTOAuthError) {
    return "chatgpt_oauth_error";
  }
  return "server_error";
}

function isContextWindowError(err: unknown): boolean {
  return /exceeds the context window|context window/i.test(String(err));
}

function isReasoningEffortValidationError(err: unknown): boolean {
  return /reasoning_effort must be a non-empty string when provided/.test(String(err));
}

function isResponsesLiteValidationError(err: unknown): boolean {
  return /responses_lite must be one of: off, on, auto|Responses Lite requires tool_choice to be the exact string auto|Responses Lite cannot use hosted (?:web_search|image_generation|programmatic_tool_calling) without a standalone executor|Responses Lite reasoning\.context must be all_turns/.test(
    String(err),
  );
}

function isRequestShapeValidationError(err: unknown): boolean {
  return /reasoning(?:\.| must|_effort conflicts)|safety_identifier|previous_response_id|prompt_cache_(?:key|options|breakpoint|retention)|service_tier|include|verbosity|text must|unsupported content block|system message content|message \d+ content|image (?:reference|detail)|multi_agent|programmatic_tool_calling|allowed_callers|output_schema|tools? must/.test(
    String(err),
  );
}

function handleError(err: unknown, res: Response): void {
  const status = errorStatus(err);
  const body = {
    error: { message: String(err), type: errorType(err) },
  };

  if (res.headersSent) {
    if (!res.writableEnded) res.end();
    return;
  }

  res.status(status).json(body);
}

function writeOpenAIStreamError(err: unknown, res: Response): void {
  if (res.writableEnded) return;
  res.write(
    `data: ${JSON.stringify({
      error: { message: String(err), type: errorType(err) },
    })}\n\n`,
  );
  res.write("data: [DONE]\n\n");
  res.end();
}

function handleAnthropicError(err: unknown, res: Response): void {
  const status = errorStatus(err);
  const body = formatAnthropicError(status, String(err));

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

export function createApp(opts?: CreateAppOptions): express.Express {
  const codexConfig = opts?.codexConfig ?? loadCodexConfig();
  const envModel = process.env.CODEX_AS_API_MODEL?.trim() || undefined;
  const model = opts?.model ?? envModel ?? codexConfig.model ?? DEFAULT_MODEL;
  const authPath = opts?.authPath ?? process.env.CODEX_AS_API_AUTH_PATH;
  const provider =
    opts?.provider ??
    new ChatGPTOAuthProvider({
      model,
      authJsonPath: authPath,
    });

  const app = express();
  app.use(express.json({ limit: "50mb" }));

  app.get("/health", (_req: Request, res: Response) => {
    try {
      const reasoningEffort = resolveReasoningEffort(undefined, codexConfig, model);
      res.json({
        status: "ok",
        auth_available: isAuthLocallyAvailable(authPath),
        model,
        codex_config_path: codexConfig.configPath,
        reasoning_effort: reasoningEffort ?? null,
        context_window: getContextWindow(model, codexConfig),
        auto_compact_token_limit: getAutoCompactTokenLimit(model, codexConfig),
      });
    } catch (err) {
      handleError(err, res);
    }
  });

  app.post(
    "/v1/chat/completions",
    async (req: Request, res: Response) => {
      try {
        const body = req.body;
        rejectUnsupportedGenerationFeatures(body);
        const messages = requestMessagesToInternal(
          body.messages || [],
        );
        const tools = parseTools(body.tools);
        const stop = normalizeStop(body.stop);
        const maxTokens =
          body.max_completion_tokens ?? body.max_tokens ?? undefined;
        const requestModel = typeof body.model === "string" && body.model
          ? body.model
          : model;

        const subagent =
          body.subagent ||
          (req.headers["x-openai-subagent"] as string | undefined);
        const memgenHeader = req.headers[
          "x-openai-memgen-request"
        ] as string | undefined;
        let memgenRequest: boolean | undefined =
          body.memgen_request;
        if (memgenRequest == null && memgenHeader != null) {
          memgenRequest = !["false", "0", ""].includes(
            memgenHeader.toLowerCase(),
          );
        }
        const reasoning = resolveReasoning(
          body.reasoning,
          body.reasoning_effort,
          codexConfig,
          requestModel,
        );

        const chatOpts = {
          model: requestModel,
          tools,
          toolChoice: body.tool_choice,
          temperature: body.temperature,
          reasoningEffort: reasoning?.effort,
          reasoning,
          maxTokens,
          stop,
          promptCacheKey: body.prompt_cache_key,
          promptCacheOptions: body.prompt_cache_options,
          safetyIdentifier: body.safety_identifier,
          subagent,
          memgenRequest,
          previousResponseId: body.previous_response_id,
          serviceTier: body.service_tier,
          text: resolveTextOptions(body.text, body.verbosity),
          clientMetadata: body.client_metadata,
          codexMetadata: body.codex_metadata,
          responsesLite: body.responses_lite,
          parallelToolCalls: body.parallel_tool_calls,
        };

        const modelId = `codex-oauth:${requestModel}`;

        if (body.stream) {
          // ChatGPTOAuthProvider builds and validates the deterministic request
          // synchronously, before Express commits streaming response headers.
          const responseStream = provider.chatStream(messages, chatOpts);
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
          res.write(`data: ${JSON.stringify(preamble)}\n\n`);

          let usageDict: Record<string, unknown> | null = null;
          let upstreamResponseId: string | null = null;
          const toolCallIndices = new Map<string, number>();

          for await (const event of responseStream) {
            const typ = event.type;
            if (typ === "content") {
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
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            } else if (typ === "reasoning_delta") {
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
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            } else if (typ === "reasoning_raw_delta") {
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
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            } else if (typ === "tool_call") {
              const toolCallId = String(event.id ?? "");
              let toolCallIndex = toolCallIndices.get(toolCallId);
              if (toolCallIndex == null) {
                toolCallIndex = toolCallIndices.size;
                toolCallIndices.set(toolCallId, toolCallIndex);
              }
              const tc = {
                index: toolCallIndex,
                id: toolCallId,
                type: "function",
                function: {
                  name: event.name,
                  arguments: JSON.stringify(
                    event.arguments || {},
                  ),
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
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            } else if (typ === "finish") {
              upstreamResponseId = typeof event.response_id === "string"
                ? event.response_id
                : null;
              if (
                typeof event.usage === "object" &&
                event.usage !== null
              ) {
                usageDict = event.usage as Record<
                  string,
                  unknown
                >;
              }
              const chunk = {
                id: requestId,
                object: "chat.completion.chunk",
                created,
                model: modelId,
                choices: [
                  {
                    index: 0,
                    delta: {},
                    finish_reason:
                      event.finish_reason || "stop",
                  },
                ],
                response_id: upstreamResponseId,
              };
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            }
          }

          if (usageDict) {
            const parsedUsage = usageFromResponse(usageDict);
            const promptTokens = parsedUsage?.prompt_tokens ?? 0;
            const completionTokens = parsedUsage?.completion_tokens ?? 0;
            const finishChunk = {
              id: requestId,
              object: "chat.completion.chunk",
              created,
              model: modelId,
              choices: [],
              response_id: upstreamResponseId,
              usage: {
                prompt_tokens: promptTokens,
                completion_tokens: completionTokens,
                total_tokens: parsedUsage?.total_tokens ?? promptTokens + completionTokens,
                prompt_tokens_details: {
                  cached_tokens: parsedUsage?.cached_tokens ?? 0,
                  cache_write_tokens: parsedUsage?.cache_write_tokens ?? 0,
                },
              },
            };
            res.write(
              `data: ${JSON.stringify(finishChunk)}\n\n`,
            );
          }

          res.write("data: [DONE]\n\n");
          res.end();
        } else {
          const response = await provider.chat(
            messages,
            chatOpts,
          );

          const choiceMessage: Record<string, unknown> = {
            role: "assistant",
            content: response.content,
          };
          if (response.tool_calls.length) {
            choiceMessage.tool_calls = response.tool_calls.map(
              (tc) => ({
                id: tc.id,
                type: "function",
                function: {
                  name: tc.name,
                  arguments: JSON.stringify(tc.arguments),
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
                finish_reason: response.finish_reason,
              },
            ],
          };
          if (response.response_id != null) {
            result.response_id = response.response_id;
          }

          if (response.usage) {
            result.usage = {
              prompt_tokens: response.usage.prompt_tokens,
              completion_tokens:
                response.usage.completion_tokens,
              total_tokens: response.usage.total_tokens,
              prompt_tokens_details: {
                cached_tokens: response.usage.cached_tokens,
                cache_write_tokens: response.usage.cache_write_tokens ?? 0,
              },
            };
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
        const body = req.body;
        rejectUnsupportedGenerationFeatures(body);
        const requestModel = typeof body.model === "string" && body.model
          ? body.model
          : model;
        const reasoning = resolveReasoning(
          body.reasoning,
          body.reasoning_effort,
          codexConfig,
          requestModel,
        );
        const images = await provider.generateImage(body.prompt, {
          model: requestModel,
          size: body.size,
          referenceImages: body.reference_images,
          reasoningEffort: reasoning?.effort,
          reasoning,
          safetyIdentifier: body.safety_identifier,
          promptCacheOptions: body.prompt_cache_options,
          text: resolveTextOptions(undefined, body.verbosity),
          responsesLite: body.responses_lite,
        });
        const data = images
          .filter((img) => img.result)
          .map((img) => ({
            url: img.result,
            revised_prompt: img.revised_prompt || body.prompt,
          }));
        res.json({ created: Math.floor(Date.now() / 1000), data });
      } catch (err) {
        handleError(err, res);
      }
    },
  );

  app.post("/v1/inspect", async (req: Request, res: Response) => {
    try {
      const body = req.body;
      rejectUnsupportedGenerationFeatures(body);
      const requestModel = model;
      const reasoning = resolveReasoning(
        body.reasoning,
        body.reasoning_effort,
        codexConfig,
        requestModel,
      );
      const result = await provider.inspectImages(
        String(body.prompt || ""),
        {
          model: requestModel,
          images: body.images || [],
          reasoningEffort: reasoning?.effort,
          reasoning,
          safetyIdentifier: body.safety_identifier,
          promptCacheOptions: body.prompt_cache_options,
          text: resolveTextOptions(undefined, body.verbosity),
          responsesLite: body.responses_lite,
        },
      );
      res.json({ content: result });
    } catch (err) {
      handleError(err, res);
    }
  });

  async function compact(req: Request, res: Response): Promise<void> {
    try {
      const body = req.body;
      rejectUnsupportedGenerationFeatures(body);
      rejectUnsupportedCompactFields(body);
      const { messages, reasoningEffort, tools } = messagesFromCompactBody(
        body,
        model,
        req.path === "/v1/messages/compact",
      );
      const checkpoint = await provider.compactMessages(messages, {
        model,
        reasoningEffort: resolveReasoningEffort(
          compactReasoningEffort(
            body.reasoning,
            body.reasoning_effort ?? reasoningEffort ?? undefined,
          ),
          codexConfig,
          model,
        ),
        responsesLite: body.responses_lite,
        tools: tools ?? undefined,
        promptCacheOptions: body.prompt_cache_options as PromptCacheOptions | undefined,
        promptCacheKey: resolvePromptCacheKey(body.prompt_cache_key),
        previousResponseId: resolvePreviousResponseId(body.previous_response_id),
        serviceTier: body.service_tier,
        text: resolveTextOptions(body.text, body.verbosity),
      });
      res.json({ checkpoint });
    } catch (err) {
      handleError(err, res);
    }
  }

  app.post("/v1/compact", compact);
  app.post("/v1/messages/compact", compact);

  app.post("/v1/messages/count_tokens", async (req: Request, res: Response) => {
    try {
      const body = req.body;
      rejectUnsupportedGenerationFeatures(body);
      const { messages } = anthropicRequestToInternal({
        model: body.model,
        messages: body.messages || [],
        system: body.system,
        maxTokens: body.max_tokens,
        tools: body.tools,
        toolChoice: body.tool_choice,
        stopSequences: body.stop_sequences,
        thinking: body.thinking,
        outputFormat: anthropicOutputFormatFromBody(body),
      });
      const inputTokens = estimateInputTokens(messages, body);
      res.json({
        input_tokens: inputTokens,
        context_window: getContextWindow(model, codexConfig),
        auto_compact_token_limit: getAutoCompactTokenLimit(model, codexConfig),
      });
    } catch (err) {
      handleAnthropicError(err, res);
    }
  });

  app.post("/v1/messages", async (req: Request, res: Response) => {
    try {
      const body = req.body;
      rejectUnsupportedGenerationFeatures(body);
      const requestId = `msg_${crypto.randomUUID().replace(/-/g, "").slice(0, 24)}`;

      const subagent =
        body.subagent ||
        (req.headers["x-openai-subagent"] as string | undefined);
      const memgenHeader = req.headers[
        "x-openai-memgen-request"
      ] as string | undefined;
      let memgenRequest: boolean | undefined = body.memgen_request;
      if (memgenRequest == null && memgenHeader != null) {
        memgenRequest = !["false", "0", ""].includes(
          memgenHeader.toLowerCase(),
        );
      }

      const { messages, tools, toolChoice, stop, reasoningEffort, text } =
        anthropicRequestToInternal({
          model: body.model,
          messages: body.messages || [],
          system: body.system,
          maxTokens: body.max_tokens,
          tools: body.tools,
          toolChoice: body.tool_choice,
          stopSequences: body.stop_sequences,
          thinking: body.thinking,
          outputFormat: anthropicOutputFormatFromBody(body),
        });

      const clientModel = body.model || "claude-sonnet-4-5";
      const requestModel = model;
      const resolvedReasoning = resolveReasoning(
        body.reasoning,
        body.reasoning_effort ?? reasoningEffort ?? undefined,
        codexConfig,
        requestModel,
      );
      const chatOpts = {
        model: requestModel,
        tools: tools ?? undefined,
        toolChoice: toolChoice ?? undefined,
        reasoningEffort: resolvedReasoning?.effort,
        reasoning: resolvedReasoning,
        maxTokens: body.max_tokens,
        stop: stop ?? undefined,
        text: resolveTextOptions(text, body.verbosity),
        promptCacheOptions: body.prompt_cache_options,
        safetyIdentifier: body.safety_identifier,
        subagent,
        memgenRequest,
        responsesLite: body.responses_lite,
      };

      if (body.stream) {
        // Keep deterministic request-shape failures as normal Anthropic JSON
        // errors instead of committing an SSE 200 response first.
        const responseStream = provider.chatStream(messages, chatOpts);
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");

        for await (const chunk of anthropicStreamAdapter(
          responseStream,
          clientModel,
          requestId,
        )) {
          res.write(chunk);
        }
        res.end();
      } else {
        const response = await provider.chat(messages, chatOpts);
        res.json(internalResponseToAnthropic(response, clientModel, requestId));
      }
    } catch (err) {
      handleAnthropicError(err, res);
    }
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

// --- Helpers ---

function getContextWindow(model: string, config: CodexConfig): number {
  const capability = capabilityForModel(model);
  if (config.modelContextWindow != null) {
    return capability.maxContextWindow == null
      ? config.modelContextWindow
      : Math.min(config.modelContextWindow, capability.maxContextWindow);
  }
  return capability.contextWindow ?? DEFAULT_CONTEXT_WINDOW;
}

function getAutoCompactTokenLimit(model: string, config: CodexConfig): number {
  if (config.modelAutoCompactTokenLimit != null) {
    return Math.min(
      config.modelAutoCompactTokenLimit,
      Math.floor(getContextWindow(model, config) * 0.9),
    );
  }
  const contextWindow = getContextWindow(model, config);
  if (config.modelContextWindow != null || capabilityForModel(model).contextWindow != null) {
    return Math.floor(contextWindow * 0.9);
  }
  return Math.floor(DEFAULT_CONTEXT_WINDOW * 0.8);
}

function resolveReasoningEffort(
  requested: unknown,
  config: CodexConfig,
  model: string,
): string | undefined {
  const effort = requested
    ?? config.modelReasoningEffort
    ?? capabilityForModel(model).defaultReasoningEffort;
  if (effort == null) return undefined;
  if (typeof effort !== "string" || effort.length === 0) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning_effort must be a non-empty string when provided",
    );
  }
  return effort;
}

function resolveReasoning(
  requestedReasoning: unknown,
  requestedEffort: unknown,
  config: CodexConfig,
  model: string,
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
  if (
    explicitEffort != null
    && (
      typeof explicitEffort !== "string"
      || explicitEffort.length === 0
    )
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning_effort must be a non-empty string when provided",
    );
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
  if (mode != null && !/^gpt-5\.6(?:$|-)/.test(model)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning.mode is supported only by GPT-5.6 models",
    );
  }
  if (mode === "pro") {
    throw new ChatGPTOAuthInvalidRequestError(
      "reasoning.mode pro is not supported by the private Codex OAuth transport",
    );
  }
  const effort = explicitEffort
    ?? config.modelReasoningEffort
    ?? (mode != null ? "medium" : undefined)
    ?? capabilityForModel(model).defaultReasoningEffort;
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
    throw new ChatGPTOAuthError("text must be an object when provided");
  }
  const text = textValue == null
    ? {}
    : { ...(textValue as Record<string, unknown>) };
  if (
    text.verbosity != null
    && (
      typeof text.verbosity !== "string"
      || !["low", "medium", "high"].includes(text.verbosity)
    )
  ) {
    throw new ChatGPTOAuthError(
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
    throw new ChatGPTOAuthError("verbosity must be one of: low, medium, high");
  }
  if (
    verbosityValue != null
    && text.verbosity != null
    && text.verbosity !== verbosityValue
  ) {
    throw new ChatGPTOAuthError(
      "verbosity conflicts with text.verbosity",
    );
  }
  if (verbosityValue != null) text.verbosity = verbosityValue;
  return Object.keys(text).length > 0 ? text : undefined;
}

function messagesFromCompactBody(
  body: Record<string, unknown>,
  model: string,
  forceAnthropic = false,
): {
  messages: Message[];
  reasoningEffort: string | null;
  tools: ToolSchema[] | null;
} {
  if (
    forceAnthropic
    || body.system != null
    || body.thinking != null
    || body.tool_choice != null
    || body.stop_sequences != null
  ) {
    const converted = anthropicRequestToInternal({
      model: String(body.model || model),
      messages: Array.isArray(body.messages) ? body.messages as Record<string, unknown>[] : [],
      system: body.system as string | Record<string, unknown>[] | undefined,
      maxTokens: typeof body.max_tokens === "number" ? body.max_tokens : undefined,
      tools: Array.isArray(body.tools) ? body.tools as Record<string, unknown>[] : undefined,
      toolChoice: typeof body.tool_choice === "object" && body.tool_choice !== null
        ? body.tool_choice as Record<string, unknown>
        : undefined,
      stopSequences: Array.isArray(body.stop_sequences) ? body.stop_sequences.map(String) : undefined,
      thinking: typeof body.thinking === "object" && body.thinking !== null
        ? body.thinking as Record<string, unknown>
        : undefined,
      outputFormat: anthropicOutputFormatFromBody(body),
    });
    return {
      messages: converted.messages,
      reasoningEffort: converted.reasoningEffort,
      tools: converted.tools,
    };
  }

  const rawMessages = Array.isArray(body.messages) ? body.messages as Record<string, unknown>[] : [];
  return {
    messages: requestMessagesToInternal(rawMessages),
    reasoningEffort: null,
    tools: parseTools(body.tools) ?? null,
  };
}

function anthropicOutputFormatFromBody(body: Record<string, unknown>): Record<string, unknown> | undefined {
  if (typeof body.output_format === "object" && body.output_format !== null && !Array.isArray(body.output_format)) {
    return body.output_format as Record<string, unknown>;
  }
  if (typeof body.output_config === "object" && body.output_config !== null && !Array.isArray(body.output_config)) {
    const format = (body.output_config as Record<string, unknown>).format;
    if (typeof format === "object" && format !== null && !Array.isArray(format)) {
      return format as Record<string, unknown>;
    }
  }
  return undefined;
}


function byteLength(value: string): number {
  return Buffer.byteLength(value, "utf8");
}

function jsonByteLength(value: unknown): number {
  return Buffer.byteLength(JSON.stringify(value) ?? "", "utf8");
}

function estimateInputTokens(messages: Message[], rawPayload?: unknown): number {
  let total = 32;
  for (const message of messages) {
    total += 12 + byteLength(message.role) + byteLength(message.content);
    total += (message.images?.length || 0) * 8500;
    total += (message.structured_content?.filter((part) => part.type === "image_url").length || 0) * 8500;
    if (message.tool_calls?.length) total += jsonByteLength(message.tool_calls);
    if (message.reasoning_content) total += byteLength(message.reasoning_content);
  }
  if (rawPayload != null) total += jsonByteLength(rawPayload);
  return Math.max(1, total);
}


function requestMessagesToInternal(
  rawMessages: Record<string, unknown>[],
): Message[] {
  const result: Message[] = [];
  for (const [messageIndex, msg] of rawMessages.entries()) {
    const role = mapRole(String(msg.role || "user"));
    const { content, structuredContent } = normalizeMessageContent(
      msg.content,
      role,
      messageIndex,
    );
    const toolCalls = msg.tool_calls
      ? parseToolCalls(
          msg.tool_calls as Record<string, unknown>[],
        )
      : undefined;
    result.push({
      role,
      content,
      tool_calls: toolCalls,
      tool_call_id:
        typeof msg.tool_call_id === "string"
          ? msg.tool_call_id
          : undefined,
      name:
        typeof msg.name === "string" ? msg.name : undefined,
      structured_content: structuredContent,
    });
  }
  return result;
}

function mapRole(role: string): MessageRole {
  const mapping: Record<string, MessageRole> = {
    system: MessageRole.SYSTEM,
    user: MessageRole.USER,
    assistant: MessageRole.ASSISTANT,
    tool: MessageRole.TOOL,
  };
  return mapping[role.toLowerCase()] ?? MessageRole.USER;
}

function normalizeMessageContent(
  content: unknown,
  role: MessageRole,
  messageIndex: number,
): { content: string; structuredContent?: MessageContentPart[] } {
  if (content == null) return { content: "" };
  if (typeof content === "string") return { content };
  if (!Array.isArray(content)) {
    throw new ChatGPTOAuthError(
      `message ${messageIndex} content must be a string or array`,
    );
  }
  const textParts: string[] = [];
  const structuredContent: MessageContentPart[] = [];
  for (const [contentIndex, rawPart] of content.entries()) {
    if (typeof rawPart !== "object" || rawPart === null || Array.isArray(rawPart)) {
      throw new ChatGPTOAuthError(
        `message ${messageIndex} content block ${contentIndex} must be an object`,
      );
    }
    const part = rawPart as Record<string, unknown>;
    const breakpoint = normalizePromptCacheBreakpoint(
      part.prompt_cache_breakpoint,
      `message ${messageIndex} content block ${contentIndex}`,
    );
    if (role !== MessageRole.USER && breakpoint != null) {
      throw new ChatGPTOAuthError(
        "prompt_cache_breakpoint is supported only on user message content",
      );
    }
    if (["text", "input_text", "output_text"].includes(String(part.type))) {
      if (typeof part.text !== "string") {
        throw new ChatGPTOAuthError(
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
      if (role !== MessageRole.USER) {
        throw new ChatGPTOAuthError(
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
        throw new ChatGPTOAuthError(
          `message ${messageIndex} content block ${contentIndex} image_url requires url`,
        );
      }
      const imageObject = typeof rawImage === "object" && rawImage !== null
        ? rawImage as Record<string, unknown>
        : {};
      const detail = imageObject.detail ?? part.detail;
      if (
        detail != null
        && (
          typeof detail !== "string"
          || !["auto", "low", "high", "original"].includes(detail)
        )
      ) {
        throw new ChatGPTOAuthError(
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
    throw new ChatGPTOAuthError(
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
    throw new ChatGPTOAuthError(
      `${source} prompt_cache_breakpoint must have mode explicit`,
    );
  }
  return { mode: "explicit" };
}

function parseToolCalls(
  raw: Record<string, unknown>[],
): ToolCall[] {
  const calls: ToolCall[] = [];
  for (const item of raw) {
    if (typeof item !== "object" || item === null) continue;
    const callId = String(
      item.id ??
        item.call_id ??
        crypto.randomUUID().replace(/-/g, ""),
    );
    const func = (
      typeof item.function === "object" && item.function !== null
        ? item.function
        : item
    ) as Record<string, unknown>;
    const name = func.name;
    const rawArgs = func.arguments;
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
    if (name) {
      calls.push({
        id: callId,
        name: String(name),
        arguments: args,
      });
    }
  }
  return calls;
}

function parseTools(raw: unknown): ToolSchema[] | undefined {
  if (!Array.isArray(raw) || !raw.length) return undefined;
  const schemas: ToolSchema[] = [];
  for (const item of raw) {
    if (typeof item !== "object" || item === null) continue;
    const func = (item.function ?? item) as Record<
      string,
      unknown
    >;
    const name = func.name;
    if (name) {
      schemas.push({
        name: String(name),
        description: String(func.description || ""),
        parameters: (typeof func.parameters === "object" &&
        func.parameters !== null
          ? func.parameters
          : {}) as Record<string, unknown>,
      });
    }
  }
  return schemas.length ? schemas : undefined;
}

function rejectUnsupportedGenerationFeatures(
  body: Record<string, unknown>,
): void {
  if (body.safety_identifier != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "safety_identifier is not supported by the private Codex OAuth HTTP transport",
    );
  }
  if (body.prompt_cache_options != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "prompt_cache_options is not supported by the private Codex OAuth HTTP transport",
    );
  }
  const { tools: _tools, ...bodyWithoutTools } = body;
  if (containsRequestField(bodyWithoutTools, "prompt_cache_breakpoint")) {
    throw new ChatGPTOAuthInvalidRequestError(
      "prompt_cache_breakpoint is not supported by the private Codex OAuth HTTP transport",
    );
  }
  if (Object.hasOwn(body, "multi_agent") && body.multi_agent != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "multi_agent is not supported by this compatibility API",
    );
  }
  if (
    Object.hasOwn(body, "programmatic_tool_calling")
    && body.programmatic_tool_calling != null
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
  }
}

function containsRequestField(value: unknown, field: string): boolean {
  if (Array.isArray(value)) {
    return value.some((item) => containsRequestField(item, field));
  }
  if (typeof value !== "object" || value === null) return false;
  const record = value as Record<string, unknown>;
  if (Object.hasOwn(record, field)) return true;
  return Object.values(record).some((item) => containsRequestField(item, field));
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
  if (Array.isArray(stop)) return stop.map(String);
  return undefined;
}

export function main(): void {
  primeCodexCliVersionCache();
  const app = createApp();
  app.listen(PORT, HOST, () => {
    console.log(`codex-as-api listening on ${HOST}:${PORT}`);
  });
}
