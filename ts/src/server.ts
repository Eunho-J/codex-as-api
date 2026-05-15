import * as crypto from "node:crypto";
import express, { type Request, type Response } from "express";
import {
  ChatGPTOAuthError,
  ChatGPTOAuthMissingError,
  isAuthLocallyAvailable,
} from "./auth.js";
import type { Message, ToolCall, ToolSchema } from "./messages.js";
import { MessageRole } from "./messages.js";
import { ChatGPTOAuthProvider } from "./provider.js";
import {
  anthropicRequestToInternal,
  internalResponseToAnthropic,
  anthropicStreamAdapter,
  formatAnthropicError,
} from "./anthropic-adapter.js";
import { loadCodexConfig } from "./codex-config.js";

const HOST = process.env.CODEX_AS_API_HOST || "127.0.0.1";
const PORT = parseInt(process.env.CODEX_AS_API_PORT || "18080", 10);
const CODEX_CONFIG = loadCodexConfig();
const MODEL = process.env.CODEX_AS_API_MODEL || CODEX_CONFIG.model || "gpt-5.5";
const AUTH_PATH = process.env.CODEX_AS_API_AUTH_PATH;
const DEFAULT_CONTEXT_WINDOW = 200_000;

function errorStatus(err: unknown): number {
  if (err instanceof ChatGPTOAuthMissingError) return 401;
  if (isContextWindowError(err)) return 400;
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

export function createApp(opts?: {
  provider?: ChatGPTOAuthProvider;
}): express.Express {
  const provider =
    opts?.provider ??
    new ChatGPTOAuthProvider({
      model: MODEL,
      authJsonPath: AUTH_PATH,
    });

  const app = express();
  app.use(express.json({ limit: "50mb" }));

  app.get("/health", (_req: Request, res: Response) => {
    res.json({
      status: "ok",
      auth_available: isAuthLocallyAvailable(AUTH_PATH),
      model: MODEL,
      codex_config_path: CODEX_CONFIG.configPath,
      context_window: getContextWindow(),
      auto_compact_token_limit: getAutoCompactTokenLimit(),
    });
  });

  app.post(
    "/v1/chat/completions",
    async (req: Request, res: Response) => {
      try {
        const body = req.body;
        const messages = requestMessagesToInternal(
          body.messages || [],
        );
        const tools = parseTools(body.tools);
        const stop = normalizeStop(body.stop);
        const maxTokens =
          body.max_completion_tokens ?? body.max_tokens ?? undefined;

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

        const chatOpts = {
          model: body.model,
          tools,
          toolChoice: body.tool_choice,
          temperature: body.temperature,
          reasoningEffort: body.reasoning_effort,
          maxTokens,
          stop,
          promptCacheKey: body.prompt_cache_key,
          subagent,
          memgenRequest,
          previousResponseId: body.previous_response_id,
          serviceTier: body.service_tier,
          text: body.text,
          clientMetadata: body.client_metadata,
        };

        const modelId = `codex-oauth:${body.model || MODEL}`;

        if (body.stream) {
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

          for await (const event of provider.chatStream(
            messages,
            chatOpts,
          )) {
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
              const tc = {
                id: event.id,
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
              };
              res.write(`data: ${JSON.stringify(chunk)}\n\n`);
            }
          }

          if (usageDict) {
            const u = usageDict;
            const finishChunk = {
              id: requestId,
              object: "chat.completion.chunk",
              created,
              model: modelId,
              choices: [],
              usage: {
                prompt_tokens: u.prompt_tokens ?? 0,
                completion_tokens: u.completion_tokens ?? 0,
                total_tokens: u.total_tokens ?? 0,
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

          if (response.usage) {
            result.usage = {
              prompt_tokens: response.usage.prompt_tokens,
              completion_tokens:
                response.usage.completion_tokens,
              total_tokens: response.usage.total_tokens,
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
        const images = await provider.generateImage(body.prompt, {
          model: body.model,
          size: body.size,
          reasoningEffort: body.reasoning_effort,
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
      const result = await provider.inspectImages(
        String(body.prompt || ""),
        {
          images: body.images || [],
          reasoningEffort: body.reasoning_effort,
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
      const { messages, reasoningEffort } = messagesFromCompactBody(body);
      const checkpoint = await provider.compactMessages(messages, {
        model: MODEL,
        reasoningEffort: body.reasoning_effort ?? reasoningEffort ?? undefined,
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
      const { messages, tools, toolChoice, stop, reasoningEffort } = anthropicRequestToInternal({
        model: body.model,
        messages: body.messages || [],
        system: body.system,
        maxTokens: body.max_tokens,
        tools: body.tools,
        toolChoice: body.tool_choice,
        stopSequences: body.stop_sequences,
        thinking: body.thinking,
      });
      const countTokens = (provider as { countTokens?: (
        messages: Message[],
        opts?: {
          model?: string;
          tools?: ToolSchema[];
          toolChoice?: string | Record<string, unknown>;
          stop?: string | string[];
          reasoningEffort?: string;
        },
      ) => Promise<number> }).countTokens;
      if (typeof countTokens !== "function") {
        throw new ChatGPTOAuthError("provider does not support real token counting");
      }
      const inputTokens = await countTokens.call(provider, messages, {
        model: MODEL,
        tools: tools ?? undefined,
        toolChoice: toolChoice ?? undefined,
        stop: stop ?? undefined,
        reasoningEffort: reasoningEffort ?? undefined,
      });
      res.json({
        input_tokens: inputTokens,
        context_window: getContextWindow(),
        auto_compact_token_limit: getAutoCompactTokenLimit(),
      });
    } catch (err) {
      handleAnthropicError(err, res);
    }
  });

  app.post("/v1/messages", async (req: Request, res: Response) => {
    try {
      const body = req.body;
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

      const { messages, tools, toolChoice, stop, reasoningEffort } =
        anthropicRequestToInternal({
          model: body.model,
          messages: body.messages || [],
          system: body.system,
          maxTokens: body.max_tokens,
          tools: body.tools,
          toolChoice: body.tool_choice,
          stopSequences: body.stop_sequences,
          thinking: body.thinking,
        });

      const clientModel = body.model || "claude-sonnet-4-5";
      const requestModel = MODEL;
      const chatOpts = {
        model: requestModel,
        tools: tools ?? undefined,
        toolChoice: toolChoice ?? undefined,
        reasoningEffort: reasoningEffort ?? undefined,
        maxTokens: body.max_tokens,
        stop: stop ?? undefined,
        subagent,
        memgenRequest,
      };

      if (body.stream) {
        res.setHeader("Content-Type", "text/event-stream");
        res.setHeader("Cache-Control", "no-cache");
        res.setHeader("Connection", "keep-alive");

        for await (const chunk of anthropicStreamAdapter(
          provider.chatStream(messages, chatOpts),
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

// --- Helpers ---

function getContextWindow(): number {
  return CODEX_CONFIG.modelContextWindow || DEFAULT_CONTEXT_WINDOW;
}

function getAutoCompactTokenLimit(): number {
  return CODEX_CONFIG.modelAutoCompactTokenLimit || Math.floor(getContextWindow() * 0.8);
}

function messagesFromCompactBody(body: Record<string, unknown>): {
  messages: Message[];
  reasoningEffort: string | null;
} {
  if (body.system != null || body.thinking != null || body.tool_choice != null || body.stop_sequences != null) {
    const converted = anthropicRequestToInternal({
      model: String(body.model || MODEL),
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
    });
    return { messages: converted.messages, reasoningEffort: converted.reasoningEffort };
  }

  const rawMessages = Array.isArray(body.messages) ? body.messages as Record<string, unknown>[] : [];
  return { messages: requestMessagesToInternal(rawMessages), reasoningEffort: null };
}


function requestMessagesToInternal(
  rawMessages: Record<string, unknown>[],
): Message[] {
  const result: Message[] = [];
  for (const msg of rawMessages) {
    const role = mapRole(String(msg.role || "user"));
    const content = normalizeContent(msg.content);
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

function normalizeContent(content: unknown): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter(
        (item) =>
          typeof item === "object" &&
          item !== null &&
          typeof item.text === "string",
      )
      .map((item) => item.text)
      .join("");
  }
  return String(content);
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

function normalizeStop(stop: unknown): string[] | undefined {
  if (stop == null) return undefined;
  if (typeof stop === "string") return [stop];
  if (Array.isArray(stop)) return stop.map(String);
  return undefined;
}

export function main(): void {
  const app = createApp();
  app.listen(PORT, HOST, () => {
    console.log(`codex-as-api listening on ${HOST}:${PORT}`);
  });
}

