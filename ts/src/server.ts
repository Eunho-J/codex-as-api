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

const HOST = process.env.CODEX_AS_API_HOST || "0.0.0.0";
const PORT = parseInt(process.env.CODEX_AS_API_PORT || "18080", 10);
const MODEL = process.env.CODEX_AS_API_MODEL || "gpt-5.5";
const AUTH_PATH = process.env.CODEX_AS_API_AUTH_PATH;

function handleError(err: unknown, res: Response): void {
  if (err instanceof ChatGPTOAuthMissingError) {
    res
      .status(401)
      .json({
        error: { message: String(err), type: "chatgpt_oauth_error" },
      });
  } else if (err instanceof ChatGPTOAuthError) {
    res
      .status(500)
      .json({
        error: { message: String(err), type: "chatgpt_oauth_error" },
      });
  } else {
    res
      .status(500)
      .json({
        error: { message: String(err), type: "server_error" },
      });
  }
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
        handleError(err, res);
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

  app.post("/v1/compact", async (req: Request, res: Response) => {
    try {
      const body = req.body;
      const rawMessages = body.messages || [];
      const messages = requestMessagesToInternal(rawMessages);
      const checkpoint = await provider.compactMessages(messages, {
        reasoningEffort: body.reasoning_effort,
      });
      res.json({ checkpoint });
    } catch (err) {
      handleError(err, res);
    }
  });

  return app;
}

// --- Helpers ---

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
