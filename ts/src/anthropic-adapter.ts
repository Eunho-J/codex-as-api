import type {
  AssistantResponse,
  Message,
  ToolCall,
  ToolSchema,
} from "./messages.js";
import { MessageRole } from "./messages.js";

// ---------------------------------------------------------------------------
// Request conversion: Anthropic → internal
// ---------------------------------------------------------------------------

export function anthropicRequestToInternal(opts: {
  model: string;
  messages: Record<string, unknown>[];
  system?: string | Record<string, unknown>[];
  maxTokens?: number;
  tools?: Record<string, unknown>[];
  toolChoice?: Record<string, unknown>;
  stopSequences?: string[];
  thinking?: Record<string, unknown>;
}): {
  messages: Message[];
  tools: ToolSchema[] | null;
  toolChoice: string | Record<string, unknown> | null;
  stop: string[] | null;
  reasoningEffort: string | null;
} {
  const internalMessages: Message[] = [];

  if (opts.system != null) {
    const sysText = extractSystemText(opts.system);
    if (sysText) {
      internalMessages.push({ role: MessageRole.SYSTEM, content: sysText });
    }
  }

  for (const msg of opts.messages) {
    const role = String(msg.role ?? "user");
    const content = msg.content;
    if (role === "user") {
      convertUserMessage(content, internalMessages);
    } else if (role === "assistant") {
      convertAssistantMessage(content, internalMessages);
    }
  }

  const internalTools = opts.tools ? convertTools(opts.tools) : null;
  const internalToolChoice = convertToolChoice(opts.toolChoice ?? null);
  const reasoningEffort = convertThinking(opts.thinking ?? null);

  return {
    messages: internalMessages,
    tools: internalTools,
    toolChoice: internalToolChoice,
    stop: opts.stopSequences ?? null,
    reasoningEffort,
  };
}

function extractSystemText(
  system: string | Record<string, unknown>[],
): string {
  if (typeof system === "string") return system;
  const parts: string[] = [];
  for (const block of system) {
    if (
      typeof block === "object" &&
      block !== null &&
      block.type === "text" &&
      typeof block.text === "string" &&
      block.text
    ) {
      parts.push(block.text);
    }
  }
  return parts.join("\n\n");
}

function convertUserMessage(
  content: unknown,
  out: Message[],
): void {
  if (typeof content === "string") {
    out.push({ role: MessageRole.USER, content });
    return;
  }
  if (!Array.isArray(content)) return;
  const textParts: string[] = [];
  const imageUrls: string[] = [];
  for (const block of content) {
    if (typeof block !== "object" || block === null) continue;
    const b = block as Record<string, unknown>;
    const blockType = b.type;
    if (blockType === "text") {
      if (typeof b.text === "string") textParts.push(b.text);
    } else if (blockType === "tool_result") {
      if (textParts.length || imageUrls.length) {
        out.push({ role: MessageRole.USER, content: textParts.join(""), images: [...imageUrls] });
        textParts.length = 0;
        imageUrls.length = 0;
      }
      const toolUseId = typeof b.tool_use_id === "string" ? b.tool_use_id : "tool-call";
      let resultContent = b.content ?? "";
      const toolResultImages: string[] = [];
      if (Array.isArray(resultContent)) {
        const textPieces: string[] = [];
        for (const p of resultContent as Record<string, unknown>[]) {
          if (typeof p !== "object" || p === null) continue;
          if (p.type === "text") {
            textPieces.push((p.text ?? "") as string);
          } else if (p.type === "image") {
            const source = p.source as Record<string, unknown> | undefined;
            if (source && source.type === "base64") {
              const mediaType = typeof source.media_type === "string" ? source.media_type : "image/png";
              const data = typeof source.data === "string" ? source.data : "";
              toolResultImages.push(`data:${mediaType};base64,${data}`);
            }
          }
        }
        resultContent = textPieces.join("");
      } else if (typeof resultContent !== "string") {
        resultContent = resultContent ? String(resultContent) : "";
      }
      out.push({
        role: MessageRole.TOOL,
        content: resultContent as string,
        tool_call_id: toolUseId,
        name: toolUseId,
      });
      if (toolResultImages.length) {
        out.push({ role: MessageRole.USER, content: "", images: toolResultImages });
      }
    } else if (blockType === "image") {
      const source = b.source as Record<string, unknown> | undefined;
      if (source && source.type === "base64") {
        const mediaType = typeof source.media_type === "string" ? source.media_type : "image/png";
        const data = typeof source.data === "string" ? source.data : "";
        imageUrls.push(`data:${mediaType};base64,${data}`);
      }
    }
  }
  if (textParts.length || imageUrls.length) {
    out.push({ role: MessageRole.USER, content: textParts.join(""), images: [...imageUrls] });
  }
}

function convertAssistantMessage(
  content: unknown,
  out: Message[],
): void {
  if (typeof content === "string") {
    out.push({ role: MessageRole.ASSISTANT, content });
    return;
  }
  if (!Array.isArray(content)) return;
  const textParts: string[] = [];
  const toolCalls: ToolCall[] = [];
  let reasoningContent: string | null = null;
  for (const block of content) {
    if (typeof block !== "object" || block === null) continue;
    const b = block as Record<string, unknown>;
    const blockType = b.type;
    if (blockType === "text") {
      if (typeof b.text === "string") textParts.push(b.text);
    } else if (blockType === "tool_use") {
      toolCalls.push({
        id: typeof b.id === "string" ? b.id : crypto.randomUUID().replace(/-/g, ""),
        name: typeof b.name === "string" ? b.name : "",
        arguments: (typeof b.input === "object" && b.input !== null && !Array.isArray(b.input)
          ? b.input
          : {}) as Record<string, unknown>,
      });
    } else if (blockType === "thinking") {
      if (typeof b.thinking === "string" && b.thinking) {
        reasoningContent = b.thinking;
      }
    }
  }
  const msg: Message = {
    role: MessageRole.ASSISTANT,
    content: textParts.join(""),
    tool_calls: toolCalls.length ? toolCalls : [],
  };
  if (reasoningContent !== null) msg.reasoning_content = reasoningContent;
  out.push(msg);
}

function convertTools(tools: Record<string, unknown>[]): ToolSchema[] {
  const result: ToolSchema[] = [];
  for (const tool of tools) {
    if (typeof tool !== "object" || tool === null) continue;
    const name = tool.name;
    if (!name) continue;
    result.push({
      name: String(name),
      description: String(tool.description ?? ""),
      parameters: (typeof tool.input_schema === "object" && tool.input_schema !== null
        ? tool.input_schema
        : {}) as Record<string, unknown>,
    });
  }
  return result;
}

function convertToolChoice(
  tc: Record<string, unknown> | null,
): string | Record<string, unknown> | null {
  if (tc === null) return null;
  const tcType = tc.type;
  if (tcType === "auto") return "auto";
  if (tcType === "any") return "required";
  if (tcType === "tool") return { type: "function", function: { name: tc.name } };
  if (tcType === "none") return "none";
  return "auto";
}

function convertThinking(thinking: Record<string, unknown> | null): string | null {
  if (thinking === null) return null;
  if (thinking.type === "enabled") return "high";
  if (thinking.type === "adaptive") return "medium";
  return null;
}

// ---------------------------------------------------------------------------
// Non-streaming response: internal → Anthropic
// ---------------------------------------------------------------------------

export function internalResponseToAnthropic(
  response: AssistantResponse,
  model: string,
  requestId: string,
): Record<string, unknown> {
  const content: Record<string, unknown>[] = [];

  if (response.reasoning_content) {
    content.push({
      type: "thinking",
      thinking: response.reasoning_content,
      signature: "sig-placeholder",
    });
  }

  if (response.content) {
    content.push({ type: "text", text: response.content });
  }

  for (const tc of response.tool_calls) {
    content.push({
      type: "tool_use",
      id: tc.id,
      name: tc.name,
      input: tc.arguments,
    });
  }

  const stopReason = mapStopReason(response.finish_reason, response.tool_calls.length > 0);

  let usageDict: Record<string, unknown> = { input_tokens: 0, output_tokens: 0 };
  if (response.usage) {
    usageDict = {
      input_tokens: response.usage.prompt_tokens,
      output_tokens: response.usage.completion_tokens,
      cache_creation_input_tokens: 0,
      cache_read_input_tokens: response.usage.cached_tokens,
    };
  }

  if (!content.length) {
    content.push({ type: "text", text: "" });
  }

  return {
    id: requestId,
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: usageDict,
  };
}

function mapStopReason(finishReason: string, hasToolCalls: boolean): string {
  if (hasToolCalls) return "tool_use";
  const mapping: Record<string, string> = {
    stop: "end_turn",
    length: "max_tokens",
    max_tokens: "max_tokens",
    tool_calls: "tool_use",
    tool_use: "tool_use",
    stop_sequence: "stop_sequence",
  };
  return mapping[finishReason] ?? "end_turn";
}

// ---------------------------------------------------------------------------
// Streaming adapter: provider events → Anthropic SSE
// ---------------------------------------------------------------------------

export async function* anthropicStreamAdapter(
  eventIterator: AsyncIterable<Record<string, unknown>>,
  model: string,
  requestId: string,
): AsyncGenerator<string> {
  yield sse("message_start", {
    type: "message_start",
    message: {
      id: requestId,
      type: "message",
      role: "assistant",
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });

  let blockIndex = 0;
  let currentBlock: "thinking" | "text" | "tool_use" | null = null;
  let hasAnyContent = false;
  let outputTokens = 0;

  for await (const event of eventIterator) {
    const typ = event.type;

    if (typ === "reasoning_delta" || typ === "reasoning_raw_delta") {
      hasAnyContent = true;
      const text = String(event.text ?? "");
      if (currentBlock !== "thinking") {
        if (currentBlock !== null) {
          yield sse("content_block_stop", { type: "content_block_stop", index: blockIndex });
          blockIndex++;
        }
        yield sse("content_block_start", {
          type: "content_block_start",
          index: blockIndex,
          content_block: { type: "thinking", thinking: "", signature: "" },
        });
        currentBlock = "thinking";
      }
      yield sse("content_block_delta", {
        type: "content_block_delta",
        index: blockIndex,
        delta: { type: "thinking_delta", thinking: text },
      });
    } else if (typ === "content") {
      hasAnyContent = true;
      const text = String(event.text ?? "");
      if (currentBlock === "thinking") {
        yield sse("content_block_delta", {
          type: "content_block_delta",
          index: blockIndex,
          delta: { type: "signature_delta", signature: "sig-placeholder" },
        });
        yield sse("content_block_stop", { type: "content_block_stop", index: blockIndex });
        blockIndex++;
        currentBlock = null;
      }
      if (currentBlock !== "text") {
        if (currentBlock !== null) {
          yield sse("content_block_stop", { type: "content_block_stop", index: blockIndex });
          blockIndex++;
        }
        yield sse("content_block_start", {
          type: "content_block_start",
          index: blockIndex,
          content_block: { type: "text", text: "" },
        });
        currentBlock = "text";
      }
      yield sse("content_block_delta", {
        type: "content_block_delta",
        index: blockIndex,
        delta: { type: "text_delta", text },
      });
    } else if (typ === "tool_call") {
      hasAnyContent = true;
      if (currentBlock !== null) {
        if (currentBlock === "thinking") {
          yield sse("content_block_delta", {
            type: "content_block_delta",
            index: blockIndex,
            delta: { type: "signature_delta", signature: "sig-placeholder" },
          });
        }
        yield sse("content_block_stop", { type: "content_block_stop", index: blockIndex });
        blockIndex++;
      }
      const toolId = String(event.id ?? "");
      const toolName = String(event.name ?? "");
      const toolArgs = (typeof event.arguments === "object" && event.arguments !== null
        ? event.arguments
        : {}) as Record<string, unknown>;
      yield sse("content_block_start", {
        type: "content_block_start",
        index: blockIndex,
        content_block: { type: "tool_use", id: toolId, name: toolName, input: {} },
      });
      yield sse("content_block_delta", {
        type: "content_block_delta",
        index: blockIndex,
        delta: { type: "input_json_delta", partial_json: JSON.stringify(toolArgs) },
      });
      yield sse("content_block_stop", { type: "content_block_stop", index: blockIndex });
      blockIndex++;
      currentBlock = null;
    } else if (typ === "finish") {
      if (currentBlock !== null) {
        if (currentBlock === "thinking") {
          yield sse("content_block_delta", {
            type: "content_block_delta",
            index: blockIndex,
            delta: { type: "signature_delta", signature: "sig-placeholder" },
          });
        }
        yield sse("content_block_stop", { type: "content_block_stop", index: blockIndex });
        currentBlock = null;
      }

      if (!hasAnyContent) {
        yield sse("content_block_start", {
          type: "content_block_start",
          index: blockIndex,
          content_block: { type: "text", text: "" },
        });
        yield sse("content_block_stop", { type: "content_block_stop", index: blockIndex });
      }

      const finishReason = String(event.finish_reason ?? "stop");
      const stopReason = mapStopReason(finishReason, false);
      const usageEvent = event.usage;
      if (typeof usageEvent === "object" && usageEvent !== null) {
        const u = usageEvent as Record<string, unknown>;
        outputTokens = Number(u.output_tokens ?? u.completion_tokens ?? 0);
      }

      yield sse("message_delta", {
        type: "message_delta",
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: { output_tokens: outputTokens },
      });
      yield sse("message_stop", { type: "message_stop" });
    }
  }
}

function sse(eventType: string, data: Record<string, unknown>): string {
  return `event: ${eventType}\ndata: ${JSON.stringify(data)}\n\n`;
}

// ---------------------------------------------------------------------------
// Error formatting
// ---------------------------------------------------------------------------

export function formatAnthropicError(
  status: number,
  message: string,
): Record<string, unknown> {
  const typeMap: Record<number, string> = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    429: "rate_limit_error",
    500: "api_error",
    529: "overloaded_error",
  };
  return {
    type: "error",
    error: {
      type: typeMap[status] ?? "api_error",
      message,
    },
  };
}
