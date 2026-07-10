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
  outputFormat?: Record<string, unknown>;
}): {
  messages: Message[];
  tools: ToolSchema[] | null;
  toolChoice: string | Record<string, unknown> | null;
  stop: string[] | null;
  reasoningEffort: string | null;
  text: Record<string, unknown> | null;
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
  const text = anthropicOutputFormatToOpenAIText(opts.outputFormat ?? null);

  return {
    messages: internalMessages,
    tools: internalTools,
    toolChoice: internalToolChoice,
    stop: opts.stopSequences ?? null,
    reasoningEffort,
    text,
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
          } else {
            const rendered = renderAnthropicContentBlock(p);
            if (rendered) textPieces.push(rendered);
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
    } else {
      const rendered = renderAnthropicContentBlock(b);
      if (rendered) textParts.push(rendered);
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
    } else if (blockType === "redacted_thinking") {
      if (reasoningContent === null) {
        reasoningContent = "[redacted_thinking omitted]";
      }
    } else if (blockType === "server_tool_use") {
      textParts.push(renderServerToolUseBlock(b));
    } else if (blockType === "web_search_tool_result") {
      textParts.push(renderWebSearchToolResultBlock(b));
    } else {
      const rendered = renderAnthropicContentBlock(b);
      if (rendered) textParts.push(rendered);
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
  for (const [index, tool] of tools.entries()) {
    if (typeof tool !== "object" || tool === null) continue;
    if (tool.type === "programmatic_tool_calling") {
      throw new Error(
        "programmatic_tool_calling tools are not supported by this compatibility API",
      );
    }
    if (Object.hasOwn(tool, "allowed_callers")) {
      throw new Error(`tool ${index} allowed_callers is not supported`);
    }
    if (Object.hasOwn(tool, "output_schema")) {
      throw new Error(`tool ${index} output_schema is not supported`);
    }
    const name = tool.name;
    if (!name) continue;
    if (isAnthropicWebSearchTool(tool)) {
      result.push({
        name: "web_search",
        description: "Anthropic hosted web search",
        parameters: anthropicWebSearchParameters(tool),
      });
      continue;
    }
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

function isAnthropicWebSearchTool(tool: Record<string, unknown>): boolean {
  return tool.name === "web_search" &&
    typeof tool.type === "string" &&
    (tool.type === "web_search" || tool.type.startsWith("web_search_"));
}

function stringArray(value: unknown): string[] | null {
  if (!Array.isArray(value)) return null;
  const result = value.filter((v): v is string => typeof v === "string" && v.length > 0);
  return result.length ? result : null;
}

function anthropicWebSearchParameters(tool: Record<string, unknown>): Record<string, unknown> {
  const blockedDomains = stringArray(tool.blocked_domains);
  if (blockedDomains) {
    throw new Error(
      "Anthropic web_search blocked_domains is not supported by OpenAI Responses web_search; use allowed_domains instead",
    );
  }

  const openaiTool: Record<string, unknown> = {
    type: "web_search",
    external_web_access: true,
  };
  const allowedDomains = stringArray(tool.allowed_domains);
  if (allowedDomains) {
    openaiTool.filters = { allowed_domains: allowedDomains };
  }
  if (
    typeof tool.user_location === "object" &&
    tool.user_location !== null &&
    !Array.isArray(tool.user_location)
  ) {
    openaiTool.user_location = tool.user_location;
  }

  return {
    __codex_as_api_tool_type: "web_search",
    openai_tool: openaiTool,
    anthropic: {
      type: tool.type,
      max_uses: tool.max_uses,
    },
  };
}

function convertToolChoice(
  tc: Record<string, unknown> | null,
): string | Record<string, unknown> | null {
  if (tc === null) return null;
  const tcType = tc.type;
  if (tcType === "auto") return "auto";
  if (tcType === "any") return "required";
  if (tcType === "tool") {
    if (tc.name === "web_search") return { type: "web_search" };
    return { type: "function", name: tc.name };
  }
  if (tcType === "none") return "none";
  return "auto";
}

function convertThinking(thinking: Record<string, unknown> | null): string | null {
  if (thinking === null) return null;
  if (thinking.type === "enabled") return "high";
  if (thinking.type === "adaptive") return "medium";
  if (thinking.type === "disabled") return "none";
  return null;
}

export function anthropicOutputFormatToOpenAIText(
  outputFormat: Record<string, unknown> | null,
): Record<string, unknown> | null {
  if (outputFormat === null || typeof outputFormat !== "object") return null;
  const type = outputFormat.type;
  if (type === "json_schema") {
    const schema = outputFormat.schema;
    if (typeof schema !== "object" || schema === null || Array.isArray(schema)) return null;
    const name = sanitizeJsonSchemaName(
      typeof outputFormat.name === "string" && outputFormat.name ? outputFormat.name : "structured_output",
    );
    const format: Record<string, unknown> = {
      type: "json_schema",
      name,
      schema,
    };
    if (typeof outputFormat.description === "string") {
      format.description = outputFormat.description;
    }
    if (typeof outputFormat.strict === "boolean") {
      format.strict = outputFormat.strict;
    }
    return { format };
  }
  if (type === "json_object") {
    return { format: { type: "json_object" } };
  }
  return null;
}

function sanitizeJsonSchemaName(name: string): string {
  const cleaned = name.replace(/[^A-Za-z0-9_-]/g, "_").slice(0, 64);
  return cleaned || "structured_output";
}

function renderAnthropicContentBlock(block: Record<string, unknown>): string {
  const type = typeof block.type === "string" ? block.type : "unknown";
  if (type === "document") return renderDocumentBlock(block);
  if (type === "search_result") return renderSearchResultBlock(block);
  if (type.endsWith("_tool_result")) return renderGenericToolResultBlock(block);
  try {
    return `\n\n[${type}] ${JSON.stringify(block)}\n`;
  } catch {
    return `\n\n[${type}]\n`;
  }
}

function renderDocumentBlock(block: Record<string, unknown>): string {
  const title = typeof block.title === "string" ? block.title : typeof block.name === "string" ? block.name : "document";
  const source = block.source;
  let body = "";
  if (typeof source === "object" && source !== null && !Array.isArray(source)) {
    const src = source as Record<string, unknown>;
    if (src.type === "text" && typeof src.data === "string") body = src.data;
    else if (src.type === "url" && typeof src.url === "string") body = src.url;
    else if (typeof src.media_type === "string") body = `[${src.media_type}]`;
  }
  return `\n\n[document: ${title}]${body ? `\n${body}` : ""}\n`;
}

function renderSearchResultBlock(block: Record<string, unknown>): string {
  const title = typeof block.title === "string" ? block.title : "search result";
  const url = typeof block.url === "string" ? block.url : "";
  const content = typeof block.content === "string" ? block.content : "";
  return `\n\n[search_result] ${title}${url ? ` (${url})` : ""}${content ? `\n${content}` : ""}\n`;
}

function renderServerToolUseBlock(block: Record<string, unknown>): string {
  const name = typeof block.name === "string" ? block.name : "server_tool";
  const input = block.input ?? {};
  return `\n\n[server_tool_use: ${name}] ${safeJson(input)}\n`;
}

function renderWebSearchToolResultBlock(block: Record<string, unknown>): string {
  return renderGenericToolResultBlock({ ...block, type: "web_search_tool_result" });
}

function renderGenericToolResultBlock(block: Record<string, unknown>): string {
  const type = typeof block.type === "string" ? block.type : "tool_result";
  const content = block.content;
  if (Array.isArray(content)) {
    const rendered = content.map((item) => {
      if (typeof item !== "object" || item === null) return String(item);
      const i = item as Record<string, unknown>;
      const title = typeof i.title === "string" ? i.title : undefined;
      const url = typeof i.url === "string" ? i.url : undefined;
      const text = typeof i.text === "string" ? i.text : typeof i.content === "string" ? i.content : undefined;
      if (title || url || text) {
        return `- ${title ?? "result"}${url ? ` (${url})` : ""}${text ? `: ${text}` : ""}`;
      }
      return safeJson(i);
    }).join("\n");
    return `\n\n[${type}]${rendered ? `\n${rendered}` : ""}\n`;
  }
  if (typeof content === "object" && content !== null) return `\n\n[${type}] ${safeJson(content)}\n`;
  if (typeof content === "string") return `\n\n[${type}]\n${content}\n`;
  return `\n\n[${type}]\n`;
}

function safeJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
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

  const webSearchBlocks = webSearchBlocksFromRaw(response.raw);
  content.push(...webSearchBlocks);

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
      cache_creation_input_tokens: response.usage.cache_write_tokens ?? 0,
      cache_read_input_tokens: response.usage.cached_tokens,
    };
  }
  usageDict = mergeServerToolUsage(usageDict, response.raw, webSearchBlocks.length / 2);

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

function webSearchBlocksFromRaw(raw: Record<string, unknown> | null): Record<string, unknown>[] {
  const events = Array.isArray(raw?.events) ? raw.events : [];
  const blocks: Record<string, unknown>[] = [];
  for (const event of events) {
    if (typeof event !== "object" || event === null) continue;
    const e = event as Record<string, unknown>;
    if (e.type !== "web_search_call") continue;
    const id = String(e.id || `srvtoolu_${blocks.length / 2}`);
    const input = (typeof e.input === "object" && e.input !== null && !Array.isArray(e.input))
      ? e.input
      : { query: "" };
    const content = Array.isArray(e.content) ? e.content : [];
    blocks.push({ type: "server_tool_use", id, name: "web_search", input });
    blocks.push({ type: "web_search_tool_result", tool_use_id: id, content });
  }
  return blocks;
}

function mergeServerToolUsage(
  usage: Record<string, unknown>,
  raw: Record<string, unknown> | null,
  webSearchRequests: number,
): Record<string, unknown> {
  const events = Array.isArray(raw?.events) ? raw.events : [];
  for (const event of events) {
    if (typeof event !== "object" || event === null) continue;
    const e = event as Record<string, unknown>;
    if (e.type !== "finish") continue;
    const rawUsage = e.usage;
    if (typeof rawUsage !== "object" || rawUsage === null || Array.isArray(rawUsage)) continue;
    const serverToolUse = (rawUsage as Record<string, unknown>).server_tool_use;
    if (serverToolUse !== undefined) {
      usage.server_tool_use = serverToolUse;
      return usage;
    }
  }
  if (webSearchRequests > 0 && usage.server_tool_use === undefined) {
    usage.server_tool_use = { web_search_requests: webSearchRequests };
  }
  return usage;
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
    pause_turn: "pause_turn",
    refusal: "refusal",
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
  yield messageStartSse(model, requestId, { input_tokens: 0, output_tokens: 0 });
  for await (const chunk of renderAnthropicStreamEvents(eventIterator)) {
    yield chunk;
  }
}

function messageStartSse(
  model: string,
  requestId: string,
  usage: Record<string, unknown>,
): string {
  return sse("message_start", {
    type: "message_start",
    message: {
      id: requestId,
      type: "message",
      role: "assistant",
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage,
    },
  });
}

function usageFromProviderEvent(usageEvent: unknown): Record<string, unknown> {
  if (typeof usageEvent !== "object" || usageEvent === null || Array.isArray(usageEvent)) {
    return { input_tokens: 0, output_tokens: 0 };
  }
  const u = usageEvent as Record<string, unknown>;
  const inputTokens = numeric(u.input_tokens ?? u.prompt_tokens);
  const outputTokens = numeric(u.output_tokens ?? u.completion_tokens);
  const tokenDetails = u.input_tokens_details ?? u.prompt_tokens_details;
  let cacheRead = numeric(u.cache_read_input_tokens ?? u.cached_input_tokens);
  let cacheWrite = numeric(
    u.cache_creation_input_tokens
    ?? u.cache_write_tokens
    ?? u.cache_write_input_tokens,
  );
  if (
    typeof tokenDetails === "object"
    && tokenDetails !== null
    && !Array.isArray(tokenDetails)
  ) {
    const details = tokenDetails as Record<string, unknown>;
    if (cacheRead === 0) cacheRead = numeric(details.cached_tokens);
    if (cacheWrite === 0) cacheWrite = numeric(details.cache_write_tokens);
  }

  const result: Record<string, unknown> = {
    input_tokens: inputTokens,
    output_tokens: outputTokens,
    cache_creation_input_tokens: cacheWrite,
    cache_read_input_tokens: cacheRead,
  };
  for (const key of ["cache_creation", "server_tool_use", "service_tier"] as const) {
    if (u[key] !== undefined) result[key] = u[key];
  }
  return result;
}

function numeric(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

async function* renderAnthropicStreamEvents(
  events: AsyncIterable<Record<string, unknown>>,
): AsyncGenerator<string> {
  let blockIndex = 0;
  let currentBlock: "thinking" | "text" | "tool_use" | null = null;
  let hasAnyContent = false;
  let webSearchRequests = 0;

  for await (const event of events) {
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
    } else if (typ === "web_search_call") {
      hasAnyContent = true;
      webSearchRequests++;
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
      const toolInput = (typeof event.input === "object" && event.input !== null
        ? event.input
        : { query: "" }) as Record<string, unknown>;
      const resultContent = Array.isArray(event.content) ? event.content : [];
      yield sse("content_block_start", {
        type: "content_block_start",
        index: blockIndex,
        content_block: { type: "server_tool_use", id: toolId, name: "web_search", input: {} },
      });
      yield sse("content_block_delta", {
        type: "content_block_delta",
        index: blockIndex,
        delta: { type: "input_json_delta", partial_json: JSON.stringify(toolInput) },
      });
      yield sse("content_block_stop", { type: "content_block_stop", index: blockIndex });
      blockIndex++;
      yield sse("content_block_start", {
        type: "content_block_start",
        index: blockIndex,
        content_block: {
          type: "web_search_tool_result",
          tool_use_id: toolId,
          content: resultContent,
        },
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

      yield sse("message_delta", {
        type: "message_delta",
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: usageWithSynthesizedWebSearch(usageFromProviderEvent(event.usage), webSearchRequests),
      });
      yield sse("message_stop", { type: "message_stop" });
    }
  }
}

function usageWithSynthesizedWebSearch(
  usage: Record<string, unknown>,
  webSearchRequests: number,
): Record<string, unknown> {
  if (webSearchRequests > 0 && usage.server_tool_use === undefined) {
    usage.server_tool_use = { web_search_requests: webSearchRequests };
  }
  return usage;
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
