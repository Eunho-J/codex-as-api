import * as crypto from "node:crypto";
import {
  ChatGPTOAuthError,
  loadTokenData,
  redactText,
  refreshToken,
} from "./auth.js";
import type {
  AssistantResponse,
  Message,
  ToolCall,
  ToolSchema,
  Usage,
} from "./messages.js";
import { MessageRole } from "./messages.js";
import {
  reasoningFromResponseItems,
  responseFailureMessage,
} from "./protocol.js";

export const CHATGPT_OAUTH_DEFAULT_BASE_URL =
  "https://chatgpt.com/backend-api/codex";
export const CHATGPT_OAUTH_DEFAULT_MODEL = "gpt-5.5";
const REMOTE_COMPACTION_MARKER = "[Remote Responses compacted history]";
const REASONING_EFFORT_VALUES = new Set([
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "xhigh",
]);

export interface ChatOptions {
  model?: string;
  tools?: ToolSchema[];
  toolChoice?: string | Record<string, unknown>;
  temperature?: number;
  reasoningEffort?: string;
  maxTokens?: number;
  stop?: string | string[];
  promptCacheKey?: string;
  subagent?: string;
  memgenRequest?: boolean;
  previousResponseId?: string;
  serviceTier?: string;
  text?: Record<string, unknown>;
  clientMetadata?: Record<string, string>;
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
      }
    }

    return {
      content: contentParts.join(""),
      tool_calls: toolCalls,
      finish_reason: finishReason,
      usage,
      reasoning_content: reasoningParts.join("") || null,
      raw: { events: rawEvents.slice(-20) },
    };
  }

  async *chatStream(
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

    const finalOutput: Record<string, unknown>[] = [];
    const reasoningParts: string[] = [];
    let sawTextDelta = false;
    let sawReasoningDelta = false;

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
        if (typeof item === "object" && item !== null) {
          const itemDict = item as Record<string, unknown>;
          finalOutput.push(itemDict);
          const tool = toolCallFromResponseItem(itemDict);
          if (tool) {
            yield {
              type: "tool_call",
              id: tool.id,
              name: tool.name,
              arguments: tool.arguments,
            };
          }
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
        const response = event.response as Record<string, unknown> | undefined;
        let usageData: unknown = undefined;
        if (typeof response === "object" && response !== null) {
          usageData = response.usage;
          if (!finalOutput.length && Array.isArray(response.output)) {
            for (const item of response.output) {
              if (typeof item === "object" && item !== null) {
                finalOutput.push(item as Record<string, unknown>);
              }
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
        }
        yield {
          type: "finish",
          finish_reason: "stop",
          usage: usageData,
          reasoning_content: reasoningParts.join("") || null,
        };
      }
    }
  }

  async generateImage(
    prompt: string,
    opts: {
      model?: string;
      referenceImages?: { image_url: string }[];
      size?: string;
      reasoningEffort?: string;
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
    const payload: Record<string, unknown> = {
      model: opts.model || this.model,
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
    setReasoningPayload(payload, opts.reasoningEffort);
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
      images: { image_url: string }[];
      reasoningEffort?: string;
    },
  ): Promise<string> {
    if (!prompt || prompt.trim() === "") {
      throw new ChatGPTOAuthError("image inspection prompt is required");
    }
    const content: Record<string, unknown>[] = [
      { type: "input_text", text: prompt },
    ];
    content.push(...validateImageContentItems(opts.images));
    const payload: Record<string, unknown> = {
      model: opts.model || this.model,
      instructions:
        "Inspect the attached image(s) and answer the user's review prompt directly.",
      input: [{ type: "message", role: "user", content }],
      tools: [],
      parallel_tool_calls: false,
      stream: true,
      store: false,
      include: [],
      prompt_cache_key: crypto.randomUUID(),
    };
    setReasoningPayload(payload, opts.reasoningEffort);
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
    opts: { model?: string; reasoningEffort?: string } = {},
  ): Promise<string> {
    const payload: Record<string, unknown> = {
      model: opts.model || this.model,
      input: messagesToResponseItems(messages),
      instructions:
        "Create a compact checkpoint of this conversation for continuation.",
      tools: [],
      parallel_tool_calls: false,
    };
    setReasoningPayload(payload, opts.reasoningEffort);
    const data = await this.postJSON("/responses/compact", payload);
    const output = data.output;
    if (!Array.isArray(output)) {
      throw new ChatGPTOAuthError(
        "remote compact response missing output array",
      );
    }
    return REMOTE_COMPACTION_MARKER + "\n" + JSON.stringify(output);
  }

  private async collectResponseOutputItems(
    payload: Record<string, unknown>,
  ): Promise<Record<string, unknown>[]> {
    const outputItems: Record<string, unknown>[] = [];
    let completedOutputSeen = false;
    const seenKeys = new Set<string>();

    const appendItem = (item: Record<string, unknown>) => {
      const keyParts = [String(item.type ?? "")];
      const idField =
        typeof item.id === "string" && item.id
          ? item.id
          : typeof item.call_id === "string" && item.call_id
            ? item.call_id
            : null;
      if (idField) {
        keyParts.push(idField);
      } else {
        keyParts.push(JSON.stringify(item));
      }
      const key = keyParts.join("\x1f");
      if (seenKeys.has(key)) return;
      seenKeys.add(key);
      outputItems.push(item);
    };

    for await (const event of this.postSSE("/responses", payload)) {
      const typ = event.type;
      if (typ === "response.output_item.done") {
        const item = event.item;
        if (typeof item === "object" && item !== null) {
          appendItem(item as Record<string, unknown>);
        }
      } else if (typ === "response.failed") {
        throw new ChatGPTOAuthError(
          responseFailureMessage(event, "failed"),
        );
      } else if (typ === "response.incomplete") {
        throw new ChatGPTOAuthError(
          responseFailureMessage(event, "incomplete"),
        );
      } else if (typ === "response.completed") {
        const response = event.response;
        if (typeof response === "object" && response !== null) {
          const r = response as Record<string, unknown>;
          if (Array.isArray(r.output)) {
            completedOutputSeen = true;
            for (const item of r.output) {
              if (typeof item === "object" && item !== null) {
                appendItem(item as Record<string, unknown>);
              }
            }
          }
        }
      }
    }
    if (!outputItems.length && !completedOutputSeen) {
      throw new ChatGPTOAuthError(
        "ChatGPT OAuth response returned no output items",
      );
    }
    return outputItems;
  }

  private responsesPayload(
    messages: Message[],
    opts: ChatOptions,
  ): Record<string, unknown> {
    const [instructions, inputItems] =
      splitInstructionsAndInput(messages);
    if (!instructions) {
      throw new ChatGPTOAuthError(
        "ChatGPT OAuth Responses request requires system instructions",
      );
    }
    const payload: Record<string, unknown> = {
      model: opts.model || this.model,
      instructions,
      input: inputItems,
      tools: opts.tools
        ? opts.tools.map(toolSchemaToResponseDict)
        : [],
      tool_choice: opts.toolChoice || "auto",
      parallel_tool_calls: false,
      stream: true,
      store: false,
      include: [],
    };
    if (opts.promptCacheKey)
      payload.prompt_cache_key = opts.promptCacheKey;
    if (opts.stop != null)
      payload.stop = Array.isArray(opts.stop)
        ? opts.stop
        : [opts.stop];
    if (opts.previousResponseId != null)
      payload.previous_response_id = opts.previousResponseId;
    if (opts.serviceTier != null)
      payload.service_tier = opts.serviceTier;
    if (opts.text != null) payload.text = opts.text;
    if (opts.clientMetadata != null)
      payload.client_metadata = opts.clientMetadata;
    setReasoningPayload(payload, opts.reasoningEffort);
    return payload;
  }

  private getHeaders(): Record<string, string> {
    const token = loadTokenData(this.authJsonPath);
    const headers: Record<string, string> = {
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

// --- Helper functions (exported for testing) ---

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
  return typeof event === "object" &&
    event !== null &&
    !Array.isArray(event)
    ? event
    : null;
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
      for (let i = 0; i < parsed.length; i++) {
        if (typeof parsed[i] !== "object" || parsed[i] === null) {
          throw new ChatGPTOAuthError(
            `remote compaction marker item ${i} must be an object`,
          );
        }
        items.push(parsed[i]);
      }
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
      if (message.content) {
        items.push(messageItem("assistant", message.content));
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
    items.push(messageItem(role, message.content, message.images));
  }
  return items;
}

export function messageItem(
  role: string,
  content: string,
  images?: string[],
): Record<string, unknown> {
  const typ = role === "assistant" ? "output_text" : "input_text";
  const contentItems: Record<string, unknown>[] = [
    { type: typ, text: content || "" },
  ];
  if (images) {
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
  return {
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
    strict: false,
  };
}

export function setReasoningPayload(
  payload: Record<string, unknown>,
  reasoningEffort?: string,
): void {
  if (reasoningEffort == null) return;
  if (typeof reasoningEffort !== "string" || !reasoningEffort) {
    throw new ChatGPTOAuthError(
      "reasoning_effort must be a non-empty string when provided",
    );
  }
  const effort = reasoningEffort.toLowerCase();
  if (!REASONING_EFFORT_VALUES.has(effort)) {
    throw new ChatGPTOAuthError(
      "reasoning_effort must be one of: " +
        [...REASONING_EFFORT_VALUES].sort().join(", "),
    );
  }
  payload.reasoning = { effort };
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
  images: { image_url: string }[],
): Record<string, string>[] {
  const items: Record<string, string>[] = [];
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
    items.push({ type: "input_image", image_url: imageUrl });
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
  if (typeof tokenDetails === "object" && tokenDetails !== null) {
    const d = tokenDetails as Record<string, unknown>;
    if (typeof d.cached_tokens === "number")
      cachedTokens = d.cached_tokens;
  } else if (typeof v.cached_input_tokens === "number") {
    cachedTokens = v.cached_input_tokens;
  } else if (typeof v.cache_read_input_tokens === "number") {
    cachedTokens = v.cache_read_input_tokens;
  }
  return {
    prompt_tokens: prompt,
    completion_tokens: completion,
    total_tokens:
      typeof total === "number" ? total : prompt + completion,
    cached_tokens: cachedTokens,
  };
}

export { REMOTE_COMPACTION_MARKER };
