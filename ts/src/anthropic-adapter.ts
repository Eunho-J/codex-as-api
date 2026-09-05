import { isDeepStrictEqual } from "node:util";

import type {
  AssistantResponse,
  Message,
  ToolCall,
  ToolSchema,
  Usage,
} from "./messages.js";
import { MessageRole } from "./messages.js";
import { normalizeFinishReason } from "./protocol.js";
import {
  ChatGPTOAuthInvalidRequestError,
  ChatGPTOAuthProtocolError,
} from "./auth.js";
import { parseJsonStrict } from "./utf8-json.js";

// ---------------------------------------------------------------------------
// Request conversion: Anthropic → internal
// ---------------------------------------------------------------------------

export function anthropicRequestToInternal(opts: {
  model?: string;
  messages: Record<string, unknown>[];
  system?: string | Record<string, unknown>[];
  maxTokens?: number;
  tools?: Record<string, unknown>[];
  toolChoice?: Record<string, unknown>;
  stopSequences?: string[];
  thinking?: Record<string, unknown>;
  outputFormat?: unknown;
  outputConfig?: unknown;
}): {
  messages: Message[];
  tools: ToolSchema[] | null;
  toolChoice: string | Record<string, unknown> | null;
  parallelToolCalls: boolean | undefined;
  stop: string[] | null;
  reasoningEffort: string | null;
  text: Record<string, unknown> | null;
} {
  const internalMessages: Message[] = [];

  if (!Array.isArray(opts.messages)) {
    throw new ChatGPTOAuthInvalidRequestError("messages must be an array");
  }

  if (opts.system !== undefined) {
    const sysText = extractSystemText(opts.system);
    if (sysText) {
      internalMessages.push({ role: MessageRole.SYSTEM, content: sysText });
    }
  }

  for (const [messageIndex, msg] of opts.messages.entries()) {
    if (typeof msg !== "object" || msg === null || Array.isArray(msg)) {
      throw new ChatGPTOAuthInvalidRequestError(`message ${messageIndex} must be an object`);
    }
    assertAllowedFields(msg, ["role", "content"], `message ${messageIndex}`);
    const role = msg.role;
    const content = msg.content;
    if (role === "user") {
      convertUserMessage(content, internalMessages);
    } else if (role === "assistant") {
      convertAssistantMessage(content, internalMessages);
    } else {
      throw new ChatGPTOAuthInvalidRequestError(
        `message ${messageIndex} role must be one of: user, assistant`,
      );
    }
  }

  const internalTools = opts.tools ? convertTools(opts.tools) : null;
  const {
    toolChoice: internalToolChoice,
    parallelToolCalls,
  } = convertToolChoice(opts.toolChoice ?? null);
  const reasoningEffort = convertReasoningEffort(
    opts.thinking ?? null,
    opts.outputConfig ?? null,
    opts.maxTokens,
  );
  const text = anthropicOutputFormatToOpenAIText(
    resolveAnthropicOutputFormat(opts.outputFormat, opts.outputConfig),
  );

  return {
    messages: internalMessages,
    tools: internalTools,
    toolChoice: internalToolChoice,
    parallelToolCalls,
    stop: opts.stopSequences ?? null,
    reasoningEffort,
    text,
  };
}

function extractSystemText(
  system: string | Record<string, unknown>[],
): string {
  if (typeof system === "string") return system;
  if (!Array.isArray(system)) {
    throw new ChatGPTOAuthInvalidRequestError("system must be a string or an array of text blocks");
  }
  const parts: string[] = [];
  for (const [index, block] of system.entries()) {
    if (
      typeof block !== "object"
      || block === null
      || Array.isArray(block)
      || block.type !== "text"
      || typeof block.text !== "string"
    ) {
      throw new ChatGPTOAuthInvalidRequestError(`system block ${index} must be a text block`);
    }
    validateNullableOmittedField(block, "citations", `system block ${index}`);
    assertAllowedFields(block, ["type", "text", "citations"], `system block ${index}`);
    parts.push(block.text);
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
  if (!Array.isArray(content)) {
    throw new ChatGPTOAuthInvalidRequestError("user message content must be a string or array");
  }
  if (content.length === 0) {
    throw new ChatGPTOAuthInvalidRequestError("user message content array must not be empty");
  }
  const textParts: string[] = [];
  const imageUrls: string[] = [];
  for (const [blockIndex, block] of content.entries()) {
    if (typeof block !== "object" || block === null || Array.isArray(block)) {
      throw new ChatGPTOAuthInvalidRequestError(`user content block ${blockIndex} must be an object`);
    }
    const b = block as Record<string, unknown>;
    const blockType = b.type;
    if (blockType === "text") {
      validateNullableOmittedField(b, "citations", `user text block ${blockIndex}`);
      assertAllowedFields(b, ["type", "text", "citations"], `user text block ${blockIndex}`);
      if (typeof b.text !== "string") {
        throw new ChatGPTOAuthInvalidRequestError(`user text block ${blockIndex} requires string text`);
      }
      textParts.push(b.text);
    } else if (blockType === "tool_result") {
      assertAllowedFields(
        b,
        ["type", "tool_use_id", "content", "is_error"],
        `tool_result block ${blockIndex}`,
      );
      if (textParts.length || imageUrls.length) {
        out.push({ role: MessageRole.USER, content: textParts.join(""), images: [...imageUrls] });
        textParts.length = 0;
        imageUrls.length = 0;
      }
      if (typeof b.tool_use_id !== "string") {
        throw new ChatGPTOAuthInvalidRequestError(`tool_result block ${blockIndex} requires string tool_use_id`);
      }
      const toolUseId = b.tool_use_id;
      let resultContent: unknown = Object.hasOwn(b, "content") ? b.content : "";
      const toolResultImages: string[] = [];
      if (Array.isArray(resultContent)) {
        const textPieces: string[] = [];
        for (const [partIndex, p] of resultContent.entries()) {
          if (typeof p !== "object" || p === null || Array.isArray(p)) {
            throw new ChatGPTOAuthInvalidRequestError(
              `tool_result block ${blockIndex} content ${partIndex} must be an object`,
            );
          }
          if (p.type === "text") {
            validateNullableOmittedField(
              p,
              "citations",
              `tool_result block ${blockIndex} content ${partIndex}`,
            );
            assertAllowedFields(
              p,
              ["type", "text", "citations"],
              `tool_result block ${blockIndex} content ${partIndex}`,
            );
            if (typeof p.text !== "string") {
              throw new ChatGPTOAuthInvalidRequestError(
                `tool_result block ${blockIndex} text content ${partIndex} requires string text`,
              );
            }
            textPieces.push(p.text);
          } else if (p.type === "image") {
            assertAllowedFields(
              p,
              ["type", "source"],
              `tool_result block ${blockIndex} content ${partIndex}`,
            );
            const imageUrl = anthropicImageSourceUrl(p.source);
            toolResultImages.push(imageUrl);
          } else {
            throw new ChatGPTOAuthInvalidRequestError(
              `tool_result block ${blockIndex} has unsupported content type ${String(p.type)}`,
            );
          }
        }
        resultContent = textPieces.join("");
      } else if (typeof resultContent !== "string") {
        throw new ChatGPTOAuthInvalidRequestError(`tool_result block ${blockIndex} content must be a string or array`);
      }
      if (Object.hasOwn(b, "is_error") && typeof b.is_error !== "boolean") {
        throw new ChatGPTOAuthInvalidRequestError(`tool_result block ${blockIndex} is_error must be a boolean`);
      }
      if (b.is_error === true) {
        resultContent = `[tool_error]\n${resultContent}`;
      }
      out.push({
        role: MessageRole.TOOL,
        content: resultContent as string,
        tool_call_id: toolUseId,
      });
      if (toolResultImages.length) {
        out.push({ role: MessageRole.USER, content: "", images: toolResultImages });
      }
    } else if (blockType === "image") {
      assertAllowedFields(b, ["type", "source"], `user image block ${blockIndex}`);
      const imageUrl = anthropicImageSourceUrl(b.source);
      imageUrls.push(imageUrl);
    } else {
      throw new ChatGPTOAuthInvalidRequestError(
        `user content block ${blockIndex} has unsupported type ${String(blockType)}`,
      );
    }
  }
  if (textParts.length || imageUrls.length) {
    out.push({ role: MessageRole.USER, content: textParts.join(""), images: [...imageUrls] });
  }
}

function anthropicImageSourceUrl(sourceValue: unknown): string {
  if (
    typeof sourceValue !== "object"
    || sourceValue === null
    || Array.isArray(sourceValue)
  ) {
    throw new ChatGPTOAuthInvalidRequestError("Anthropic image source must be an object");
  }
  const source = sourceValue as Record<string, unknown>;
  if (source.type === "base64") {
    assertAllowedFields(source, ["type", "media_type", "data"], "Anthropic base64 image source");
    const mediaType = source.media_type;
    if (
      typeof mediaType !== "string"
      || !["image/jpeg", "image/png", "image/gif", "image/webp"].includes(mediaType)
    ) {
      throw new ChatGPTOAuthInvalidRequestError(
        "Anthropic base64 image source media_type must be one of: image/jpeg, image/png, image/gif, image/webp",
      );
    }
    if (typeof source.data !== "string" || source.data.length === 0) {
      throw new ChatGPTOAuthInvalidRequestError("Anthropic base64 image source requires non-empty data");
    }
    return `data:${mediaType};base64,${source.data}`;
  }
  if (source.type === "url") {
    assertAllowedFields(source, ["type", "url"], "Anthropic URL image source");
    if (typeof source.url !== "string" || source.url.length === 0) {
      throw new ChatGPTOAuthInvalidRequestError("Anthropic URL image source requires a non-empty url");
    }
    return source.url;
  }
  throw new ChatGPTOAuthInvalidRequestError(
    "Anthropic image source type must be one of: base64, url",
  );
}

function convertAssistantMessage(
  content: unknown,
  out: Message[],
): void {
  if (typeof content === "string") {
    out.push({ role: MessageRole.ASSISTANT, content });
    return;
  }
  if (!Array.isArray(content)) {
    throw new ChatGPTOAuthInvalidRequestError("assistant message content must be a string or array");
  }
  if (content.length === 0) {
    out.push({ role: MessageRole.ASSISTANT, content: "", tool_calls: [] });
    return;
  }
  const textParts: string[] = [];
  const toolCalls: ToolCall[] = [];
  const toolCallIds = new Set<string>();
  for (const [blockIndex, block] of content.entries()) {
    if (typeof block !== "object" || block === null || Array.isArray(block)) {
      throw new ChatGPTOAuthInvalidRequestError(`assistant content block ${blockIndex} must be an object`);
    }
    const b = block as Record<string, unknown>;
    const blockType = b.type;
    if (blockType === "text") {
      validateNullableOmittedField(b, "citations", `assistant text block ${blockIndex}`);
      assertAllowedFields(b, ["type", "text", "citations"], `assistant text block ${blockIndex}`);
      if (typeof b.text !== "string") {
        throw new ChatGPTOAuthInvalidRequestError(`assistant text block ${blockIndex} requires string text`);
      }
      textParts.push(b.text);
    } else if (blockType === "tool_use") {
      assertAllowedFields(
        b,
        ["type", "id", "name", "input", "caller"],
        `assistant tool_use block ${blockIndex}`,
      );
      if (Object.hasOwn(b, "caller")) {
        const caller = requireRecord(
          b.caller,
          `assistant tool_use block ${blockIndex} caller`,
        );
        assertAllowedFields(
          caller,
          ["type"],
          `assistant tool_use block ${blockIndex} caller`,
        );
        if (caller.type !== "direct") {
          throw new ChatGPTOAuthInvalidRequestError(
            `assistant tool_use block ${blockIndex} caller.type must be direct`,
          );
        }
      }
      if (typeof b.id !== "string") {
        throw new ChatGPTOAuthInvalidRequestError(`assistant tool_use block ${blockIndex} requires id`);
      }
      if (typeof b.name !== "string") {
        throw new ChatGPTOAuthInvalidRequestError(`assistant tool_use block ${blockIndex} requires name`);
      }
      if (typeof b.input !== "object" || b.input === null || Array.isArray(b.input)) {
        throw new ChatGPTOAuthInvalidRequestError(`assistant tool_use block ${blockIndex} input must be an object`);
      }
      if (toolCallIds.has(b.id)) {
        throw new ChatGPTOAuthInvalidRequestError(
          `assistant tool_use blocks contain duplicate id ${JSON.stringify(b.id)}`,
        );
      }
      toolCallIds.add(b.id);
      toolCalls.push({
        id: b.id,
        name: b.name,
        arguments: JSON.stringify(b.input),
      });
    } else if (blockType === "thinking") {
      throw new ChatGPTOAuthInvalidRequestError(
        "assistant thinking history cannot be represented by the Codex OAuth transport",
      );
    } else if (blockType === "redacted_thinking") {
      throw new ChatGPTOAuthInvalidRequestError(
        "redacted_thinking cannot be represented by the Codex OAuth transport",
      );
    } else if (blockType === "server_tool_use" || blockType === "web_search_tool_result") {
      throw new ChatGPTOAuthInvalidRequestError(
        `${String(blockType)} history cannot be represented by the Codex OAuth transport`,
      );
    } else {
      throw new ChatGPTOAuthInvalidRequestError(
        `assistant content block ${blockIndex} has unsupported type ${String(blockType)}`,
      );
    }
  }
  if (textParts.length === 0 && toolCalls.length === 0) {
    throw new ChatGPTOAuthInvalidRequestError(
      "assistant message content array must not be empty",
    );
  }
  const msg: Message = {
    role: MessageRole.ASSISTANT,
    content: textParts.join(""),
    tool_calls: toolCalls.length ? toolCalls : [],
  };
  out.push(msg);
}

function convertTools(tools: Record<string, unknown>[]): ToolSchema[] {
  const result: ToolSchema[] = [];
  const toolNames = new Set<string>();
  for (const [index, tool] of tools.entries()) {
    if (typeof tool !== "object" || tool === null || Array.isArray(tool)) {
      throw new ChatGPTOAuthInvalidRequestError(`tool ${index} must be an object`);
    }
    for (const field of ["defer_loading", "eager_input_streaming"] as const) {
      const value = tool[field];
      if ((field === "defer_loading" && Object.hasOwn(tool, field)) || value != null) {
        throw new ChatGPTOAuthInvalidRequestError(`tool ${index} ${field} is not supported`);
      }
    }
    if (tool.type === "programmatic_tool_calling") {
      throw new ChatGPTOAuthInvalidRequestError(
        "programmatic_tool_calling tools are not supported by this compatibility API",
      );
    }
    if (Object.hasOwn(tool, "allowed_callers")) {
      throw new ChatGPTOAuthInvalidRequestError(`tool ${index} allowed_callers is not supported`);
    }
    if (Object.hasOwn(tool, "output_schema")) {
      throw new ChatGPTOAuthInvalidRequestError(`tool ${index} output_schema is not supported`);
    }
    const webSearch = isAnthropicWebSearchTool(tool);
    if (webSearch) {
      throw new ChatGPTOAuthInvalidRequestError(
        "Anthropic hosted web_search cannot be represented losslessly by this facade",
      );
    }
    if (Object.hasOwn(tool, "strict") && typeof tool.strict !== "boolean") {
      throw new ChatGPTOAuthInvalidRequestError(
        `tool ${index} strict must be a boolean`,
      );
    }
    assertAllowedFields(
      tool,
      [
        "type",
        "name",
        "description",
        "input_schema",
        "strict",
        "defer_loading",
        "eager_input_streaming",
        "allowed_callers",
        "output_schema",
      ],
      `tool ${index}`,
    );
    const name = tool.name;
    if (typeof name !== "string" || name.length === 0) {
      throw new ChatGPTOAuthInvalidRequestError(`tool ${index} name must be a non-empty string`);
    }
    if (toolNames.has(name)) {
      throw new ChatGPTOAuthInvalidRequestError(
        `tools contains duplicate name ${JSON.stringify(name)}`,
      );
    }
    toolNames.add(name);
    if (tool.type !== undefined && tool.type !== null && tool.type !== "custom") {
      throw new ChatGPTOAuthInvalidRequestError(
        `tool ${index} type must be custom or null`,
      );
    }
    result.push({
      name,
      parameters: requireRecord(tool.input_schema, `tool ${index} input_schema`),
      ...(!Object.hasOwn(tool, "description")
        ? {}
        : { description: requireString(tool.description, `tool ${index} description`) }),
      ...(!Object.hasOwn(tool, "strict") ? {} : { strict: tool.strict as boolean }),
    });
  }
  return result;
}

function isAnthropicWebSearchTool(tool: Record<string, unknown>): boolean {
  return tool.name === "web_search" &&
    typeof tool.type === "string" &&
    ["web_search", "web_search_20250305", "web_search_20260209"].includes(tool.type);
}

function convertToolChoice(
  tc: Record<string, unknown> | null,
): {
  toolChoice: string | Record<string, unknown> | null;
  parallelToolCalls: boolean | undefined;
} {
  if (tc === null) {
    return { toolChoice: null, parallelToolCalls: undefined };
  }
  const tcType = tc.type;
  assertAllowedFields(
    tc,
    tcType === "tool"
      ? ["type", "name", "disable_parallel_tool_use"]
      : tcType === "none"
      ? ["type"]
      : ["type", "disable_parallel_tool_use"],
    "tool_choice",
  );
  const hasParallelControl = Object.hasOwn(tc, "disable_parallel_tool_use");
  if (hasParallelControl && typeof tc.disable_parallel_tool_use !== "boolean") {
    throw new ChatGPTOAuthInvalidRequestError(
      "tool_choice.disable_parallel_tool_use must be a boolean when provided",
    );
  }
  const parallelToolCalls = !hasParallelControl
    ? undefined
    : !tc.disable_parallel_tool_use;
  if (tcType === "auto") return { toolChoice: "auto", parallelToolCalls };
  if (tcType === "any") return { toolChoice: "required", parallelToolCalls };
  if (tcType === "tool") {
    if (typeof tc.name !== "string" || tc.name.length === 0) {
      throw new ChatGPTOAuthInvalidRequestError("tool_choice type tool requires a non-empty name");
    }
    if (tc.name === "web_search") {
      throw new ChatGPTOAuthInvalidRequestError(
        "Anthropic hosted web_search cannot be represented losslessly by this facade",
      );
    }
    return {
      toolChoice: { type: "function", name: tc.name },
      parallelToolCalls,
    };
  }
  if (tcType === "none") return { toolChoice: "none", parallelToolCalls };
  throw new ChatGPTOAuthInvalidRequestError("tool_choice.type must be one of: auto, any, tool, none");
}

function convertThinking(
  thinking: Record<string, unknown> | null,
  maxTokens: number | undefined,
): string | null {
  if (thinking === null) return null;
  if (thinking.type === "enabled") {
    assertAllowedFields(thinking, ["type", "budget_tokens", "display"], "thinking");
    if (thinking.display != null && thinking.display !== "omitted") {
      throw new ChatGPTOAuthInvalidRequestError(
        "thinking.display must be omitted when provided",
      );
    }
    if (
      typeof thinking.budget_tokens !== "number"
      || !Number.isSafeInteger(thinking.budget_tokens)
      || thinking.budget_tokens < 1024
    ) {
      throw new ChatGPTOAuthInvalidRequestError(
        "thinking.budget_tokens must be an integer greater than or equal to 1024",
      );
    }
    if (maxTokens != null && thinking.budget_tokens >= maxTokens) {
      throw new ChatGPTOAuthInvalidRequestError(
        "thinking.budget_tokens must be less than max_tokens",
      );
    }
    return "high";
  }
  if (thinking.type === "adaptive") {
    assertAllowedFields(thinking, ["type", "display"], "thinking");
    if (thinking.display != null && thinking.display !== "omitted") {
      throw new ChatGPTOAuthInvalidRequestError(
        "thinking.display must be omitted when provided",
      );
    }
    return "medium";
  }
  if (thinking.type === "disabled") {
    assertAllowedFields(thinking, ["type"], "thinking");
    return "none";
  }
  throw new ChatGPTOAuthInvalidRequestError("thinking.type must be one of: enabled, adaptive, disabled");
}

function requireRecord(value: unknown, field: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be an object`);
  }
  return value as Record<string, unknown>;
}

function requireString(value: unknown, field: string): string {
  if (typeof value !== "string") {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be a string`);
  }
  return value;
}

function assertAllowedFields(
  value: Record<string, unknown>,
  allowed: readonly string[],
  field: string,
): void {
  const allowedSet = new Set(allowed);
  const unknown = Object.keys(value).find((key) => !allowedSet.has(key));
  if (unknown != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      `${field} does not support field ${JSON.stringify(unknown)}`,
    );
  }
}

function validateNullableOmittedField(
  value: Record<string, unknown>,
  field: string,
  location: string,
): void {
  if (Object.hasOwn(value, field) && value[field] !== null) {
    throw new ChatGPTOAuthInvalidRequestError(
      `${location}.${field} cannot be preserved by this facade`,
    );
  }
}

function convertReasoningEffort(
  thinking: Record<string, unknown> | null,
  outputConfigValue: unknown,
  maxTokens: number | undefined,
): string | null {
  const thinkingEffort = convertThinking(thinking, maxTokens);
  if (outputConfigValue == null) return thinkingEffort;
  if (
    typeof outputConfigValue !== "object"
    || Array.isArray(outputConfigValue)
  ) {
    throw new ChatGPTOAuthInvalidRequestError("output_config must be an object");
  }

  const outputConfig = outputConfigValue as Record<string, unknown>;
  for (const key of Object.keys(outputConfig)) {
    if (!["effort", "format", "task_budget"].includes(key)) {
      throw new ChatGPTOAuthInvalidRequestError(`output_config.${key} is not supported`);
    }
  }
  if (outputConfig.task_budget != null) {
    throw new ChatGPTOAuthInvalidRequestError(
      "output_config.task_budget is not supported by the Codex OAuth backend",
    );
  }
  if (outputConfig.effort == null) return thinkingEffort;
  if (
    typeof outputConfig.effort !== "string"
    || !["low", "medium", "high", "xhigh", "max"].includes(
      outputConfig.effort,
    )
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "output_config.effort must be one of: low, medium, high, xhigh, max",
    );
  }
  if (thinkingEffort === "none") {
    return "none";
  }
  return outputConfig.effort;
}

export function anthropicOutputFormatToOpenAIText(
  outputFormat: Record<string, unknown> | null,
): Record<string, unknown> | null {
  if (outputFormat === null) return null;
  validateAnthropicOutputFormat("output_format", outputFormat);
  const type = outputFormat.type;
  if (type === "json_schema") {
    const schema = outputFormat.schema as Record<string, unknown>;
    const name = typeof outputFormat.name === "string"
      ? outputFormat.name
      : "codex_output_schema";
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
  throw new ChatGPTOAuthInvalidRequestError(
    "output_format.type must be one of: json_schema, json_object",
  );
}

function resolveAnthropicOutputFormat(
  outputFormatValue: unknown,
  outputConfigValue: unknown,
): Record<string, unknown> | null {
  const outputFormat = outputFormatRecord(outputFormatValue, "output_format");
  let nestedFormat: Record<string, unknown> | null = null;
  if (outputConfigValue != null) {
    if (
      typeof outputConfigValue !== "object"
      || Array.isArray(outputConfigValue)
    ) {
      throw new ChatGPTOAuthInvalidRequestError("output_config must be an object");
    }
    nestedFormat = outputFormatRecord(
      (outputConfigValue as Record<string, unknown>).format,
      "output_config.format",
    );
  }
  if (
    outputFormat !== null
    && nestedFormat !== null
    && !isDeepStrictEqual(outputFormat, nestedFormat)
  ) {
    throw new ChatGPTOAuthInvalidRequestError("output_format conflicts with output_config.format");
  }
  const selected = outputFormat ?? nestedFormat;
  if (selected === null) return null;
  validateAnthropicOutputFormat(
    outputFormat !== null ? "output_format" : "output_config.format",
    selected,
  );
  return selected;
}

function outputFormatRecord(
  value: unknown,
  field: string,
): Record<string, unknown> | null {
  if (value == null) return null;
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new ChatGPTOAuthInvalidRequestError(`${field} must be an object`);
  }
  return value as Record<string, unknown>;
}

function validateAnthropicOutputFormat(
  field: string,
  outputFormat: Record<string, unknown>,
): void {
  const type = outputFormat.type;
  if (typeof type !== "string") {
    throw new ChatGPTOAuthInvalidRequestError(`${field}.type must be a string`);
  }

  let allowedFields: ReadonlySet<string>;
  if (type === "json_object") {
    allowedFields = new Set(["type"]);
  } else if (type === "json_schema") {
    allowedFields = new Set([
      "type",
      "schema",
      "name",
      "description",
      "strict",
    ]);
  } else {
    throw new ChatGPTOAuthInvalidRequestError(`${field}.type must be one of: json_object, json_schema`);
  }

  const unknownField = Object.keys(outputFormat).find(
    (key) => !allowedFields.has(key),
  );
  if (unknownField !== undefined) {
    throw new ChatGPTOAuthInvalidRequestError(`${field}.${unknownField} is not supported`);
  }

  if (type === "json_schema") {
    const schema = outputFormat.schema;
    if (typeof schema !== "object" || schema === null || Array.isArray(schema)) {
      throw new ChatGPTOAuthInvalidRequestError(`${field}.schema must be an object`);
    }
    if (
      outputFormat.name !== undefined
      && (typeof outputFormat.name !== "string" || !/^[A-Za-z0-9_-]{1,64}$/.test(outputFormat.name))
    ) {
      throw new ChatGPTOAuthInvalidRequestError(
        `${field}.name must contain only ASCII letters, digits, underscores, or hyphens and be at most 64 characters`,
      );
    }
  }
  if (
    Object.hasOwn(outputFormat, "description")
    && typeof outputFormat.description !== "string"
  ) {
    throw new ChatGPTOAuthInvalidRequestError(`${field}.description must be a string`);
  }
  if (outputFormat.strict != null && typeof outputFormat.strict !== "boolean") {
    throw new ChatGPTOAuthInvalidRequestError(`${field}.strict must be a boolean`);
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

  rejectUnrepresentableResponseEvents(response.raw);

  if (response.content) {
    content.push({ type: "text", text: response.content, citations: null });
  }

  for (const tc of response.tool_calls) {
    const input = parseAnthropicToolArguments(tc.arguments);
    content.push({
      type: "tool_use",
      id: tc.id,
      name: tc.name,
      input,
      caller: { type: "direct" },
    });
  }

  const stopReason = mapStopReason(response.finish_reason, response.tool_calls.length > 0);

  if (response.usage == null) {
    throw new ChatGPTOAuthProtocolError("provider response requires authoritative usage");
  }
  const usageDict = mergeUsageExtensions(
    usageFromInternalResponse(response.usage),
    response.raw,
  );

  return {
    id: requestId,
    type: "message",
    role: "assistant",
    model,
    container: null,
    content,
    context_management: null,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: usageDict,
    ...(response.reasoning_content == null
      ? {}
      : { codex_reasoning: response.reasoning_content }),
  };
}

function parseAnthropicToolArguments(value: unknown): Record<string, unknown> {
  if (typeof value !== "string") {
    throw new ChatGPTOAuthProtocolError("tool call arguments must be a JSON object string");
  }
  let parsed: unknown;
  try {
    parsed = parseJsonStrict(value);
  } catch (error) {
    throw new ChatGPTOAuthProtocolError(
      `tool call arguments must contain valid JSON: ${String(error)}`,
    );
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new ChatGPTOAuthProtocolError("tool call arguments JSON must be an object");
  }
  return parsed as Record<string, unknown>;
}

function rejectUnrepresentableResponseEvents(raw: Record<string, unknown> | null): void {
  const events = rawEvents(raw);
  for (const [index, event] of events.entries()) {
    if (typeof event !== "object" || event === null || Array.isArray(event)) {
      throw new ChatGPTOAuthProtocolError(`provider raw event ${index} must be an object`);
    }
    const e = event as Record<string, unknown>;
    if (e.type !== "web_search_call") continue;
    throw new ChatGPTOAuthProtocolError(
      "provider web_search_call output cannot be represented losslessly by the Anthropic facade",
    );
  }
}

function mergeUsageExtensions(
  usage: Record<string, unknown>,
  raw: Record<string, unknown> | null,
): Record<string, unknown> {
  const events = rawEvents(raw);
  for (const [index, event] of events.entries()) {
    if (typeof event !== "object" || event === null || Array.isArray(event)) {
      throw new ChatGPTOAuthProtocolError(`provider raw event ${index} must be an object`);
    }
    const e = event as Record<string, unknown>;
    if (e.type !== "finish") continue;
    const rawUsage = e.usage;
    if (rawUsage == null) continue;
    if (typeof rawUsage !== "object" || Array.isArray(rawUsage)) {
      throw new ChatGPTOAuthProtocolError("provider finish event usage must be an object");
    }
    const u = rawUsage as Record<string, unknown>;
    for (const [key, required] of [
      ["cache_creation", ["ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"]],
      ["server_tool_use", ["web_search_requests", "web_fetch_requests"]],
    ] as const) {
      if (u[key] != null) {
        usage[key] = requireUsageBreakdown(u[key], key, required);
      }
    }
    if (u.service_tier != null) {
      if (
        typeof u.service_tier !== "string"
        || !["standard", "priority", "batch"].includes(u.service_tier)
      ) {
        throw new ChatGPTOAuthProtocolError(
          "provider usage service_tier must be standard, priority, batch, or null",
        );
      }
      usage.service_tier = u.service_tier;
    }
    return usage;
  }
  return usage;
}

function rawEvents(raw: Record<string, unknown> | null): unknown[] {
  if (raw == null) return [];
  if (typeof raw !== "object" || Array.isArray(raw)) {
    throw new ChatGPTOAuthProtocolError("provider raw response must be an object");
  }
  if (!Array.isArray(raw.events)) {
    throw new ChatGPTOAuthProtocolError("provider raw response requires an events array");
  }
  for (const [index, event] of raw.events.entries()) {
    if (typeof event !== "object" || event === null || Array.isArray(event)) {
      throw new ChatGPTOAuthProtocolError(`provider raw event ${index} must be an object`);
    }
    const type = (event as Record<string, unknown>).type;
    if (typeof type !== "string" || type.length === 0) {
      throw new ChatGPTOAuthProtocolError(
        `provider raw event ${index} requires a non-empty string type`,
      );
    }
  }
  return raw.events;
}

function mapStopReason(value: unknown, hasToolCalls: boolean): string {
  const finishReason = normalizeFinishReason(value);
  if (finishReason == null) {
    throw new ChatGPTOAuthProtocolError("provider response requires a non-null finish_reason");
  }
  if (finishReason === "tool_calls") {
    if (!hasToolCalls) {
      throw new ChatGPTOAuthProtocolError(
        "provider finish_reason tool_calls requires at least one tool call",
      );
    }
    return "tool_use";
  }
  if (hasToolCalls) {
    throw new ChatGPTOAuthProtocolError(
      "provider finish_reason stop conflicts with emitted tool calls",
    );
  }
  return "end_turn";
}

// ---------------------------------------------------------------------------
// Streaming adapter: provider events → Anthropic SSE
// ---------------------------------------------------------------------------

export async function* anthropicStreamAdapter(
  eventIterator: AsyncIterable<Record<string, unknown>>,
  model: string,
  requestId: string,
): AsyncGenerator<string> {
  yield messageStartSse(model, requestId);
  for await (const chunk of renderAnthropicStreamEvents(eventIterator)) {
    yield chunk;
  }
}

function messageStartSse(
  model: string,
  requestId: string,
): string {
  return sse("message_start", {
    type: "message_start",
    message: {
      id: requestId,
      type: "message",
      role: "assistant",
      model,
      container: null,
      content: [],
      context_management: null,
      stop_reason: null,
      stop_sequence: null,
    },
  });
}

function usageFromInternalResponse(usage: Usage): Record<string, unknown> {
  const inputTokens = requireUsageNumber(usage.prompt_tokens, "prompt_tokens");
  const outputTokens = requireUsageNumber(usage.completion_tokens, "completion_tokens");
  const totalTokens = requireUsageNumber(usage.total_tokens, "total_tokens");
  if (totalTokens !== inputTokens + outputTokens) {
    throw new ChatGPTOAuthProtocolError(
      "upstream response usage total_tokens must equal prompt_tokens plus completion_tokens",
    );
  }
  const result: Record<string, unknown> = {
    cache_creation: null,
    cache_creation_input_tokens: usage.cache_write_tokens == null
      ? null
      : requireUsageNumber(usage.cache_write_tokens, "cache_write_tokens"),
    cache_read_input_tokens: usage.cached_tokens == null
      ? null
      : requireUsageNumber(usage.cached_tokens, "cached_tokens"),
    inference_geo: null,
    input_tokens: inputTokens,
    iterations: null,
    output_tokens: outputTokens,
    server_tool_use: null,
    service_tier: null,
    speed: null,
  };
  return result;
}

function usageFromProviderEvent(usageEvent: unknown): Record<string, unknown> {
  if (usageEvent == null) {
    throw new ChatGPTOAuthProtocolError("finish event requires authoritative usage");
  }
  if (typeof usageEvent !== "object" || Array.isArray(usageEvent)) {
    throw new ChatGPTOAuthProtocolError("finish event usage must be an object");
  }
  const u = usageEvent as Record<string, unknown>;
  const unsupportedAlias = [
    "prompt_tokens",
    "completion_tokens",
    "prompt_tokens_details",
    "cached_input_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
  ].find((field) => Object.hasOwn(u, field));
  if (unsupportedAlias != null) {
    throw new ChatGPTOAuthProtocolError(
      `finish event usage does not support Chat Completions field ${unsupportedAlias}`,
    );
  }
  const inputTokens = requireUsageNumber(u.input_tokens, "input_tokens");
  const outputTokens = requireUsageNumber(u.output_tokens, "output_tokens");
  const totalTokens = requireUsageNumber(u.total_tokens, "total_tokens");
  if (totalTokens !== inputTokens + outputTokens) {
    throw new ChatGPTOAuthProtocolError(
      "finish event usage total_tokens must equal input_tokens plus output_tokens",
    );
  }
  const tokenDetails = u.input_tokens_details;
  const result: Record<string, unknown> = {
    cache_creation_input_tokens: null,
    cache_read_input_tokens: null,
    input_tokens: inputTokens,
    iterations: null,
    output_tokens: outputTokens,
    server_tool_use: null,
  };
  if (tokenDetails != null) {
    if (typeof tokenDetails !== "object" || Array.isArray(tokenDetails)) {
      throw new ChatGPTOAuthProtocolError("finish event usage input token details must be an object");
    }
    const details = tokenDetails as Record<string, unknown>;
    result.cache_read_input_tokens = requireUsageNumber(
      details.cached_tokens,
      "input_tokens_details.cached_tokens",
    );
    const cacheWriteValue = details.cache_write_tokens;
    if (cacheWriteValue != null) {
      result.cache_creation_input_tokens = requireUsageNumber(
        cacheWriteValue,
        "input_tokens_details.cache_write_tokens",
      );
    }
  }
  if (u.server_tool_use != null) {
    result.server_tool_use = requireUsageBreakdown(
      u.server_tool_use,
      "server_tool_use",
      ["web_search_requests", "web_fetch_requests"],
    );
  }
  if (u.cache_creation != null) {
    requireUsageBreakdown(
      u.cache_creation,
      "cache_creation",
      ["ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"],
    );
  }
  if (u.service_tier != null) {
    if (
      typeof u.service_tier !== "string"
      || !["standard", "priority", "batch"].includes(u.service_tier)
    ) {
      throw new ChatGPTOAuthProtocolError(
        "finish event usage service_tier must be standard, priority, batch, or null",
      );
    }
  }
  return result;
}

function requireUsageBreakdown(
  value: unknown,
  field: string,
  allowed: readonly string[],
): Record<string, number> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ChatGPTOAuthProtocolError(`finish event usage ${field} must be an object`);
  }
  const result: Record<string, number> = {};
  for (const [key, count] of Object.entries(value as Record<string, unknown>)) {
    if (!allowed.includes(key)) {
      throw new ChatGPTOAuthProtocolError(
        `finish event usage ${field} contains an unsupported field`,
      );
    }
    result[key] = requireUsageNumber(count, `${field}.${key}`);
  }
  const missing = allowed.filter((key) => !Object.hasOwn(result, key));
  if (missing.length > 0) {
    throw new ChatGPTOAuthProtocolError(
      `finish event usage ${field} is missing required fields: ${missing.join(", ")}`,
    );
  }
  return result;
}

function requireUsageNumber(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) {
    throw new ChatGPTOAuthProtocolError(`finish event usage ${field} must be a non-negative integer`);
  }
  return value;
}

async function* renderAnthropicStreamEvents(
  events: AsyncIterable<Record<string, unknown>>,
): AsyncGenerator<string> {
  let blockIndex = 0;
  let currentBlock: "text" | "tool_use" | null = null;
  let sawFinish = false;
  let hasToolCalls = false;

  for await (const event of events) {
    if (sawFinish) {
      throw new ChatGPTOAuthProtocolError("provider emitted an event after finish");
    }
    const typ = event.type;

    if (typ === "reasoning_delta" || typ === "reasoning_raw_delta") {
      if (typeof event.text !== "string") {
        throw new ChatGPTOAuthProtocolError("reasoning event text must be a string");
      }
      yield sse("codex_reasoning_delta", {
        type: "codex_reasoning_delta",
        delta: event.text,
      });
      continue;
    } else if (typ === "reasoning_section_break") {
      continue;
    } else if (typ === "content") {
      if (typeof event.text !== "string") {
        throw new ChatGPTOAuthProtocolError("content event text must be a string");
      }
      const text = event.text;
      if (currentBlock !== "text") {
        if (currentBlock !== null) {
          yield sse("content_block_stop", { type: "content_block_stop", index: blockIndex });
          blockIndex++;
        }
        yield sse("content_block_start", {
          type: "content_block_start",
          index: blockIndex,
          content_block: { type: "text", text: "", citations: null },
        });
        currentBlock = "text";
      }
      yield sse("content_block_delta", {
        type: "content_block_delta",
        index: blockIndex,
        delta: { type: "text_delta", text },
      });
    } else if (typ === "tool_call") {
      hasToolCalls = true;
      if (currentBlock !== null) {
        yield sse("content_block_stop", { type: "content_block_stop", index: blockIndex });
        blockIndex++;
      }
      if (typeof event.id !== "string") {
        throw new ChatGPTOAuthProtocolError("tool_call event requires a string id");
      }
      if (typeof event.name !== "string") {
        throw new ChatGPTOAuthProtocolError("tool_call event requires a string name");
      }
      parseAnthropicToolArguments(event.arguments);
      const toolId = event.id;
      const toolName = event.name;
      const toolArgs = event.arguments as string;
      yield sse("content_block_start", {
        type: "content_block_start",
        index: blockIndex,
        content_block: {
          type: "tool_use",
          id: toolId,
          name: toolName,
          input: {},
          caller: { type: "direct" },
        },
      });
      yield sse("content_block_delta", {
        type: "content_block_delta",
        index: blockIndex,
        delta: { type: "input_json_delta", partial_json: toolArgs },
      });
      yield sse("content_block_stop", { type: "content_block_stop", index: blockIndex });
      blockIndex++;
      currentBlock = null;
    } else if (typ === "web_search_call") {
      throw new ChatGPTOAuthProtocolError(
        "provider web_search_call output cannot be represented losslessly by the Anthropic facade",
      );
    } else if (typ === "finish") {
      if (currentBlock !== null) {
        yield sse("content_block_stop", { type: "content_block_stop", index: blockIndex });
        currentBlock = null;
      }

      const stopReason = mapStopReason(event.finish_reason, hasToolCalls);
      sawFinish = true;

      const usage = usageFromProviderEvent(event.usage);
      yield sse("message_delta", {
        type: "message_delta",
        context_management: null,
        delta: { container: null, stop_reason: stopReason, stop_sequence: null },
        usage,
      });
      yield sse("message_stop", { type: "message_stop" });
      return;
    } else {
      throw new ChatGPTOAuthProtocolError(
        "provider emitted an unsupported event type",
      );
    }
  }
  if (!sawFinish) {
    throw new ChatGPTOAuthProtocolError("provider stream ended before finish");
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
