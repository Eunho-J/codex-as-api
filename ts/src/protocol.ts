import { ChatGPTOAuthProtocolError } from "./auth.js";
import type { FinishReason } from "./messages.js";

export function normalizeFinishReason(value: unknown): FinishReason {
  if (value === null || value === "stop" || value === "tool_calls") return value;
  throw new ChatGPTOAuthProtocolError(
    "finish_reason must be null, stop, or tool_calls",
  );
}

export function normalizeStreamContent(content: unknown): string {
  if (content === null || content === undefined) {
    return "";
  }
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    const parts: string[] = [];
    for (const [index, item] of content.entries()) {
      if (typeof item !== "object" || item === null || Array.isArray(item)) {
        throw new ChatGPTOAuthProtocolError(`stream content item ${index} must be an object`);
      }
      const text = (item as Record<string, unknown>).text;
      if (typeof text !== "string") {
        throw new ChatGPTOAuthProtocolError(`stream content item ${index} text must be a string`);
      }
      parts.push(text);
    }
    return parts.join("");
  }
  throw new ChatGPTOAuthProtocolError("stream content must be a string, array, or null");
}

export function responseFailureMessage(event: Record<string, unknown>, status: string): string {
  const response = event["response"];
  let error: unknown = event["error"];
  let incompleteDetails: unknown = event["incomplete_details"];
  if (response !== null && typeof response === "object") {
    const r = response as Record<string, unknown>;
    error = r["error"] ?? error;
    incompleteDetails = r["incomplete_details"] ?? incompleteDetails;
  }
  const detailParts: string[] = [];
  if (error !== null && typeof error === "object") {
    const e = error as Record<string, unknown>;
    const message = e["message"] || e["code"] || e["type"];
    if (typeof message === "string" && message) {
      detailParts.push(message);
    }
  } else if (typeof error === "string" && error) {
    detailParts.push(error);
  }
  if (incompleteDetails !== null && typeof incompleteDetails === "object") {
    const d = incompleteDetails as Record<string, unknown>;
    const reason = d["reason"] || d["message"];
    if (typeof reason === "string" && reason) {
      detailParts.push(reason);
    }
  } else if (typeof incompleteDetails === "string" && incompleteDetails) {
    detailParts.push(incompleteDetails);
  }
  const detail =
    detailParts.length > 0
      ? detailParts.join("; ")
      : JSON.stringify(event).slice(0, 500);
  return `OpenAI protocol response ${status}: ${detail}`;
}

export function reasoningPartsFromResponseItems(
  items: Record<string, unknown>[],
): { summary: string; content: string } {
  const summaryParts: string[] = [];
  const contentParts: string[] = [];
  for (const item of items) {
    if (item["type"] !== "reasoning") {
      continue;
    }
    for (const field of ["summary", "content"]) {
      const value = item[field];
      if (field === "content" && value == null) continue;
      if (!Array.isArray(value)) {
        throw new ChatGPTOAuthProtocolError(`reasoning item ${field} must be an array`);
      }
      const expectedTypes = field === "summary"
        ? new Set(["summary_text"])
        : new Set(["reasoning_text", "text"]);
      for (const [index, part] of value.entries()) {
        if (part === null || typeof part !== "object" || Array.isArray(part)) {
          throw new ChatGPTOAuthProtocolError(
            `reasoning item ${field}[${index}] must be an object`,
          );
        }
        const record = part as Record<string, unknown>;
        if (typeof record.type !== "string" || !expectedTypes.has(record.type)) {
          throw new ChatGPTOAuthProtocolError(
            `reasoning item ${field}[${index}] has an unsupported type`,
          );
        }
        const text = record.text;
        if (typeof text !== "string") {
          throw new ChatGPTOAuthProtocolError(
            `reasoning item ${field}[${index}] text must be a string`,
          );
        }
        if (text) {
          (field === "summary" ? summaryParts : contentParts).push(text);
        }
      }
    }
  }
  return { summary: summaryParts.join(""), content: contentParts.join("") };
}

export function reasoningFromResponseItems(items: Record<string, unknown>[]): string {
  const { summary, content } = reasoningPartsFromResponseItems(items);
  return summary + content;
}
