function getValue(value: unknown, key: string, defaultValue: unknown = undefined): unknown {
  if (value !== null && typeof value === "object" && key in (value as Record<string, unknown>)) {
    return (value as Record<string, unknown>)[key];
  }
  return defaultValue;
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
    for (const item of content) {
      const text = getValue(item, "text");
      if (text) {
        parts.push(String(text));
      }
    }
    return parts.join("");
  }
  return String(content);
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

export function reasoningFromResponseItems(items: Record<string, unknown>[]): string {
  const parts: string[] = [];
  for (const item of items) {
    if (item["type"] !== "reasoning") {
      continue;
    }
    for (const field of ["summary", "content"]) {
      const value = item[field];
      if (typeof value === "string" && value) {
        parts.push(value);
        continue;
      }
      if (!Array.isArray(value)) {
        continue;
      }
      for (const part of value) {
        if (typeof part === "string" && part) {
          parts.push(part);
        } else if (part !== null && typeof part === "object") {
          const text = (part as Record<string, unknown>)["text"];
          if (typeof text === "string" && text) {
            parts.push(text);
          }
        }
      }
    }
  }
  return parts.join("");
}
