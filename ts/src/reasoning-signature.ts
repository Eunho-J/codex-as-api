import { ChatGPTOAuthInvalidRequestError } from "./auth.js";
import { decodeUtf8Strict, parseJsonStrict } from "./utf8-json.js";

export const REASONING_SIGNATURE_PREFIX = "codex-reasoning-v1:";
export const REASONING_SIGNATURE_VERSION = 1 as const;

export const REASONING_COMPATIBLE_MODELS: ReadonlySet<string> = new Set([
  "gpt-6.1-sol",
  "gpt-6-astra",
  "gpt-6-sol",
  "gpt-6-luna",
  "gpt-daybreak-blue-latest",
]);

export interface ReasoningSummaryTextPart {
  type: "summary_text";
  text: string;
}

export interface ReasoningTextContentPart {
  type: "reasoning_text";
  text: string;
}

export interface ReplayableReasoningItem {
  type: "reasoning";
  id?: string;
  encrypted_content: string;
  summary: ReasoningSummaryTextPart[];
}

export interface ReasoningSignatureEnvelope {
  v: typeof REASONING_SIGNATURE_VERSION;
  model: string;
  item: ReplayableReasoningItem;
}

export interface AssistantReasoningItem {
  type?: "reasoning";
  model?: string;
  id?: string;
  encrypted_content?: string | null;
  summary: ReasoningSummaryTextPart[];
  content?: ReasoningTextContentPart[];
}

const RECOGNIZED_SIGNATURE_PREFIX_RE = /^\s*codex[-_]reasoning/i;
const BASE64URL_UNPADDED_RE = /^[A-Za-z0-9_-]+$/;
const ENVELOPE_ALLOWED_KEYS = new Set(["v", "model", "item"]);
const ITEM_ALLOWED_KEYS = new Set(["type", "id", "encrypted_content", "summary"]);
const SUMMARY_PART_ALLOWED_KEYS = new Set(["type", "text"]);

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function assertNoDuplicateJsonKeys(jsonText: string): void {
  const stack: Array<
    | { kind: "object"; keys: Set<string>; expectingKey: boolean }
    | { kind: "array" }
  > = [];
  let i = 0;
  const len = jsonText.length;

  while (i < len) {
    const ch = jsonText[i];
    if (ch === " " || ch === "\t" || ch === "\r" || ch === "\n") {
      i += 1;
      continue;
    }
    if (ch === "{") {
      stack.push({ kind: "object", keys: new Set<string>(), expectingKey: true });
      i += 1;
      continue;
    }
    if (ch === "}") {
      stack.pop();
      i += 1;
      continue;
    }
    if (ch === "[") {
      stack.push({ kind: "array" });
      i += 1;
      continue;
    }
    if (ch === "]") {
      stack.pop();
      i += 1;
      continue;
    }
    if (ch === ",") {
      const top = stack[stack.length - 1];
      if (top && top.kind === "object") {
        top.expectingKey = true;
      }
      i += 1;
      continue;
    }
    if (ch === ":") {
      const top = stack[stack.length - 1];
      if (top && top.kind === "object") {
        top.expectingKey = false;
      }
      i += 1;
      continue;
    }
    if (ch === '"') {
      const start = i;
      i += 1;
      while (i < len) {
        const c = jsonText[i];
        if (c === "\\") {
          i += 2;
          continue;
        }
        if (c === '"') {
          i += 1;
          break;
        }
        i += 1;
      }
      const top = stack[stack.length - 1];
      if (top && top.kind === "object" && top.expectingKey) {
        const key = JSON.parse(jsonText.slice(start, i)) as string;
        if (top.keys.has(key)) {
          throw new ChatGPTOAuthInvalidRequestError(
            "Invalid thinking signature: duplicate JSON key",
          );
        }
        top.keys.add(key);
        top.expectingKey = false;
      }
      continue;
    }
    i += 1;
  }
}

export function isReasoningModelCompatible(
  envelopeModel: string,
  targetModel: string,
): boolean {
  if (
    typeof envelopeModel !== "string" ||
    envelopeModel.length === 0 ||
    envelopeModel.trim() !== envelopeModel ||
    typeof targetModel !== "string" ||
    targetModel.length === 0 ||
    targetModel.trim() !== targetModel
  ) {
    return false;
  }
  if (envelopeModel === targetModel) {
    return true;
  }
  return (
    REASONING_COMPATIBLE_MODELS.has(envelopeModel) &&
    REASONING_COMPATIBLE_MODELS.has(targetModel)
  );
}

export function validateReplayableReasoningItem(
  rawItem: unknown,
): ReplayableReasoningItem {
  if (!isPlainObject(rawItem)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid thinking signature: item must be an object",
    );
  }
  for (const key of Object.keys(rawItem)) {
    if (!ITEM_ALLOWED_KEYS.has(key)) {
      throw new ChatGPTOAuthInvalidRequestError(
        "Invalid thinking signature: unsupported item field",
      );
    }
  }
  if (rawItem.type !== "reasoning") {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid thinking signature: item.type must be 'reasoning'",
    );
  }

  let normalizedId: string | undefined;
  if (Object.prototype.hasOwnProperty.call(rawItem, "id")) {
    if (typeof rawItem.id !== "string" || rawItem.id.length === 0) {
      throw new ChatGPTOAuthInvalidRequestError(
        "Invalid thinking signature: item.id must be a non-empty string when present",
      );
    }
    normalizedId = rawItem.id;
  }

  if (
    typeof rawItem.encrypted_content !== "string" ||
    rawItem.encrypted_content.length === 0
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid thinking signature: item.encrypted_content must be a non-empty string",
    );
  }

  if (!Array.isArray(rawItem.summary)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid thinking signature: item.summary must be an array",
    );
  }

  const normalizedSummary: ReasoningSummaryTextPart[] = [];
  for (const rawPart of rawItem.summary) {
    if (!isPlainObject(rawPart)) {
      throw new ChatGPTOAuthInvalidRequestError(
        "Invalid thinking signature: summary entry must be an object",
      );
    }
    for (const key of Object.keys(rawPart)) {
      if (!SUMMARY_PART_ALLOWED_KEYS.has(key)) {
        throw new ChatGPTOAuthInvalidRequestError(
          "Invalid thinking signature: unsupported summary field",
        );
      }
    }
    if (rawPart.type !== "summary_text" || typeof rawPart.text !== "string") {
      throw new ChatGPTOAuthInvalidRequestError(
        "Invalid thinking signature: summary entry must be {type:'summary_text', text:string}",
      );
    }
    normalizedSummary.push({
      type: "summary_text",
      text: rawPart.text,
    });
  }

  const result: ReplayableReasoningItem = {
    type: "reasoning",
    ...(normalizedId !== undefined ? { id: normalizedId } : {}),
    encrypted_content: rawItem.encrypted_content,
    summary: normalizedSummary,
  };
  return result;
}

export function encodeReasoningSignature(
  envelopeOrModel:
    | {
        model: string;
        item: {
          type?: "reasoning";
          id?: string | null;
          encrypted_content: string;
          summary: ReadonlyArray<{ type: "summary_text"; text: string }>;
        };
      }
    | string,
  maybeItem?: {
    type?: "reasoning";
    id?: string | null;
    encrypted_content: string;
    summary: ReadonlyArray<{ type: "summary_text"; text: string }>;
  },
): string {
  const model =
    typeof envelopeOrModel === "string" ? envelopeOrModel : envelopeOrModel.model;
  const rawItem =
    typeof envelopeOrModel === "string" ? maybeItem : envelopeOrModel.item;

  if (typeof model !== "string" || model.length === 0 || model.trim() !== model) {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid reasoning signature model: must be a non-empty trimmed string",
    );
  }
  if (!rawItem || typeof rawItem !== "object") {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid reasoning signature item: must be an object",
    );
  }

  const normalizedRawSummary = Array.isArray(rawItem.summary)
    ? rawItem.summary.map((part: unknown) =>
        isPlainObject(part) &&
        part.type === "summary_text" &&
        typeof part.text === "string"
          ? { type: "summary_text", text: part.text }
          : part,
      )
    : rawItem.summary;

  const candidateItem: Record<string, unknown> = {
    type: rawItem.type ?? "reasoning",
    ...(rawItem.id !== undefined && rawItem.id !== null ? { id: rawItem.id } : {}),
    encrypted_content: rawItem.encrypted_content,
    summary: normalizedRawSummary,
  };
  const validatedItem = validateReplayableReasoningItem(candidateItem);

  const orderedItem: ReplayableReasoningItem =
    validatedItem.id !== undefined
      ? {
          type: "reasoning",
          id: validatedItem.id,
          encrypted_content: validatedItem.encrypted_content,
          summary: validatedItem.summary.map((part) => ({
            type: "summary_text",
            text: part.text,
          })),
        }
      : {
          type: "reasoning",
          encrypted_content: validatedItem.encrypted_content,
          summary: validatedItem.summary.map((part) => ({
            type: "summary_text",
            text: part.text,
          })),
        };

  const orderedEnvelope: ReasoningSignatureEnvelope = {
    v: REASONING_SIGNATURE_VERSION,
    model,
    item: orderedItem,
  };

  const json = JSON.stringify(orderedEnvelope);
  const encoded = Buffer.from(json, "utf8").toString("base64url");
  return `${REASONING_SIGNATURE_PREFIX}${encoded}`;
}

export function decodeReasoningSignature(
  signature: unknown,
): ReasoningSignatureEnvelope | null {
  if (signature === undefined || signature === null) {
    return null;
  }
  if (typeof signature !== "string") {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid thinking signature: signature must be a string",
    );
  }
  if (signature.trim().length === 0) {
    return null;
  }
  if (!RECOGNIZED_SIGNATURE_PREFIX_RE.test(signature)) {
    return null;
  }
  if (!signature.startsWith(REASONING_SIGNATURE_PREFIX)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid thinking signature: unsupported or malformed codex-reasoning prefix",
    );
  }

  const payload = signature.slice(REASONING_SIGNATURE_PREFIX.length);
  if (
    payload.length === 0 ||
    payload.length % 4 === 1 ||
    !BASE64URL_UNPADDED_RE.test(payload)
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid thinking signature: malformed base64url payload",
    );
  }

  const decodedBytes = Buffer.from(payload, "base64url");
  if (decodedBytes.toString("base64url") !== payload) {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid thinking signature: non-canonical base64url payload",
    );
  }

  let jsonText: string;
  try {
    jsonText = decodeUtf8Strict(decodedBytes);
  } catch {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid thinking signature: payload is not valid UTF-8",
    );
  }

  let parsed: unknown;
  try {
    parsed = parseJsonStrict(jsonText);
  } catch {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid thinking signature: payload is not valid JSON",
    );
  }

  assertNoDuplicateJsonKeys(jsonText);

  if (!isPlainObject(parsed)) {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid thinking signature: envelope must be a JSON object",
    );
  }

  for (const key of Object.keys(parsed)) {
    if (!ENVELOPE_ALLOWED_KEYS.has(key)) {
      throw new ChatGPTOAuthInvalidRequestError(
        "Invalid thinking signature: unsupported envelope field",
      );
    }
  }

  if (parsed.v !== REASONING_SIGNATURE_VERSION) {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid thinking signature: unsupported envelope version",
    );
  }

  if (
    typeof parsed.model !== "string" ||
    parsed.model.length === 0 ||
    parsed.model.trim() !== parsed.model
  ) {
    throw new ChatGPTOAuthInvalidRequestError(
      "Invalid thinking signature: model must be a non-empty trimmed string",
    );
  }

  const item = validateReplayableReasoningItem(parsed.item);

  return {
    v: REASONING_SIGNATURE_VERSION,
    model: parsed.model,
    item,
  };
}
