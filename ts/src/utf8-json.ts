export function decodeUtf8Strict(bytes: Uint8Array): string {
  return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
}

function hasUnpairedSurrogate(value: string): boolean {
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (!(next >= 0xdc00 && next <= 0xdfff)) return true;
      index++;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return true;
    }
  }
  return false;
}

function validateJsonValue(value: unknown): void {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) throw new SyntaxError("JSON numbers must be finite");
    if (Number.isInteger(value) && !Number.isSafeInteger(value)) {
      throw new SyntaxError("JSON integers must be exactly representable");
    }
    return;
  }
  if (typeof value === "string") {
    if (hasUnpairedSurrogate(value)) {
      throw new SyntaxError("JSON strings must not contain unpaired surrogates");
    }
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) validateJsonValue(item);
    return;
  }
  if (value !== null && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      if (hasUnpairedSurrogate(key)) {
        throw new SyntaxError("JSON object keys must not contain unpaired surrogates");
      }
      validateJsonValue(item);
    }
  }
}

export function parseJsonStrict(text: string): unknown {
  const value: unknown = JSON.parse(text);
  validateJsonValue(value);
  return value;
}

export async function parseJsonResponseStrict(
  response: globalThis.Response,
): Promise<unknown> {
  const bytes = new Uint8Array(await response.arrayBuffer());
  return parseJsonStrict(decodeUtf8Strict(bytes));
}
