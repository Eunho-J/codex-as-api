import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

declare const __O200K_RANK_DATA__: string | undefined;

const O200K_RANK_SHA256 =
  "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d";
const MAX_RANK = 0xffff_ffff;

// Ported from openai/tiktoken 0.13.0 at
// 08a5f3b2c987ada4fc5aa1f16c643c203fa8acaa; last synchronized 2026-07-14.
// The inline case-insensitive contraction group is expanded because JavaScript
// does not support scoped flags. U+017F is the Unicode simple-case fold of "s".
const CONTRACTION =
  "(?:'[sS\u017f]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?";
const O200K_PATTERN = new RegExp(
  [
    `[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+${CONTRACTION}`,
    `[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*${CONTRACTION}`,
    "\\p{N}{1,3}",
    " ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*",
    "\\s*[\\r\\n]+",
    "\\s+(?!\\S)",
    "\\s+",
  ].join("|"),
  "gu",
);

type Merge = { start: number; rank: number };

class MergeHeap {
  private readonly items: Merge[] = [];

  push(item: Merge): void {
    let index = this.items.length;
    this.items.push(item);
    while (index > 0) {
      const parent = (index - 1) >>> 1;
      if (!MergeHeap.before(item, this.items[parent])) break;
      this.items[index] = this.items[parent];
      index = parent;
    }
    this.items[index] = item;
  }

  pop(): Merge | undefined {
    const first = this.items[0];
    const last = this.items.pop();
    if (first == null || last == null || this.items.length === 0) return first;

    let index = 0;
    while (true) {
      const left = index * 2 + 1;
      if (left >= this.items.length) break;
      const right = left + 1;
      const child = right < this.items.length
        && MergeHeap.before(this.items[right], this.items[left])
        ? right
        : left;
      if (!MergeHeap.before(this.items[child], last)) break;
      this.items[index] = this.items[child];
      index = child;
    }
    this.items[index] = last;
    return first;
  }

  private static before(left: Merge, right: Merge): boolean {
    return left.rank < right.rank
      || (left.rank === right.rank && left.start < right.start);
  }
}

let mergeableRanks: Map<string, number> | undefined;

function rankData(): string {
  if (typeof __O200K_RANK_DATA__ === "string") return __O200K_RANK_DATA__;

  for (const path of [
    resolve(process.cwd(), "config/o200k_base.tiktoken"),
    resolve(process.cwd(), "../config/o200k_base.tiktoken"),
  ]) {
    if (existsSync(path)) return readFileSync(path, "utf8");
  }
  throw new Error("config/o200k_base.tiktoken is unavailable");
}

function ranks(): Map<string, number> {
  if (mergeableRanks != null) return mergeableRanks;

  const data = rankData();
  const digest = createHash("sha256").update(data).digest("hex");
  if (digest !== O200K_RANK_SHA256) {
    throw new Error(
      `o200k_base.tiktoken SHA-256 mismatch: expected ${O200K_RANK_SHA256}, got ${digest}`,
    );
  }

  const parsed = new Map<string, number>();
  for (const [index, line] of data.split("\n").entries()) {
    if (line === "") continue;
    const separator = line.lastIndexOf(" ");
    const rank = Number(line.slice(separator + 1));
    if (separator <= 0 || !Number.isSafeInteger(rank) || rank < 0) {
      throw new Error(`invalid o200k_base rank data at line ${index + 1}`);
    }
    parsed.set(Buffer.from(line.slice(0, separator), "base64").toString("latin1"), rank);
  }
  if (parsed.size !== 199_998) {
    throw new Error(`invalid o200k_base rank count: ${parsed.size}`);
  }
  mergeableRanks = parsed;
  return parsed;
}

function key(bytes: Uint8Array, start = 0, end = bytes.length): string {
  return Buffer.from(bytes.buffer, bytes.byteOffset + start, end - start).toString("latin1");
}

function rankOf(
  encoder: Map<string, number>,
  bytes: Uint8Array,
  start: number,
  end: number,
): number | undefined {
  return encoder.get(key(bytes, start, end));
}

function bytePairEncodeSmall(
  bytes: Uint8Array,
  encoder: Map<string, number>,
): number[] {
  const parts: Array<{ start: number; rank: number }> = [];
  let minRank = MAX_RANK;
  let minIndex = MAX_RANK;
  for (let index = 0; index < bytes.length - 1; index += 1) {
    const rank = rankOf(encoder, bytes, index, index + 2) ?? MAX_RANK;
    if (rank < minRank) {
      minRank = rank;
      minIndex = index;
    }
    parts.push({ start: index, rank });
  }
  parts.push({ start: bytes.length - 1, rank: MAX_RANK });
  parts.push({ start: bytes.length, rank: MAX_RANK });

  const mergedRank = (index: number): number => {
    if (index + 3 >= parts.length) return MAX_RANK;
    return rankOf(
      encoder,
      bytes,
      parts[index].start,
      parts[index + 3].start,
    ) ?? MAX_RANK;
  };

  while (minRank !== MAX_RANK) {
    if (minIndex > 0) parts[minIndex - 1].rank = mergedRank(minIndex - 1);
    parts[minIndex].rank = mergedRank(minIndex);
    parts.splice(minIndex + 1, 1);

    minRank = MAX_RANK;
    minIndex = MAX_RANK;
    for (let index = 0; index < parts.length - 1; index += 1) {
      if (parts[index].rank < minRank) {
        minRank = parts[index].rank;
        minIndex = index;
      }
    }
  }

  const result: number[] = [];
  for (let index = 0; index < parts.length - 1; index += 1) {
    const rank = rankOf(
      encoder,
      bytes,
      parts[index].start,
      parts[index + 1].start,
    );
    if (rank == null) throw new Error("o200k_base BPE produced an unknown token");
    result.push(rank);
  }
  return result;
}

function bytePairEncodeLarge(
  bytes: Uint8Array,
  encoder: Map<string, number>,
): number[] {
  const length = bytes.length;
  const previous = new Uint32Array(length);
  const end = new Uint32Array(length);
  const nextEnd = new Uint32Array(length);
  const nextRank = new Uint32Array(length);
  const currentRank = new Uint32Array(length);
  nextRank.fill(MAX_RANK);
  currentRank.fill(MAX_RANK);

  previous[0] = MAX_RANK;
  end[0] = 1;
  nextEnd[0] = 2;
  const heap = new MergeHeap();
  for (let index = 0; index < length - 1; index += 1) {
    const rank = rankOf(encoder, bytes, index, index + 2);
    if (rank != null) {
      heap.push({ start: index, rank });
      nextRank[index] = rank;
    }
    previous[index + 1] = index;
    end[index + 1] = index + 2;
    nextEnd[index + 1] = index + 3;
  }

  const potentialMerge = (start: number, followingEnd: number): void => {
    nextEnd[start] = followingEnd;
    nextRank[start] = MAX_RANK;
    if (followingEnd <= length) {
      const rank = rankOf(encoder, bytes, start, followingEnd);
      if (rank != null) {
        heap.push({ start, rank });
        nextRank[start] = rank;
      }
    }
  };

  for (let merge = heap.pop(); merge != null; merge = heap.pop()) {
    if (merge.rank !== nextRank[merge.start]) continue;

    const leftStart = merge.start;
    const rightStart = end[leftStart];
    const rightEnd = nextEnd[leftStart];
    const rightNextEnd = nextEnd[rightStart];
    currentRank[leftStart] = nextRank[leftStart];
    end[leftStart] = rightEnd;
    potentialMerge(leftStart, rightNextEnd);
    if (rightEnd < length) previous[rightEnd] = leftStart;
    if (leftStart > 0) potentialMerge(previous[leftStart], rightEnd);
    nextRank[rightStart] = MAX_RANK;
  }

  const result: number[] = [];
  for (let index = 0; index < length; index = end[index]) {
    const rank = currentRank[index] !== MAX_RANK
      ? currentRank[index]
      : rankOf(encoder, bytes, index, end[index]);
    if (rank == null) throw new Error("o200k_base BPE produced an unknown token");
    result.push(rank);
  }
  return result;
}

function encodePiece(piece: string, encoder: Map<string, number>): number[] {
  const bytes = Buffer.from(piece, "utf8");
  const directRank = encoder.get(key(bytes));
  if (directRank != null) return [directRank];
  if (bytes.length === 1) {
    const rank = encoder.get(key(bytes));
    if (rank == null) throw new Error("o200k_base is missing a byte token");
    return [rank];
  }
  return bytes.length < 100
    ? bytePairEncodeSmall(bytes, encoder)
    : bytePairEncodeLarge(bytes, encoder);
}

function forEachEncodedPiece(text: string, consume: (tokens: number[]) => void): void {
  const encoder = ranks();
  let matchedUntil = 0;
  for (const match of text.matchAll(O200K_PATTERN)) {
    if (match.index !== matchedUntil || match[0].length === 0) {
      throw new Error(`o200k_base regex did not cover UTF-16 offset ${matchedUntil}`);
    }
    consume(encodePiece(match[0], encoder));
    matchedUntil += match[0].length;
  }
  if (matchedUntil !== text.length) {
    throw new Error(`o200k_base regex did not cover UTF-16 offset ${matchedUntil}`);
  }
}

/** Encode text exactly like tiktoken's o200k_base encode_ordinary. */
export function encodeO200kOrdinary(text: string): number[] {
  const result: number[] = [];
  forEachEncodedPiece(text, (tokens) => {
    for (const token of tokens) result.push(token);
  });
  return result;
}

/** Count text tokens without treating special-token-shaped literals specially. */
export function countO200kOrdinaryTokens(text: string): number {
  let count = 0;
  forEachEncodedPiece(text, (tokens) => {
    count += tokens.length;
  });
  return count;
}
