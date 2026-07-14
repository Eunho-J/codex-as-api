"""Bundled o200k_base BPE implementation without the tiktoken package."""

from __future__ import annotations

import base64
import hashlib
import heapq
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Final

import regex

# Ported from openai/tiktoken 0.13.0 at commit
# 08a5f3b2c987ada4fc5aa1f16c643c203fa8acaa. Last synchronization check:
# 2026-07-14. Special-token-looking text intentionally follows encode_ordinary
# and is therefore encoded as ordinary text.
_O200K_PATTERN: Final = regex.compile(
    "|".join(
        [
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*"
            r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+"
            r"[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
            r"\p{N}{1,3}",
            r" ?[^\s\p{L}\p{N}]+[\r\n/]*",
            r"\s*[\r\n]+",
            r"\s+(?!\S)",
            r"\s+",
        ]
    )
)
_RANKS_SHA256: Final = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"
_RANK_COUNT: Final = 199_998
_MAX_RANK: Final = 2**32 - 1


def _rank_asset_bytes() -> bytes:
    installed_asset = files("codex_as_api").joinpath("o200k_base.tiktoken")
    if installed_asset.is_file():
        return installed_asset.read_bytes()

    # Editable/source checkouts keep one shared copy for every implementation.
    source_asset = Path(__file__).resolve().parents[2] / "config" / "o200k_base.tiktoken"
    return source_asset.read_bytes()


@lru_cache(maxsize=1)
def _mergeable_ranks() -> dict[bytes, int]:
    data = _rank_asset_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != _RANKS_SHA256:
        raise RuntimeError(f"o200k_base rank asset checksum mismatch: expected {_RANKS_SHA256}, got {digest}")

    ranks: dict[bytes, int] = {}
    for expected_rank, line in enumerate(data.splitlines()):
        try:
            encoded_token, rank_text = line.split()
            rank = int(rank_text)
            token = base64.b64decode(encoded_token, validate=True)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid o200k_base rank row {expected_rank + 1}") from exc
        if rank != expected_rank:
            raise RuntimeError(
                f"invalid o200k_base rank ordering at row {expected_rank + 1}: expected {expected_rank}, got {rank}"
            )
        if token in ranks:
            raise RuntimeError(f"duplicate o200k_base token at row {expected_rank + 1}")
        ranks[token] = rank

    if len(ranks) != _RANK_COUNT:
        raise RuntimeError(f"invalid o200k_base rank count: expected {_RANK_COUNT}, got {len(ranks)}")
    return ranks


def _normalise_invalid_unicode(text: str) -> str:
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return text


def _byte_pair_encode(piece: bytes, ranks: dict[bytes, int]) -> list[int]:
    direct_rank = ranks.get(piece)
    if direct_rank is not None:
        return [direct_rank]

    piece_length = len(piece)
    if piece_length == 1:
        return [ranks[piece]]

    # Heap-based equivalent of tiktoken's large-piece merge. Each live token is
    # represented by its byte start and end; stale potential merges remain in
    # the heap and are discarded by comparing their recorded rank with state.
    previous = [index - 1 for index in range(piece_length)]
    ends = [index + 1 for index in range(piece_length)]
    next_ends = [index + 2 for index in range(piece_length)]
    next_ranks = [_MAX_RANK] * piece_length
    current_ranks = [_MAX_RANK] * piece_length
    heap: list[tuple[int, int]] = []

    for start in range(piece_length - 1):
        rank = ranks.get(piece[start : start + 2])
        if rank is not None:
            next_ranks[start] = rank
            heapq.heappush(heap, (rank, start))

    def register_potential_merge(start: int, next_end: int) -> None:
        next_ends[start] = next_end
        next_ranks[start] = _MAX_RANK
        if next_end <= piece_length:
            rank = ranks.get(piece[start:next_end])
            if rank is not None:
                next_ranks[start] = rank
                heapq.heappush(heap, (rank, start))

    while heap:
        rank, left_start = heapq.heappop(heap)
        if rank != next_ranks[left_start]:
            continue

        right_start = ends[left_start]
        right_end = next_ends[left_start]
        right_next_end = next_ends[right_start]

        current_ranks[left_start] = rank
        ends[left_start] = right_end
        register_potential_merge(left_start, right_next_end)
        if right_end < piece_length:
            previous[right_end] = left_start
        if left_start > 0:
            register_potential_merge(previous[left_start], right_end)
        next_ranks[right_start] = _MAX_RANK

    result: list[int] = []
    start = 0
    while start < piece_length:
        rank = current_ranks[start]
        if rank == _MAX_RANK:
            rank = ranks[piece[start : ends[start]]]
        result.append(rank)
        start = ends[start]
    return result


def encode_ordinary(text: str) -> list[int]:
    """Encode text with o200k_base while treating special-token literals ordinarily."""
    ranks = _mergeable_ranks()
    text = _normalise_invalid_unicode(text)
    tokens: list[int] = []
    for match in _O200K_PATTERN.finditer(text):
        tokens.extend(_byte_pair_encode(match.group().encode("utf-8"), ranks))
    return tokens


def count_ordinary(text: str) -> int:
    """Return the exact number of o200k_base encode_ordinary tokens in text."""
    ranks = _mergeable_ranks()
    text = _normalise_invalid_unicode(text)
    count = 0
    for match in _O200K_PATTERN.finditer(text):
        count += len(_byte_pair_encode(match.group().encode("utf-8"), ranks))
    return count
