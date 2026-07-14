from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from codex_as_api.o200k_tokenizer import count_ordinary, encode_ordinary

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "o200k_base_encode_ordinary.json"
_CASES = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _CASES, ids=lambda case: repr(case["text"])[:48])
def test_encode_ordinary_matches_tiktoken_0_13_0_fixture(case):
    expected = case["tokens"]

    assert encode_ordinary(case["text"]) == expected
    assert count_ordinary(case["text"]) == len(expected)


def test_encode_ordinary_handles_a_4000_byte_single_piece_without_quadratic_slowdown():
    text = "abcd" * 1000
    encode_ordinary("warm rank cache")

    started = time.perf_counter()
    tokens = encode_ordinary(text)
    elapsed = time.perf_counter() - started

    assert len(tokens) == 1000
    assert count_ordinary(text) == 1000
    assert elapsed < 1.0
