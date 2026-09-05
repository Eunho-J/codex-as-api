from __future__ import annotations

import pytest

from codex_as_api.strict_json import strict_json_loads


@pytest.mark.parametrize(
    "document",
    [
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
        '{"value":9007199254740992}',
        '{"value":-9007199254740992}',
        '{"value":9007199254740992.5}',
        '{"value":"\\ud800"}',
        '{"\\udfff":"value"}',
    ],
)
def test_strict_json_rejects_nonstandard_numbers_and_lone_surrogates(document):
    with pytest.raises(ValueError):
        strict_json_loads(document)


def test_strict_json_accepts_safe_numbers_and_paired_surrogates():
    assert strict_json_loads('{"value":1.25,"integer":9007199254740991,"emoji":"\\ud83d\\ude00"}') == {
        "value": 1.25,
        "integer": 9_007_199_254_740_991,
        "emoji": "😀",
    }
