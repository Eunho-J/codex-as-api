from __future__ import annotations

import json
import math
from typing import Any, NoReturn

JS_SAFE_INTEGER = 9_007_199_254_740_991


def as_js_safe_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if abs(value) <= JS_SAFE_INTEGER else None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer() and abs(value) <= JS_SAFE_INTEGER:
        return int(value)
    return None


def _reject_nonstandard_constant(_value: str) -> NoReturn:
    raise ValueError("non-standard JSON constants are not allowed")


def _parse_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("JSON numbers must be finite")
    if parsed.is_integer() and abs(parsed) > JS_SAFE_INTEGER:
        raise ValueError("JSON integers must be JavaScript-safe")
    return parsed


def _parse_safe_integer(value: str) -> int:
    parsed = int(value)
    if abs(parsed) > JS_SAFE_INTEGER:
        raise ValueError("JSON integers must be JavaScript-safe")
    return parsed


def _validate_strings_and_numbers(value: Any) -> None:
    if isinstance(value, str):
        if any("\ud800" <= character <= "\udfff" for character in value):
            raise ValueError("JSON strings must not contain lone surrogate code points")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
        return
    if isinstance(value, list):
        for item in value:
            _validate_strings_and_numbers(item)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_strings_and_numbers(key)
            _validate_strings_and_numbers(item)


def strict_json_loads(data: str | bytes | bytearray) -> Any:
    if isinstance(data, bytearray):
        data = bytes(data)
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    value = json.loads(
        data,
        parse_constant=_reject_nonstandard_constant,
        parse_float=_parse_finite_float,
        parse_int=_parse_safe_integer,
    )
    _validate_strings_and_numbers(value)
    return value
