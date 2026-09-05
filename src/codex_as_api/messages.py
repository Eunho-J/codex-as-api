from __future__ import annotations

import dataclasses
import enum


class MessageRole(enum.Enum):
    SYSTEM = "system"
    DEVELOPER = "developer"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str):
            raise TypeError("ToolCall.id must be a string")
        if not isinstance(self.name, str):
            raise TypeError("ToolCall.name must be a string")
        if not isinstance(self.arguments, str):
            raise TypeError("ToolCall.arguments must be a string")


@dataclasses.dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None
    images: tuple[str, ...] = ()
    content_parts: tuple[dict[str, object], ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise TypeError("Message.role must be a MessageRole")
        if not isinstance(self.content, str):
            raise TypeError("Message.content must be a string")
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.content_parts is not None and not isinstance(self.content_parts, tuple):
            object.__setattr__(self, "content_parts", tuple(self.content_parts))
        if not isinstance(self.images, tuple):
            object.__setattr__(self, "images", tuple(self.images))
        if any(not isinstance(image, str) or not image for image in self.images):
            raise ValueError("Message.images must contain non-empty strings")
        if self.content_parts is not None and any(not isinstance(part, dict) for part in self.content_parts):
            raise TypeError("Message.content_parts must contain dict values")
        if self.reasoning_content is not None and not isinstance(
            self.reasoning_content,
            str,
        ):
            raise TypeError("Message.reasoning_content must be a string or None")
        if self.role is MessageRole.TOOL:
            if not isinstance(self.tool_call_id, str):
                raise TypeError("tool messages require a string tool_call_id")
            if self.name is not None and (not isinstance(self.name, str) or not self.name):
                raise ValueError("tool message name must be non-empty when provided")
        elif self.tool_call_id is not None or self.name is not None:
            raise ValueError("tool_call_id and name are only allowed on tool messages")
        if self.tool_calls and self.role is not MessageRole.ASSISTANT:
            raise ValueError("tool_calls are only allowed on assistant messages")


@dataclasses.dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None

    def __post_init__(self) -> None:
        for field, value in (
            ("prompt_tokens", self.prompt_tokens),
            ("completion_tokens", self.completion_tokens),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Usage.{field} must be a non-negative integer")
        if self.cache_write_tokens is not None and (
            not isinstance(self.cache_write_tokens, int)
            or isinstance(self.cache_write_tokens, bool)
            or self.cache_write_tokens < 0
        ):
            raise ValueError("Usage.cache_write_tokens must be a non-negative integer or None")
        if self.cached_tokens is not None and (
            not isinstance(self.cached_tokens, int) or isinstance(self.cached_tokens, bool) or self.cached_tokens < 0
        ):
            raise ValueError("Usage.cached_tokens must be a non-negative integer or None")
        if not isinstance(self.total_tokens, int) or isinstance(self.total_tokens, bool) or self.total_tokens < 0:
            raise ValueError("Usage.total_tokens must be a non-negative integer")
        if self.total_tokens != self.prompt_tokens + self.completion_tokens:
            raise ValueError("Usage.total_tokens must equal prompt_tokens plus completion_tokens")


@dataclasses.dataclass(frozen=True, slots=True)
class AssistantResponse:
    content: str
    finish_reason: str | None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage | None = None
    reasoning_content: str | None = None
    raw: dict | None = None
    response_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.content, str):
            raise TypeError("AssistantResponse.content must be a string")
        if self.finish_reason not in {None, "stop", "tool_calls"}:
            raise ValueError("AssistantResponse.finish_reason must be null, stop, or tool_calls")
        if self.response_id is not None and (not isinstance(self.response_id, str) or not self.response_id):
            raise ValueError("AssistantResponse.response_id must be non-empty when provided")
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))


class InterruptIdleSignal(Exception):  # noqa: N818 - this is an internal control-flow signal.
    """Raised when an interrupted agent turn should return to idle."""


@dataclasses.dataclass(frozen=True, slots=True)
class TerminatorCall:
    name: str
    arguments: dict

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("TerminatorCall.name must be non-empty")
        if not isinstance(self.arguments, dict):
            raise TypeError("TerminatorCall.arguments must be a dict")


@dataclasses.dataclass(frozen=True, slots=True)
class AgentResponse:
    text: str
    finish_reason: str
    terminator_call: TerminatorCall | None = None
    usage: Usage | None = None
    reasoning_content: str | None = None
    raw: dict | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    content: str
    payload: dict | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class ToolSchema:
    name: str
    description: str | None
    parameters: dict
    strict: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ToolSchema.name must be a non-empty string")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("ToolSchema.description must be a string or None")
        if not isinstance(self.parameters, dict):
            raise TypeError("ToolSchema.parameters must be a dict")
        if not isinstance(self.strict, bool):
            raise TypeError("ToolSchema.strict must be a boolean")
