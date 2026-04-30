from __future__ import annotations

import dataclasses
import enum


class MessageRole(enum.Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclasses.dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclasses.dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))
        if self.role is MessageRole.TOOL:
            if self.tool_call_id is None or self.name is None:
                raise ValueError("tool messages require tool_call_id and name")
        elif self.tool_call_id is not None or self.name is not None:
            raise ValueError("tool_call_id and name are only allowed on tool messages")
        if self.tool_calls and self.role is not MessageRole.ASSISTANT:
            raise ValueError("tool_calls are only allowed on assistant messages")


@dataclasses.dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int | None = None
    cached_tokens: int = 0

    def __post_init__(self) -> None:
        if self.total_tokens is None:
            object.__setattr__(
                self,
                "total_tokens",
                self.prompt_tokens + self.completion_tokens,
            )

    @property
    def cache_hit_rate(self) -> float:
        if self.prompt_tokens <= 0:
            return 0.0
        return self.cached_tokens / self.prompt_tokens


@dataclasses.dataclass(frozen=True, slots=True)
class AssistantResponse:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    usage: Usage | None = None
    reasoning_content: str | None = None
    raw: dict | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tool_calls, tuple):
            object.__setattr__(self, "tool_calls", tuple(self.tool_calls))


class InterruptIdleSignal(Exception):
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
    terminator_call: TerminatorCall | None = None
    finish_reason: str = "stop"
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
    description: str
    parameters: dict
