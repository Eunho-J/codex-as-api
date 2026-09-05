export enum MessageRole {
  SYSTEM = "system",
  DEVELOPER = "developer",
  USER = "user",
  ASSISTANT = "assistant",
  TOOL = "tool",
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: string;
}

export interface PromptCacheBreakpoint {
  mode: "explicit";
}

export type MessageContentPart =
  | {
      type: "text";
      text: string;
      prompt_cache_breakpoint?: PromptCacheBreakpoint;
    }
  | {
      type: "image_url";
      image_url: string;
      detail?: "auto" | "low" | "high" | "original";
      prompt_cache_breakpoint?: PromptCacheBreakpoint;
    }
  | {
      type: "input_audio";
      audio_url: string;
      prompt_cache_breakpoint?: PromptCacheBreakpoint;
    };

export interface Message {
  role: MessageRole;
  content: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
  reasoning_content?: string;
  images?: string[];
  structured_content?: MessageContentPart[];
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens?: number;
  cache_write_tokens?: number;
}

export type FinishReason = "stop" | "tool_calls" | null;

export interface AssistantResponse {
  content: string;
  tool_calls: ToolCall[];
  finish_reason: FinishReason;
  usage: Usage | null;
  reasoning_content: string | null;
  raw: Record<string, unknown> | null;
  response_id?: string | null;
}

export interface ToolSchema {
  name: string;
  description?: string;
  parameters: Record<string, unknown>;
  strict?: boolean;
  allowed_callers?: unknown;
  output_schema?: unknown;
}
