export enum MessageRole {
  SYSTEM = "system",
  USER = "user",
  ASSISTANT = "assistant",
  TOOL = "tool",
}

export interface ToolCall {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
}

export interface Message {
  role: MessageRole;
  content: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
  reasoning_content?: string;
  images?: string[];
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  cached_tokens: number;
}

export interface AssistantResponse {
  content: string;
  tool_calls: ToolCall[];
  finish_reason: string;
  usage: Usage | null;
  reasoning_content: string | null;
  raw: Record<string, unknown> | null;
}

export interface ToolSchema {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
}
