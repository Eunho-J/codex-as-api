use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
    pub reasoning_content: Option<String>,
    pub images: Vec<String>,
}

impl Message {
    pub fn new(
        role: MessageRole,
        content: String,
        tool_calls: Vec<ToolCall>,
        tool_call_id: Option<String>,
        name: Option<String>,
    ) -> Result<Self, MessageError> {
        if role == MessageRole::Tool {
            if tool_call_id.is_none() || name.is_none() {
                return Err(MessageError::Validation(
                    "tool messages require tool_call_id and name".to_string(),
                ));
            }
        } else if tool_call_id.is_some() || name.is_some() {
            return Err(MessageError::Validation(
                "tool_call_id and name are only allowed on tool messages".to_string(),
            ));
        }
        if !tool_calls.is_empty() && role != MessageRole::Assistant {
            return Err(MessageError::Validation(
                "tool_calls are only allowed on assistant messages".to_string(),
            ));
        }
        Ok(Self {
            role,
            content,
            tool_calls,
            tool_call_id,
            name,
            reasoning_content: None,
            images: vec![],
        })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MessageError {
    #[error("{0}")]
    Validation(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: i64,
    pub completion_tokens: i64,
    pub total_tokens: i64,
    pub cached_tokens: i64,
}

impl Usage {
    pub fn new(
        prompt_tokens: i64,
        completion_tokens: i64,
        total_tokens: Option<i64>,
        cached_tokens: i64,
    ) -> Self {
        let total = total_tokens.unwrap_or(prompt_tokens + completion_tokens);
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: total,
            cached_tokens,
        }
    }

    pub fn cache_hit_rate(&self) -> f64 {
        if self.prompt_tokens <= 0 {
            return 0.0;
        }
        self.cached_tokens as f64 / self.prompt_tokens as f64
    }
}

#[derive(Debug, Clone)]
pub struct AssistantResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
    pub usage: Option<Usage>,
    pub reasoning_content: Option<String>,
    pub raw: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_user() {
        let msg = Message::new(
            MessageRole::User,
            "hello".to_string(),
            vec![],
            None,
            None,
        )
        .unwrap();
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn test_message_tool_requires_fields() {
        let result = Message::new(
            MessageRole::Tool,
            "output".to_string(),
            vec![],
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_message_tool_valid() {
        let msg = Message::new(
            MessageRole::Tool,
            "output".to_string(),
            vec![],
            Some("call-1".to_string()),
            Some("my_tool".to_string()),
        )
        .unwrap();
        assert_eq!(msg.role, MessageRole::Tool);
    }

    #[test]
    fn test_message_non_tool_with_tool_call_id() {
        let result = Message::new(
            MessageRole::User,
            "hi".to_string(),
            vec![],
            Some("call-1".to_string()),
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_message_tool_calls_on_non_assistant() {
        let tc = ToolCall {
            id: "c1".to_string(),
            name: "fn".to_string(),
            arguments: HashMap::new(),
        };
        let result = Message::new(
            MessageRole::User,
            "hi".to_string(),
            vec![tc],
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_message_assistant_with_tool_calls() {
        let tc = ToolCall {
            id: "c1".to_string(),
            name: "fn".to_string(),
            arguments: HashMap::new(),
        };
        let msg = Message::new(
            MessageRole::Assistant,
            "".to_string(),
            vec![tc],
            None,
            None,
        )
        .unwrap();
        assert_eq!(msg.tool_calls.len(), 1);
    }

    #[test]
    fn test_usage_auto_total() {
        let u = Usage::new(100, 50, None, 20);
        assert_eq!(u.total_tokens, 150);
        assert_eq!(u.cached_tokens, 20);
    }

    #[test]
    fn test_usage_explicit_total() {
        let u = Usage::new(100, 50, Some(200), 0);
        assert_eq!(u.total_tokens, 200);
    }

    #[test]
    fn test_usage_cache_hit_rate() {
        let u = Usage::new(100, 50, None, 80);
        assert!((u.cache_hit_rate() - 0.8).abs() < f64::EPSILON);
    }

    #[test]
    fn test_usage_cache_hit_rate_zero_prompt() {
        let u = Usage::new(0, 50, None, 0);
        assert!((u.cache_hit_rate() - 0.0).abs() < f64::EPSILON);
    }
}
