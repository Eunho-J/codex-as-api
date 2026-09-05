use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    Developer,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
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
    /// Normalized Responses API content blocks when the caller supplied
    /// structured Chat Completions content. `None` retains the legacy
    /// `content` + `images` representation.
    pub structured_content: Option<Vec<serde_json::Value>>,
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
            if tool_call_id.is_none() {
                return Err(MessageError::Validation(
                    "tool messages require a string tool_call_id".to_string(),
                ));
            }
            if name.as_deref().is_some_and(str::is_empty) {
                return Err(MessageError::Validation(
                    "tool message name must be non-empty when provided".to_string(),
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
            structured_content: None,
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
    pub cached_tokens: Option<i64>,
    pub cache_write_tokens: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct AssistantResponse {
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: Option<String>,
    pub usage: Option<Usage>,
    pub reasoning_content: Option<String>,
    pub raw: Option<serde_json::Value>,
    pub response_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
    pub strict: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_user() {
        let msg = Message::new(MessageRole::User, "hello".to_string(), vec![], None, None).unwrap();
        assert_eq!(msg.role, MessageRole::User);
        assert_eq!(msg.content, "hello");
    }

    #[test]
    fn test_message_tool_requires_fields() {
        let result = Message::new(MessageRole::Tool, "output".to_string(), vec![], None, None);
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
    fn test_message_tool_name_is_optional() {
        let msg = Message::new(
            MessageRole::Tool,
            "output".to_string(),
            vec![],
            Some("call-1".to_string()),
            None,
        )
        .unwrap();
        assert_eq!(msg.tool_call_id.as_deref(), Some("call-1"));
        assert!(msg.name.is_none());
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
            arguments: "{}".to_string(),
        };
        let result = Message::new(MessageRole::User, "hi".to_string(), vec![tc], None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_message_assistant_with_tool_calls() {
        let tc = ToolCall {
            id: "c1".to_string(),
            name: "fn".to_string(),
            arguments: "{}".to_string(),
        };
        let msg =
            Message::new(MessageRole::Assistant, "".to_string(), vec![tc], None, None).unwrap();
        assert_eq!(msg.tool_calls.len(), 1);
    }

    #[test]
    fn usage_preserves_explicit_total() {
        let u = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cached_tokens: Some(20),
            cache_write_tokens: None,
        };
        assert_eq!(u.total_tokens, 150);
        assert_eq!(u.cached_tokens, Some(20));
    }

    #[test]
    fn usage_preserves_cache_write_tokens() {
        let u = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cached_tokens: Some(0),
            cache_write_tokens: Some(9),
        };
        assert_eq!(u.cache_write_tokens, Some(9));
    }

    #[test]
    fn usage_serialization_keeps_actual_counts() {
        let u = Usage {
            prompt_tokens: 4,
            completion_tokens: 3,
            total_tokens: 7,
            cached_tokens: Some(2),
            cache_write_tokens: None,
        };
        let value = serde_json::to_value(u).unwrap();
        assert_eq!(value["prompt_tokens"], 4);
        assert_eq!(value["completion_tokens"], 3);
        assert_eq!(value["total_tokens"], 7);
    }

    #[test]
    fn usage_deserialization_requires_all_required_counts() {
        assert!(serde_json::from_value::<Usage>(serde_json::json!({
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "cached_tokens": 0,
            "cache_write_tokens": null
        }))
        .is_err());
    }
}
