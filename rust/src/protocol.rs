use serde_json::Value;
use std::collections::HashMap;

pub fn get_value<'a>(value: &'a Value, key: &str) -> &'a Value {
    match value {
        Value::Object(map) => map.get(key).unwrap_or(&Value::Null),
        _ => &Value::Null,
    }
}

pub fn normalize_stream_content(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(arr) => {
            let mut parts = Vec::new();
            for item in arr {
                if let Some(text) = get_value(item, "text").as_str() {
                    if !text.is_empty() {
                        parts.push(text.to_string());
                    }
                }
            }
            parts.join("")
        }
        other => other.to_string(),
    }
}

pub fn normalize_openai_chat_completion_chunk(chunk: &Value) -> Vec<HashMap<String, Value>> {
    let mut events = Vec::new();
    let choices = get_value(chunk, "choices");
    let choices_arr = match choices.as_array() {
        Some(arr) if !arr.is_empty() => arr,
        _ => return events,
    };
    let choice = &choices_arr[0];
    let delta = get_value(choice, "delta");
    let content = normalize_stream_content(get_value(delta, "content"));
    if !content.is_empty() {
        let mut ev = HashMap::new();
        ev.insert("type".to_string(), Value::String("content".to_string()));
        ev.insert("text".to_string(), Value::String(content));
        events.push(ev);
    }
    for key in &["reasoning_content", "reasoning", "reasoning_summary"] {
        let reasoning = normalize_stream_content(get_value(delta, key));
        if !reasoning.is_empty() {
            let mut ev = HashMap::new();
            ev.insert(
                "type".to_string(),
                Value::String("reasoning_delta".to_string()),
            );
            ev.insert("text".to_string(), Value::String(reasoning));
            ev.insert(
                "source_key".to_string(),
                Value::String(key.to_string()),
            );
            events.push(ev);
        }
    }
    let raw_reasoning = normalize_stream_content(get_value(delta, "reasoning_text"));
    if !raw_reasoning.is_empty() {
        let mut ev = HashMap::new();
        ev.insert(
            "type".to_string(),
            Value::String("reasoning_raw_delta".to_string()),
        );
        ev.insert("text".to_string(), Value::String(raw_reasoning));
        ev.insert(
            "source_key".to_string(),
            Value::String("reasoning_text".to_string()),
        );
        events.push(ev);
    }
    let tool_calls = get_value(delta, "tool_calls");
    if !tool_calls.is_null() {
        let mut ev = HashMap::new();
        ev.insert(
            "type".to_string(),
            Value::String("tool_call_delta".to_string()),
        );
        ev.insert("delta".to_string(), tool_calls.clone());
        events.push(ev);
    }
    let finish_reason = get_value(choice, "finish_reason");
    if !finish_reason.is_null() {
        let mut ev = HashMap::new();
        ev.insert("type".to_string(), Value::String("finish".to_string()));
        ev.insert("finish_reason".to_string(), finish_reason.clone());
        events.push(ev);
    }
    events
}

pub fn response_failure_message(event: &Value, status: &str) -> String {
    let response = get_value(event, "response");
    let mut error = get_value(event, "error").clone();
    let mut incomplete_details = get_value(event, "incomplete_details").clone();

    if let Some(resp_obj) = response.as_object() {
        if let Some(e) = resp_obj.get("error") {
            if !e.is_null() {
                error = e.clone();
            }
        }
        if let Some(d) = resp_obj.get("incomplete_details") {
            if !d.is_null() {
                incomplete_details = d.clone();
            }
        }
    }

    let mut detail_parts: Vec<String> = Vec::new();

    match &error {
        Value::Object(map) => {
            let message = map
                .get("message")
                .or_else(|| map.get("code"))
                .or_else(|| map.get("type"));
            if let Some(Value::String(s)) = message {
                if !s.is_empty() {
                    detail_parts.push(s.clone());
                }
            }
        }
        Value::String(s) if !s.is_empty() => {
            detail_parts.push(s.clone());
        }
        _ => {}
    }

    match &incomplete_details {
        Value::Object(map) => {
            let reason = map.get("reason").or_else(|| map.get("message"));
            if let Some(Value::String(s)) = reason {
                if !s.is_empty() {
                    detail_parts.push(s.clone());
                }
            }
        }
        Value::String(s) if !s.is_empty() => {
            detail_parts.push(s.clone());
        }
        _ => {}
    }

    let detail = if detail_parts.is_empty() {
        let serialized = serde_json::to_string(event).unwrap_or_default();
        if serialized.len() > 500 {
            serialized[..500].to_string()
        } else {
            serialized
        }
    } else {
        detail_parts.join("; ")
    };

    format!("OpenAI protocol response {}: {}", status, detail)
}

pub fn reasoning_from_response_items(items: &[Value]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for item in items {
        if get_value(item, "type").as_str() != Some("reasoning") {
            continue;
        }
        for field in &["summary", "content"] {
            let value = get_value(item, field);
            match value {
                Value::String(s) if !s.is_empty() => {
                    parts.push(s.clone());
                }
                Value::Array(arr) => {
                    for part in arr {
                        match part {
                            Value::String(s) if !s.is_empty() => {
                                parts.push(s.clone());
                            }
                            Value::Object(map) => {
                                if let Some(Value::String(text)) = map.get("text") {
                                    if !text.is_empty() {
                                        parts.push(text.clone());
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }
    }
    parts.join("")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_normalize_stream_content_null() {
        assert_eq!(normalize_stream_content(&Value::Null), "");
    }

    #[test]
    fn test_normalize_stream_content_string() {
        assert_eq!(
            normalize_stream_content(&Value::String("hello".to_string())),
            "hello"
        );
    }

    #[test]
    fn test_normalize_stream_content_array() {
        let arr = json!([{"text": "hello "}, {"text": "world"}]);
        assert_eq!(normalize_stream_content(&arr), "hello world");
    }

    #[test]
    fn test_normalize_stream_content_number() {
        assert_eq!(normalize_stream_content(&json!(42)), "42");
    }

    #[test]
    fn test_normalize_openai_chunk_content() {
        let chunk = json!({
            "choices": [{"delta": {"content": "hi"}, "finish_reason": null}]
        });
        let events = normalize_openai_chat_completion_chunk(&chunk);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["type"], Value::String("content".to_string()));
        assert_eq!(events[0]["text"], Value::String("hi".to_string()));
    }

    #[test]
    fn test_normalize_openai_chunk_finish() {
        let chunk = json!({
            "choices": [{"delta": {}, "finish_reason": "stop"}]
        });
        let events = normalize_openai_chat_completion_chunk(&chunk);
        assert_eq!(events.len(), 1);
        assert_eq!(events[0]["type"], Value::String("finish".to_string()));
    }

    #[test]
    fn test_normalize_openai_chunk_empty_choices() {
        let chunk = json!({"choices": []});
        let events = normalize_openai_chat_completion_chunk(&chunk);
        assert!(events.is_empty());
    }

    #[test]
    fn test_normalize_openai_chunk_reasoning() {
        let chunk = json!({
            "choices": [{"delta": {"reasoning_content": "thinking..."}, "finish_reason": null}]
        });
        let events = normalize_openai_chat_completion_chunk(&chunk);
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0]["type"],
            Value::String("reasoning_delta".to_string())
        );
    }

    #[test]
    fn test_response_failure_message_with_error() {
        let event = json!({
            "error": {"message": "rate limit exceeded"}
        });
        let msg = response_failure_message(&event, "failed");
        assert_eq!(
            msg,
            "OpenAI protocol response failed: rate limit exceeded"
        );
    }

    #[test]
    fn test_response_failure_message_string_error() {
        let event = json!({"error": "something went wrong"});
        let msg = response_failure_message(&event, "incomplete");
        assert_eq!(
            msg,
            "OpenAI protocol response incomplete: something went wrong"
        );
    }

    #[test]
    fn test_response_failure_message_fallback() {
        let event = json!({"some_field": 123});
        let msg = response_failure_message(&event, "failed");
        assert!(msg.starts_with("OpenAI protocol response failed:"));
        assert!(msg.contains("some_field"));
    }

    #[test]
    fn test_response_failure_with_response_wrapper() {
        let event = json!({
            "response": {
                "error": {"message": "inner error"},
                "incomplete_details": {"reason": "max_tokens"}
            }
        });
        let msg = response_failure_message(&event, "failed");
        assert!(msg.contains("inner error"));
        assert!(msg.contains("max_tokens"));
    }

    #[test]
    fn test_reasoning_from_response_items_summary() {
        let items = vec![json!({
            "type": "reasoning",
            "summary": "thinking about it"
        })];
        assert_eq!(
            reasoning_from_response_items(&items),
            "thinking about it"
        );
    }

    #[test]
    fn test_reasoning_from_response_items_content_array() {
        let items = vec![json!({
            "type": "reasoning",
            "content": [{"text": "step 1"}, {"text": "step 2"}]
        })];
        assert_eq!(reasoning_from_response_items(&items), "step 1step 2");
    }

    #[test]
    fn test_reasoning_from_response_items_skips_non_reasoning() {
        let items = vec![
            json!({"type": "message", "content": "not reasoning"}),
            json!({"type": "reasoning", "summary": "yes"}),
        ];
        assert_eq!(reasoning_from_response_items(&items), "yes");
    }

    #[test]
    fn test_reasoning_from_response_items_empty() {
        let items: Vec<Value> = vec![];
        assert_eq!(reasoning_from_response_items(&items), "");
    }
}
