use serde_json::Value;

pub fn get_value<'a>(value: &'a Value, key: &str) -> &'a Value {
    match value {
        Value::Object(map) => map.get(key).unwrap_or(&Value::Null),
        _ => &Value::Null,
    }
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
        serde_json::to_string(event)
            .expect("serde_json::Value must serialize")
            .chars()
            .take(500)
            .collect()
    } else {
        detail_parts.join("; ")
    };

    format!("OpenAI protocol response {}: {}", status, detail)
}

pub fn reasoning_parts_from_response_items(items: &[Value]) -> Result<(String, String), String> {
    let mut summary_parts: Vec<String> = Vec::new();
    let mut content_parts: Vec<String> = Vec::new();
    for (item_index, item) in items.iter().enumerate() {
        let object = item
            .as_object()
            .ok_or_else(|| format!("response output item {item_index} must be an object"))?;
        let item_type = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("response output item {item_index} requires a string type"))?;
        if item_type != "reasoning" {
            continue;
        }
        let summary = object
            .get("summary")
            .and_then(Value::as_array)
            .ok_or_else(|| "reasoning summary must be an array".to_string())?;
        for (field, value, expected_types, is_summary) in [
            ("summary", Some(summary), &["summary_text"][..], true),
            (
                "content",
                object
                    .get("content")
                    .filter(|value| !value.is_null())
                    .map(|value| {
                        value
                            .as_array()
                            .ok_or_else(|| "reasoning content must be an array or null".to_string())
                    })
                    .transpose()?,
                &["reasoning_text", "text"][..],
                false,
            ),
        ] {
            let Some(value) = value else {
                continue;
            };
            for (part_index, part) in value.iter().enumerate() {
                let map = part.as_object().ok_or_else(|| {
                    format!("reasoning {field} item {part_index} must be an object")
                })?;
                if !map
                    .get("type")
                    .and_then(Value::as_str)
                    .is_some_and(|part_type| expected_types.contains(&part_type))
                {
                    return Err(format!(
                        "reasoning {field} item {part_index} has an unsupported type"
                    ));
                }
                let text = map.get("text").and_then(Value::as_str).ok_or_else(|| {
                    format!("reasoning {field} item {part_index} requires string text")
                })?;
                if is_summary {
                    summary_parts.push(text.to_string());
                } else {
                    content_parts.push(text.to_string());
                }
            }
        }
    }
    Ok((summary_parts.join(""), content_parts.join("")))
}

pub fn reasoning_from_response_items(items: &[Value]) -> Result<String, String> {
    let (summary, content) = reasoning_parts_from_response_items(items)?;
    Ok(summary + &content)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn get_value_returns_null_for_non_object() {
        assert_eq!(get_value(&Value::Null, "missing"), &Value::Null);
    }

    #[test]
    fn get_value_returns_null_for_missing_field() {
        assert_eq!(
            get_value(&json!({"present": true}), "missing"),
            &Value::Null
        );
    }

    #[test]
    fn response_failure_uses_error_code_when_message_is_absent() {
        let event = json!({"error": {"code": "quota_exceeded"}});
        assert_eq!(
            response_failure_message(&event, "failed"),
            "OpenAI protocol response failed: quota_exceeded"
        );
    }

    #[test]
    fn response_failure_uses_string_incomplete_details() {
        let event = json!({"incomplete_details": "upstream stopped"});
        assert_eq!(
            response_failure_message(&event, "incomplete"),
            "OpenAI protocol response incomplete: upstream stopped"
        );
    }

    #[test]
    fn response_failure_fallback_is_bounded() {
        let event = json!({"payload": "x".repeat(800)});
        let message = response_failure_message(&event, "failed");
        assert!(message.len() < 600);
    }

    #[test]
    fn reasoning_reads_summary_arrays() {
        let items = vec![json!({
            "type": "reasoning",
            "summary": [
                {"type": "summary_text", "text": "one"},
                {"type": "summary_text", "text": "two"}
            ]
        })];
        assert_eq!(reasoning_from_response_items(&items).unwrap(), "onetwo");
    }

    #[test]
    fn reasoning_reads_typed_content() {
        for part_type in ["reasoning_text", "text"] {
            let items = vec![json!({
                "type": "reasoning",
                "summary": [],
                "content": [{"type": part_type, "text": "detail"}]
            })];
            assert_eq!(reasoning_from_response_items(&items).unwrap(), "detail");
        }
    }

    #[test]
    fn malformed_reasoning_items_fail_instead_of_returning_partial_text() {
        let malformed = [
            json!("reasoning"),
            json!({"type": 7, "summary": "partial"}),
            json!({"type": "reasoning", "summary": 42, "content": "partial"}),
            json!({"type": "reasoning", "summary": [42], "content": "partial"}),
            json!({"type": "reasoning", "summary": [{"other": "missing text"}], "content": "partial"}),
            json!({"type": "reasoning", "summary": [{"text": 42}], "content": "partial"}),
        ];
        for item in malformed {
            assert!(reasoning_from_response_items(&[item]).is_err());
        }
    }

    #[test]
    fn test_response_failure_message_with_error() {
        let event = json!({
            "error": {"message": "rate limit exceeded"}
        });
        let msg = response_failure_message(&event, "failed");
        assert_eq!(msg, "OpenAI protocol response failed: rate limit exceeded");
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
            "summary": [{"type": "summary_text", "text": "thinking about it"}]
        })];
        assert_eq!(
            reasoning_from_response_items(&items).unwrap(),
            "thinking about it"
        );
    }

    #[test]
    fn test_reasoning_from_response_items_content_array() {
        let items = vec![json!({
            "type": "reasoning",
            "summary": [],
            "content": [
                {"type": "reasoning_text", "text": "step 1"},
                {"type": "text", "text": "step 2"}
            ]
        })];
        assert_eq!(
            reasoning_from_response_items(&items).unwrap(),
            "step 1step 2"
        );
    }

    #[test]
    fn test_reasoning_from_response_items_skips_non_reasoning() {
        let items = vec![
            json!({"type": "message", "content": "not reasoning"}),
            json!({"type": "reasoning", "summary": [{"type": "summary_text", "text": "yes"}]}),
        ];
        assert_eq!(reasoning_from_response_items(&items).unwrap(), "yes");
    }

    #[test]
    fn test_reasoning_from_response_items_empty() {
        let items: Vec<Value> = vec![];
        assert_eq!(reasoning_from_response_items(&items).unwrap(), "");
    }

    #[test]
    fn test_reasoning_from_response_items_requires_summary_array() {
        let items = vec![json!({
            "type": "reasoning",
            "summary": null,
            "content": null
        })];
        assert!(reasoning_from_response_items(&items).is_err());
    }
}
