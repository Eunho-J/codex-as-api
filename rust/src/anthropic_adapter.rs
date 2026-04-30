use crate::messages::{AssistantResponse, Message, MessageRole, ToolCall, ToolSchema};
use serde_json::{json, Value};
use std::collections::HashMap;

pub fn anthropic_request_to_internal(
    body: &Value,
) -> (
    Vec<Message>,
    Option<Vec<ToolSchema>>,
    Option<Value>,
    Option<Vec<String>>,
    Option<String>,
) {
    let mut messages: Vec<Message> = Vec::new();

    let system = body.get("system");
    if let Some(sys) = system {
        let sys_text = extract_system_text(sys);
        if !sys_text.is_empty() {
            messages.push(Message {
                role: MessageRole::System,
                content: sys_text,
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
            });
        }
    }

    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let empty = Value::String(String::new());
            let content = msg.get("content").unwrap_or(&empty);
            if role == "user" {
                convert_user_message(content, &mut messages);
            } else if role == "assistant" {
                convert_assistant_message(content, &mut messages);
            }
        }
    }

    let tools = body
        .get("tools")
        .and_then(|v| v.as_array())
        .filter(|a| !a.is_empty())
        .map(|arr| convert_tools(arr));

    let tool_choice = body
        .get("tool_choice")
        .map(|tc| convert_tool_choice(tc));

    let stop = body
        .get("stop_sequences")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        });

    let reasoning_effort = body
        .get("thinking")
        .map(|t| convert_thinking(t))
        .flatten();

    (messages, tools, tool_choice, stop, reasoning_effort)
}

fn extract_system_text(system: &Value) -> String {
    match system {
        Value::String(s) => s.clone(),
        Value::Array(arr) => {
            let parts: Vec<String> = arr
                .iter()
                .filter_map(|block| {
                    if block.get("type").and_then(|v| v.as_str()) == Some("text") {
                        block
                            .get("text")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                    } else {
                        None
                    }
                })
                .collect();
            parts.join("\n\n")
        }
        _ => String::new(),
    }
}

fn convert_user_message(content: &Value, out: &mut Vec<Message>) {
    match content {
        Value::String(s) => {
            out.push(Message {
                role: MessageRole::User,
                content: s.clone(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
            });
        }
        Value::Array(arr) => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut image_urls: Vec<String> = Vec::new();
            for block in arr {
                let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match block_type {
                    "text" => {
                        if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                            text_parts.push(text.to_string());
                        }
                    }
                    "tool_result" => {
                        if !text_parts.is_empty() || !image_urls.is_empty() {
                            out.push(Message {
                                role: MessageRole::User,
                                content: text_parts.join(""),
                                tool_calls: vec![],
                                tool_call_id: None,
                                name: None,
                                reasoning_content: None,
                                images: std::mem::take(&mut image_urls),
                            });
                            text_parts = Vec::new();
                        }
                        let tool_use_id = block
                            .get("tool_use_id")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .unwrap_or("tool-call")
                            .to_string();
                        let raw_content = block.get("content").unwrap_or(&Value::Null);
                        let (result_content, tool_result_images) = extract_tool_result_content_with_images(raw_content);
                        out.push(Message {
                            role: MessageRole::Tool,
                            content: result_content,
                            tool_calls: vec![],
                            tool_call_id: Some(tool_use_id.clone()),
                            name: Some(tool_use_id),
                            reasoning_content: None,
                            images: vec![],
                        });
                        if !tool_result_images.is_empty() {
                            out.push(Message {
                                role: MessageRole::User,
                                content: String::new(),
                                tool_calls: vec![],
                                tool_call_id: None,
                                name: None,
                                reasoning_content: None,
                                images: tool_result_images,
                            });
                        }
                    }
                    "image" => {
                        if let Some(source) = block.get("source").and_then(|v| v.as_object()) {
                            if source.get("type").and_then(|v| v.as_str()) == Some("base64") {
                                let media_type = source
                                    .get("media_type")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("image/png");
                                let data = source
                                    .get("data")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("");
                                image_urls.push(format!(
                                    "data:{};base64,{}",
                                    media_type, data
                                ));
                            }
                        }
                    }
                    _ => {}
                }
            }
            if !text_parts.is_empty() || !image_urls.is_empty() {
                out.push(Message {
                    role: MessageRole::User,
                    content: text_parts.join(""),
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    reasoning_content: None,
                    images: image_urls,
                });
            }
        }
        _ => {}
    }
}

fn extract_tool_result_content_with_images(content: &Value) -> (String, Vec<String>) {
    match content {
        Value::String(s) => (s.clone(), vec![]),
        Value::Array(arr) => {
            let mut text_pieces: Vec<String> = Vec::new();
            let mut images: Vec<String> = Vec::new();
            for p in arr {
                let typ = p.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if typ == "text" {
                    if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                        text_pieces.push(t.to_string());
                    }
                } else if typ == "image" {
                    if let Some(source) = p.get("source").and_then(|v| v.as_object()) {
                        if source.get("type").and_then(|v| v.as_str()) == Some("base64") {
                            let media_type = source
                                .get("media_type")
                                .and_then(|v| v.as_str())
                                .unwrap_or("image/png");
                            let data = source
                                .get("data")
                                .and_then(|v| v.as_str())
                                .unwrap_or("");
                            images.push(format!("data:{};base64,{}", media_type, data));
                        }
                    }
                }
            }
            (text_pieces.join(""), images)
        }
        Value::Null => (String::new(), vec![]),
        other => (other.to_string(), vec![]),
    }
}

fn convert_assistant_message(content: &Value, out: &mut Vec<Message>) {
    match content {
        Value::String(s) => {
            out.push(Message {
                role: MessageRole::Assistant,
                content: s.clone(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
            });
        }
        Value::Array(arr) => {
            let mut text_parts: Vec<String> = Vec::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut reasoning_content: Option<String> = None;

            for block in arr {
                let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match block_type {
                    "text" => {
                        if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                            text_parts.push(text.to_string());
                        }
                    }
                    "tool_use" => {
                        let id = block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
                        let name = block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let arguments: HashMap<String, Value> = block
                            .get("input")
                            .and_then(|v| v.as_object())
                            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                            .unwrap_or_default();
                        tool_calls.push(ToolCall { id, name, arguments });
                    }
                    "thinking" => {
                        if let Some(thinking_text) = block.get("thinking").and_then(|v| v.as_str()) {
                            if !thinking_text.is_empty() {
                                reasoning_content = Some(thinking_text.to_string());
                            }
                        }
                    }
                    _ => {}
                }
            }

            out.push(Message {
                role: MessageRole::Assistant,
                content: text_parts.join(""),
                tool_calls,
                tool_call_id: None,
                name: None,
                reasoning_content,
                images: vec![],
            });
        }
        _ => {}
    }
}

fn convert_tools(tools: &[Value]) -> Vec<ToolSchema> {
    let mut result = Vec::new();
    for tool in tools {
        let name = match tool.get("name").and_then(|v| v.as_str()) {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => continue,
        };
        let description = tool
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let parameters = tool
            .get("input_schema")
            .cloned()
            .unwrap_or_else(|| json!({}));
        result.push(ToolSchema {
            name,
            description,
            parameters,
        });
    }
    result
}

fn convert_tool_choice(tc: &Value) -> Value {
    let tc_type = tc.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match tc_type {
        "auto" => json!("auto"),
        "any" => json!("required"),
        "tool" => {
            let name = tc.get("name").and_then(|v| v.as_str()).unwrap_or("");
            json!({"type": "function", "function": {"name": name}})
        }
        "none" => json!("none"),
        _ => json!("auto"),
    }
}

fn convert_thinking(thinking: &Value) -> Option<String> {
    match thinking.get("type").and_then(|v| v.as_str()) {
        Some("enabled") => Some("high".to_string()),
        Some("adaptive") => Some("medium".to_string()),
        _ => None,
    }
}

pub fn internal_response_to_anthropic(
    response: &AssistantResponse,
    model: &str,
    request_id: &str,
) -> Value {
    let mut content: Vec<Value> = Vec::new();

    if let Some(rc) = &response.reasoning_content {
        content.push(json!({
            "type": "thinking",
            "thinking": rc,
            "signature": "sig-placeholder",
        }));
    }

    if !response.content.is_empty() {
        content.push(json!({"type": "text", "text": response.content}));
    }

    for tc in &response.tool_calls {
        content.push(json!({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.name,
            "input": tc.arguments,
        }));
    }

    let stop_reason = map_stop_reason(&response.finish_reason, !response.tool_calls.is_empty());

    let usage_dict = match &response.usage {
        Some(u) => json!({
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": u.cached_tokens,
        }),
        None => json!({"input_tokens": 0, "output_tokens": 0}),
    };

    if content.is_empty() {
        content.push(json!({"type": "text", "text": ""}));
    }

    json!({
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": null,
        "usage": usage_dict,
    })
}

fn map_stop_reason(finish_reason: &str, has_tool_calls: bool) -> &'static str {
    if has_tool_calls {
        return "tool_use";
    }
    match finish_reason {
        "stop" => "end_turn",
        "length" | "max_tokens" => "max_tokens",
        "tool_calls" | "tool_use" => "tool_use",
        "stop_sequence" => "stop_sequence",
        _ => "end_turn",
    }
}

pub fn anthropic_stream_adapter(events: &[Value], model: &str, request_id: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();

    out.push(sse(
        "message_start",
        &json!({
            "type": "message_start",
            "message": {
                "id": request_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }),
    ));

    let mut block_index: u32 = 0;
    let mut current_block: Option<&'static str> = None;
    let mut has_any_content = false;
    let mut output_tokens: i64 = 0;

    for event in events {
        let typ = event.get("type").and_then(|v| v.as_str()).unwrap_or("");

        match typ {
            "reasoning_delta" | "reasoning_raw_delta" => {
                has_any_content = true;
                let text = event
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if current_block != Some("thinking") {
                    if current_block.is_some() {
                        out.push(sse(
                            "content_block_stop",
                            &json!({"type": "content_block_stop", "index": block_index}),
                        ));
                        block_index += 1;
                    }
                    out.push(sse(
                        "content_block_start",
                        &json!({
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
                        }),
                    ));
                    current_block = Some("thinking");
                }
                out.push(sse(
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "thinking_delta", "thinking": text},
                    }),
                ));
            }

            "content" => {
                has_any_content = true;
                let text = event
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                if current_block == Some("thinking") {
                    out.push(sse(
                        "content_block_delta",
                        &json!({
                            "type": "content_block_delta",
                            "index": block_index,
                            "delta": {"type": "signature_delta", "signature": "sig-placeholder"},
                        }),
                    ));
                    out.push(sse(
                        "content_block_stop",
                        &json!({"type": "content_block_stop", "index": block_index}),
                    ));
                    block_index += 1;
                    current_block = None;
                }
                if current_block != Some("text") {
                    if current_block.is_some() {
                        out.push(sse(
                            "content_block_stop",
                            &json!({"type": "content_block_stop", "index": block_index}),
                        ));
                        block_index += 1;
                    }
                    out.push(sse(
                        "content_block_start",
                        &json!({
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        }),
                    ));
                    current_block = Some("text");
                }
                out.push(sse(
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {"type": "text_delta", "text": text},
                    }),
                ));
            }

            "tool_call" => {
                has_any_content = true;
                if let Some(cb) = current_block {
                    if cb == "thinking" {
                        out.push(sse(
                            "content_block_delta",
                            &json!({
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "signature_delta", "signature": "sig-placeholder"},
                            }),
                        ));
                    }
                    out.push(sse(
                        "content_block_stop",
                        &json!({"type": "content_block_stop", "index": block_index}),
                    ));
                    block_index += 1;
                }
                let tool_id = event.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let tool_name = event.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let tool_args = event.get("arguments").cloned().unwrap_or_else(|| json!({}));
                out.push(sse(
                    "content_block_start",
                    &json!({
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name, "input": {}},
                    }),
                ));
                out.push(sse(
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": serde_json::to_string(&tool_args).unwrap_or_else(|_| "{}".to_string()),
                        },
                    }),
                ));
                out.push(sse(
                    "content_block_stop",
                    &json!({"type": "content_block_stop", "index": block_index}),
                ));
                block_index += 1;
                current_block = None;
            }

            "finish" => {
                if let Some(cb) = current_block {
                    if cb == "thinking" {
                        out.push(sse(
                            "content_block_delta",
                            &json!({
                                "type": "content_block_delta",
                                "index": block_index,
                                "delta": {"type": "signature_delta", "signature": "sig-placeholder"},
                            }),
                        ));
                    }
                    out.push(sse(
                        "content_block_stop",
                        &json!({"type": "content_block_stop", "index": block_index}),
                    ));
                    current_block = None;
                }

                if !has_any_content {
                    out.push(sse(
                        "content_block_start",
                        &json!({
                            "type": "content_block_start",
                            "index": block_index,
                            "content_block": {"type": "text", "text": ""},
                        }),
                    ));
                    out.push(sse(
                        "content_block_stop",
                        &json!({"type": "content_block_stop", "index": block_index}),
                    ));
                }

                let finish_reason = event
                    .get("finish_reason")
                    .and_then(|v| v.as_str())
                    .unwrap_or("stop");
                let stop_reason = map_stop_reason(finish_reason, false);

                if let Some(usage) = event.get("usage").and_then(|v| v.as_object()) {
                    output_tokens = usage
                        .get("output_tokens")
                        .or_else(|| usage.get("completion_tokens"))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(0);
                }

                out.push(sse(
                    "message_delta",
                    &json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                        "usage": {"output_tokens": output_tokens},
                    }),
                ));
                out.push(sse("message_stop", &json!({"type": "message_stop"})));
            }

            _ => {}
        }
    }

    out
}

fn sse(event_type: &str, data: &Value) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        event_type,
        serde_json::to_string(data).unwrap_or_else(|_| "{}".to_string())
    )
}

pub fn format_anthropic_error(status: u16, message: &str) -> Value {
    let error_type = match status {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "permission_error",
        404 => "not_found_error",
        429 => "rate_limit_error",
        500 => "api_error",
        529 => "overloaded_error",
        _ => "api_error",
    };
    json!({
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // Request conversion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_system_string() {
        let body = json!({
            "messages": [],
            "system": "You are a helpful assistant.",
        });
        let (msgs, _, _, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, MessageRole::System);
        assert_eq!(msgs[0].content, "You are a helpful assistant.");
    }

    #[test]
    fn test_system_blocks() {
        let body = json!({
            "messages": [],
            "system": [
                {"type": "text", "text": "Part one."},
                {"type": "text", "text": "Part two."},
            ],
        });
        let (msgs, _, _, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(msgs[0].content, "Part one.\n\nPart two.");
    }

    #[test]
    fn test_system_blocks_skips_non_text() {
        let body = json!({
            "messages": [],
            "system": [
                {"type": "image", "source": {}},
                {"type": "text", "text": "Only text."},
            ],
        });
        let (msgs, _, _, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(msgs[0].content, "Only text.");
    }

    #[test]
    fn test_user_text_string() {
        let body = json!({
            "messages": [{"role": "user", "content": "Hello"}],
        });
        let (msgs, _, _, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, MessageRole::User);
        assert_eq!(msgs[0].content, "Hello");
    }

    #[test]
    fn test_user_text_content_blocks() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Hello "},
                {"type": "text", "text": "world"},
            ]}],
        });
        let (msgs, _, _, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content, "Hello world");
    }

    #[test]
    fn test_user_tool_result_flushes_text() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Before"},
                {"type": "tool_result", "tool_use_id": "call-1", "content": "result"},
            ]}],
        });
        let (msgs, _, _, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, MessageRole::User);
        assert_eq!(msgs[0].content, "Before");
        assert_eq!(msgs[1].role, MessageRole::Tool);
        assert_eq!(msgs[1].tool_call_id, Some("call-1".to_string()));
        assert_eq!(msgs[1].content, "result");
    }

    #[test]
    fn test_user_tool_result_list_content() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call-2", "content": [
                    {"type": "text", "text": "part A"},
                    {"type": "text", "text": " part B"},
                ]},
            ]}],
        });
        let (msgs, _, _, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, MessageRole::Tool);
        assert_eq!(msgs[0].content, "part A part B");
    }

    #[test]
    fn test_user_tool_result_with_image() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call-img", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo"}},
                ]},
            ]}],
        });
        let (msgs, _, _, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, MessageRole::Tool);
        assert_eq!(msgs[0].tool_call_id, Some("call-img".to_string()));
        assert_eq!(msgs[0].content, "");
        assert_eq!(msgs[1].role, MessageRole::User);
        assert_eq!(msgs[1].images.len(), 1);
        assert_eq!(msgs[1].images[0], "data:image/png;base64,iVBORw0KGgo");
    }

    #[test]
    fn test_user_tool_result_with_text_and_image() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call-mix", "content": [
                    {"type": "text", "text": "file contents"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQ"}},
                ]},
            ]}],
        });
        let (msgs, _, _, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, MessageRole::Tool);
        assert_eq!(msgs[0].content, "file contents");
        assert_eq!(msgs[1].role, MessageRole::User);
        assert_eq!(msgs[1].images[0], "data:image/jpeg;base64,/9j/4AAQ");
    }

    #[test]
    fn test_user_tool_result_default_id() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "", "content": "x"},
            ]}],
        });
        let (msgs, _, _, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(msgs[0].tool_call_id, Some("tool-call".to_string()));
    }

    #[test]
    fn test_assistant_text_string() {
        let body = json!({
            "messages": [{"role": "assistant", "content": "I can help."}],
        });
        let (msgs, _, _, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(msgs[0].role, MessageRole::Assistant);
        assert_eq!(msgs[0].content, "I can help.");
    }

    #[test]
    fn test_assistant_text_and_tool_use() {
        let body = json!({
            "messages": [{"role": "assistant", "content": [
                {"type": "text", "text": "Calling tool."},
                {"type": "tool_use", "id": "tc-1", "name": "search", "input": {"q": "rust"}},
            ]}],
        });
        let (msgs, _, _, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(msgs[0].content, "Calling tool.");
        assert_eq!(msgs[0].tool_calls.len(), 1);
        assert_eq!(msgs[0].tool_calls[0].id, "tc-1");
        assert_eq!(msgs[0].tool_calls[0].name, "search");
        assert_eq!(
            msgs[0].tool_calls[0].arguments.get("q"),
            Some(&json!("rust"))
        );
    }

    #[test]
    fn test_assistant_thinking_block() {
        let body = json!({
            "messages": [{"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "Answer."},
            ]}],
        });
        let (msgs, _, _, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(msgs[0].reasoning_content, Some("Let me think...".to_string()));
        assert_eq!(msgs[0].content, "Answer.");
    }

    #[test]
    fn test_tools_conversion() {
        let body = json!({
            "messages": [],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {"location": {"type": "string"}}},
            }],
        });
        let (_, tools, _, _, _) = anthropic_request_to_internal(&body);
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].description, "Get weather");
        assert!(tools[0].parameters.get("properties").is_some());
    }

    #[test]
    fn test_tool_choice_auto() {
        let body = json!({"messages": [], "tool_choice": {"type": "auto"}});
        let (_, _, tc, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(tc, Some(json!("auto")));
    }

    #[test]
    fn test_tool_choice_any() {
        let body = json!({"messages": [], "tool_choice": {"type": "any"}});
        let (_, _, tc, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(tc, Some(json!("required")));
    }

    #[test]
    fn test_tool_choice_tool() {
        let body = json!({"messages": [], "tool_choice": {"type": "tool", "name": "my_fn"}});
        let (_, _, tc, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(
            tc,
            Some(json!({"type": "function", "function": {"name": "my_fn"}}))
        );
    }

    #[test]
    fn test_tool_choice_none() {
        let body = json!({"messages": [], "tool_choice": {"type": "none"}});
        let (_, _, tc, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(tc, Some(json!("none")));
    }

    #[test]
    fn test_tool_choice_absent() {
        let body = json!({"messages": []});
        let (_, _, tc, _, _) = anthropic_request_to_internal(&body);
        assert_eq!(tc, None);
    }

    #[test]
    fn test_thinking_enabled() {
        let body = json!({"messages": [], "thinking": {"type": "enabled"}});
        let (_, _, _, _, effort) = anthropic_request_to_internal(&body);
        assert_eq!(effort, Some("high".to_string()));
    }

    #[test]
    fn test_thinking_adaptive() {
        let body = json!({"messages": [], "thinking": {"type": "adaptive"}});
        let (_, _, _, _, effort) = anthropic_request_to_internal(&body);
        assert_eq!(effort, Some("medium".to_string()));
    }

    #[test]
    fn test_thinking_disabled() {
        let body = json!({"messages": [], "thinking": {"type": "disabled"}});
        let (_, _, _, _, effort) = anthropic_request_to_internal(&body);
        assert_eq!(effort, None);
    }

    #[test]
    fn test_stop_sequences() {
        let body = json!({"messages": [], "stop_sequences": ["STOP", "END"]});
        let (_, _, _, stop, _) = anthropic_request_to_internal(&body);
        assert_eq!(stop, Some(vec!["STOP".to_string(), "END".to_string()]));
    }

    // -----------------------------------------------------------------------
    // Non-streaming response tests
    // -----------------------------------------------------------------------

    fn make_response(
        content: &str,
        tool_calls: Vec<ToolCall>,
        finish_reason: &str,
        usage: Option<crate::messages::Usage>,
        reasoning_content: Option<String>,
    ) -> AssistantResponse {
        AssistantResponse {
            content: content.to_string(),
            tool_calls,
            finish_reason: finish_reason.to_string(),
            usage,
            reasoning_content,
            raw: None,
        }
    }

    #[test]
    fn test_response_text_only() {
        let resp = make_response("Hello!", vec![], "stop", None, None);
        let out = internal_response_to_anthropic(&resp, "claude-3", "msg_abc");
        assert_eq!(out["id"], "msg_abc");
        assert_eq!(out["role"], "assistant");
        assert_eq!(out["stop_reason"], "end_turn");
        assert_eq!(out["content"][0]["type"], "text");
        assert_eq!(out["content"][0]["text"], "Hello!");
    }

    #[test]
    fn test_response_tool_use() {
        let tc = ToolCall {
            id: "tc-1".to_string(),
            name: "search".to_string(),
            arguments: [("q".to_string(), json!("rust"))].into_iter().collect(),
        };
        let resp = make_response("", vec![tc], "tool_calls", None, None);
        let out = internal_response_to_anthropic(&resp, "claude-3", "msg_xyz");
        assert_eq!(out["stop_reason"], "tool_use");
        let content = out["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["name"], "search");
    }

    #[test]
    fn test_response_reasoning() {
        let resp = make_response("Answer.", vec![], "stop", None, Some("My reasoning.".to_string()));
        let out = internal_response_to_anthropic(&resp, "claude-3", "msg_r");
        let content = out["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "thinking");
        assert_eq!(content[0]["thinking"], "My reasoning.");
        assert_eq!(content[1]["type"], "text");
    }

    #[test]
    fn test_response_empty_content_gets_text_block() {
        let resp = make_response("", vec![], "stop", None, None);
        let out = internal_response_to_anthropic(&resp, "claude-3", "msg_e");
        let content = out["content"].as_array().unwrap();
        assert_eq!(content.len(), 1);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[0]["text"], "");
    }

    #[test]
    fn test_response_usage_present() {
        let usage = crate::messages::Usage::new(100, 50, None, 20);
        let resp = make_response("Hi", vec![], "stop", Some(usage), None);
        let out = internal_response_to_anthropic(&resp, "claude-3", "msg_u");
        assert_eq!(out["usage"]["input_tokens"], 100);
        assert_eq!(out["usage"]["output_tokens"], 50);
        assert_eq!(out["usage"]["cache_read_input_tokens"], 20);
        assert_eq!(out["usage"]["cache_creation_input_tokens"], 0);
    }

    #[test]
    fn test_response_usage_absent() {
        let resp = make_response("Hi", vec![], "stop", None, None);
        let out = internal_response_to_anthropic(&resp, "claude-3", "msg_nu");
        assert_eq!(out["usage"]["input_tokens"], 0);
        assert_eq!(out["usage"]["output_tokens"], 0);
    }

    #[test]
    fn test_response_stop_reason_length() {
        let resp = make_response("truncated", vec![], "length", None, None);
        let out = internal_response_to_anthropic(&resp, "m", "id");
        assert_eq!(out["stop_reason"], "max_tokens");
    }

    #[test]
    fn test_response_stop_reason_max_tokens() {
        let resp = make_response("truncated", vec![], "max_tokens", None, None);
        let out = internal_response_to_anthropic(&resp, "m", "id");
        assert_eq!(out["stop_reason"], "max_tokens");
    }

    // -----------------------------------------------------------------------
    // Streaming adapter tests
    // -----------------------------------------------------------------------

    fn get_sse_events(chunks: &[String]) -> Vec<(String, Value)> {
        chunks
            .iter()
            .map(|chunk| {
                let lines: Vec<&str> = chunk.trim_end_matches('\n').lines().collect();
                let event_line = lines[0].strip_prefix("event: ").unwrap_or("").to_string();
                let data_line = lines[1].strip_prefix("data: ").unwrap_or("{}");
                let data: Value = serde_json::from_str(data_line).unwrap();
                (event_line, data)
            })
            .collect()
    }

    #[test]
    fn test_stream_text_only() {
        let events = vec![
            json!({"type": "content", "text": "Hello"}),
            json!({"type": "finish", "finish_reason": "stop"}),
        ];
        let chunks = anthropic_stream_adapter(&events, "claude-3", "msg_s1");
        let parsed = get_sse_events(&chunks);
        assert_eq!(parsed[0].0, "message_start");
        assert_eq!(parsed[1].0, "content_block_start");
        assert_eq!(parsed[1].1["content_block"]["type"], "text");
        assert_eq!(parsed[2].0, "content_block_delta");
        assert_eq!(parsed[2].1["delta"]["text"], "Hello");
        assert_eq!(parsed[3].0, "content_block_stop");
        assert_eq!(parsed[4].0, "message_delta");
        assert_eq!(parsed[4].1["delta"]["stop_reason"], "end_turn");
        assert_eq!(parsed[5].0, "message_stop");
    }

    #[test]
    fn test_stream_thinking_then_text() {
        let events = vec![
            json!({"type": "reasoning_delta", "text": "thinking..."}),
            json!({"type": "content", "text": "answer"}),
            json!({"type": "finish", "finish_reason": "stop"}),
        ];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);
        assert_eq!(parsed[1].0, "content_block_start");
        assert_eq!(parsed[1].1["content_block"]["type"], "thinking");
        assert_eq!(parsed[2].0, "content_block_delta");
        assert_eq!(parsed[2].1["delta"]["type"], "thinking_delta");
        assert_eq!(parsed[3].0, "content_block_delta");
        assert_eq!(parsed[3].1["delta"]["type"], "signature_delta");
        assert_eq!(parsed[4].0, "content_block_stop");
        assert_eq!(parsed[5].0, "content_block_start");
        assert_eq!(parsed[5].1["content_block"]["type"], "text");
    }

    #[test]
    fn test_stream_tool_call() {
        let events = vec![
            json!({"type": "tool_call", "id": "tc-1", "name": "search", "arguments": {"q": "rust"}}),
            json!({"type": "finish", "finish_reason": "tool_calls"}),
        ];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);
        assert_eq!(parsed[1].0, "content_block_start");
        assert_eq!(parsed[1].1["content_block"]["type"], "tool_use");
        assert_eq!(parsed[1].1["content_block"]["name"], "search");
        assert_eq!(parsed[2].0, "content_block_delta");
        assert_eq!(parsed[2].1["delta"]["type"], "input_json_delta");
        assert_eq!(parsed[3].0, "content_block_stop");
        assert_eq!(parsed[4].0, "message_delta");
        assert_eq!(parsed[4].1["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn test_stream_text_then_tool() {
        let events = vec![
            json!({"type": "content", "text": "First"}),
            json!({"type": "tool_call", "id": "tc-2", "name": "fn", "arguments": {}}),
            json!({"type": "finish", "finish_reason": "stop"}),
        ];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);
        let event_types: Vec<&str> = parsed.iter().map(|(e, _)| e.as_str()).collect();
        assert!(event_types.contains(&"content_block_start"));
        let text_start = parsed
            .iter()
            .find(|(e, d)| e == "content_block_start" && d["content_block"]["type"] == "text");
        assert!(text_start.is_some());
        let tool_start = parsed
            .iter()
            .find(|(e, d)| e == "content_block_start" && d["content_block"]["type"] == "tool_use");
        assert!(tool_start.is_some());
    }

    #[test]
    fn test_stream_empty_emits_text_block() {
        let events = vec![json!({"type": "finish", "finish_reason": "stop"})];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);
        let has_empty_text = parsed
            .iter()
            .any(|(e, d)| e == "content_block_start" && d["content_block"]["type"] == "text");
        assert!(has_empty_text);
    }

    #[test]
    fn test_stream_stop_reason_length() {
        let events = vec![json!({"type": "finish", "finish_reason": "length"})];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);
        let msg_delta = parsed.iter().find(|(e, _)| e == "message_delta").unwrap();
        assert_eq!(msg_delta.1["delta"]["stop_reason"], "max_tokens");
    }

    #[test]
    fn test_stream_usage_passed_through() {
        let events = vec![json!({
            "type": "finish",
            "finish_reason": "stop",
            "usage": {"output_tokens": 42, "completion_tokens": 0},
        })];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);
        let msg_delta = parsed.iter().find(|(e, _)| e == "message_delta").unwrap();
        assert_eq!(msg_delta.1["usage"]["output_tokens"], 42);
    }

    #[test]
    fn test_stream_multiple_tool_calls() {
        let events = vec![
            json!({"type": "tool_call", "id": "tc-1", "name": "fn1", "arguments": {}}),
            json!({"type": "tool_call", "id": "tc-2", "name": "fn2", "arguments": {}}),
            json!({"type": "finish", "finish_reason": "tool_calls"}),
        ];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);
        let tool_starts: Vec<_> = parsed
            .iter()
            .filter(|(e, d)| e == "content_block_start" && d["content_block"]["type"] == "tool_use")
            .collect();
        assert_eq!(tool_starts.len(), 2);
        assert_eq!(tool_starts[0].1["content_block"]["name"], "fn1");
        assert_eq!(tool_starts[1].1["content_block"]["name"], "fn2");
    }

    // -----------------------------------------------------------------------
    // Error formatting tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_error_400() {
        let e = format_anthropic_error(400, "bad request");
        assert_eq!(e["type"], "error");
        assert_eq!(e["error"]["type"], "invalid_request_error");
        assert_eq!(e["error"]["message"], "bad request");
    }

    #[test]
    fn test_error_401() {
        let e = format_anthropic_error(401, "unauthorized");
        assert_eq!(e["error"]["type"], "authentication_error");
    }

    #[test]
    fn test_error_403() {
        let e = format_anthropic_error(403, "forbidden");
        assert_eq!(e["error"]["type"], "permission_error");
    }

    #[test]
    fn test_error_404() {
        let e = format_anthropic_error(404, "not found");
        assert_eq!(e["error"]["type"], "not_found_error");
    }

    #[test]
    fn test_error_429() {
        let e = format_anthropic_error(429, "rate limited");
        assert_eq!(e["error"]["type"], "rate_limit_error");
    }

    #[test]
    fn test_error_500() {
        let e = format_anthropic_error(500, "server error");
        assert_eq!(e["error"]["type"], "api_error");
    }

    #[test]
    fn test_error_529() {
        let e = format_anthropic_error(529, "overloaded");
        assert_eq!(e["error"]["type"], "overloaded_error");
    }

    #[test]
    fn test_error_unknown_status() {
        let e = format_anthropic_error(503, "unavailable");
        assert_eq!(e["error"]["type"], "api_error");
    }
}
