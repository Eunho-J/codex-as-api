use crate::messages::{AssistantResponse, Message, MessageRole, ToolCall, ToolSchema};
use serde_json::{json, Value};
use std::collections::HashSet;

pub type AnthropicInternalRequest = (
    Vec<Message>,
    Option<Vec<ToolSchema>>,
    Option<Value>,
    Option<Vec<String>>,
    Option<String>,
    Option<Value>,
    Option<bool>,
);

pub fn anthropic_request_to_internal(body: &Value) -> Result<AnthropicInternalRequest, String> {
    body.as_object()
        .ok_or_else(|| "Anthropic request body must be an object".to_string())?;
    reject_explicit_null_top_level_fields(body)?;
    validate_anthropic_cache_controls(body, true)?;
    let mut messages: Vec<Message> = Vec::new();

    let system = body.get("system");
    if let Some(sys) = system {
        let sys_text = extract_system_text(sys)?;
        if !sys_text.is_empty() {
            messages.push(Message {
                role: MessageRole::System,
                content: sys_text,
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            });
        }
    }

    let request_messages = body
        .get("messages")
        .and_then(Value::as_array)
        .filter(|messages| !messages.is_empty())
        .ok_or_else(|| "messages must be a non-empty array".to_string())?;
    for (index, msg) in request_messages.iter().enumerate() {
        let object = msg
            .as_object()
            .ok_or_else(|| format!("messages[{index}] must be an object"))?;
        reject_unknown_fields(object, &format!("messages[{index}]"), &["role", "content"])?;
        let role = object
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("messages[{index}].role must be a string"))?;
        let content = object
            .get("content")
            .ok_or_else(|| format!("messages[{index}].content is required"))?;
        match role {
            "user" => convert_user_message(content, &mut messages)?,
            "assistant" => convert_assistant_message(content, &mut messages)?,
            _ => return Err(format!("messages[{index}].role must be user or assistant")),
        }
    }

    let tools = match body.get("tools") {
        None => None,
        Some(Value::Array(tools)) if tools.is_empty() => None,
        Some(Value::Array(tools)) => {
            validate_anthropic_tool_controls(tools)?;
            Some(convert_tools(tools)?)
        }
        Some(_) => return Err("tools must be an array".to_string()),
    };

    let (tool_choice, parallel_tool_calls) = match body.get("tool_choice") {
        None => (None, None),
        Some(tool_choice) => {
            let (tool_choice, parallel_tool_calls) = convert_tool_choice(tool_choice)?;
            (Some(tool_choice), parallel_tool_calls)
        }
    };

    let stop = match body.get("stop_sequences") {
        None => None,
        Some(Value::Array(values)) => {
            let mut result = Vec::with_capacity(values.len());
            for (index, value) in values.iter().enumerate() {
                let value = value
                    .as_str()
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| format!("stop_sequences[{index}] must be a non-empty string"))?;
                result.push(value.to_string());
            }
            Some(result)
        }
        Some(_) => return Err("stop_sequences must be an array".to_string()),
    };

    let reasoning_effort = convert_reasoning_effort(body)?;
    let text = match anthropic_output_format_from_body(body)? {
        Some(output_format) => anthropic_output_format_to_openai_text(output_format)?,
        None => None,
    };

    Ok((
        messages,
        tools,
        tool_choice,
        stop,
        reasoning_effort,
        text,
        parallel_tool_calls,
    ))
}

fn reject_explicit_null_top_level_fields(body: &Value) -> Result<(), String> {
    for field in [
        "system",
        "tools",
        "tool_choice",
        "stop_sequences",
        "thinking",
        "output_config",
        "stream",
        "service_tier",
    ] {
        if body.get(field) == Some(&Value::Null) {
            return Err(format!("{field} must not be null"));
        }
    }
    Ok(())
}

fn reject_unknown_fields(
    object: &serde_json::Map<String, Value>,
    path: &str,
    allowed: &[&str],
) -> Result<(), String> {
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(format!("{path}.{field} is not supported"));
    }
    Ok(())
}

fn validate_nullable_unrepresentable_field(
    object: &serde_json::Map<String, Value>,
    field: &str,
    path: &str,
) -> Result<(), String> {
    if object.get(field).is_some_and(|value| !value.is_null()) {
        return Err(format!(
            "{path}.{field} cannot be represented by this facade"
        ));
    }
    Ok(())
}

pub(crate) fn validate_anthropic_cache_controls(
    body: &Value,
    allow_cache_control: bool,
) -> Result<(), String> {
    validate_cache_control_at(body, "cache_control", allow_cache_control)?;

    if let Some(system) = body.get("system") {
        match system {
            Value::Array(blocks) => {
                for (index, block) in blocks.iter().enumerate() {
                    validate_cache_control_at(
                        block,
                        &format!("system[{index}].cache_control"),
                        allow_cache_control,
                    )?;
                }
            }
            Value::Object(_) => {
                validate_cache_control_at(system, "system.cache_control", allow_cache_control)?;
            }
            _ => {}
        }
    }

    if let Some(messages) = body.get("messages").and_then(Value::as_array) {
        for (message_index, message) in messages.iter().enumerate() {
            if let Some(blocks) = message.get("content").and_then(Value::as_array) {
                validate_message_content_cache_controls(
                    blocks,
                    message_index,
                    "content",
                    allow_cache_control,
                )?;
            }
        }
    }

    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for (index, tool) in tools.iter().enumerate() {
            validate_cache_control_at(
                tool,
                &format!("tools[{index}].cache_control"),
                allow_cache_control,
            )?;
        }
    }

    Ok(())
}

fn validate_message_content_cache_controls(
    blocks: &[Value],
    message_index: usize,
    path: &str,
    allow_cache_control: bool,
) -> Result<(), String> {
    for (block_index, block) in blocks.iter().enumerate() {
        let block_path = format!("messages[{message_index}].{path}[{block_index}]");
        validate_cache_control_at(
            block,
            &format!("{block_path}.cache_control"),
            allow_cache_control,
        )?;
        if let Some(nested) = block.get("content").and_then(Value::as_array) {
            validate_message_content_cache_controls(
                nested,
                message_index,
                &format!("{path}[{block_index}].content"),
                allow_cache_control,
            )?;
        }
    }
    Ok(())
}

fn validate_cache_control_at(
    value: &Value,
    location: &str,
    allow_cache_control: bool,
) -> Result<(), String> {
    let Some(cache_control) = value.get("cache_control") else {
        return Ok(());
    };
    if cache_control.is_null() {
        return Ok(());
    }
    if !allow_cache_control {
        return Err(format!(
            "{location} is accepted without forwarding only for Claude Code requests"
        ));
    }
    let object = cache_control
        .as_object()
        .ok_or_else(|| format!("{location} must be an object"))?;
    if let Some(field) = object
        .keys()
        .find(|field| !matches!(field.as_str(), "type" | "ttl"))
    {
        return Err(format!("{location}.{field} is not supported"));
    }
    if object.get("type").and_then(Value::as_str) != Some("ephemeral") {
        return Err(format!("{location}.type must be ephemeral"));
    }
    match object.get("ttl") {
        None => Ok(()),
        Some(Value::String(ttl)) if matches!(ttl.as_str(), "5m" | "1h") => Ok(()),
        Some(_) => Err(format!("{location}.ttl must be one of: 5m, 1h")),
    }
}

fn extract_system_text(system: &Value) -> Result<String, String> {
    match system {
        Value::String(s) => Ok(s.clone()),
        Value::Array(arr) => {
            let mut parts = Vec::with_capacity(arr.len());
            for (index, block) in arr.iter().enumerate() {
                let object = block
                    .as_object()
                    .ok_or_else(|| format!("system[{index}] must be an object"))?;
                validate_nullable_unrepresentable_field(
                    object,
                    "citations",
                    &format!("system[{index}]"),
                )?;
                reject_unknown_fields(
                    object,
                    &format!("system[{index}]"),
                    &["type", "text", "cache_control", "citations"],
                )?;
                if object.get("type").and_then(Value::as_str) != Some("text") {
                    return Err(format!("system[{index}].type must be text"));
                }
                let text = object
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("system[{index}].text must be a string"))?;
                parts.push(text.to_string());
            }
            Ok(parts.join("\n\n"))
        }
        _ => Err("system must be a string or array of text blocks".to_string()),
    }
}

fn convert_user_message(content: &Value, out: &mut Vec<Message>) -> Result<(), String> {
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
                structured_content: None,
            });
        }
        Value::Array(arr) => {
            if arr.is_empty() {
                return Err("user message content must be a non-empty array".to_string());
            }
            let initial_message_count = out.len();
            let mut text_parts: Vec<String> = Vec::new();
            let mut image_urls: Vec<String> = Vec::new();
            for (block_index, block) in arr.iter().enumerate() {
                let object = block
                    .as_object()
                    .ok_or_else(|| format!("user content block {block_index} must be an object"))?;
                let block_type = object.get("type").and_then(Value::as_str).ok_or_else(|| {
                    format!("user content block {block_index} requires a string type")
                })?;
                match block_type {
                    "text" => {
                        validate_nullable_unrepresentable_field(
                            object,
                            "citations",
                            &format!("messages user content[{block_index}]"),
                        )?;
                        reject_unknown_fields(
                            object,
                            &format!("messages user content[{block_index}]"),
                            &["type", "text", "cache_control", "citations"],
                        )?;
                        let text = object.get("text").and_then(Value::as_str).ok_or_else(|| {
                            format!("user content block {block_index} requires string text")
                        })?;
                        text_parts.push(text.to_string());
                    }
                    "tool_result" => {
                        reject_unknown_fields(
                            object,
                            &format!("messages user content[{block_index}]"),
                            &[
                                "type",
                                "tool_use_id",
                                "content",
                                "is_error",
                                "cache_control",
                            ],
                        )?;
                        if !text_parts.is_empty() || !image_urls.is_empty() {
                            out.push(Message {
                                role: MessageRole::User,
                                content: text_parts.join(""),
                                tool_calls: vec![],
                                tool_call_id: None,
                                name: None,
                                reasoning_content: None,
                                images: std::mem::take(&mut image_urls),
                                structured_content: None,
                            });
                            text_parts = Vec::new();
                        }
                        let tool_use_id = block
                            .get("tool_use_id")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| {
                                format!(
                                    "user tool_result block {block_index} requires string tool_use_id"
                                )
                            })?
                            .to_string();
                        let (mut result_content, tool_result_images) = match block.get("content") {
                            None => (String::new(), Vec::new()),
                            Some(raw_content) => {
                                extract_tool_result_content_with_images(raw_content)?
                            }
                        };
                        let is_error = match block.get("is_error") {
                            None => false,
                            Some(Value::Bool(value)) => *value,
                            Some(_) => {
                                return Err(format!(
                                    "user tool_result block {block_index} is_error must be a boolean"
                                ));
                            }
                        };
                        if is_error {
                            result_content = format!("[tool_error]\n{result_content}");
                        }
                        out.push(Message {
                            role: MessageRole::Tool,
                            content: result_content,
                            tool_calls: vec![],
                            tool_call_id: Some(tool_use_id),
                            name: None,
                            reasoning_content: None,
                            images: vec![],
                            structured_content: None,
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
                                structured_content: None,
                            });
                        }
                    }
                    "image" => {
                        reject_unknown_fields(
                            object,
                            &format!("messages user content[{block_index}]"),
                            &["type", "source", "cache_control"],
                        )?;
                        image_urls.push(anthropic_image_source_url(block.get("source"))?);
                    }
                    "document" | "search_result" | "server_tool_use" | "web_search_tool_result" => {
                        return Err(format!(
                            "user content block type {block_type:?} cannot be represented losslessly by the Codex OAuth backend"
                        ));
                    }
                    _ => {
                        return Err(format!(
                            "unsupported user content block type {block_type:?}"
                        ));
                    }
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
                    structured_content: None,
                });
            }
            if out.len() == initial_message_count {
                return Err(
                    "user message content did not contain a representable content block"
                        .to_string(),
                );
            }
        }
        _ => return Err("user message content must be a string or array".to_string()),
    }
    Ok(())
}

fn extract_tool_result_content_with_images(
    content: &Value,
) -> Result<(String, Vec<String>), String> {
    Ok(match content {
        Value::String(s) => (s.clone(), vec![]),
        Value::Array(arr) => {
            let mut text_pieces: Vec<String> = Vec::new();
            let mut images: Vec<String> = Vec::new();
            for (index, p) in arr.iter().enumerate() {
                let object = p
                    .as_object()
                    .ok_or_else(|| format!("tool_result content item {index} must be an object"))?;
                let typ = object.get("type").and_then(Value::as_str).ok_or_else(|| {
                    format!("tool_result content item {index} requires a string type")
                })?;
                if typ == "text" {
                    validate_nullable_unrepresentable_field(
                        object,
                        "citations",
                        &format!("tool_result content item {index}"),
                    )?;
                    reject_unknown_fields(
                        object,
                        &format!("tool_result content item {index}"),
                        &["type", "text", "cache_control", "citations"],
                    )?;
                    let text = object.get("text").and_then(Value::as_str).ok_or_else(|| {
                        format!("tool_result content item {index} requires string text")
                    })?;
                    text_pieces.push(text.to_string());
                } else if typ == "image" {
                    reject_unknown_fields(
                        object,
                        &format!("tool_result content item {index}"),
                        &["type", "source", "cache_control"],
                    )?;
                    images.push(anthropic_image_source_url(p.get("source"))?);
                } else if matches!(typ, "document" | "search_result" | "web_search_tool_result") {
                    return Err(format!(
                        "tool_result content type {typ:?} cannot be represented losslessly by the Codex OAuth backend"
                    ));
                } else {
                    return Err(format!("unsupported tool_result content type {typ:?}"));
                }
            }
            (text_pieces.join(""), images)
        }
        _ => return Err("tool_result content must be a string or array".to_string()),
    })
}

fn anthropic_image_source_url(source: Option<&Value>) -> Result<String, String> {
    let source = source
        .and_then(Value::as_object)
        .ok_or_else(|| "Anthropic image block requires an object source".to_string())?;
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            reject_unknown_fields(
                source,
                "Anthropic image source",
                &["type", "media_type", "data"],
            )?;
            let media_type = source
                .get("media_type")
                .and_then(Value::as_str)
                .filter(|media_type| {
                    matches!(
                        *media_type,
                        "image/jpeg" | "image/png" | "image/gif" | "image/webp"
                    )
                })
                .ok_or_else(|| {
                    "Anthropic base64 image source media_type must be one of: image/jpeg, image/png, image/gif, image/webp"
                        .to_string()
                })?;
            let data = source
                .get("data")
                .and_then(Value::as_str)
                .filter(|data| !data.is_empty())
                .ok_or_else(|| {
                    "Anthropic base64 image source requires non-empty data".to_string()
                })?;
            Ok(format!("data:{media_type};base64,{data}"))
        }
        Some("url") => {
            reject_unknown_fields(source, "Anthropic image source", &["type", "url"])?;
            let url = source
                .get("url")
                .and_then(Value::as_str)
                .filter(|url| !url.is_empty())
                .ok_or_else(|| "Anthropic URL image source requires a non-empty url".to_string())?;
            Ok(url.to_string())
        }
        _ => Err("Anthropic image source type must be one of: base64, url".to_string()),
    }
}

fn convert_assistant_message(content: &Value, out: &mut Vec<Message>) -> Result<(), String> {
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
                structured_content: None,
            });
        }
        Value::Array(arr) => {
            if arr.is_empty() {
                out.push(Message {
                    role: MessageRole::Assistant,
                    content: String::new(),
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    reasoning_content: None,
                    images: vec![],
                    structured_content: None,
                });
                return Ok(());
            }
            let mut text_parts: Vec<String> = Vec::new();
            let mut tool_calls: Vec<ToolCall> = Vec::new();
            let mut tool_call_ids: HashSet<String> = HashSet::new();

            for (block_index, block) in arr.iter().enumerate() {
                let object = block.as_object().ok_or_else(|| {
                    format!("assistant content block {block_index} must be an object")
                })?;
                let block_type = object.get("type").and_then(Value::as_str).ok_or_else(|| {
                    format!("assistant content block {block_index} requires a string type")
                })?;
                match block_type {
                    "text" => {
                        validate_nullable_unrepresentable_field(
                            object,
                            "citations",
                            &format!("messages assistant content[{block_index}]"),
                        )?;
                        reject_unknown_fields(
                            object,
                            &format!("messages assistant content[{block_index}]"),
                            &["type", "text", "cache_control", "citations"],
                        )?;
                        let text = object.get("text").and_then(Value::as_str).ok_or_else(|| {
                            format!("assistant content block {block_index} requires string text")
                        })?;
                        text_parts.push(text.to_string());
                    }
                    "tool_use" => {
                        reject_unknown_fields(
                            object,
                            &format!("messages assistant content[{block_index}]"),
                            &["type", "id", "name", "input", "cache_control", "caller"],
                        )?;
                        if let Some(caller) = object.get("caller") {
                            let caller = caller.as_object().ok_or_else(|| {
                                format!(
                                    "assistant tool_use block {block_index} caller must be an object"
                                )
                            })?;
                            reject_unknown_fields(
                                caller,
                                &format!("assistant tool_use block {block_index} caller"),
                                &["type"],
                            )?;
                            if caller.get("type").and_then(Value::as_str) != Some("direct") {
                                return Err(format!(
                                    "assistant tool_use block {block_index} caller.type must be direct"
                                ));
                            }
                        }
                        let id = block
                            .get("id")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| {
                                format!(
                                    "assistant tool_use block {block_index} requires a string id"
                                )
                            })?
                            .to_string();
                        let name = block
                            .get("name")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| {
                                format!(
                                    "assistant tool_use block {block_index} requires a string name"
                                )
                            })?
                            .to_string();
                        let arguments =
                            block
                                .get("input")
                                .and_then(|v| v.as_object())
                                .ok_or_else(|| {
                                    format!(
                                    "assistant tool_use block {block_index} requires object input"
                                )
                                })?;
                        let arguments = serde_json::to_string(arguments)
                            .expect("serde_json::Map must serialize");
                        if !tool_call_ids.insert(id.clone()) {
                            return Err(format!(
                                "assistant tool_use blocks contain duplicate id {id:?}"
                            ));
                        }
                        tool_calls.push(ToolCall {
                            id,
                            name,
                            arguments,
                        });
                    }
                    "thinking" | "redacted_thinking" => {
                        return Err(format!(
                            "assistant {block_type} history cannot be represented losslessly by the Codex OAuth backend"
                        ));
                    }
                    "server_tool_use"
                    | "web_search_tool_result"
                    | "document"
                    | "search_result"
                    | "tool_result" => {
                        return Err(format!(
                            "assistant content block type {block_type:?} cannot be represented losslessly by the Codex OAuth backend"
                        ));
                    }
                    _ => {
                        return Err(format!(
                            "unsupported assistant content block type {block_type:?}"
                        ));
                    }
                }
            }

            if text_parts.is_empty() && tool_calls.is_empty() {
                return Err(
                    "assistant message content did not contain a representable content block"
                        .to_string(),
                );
            }

            out.push(Message {
                role: MessageRole::Assistant,
                content: text_parts.join(""),
                tool_calls,
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            });
        }
        _ => return Err("assistant message content must be a string or array".to_string()),
    }
    Ok(())
}

fn convert_tools(tools: &[Value]) -> Result<Vec<ToolSchema>, String> {
    let mut result = Vec::new();
    for (index, tool) in tools.iter().enumerate() {
        let object = tool
            .as_object()
            .ok_or_else(|| format!("tools[{index}] must be an object"))?;
        let name = match tool.get("name").and_then(|v| v.as_str()) {
            Some(n) if !n.is_empty() => n.to_string(),
            _ => return Err(format!("tools[{index}].name must be a non-empty string")),
        };
        if is_anthropic_web_search_tool(tool) {
            return Err(
                "Anthropic hosted web_search cannot be represented losslessly by this facade"
                    .to_string(),
            );
        }
        reject_unknown_fields(
            object,
            &format!("tools[{index}]"),
            &[
                "type",
                "name",
                "description",
                "input_schema",
                "cache_control",
                "strict",
                "defer_loading",
                "eager_input_streaming",
                "allowed_callers",
                "output_schema",
            ],
        )?;
        match object.get("type") {
            None | Some(Value::Null) => {}
            Some(Value::String(value)) if value == "custom" => {}
            Some(_) => {
                return Err(format!("tools[{index}].type must be custom or null"));
            }
        }
        let description = match object.get("description") {
            None => None,
            Some(Value::String(value)) => Some(value.clone()),
            Some(_) => {
                return Err(format!("tools[{index}].description must be a string"));
            }
        };
        let strict = match object.get("strict") {
            None => false,
            Some(Value::Bool(value)) => *value,
            Some(_) => return Err(format!("tools[{index}].strict must be a boolean")),
        };
        let parameters = object
            .get("input_schema")
            .filter(|value| value.is_object())
            .cloned()
            .ok_or_else(|| format!("tools[{index}].input_schema must be an object"))?;
        result.push(ToolSchema {
            name,
            description,
            parameters,
            strict,
        });
    }
    Ok(result)
}

fn validate_anthropic_tool_controls(tools: &[Value]) -> Result<(), String> {
    for tool in tools {
        let Some(tool) = tool.as_object() else {
            return Err("tools entries must be objects".to_string());
        };
        if is_anthropic_web_search_tool(&Value::Object(tool.clone())) {
            return Err(
                "Anthropic hosted web_search cannot be represented losslessly by this facade"
                    .to_string(),
            );
        }
        for field in ["allowed_callers"] {
            if tool.contains_key(field) {
                return Err(format!(
                    "tool.{field} is not supported by the Codex OAuth backend"
                ));
            }
        }
        if tool.contains_key("output_schema") {
            return Err(
                "tool.output_schema is not supported by the Codex OAuth backend".to_string(),
            );
        }
        match tool.get("strict") {
            None | Some(Value::Bool(false)) => {}
            Some(Value::Bool(true)) => {}
            Some(_) => return Err("tool.strict must be a boolean when provided".to_string()),
        }
        if tool.contains_key("defer_loading") {
            return Err(
                "tool.defer_loading is not supported by the Codex OAuth backend".to_string(),
            );
        }
        for field in ["eager_input_streaming"] {
            match tool.get(field) {
                None | Some(Value::Null) => {}
                Some(Value::Bool(_)) => {
                    return Err(format!(
                        "tool.{field} is not supported by the Codex OAuth backend"
                    ));
                }
                Some(_) => {
                    return Err(format!("tool.{field} must be a boolean when provided"));
                }
            }
        }
    }
    Ok(())
}

fn is_anthropic_web_search_tool(tool: &Value) -> bool {
    tool.get("name").and_then(|v| v.as_str()) == Some("web_search")
        && tool
            .get("type")
            .and_then(|v| v.as_str())
            .map(|s| {
                matches!(
                    s,
                    "web_search" | "web_search_20250305" | "web_search_20260209"
                )
            })
            .unwrap_or(false)
}

fn convert_tool_choice(tc: &Value) -> Result<(Value, Option<bool>), String> {
    let object = tc
        .as_object()
        .ok_or_else(|| "tool_choice must be an object".to_string())?;
    let tc_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| "tool_choice.type must be a string".to_string())?;
    let allowed_fields: &[&str] = if tc_type == "tool" {
        &["type", "name", "disable_parallel_tool_use"]
    } else if tc_type == "none" {
        &["type"]
    } else {
        &["type", "disable_parallel_tool_use"]
    };
    reject_unknown_fields(object, "tool_choice", allowed_fields)?;
    let parallel_tool_calls = match object.get("disable_parallel_tool_use") {
        None => None,
        Some(Value::Bool(disable)) => Some(!disable),
        Some(_) => {
            return Err("tool_choice.disable_parallel_tool_use must be a boolean".to_string())
        }
    };
    let choice = match tc_type {
        "auto" => json!("auto"),
        "any" => json!("required"),
        "tool" => {
            let name = object
                .get("name")
                .and_then(Value::as_str)
                .filter(|name| !name.is_empty())
                .ok_or_else(|| "tool_choice.name must be a non-empty string".to_string())?;
            if name == "web_search" {
                return Err(
                    "Anthropic hosted web_search cannot be represented losslessly by this facade"
                        .to_string(),
                );
            } else {
                json!({"type": "function", "name": name})
            }
        }
        "none" => json!("none"),
        _ => return Err(format!("unsupported tool_choice.type {tc_type:?}")),
    };
    Ok((choice, parallel_tool_calls))
}

fn convert_thinking(thinking: &Value, max_tokens: Option<i64>) -> Result<Option<String>, String> {
    let object = thinking
        .as_object()
        .ok_or_else(|| "thinking must be an object".to_string())?;
    let thinking_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| "thinking.type must be one of: enabled, adaptive, disabled".to_string())?;
    match thinking_type {
        "enabled" => {
            reject_unknown_fields(object, "thinking", &["type", "budget_tokens", "display"])?;
            match object.get("display") {
                None | Some(Value::Null) => {}
                Some(Value::String(value)) if value == "omitted" => {}
                Some(_) => return Err("thinking.display must be omitted or null".to_string()),
            }
            let budget_tokens = object
                .get("budget_tokens")
                .and_then(Value::as_number)
                .and_then(crate::strict_json::as_js_safe_integer)
                .ok_or_else(|| {
                    "thinking.budget_tokens must be an integer greater than or equal to 1024"
                        .to_string()
                })?;
            if budget_tokens < 1024 {
                return Err(
                    "thinking.budget_tokens must be an integer greater than or equal to 1024"
                        .to_string(),
                );
            }
            if max_tokens.is_some_and(|max_tokens| budget_tokens >= max_tokens) {
                return Err("thinking.budget_tokens must be less than max_tokens".to_string());
            }
            Ok(Some("high".to_string()))
        }
        "adaptive" => {
            reject_unknown_fields(object, "thinking", &["type", "display"])?;
            match object.get("display") {
                None | Some(Value::Null) => {}
                Some(Value::String(value)) if value == "omitted" => {}
                Some(_) => {
                    return Err("thinking.display must be omitted, null, or \"omitted\"".to_string())
                }
            }
            Ok(Some("medium".to_string()))
        }
        "disabled" => {
            reject_unknown_fields(object, "thinking", &["type"])?;
            Ok(Some("none".to_string()))
        }
        _ => Err("thinking.type must be one of: enabled, adaptive, disabled".to_string()),
    }
}

fn convert_reasoning_effort(body: &Value) -> Result<Option<String>, String> {
    let max_tokens = body
        .get("max_tokens")
        .and_then(Value::as_number)
        .and_then(crate::strict_json::as_js_safe_integer);
    let thinking_effort = body
        .get("thinking")
        .filter(|value| !value.is_null())
        .map(|thinking| convert_thinking(thinking, max_tokens))
        .transpose()?
        .flatten();
    let output_config = match body.get("output_config") {
        None | Some(Value::Null) => None,
        Some(Value::Object(output_config)) => Some(output_config),
        Some(_) => return Err("output_config must be an object".to_string()),
    };
    if let Some(unknown_field) = output_config.and_then(|output_config| {
        output_config
            .keys()
            .find(|field| !matches!(field.as_str(), "effort" | "format" | "task_budget"))
    }) {
        return Err(format!(
            "output_config.{unknown_field} is not supported by the Codex OAuth backend"
        ));
    }
    if output_config
        .and_then(|output_config| output_config.get("task_budget"))
        .is_some_and(|task_budget| !task_budget.is_null())
    {
        return Err(
            "output_config.task_budget is not supported by the Codex OAuth backend".to_string(),
        );
    }
    let output_effort = output_config.and_then(|output_config| output_config.get("effort"));

    match output_effort {
        None | Some(Value::Null) => Ok(thinking_effort),
        Some(Value::String(effort)) if effort.trim().is_empty() => {
            Err("output_config.effort must be a non-empty string when provided".to_string())
        }
        Some(Value::String(effort))
            if matches!(effort.as_str(), "low" | "medium" | "high" | "xhigh" | "max") =>
        {
            if thinking_effort.as_deref() == Some("none") {
                Ok(Some("none".to_string()))
            } else {
                Ok(Some(effort.clone()))
            }
        }
        Some(Value::String(_)) => {
            Err("output_config.effort must be one of: low, medium, high, xhigh, max".to_string())
        }
        Some(_) => Err("output_config.effort must be a non-empty string when provided".to_string()),
    }
}

fn anthropic_output_format_from_body(body: &Value) -> Result<Option<&Value>, String> {
    let top_level = body.get("output_format").filter(|value| !value.is_null());
    let nested = body
        .get("output_config")
        .and_then(Value::as_object)
        .and_then(|output_config| output_config.get("format"))
        .filter(|value| !value.is_null());

    if top_level.is_some() && nested.is_some() && top_level != nested {
        return Err("output_format conflicts with output_config.format".to_string());
    }

    let (field, selected) = if let Some(output_format) = top_level {
        ("output_format", output_format)
    } else if let Some(output_format) = nested {
        ("output_config.format", output_format)
    } else {
        return Ok(None);
    };
    validate_anthropic_output_format(field, selected)?;
    Ok(Some(selected))
}

fn validate_anthropic_output_format(field: &str, output_format: &Value) -> Result<(), String> {
    let object = output_format
        .as_object()
        .ok_or_else(|| format!("{field} must be an object when provided"))?;
    let format_type = object
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| format!("{field}.type must be a string"))?;

    let allowed_fields: &[&str] = match format_type {
        "json_object" => &["type"],
        "json_schema" => &["type", "schema", "name", "description", "strict"],
        _ => {
            return Err(format!(
                "{field}.type must be one of: json_object, json_schema"
            ));
        }
    };
    if let Some(unknown_field) = object
        .keys()
        .find(|key| !allowed_fields.contains(&key.as_str()))
    {
        return Err(format!("{field}.{unknown_field} is not supported"));
    }
    if format_type == "json_schema" && !object.get("schema").is_some_and(Value::is_object) {
        return Err(format!("{field}.schema must be an object"));
    }
    if format_type == "json_schema" {
        let name = object
            .get("name")
            .map(|value| {
                value
                    .as_str()
                    .filter(|name| !name.is_empty())
                    .ok_or_else(|| format!("{field}.name must be a non-empty string"))
            })
            .transpose()?
            .unwrap_or("codex_output_schema");
        if name.len() > 64
            || !name.chars().all(|character| {
                character.is_ascii_alphanumeric() || matches!(character, '_' | '-')
            })
        {
            return Err(format!(
                "{field}.name must contain only ASCII letters, digits, underscores, or hyphens and be at most 64 bytes"
            ));
        }
    } else if object.get("name").is_some() {
        return Err(format!("{field}.name is not supported for json_object"));
    }
    if object
        .get("description")
        .is_some_and(|value| !value.is_string())
    {
        return Err(format!("{field}.description must be a string"));
    }
    if object
        .get("strict")
        .is_some_and(|value| !value.is_null() && !value.is_boolean())
    {
        return Err(format!("{field}.strict must be a boolean"));
    }
    Ok(())
}

pub fn anthropic_output_format_to_openai_text(
    output_format: &Value,
) -> Result<Option<Value>, String> {
    validate_anthropic_output_format("output_format", output_format)?;
    let typ = output_format
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| "output format requires a string type".to_string())?;
    match typ {
        "json_schema" => {
            let schema = output_format
                .get("schema")
                .and_then(Value::as_object)
                .ok_or_else(|| "json_schema output format requires an object schema".to_string())?;
            let name = match output_format.get("name") {
                None => "codex_output_schema",
                Some(Value::String(name)) => name.as_str(),
                Some(_) => return Err("output_format.name must be a non-empty string".to_string()),
            };
            if name.len() > 64
                || !name.chars().all(|character| {
                    character.is_ascii_alphanumeric() || matches!(character, '_' | '-')
                })
            {
                return Err(
                    "json_schema output format name must contain only ASCII letters, digits, underscores, or hyphens and be at most 64 bytes"
                        .to_string(),
                );
            }
            let mut format = serde_json::Map::new();
            format.insert("type".to_string(), json!("json_schema"));
            format.insert("name".to_string(), json!(name));
            format.insert("schema".to_string(), Value::Object(schema.clone()));
            if let Some(description) = output_format.get("description").and_then(|v| v.as_str()) {
                format.insert("description".to_string(), json!(description));
            }
            if let Some(strict) = output_format.get("strict").and_then(|v| v.as_bool()) {
                format.insert("strict".to_string(), json!(strict));
            }
            Ok(Some(json!({"format": Value::Object(format)})))
        }
        "json_object" => Ok(Some(json!({"format": {"type": "json_object"}}))),
        _ => Err(format!("unsupported output format type {typ:?}")),
    }
}

fn parse_anthropic_tool_arguments(value: &str) -> Result<Value, String> {
    let parsed = crate::strict_json::parse_str(value)
        .map_err(|error| format!("tool call arguments must contain valid JSON: {error}"))?;
    if !parsed.is_object() {
        return Err("tool call arguments JSON must be an object".to_string());
    }
    Ok(parsed)
}

pub fn internal_response_to_anthropic(
    response: &AssistantResponse,
    model: &str,
    request_id: &str,
) -> Result<Value, String> {
    let mut content: Vec<Value> = Vec::new();

    reject_unrepresentable_response_events(response.raw.as_ref())?;

    if !response.content.is_empty() {
        content.push(json!({"type": "text", "text": response.content, "citations": null}));
    }

    for tc in &response.tool_calls {
        let arguments = parse_anthropic_tool_arguments(&tc.arguments)?;
        content.push(json!({
            "type": "tool_use",
            "id": tc.id,
            "name": tc.name,
            "input": arguments,
            "caller": {"type": "direct"},
        }));
    }

    let stop_reason = map_stop_reason(
        response.finish_reason.as_deref(),
        !response.tool_calls.is_empty(),
    )?;

    let mut result = json!({
        "id": request_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "container": null,
        "content": content,
        "context_management": null,
        "stop_reason": stop_reason,
        "stop_sequence": null,
    });
    let object = result
        .as_object_mut()
        .expect("Anthropic response literal must be an object");
    if let Some(reasoning) = response
        .reasoning_content
        .as_ref()
        .filter(|reasoning| !reasoning.is_empty())
    {
        object.insert("codex_reasoning".to_string(), json!(reasoning));
    }
    let usage = response
        .usage
        .as_ref()
        .ok_or_else(|| "provider response requires authoritative usage".to_string())?;
    if usage.prompt_tokens < 0
        || usage.completion_tokens < 0
        || usage.total_tokens < 0
        || usage.prompt_tokens.checked_add(usage.completion_tokens) != Some(usage.total_tokens)
        || usage.cached_tokens.is_some_and(|value| value < 0)
        || usage.cache_write_tokens.is_some_and(|value| value < 0)
    {
        return Err("Anthropic response usage is inconsistent or negative".to_string());
    }
    let usage_dict = json!({
        "cache_creation": null,
        "cache_creation_input_tokens": usage.cache_write_tokens,
        "cache_read_input_tokens": usage.cached_tokens,
        "inference_geo": null,
        "input_tokens": usage.prompt_tokens,
        "iterations": null,
        "output_tokens": usage.completion_tokens,
        "server_tool_use": null,
        "service_tier": null,
        "speed": null,
    });
    let usage_dict = merge_actual_usage_extensions(usage_dict, response.raw.as_ref())?;
    object.insert("usage".to_string(), usage_dict);
    Ok(result)
}

fn reject_unrepresentable_response_events(raw: Option<&Value>) -> Result<(), String> {
    let Some(raw) = raw else {
        return Ok(());
    };
    let events = raw
        .get("events")
        .and_then(Value::as_array)
        .ok_or_else(|| "normalized response raw.events must be an array".to_string())?;
    for (index, event) in events.iter().enumerate() {
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("normalized raw event {index} requires a string type"))?;
        if event_type.is_empty() {
            return Err(format!(
                "normalized raw event {index} requires a non-empty string type"
            ));
        }
        if event_type != "web_search_call" {
            continue;
        }
        return Err(
            "provider web_search_call output cannot be represented losslessly by the Anthropic facade"
                .to_string(),
        );
    }
    Ok(())
}

fn merge_actual_usage_extensions(mut usage: Value, raw: Option<&Value>) -> Result<Value, String> {
    let Some(raw) = raw else {
        return Ok(usage);
    };
    let events = raw
        .get("events")
        .and_then(Value::as_array)
        .ok_or_else(|| "normalized response raw.events must be an array".to_string())?;
    for (index, event) in events.iter().enumerate() {
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("normalized raw event {index} requires a string type"))?;
        if event_type != "finish" {
            continue;
        }
        let Some(event_usage_value) = event.get("usage").filter(|usage| !usage.is_null()) else {
            return Ok(usage);
        };
        let event_usage = event_usage_value
            .as_object()
            .ok_or_else(|| "normalized finish event usage must be an object".to_string())?;
        for key in ["cache_creation", "server_tool_use", "service_tier"] {
            let Some(value) = event_usage.get(key) else {
                continue;
            };
            match key {
                "cache_creation" if value.is_null() => continue,
                "cache_creation" => validate_usage_counter_object(
                    value,
                    key,
                    &["ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"],
                )?,
                "server_tool_use" if value.is_null() => continue,
                "server_tool_use" => validate_usage_counter_object(
                    value,
                    key,
                    &["web_search_requests", "web_fetch_requests"],
                )?,
                "service_tier" if value.is_null() => continue,
                "service_tier"
                    if !matches!(value.as_str(), Some("standard" | "priority" | "batch")) =>
                {
                    return Err(
                        "usage.service_tier must be standard, priority, batch, or null".to_string(),
                    );
                }
                "service_tier" => {}
                _ => unreachable!("fixed usage extension field set"),
            }
            usage
                .as_object_mut()
                .expect("Anthropic usage literal must be an object")
                .insert(key.to_string(), value.clone());
        }
        return Ok(usage);
    }
    Ok(usage)
}

fn map_stop_reason(
    finish_reason: Option<&str>,
    has_tool_calls: bool,
) -> Result<&'static str, String> {
    match finish_reason {
        None => Err("provider response requires a non-null finish_reason".to_string()),
        Some("stop") if has_tool_calls => {
            Err("provider finish_reason stop conflicts with emitted tool calls".to_string())
        }
        Some("stop") => Ok("end_turn"),
        Some("tool_calls") if !has_tool_calls => {
            Err("provider finish_reason tool_calls requires at least one tool call".to_string())
        }
        Some("tool_calls") => Ok("tool_use"),
        _ => Err(format!("unsupported finish_reason {finish_reason:?}")),
    }
}

pub struct AnthropicStreamAdapter {
    model: String,
    request_id: String,
    started: bool,
    block_index: u32,
    current_block: Option<&'static str>,
    has_tool_calls: bool,
    finished: bool,
}

impl AnthropicStreamAdapter {
    pub fn new(model: &str, request_id: &str) -> Self {
        Self {
            model: model.to_string(),
            request_id: request_id.to_string(),
            started: false,
            block_index: 0,
            current_block: None,
            has_tool_calls: false,
            finished: false,
        }
    }

    pub fn start(&mut self) -> Vec<String> {
        if self.started {
            return vec![];
        }
        self.started = true;
        vec![message_start_sse(&self.model, &self.request_id)]
    }

    pub fn push(&mut self, event: &Value) -> Result<Vec<String>, String> {
        if self.finished {
            return Err("normalized response emitted an event after finish".to_string());
        }
        let mut out = Vec::new();
        let mut block_index = self.block_index;
        let mut current_block = self.current_block;
        let mut has_tool_calls = self.has_tool_calls;
        let typ = event
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| "normalized response event requires a string type".to_string())?;

        match typ {
            "reasoning_delta" | "reasoning_raw_delta" => {
                let text = event
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| format!("{typ} event requires string text"))?
                    .to_string();
                out.push(sse(
                    "codex_reasoning_delta",
                    &json!({"type": "codex_reasoning_delta", "delta": text}),
                ));
            }

            "content" => {
                let text = event
                    .get("text")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "content event requires string text".to_string())?
                    .to_string();
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
                            "content_block": {"type": "text", "text": "", "citations": null},
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
                has_tool_calls = true;
                if current_block.is_some() {
                    out.push(sse(
                        "content_block_stop",
                        &json!({"type": "content_block_stop", "index": block_index}),
                    ));
                    block_index += 1;
                }
                let tool_id = event
                    .get("id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "tool_call event requires a string id".to_string())?
                    .to_string();
                let tool_name = event
                    .get("name")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "tool_call event requires a string name".to_string())?
                    .to_string();
                let tool_args = event
                    .get("arguments")
                    .and_then(Value::as_str)
                    .ok_or_else(|| "tool_call event requires string arguments".to_string())?;
                parse_anthropic_tool_arguments(tool_args)?;
                out.push(sse(
                    "content_block_start",
                    &json!({
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": tool_name,
                            "input": {},
                            "caller": {"type": "direct"},
                        },
                    }),
                ));
                out.push(sse(
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": tool_args,
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

            "web_search_call" => {
                return Err(
                    "provider web_search_call output cannot be represented losslessly by the Anthropic facade"
                        .to_string(),
                );
            }

            "finish" => {
                if current_block.is_some() {
                    out.push(sse(
                        "content_block_stop",
                        &json!({"type": "content_block_stop", "index": block_index}),
                    ));
                    current_block = None;
                }

                let finish_reason = match event.get("finish_reason") {
                    None | Some(Value::Null) => None,
                    Some(Value::String(reason)) if !reason.is_empty() => Some(reason.as_str()),
                    Some(_) => {
                        return Err(
                            "finish event finish_reason must be non-empty or null".to_string()
                        );
                    }
                };
                let stop_reason = map_stop_reason(finish_reason, has_tool_calls)?;
                let usage_value = event
                    .get("usage")
                    .filter(|usage| !usage.is_null())
                    .ok_or_else(|| "finish event requires authoritative usage".to_string())?;
                let usage = anthropic_usage_from_provider(usage_value)?;
                let message_delta = json!({
                    "type": "message_delta",
                    "context_management": null,
                    "delta": {"container": null, "stop_reason": stop_reason, "stop_sequence": null},
                    "usage": usage,
                });

                out.push(sse("message_delta", &message_delta));
                out.push(sse("message_stop", &json!({"type": "message_stop"})));
                self.finished = true;
            }

            "reasoning_section_break" => {}
            _ => {
                return Err(format!(
                    "unsupported normalized response event type {typ:?}"
                ))
            }
        }

        self.block_index = block_index;
        self.current_block = current_block;
        self.has_tool_calls = has_tool_calls;
        Ok(out)
    }
}

#[cfg(test)]
pub fn anthropic_stream_adapter(events: &[Value], model: &str, request_id: &str) -> Vec<String> {
    let mut adapter = AnthropicStreamAdapter::new(model, request_id);
    let mut out = adapter.start();
    for event in events {
        out.extend(
            adapter
                .push(event)
                .expect("test events must satisfy the normalized event contract"),
        );
    }
    out
}

fn message_start_sse(model: &str, request_id: &str) -> String {
    sse(
        "message_start",
        &json!({
            "type": "message_start",
            "message": {
                "id": request_id,
                "type": "message",
                "role": "assistant",
                "model": model,
                "container": null,
                "content": [],
                "context_management": null,
                "stop_reason": null,
                "stop_sequence": null,
            },
        }),
    )
}

fn optional_usage_count(usage: &Value, key: &str) -> Result<Option<i64>, String> {
    match usage.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_i64()
            .filter(|count| *count >= 0)
            .map(Some)
            .ok_or_else(|| format!("usage.{key} must be a non-negative integer")),
    }
}

fn required_usage_count(usage: &Value, key: &str) -> Result<i64, String> {
    optional_usage_count(usage, key)?.ok_or_else(|| format!("usage requires {key}"))
}

fn usage_details_count(usage: &Value, key: &str) -> Result<Option<i64>, String> {
    let details = match usage.get("input_tokens_details") {
        Some(details) => details,
        None => return Ok(None),
    };
    if details.is_null() {
        return Ok(None);
    }
    let object = details
        .as_object()
        .ok_or_else(|| "usage input token details must be an object or null".to_string())?;
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_i64()
            .filter(|count| *count >= 0)
            .map(Some)
            .ok_or_else(|| format!("usage input token details.{key} must be non-negative")),
    }
}

fn validate_usage_counter_object(
    value: &Value,
    field: &str,
    required: &[&str],
) -> Result<(), String> {
    let object = value
        .as_object()
        .ok_or_else(|| format!("usage.{field} must be an object"))?;
    if let Some(key) = object.keys().find(|key| !required.contains(&key.as_str())) {
        return Err(format!("usage.{field} contains unsupported field {key:?}"));
    }
    if let Some(key) = required.iter().find(|key| !object.contains_key(**key)) {
        return Err(format!("usage.{field} is missing required field {key:?}"));
    }
    for (key, count) in object {
        if count.as_i64().is_none_or(|count| count < 0) {
            return Err(format!(
                "usage.{field}.{key} must be a non-negative integer"
            ));
        }
    }
    Ok(())
}

fn anthropic_usage_from_provider(usage: &Value) -> Result<Value, String> {
    let usage = usage
        .as_object()
        .ok_or_else(|| "usage must be an object".to_string())?;
    if let Some(field) = [
        "prompt_tokens",
        "completion_tokens",
        "prompt_tokens_details",
        "cached_input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ]
    .into_iter()
    .find(|field| usage.contains_key(*field))
    {
        return Err(format!(
            "usage contains unsupported public alias field {field:?}"
        ));
    }
    let usage = &Value::Object(usage.clone());
    let input_tokens = required_usage_count(usage, "input_tokens")?;
    let output_tokens = required_usage_count(usage, "output_tokens")?;
    let total_tokens = optional_usage_count(usage, "total_tokens")?
        .ok_or_else(|| "usage requires total_tokens".to_string())?;
    if input_tokens.checked_add(output_tokens) != Some(total_tokens) {
        return Err("usage.total_tokens must equal input_tokens plus output_tokens".to_string());
    }
    if usage
        .get("input_tokens_details")
        .and_then(Value::as_object)
        .is_some_and(|details| !details.contains_key("cached_tokens"))
    {
        return Err("usage input token details requires cached_tokens".to_string());
    }
    let cache_read = usage_details_count(usage, "cached_tokens")?;
    let cache_creation = usage_details_count(usage, "cache_write_tokens")?;

    let mut out = serde_json::Map::new();
    out.insert(
        "cache_creation_input_tokens".to_string(),
        cache_creation.map_or(Value::Null, |value| json!(value)),
    );
    out.insert(
        "cache_read_input_tokens".to_string(),
        cache_read.map_or(Value::Null, |value| json!(value)),
    );
    out.insert("input_tokens".to_string(), json!(input_tokens));
    out.insert("iterations".to_string(), Value::Null);
    out.insert("output_tokens".to_string(), json!(output_tokens));
    out.insert("server_tool_use".to_string(), Value::Null);
    for key in ["cache_creation", "server_tool_use", "service_tier"] {
        if let Some(value) = usage.get(key) {
            match key {
                "cache_creation" if value.is_null() => continue,
                "cache_creation" => validate_usage_counter_object(
                    value,
                    key,
                    &["ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"],
                )?,
                "server_tool_use" if value.is_null() => continue,
                "server_tool_use" => {
                    validate_usage_counter_object(
                        value,
                        key,
                        &["web_search_requests", "web_fetch_requests"],
                    )?;
                    out.insert(key.to_string(), value.clone());
                }
                "service_tier" if value.is_null() => continue,
                "service_tier"
                    if !matches!(value.as_str(), Some("standard" | "priority" | "batch")) =>
                {
                    return Err(
                        "usage.service_tier must be standard, priority, batch, or null".to_string(),
                    );
                }
                "service_tier" => {}
                _ => unreachable!("fixed usage field set"),
            }
        }
    }
    Ok(Value::Object(out))
}
fn sse(event_type: &str, data: &Value) -> String {
    format!(
        "event: {}\ndata: {}\n\n",
        event_type,
        serde_json::to_string(data).expect("serde_json::Value must serialize")
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
mod strict_adapter_tests {
    use super::*;

    fn has_key(value: &Value, key: &str) -> bool {
        match value {
            Value::Object(object) => {
                object.contains_key(key) || object.values().any(|value| has_key(value, key))
            }
            Value::Array(values) => values.iter().any(|value| has_key(value, key)),
            _ => false,
        }
    }

    #[test]
    fn assistant_thinking_history_is_rejected_instead_of_fabricated() {
        for block in [
            json!({"type": "thinking", "thinking": "opaque"}),
            json!({"type": "redacted_thinking", "data": "opaque"}),
        ] {
            let request = json!({
                "messages": [{"role": "assistant", "content": [block]}]
            });
            assert!(anthropic_request_to_internal(&request).is_err());
        }
    }

    #[test]
    fn non_stream_reasoning_uses_only_proxy_extension() {
        let response = AssistantResponse {
            content: "answer".to_string(),
            tool_calls: vec![],
            finish_reason: Some("stop".to_string()),
            usage: Some(crate::messages::Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
                cached_tokens: Some(0),
                cache_write_tokens: None,
            }),
            reasoning_content: Some("real reasoning".to_string()),
            raw: None,
            response_id: Some("resp".to_string()),
        };
        let value = internal_response_to_anthropic(&response, "claude-facade", "msg-1").unwrap();
        assert_eq!(value["codex_reasoning"], "real reasoning");
        assert!(value["content"]
            .as_array()
            .unwrap()
            .iter()
            .all(|block| block.get("type").and_then(Value::as_str) != Some("thinking")));
        assert!(!has_key(&value, "signature"));
    }

    #[test]
    fn stream_reasoning_is_a_proxy_extension_without_thinking_or_signature() {
        let chunks = anthropic_stream_adapter(
            &[
                json!({"type": "reasoning_delta", "text": "real"}),
                json!({"type": "content", "text": "answer"}),
                json!({
                    "type": "finish",
                    "finish_reason": "stop",
                    "usage": {
                        "input_tokens": 1,
                        "output_tokens": 1,
                        "total_tokens": 2,
                        "input_tokens_details": {"cached_tokens": 0}
                    },
                    "response_id": "resp"
                }),
            ],
            "claude-facade",
            "msg-1",
        );
        let parsed: Vec<(&str, Value)> = chunks
            .iter()
            .map(|chunk| {
                let mut lines = chunk.trim_end().lines();
                let event = lines.next().unwrap().strip_prefix("event: ").unwrap();
                let data = lines.next().unwrap().strip_prefix("data: ").unwrap();
                (event, serde_json::from_str(data).unwrap())
            })
            .collect();
        assert!(parsed.iter().any(|(event, value)| {
            *event == "codex_reasoning_delta"
                && value.get("type").and_then(Value::as_str) == Some("codex_reasoning_delta")
                && value.get("delta").and_then(Value::as_str) == Some("real")
        }));
        assert!(parsed.iter().all(|(_, value)| {
            value.get("type").and_then(Value::as_str) != Some("thinking_delta")
                && !has_key(value, "signature")
        }));
        let start = parsed
            .iter()
            .find(|(event, _)| *event == "message_start")
            .expect("stream must begin with message_start");
        assert!(start.1["message"].get("usage").is_none());
    }

    #[test]
    fn reasoning_only_response_is_not_padded_with_fake_text() {
        let mut adapter = AnthropicStreamAdapter::new("claude-facade", "msg-1");
        let reasoning = adapter
            .push(&json!({"type": "reasoning_delta", "text": "real"}))
            .unwrap();
        let finished = adapter
            .push(&json!({
                "type": "finish",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0}
                },
                "response_id": "resp"
            }))
            .unwrap();
        assert!(reasoning
            .iter()
            .all(|event| !event.contains("content_block_start")));
        assert!(finished
            .iter()
            .all(|event| !event.contains("content_block_start")));
        assert!(finished.iter().any(|event| event.contains("message_stop")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    type ConvertedRequest = (
        Vec<Message>,
        Option<Vec<ToolSchema>>,
        Option<Value>,
        Option<Vec<String>>,
        Option<String>,
        Option<Value>,
    );

    fn anthropic_request_to_internal(body: &Value) -> Result<ConvertedRequest, String> {
        let mut body = body.clone();
        if body
            .get("messages")
            .and_then(Value::as_array)
            .is_some_and(Vec::is_empty)
        {
            body["messages"] = json!([{"role": "user", "content": "baseline fixture message"}]);
        }
        let (messages, tools, tool_choice, stop, reasoning_effort, text, _parallel_tool_calls) =
            super::anthropic_request_to_internal(&body)?;
        Ok((messages, tools, tool_choice, stop, reasoning_effort, text))
    }

    // -----------------------------------------------------------------------
    // Request conversion tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_system_string() {
        let body = json!({
            "messages": [],
            "system": "You are a helpful assistant.",
        });
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(msgs.len(), 2);
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
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(msgs[0].content, "Part one.\n\nPart two.");
    }

    #[test]
    fn test_empty_system_is_omitted_but_explicit_null_is_rejected() {
        let empty = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "system": [],
        });
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&empty).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, MessageRole::User);

        let explicit_null = json!({
            "messages": [{"role": "user", "content": "hello"}],
            "system": null,
        });
        assert!(anthropic_request_to_internal(&explicit_null).is_err());
    }

    #[test]
    fn test_system_blocks_reject_non_text() {
        let body = json!({
            "messages": [],
            "system": [
                {"type": "image", "source": {}},
                {"type": "text", "text": "Only text."},
            ],
        });
        assert!(anthropic_request_to_internal(&body).is_err());
    }

    #[test]
    fn cache_control_hints_are_validated_and_stripped_from_conversion() {
        let body = json!({
            "cache_control": {"type": "ephemeral"},
            "system": [{
                "type": "text",
                "text": "Cached system",
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }],
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral"}
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_1",
                        "content": [{
                            "type": "text",
                            "text": "kept",
                            "cache_control": {"type": "ephemeral"}
                        }],
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            }],
            "tools": [{
                "name": "lookup",
                "description": "Look something up",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"}
            }]
        });

        let (messages, tools, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].role, MessageRole::System);
        assert_eq!(messages[1].role, MessageRole::User);
        assert_eq!(messages[2].role, MessageRole::Tool);
        assert_eq!(messages[2].content, "kept");
        assert_eq!(tools.unwrap()[0].name, "lookup");
    }

    #[test]
    fn message_level_cache_control_is_not_part_of_the_sdk_message_shape() {
        let body = json!({
            "messages": [{
                "role": "user",
                "content": "hello",
                "cache_control": {"type": "ephemeral"}
            }]
        });
        assert!(anthropic_request_to_internal(&body).is_err());
    }

    #[test]
    fn malformed_or_unknown_cache_control_hints_fail_loudly() {
        let invalid = [
            json!({"cache_control": {"type": "persistent"}}),
            json!({"cache_control": {"type": "ephemeral", "ttl": "2h"}}),
            json!({"cache_control": {"type": "ephemeral", "scope": "global"}}),
            json!({
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Hello",
                        "cache_control": "ephemeral"
                    }]
                }]
            }),
            json!({
                "tools": [{
                    "name": "lookup",
                    "input_schema": {"type": "object"},
                    "cache_control": {"ttl": "5m"}
                }]
            }),
        ];

        for body in invalid {
            assert!(
                anthropic_request_to_internal(&body).is_err(),
                "accepted invalid cache_control: {body}"
            );
        }
        assert!(validate_anthropic_cache_controls(&json!({"cache_control": null}), false).is_ok());
    }

    #[test]
    fn test_user_text_string() {
        let body = json!({
            "messages": [{"role": "user", "content": "Hello"}],
        });
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
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
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
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
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
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
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
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
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
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
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, MessageRole::Tool);
        assert_eq!(msgs[0].content, "file contents");
        assert_eq!(msgs[1].role, MessageRole::User);
        assert_eq!(msgs[1].images[0], "data:image/jpeg;base64,/9j/4AAQ");
    }

    #[test]
    fn test_url_images_map_in_direct_and_tool_result_content() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "image", "source": {"type": "url", "url": "https://example.com/direct.png"}},
                {"type": "tool_result", "tool_use_id": "call-url", "content": [
                    {"type": "image", "source": {"type": "url", "url": "https://example.com/result.png"}}
                ]}
            ]}],
        });
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].role, MessageRole::User);
        assert_eq!(msgs[0].images, vec!["https://example.com/direct.png"]);
        assert_eq!(msgs[1].role, MessageRole::Tool);
        assert_eq!(msgs[1].tool_call_id, Some("call-url".to_string()));
        assert_eq!(msgs[2].role, MessageRole::User);
        assert_eq!(msgs[2].images, vec!["https://example.com/result.png"]);
    }

    #[test]
    fn test_malformed_known_image_sources_fail_loudly_in_direct_content() {
        for (source, expected_error) in [
            (
                json!({"type": "url", "url": ""}),
                "Anthropic URL image source requires a non-empty url",
            ),
            (
                json!({"type": "url", "url": 42}),
                "Anthropic URL image source requires a non-empty url",
            ),
            (
                json!({"type": "base64", "media_type": "image/png", "data": ""}),
                "Anthropic base64 image source requires non-empty data",
            ),
            (
                json!({"type": "base64", "media_type": 42, "data": "AAAA"}),
                "Anthropic base64 image source media_type must be one of: image/jpeg, image/png, image/gif, image/webp",
            ),
        ] {
            let body = json!({
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": source}
                ]}]
            });
            assert_eq!(
                anthropic_request_to_internal(&body).unwrap_err(),
                expected_error
            );
        }
    }

    #[test]
    fn test_missing_non_object_and_unknown_image_sources_fail_loudly() {
        for (image, expected_error) in [
            (
                json!({"type": "image"}),
                "Anthropic image block requires an object source",
            ),
            (
                json!({"type": "image", "source": "not-an-object"}),
                "Anthropic image block requires an object source",
            ),
            (
                json!({"type": "image", "source": {}}),
                "Anthropic image source type must be one of: base64, url",
            ),
            (
                json!({"type": "image", "source": {"type": "file", "file_id": "file-1"}}),
                "Anthropic image source type must be one of: base64, url",
            ),
        ] {
            let body = json!({
                "messages": [{"role": "user", "content": [image]}]
            });
            assert_eq!(
                anthropic_request_to_internal(&body).unwrap_err(),
                expected_error
            );
        }
    }

    #[test]
    fn test_malformed_known_image_source_fails_loudly_in_tool_result() {
        let body = json!({
            "messages": [{"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": "call-image",
                "content": [{
                    "type": "image",
                    "source": {"type": "url", "url": null}
                }]
            }]}]
        });
        assert_eq!(
            anthropic_request_to_internal(&body).unwrap_err(),
            "Anthropic URL image source requires a non-empty url"
        );
    }

    #[test]
    fn test_tool_result_error_preserves_explicit_prefix() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-error",
                    "is_error": true,
                    "content": [{"type": "text", "text": "command failed"}]
                }
            ]}],
        });
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, MessageRole::Tool);
        assert_eq!(msgs[0].content, "[tool_error]\ncommand failed");
    }

    #[test]
    fn test_user_tool_result_rejects_missing_or_non_string_id() {
        for block in [
            json!({"type": "tool_result", "content": "x"}),
            json!({"type": "tool_result", "tool_use_id": null, "content": "x"}),
            json!({"type": "tool_result", "tool_use_id": 1, "content": "x"}),
        ] {
            let body = json!({
                "messages": [{"role": "user", "content": [block]}],
            });
            assert!(anthropic_request_to_internal(&body)
                .unwrap_err()
                .contains("string tool_use_id"));
        }
    }

    #[test]
    fn test_assistant_text_string() {
        let body = json!({
            "messages": [{"role": "assistant", "content": "I can help."}],
        });
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(msgs[0].role, MessageRole::Assistant);
        assert_eq!(msgs[0].content, "I can help.");
    }

    #[test]
    fn test_assistant_text_and_tool_use() {
        let body = json!({
            "messages": [{"role": "assistant", "content": [
                {"type": "text", "text": "Calling tool."},
                {
                    "type": "tool_use",
                    "id": "tc-1",
                    "name": "search",
                    "input": {"q": "rust"},
                    "caller": {"type": "direct"}
                },
            ]}],
        });
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(msgs[0].content, "Calling tool.");
        assert_eq!(msgs[0].tool_calls.len(), 1);
        assert_eq!(msgs[0].tool_calls[0].id, "tc-1");
        assert_eq!(msgs[0].tool_calls[0].name, "search");
        assert_eq!(msgs[0].tool_calls[0].arguments, "{\"q\":\"rust\"}");
    }

    #[test]
    fn test_assistant_tool_use_rejects_non_direct_or_malformed_caller() {
        for caller in [
            Value::Null,
            json!("direct"),
            json!({}),
            json!({"type": "code_execution_20260120", "tool_id": "srv_1"}),
            json!({"type": "direct", "extra": true}),
        ] {
            let body = json!({
                "messages": [{"role": "assistant", "content": [{
                    "type": "tool_use",
                    "id": "tc-1",
                    "name": "search",
                    "input": {"q": "rust"},
                    "caller": caller
                }]}]
            });
            assert!(anthropic_request_to_internal(&body).is_err());
        }
    }

    #[test]
    fn test_assistant_thinking_block_is_rejected() {
        let body = json!({
            "messages": [{"role": "assistant", "content": [
                {"type": "thinking", "thinking": "Let me think..."},
                {"type": "text", "text": "Answer."},
            ]}],
        });
        assert!(anthropic_request_to_internal(&body).is_err());
    }

    #[test]
    fn test_rejects_lossy_assistant_server_web_search_history() {
        let body = json!({
            "messages": [{"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "codex"}},
                {"type": "web_search_tool_result", "tool_use_id": "srv_1", "content": [
                    {"title": "Codex", "url": "https://example.com", "page_age": "1d"},
                ]},
                {"type": "text", "text": "Summary"},
            ]}],
        });
        assert!(anthropic_request_to_internal(&body).is_err());
    }

    #[test]
    fn test_rejects_lossy_non_text_tool_result_blocks() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call-1", "content": [
                    {"type": "search_result", "title": "Docs", "url": "https://docs.example", "content": "body"},
                    {"type": "document", "title": "Spec", "source": {"type": "text", "data": "document body"}},
                ]},
            ]}],
        });
        assert!(anthropic_request_to_internal(&body).is_err());
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
        let (_, tools, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        let tools = tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "get_weather");
        assert_eq!(tools[0].description.as_deref(), Some("Get weather"));
        assert!(tools[0].parameters.get("properties").is_some());
    }

    #[test]
    fn test_nullable_eager_tool_control_is_omitted() {
        let body = json!({
            "messages": [],
            "tools": [{
                "name": "get_weather",
                "strict": false,
                "eager_input_streaming": null,
                "input_schema": {
                    "type": "object",
                    "properties": {"strict": {"type": "boolean"}}
                }
            }],
        });
        let (_, tools, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(tools.unwrap().len(), 1);
    }

    #[test]
    fn test_non_nullable_tool_controls_reject_null() {
        for control in [
            json!({"strict": null}),
            json!({"description": null}),
            json!({"defer_loading": null}),
            json!({"allowed_callers": null}),
        ] {
            let mut tool = json!({"name": "get_weather", "input_schema": {}});
            tool.as_object_mut()
                .unwrap()
                .extend(control.as_object().unwrap().clone());
            let body = json!({
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [tool]
            });
            assert!(anthropic_request_to_internal(&body).is_err());
        }
    }

    #[test]
    fn test_custom_tool_discriminator_variants() {
        for tool_type in [None, Some(json!("custom")), Some(json!(null))] {
            let mut tool = json!({"name": "lookup", "input_schema": {}});
            if let Some(tool_type) = tool_type {
                tool["type"] = tool_type;
            }
            let body = json!({
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [tool]
            });
            assert!(anthropic_request_to_internal(&body).is_ok());
        }
        for tool_type in [json!("future"), json!(1), json!(false), json!({})] {
            let body = json!({
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": tool_type, "name": "lookup", "input_schema": {}}]
            });
            assert!(anthropic_request_to_internal(&body).is_err());
        }
    }

    #[test]
    fn test_strict_is_forwarded_and_unrepresentable_tool_controls_fail_loudly() {
        let strict_body = json!({
            "messages": [],
            "tools": [{
                "name": "get_weather",
                "description": "",
                "strict": true,
                "input_schema": {"type": "object"},
            }],
        });
        let (_, tools, _, _, _, _) = anthropic_request_to_internal(&strict_body).unwrap();
        let tool = &tools.unwrap()[0];
        assert!(tool.strict);
        assert_eq!(tool.description.as_deref(), Some(""));

        for field in ["defer_loading", "eager_input_streaming"] {
            for value in [false, true] {
                let mut tool = json!({
                    "name": "get_weather",
                    "input_schema": {"type": "object"},
                });
                tool.as_object_mut()
                    .unwrap()
                    .insert(field.to_string(), json!(value));
                let body = json!({"messages": [], "tools": [tool]});
                assert_eq!(
                    anthropic_request_to_internal(&body).unwrap_err(),
                    format!("tool.{field} is not supported by the Codex OAuth backend")
                );
            }
        }
    }

    #[test]
    fn test_tool_choice_auto() {
        let body = json!({"messages": [], "tool_choice": {"type": "auto"}});
        let (_, _, tc, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(tc, Some(json!("auto")));
    }

    #[test]
    fn test_tool_choice_any() {
        let body = json!({"messages": [], "tool_choice": {"type": "any"}});
        let (_, _, tc, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(tc, Some(json!("required")));
    }

    #[test]
    fn test_tool_choice_tool() {
        let body = json!({"messages": [], "tool_choice": {"type": "tool", "name": "my_fn"}});
        let (_, _, tc, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(tc, Some(json!({"type": "function", "name": "my_fn"})));
    }

    #[test]
    fn test_tool_choice_web_search_is_rejected() {
        let body = json!({"messages": [], "tool_choice": {"type": "tool", "name": "web_search"}});
        assert!(anthropic_request_to_internal(&body)
            .unwrap_err()
            .contains("cannot be represented losslessly"));
    }

    #[test]
    fn test_tool_choice_none() {
        let body = json!({"messages": [], "tool_choice": {"type": "none"}});
        let (_, _, tc, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(tc, Some(json!("none")));
    }

    #[test]
    fn test_tool_choice_absent() {
        let body = json!({"messages": []});
        let (_, _, tc, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(tc, None);
    }

    #[test]
    fn test_hosted_web_search_tools_are_rejected() {
        for tool in [
            json!({"type": "web_search", "name": "web_search"}),
            json!({
                "type": "web_search_20250305",
                "name": "web_search",
                "blocked_domains": ["example.com"]
            }),
            json!({
                "type": "web_search_20260209",
                "name": "web_search",
                "allowed_domains": ["example.com"],
                "max_uses": 8,
                "strict": false,
                "user_location": {"type": "approximate", "country": "US"}
            }),
        ] {
            let body = json!({
                "messages": [],
                "tools": [tool],
            });
            assert!(anthropic_request_to_internal(&body)
                .unwrap_err()
                .contains("cannot be represented losslessly"));
        }

        // Unknown versioned declarations still fail as invalid custom tool types.
        for tool_type in ["web_search_20240101", "web_search_future"] {
            let body = json!({
                "messages": [],
                "tools": [{"type": tool_type, "name": "web_search"}],
            });
            assert!(anthropic_request_to_internal(&body).is_err());
        }
    }

    #[test]
    fn test_thinking_enabled() {
        let body = json!({"messages": [], "thinking": {"type": "enabled", "budget_tokens": 1024}});
        let (_, _, _, _, effort, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(effort, Some("high".to_string()));
    }

    #[test]
    fn test_thinking_enabled_accepts_safe_integral_json_numbers() {
        for budget in [json!(1024.0), json!(2e3), json!(9_007_199_254_740_991.0)] {
            let body = json!({
                "messages": [],
                "thinking": {"type": "enabled", "budget_tokens": budget}
            });
            let (_, _, _, _, effort, _) = anthropic_request_to_internal(&body).unwrap();
            assert_eq!(effort, Some("high".to_string()));
        }
    }

    #[test]
    fn test_thinking_enabled_requires_a_positive_integer_budget() {
        for budget in [
            None,
            Some(json!(null)),
            Some(json!(0)),
            Some(json!(1023)),
            Some(json!(-1)),
            Some(json!(1.5)),
            Some(json!("1024")),
        ] {
            let mut thinking = json!({"type": "enabled"});
            if let Some(budget) = budget {
                thinking["budget_tokens"] = budget;
            }
            let body = json!({"messages": [], "thinking": thinking});
            assert!(anthropic_request_to_internal(&body).is_err());
        }
    }

    #[test]
    fn test_enabled_thinking_budget_is_below_max_tokens_and_display_is_lossless() {
        let valid = json!({
            "messages": [],
            "max_tokens": 2048,
            "thinking": {"type": "enabled", "budget_tokens": 1024, "display": "omitted"}
        });
        assert!(anthropic_request_to_internal(&valid).is_ok());
        for thinking in [
            json!({"type": "enabled", "budget_tokens": 2048}),
            json!({"type": "enabled", "budget_tokens": 4096}),
            json!({"type": "enabled", "budget_tokens": 1024, "display": "summarized"}),
        ] {
            let body = json!({
                "messages": [],
                "max_tokens": 2048,
                "thinking": thinking
            });
            assert!(anthropic_request_to_internal(&body).is_err());
        }
    }

    #[test]
    fn test_tool_choice_none_rejects_parallel_control() {
        let body = json!({
            "messages": [],
            "tool_choice": {"type": "none", "disable_parallel_tool_use": false}
        });
        assert!(anthropic_request_to_internal(&body).is_err());
    }

    #[test]
    fn test_thinking_adaptive() {
        let body = json!({"messages": [], "thinking": {"type": "adaptive"}});
        let (_, _, _, _, effort, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(effort, Some("medium".to_string()));
    }

    #[test]
    fn test_thinking_adaptive_rejects_unrepresentable_display_modes() {
        for display in [json!("summarized"), json!(""), json!(false), json!({})] {
            let body = json!({
                "messages": [],
                "thinking": {"type": "adaptive", "display": display}
            });
            assert!(anthropic_request_to_internal(&body).is_err());
        }
    }

    #[test]
    fn test_thinking_disabled() {
        let body = json!({"messages": [], "thinking": {"type": "disabled"}});
        let (_, _, _, _, effort, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(effort, Some("none".to_string()));
    }

    #[test]
    fn test_thinking_unspecified_has_no_request_override() {
        let body = json!({"messages": []});
        let (_, _, _, _, effort, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(effort, None);
    }

    #[test]
    fn test_output_config_effort_precedes_enabled_and_adaptive_thinking() {
        for effort in ["low", "medium", "high", "xhigh", "max"] {
            let body = json!({
                "messages": [],
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": effort},
            });
            let (_, _, _, _, converted, _) = anthropic_request_to_internal(&body).unwrap();
            assert_eq!(converted, Some(effort.to_string()));
        }

        let enabled = json!({
            "messages": [],
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "output_config": {"effort": "low"},
        });
        let (_, _, _, _, converted, _) = anthropic_request_to_internal(&enabled).unwrap();
        assert_eq!(converted, Some("low".to_string()));
    }

    #[test]
    fn test_disabled_thinking_precedes_output_config_effort() {
        for effort in ["low", "medium", "high", "xhigh", "max"] {
            let body = json!({
                "messages": [],
                "thinking": {"type": "disabled"},
                "output_config": {"effort": effort},
            });
            let (_, _, _, _, converted, _) = anthropic_request_to_internal(&body).unwrap();
            assert_eq!(converted, Some("none".to_string()));
        }
    }

    #[test]
    fn test_output_config_effort_is_validated_before_disabled_thinking_precedence() {
        let invalid_efforts = [
            (
                json!(""),
                "output_config.effort must be a non-empty string when provided",
            ),
            (
                json!(42),
                "output_config.effort must be a non-empty string when provided",
            ),
            (
                json!("ultra"),
                "output_config.effort must be one of: low, medium, high, xhigh, max",
            ),
        ];
        for (effort, expected_error) in invalid_efforts {
            let body = json!({
                "messages": [],
                "thinking": {"type": "disabled"},
                "output_config": {"effort": effort},
            });
            assert_eq!(
                anthropic_request_to_internal(&body).unwrap_err(),
                expected_error
            );
        }
    }

    #[test]
    fn test_output_config_effort_must_be_a_non_empty_string() {
        for effort in [json!(""), json!("   "), json!(42), json!({"level": "high"})] {
            let body = json!({
                "messages": [],
                "output_config": {"effort": effort},
            });
            assert_eq!(
                anthropic_request_to_internal(&body).unwrap_err(),
                "output_config.effort must be a non-empty string when provided"
            );
        }
    }

    #[test]
    fn test_output_config_effort_rejects_unknown_values() {
        for effort in ["none", "ultra", "custom"] {
            let body = json!({
                "messages": [],
                "output_config": {"effort": effort},
            });
            assert_eq!(
                anthropic_request_to_internal(&body).unwrap_err(),
                "output_config.effort must be one of: low, medium, high, xhigh, max"
            );
        }
    }

    #[test]
    fn test_output_config_rejects_unknown_fields() {
        let body = json!({
            "messages": [],
            "output_config": {"effort": "low", "experimental": true},
        });
        assert_eq!(
            anthropic_request_to_internal(&body).unwrap_err(),
            "output_config.experimental is not supported by the Codex OAuth backend"
        );
    }

    #[test]
    fn test_output_config_task_budget_fails_loudly() {
        let body = json!({
            "messages": [],
            "output_config": {
                "task_budget": {"type": "tokens", "total": 20_000}
            },
        });
        assert_eq!(
            anthropic_request_to_internal(&body).unwrap_err(),
            "output_config.task_budget is not supported by the Codex OAuth backend"
        );

        let null_body = json!({
            "messages": [],
            "output_config": {"task_budget": null},
        });
        assert!(anthropic_request_to_internal(&null_body).is_ok());
    }

    #[test]
    fn test_stop_sequences() {
        let body = json!({"messages": [], "stop_sequences": ["STOP", "END"]});
        let (_, _, _, stop, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(stop, Some(vec!["STOP".to_string(), "END".to_string()]));
    }

    #[test]
    fn test_output_format_json_schema_maps_to_text_format() {
        let body = json!({
            "messages": [],
            "output_format": {
                "type": "json_schema",
                "name": "my_schema",
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
                "strict": false,
            },
        });
        let (_, _, _, _, _, text) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(
            text,
            Some(json!({"format": {
                "type": "json_schema",
                "name": "my_schema",
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
                "strict": false,
            }}))
        );
    }

    #[test]
    fn test_output_format_json_schema_uses_pinned_default_name() {
        let body = json!({
            "messages": [{"role": "user", "content": "hi"}],
            "output_format": {
                "type": "json_schema",
                "schema": {"type": "object"}
            },
        });
        let (_, _, _, _, _, text) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(
            text,
            Some(json!({"format": {
                "type": "json_schema",
                "name": "codex_output_schema",
                "schema": {"type": "object"},
            }}))
        );
    }

    #[test]
    fn test_output_format_nullable_strict_is_omitted() {
        let body = json!({
            "messages": [],
            "output_format": {
                "type": "json_schema",
                "schema": {"type": "object"},
                "strict": null,
            },
        });
        let (_, _, _, _, _, text) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(
            text,
            Some(json!({"format": {
                "type": "json_schema",
                "name": "codex_output_schema",
                "schema": {"type": "object"},
            }}))
        );
    }

    #[test]
    fn test_output_format_rejects_null_non_nullable_extensions() {
        for field in ["name", "description"] {
            let mut format = json!({
                "type": "json_schema",
                "schema": {"type": "object"},
            });
            format[field] = Value::Null;
            let body = json!({"messages": [], "output_format": format});
            assert!(anthropic_request_to_internal(&body)
                .unwrap_err()
                .contains(&format!("output_format.{field}")));
        }
    }

    #[test]
    fn test_output_config_effort_preserves_nested_format_mapping() {
        let body = json!({
            "messages": [],
            "output_config": {
                "effort": "medium",
                "format": {"type": "json_object"},
            },
        });
        let (_, _, _, _, effort, text) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(effort, Some("medium".to_string()));
        assert_eq!(text, Some(json!({"format": {"type": "json_object"}})));
    }

    #[test]
    fn test_output_formats_fail_loudly_in_top_level_and_nested_locations() {
        for (field, value, expected_error) in [
            (
                "output_format",
                json!("json_object"),
                "output_format must be an object when provided",
            ),
            (
                "output_format",
                json!({}),
                "output_format.type must be a string",
            ),
            (
                "output_format",
                json!({"type": "text"}),
                "output_format.type must be one of: json_object, json_schema",
            ),
            (
                "output_config.format",
                json!({"type": "json_schema"}),
                "output_config.format.schema must be an object",
            ),
            (
                "output_config.format",
                json!({"type": "json_object", "extra": true}),
                "output_config.format.extra is not supported",
            ),
        ] {
            let mut body = json!({"messages": []});
            if field == "output_format" {
                body["output_format"] = value;
            } else {
                body["output_config"] = json!({"format": value});
            }
            assert_eq!(
                anthropic_request_to_internal(&body).unwrap_err(),
                expected_error
            );
        }
    }

    #[test]
    fn test_dual_output_formats_must_match() {
        let conflicting = json!({
            "messages": [],
            "output_format": {"type": "json_object"},
            "output_config": {
                "format": {"type": "json_schema", "schema": {"type": "object"}}
            },
        });
        assert_eq!(
            anthropic_request_to_internal(&conflicting).unwrap_err(),
            "output_format conflicts with output_config.format"
        );

        let identical = json!({
            "messages": [],
            "output_format": {"type": "json_object"},
            "output_config": {"format": {"type": "json_object"}},
        });
        let (_, _, _, _, _, text) = anthropic_request_to_internal(&identical).unwrap();
        assert_eq!(text, Some(json!({"format": {"type": "json_object"}})));
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
            finish_reason: Some(finish_reason.to_string()),
            usage: usage.or(Some(crate::messages::Usage {
                prompt_tokens: 1,
                completion_tokens: 1,
                total_tokens: 2,
                cached_tokens: Some(0),
                cache_write_tokens: None,
            })),
            reasoning_content,
            raw: None,
            response_id: None,
        }
    }

    #[test]
    fn test_response_text_only() {
        let resp = make_response("Hello!", vec![], "stop", None, None);
        let out = internal_response_to_anthropic(&resp, "claude-3", "msg_abc").unwrap();
        assert_eq!(out["id"], "msg_abc");
        assert_eq!(out["role"], "assistant");
        assert_eq!(out["stop_reason"], "end_turn");
        assert_eq!(out["content"][0]["type"], "text");
        assert_eq!(out["content"][0]["text"], "Hello!");
        assert!(out["content"][0]["citations"].is_null());
        assert!(out["container"].is_null());
        assert!(out["context_management"].is_null());
    }

    #[test]
    fn test_response_tool_use() {
        let tc = ToolCall {
            id: "tc-1".to_string(),
            name: "search".to_string(),
            arguments: "{\"q\":\"rust\"}".to_string(),
        };
        let resp = make_response("", vec![tc], "tool_calls", None, None);
        let out = internal_response_to_anthropic(&resp, "claude-3", "msg_xyz").unwrap();
        assert_eq!(out["stop_reason"], "tool_use");
        let content = out["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "tool_use");
        assert_eq!(content[0]["name"], "search");
        assert_eq!(content[0]["caller"], json!({"type": "direct"}));

        let replay = json!({
            "messages": [{"role": "assistant", "content": content}]
        });
        let (messages, _, _, _, _, _) = anthropic_request_to_internal(&replay).unwrap();
        assert_eq!(messages[0].tool_calls.len(), 1);
        assert_eq!(messages[0].tool_calls[0].id, "tc-1");
        assert_eq!(messages[0].tool_calls[0].name, "search");
        assert_eq!(messages[0].tool_calls[0].arguments, "{\"q\":\"rust\"}");
    }

    #[test]
    fn test_empty_tool_use_id_roundtrips_through_tool_result() {
        let tc = ToolCall {
            id: String::new(),
            name: "lookup".to_string(),
            arguments: "{}".to_string(),
        };
        let resp = make_response("", vec![tc.clone()], "tool_calls", None, None);
        let out = internal_response_to_anthropic(&resp, "claude-3", "msg_empty_call").unwrap();
        let replay = json!({
            "messages": [
                {"role": "assistant", "content": out["content"]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "", "content": "done"}
                ]}
            ]
        });

        let (messages, _, _, _, _, _) = anthropic_request_to_internal(&replay).unwrap();
        assert_eq!(messages[0].tool_calls.len(), 1);
        assert_eq!(messages[0].tool_calls[0].id, tc.id);
        assert_eq!(messages[0].tool_calls[0].name, tc.name);
        assert_eq!(messages[0].tool_calls[0].arguments, tc.arguments);
        assert_eq!(messages[1].role, MessageRole::Tool);
        assert_eq!(messages[1].tool_call_id, Some(String::new()));
        assert_eq!(messages[1].content, "done");
    }

    #[test]
    fn test_response_reasoning() {
        let resp = make_response(
            "Answer.",
            vec![],
            "stop",
            None,
            Some("My reasoning.".to_string()),
        );
        let out = internal_response_to_anthropic(&resp, "claude-3", "msg_r").unwrap();
        let content = out["content"].as_array().unwrap();
        assert_eq!(out["codex_reasoning"], "My reasoning.");
        assert_eq!(content[0]["type"], "text");
        assert!(content
            .iter()
            .all(|block| block["type"].as_str() != Some("thinking")));
    }

    #[test]
    fn test_response_empty_content_is_an_empty_content_array() {
        let resp = make_response("", vec![], "stop", None, None);
        let out = internal_response_to_anthropic(&resp, "claude-3", "msg_e").unwrap();
        assert_eq!(out["content"], json!([]));
        assert_eq!(out["stop_reason"], "end_turn");
        let replay = json!({
            "messages": [{"role": "assistant", "content": out["content"]}]
        });
        let (messages, _, _, _, _, _) = anthropic_request_to_internal(&replay).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].role, MessageRole::Assistant);
        assert!(messages[0].content.is_empty());
    }

    #[test]
    fn test_response_usage_present() {
        let usage = crate::messages::Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
            cached_tokens: Some(20),
            cache_write_tokens: Some(9),
        };
        let resp = make_response("Hi", vec![], "stop", Some(usage), None);
        let out = internal_response_to_anthropic(&resp, "claude-3", "msg_u").unwrap();
        assert_eq!(out["usage"]["input_tokens"], 100);
        assert_eq!(out["usage"]["output_tokens"], 50);
        assert_eq!(out["usage"]["cache_read_input_tokens"], 20);
        assert_eq!(out["usage"]["cache_creation_input_tokens"], 9);
        assert!(out["usage"]["cache_creation"].is_null());
        assert!(out["usage"]["inference_geo"].is_null());
        assert!(out["usage"]["iterations"].is_null());
        assert!(out["usage"]["server_tool_use"].is_null());
        assert!(out["usage"]["service_tier"].is_null());
        assert!(out["usage"]["speed"].is_null());
    }

    #[test]
    fn test_response_usage_absent_is_rejected() {
        let mut resp = make_response("Hi", vec![], "stop", None, None);
        resp.usage = None;
        assert!(internal_response_to_anthropic(&resp, "claude-3", "msg_nu")
            .unwrap_err()
            .contains("authoritative usage"));
    }

    #[test]
    fn test_response_null_finish_reason_is_rejected() {
        let mut resp = make_response("Hi", vec![], "stop", None, None);
        resp.finish_reason = None;
        assert!(
            internal_response_to_anthropic(&resp, "claude-3", "msg_null_finish")
                .unwrap_err()
                .contains("non-null finish_reason")
        );
    }

    #[test]
    fn test_response_web_search_output_is_rejected() {
        let mut resp = make_response("answer", vec![], "stop", None, None);
        resp.raw = Some(json!({"events": [{"type": "web_search_call"}]}));
        assert!(internal_response_to_anthropic(&resp, "claude-3", "msg_web")
            .unwrap_err()
            .contains("cannot be represented losslessly"));
    }

    #[test]
    fn test_response_diagnostic_raw_events_do_not_require_a_duplicate_finish() {
        let mut resp = make_response("answer", vec![], "stop", None, None);
        resp.raw = Some(json!({
            "events": [{"type": "reasoning_section_break", "summary_index": 1}]
        }));

        let output = internal_response_to_anthropic(&resp, "claude-3", "msg_diagnostic").unwrap();
        assert_eq!(output["stop_reason"], "end_turn");
        assert_eq!(output["usage"]["input_tokens"], 1);
    }

    #[test]
    fn test_response_stop_reason_length() {
        let resp = make_response("truncated", vec![], "length", None, None);
        assert_eq!(
            internal_response_to_anthropic(&resp, "m", "id").unwrap_err(),
            "unsupported finish_reason Some(\"length\")"
        );
    }

    #[test]
    fn test_response_stop_reason_max_tokens() {
        let resp = make_response("truncated", vec![], "max_tokens", None, None);
        assert_eq!(
            internal_response_to_anthropic(&resp, "m", "id").unwrap_err(),
            "unsupported finish_reason Some(\"max_tokens\")"
        );
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

    fn provider_usage(input_tokens: i64, output_tokens: i64) -> Value {
        json!({
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_tokens_details": {"cached_tokens": 0}
        })
    }

    fn finish_event(reason: &str, input_tokens: i64, output_tokens: i64) -> Value {
        json!({
            "type": "finish",
            "finish_reason": reason,
            "usage": provider_usage(input_tokens, output_tokens),
        })
    }

    #[test]
    fn test_stream_text_only() {
        let events = vec![
            json!({"type": "content", "text": "Hello"}),
            finish_event("stop", 1, 1),
        ];
        let chunks = anthropic_stream_adapter(&events, "claude-3", "msg_s1");
        let parsed = get_sse_events(&chunks);
        assert_eq!(parsed[0].0, "message_start");
        assert!(parsed[0].1["message"].get("usage").is_none());
        assert!(parsed[0].1["message"]["container"].is_null());
        assert!(parsed[0].1["message"]["context_management"].is_null());
        assert_eq!(parsed[1].0, "content_block_start");
        assert_eq!(parsed[1].1["content_block"]["type"], "text");
        assert!(parsed[1].1["content_block"]["citations"].is_null());
        assert_eq!(parsed[2].0, "content_block_delta");
        assert_eq!(parsed[2].1["delta"]["text"], "Hello");
        assert_eq!(parsed[3].0, "content_block_stop");
        assert_eq!(parsed[4].0, "message_delta");
        assert_eq!(parsed[4].1["delta"]["stop_reason"], "end_turn");
        assert_eq!(parsed[5].0, "message_stop");
    }

    #[test]
    fn stream_preserves_empty_text_and_reasoning_deltas() {
        let events = vec![
            json!({"type": "reasoning_delta", "text": ""}),
            json!({"type": "reasoning_raw_delta", "text": ""}),
            json!({"type": "content", "text": ""}),
            finish_event("stop", 0, 0),
        ];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);

        let reasoning: Vec<Value> = parsed
            .iter()
            .filter(|(event, _)| event == "codex_reasoning_delta")
            .map(|(_, data)| data["delta"].clone())
            .collect();
        let text: Vec<Value> = parsed
            .iter()
            .filter(|(event, data)| {
                event == "content_block_delta" && data["delta"]["type"] == "text_delta"
            })
            .map(|(_, data)| data["delta"]["text"].clone())
            .collect();
        assert_eq!(reasoning, vec![json!(""), json!("")]);
        assert_eq!(text, vec![json!("")]);
    }

    #[test]
    fn test_stream_thinking_then_text() {
        let events = vec![
            json!({"type": "reasoning_delta", "text": "thinking..."}),
            json!({"type": "content", "text": "answer"}),
            finish_event("stop", 1, 1),
        ];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);
        assert_eq!(parsed[1].0, "codex_reasoning_delta");
        assert_eq!(
            parsed[1].1,
            json!({"type": "codex_reasoning_delta", "delta": "thinking..."})
        );
        assert_eq!(parsed[2].0, "content_block_start");
        assert_eq!(parsed[2].1["content_block"]["type"], "text");
        assert!(parsed.iter().all(|(_, data)| {
            data["content_block"]["type"] != "thinking"
                && data["delta"]["type"] != "signature_delta"
        }));
    }

    #[test]
    fn test_stream_tool_call() {
        let events = vec![
            json!({"type": "tool_call", "id": "tc-1", "name": "search", "arguments": "{\"q\":\"rust\"}"}),
            finish_event("tool_calls", 1, 1),
        ];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);
        assert_eq!(parsed[1].0, "content_block_start");
        assert_eq!(parsed[1].1["content_block"]["type"], "tool_use");
        assert_eq!(parsed[1].1["content_block"]["name"], "search");
        assert_eq!(
            parsed[1].1["content_block"]["caller"],
            json!({"type": "direct"})
        );
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
            json!({"type": "tool_call", "id": "tc-2", "name": "fn", "arguments": "{}"}),
            finish_event("tool_calls", 1, 1),
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
        let message_delta = parsed
            .iter()
            .find(|(event, _)| event == "message_delta")
            .unwrap();
        assert_eq!(message_delta.1["delta"]["stop_reason"], "tool_use");
    }

    #[test]
    fn test_stream_empty_response_finishes_without_content_blocks() {
        let mut adapter = AnthropicStreamAdapter::new("m", "id");
        let mut chunks = adapter.start();
        chunks.extend(adapter.push(&finish_event("stop", 1, 0)).unwrap());
        let parsed = get_sse_events(&chunks);
        assert_eq!(
            parsed
                .iter()
                .map(|(event, _)| event.as_str())
                .collect::<Vec<_>>(),
            vec!["message_start", "message_delta", "message_stop"]
        );
    }

    #[test]
    fn test_stream_stop_reason_length() {
        let mut adapter = AnthropicStreamAdapter::new("m", "id");
        assert_eq!(
            adapter.push(&finish_event("length", 1, 1)).unwrap_err(),
            "unsupported finish_reason Some(\"length\")"
        );
    }

    #[test]
    fn test_stream_usage_passed_through() {
        let events = vec![
            json!({
                "type": "content",
                "text": "answer"
            }),
            finish_event("stop", 8, 42),
        ];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);
        let msg_delta = parsed.iter().find(|(e, _)| e == "message_delta").unwrap();
        assert_eq!(msg_delta.1["usage"]["output_tokens"], 42);
    }

    #[test]
    fn test_stream_maps_provider_cache_token_details() {
        let events = vec![
            json!({
                "type": "content",
                "text": "answer"
            }),
            json!({
                "type": "finish",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 3,
                    "total_tokens": 23,
                    "input_tokens_details": {
                        "cached_tokens": 7,
                        "cache_write_tokens": 11
                    }
                },
            }),
        ];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);
        let msg_delta = parsed
            .iter()
            .find(|(event, _)| event == "message_delta")
            .unwrap();
        assert_eq!(msg_delta.1["usage"]["cache_read_input_tokens"], 7);
        assert_eq!(msg_delta.1["usage"]["cache_creation_input_tokens"], 11);
    }

    #[test]
    fn test_stream_routes_real_cumulative_usage_in_message_delta() {
        let events = vec![
            json!({"type": "content", "text": "hi"}),
            json!({
                "type": "finish",
                "finish_reason": "stop",
                "usage": {
                    "input_tokens": 123,
                    "output_tokens": 7,
                    "total_tokens": 130,
                    "input_tokens_details": {
                        "cached_tokens": 13,
                        "cache_write_tokens": 11
                    },
                    "cache_creation": {"ephemeral_5m_input_tokens": 11, "ephemeral_1h_input_tokens": 0},
                    "server_tool_use": {"web_search_requests": 2, "web_fetch_requests": 1}
                },
            }),
        ];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);
        let msg_start = parsed.iter().find(|(e, _)| e == "message_start").unwrap();
        assert!(msg_start.1["message"].get("usage").is_none());
        let msg_delta = parsed.iter().find(|(e, _)| e == "message_delta").unwrap();
        assert_eq!(msg_delta.1["usage"]["input_tokens"], 123);
        assert_eq!(msg_delta.1["usage"]["output_tokens"], 7);
        assert_eq!(msg_delta.1["usage"]["cache_creation_input_tokens"], 11);
        assert_eq!(msg_delta.1["usage"]["cache_read_input_tokens"], 13);
        assert!(msg_delta.1["usage"]["iterations"].is_null());
        assert_eq!(
            msg_delta.1["usage"]["server_tool_use"]["web_search_requests"],
            2
        );
        assert_eq!(
            msg_delta.1["usage"]["server_tool_use"]["web_fetch_requests"],
            1
        );
        assert!(msg_delta.1["context_management"].is_null());
        assert!(msg_delta.1["delta"]["container"].is_null());
    }

    #[test]
    fn malformed_actual_usage_extension_counters_fail_loudly() {
        let invalid_extensions = [
            ("cache_creation", json!("invalid")),
            ("cache_creation", json!({"future_counter": 1})),
            ("cache_creation", json!({"ephemeral_5m_input_tokens": -1})),
            (
                "server_tool_use",
                json!({"web_search_requests": 1, "web_fetch_requests": 0, "future_counter": 1}),
            ),
            (
                "server_tool_use",
                json!({"web_search_requests": 1.5, "web_fetch_requests": 0}),
            ),
            ("server_tool_use", json!({"web_search_requests": 1})),
            ("service_tier", json!({"id": "priority"})),
            ("service_tier", json!("")),
        ];

        for (field, value) in invalid_extensions {
            let mut usage = provider_usage(1, 1);
            usage[field] = value;
            assert!(anthropic_usage_from_provider(&usage).is_err());

            let raw = json!({"events": [{"type": "finish", "usage": usage}]});
            assert!(merge_actual_usage_extensions(json!({}), Some(&raw)).is_err());
        }
    }

    #[test]
    fn test_stream_multiple_tool_calls() {
        let events = vec![
            json!({"type": "tool_call", "id": "tc-1", "name": "fn1", "arguments": "{}"}),
            json!({"type": "tool_call", "id": "tc-2", "name": "fn2", "arguments": "{}"}),
            finish_event("tool_calls", 1, 1),
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

    #[test]
    fn test_stream_rejects_missing_final_contract_and_web_search_output() {
        let mut adapter = AnthropicStreamAdapter::new("m", "id");
        assert!(adapter
            .push(&json!({"type": "finish", "finish_reason": null, "usage": provider_usage(1, 1)}))
            .unwrap_err()
            .contains("non-null finish_reason"));

        let mut adapter = AnthropicStreamAdapter::new("m", "id");
        assert!(adapter
            .push(&json!({"type": "finish", "finish_reason": "stop"}))
            .unwrap_err()
            .contains("authoritative usage"));

        let mut adapter = AnthropicStreamAdapter::new("m", "id");
        assert!(adapter
            .push(&json!({"type": "web_search_call"}))
            .unwrap_err()
            .contains("cannot be represented losslessly"));
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
