use crate::messages::{AssistantResponse, Message, MessageRole, ToolCall, ToolSchema};
use serde_json::{json, Value};
use std::collections::HashMap;

pub fn anthropic_request_to_internal(
    body: &Value,
) -> Result<
    (
        Vec<Message>,
        Option<Vec<ToolSchema>>,
        Option<Value>,
        Option<Vec<String>>,
        Option<String>,
        Option<Value>,
    ),
    String,
> {
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
                structured_content: None,
            });
        }
    }

    if let Some(msgs) = body.get("messages").and_then(|v| v.as_array()) {
        for msg in msgs {
            let role = msg.get("role").and_then(|v| v.as_str()).unwrap_or("user");
            let empty = Value::String(String::new());
            let content = msg.get("content").unwrap_or(&empty);
            if role == "user" {
                convert_user_message(content, &mut messages)?;
            } else if role == "assistant" {
                convert_assistant_message(content, &mut messages);
            }
        }
    }

    let tools = if let Some(tools) = body
        .get("tools")
        .and_then(|v| v.as_array())
        .filter(|a| !a.is_empty())
    {
        validate_anthropic_tool_controls(tools)?;
        Some(convert_tools(tools))
    } else {
        None
    };

    let tool_choice = body.get("tool_choice").map(|tc| convert_tool_choice(tc));

    let stop = body
        .get("stop_sequences")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect()
        });

    let reasoning_effort = convert_reasoning_effort(body)?;
    let text =
        anthropic_output_format_from_body(body)?.and_then(anthropic_output_format_to_openai_text);

    Ok((messages, tools, tool_choice, stop, reasoning_effort, text))
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
                                structured_content: None,
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
                        let (mut result_content, tool_result_images) =
                            extract_tool_result_content_with_images(raw_content)?;
                        if block.get("is_error").and_then(Value::as_bool) == Some(true) {
                            result_content = format!("[tool_error]\n{result_content}");
                        }
                        out.push(Message {
                            role: MessageRole::Tool,
                            content: result_content,
                            tool_calls: vec![],
                            tool_call_id: Some(tool_use_id.clone()),
                            name: Some(tool_use_id),
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
                        image_urls.push(anthropic_image_source_url(block.get("source"))?);
                    }
                    _ => {
                        let rendered = render_anthropic_content_block(block);
                        if !rendered.is_empty() {
                            text_parts.push(rendered);
                        }
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
        }
        _ => {}
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
            for p in arr {
                let typ = p.get("type").and_then(|v| v.as_str()).unwrap_or("");
                if typ == "text" {
                    if let Some(t) = p.get("text").and_then(|v| v.as_str()) {
                        text_pieces.push(t.to_string());
                    }
                } else if typ == "image" {
                    images.push(anthropic_image_source_url(p.get("source"))?);
                } else {
                    let rendered = render_anthropic_content_block(p);
                    if !rendered.is_empty() {
                        text_pieces.push(rendered);
                    }
                }
            }
            (text_pieces.join(""), images)
        }
        Value::Null => (String::new(), vec![]),
        other => (other.to_string(), vec![]),
    })
}

fn anthropic_image_source_url(source: Option<&Value>) -> Result<String, String> {
    let source = source
        .and_then(Value::as_object)
        .ok_or_else(|| "Anthropic image block requires an object source".to_string())?;
    match source.get("type").and_then(Value::as_str) {
        Some("base64") => {
            let media_type = match source.get("media_type") {
                None => "image/png",
                Some(Value::String(media_type)) => media_type,
                Some(_) => {
                    return Err(
                        "Anthropic base64 image source requires a string media_type".to_string()
                    );
                }
            };
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
                structured_content: None,
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
                        tool_calls.push(ToolCall {
                            id,
                            name,
                            arguments,
                        });
                    }
                    "thinking" => {
                        if let Some(thinking_text) = block.get("thinking").and_then(|v| v.as_str())
                        {
                            if !thinking_text.is_empty() {
                                reasoning_content = Some(thinking_text.to_string());
                            }
                        }
                    }
                    "redacted_thinking" => {
                        if reasoning_content.is_none() {
                            reasoning_content = Some("[redacted_thinking omitted]".to_string());
                        }
                    }
                    "server_tool_use" => {
                        text_parts.push(render_server_tool_use_block(block));
                    }
                    "web_search_tool_result" => {
                        text_parts.push(render_generic_tool_result_block(block));
                    }
                    _ => {
                        let rendered = render_anthropic_content_block(block);
                        if !rendered.is_empty() {
                            text_parts.push(rendered);
                        }
                    }
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
                structured_content: None,
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
        if is_anthropic_web_search_tool(tool) {
            result.push(ToolSchema {
                name: "web_search".to_string(),
                description: "Anthropic hosted web search".to_string(),
                parameters: anthropic_web_search_parameters(tool),
            });
            continue;
        }
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

fn validate_anthropic_tool_controls(tools: &[Value]) -> Result<(), String> {
    for tool in tools {
        let Some(tool) = tool.as_object() else {
            continue;
        };
        for field in ["strict", "defer_loading", "eager_input_streaming"] {
            match tool.get(field) {
                None | Some(Value::Null) | Some(Value::Bool(false)) => {}
                Some(Value::Bool(true)) => {
                    return Err(format!(
                        "tool.{field}=true is not supported by the Codex OAuth backend"
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
            .map(|s| s == "web_search" || s.starts_with("web_search_"))
            .unwrap_or(false)
}

fn string_array(value: Option<&Value>) -> Option<Vec<String>> {
    let arr = value?.as_array()?;
    let out: Vec<String> = arr
        .iter()
        .filter_map(|v| v.as_str().filter(|s| !s.is_empty()).map(|s| s.to_string()))
        .collect();
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn anthropic_web_search_parameters(tool: &Value) -> Value {
    if string_array(tool.get("blocked_domains")).is_some() {
        return json!({
            "__codex_as_api_tool_type": "web_search",
            "__codex_as_api_error": "Anthropic web_search blocked_domains is not supported by OpenAI Responses web_search; use allowed_domains instead",
        });
    }
    let mut openai_tool = serde_json::Map::new();
    openai_tool.insert("type".to_string(), json!("web_search"));
    openai_tool.insert("external_web_access".to_string(), json!(true));
    if let Some(allowed) = string_array(tool.get("allowed_domains")) {
        openai_tool.insert("filters".to_string(), json!({"allowed_domains": allowed}));
    }
    if let Some(user_location) = tool.get("user_location").filter(|v| v.is_object()) {
        openai_tool.insert("user_location".to_string(), user_location.clone());
    }
    json!({
        "__codex_as_api_tool_type": "web_search",
        "openai_tool": Value::Object(openai_tool),
        "anthropic": {
            "type": tool.get("type").cloned().unwrap_or(Value::Null),
            "max_uses": tool.get("max_uses").cloned().unwrap_or(Value::Null),
        },
    })
}

fn convert_tool_choice(tc: &Value) -> Value {
    let tc_type = tc.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match tc_type {
        "auto" => json!("auto"),
        "any" => json!("required"),
        "tool" => {
            let name = tc.get("name").and_then(|v| v.as_str()).unwrap_or("");
            if name == "web_search" {
                json!({"type": "web_search"})
            } else {
                json!({"type": "function", "name": name})
            }
        }
        "none" => json!("none"),
        _ => json!("auto"),
    }
}

fn convert_thinking(thinking: &Value) -> Option<String> {
    match thinking.get("type").and_then(|v| v.as_str()) {
        Some("enabled") => Some("high".to_string()),
        Some("adaptive") => Some("medium".to_string()),
        Some("disabled") => Some("none".to_string()),
        _ => None,
    }
}

fn convert_reasoning_effort(body: &Value) -> Result<Option<String>, String> {
    let thinking_effort = body.get("thinking").and_then(convert_thinking);
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
        Some(Value::String(_)) if thinking_effort.as_deref() == Some("none") => {
            Err("output_config.effort cannot be used when thinking.type is disabled".to_string())
        }
        Some(Value::String(effort))
            if matches!(effort.as_str(), "low" | "medium" | "high" | "xhigh" | "max") =>
        {
            Ok(Some(effort.clone()))
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
    if object
        .get("name")
        .is_some_and(|value| !value.as_str().is_some_and(|name| !name.is_empty()))
    {
        return Err(format!("{field}.name must be a non-empty string"));
    }
    if object
        .get("description")
        .is_some_and(|value| !value.is_string())
    {
        return Err(format!("{field}.description must be a string"));
    }
    if object
        .get("strict")
        .is_some_and(|value| !value.is_boolean())
    {
        return Err(format!("{field}.strict must be a boolean"));
    }
    Ok(())
}

pub fn anthropic_output_format_to_openai_text(output_format: &Value) -> Option<Value> {
    let typ = output_format.get("type").and_then(|v| v.as_str())?;
    match typ {
        "json_schema" => {
            let schema = output_format.get("schema")?.as_object()?;
            let name = sanitize_json_schema_name(
                output_format
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("structured_output"),
            );
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
            Some(json!({"format": Value::Object(format)}))
        }
        "json_object" => Some(json!({"format": {"type": "json_object"}})),
        _ => None,
    }
}

fn sanitize_json_schema_name(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .take(64)
        .collect();
    if cleaned.is_empty() {
        "structured_output".to_string()
    } else {
        cleaned
    }
}

fn render_anthropic_content_block(block: &Value) -> String {
    let typ = block
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    match typ {
        "document" => render_document_block(block),
        "search_result" => render_search_result_block(block),
        _ if typ.ends_with("_tool_result") => render_generic_tool_result_block(block),
        _ => format!("\n\n[{}] {}\n", typ, safe_json(block)),
    }
}

fn render_document_block(block: &Value) -> String {
    let title = block
        .get("title")
        .or_else(|| block.get("name"))
        .and_then(|v| v.as_str())
        .unwrap_or("document");
    let mut body = String::new();
    if let Some(source) = block.get("source").and_then(|v| v.as_object()) {
        if source.get("type").and_then(|v| v.as_str()) == Some("text") {
            body = source
                .get("data")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
        } else if source.get("type").and_then(|v| v.as_str()) == Some("url") {
            body = source
                .get("url")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
        } else if let Some(media_type) = source.get("media_type").and_then(|v| v.as_str()) {
            body = format!("[{}]", media_type);
        }
    }
    if body.is_empty() {
        format!("\n\n[document: {}]\n", title)
    } else {
        format!("\n\n[document: {}]\n{}\n", title, body)
    }
}

fn render_search_result_block(block: &Value) -> String {
    let title = block
        .get("title")
        .and_then(|v| v.as_str())
        .unwrap_or("search result");
    let url = block.get("url").and_then(|v| v.as_str()).unwrap_or("");
    let content = block.get("content").and_then(|v| v.as_str()).unwrap_or("");
    format!(
        "\n\n[search_result] {}{}{}\n",
        title,
        if url.is_empty() {
            String::new()
        } else {
            format!(" ({})", url)
        },
        if content.is_empty() {
            String::new()
        } else {
            format!("\n{}", content)
        },
    )
}

fn render_server_tool_use_block(block: &Value) -> String {
    let name = block
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("server_tool");
    let input = block.get("input").unwrap_or(&Value::Null);
    format!("\n\n[server_tool_use: {}] {}\n", name, safe_json(input))
}

fn render_generic_tool_result_block(block: &Value) -> String {
    let typ = block
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("tool_result");
    let Some(content) = block.get("content") else {
        return format!("\n\n[{}]\n", typ);
    };
    if let Some(arr) = content.as_array() {
        let lines: Vec<String> = arr
            .iter()
            .map(|item| {
                if let Some(obj) = item.as_object() {
                    let title = obj.get("title").and_then(|v| v.as_str());
                    let url = obj.get("url").and_then(|v| v.as_str()).unwrap_or("");
                    let text = obj
                        .get("text")
                        .or_else(|| obj.get("content"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    if title.is_some() || !url.is_empty() || !text.is_empty() {
                        format!(
                            "- {}{}{}",
                            title.unwrap_or("result"),
                            if url.is_empty() {
                                String::new()
                            } else {
                                format!(" ({})", url)
                            },
                            if text.is_empty() {
                                String::new()
                            } else {
                                format!(": {}", text)
                            },
                        )
                    } else {
                        safe_json(item)
                    }
                } else {
                    item.to_string()
                }
            })
            .collect();
        if lines.is_empty() {
            format!("\n\n[{}]\n", typ)
        } else {
            format!("\n\n[{}]\n{}\n", typ, lines.join("\n"))
        }
    } else if let Some(s) = content.as_str() {
        format!("\n\n[{}]\n{}\n", typ, s)
    } else {
        format!("\n\n[{}] {}\n", typ, safe_json(content))
    }
}

fn safe_json(value: &Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| value.to_string())
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

    let web_search_blocks = web_search_blocks_from_raw(response.raw.as_ref());
    content.extend(web_search_blocks.clone());

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

    let mut usage_dict = match &response.usage {
        Some(u) => json!({
            "input_tokens": u.prompt_tokens,
            "output_tokens": u.completion_tokens,
            "cache_creation_input_tokens": u.cache_write_tokens,
            "cache_read_input_tokens": u.cached_tokens,
        }),
        None => json!({"input_tokens": 0, "output_tokens": 0}),
    };
    usage_dict = merge_server_tool_usage(
        usage_dict,
        response.raw.as_ref(),
        web_search_blocks.len() / 2,
    );

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

fn web_search_blocks_from_raw(raw: Option<&Value>) -> Vec<Value> {
    let mut blocks = Vec::new();
    let Some(events) = raw.and_then(|r| r.get("events")).and_then(|v| v.as_array()) else {
        return blocks;
    };
    for event in events {
        if event.get("type").and_then(|v| v.as_str()) != Some("web_search_call") {
            continue;
        }
        let tool_id = event
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| format!("srvtoolu_{}", blocks.len() / 2));
        let input = event
            .get("input")
            .filter(|v| v.is_object())
            .cloned()
            .unwrap_or_else(|| json!({"query": ""}));
        let content = event
            .get("content")
            .filter(|v| v.is_array())
            .cloned()
            .unwrap_or_else(|| json!([]));
        blocks.push(
            json!({"type": "server_tool_use", "id": tool_id, "name": "web_search", "input": input}),
        );
        blocks.push(
            json!({"type": "web_search_tool_result", "tool_use_id": tool_id, "content": content}),
        );
    }
    blocks
}

fn merge_server_tool_usage(
    mut usage: Value,
    raw: Option<&Value>,
    web_search_requests: usize,
) -> Value {
    if let Some(events) = raw.and_then(|r| r.get("events")).and_then(|v| v.as_array()) {
        for event in events {
            if event.get("type").and_then(|v| v.as_str()) != Some("finish") {
                continue;
            }
            if let Some(server_tool_use) = event
                .get("usage")
                .and_then(|u| u.get("server_tool_use"))
                .cloned()
            {
                if let Some(map) = usage.as_object_mut() {
                    map.insert("server_tool_use".to_string(), server_tool_use);
                }
                return usage;
            }
        }
    }
    if web_search_requests > 0 {
        if let Some(map) = usage.as_object_mut() {
            map.entry("server_tool_use".to_string())
                .or_insert_with(|| json!({"web_search_requests": web_search_requests}));
        }
    }
    usage
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
        "pause_turn" => "pause_turn",
        "refusal" => "refusal",
        _ => "end_turn",
    }
}

pub struct AnthropicStreamAdapter {
    model: String,
    request_id: String,
    started: bool,
    block_index: u32,
    current_block: Option<&'static str>,
    has_any_content: bool,
    has_tool_calls: bool,
    web_search_requests: usize,
}

impl AnthropicStreamAdapter {
    pub fn new(model: &str, request_id: &str) -> Self {
        Self {
            model: model.to_string(),
            request_id: request_id.to_string(),
            started: false,
            block_index: 0,
            current_block: None,
            has_any_content: false,
            has_tool_calls: false,
            web_search_requests: 0,
        }
    }

    pub fn start(&mut self) -> Vec<String> {
        if self.started {
            return vec![];
        }
        self.started = true;
        vec![message_start_sse(
            &self.model,
            &self.request_id,
            &json!({"input_tokens": 0, "output_tokens": 0}),
        )]
    }

    pub fn push(&mut self, event: &Value) -> Vec<String> {
        let mut out = Vec::new();
        let mut block_index = self.block_index;
        let mut current_block = self.current_block;
        let mut has_any_content = self.has_any_content;
        let mut has_tool_calls = self.has_tool_calls;
        let mut web_search_requests = self.web_search_requests;
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
                has_tool_calls = true;
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
                let tool_id = event
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let tool_name = event
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
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

            "web_search_call" => {
                has_any_content = true;
                web_search_requests += 1;
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
                let tool_id = event.get("id").and_then(|v| v.as_str()).unwrap_or("");
                let tool_input = event
                    .get("input")
                    .filter(|v| v.is_object())
                    .cloned()
                    .unwrap_or_else(|| json!({"query": ""}));
                let result_content = event
                    .get("content")
                    .filter(|v| v.is_array())
                    .cloned()
                    .unwrap_or_else(|| json!([]));
                out.push(sse(
                    "content_block_start",
                    &json!({
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {"type": "server_tool_use", "id": tool_id, "name": "web_search", "input": {}},
                    }),
                ));
                out.push(sse(
                    "content_block_delta",
                    &json!({
                        "type": "content_block_delta",
                        "index": block_index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": serde_json::to_string(&tool_input).unwrap_or_else(|_| "{}".to_string()),
                        },
                    }),
                ));
                out.push(sse(
                    "content_block_stop",
                    &json!({"type": "content_block_stop", "index": block_index}),
                ));
                block_index += 1;
                out.push(sse(
                    "content_block_start",
                    &json!({
                        "type": "content_block_start",
                        "index": block_index,
                        "content_block": {
                            "type": "web_search_tool_result",
                            "tool_use_id": tool_id,
                            "content": result_content,
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
                let stop_reason = map_stop_reason(finish_reason, has_tool_calls);

                out.push(sse(
                    "message_delta",
                    &json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": null},
                        "usage": usage_with_synthesized_web_search(event
                            .get("usage")
                            .map(anthropic_usage_from_provider)
                            .unwrap_or_else(|| json!({"input_tokens": 0, "output_tokens": 0})), web_search_requests),
                    }),
                ));
                out.push(sse("message_stop", &json!({"type": "message_stop"})));
            }

            _ => {}
        }

        self.block_index = block_index;
        self.current_block = current_block;
        self.has_any_content = has_any_content;
        self.has_tool_calls = has_tool_calls;
        self.web_search_requests = web_search_requests;
        out
    }
}

#[cfg(test)]
pub fn anthropic_stream_adapter(events: &[Value], model: &str, request_id: &str) -> Vec<String> {
    let mut adapter = AnthropicStreamAdapter::new(model, request_id);
    let mut out = adapter.start();
    for event in events {
        out.extend(adapter.push(event));
    }
    out
}

fn usage_with_synthesized_web_search(mut usage: Value, web_search_requests: usize) -> Value {
    if web_search_requests > 0 {
        if let Some(map) = usage.as_object_mut() {
            map.entry("server_tool_use".to_string())
                .or_insert_with(|| json!({"web_search_requests": web_search_requests}));
        }
    }
    usage
}

fn message_start_sse(model: &str, request_id: &str, usage: &Value) -> String {
    sse(
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
                "usage": usage,
            },
        }),
    )
}

fn usage_i64(usage: &Value, key: &str, fallback_key: Option<&str>) -> i64 {
    usage
        .get(key)
        .or_else(|| fallback_key.and_then(|k| usage.get(k)))
        .and_then(|v| v.as_i64())
        .unwrap_or(0)
}

fn anthropic_usage_from_provider(usage: &Value) -> Value {
    let mut cache_read = usage_i64(
        usage,
        "cache_read_input_tokens",
        Some("cached_input_tokens"),
    );
    if cache_read == 0 {
        cache_read = usage
            .get("input_tokens_details")
            .or_else(|| usage.get("prompt_tokens_details"))
            .and_then(|details| details.get("cached_tokens"))
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
    }
    let mut cache_creation = usage_i64(usage, "cache_creation_input_tokens", None);
    if cache_creation == 0 {
        cache_creation = usage
            .get("input_tokens_details")
            .or_else(|| usage.get("prompt_tokens_details"))
            .and_then(|details| details.get("cache_write_tokens"))
            .and_then(Value::as_i64)
            .unwrap_or(0);
    }
    let mut out = serde_json::Map::new();
    out.insert(
        "input_tokens".to_string(),
        json!(usage_i64(usage, "input_tokens", Some("prompt_tokens"))),
    );
    out.insert(
        "output_tokens".to_string(),
        json!(usage_i64(usage, "output_tokens", Some("completion_tokens"))),
    );
    out.insert(
        "cache_creation_input_tokens".to_string(),
        json!(cache_creation),
    );
    out.insert("cache_read_input_tokens".to_string(), json!(cache_read));
    for key in ["cache_creation", "server_tool_use", "service_tier"] {
        if let Some(value) = usage.get(key) {
            out.insert(key.to_string(), value.clone());
        }
    }
    Value::Object(out)
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
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
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
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
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
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(msgs[0].content, "Only text.");
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
                json!({"type": "base64", "data": ""}),
                "Anthropic base64 image source requires non-empty data",
            ),
            (
                json!({"type": "base64", "media_type": 42, "data": "AAAA"}),
                "Anthropic base64 image source requires a string media_type",
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
    fn test_user_tool_result_default_id() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "", "content": "x"},
            ]}],
        });
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(msgs[0].tool_call_id, Some("tool-call".to_string()));
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
                {"type": "tool_use", "id": "tc-1", "name": "search", "input": {"q": "rust"}},
            ]}],
        });
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
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
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(
            msgs[0].reasoning_content,
            Some("Let me think...".to_string())
        );
        assert_eq!(msgs[0].content, "Answer.");
    }

    #[test]
    fn test_preserves_assistant_server_web_search_history() {
        let body = json!({
            "messages": [{"role": "assistant", "content": [
                {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {"query": "codex"}},
                {"type": "web_search_tool_result", "tool_use_id": "srv_1", "content": [
                    {"title": "Codex", "url": "https://example.com", "page_age": "1d"},
                ]},
                {"type": "text", "text": "Summary"},
            ]}],
        });
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert!(msgs[0].content.contains("server_tool_use: web_search"));
        assert!(msgs[0].content.contains("https://example.com"));
        assert!(msgs[0].content.contains("Summary"));
    }

    #[test]
    fn test_preserves_non_text_tool_result_blocks() {
        let body = json!({
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call-1", "content": [
                    {"type": "search_result", "title": "Docs", "url": "https://docs.example", "content": "body"},
                    {"type": "document", "title": "Spec", "source": {"type": "text", "data": "document body"}},
                ]},
            ]}],
        });
        let (msgs, _, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(msgs[0].role, MessageRole::Tool);
        assert!(msgs[0].content.contains("Docs"));
        assert!(msgs[0].content.contains("document body"));
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
        assert_eq!(tools[0].description, "Get weather");
        assert!(tools[0].parameters.get("properties").is_some());
    }

    #[test]
    fn test_false_and_null_tool_controls_are_noops() {
        let body = json!({
            "messages": [],
            "tools": [{
                "name": "get_weather",
                "strict": false,
                "defer_loading": null,
                "eager_input_streaming": false,
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
    fn test_true_tool_controls_fail_loudly() {
        for field in ["strict", "defer_loading", "eager_input_streaming"] {
            let mut tool = json!({
                "name": "get_weather",
                "input_schema": {"type": "object"},
            });
            tool.as_object_mut()
                .unwrap()
                .insert(field.to_string(), json!(true));
            let body = json!({"messages": [], "tools": [tool]});
            assert_eq!(
                anthropic_request_to_internal(&body).unwrap_err(),
                format!("tool.{field}=true is not supported by the Codex OAuth backend")
            );
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
    fn test_tool_choice_web_search() {
        let body = json!({"messages": [], "tool_choice": {"type": "tool", "name": "web_search"}});
        let (_, _, tc, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(tc, Some(json!({"type": "web_search"})));
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
    fn test_unsuffixed_web_search_tool_conversion() {
        let body = json!({
            "messages": [],
            "tools": [{"type": "web_search", "name": "web_search"}],
        });
        let (_, tools, _, _, _, _) = anthropic_request_to_internal(&body).unwrap();
        let tools = tools.unwrap();
        assert_eq!(
            tools[0].parameters.get("__codex_as_api_tool_type"),
            Some(&json!("web_search"))
        );
    }

    #[test]
    fn test_thinking_enabled() {
        let body = json!({"messages": [], "thinking": {"type": "enabled"}});
        let (_, _, _, _, effort, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(effort, Some("high".to_string()));
    }

    #[test]
    fn test_thinking_adaptive() {
        let body = json!({"messages": [], "thinking": {"type": "adaptive"}});
        let (_, _, _, _, effort, _) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(effort, Some("medium".to_string()));
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
            "thinking": {"type": "enabled"},
            "output_config": {"effort": "low"},
        });
        let (_, _, _, _, converted, _) = anthropic_request_to_internal(&enabled).unwrap();
        assert_eq!(converted, Some("low".to_string()));
    }

    #[test]
    fn test_output_config_effort_rejects_disabled_thinking() {
        let body = json!({
            "messages": [],
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "high"},
        });
        assert_eq!(
            anthropic_request_to_internal(&body).unwrap_err(),
            "output_config.effort cannot be used when thinking.type is disabled"
        );
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
                "name": "my schema!",
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
                "strict": false,
            },
        });
        let (_, _, _, _, _, text) = anthropic_request_to_internal(&body).unwrap();
        assert_eq!(
            text,
            Some(json!({"format": {
                "type": "json_schema",
                "name": "my_schema_",
                "schema": {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]},
                "strict": false,
            }}))
        );
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
            finish_reason: finish_reason.to_string(),
            usage,
            reasoning_content,
            raw: None,
            response_id: None,
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
        let resp = make_response(
            "Answer.",
            vec![],
            "stop",
            None,
            Some("My reasoning.".to_string()),
        );
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
        let mut usage = crate::messages::Usage::new(100, 50, None, 20);
        usage.cache_write_tokens = 9;
        let resp = make_response("Hi", vec![], "stop", Some(usage), None);
        let out = internal_response_to_anthropic(&resp, "claude-3", "msg_u");
        assert_eq!(out["usage"]["input_tokens"], 100);
        assert_eq!(out["usage"]["output_tokens"], 50);
        assert_eq!(out["usage"]["cache_read_input_tokens"], 20);
        assert_eq!(out["usage"]["cache_creation_input_tokens"], 9);
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
        let message_delta = parsed
            .iter()
            .find(|(event, _)| event == "message_delta")
            .unwrap();
        assert_eq!(message_delta.1["delta"]["stop_reason"], "tool_use");
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
    fn test_stream_maps_provider_cache_token_details() {
        let events = vec![json!({
            "type": "finish",
            "finish_reason": "stop",
            "usage": {
                "input_tokens": 20,
                "output_tokens": 3,
                "input_tokens_details": {
                    "cached_tokens": 7,
                    "cache_write_tokens": 11
                }
            },
        })];
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
                    "cache_creation_input_tokens": 11,
                    "cache_read_input_tokens": 13,
                    "cache_creation": {"ephemeral_5m_input_tokens": 11, "ephemeral_1h_input_tokens": 0},
                    "server_tool_use": {"web_search_requests": 2}
                },
            }),
        ];
        let chunks = anthropic_stream_adapter(&events, "m", "id");
        let parsed = get_sse_events(&chunks);
        let msg_start = parsed.iter().find(|(e, _)| e == "message_start").unwrap();
        assert_eq!(msg_start.1["message"]["usage"]["input_tokens"], 0);
        assert_eq!(msg_start.1["message"]["usage"]["output_tokens"], 0);
        let msg_delta = parsed.iter().find(|(e, _)| e == "message_delta").unwrap();
        assert_eq!(msg_delta.1["usage"]["input_tokens"], 123);
        assert_eq!(msg_delta.1["usage"]["output_tokens"], 7);
        assert_eq!(msg_delta.1["usage"]["cache_creation_input_tokens"], 11);
        assert_eq!(msg_delta.1["usage"]["cache_read_input_tokens"], 13);
        assert_eq!(
            msg_delta.1["usage"]["server_tool_use"]["web_search_requests"],
            2
        );
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
