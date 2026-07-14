use axum::extract::{OriginalUri, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};
use tokio::task;

use crate::anthropic_adapter::{
    anthropic_request_to_internal, format_anthropic_error, internal_response_to_anthropic,
    AnthropicStreamAdapter,
};
use crate::auth;
use crate::codex_config::CodexConfig;
use crate::messages::{Message, MessageRole, ToolCall, ToolSchema};
use crate::model_capabilities::capability_for_model;
use crate::o200k_tokenizer::count_ordinary;
use crate::provider::{ChatGPTOAuthProvider, CompactControls, GenerationControls, ProviderError};

const DEFAULT_CONTEXT_WINDOW: i64 = 200_000;

#[derive(Clone)]
pub struct AppState {
    pub model: String,
    pub auth_path: Option<String>,
    pub codex_config: CodexConfig,
    pub provider: Arc<ChatGPTOAuthProvider>,
}

#[derive(Debug, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<Value>,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<Value>>,
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub max_completion_tokens: Option<i64>,
    pub stop: Option<StopValue>,
    pub tools: Option<Vec<Value>>,
    pub tool_choice: Option<Value>,
    pub reasoning_effort: Option<String>,
    pub reasoning: Option<Value>,
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_options: Option<Value>,
    pub safety_identifier: Option<String>,
    pub verbosity: Option<String>,
    pub multi_agent: Option<Value>,
    pub programmatic_tool_calling: Option<Value>,
    pub top_p: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub user: Option<String>,
    pub subagent: Option<String>,
    pub memgen_request: Option<bool>,
    pub previous_response_id: Option<String>,
    pub service_tier: Option<String>,
    pub text: Option<Value>,
    pub client_metadata: Option<HashMap<String, String>>,
    pub codex_metadata: Option<bool>,
    pub responses_lite: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopValue {
    Single(String),
    Multiple(Vec<String>),
}

impl StopValue {
    fn to_vec(&self) -> Vec<String> {
        match self {
            StopValue::Single(s) => vec![s.clone()],
            StopValue::Multiple(v) => v.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    pub size: Option<String>,
    pub tools: Option<Vec<Value>>,
    pub programmatic_tool_calling: Option<Value>,
    pub reasoning_effort: Option<String>,
    pub reasoning: Option<Value>,
    pub prompt_cache_options: Option<Value>,
    pub safety_identifier: Option<String>,
    pub verbosity: Option<String>,
    pub multi_agent: Option<Value>,
    pub responses_lite: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct InspectRequest {
    pub prompt: Option<String>,
    pub images: Option<Vec<Value>>,
    pub tools: Option<Vec<Value>>,
    pub programmatic_tool_calling: Option<Value>,
    pub reasoning_effort: Option<String>,
    pub reasoning: Option<Value>,
    pub prompt_cache_options: Option<Value>,
    pub safety_identifier: Option<String>,
    pub verbosity: Option<String>,
    pub multi_agent: Option<Value>,
    pub responses_lite: Option<Value>,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
}

fn error_response(status: StatusCode, message: String) -> impl IntoResponse {
    (
        status,
        Json(ErrorResponse {
            error: ErrorDetail {
                message,
                r#type: "chatgpt_oauth_error".to_string(),
            },
        }),
    )
}

fn anthropic_sse_event(chunk: &str) -> Event {
    let mut lines = chunk.trim_end_matches('\n').splitn(2, '\n');
    let event_type = lines
        .next()
        .and_then(|line| line.strip_prefix("event: "))
        .unwrap_or("message");
    let data = lines
        .next()
        .and_then(|line| line.strip_prefix("data: "))
        .unwrap_or("");
    Event::default().event(event_type).data(data)
}

fn anthropic_stream_error_event(status: u16, message: &str) -> Event {
    Event::default().event("error").data(
        serde_json::to_string(&format_anthropic_error(status, message))
            .unwrap_or_else(|_| "{\"type\":\"error\"}".to_string()),
    )
}

fn map_error_status(e: &ProviderError) -> StatusCode {
    match e {
        ProviderError::Auth(_) => StatusCode::UNAUTHORIZED,
        ProviderError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
        ProviderError::UpstreamHttp { status, .. } => {
            StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
        }
        _ if e.to_string().to_lowercase().contains("context window") => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/images/generations", post(images_generations))
        .route("/v1/inspect", post(inspect))
        .route("/v1/compact", post(compact))
        .route("/v1/messages/compact", post(compact))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/messages", post(anthropic_messages))
        .with_state(state)
}

fn openai_model_id(model: &str) -> String {
    format!("codex-oauth:{}", model)
}

fn is_known_codex_model(model: &str) -> bool {
    static CATALOG: OnceLock<Value> = OnceLock::new();
    CATALOG
        .get_or_init(|| {
            serde_json::from_str(include_str!("model-capabilities.json"))
                .expect("embedded model-capabilities.json must be valid")
        })
        .get("models")
        .and_then(Value::as_object)
        .is_some_and(|models| models.contains_key(model))
}

fn request_messages_to_internal(messages: &[ChatMessage]) -> Result<Vec<Message>, ProviderError> {
    let mut result = Vec::new();
    for (message_index, msg) in messages.iter().enumerate() {
        let role = map_role(&msg.role);
        let (content, structured_content) = normalize_content(&msg.content, role, message_index)?;
        let tool_calls = parse_tool_calls(&msg.tool_calls);
        let m = Message {
            role,
            content,
            tool_calls,
            tool_call_id: msg.tool_call_id.clone(),
            name: msg.name.clone(),
            reasoning_content: None,
            images: vec![],
            structured_content,
        };
        result.push(m);
    }
    Ok(result)
}

fn map_role(role: &str) -> MessageRole {
    match role.to_lowercase().as_str() {
        "system" => MessageRole::System,
        "assistant" => MessageRole::Assistant,
        "tool" => MessageRole::Tool,
        _ => MessageRole::User,
    }
}

fn normalize_content(
    content: &Option<Value>,
    role: MessageRole,
    message_index: usize,
) -> Result<(String, Option<Vec<Value>>), ProviderError> {
    match content {
        None => Ok((String::new(), None)),
        Some(Value::String(s)) => Ok((s.clone(), None)),
        Some(Value::Array(arr)) => {
            let mut text_parts = Vec::new();
            let mut blocks = Vec::new();
            for (block_index, item) in arr.iter().enumerate() {
                let object = item.as_object().ok_or_else(|| {
                    ProviderError::InvalidRequest(format!(
                        "messages item {message_index} content item {block_index} must be an object"
                    ))
                })?;
                let block_type = object.get("type").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::InvalidRequest(format!(
                        "messages item {message_index} content item {block_index} requires a string type"
                    ))
                })?;
                let breakpoint = validate_prompt_cache_breakpoint(
                    object.get("prompt_cache_breakpoint"),
                    message_index,
                    block_index,
                )?;
                if breakpoint.is_some() && role == MessageRole::System {
                    return Err(ProviderError::InvalidRequest(
                        "system message content cannot use prompt_cache_breakpoint".to_string(),
                    ));
                }
                if breakpoint.is_some() && role != MessageRole::User {
                    return Err(ProviderError::InvalidRequest(
                        "prompt_cache_breakpoint is supported only on user text/image content"
                            .to_string(),
                    ));
                }
                match block_type {
                    "text" | "input_text" | "output_text" => {
                        let text = object.get("text").and_then(Value::as_str).ok_or_else(|| {
                            ProviderError::InvalidRequest(format!(
                                "messages item {message_index} content item {block_index} requires string text"
                            ))
                        })?;
                        text_parts.push(text.to_string());
                        if role != MessageRole::System && role != MessageRole::Tool {
                            let mut normalized = serde_json::Map::new();
                            normalized.insert(
                                "type".to_string(),
                                Value::String(
                                    if role == MessageRole::Assistant {
                                        "output_text"
                                    } else {
                                        "input_text"
                                    }
                                    .to_string(),
                                ),
                            );
                            normalized.insert("text".to_string(), Value::String(text.to_string()));
                            if let Some(value) = breakpoint {
                                normalized.insert("prompt_cache_breakpoint".to_string(), value);
                            }
                            blocks.push(Value::Object(normalized));
                        }
                    }
                    "image_url" | "input_image" => {
                        if role != MessageRole::User {
                            return Err(ProviderError::InvalidRequest(
                                "image_url content is supported only on user messages".to_string(),
                            ));
                        }
                        let (url, detail) =
                            parse_image_url_block(object, message_index, block_index)?;
                        let mut normalized = serde_json::Map::new();
                        normalized.insert("type".to_string(), json!("input_image"));
                        normalized.insert("image_url".to_string(), Value::String(url));
                        if let Some(detail) = detail {
                            normalized.insert("detail".to_string(), Value::String(detail));
                        }
                        if let Some(value) = breakpoint {
                            normalized.insert("prompt_cache_breakpoint".to_string(), value);
                        }
                        blocks.push(Value::Object(normalized));
                    }
                    "file" | "input_file" | "audio" | "input_audio" => {
                        return Err(ProviderError::InvalidRequest(format!(
                            "messages content type {block_type} is not supported"
                        )));
                    }
                    _ => {
                        return Err(ProviderError::InvalidRequest(format!(
                            "messages content type {block_type} is not supported"
                        )));
                    }
                }
            }
            Ok((text_parts.join(""), (!blocks.is_empty()).then_some(blocks)))
        }
        Some(_) => Err(ProviderError::InvalidRequest(
            "message content must be a string or an array".to_string(),
        )),
    }
}

fn validate_prompt_cache_breakpoint(
    value: Option<&Value>,
    message_index: usize,
    block_index: usize,
) -> Result<Option<Value>, ProviderError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let valid = value.as_object().is_some_and(|object| {
        object.len() == 1 && object.get("mode").and_then(Value::as_str) == Some("explicit")
    });
    if !valid {
        return Err(ProviderError::InvalidRequest(format!(
            "messages item {message_index} content item {block_index} prompt_cache_breakpoint must be {{\"mode\":\"explicit\"}}"
        )));
    }
    Ok(Some(value.clone()))
}

fn parse_image_url_block(
    object: &serde_json::Map<String, Value>,
    message_index: usize,
    block_index: usize,
) -> Result<(String, Option<String>), ProviderError> {
    let image_url = object.get("image_url").ok_or_else(|| {
        ProviderError::InvalidRequest(format!(
            "messages item {message_index} content item {block_index} requires image_url"
        ))
    })?;
    let (url, detail_value) = match image_url {
        Value::String(url) => (url.clone(), object.get("detail")),
        Value::Object(image) => (
            image
                .get("url")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            image.get("detail").or_else(|| object.get("detail")),
        ),
        _ => (String::new(), None),
    };
    let detail = match detail_value {
        None | Some(Value::Null) => None,
        Some(Value::String(detail)) => Some(detail.clone()),
        Some(_) => {
            return Err(ProviderError::InvalidRequest(
                "image detail must be one of: auto, low, high, original".to_string(),
            ));
        }
    };
    if url.trim().is_empty() {
        return Err(ProviderError::InvalidRequest(format!(
            "messages item {message_index} content item {block_index} requires a non-empty image URL"
        )));
    }
    if detail
        .as_deref()
        .is_some_and(|detail| !matches!(detail, "auto" | "low" | "high" | "original"))
    {
        return Err(ProviderError::InvalidRequest(
            "image detail must be one of: auto, low, high, original".to_string(),
        ));
    }
    Ok((url, detail))
}

fn parse_tool_calls(raw: &Option<Vec<Value>>) -> Vec<ToolCall> {
    let items = match raw {
        Some(v) => v,
        None => return vec![],
    };
    let mut calls = Vec::new();
    for item in items {
        let obj = match item.as_object() {
            Some(o) => o,
            None => continue,
        };
        let call_id = obj
            .get("id")
            .or_else(|| obj.get("call_id"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());

        let func = obj.get("function").and_then(|v| v.as_object());
        let name = func
            .and_then(|f| f.get("name"))
            .or_else(|| obj.get("name"))
            .and_then(|v| v.as_str());
        let args_raw = func
            .and_then(|f| f.get("arguments"))
            .or_else(|| obj.get("arguments"));

        let parsed: HashMap<String, Value> = match args_raw {
            Some(Value::String(s)) => {
                if s.is_empty() {
                    HashMap::new()
                } else {
                    serde_json::from_str(s).unwrap_or_else(|_| {
                        let mut m = HashMap::new();
                        m.insert("input".to_string(), Value::String(s.clone()));
                        m
                    })
                }
            }
            Some(Value::Object(map)) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            _ => HashMap::new(),
        };

        if let Some(n) = name {
            calls.push(ToolCall {
                id: call_id,
                name: n.to_string(),
                arguments: parsed,
            });
        }
    }
    calls
}

fn parse_tools(raw: &Option<Vec<Value>>) -> Result<Option<Vec<ToolSchema>>, ProviderError> {
    let items = match raw {
        Some(v) if !v.is_empty() => v,
        _ => return Ok(None),
    };
    let mut schemas = Vec::new();
    for item in items {
        let obj = match item.as_object() {
            Some(o) => o,
            None => {
                return Err(ProviderError::InvalidRequest(
                    "tools entries must be objects".to_string(),
                ));
            }
        };
        if obj.get("type").and_then(Value::as_str) == Some("programmatic_tool_calling") {
            return Err(ProviderError::InvalidRequest(
                "programmatic_tool_calling is not supported by the Chat Completions facade"
                    .to_string(),
            ));
        }
        let func = obj
            .get("function")
            .and_then(|v| v.as_object())
            .unwrap_or(obj);
        if obj.contains_key("allowed_callers") || func.contains_key("allowed_callers") {
            return Err(ProviderError::InvalidRequest(
                "tool allowed_callers is not supported by the Chat Completions facade".to_string(),
            ));
        }
        if obj.contains_key("output_schema") || func.contains_key("output_schema") {
            return Err(ProviderError::InvalidRequest(
                "tool output_schema is not supported by the Chat Completions facade".to_string(),
            ));
        }
        let name = func.get("name").and_then(|v| v.as_str());
        let desc = func
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let params = func.get("parameters").cloned().unwrap_or(json!({}));
        if let Some(n) = name {
            schemas.push(ToolSchema {
                name: n.to_string(),
                description: desc.to_string(),
                parameters: if params.is_object() {
                    params
                } else {
                    json!({})
                },
            });
        }
    }
    Ok(if schemas.is_empty() {
        None
    } else {
        Some(schemas)
    })
}

fn max_tokens_from_request(req: &ChatCompletionRequest) -> Option<i64> {
    req.max_completion_tokens.or(req.max_tokens)
}

fn context_window_for_model(state: &AppState, model: &str) -> i64 {
    let capability = capability_for_model(model);
    if let Some(configured) = state.codex_config.model_context_window {
        return capability
            .max_context_window
            .map_or(configured, |maximum| configured.min(maximum));
    }
    capability.context_window.unwrap_or(DEFAULT_CONTEXT_WINDOW)
}

fn context_window(state: &AppState) -> i64 {
    context_window_for_model(state, &state.model)
}

fn auto_compact_token_limit_for_model(state: &AppState, model: &str) -> i64 {
    let context_window = context_window_for_model(state, model);
    if let Some(limit) = state.codex_config.model_auto_compact_token_limit {
        return limit.min((context_window * 9) / 10);
    }

    let catalog_context = capability_for_model(model).context_window;
    if state.codex_config.model_context_window.is_some() || catalog_context.is_some() {
        (context_window * 9) / 10
    } else {
        (DEFAULT_CONTEXT_WINDOW * 8) / 10
    }
}

fn auto_compact_token_limit(state: &AppState) -> i64 {
    auto_compact_token_limit_for_model(state, &state.model)
}

fn effective_reasoning_effort(
    state: &AppState,
    requested: Option<&str>,
    model: &str,
) -> Option<String> {
    requested
        .map(|effort| effort.to_string())
        .or_else(|| state.codex_config.model_reasoning_effort.clone())
        .or_else(|| capability_for_model(model).default_reasoning_effort)
}

fn effective_reasoning_effort_with_options(
    state: &AppState,
    requested: Option<&str>,
    reasoning: Option<&Value>,
    model: &str,
) -> Result<Option<String>, ProviderError> {
    if requested.is_some_and(str::is_empty) {
        return Err(ProviderError::InvalidRequest(
            "reasoning_effort must be a non-empty string when provided".to_string(),
        ));
    }
    let mut nested_effort: Option<&str> = None;
    let mut explicit_mode = false;
    if let Some(reasoning) = reasoning {
        let object = reasoning.as_object().ok_or_else(|| {
            ProviderError::InvalidRequest("reasoning must be an object".to_string())
        })?;
        if let Some(value) = object.get("effort") {
            if !value.is_null() {
                nested_effort = Some(
                    value
                        .as_str()
                        .filter(|value| !value.is_empty())
                        .ok_or_else(|| {
                            ProviderError::InvalidRequest(
                                "reasoning.effort must be a non-empty string when provided"
                                    .to_string(),
                            )
                        })?,
                );
            }
        }
        if let Some(value) = object.get("mode").filter(|value| !value.is_null()) {
            match value.as_str() {
                Some("standard") => explicit_mode = true,
                Some("pro") => {
                    return Err(ProviderError::InvalidRequest(
                        "reasoning.mode pro is not supported by the Codex OAuth HTTP transport"
                            .to_string(),
                    ));
                }
                _ => {
                    return Err(ProviderError::InvalidRequest(
                        "reasoning.mode must be one of: standard, pro".to_string(),
                    ));
                }
            }
        }
    }
    if let (Some(top_level), Some(nested)) = (requested, nested_effort) {
        if top_level != nested {
            return Err(ProviderError::InvalidRequest(
                "reasoning_effort conflicts with reasoning.effort".to_string(),
            ));
        }
    }
    Ok(requested
        .or(nested_effort)
        .map(str::to_string)
        .or_else(|| state.codex_config.model_reasoning_effort.clone())
        .or_else(|| explicit_mode.then(|| "medium".to_string()))
        .or_else(|| capability_for_model(model).default_reasoning_effort))
}

fn generation_controls(
    reasoning: Option<Value>,
    safety_identifier: Option<String>,
    prompt_cache_options: Option<Value>,
    verbosity: Option<String>,
    multi_agent: Option<&Value>,
) -> Result<GenerationControls, ProviderError> {
    if multi_agent.is_some_and(|value| !value.is_null()) {
        return Err(ProviderError::InvalidRequest(
            "multi_agent is not supported by the Chat Completions facade".to_string(),
        ));
    }
    if safety_identifier.is_some() {
        return Err(ProviderError::InvalidRequest(
            "safety_identifier is not supported by the Codex OAuth HTTP transport".to_string(),
        ));
    }
    if prompt_cache_options
        .as_ref()
        .is_some_and(|value| !value.is_null())
    {
        return Err(ProviderError::InvalidRequest(
            "prompt_cache_options is not supported by the Codex OAuth HTTP transport".to_string(),
        ));
    }
    Ok(GenerationControls {
        reasoning,
        safety_identifier: None,
        prompt_cache_options: None,
        verbosity,
    })
}

fn optional_string_field(body: &Value, field: &str) -> Result<Option<String>, ProviderError> {
    match body.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(ProviderError::InvalidRequest(format!(
            "{field} must be a string when provided"
        ))),
    }
}

fn validate_anthropic_context_management(body: &Value) -> Result<(), ProviderError> {
    let Some(context_management) = body.get("context_management") else {
        return Ok(());
    };
    if context_management.is_null()
        || context_management
            == &json!({
                "edits": [{
                    "type": "clear_thinking_20251015",
                    "keep": "all",
                }],
            })
    {
        return Ok(());
    }
    Err(ProviderError::InvalidRequest(
        "context_management supports only clear_thinking_20251015 with keep set to all".to_string(),
    ))
}

fn anthropic_service_tier(body: &Value) -> Result<Option<String>, ProviderError> {
    let service_tier = optional_string_field(body, "service_tier")?;
    if service_tier
        .as_deref()
        .is_some_and(|service_tier| service_tier.trim().is_empty())
    {
        return Err(ProviderError::InvalidRequest(
            "service_tier must be a non-empty string when provided".to_string(),
        ));
    }

    let (speed_tier, equivalent_tiers): (&str, &[&str]) = match body.get("speed") {
        None | Some(Value::Null) => return Ok(service_tier),
        Some(Value::String(speed)) if speed == "fast" => ("fast", &["fast", "priority"]),
        Some(Value::String(speed)) if speed == "standard" => ("default", &["default"]),
        Some(_) => {
            return Err(ProviderError::InvalidRequest(
                "speed must be one of: fast, standard".to_string(),
            ));
        }
    };
    if service_tier
        .as_deref()
        .is_some_and(|service_tier| !equivalent_tiers.contains(&service_tier))
    {
        return Err(ProviderError::InvalidRequest(
            "speed conflicts with service_tier".to_string(),
        ));
    }
    Ok(Some(speed_tier.to_string()))
}

fn reject_unsupported_tool_features(body: &Value) -> Result<(), ProviderError> {
    reject_unsupported_generation_tool_features(
        body.get("tools")
            .and_then(Value::as_array)
            .map(Vec::as_slice),
        body.get("programmatic_tool_calling"),
    )
}

fn reject_unsupported_generation_tool_features(
    tools: Option<&[Value]>,
    programmatic_tool_calling: Option<&Value>,
) -> Result<(), ProviderError> {
    if programmatic_tool_calling.is_some_and(|value| !value.is_null()) {
        return Err(ProviderError::InvalidRequest(
            "programmatic_tool_calling is not supported by this compatibility API".to_string(),
        ));
    }
    let Some(tools) = tools else {
        return Ok(());
    };
    for tool in tools {
        let Some(object) = tool.as_object() else {
            continue;
        };
        let function = object
            .get("function")
            .and_then(Value::as_object)
            .unwrap_or(object);
        if object.get("type").and_then(Value::as_str) == Some("programmatic_tool_calling") {
            return Err(ProviderError::InvalidRequest(
                "programmatic_tool_calling tools are not supported by this compatibility API"
                    .to_string(),
            ));
        }
        if object.contains_key("allowed_callers") || function.contains_key("allowed_callers") {
            return Err(ProviderError::InvalidRequest(
                "programmatic tool allowed_callers is not supported".to_string(),
            ));
        }
        if object.contains_key("output_schema") || function.contains_key("output_schema") {
            return Err(ProviderError::InvalidRequest(
                "programmatic tool output_schema is not supported".to_string(),
            ));
        }
    }
    Ok(())
}

const BASE_PROMPT_TOKENS: usize = 8;
const MESSAGE_BOUNDARY_TOKENS: usize = 3;
const IMAGE_TOKEN_ESTIMATE: usize = 8_500;

fn json_token_count<T: Serialize + ?Sized>(value: &T) -> usize {
    let json =
        serde_json::to_string(value).expect("internal messages and tools must serialize as JSON");
    count_ordinary(&json)
}

fn estimate_input_tokens(messages: &[Message], tools: Option<&[ToolSchema]>) -> i64 {
    let mut text_tokens = BASE_PROMPT_TOKENS;
    let mut image_tokens = 0;
    for message in messages {
        let role = match message.role {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::Tool => "tool",
        };
        text_tokens += MESSAGE_BOUNDARY_TOKENS;
        text_tokens += count_ordinary(role);
        text_tokens += count_ordinary(&message.content);
        image_tokens += message.images.len() * IMAGE_TOKEN_ESTIMATE;
        image_tokens += message
            .structured_content
            .as_ref()
            .map(|parts| {
                parts
                    .iter()
                    .filter(|part| part.get("type").and_then(Value::as_str) == Some("input_image"))
                    .count()
                    * IMAGE_TOKEN_ESTIMATE
            })
            .unwrap_or(0);
        if !message.tool_calls.is_empty() {
            text_tokens += json_token_count(&message.tool_calls);
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            text_tokens += count_ordinary(tool_call_id);
        }
        if let Some(name) = &message.name {
            text_tokens += count_ordinary(name);
        }
        if let Some(reasoning) = &message.reasoning_content {
            text_tokens += count_ordinary(reasoning);
        }
    }
    if let Some(tools) = tools.filter(|tools| !tools.is_empty()) {
        text_tokens += json_token_count(tools);
    }
    std::cmp::max(1, (text_tokens + image_tokens) as i64)
}

fn messages_from_compact_body(
    body: &Value,
    force_anthropic: bool,
) -> Result<
    (
        Vec<Message>,
        Option<Vec<ToolSchema>>,
        Option<String>,
        Option<Value>,
    ),
    ProviderError,
> {
    if force_anthropic
        || body.get("system").is_some()
        || body.get("thinking").is_some()
        || body.get("tool_choice").is_some()
        || body.get("stop_sequences").is_some()
    {
        let (messages, tools, _tool_choice, _stop, reasoning_effort, text) =
            anthropic_request_to_internal(body).map_err(ProviderError::InvalidRequest)?;
        return Ok((messages, tools, reasoning_effort, text));
    }

    let raw_items = match body.get("messages") {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(items)) => items.as_slice(),
        Some(_) => {
            return Err(ProviderError::InvalidRequest(
                "messages must be an array".to_string(),
            ));
        }
    };
    let raw_messages: Vec<ChatMessage> = raw_items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            serde_json::from_value(item.clone()).map_err(|error| {
                ProviderError::InvalidRequest(format!("messages item {index} is invalid: {error}"))
            })
        })
        .collect::<Result<_, _>>()?;
    let raw_tools = match body.get("tools") {
        None | Some(Value::Null) => None,
        Some(Value::Array(items)) => Some(items.clone()),
        Some(_) => {
            return Err(ProviderError::InvalidRequest(
                "tools must be an array".to_string(),
            ));
        }
    };
    Ok((
        request_messages_to_internal(&raw_messages)?,
        parse_tools(&raw_tools)?,
        None,
        None,
    ))
}

fn merge_anthropic_compact_text(
    direct_text: Option<&Value>,
    converted_text: Option<Value>,
) -> Result<Option<Value>, ProviderError> {
    let direct_present = direct_text.is_some_and(|value| !value.is_null());
    let mut merged = match direct_text {
        None | Some(Value::Null) => serde_json::Map::new(),
        Some(Value::Object(object)) => object.clone(),
        Some(_) => {
            return Err(ProviderError::InvalidRequest(
                "text must be an object when provided".to_string(),
            ));
        }
    };

    if let Some(converted_text) = converted_text {
        let converted = converted_text.as_object().ok_or_else(|| {
            ProviderError::InvalidRequest(
                "converted Anthropic output format must be an object".to_string(),
            )
        })?;
        for (key, value) in converted {
            if merged
                .get(key)
                .is_some_and(|existing| !existing.is_null() && existing != value)
            {
                return Err(ProviderError::InvalidRequest(format!(
                    "text.{key} conflicts with the Anthropic output format"
                )));
            }
            merged.insert(key.clone(), value.clone());
        }
    }

    Ok((direct_present || !merged.is_empty()).then_some(Value::Object(merged)))
}

async fn health(State(state): State<AppState>) -> Result<Json<Value>, axum::response::Response> {
    let auth_available = auth::is_auth_locally_available(state.auth_path.as_deref());
    let reasoning_effort = effective_reasoning_effort(&state, None, &state.model);
    if reasoning_effort.as_deref() == Some("") {
        let error = ProviderError::Request(
            "reasoning_effort must be a non-empty string when provided".to_string(),
        );
        return Err(error_response(map_error_status(&error), error.to_string()).into_response());
    }
    Ok(Json(json!({
        "status": "ok",
        "auth_available": auth_available,
        "model": state.model,
        "codex_home": state.codex_config.codex_home,
        "codex_config_path": state.codex_config.config_path,
        "reasoning_effort": reasoning_effort,
        "context_window": context_window(&state),
        "auto_compact_token_limit": auto_compact_token_limit(&state),
    })))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, axum::response::Response> {
    reject_unsupported_generation_tool_features(
        request.tools.as_deref(),
        request.programmatic_tool_calling.as_ref(),
    )
    .map_err(|error| error_response(map_error_status(&error), error.to_string()).into_response())?;
    let messages = request_messages_to_internal(&request.messages).map_err(|error| {
        error_response(map_error_status(&error), error.to_string()).into_response()
    })?;
    let tools = parse_tools(&request.tools).map_err(|error| {
        error_response(map_error_status(&error), error.to_string()).into_response()
    })?;
    let stop = request.stop.as_ref().map(|s| s.to_vec());
    let max_tokens = max_tokens_from_request(&request);

    let subagent = request.subagent.clone().or_else(|| {
        headers
            .get("x-openai-subagent")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string())
    });

    let memgen_header = headers
        .get("x-openai-memgen-request")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let memgen_request = request.memgen_request.or_else(|| {
        memgen_header.map(|h| !matches!(h.to_lowercase().as_str(), "false" | "0" | ""))
    });

    let previous_response_id = request.previous_response_id.clone();
    let reasoning_effort = effective_reasoning_effort_with_options(
        &state,
        request.reasoning_effort.as_deref(),
        request.reasoning.as_ref(),
        &request.model,
    )
    .map_err(|error| error_response(map_error_status(&error), error.to_string()).into_response())?;
    let controls = generation_controls(
        request.reasoning.clone(),
        request.safety_identifier.clone(),
        request.prompt_cache_options.clone(),
        request.verbosity.clone(),
        request.multi_agent.as_ref(),
    )
    .map_err(|error| error_response(map_error_status(&error), error.to_string()).into_response())?;

    if request.stream {
        let provider = state.provider.clone();
        let model_id = openai_model_id(&request.model);
        let request_id = format!(
            "chatcmpl-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..24]
        );
        let created = chrono::Utc::now().timestamp();
        let temperature = request.temperature;
        let prompt_cache_key = request.prompt_cache_key.clone();
        let request_model = request.model.clone();
        let tool_choice = request.tool_choice.clone();

        let service_tier = request.service_tier.clone();
        let text = request.text.clone();
        let client_metadata = request.client_metadata.clone();
        let codex_metadata = request.codex_metadata;
        let responses_lite = request.responses_lite.clone();
        let parallel_tool_calls = request.parallel_tool_calls;

        let result = task::spawn_blocking(move || {
            let tools_ref = tools.as_deref();
            let stop_ref: Option<Vec<String>> = stop;
            let stop_slice = stop_ref.as_deref();

            provider.chat_stream_with_controls(
                &messages,
                tools_ref,
                temperature,
                reasoning_effort.as_deref(),
                max_tokens,
                stop_slice,
                prompt_cache_key.as_deref(),
                subagent.as_deref(),
                memgen_request,
                previous_response_id.as_deref(),
                Some(request_model.as_str()),
                tool_choice.as_ref(),
                service_tier.as_deref(),
                text.as_ref(),
                client_metadata.as_ref(),
                codex_metadata,
                responses_lite.as_ref(),
                parallel_tool_calls,
                &controls,
            )
        })
        .await
        .unwrap();

        let events = match result {
            Ok(evts) => evts,
            Err(e) => {
                let status = map_error_status(&e);
                return Err(error_response(status, e.to_string()).into_response());
            }
        };

        let mut sse_events: Vec<Result<Event, std::convert::Infallible>> = Vec::new();

        let preamble = json!({
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant"},
                "finish_reason": null,
            }],
        });
        sse_events.push(Ok(
            Event::default().data(serde_json::to_string(&preamble).unwrap())
        ));

        let mut usage_dict: Option<Value> = None;
        let mut final_response_id: Option<Value> = None;
        let mut tool_call_index: usize = 0;

        for event in &events {
            let typ = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match typ {
                "content" => {
                    let text = event.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    let chunk = json!({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text},
                            "finish_reason": null,
                        }],
                    });
                    sse_events.push(Ok(
                        Event::default().data(serde_json::to_string(&chunk).unwrap())
                    ));
                }
                "reasoning_delta" => {
                    let text = event.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    let chunk = json!({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"reasoning_content": text},
                            "finish_reason": null,
                        }],
                    });
                    sse_events.push(Ok(
                        Event::default().data(serde_json::to_string(&chunk).unwrap())
                    ));
                }
                "reasoning_raw_delta" => {
                    let text = event.get("text").and_then(|v| v.as_str()).unwrap_or("");
                    let chunk = json!({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"reasoning": text},
                            "finish_reason": null,
                        }],
                    });
                    sse_events.push(Ok(
                        Event::default().data(serde_json::to_string(&chunk).unwrap())
                    ));
                }
                "tool_call" => {
                    let tc = json!({
                        "index": tool_call_index,
                        "id": event.get("id"),
                        "type": "function",
                        "function": {
                            "name": event.get("name"),
                            "arguments": serde_json::to_string(
                                event.get("arguments").unwrap_or(&json!({}))
                            ).unwrap_or_else(|_| "{}".to_string()),
                        },
                    });
                    let chunk = json!({
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {"tool_calls": [tc]},
                            "finish_reason": null,
                        }],
                    });
                    sse_events.push(Ok(
                        Event::default().data(serde_json::to_string(&chunk).unwrap())
                    ));
                    tool_call_index += 1;
                }
                "finish" => {
                    final_response_id = event.get("response_id").cloned();
                    if let Some(usage) = event.get("usage") {
                        if usage.is_object() {
                            usage_dict = Some(usage.clone());
                        }
                    }
                    let finish_reason = event
                        .get("finish_reason")
                        .and_then(|v| v.as_str())
                        .unwrap_or("stop");
                    let chunk = json!({
                        "id": request_id,
                        "response_id": event.get("response_id"),
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }],
                    });
                    sse_events.push(Ok(
                        Event::default().data(serde_json::to_string(&chunk).unwrap())
                    ));
                }
                _ => {}
            }
        }

        if let Some(u) = &usage_dict {
            let input_details = u
                .get("input_tokens_details")
                .or_else(|| u.get("prompt_tokens_details"));
            let cached_tokens = input_details
                .and_then(|value| value.get("cached_tokens"))
                .or_else(|| u.get("cached_input_tokens"))
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let cache_write_tokens = input_details
                .and_then(|value| value.get("cache_write_tokens"))
                .or_else(|| u.get("cache_write_input_tokens"))
                .and_then(Value::as_i64)
                .unwrap_or(0);
            let finish_chunk = json!({
                "id": request_id,
                "response_id": final_response_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [],
                "usage": {
                    "prompt_tokens": u.get("prompt_tokens").or_else(|| u.get("input_tokens")).and_then(|v| v.as_i64()).unwrap_or(0),
                    "completion_tokens": u.get("completion_tokens").or_else(|| u.get("output_tokens")).and_then(|v| v.as_i64()).unwrap_or(0),
                    "total_tokens": u.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0),
                    "prompt_tokens_details": {
                        "cached_tokens": cached_tokens,
                        "cache_write_tokens": cache_write_tokens,
                    },
                },
            });
            sse_events.push(Ok(
                Event::default().data(serde_json::to_string(&finish_chunk).unwrap())
            ));
        }

        sse_events.push(Ok(Event::default().data("[DONE]")));

        let sse = Sse::new(stream::iter(sse_events));
        Ok(sse.into_response())
    } else {
        let provider = state.provider.clone();
        let model_id = openai_model_id(&request.model);
        let temperature = request.temperature;
        let prompt_cache_key = request.prompt_cache_key.clone();
        let request_model = request.model.clone();
        let tool_choice = request.tool_choice.clone();
        let service_tier = request.service_tier.clone();
        let text = request.text.clone();
        let client_metadata = request.client_metadata.clone();
        let codex_metadata = request.codex_metadata;
        let responses_lite = request.responses_lite.clone();
        let parallel_tool_calls = request.parallel_tool_calls;

        let result = task::spawn_blocking(move || {
            let tools_ref = tools.as_deref();
            let stop_ref: Option<Vec<String>> = stop;
            let stop_slice = stop_ref.as_deref();

            provider.chat_with_controls(
                &messages,
                tools_ref,
                temperature,
                reasoning_effort.as_deref(),
                max_tokens,
                stop_slice,
                prompt_cache_key.as_deref(),
                subagent.as_deref(),
                memgen_request,
                previous_response_id.as_deref(),
                Some(request_model.as_str()),
                tool_choice.as_ref(),
                service_tier.as_deref(),
                text.as_ref(),
                client_metadata.as_ref(),
                codex_metadata,
                responses_lite.as_ref(),
                parallel_tool_calls,
                &controls,
            )
        })
        .await
        .unwrap();

        let response = match result {
            Ok(resp) => resp,
            Err(e) => {
                let status = map_error_status(&e);
                return Err(error_response(status, e.to_string()).into_response());
            }
        };

        let mut message_obj = json!({
            "role": "assistant",
            "content": response.content,
        });

        if !response.tool_calls.is_empty() {
            let tc_array: Vec<Value> = response
                .tool_calls
                .iter()
                .map(|tc| {
                    json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".to_string()),
                        },
                    })
                })
                .collect();
            message_obj
                .as_object_mut()
                .unwrap()
                .insert("tool_calls".to_string(), Value::Array(tc_array));
        }

        if let Some(rc) = &response.reasoning_content {
            message_obj
                .as_object_mut()
                .unwrap()
                .insert("reasoning_content".to_string(), Value::String(rc.clone()));
        }

        let mut result_obj = json!({
            "id": format!("chatcmpl-{}", &uuid::Uuid::new_v4().simple().to_string()[..24]),
            "object": "chat.completion",
            "created": chrono::Utc::now().timestamp(),
            "model": model_id,
            "response_id": response.response_id,
            "choices": [{
                "index": 0,
                "message": message_obj,
                "finish_reason": response.finish_reason,
            }],
        });

        if let Some(usage) = &response.usage {
            result_obj.as_object_mut().unwrap().insert(
                "usage".to_string(),
                json!({
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                    "prompt_tokens_details": {
                        "cached_tokens": usage.cached_tokens,
                        "cache_write_tokens": usage.cache_write_tokens,
                    },
                }),
            );
        }

        Ok(Json(result_obj).into_response())
    }
}

async fn images_generations(
    State(state): State<AppState>,
    Json(request): Json<ImageGenerationRequest>,
) -> Result<Json<Value>, axum::response::Response> {
    reject_unsupported_generation_tool_features(
        request.tools.as_deref(),
        request.programmatic_tool_calling.as_ref(),
    )
    .map_err(|error| error_response(map_error_status(&error), error.to_string()).into_response())?;
    let provider = state.provider.clone();
    let prompt = request.prompt.clone();
    let size = request.size.clone();
    let request_model = request.model.clone();
    let reasoning_effort = effective_reasoning_effort_with_options(
        &state,
        request.reasoning_effort.as_deref(),
        request.reasoning.as_ref(),
        &request_model,
    )
    .map_err(|error| error_response(map_error_status(&error), error.to_string()).into_response())?;
    let controls = generation_controls(
        request.reasoning.clone(),
        request.safety_identifier.clone(),
        request.prompt_cache_options.clone(),
        request.verbosity.clone(),
        request.multi_agent.as_ref(),
    )
    .map_err(|error| error_response(map_error_status(&error), error.to_string()).into_response())?;
    let responses_lite = request.responses_lite.clone();

    let result = task::spawn_blocking(move || {
        provider.generate_image_with_controls(
            &prompt,
            &[],
            size.as_deref(),
            reasoning_effort.as_deref(),
            Some(request_model.as_str()),
            responses_lite.as_ref(),
            &controls,
        )
    })
    .await
    .unwrap();

    let images = result.map_err(|e| {
        let status = map_error_status(&e);
        error_response(status, e.to_string()).into_response()
    })?;

    let data: Vec<Value> = images
        .iter()
        .filter_map(|img| {
            let result_url = img.get("result").and_then(|v| v.as_str())?;
            Some(json!({
                "url": result_url,
                "revised_prompt": img.get("revised_prompt").and_then(|v| v.as_str()).unwrap_or(&request.prompt),
            }))
        })
        .collect();

    Ok(Json(json!({
        "created": chrono::Utc::now().timestamp(),
        "data": data,
    })))
}

async fn inspect(
    State(state): State<AppState>,
    Json(request): Json<InspectRequest>,
) -> Result<Json<Value>, axum::response::Response> {
    reject_unsupported_generation_tool_features(
        request.tools.as_deref(),
        request.programmatic_tool_calling.as_ref(),
    )
    .map_err(|error| error_response(map_error_status(&error), error.to_string()).into_response())?;
    let provider = state.provider.clone();
    let prompt = request.prompt.clone().unwrap_or_default();
    let images = request.images.clone().unwrap_or_default();
    let reasoning_effort = effective_reasoning_effort_with_options(
        &state,
        request.reasoning_effort.as_deref(),
        request.reasoning.as_ref(),
        &state.model,
    )
    .map_err(|error| error_response(map_error_status(&error), error.to_string()).into_response())?;
    let controls = generation_controls(
        request.reasoning.clone(),
        request.safety_identifier.clone(),
        request.prompt_cache_options.clone(),
        request.verbosity.clone(),
        request.multi_agent.as_ref(),
    )
    .map_err(|error| error_response(map_error_status(&error), error.to_string()).into_response())?;
    let responses_lite = request.responses_lite.clone();

    let result = task::spawn_blocking(move || {
        provider.inspect_image_values_with_controls(
            &prompt,
            &images,
            reasoning_effort.as_deref(),
            None,
            responses_lite.as_ref(),
            &controls,
        )
    })
    .await
    .unwrap();

    let content = result.map_err(|e| {
        let status = map_error_status(&e);
        error_response(status, e.to_string()).into_response()
    })?;

    Ok(Json(json!({"content": content})))
}

async fn compact(
    State(state): State<AppState>,
    OriginalUri(uri): OriginalUri,
    Json(body): Json<Value>,
) -> Result<Json<Value>, axum::response::Response> {
    reject_unsupported_tool_features(&body).map_err(|error| {
        error_response(map_error_status(&error), error.to_string()).into_response()
    })?;
    for field in ["safety_identifier", "include", "prompt_cache_retention"] {
        if body.get(field).is_some_and(|value| !value.is_null()) {
            let error = ProviderError::InvalidRequest(format!(
                "{field} is not supported by the compact facade"
            ));
            return Err(error_response(map_error_status(&error), error.to_string()).into_response());
        }
    }
    let nested_reasoning_effort = match body.get("reasoning") {
        None | Some(Value::Null) => None,
        Some(Value::Object(reasoning)) => {
            if reasoning.contains_key("mode") || reasoning.contains_key("context") {
                let error = ProviderError::InvalidRequest(
                    "reasoning.mode and reasoning.context are not supported by compact".to_string(),
                );
                return Err(
                    error_response(map_error_status(&error), error.to_string()).into_response()
                );
            }
            if reasoning.keys().any(|key| key != "effort") {
                let error = ProviderError::InvalidRequest(
                    "compact reasoning supports only effort".to_string(),
                );
                return Err(
                    error_response(map_error_status(&error), error.to_string()).into_response()
                );
            }
            match reasoning.get("effort") {
                None | Some(Value::Null) => None,
                Some(Value::String(value)) if !value.is_empty() => Some(value.clone()),
                Some(_) => {
                    let error = ProviderError::InvalidRequest(
                        "reasoning.effort must be a non-empty string when provided".to_string(),
                    );
                    return Err(
                        error_response(map_error_status(&error), error.to_string()).into_response()
                    );
                }
            }
        }
        Some(_) => {
            let error = ProviderError::InvalidRequest("reasoning must be an object".to_string());
            return Err(error_response(map_error_status(&error), error.to_string()).into_response());
        }
    };
    if body
        .get("multi_agent")
        .is_some_and(|value| !value.is_null())
    {
        let error = ProviderError::InvalidRequest(
            "multi_agent is not supported by the compact facade".to_string(),
        );
        return Err(error_response(map_error_status(&error), error.to_string()).into_response());
    }
    let provider = state.provider.clone();
    let force_anthropic = uri.path() == "/v1/messages/compact";
    let (messages, tools, converted_reasoning_effort, converted_text) =
        messages_from_compact_body(&body, force_anthropic).map_err(|error| {
            error_response(map_error_status(&error), error.to_string()).into_response()
        })?;
    let top_level_reasoning_effort = match body.get("reasoning_effort") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) if !value.trim().is_empty() => Some(value.clone()),
        Some(_) => {
            let error = ProviderError::InvalidRequest(
                "reasoning_effort must be a non-empty string when provided".to_string(),
            );
            return Err(error_response(map_error_status(&error), error.to_string()).into_response());
        }
    };
    let requested_efforts = [
        top_level_reasoning_effort.as_deref(),
        nested_reasoning_effort.as_deref(),
        converted_reasoning_effort.as_deref(),
    ];
    let first_effort = requested_efforts.iter().flatten().next().copied();
    if first_effort.is_some_and(|first| {
        requested_efforts
            .iter()
            .flatten()
            .any(|candidate| *candidate != first)
    }) {
        let error = ProviderError::InvalidRequest(
            "reasoning effort fields conflict in compact request".to_string(),
        );
        return Err(error_response(map_error_status(&error), error.to_string()).into_response());
    }
    let requested_reasoning_effort = first_effort.map(str::to_string);
    let request_model = if force_anthropic {
        body.get("model")
            .and_then(Value::as_str)
            .filter(|model| is_known_codex_model(model))
            .map(str::to_string)
            .unwrap_or_else(|| state.model.clone())
    } else {
        state.model.clone()
    };
    let reasoning_effort = effective_reasoning_effort(
        &state,
        requested_reasoning_effort.as_deref(),
        &request_model,
    );
    let responses_lite = body.get("responses_lite").cloned();
    let previous_response_id =
        optional_string_field(&body, "previous_response_id").map_err(|error| {
            error_response(map_error_status(&error), error.to_string()).into_response()
        })?;
    if previous_response_id.as_deref().is_some_and(str::is_empty) {
        let error = ProviderError::InvalidRequest(
            "previous_response_id must be a non-empty string".to_string(),
        );
        return Err(error_response(map_error_status(&error), error.to_string()).into_response());
    }
    let compact_service_tier = if force_anthropic {
        anthropic_service_tier(&body)
    } else {
        optional_string_field(&body, "service_tier")
    }
    .map_err(|error| error_response(map_error_status(&error), error.to_string()).into_response())?;
    let compact_text = if force_anthropic {
        merge_anthropic_compact_text(body.get("text"), converted_text).map_err(|error| {
            error_response(map_error_status(&error), error.to_string()).into_response()
        })?
    } else {
        body.get("text").filter(|value| !value.is_null()).cloned()
    };
    let compact_controls = CompactControls {
        previous_response_id,
        prompt_cache_key: optional_string_field(&body, "prompt_cache_key").map_err(|error| {
            error_response(map_error_status(&error), error.to_string()).into_response()
        })?,
        prompt_cache_options: body
            .get("prompt_cache_options")
            .filter(|value| !value.is_null())
            .cloned(),
        service_tier: compact_service_tier,
        text: compact_text,
        verbosity: optional_string_field(&body, "verbosity").map_err(|error| {
            error_response(map_error_status(&error), error.to_string()).into_response()
        })?,
    };

    let result = task::spawn_blocking(move || {
        provider.compact_messages_with_controls(
            &messages,
            tools.as_deref(),
            reasoning_effort.as_deref(),
            Some(request_model.as_str()),
            responses_lite.as_ref(),
            &compact_controls,
        )
    })
    .await
    .unwrap();

    let checkpoint = result.map_err(|e| {
        let status = map_error_status(&e);
        error_response(status, e.to_string()).into_response()
    })?;

    Ok(Json(json!({"checkpoint": checkpoint})))
}

async fn anthropic_count_tokens(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, axum::response::Response> {
    if let Err(error) = validate_anthropic_context_management(&body) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(format_anthropic_error(400, &error.to_string())),
        )
            .into_response());
    }
    if body
        .get("multi_agent")
        .is_some_and(|value| !value.is_null())
    {
        let message = "multi_agent is not supported by the Anthropic count_tokens facade";
        return Err((
            StatusCode::BAD_REQUEST,
            Json(format_anthropic_error(400, message)),
        )
            .into_response());
    }
    if let Err(error) = reject_unsupported_tool_features(&body) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(format_anthropic_error(400, &error.to_string())),
        )
            .into_response());
    }
    let (messages, tools, _tool_choice, _stop, _reasoning_effort, _text) =
        anthropic_request_to_internal(&body).map_err(|message| {
            (
                StatusCode::BAD_REQUEST,
                Json(format_anthropic_error(400, &message)),
            )
                .into_response()
        })?;
    let input_tokens = estimate_input_tokens(&messages, tools.as_deref());
    let request_model = body
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| is_known_codex_model(model))
        .unwrap_or(&state.model);
    Ok(Json(json!({
        "input_tokens": input_tokens,
        "context_window": context_window_for_model(&state, request_model),
        "auto_compact_token_limit": auto_compact_token_limit_for_model(&state, request_model),
    })))
}

async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, axum::response::Response> {
    let request_id = format!("msg_{}", &uuid::Uuid::new_v4().simple().to_string()[..24]);
    validate_anthropic_context_management(&body).map_err(|error| {
        (
            StatusCode::BAD_REQUEST,
            Json(format_anthropic_error(400, &error.to_string())),
        )
            .into_response()
    })?;
    reject_unsupported_tool_features(&body).map_err(|error| {
        error_response(map_error_status(&error), error.to_string()).into_response()
    })?;

    let subagent = body
        .get("subagent")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .or_else(|| {
            headers
                .get("x-anthropic-subagent")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
        });

    let memgen_request = body
        .get("memgen_request")
        .and_then(|v| v.as_bool())
        .or_else(|| {
            headers
                .get("x-anthropic-memgen-request")
                .and_then(|v| v.to_str().ok())
                .map(|h| !matches!(h.to_lowercase().as_str(), "false" | "0" | ""))
        });

    let (messages, tools, tool_choice, stop, converted_reasoning_effort, text) =
        anthropic_request_to_internal(&body).map_err(|message| {
            (
                StatusCode::BAD_REQUEST,
                Json(format_anthropic_error(400, &message)),
            )
                .into_response()
        })?;

    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let default_client_model = "claude-sonnet-4-5".to_string();
    let client_model = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&default_client_model)
        .to_string();
    let request_model = if is_known_codex_model(&client_model) {
        client_model.clone()
    } else {
        state.model.clone()
    };
    let explicit_reasoning_effort =
        optional_string_field(&body, "reasoning_effort").map_err(|error| {
            (
                StatusCode::BAD_REQUEST,
                Json(format_anthropic_error(400, &error.to_string())),
            )
                .into_response()
        })?;
    if explicit_reasoning_effort.is_some()
        && converted_reasoning_effort.is_some()
        && explicit_reasoning_effort != converted_reasoning_effort
    {
        let error = ProviderError::InvalidRequest(
            "reasoning_effort conflicts with Anthropic thinking effort".to_string(),
        );
        return Err((
            StatusCode::BAD_REQUEST,
            Json(format_anthropic_error(400, &error.to_string())),
        )
            .into_response());
    }
    let requested_reasoning_effort = explicit_reasoning_effort.or(converted_reasoning_effort);
    let reasoning = body
        .get("reasoning")
        .filter(|value| !value.is_null())
        .cloned();
    let reasoning_effort = effective_reasoning_effort_with_options(
        &state,
        requested_reasoning_effort.as_deref(),
        reasoning.as_ref(),
        &request_model,
    )
    .map_err(|error| error_response(map_error_status(&error), error.to_string()).into_response())?;
    let service_tier = anthropic_service_tier(&body).map_err(|error| {
        (
            StatusCode::BAD_REQUEST,
            Json(format_anthropic_error(400, &error.to_string())),
        )
            .into_response()
    })?;
    let controls = generation_controls(
        reasoning,
        optional_string_field(&body, "safety_identifier").map_err(|error| {
            error_response(map_error_status(&error), error.to_string()).into_response()
        })?,
        body.get("prompt_cache_options")
            .filter(|value| !value.is_null())
            .cloned(),
        optional_string_field(&body, "verbosity").map_err(|error| {
            error_response(map_error_status(&error), error.to_string()).into_response()
        })?,
        body.get("multi_agent"),
    )
    .map_err(|error| error_response(map_error_status(&error), error.to_string()).into_response())?;

    let max_tokens = body.get("max_tokens").and_then(|v| v.as_i64());

    let tool_choice_val: Option<Value> = tool_choice;
    let text_val: Option<Value> = text;
    let responses_lite_val = body.get("responses_lite").cloned();

    if stream {
        let provider = state.provider.clone();
        let request_model_clone = request_model.clone();

        let prepared = task::spawn_blocking({
            let provider = provider.clone();
            move || {
                let tools_ref = tools.as_deref();
                let stop_ref = stop.as_deref();
                let tc_ref = tool_choice_val.as_ref();
                let text_ref = text_val.as_ref();

                provider.prepare_chat_stream_with_controls(
                    &messages,
                    tools_ref,
                    None,
                    reasoning_effort.as_deref(),
                    max_tokens,
                    stop_ref,
                    None,
                    subagent.as_deref(),
                    memgen_request,
                    None,
                    Some(request_model_clone.as_str()),
                    tc_ref,
                    service_tier.as_deref(),
                    text_ref,
                    None,
                    None,
                    responses_lite_val.as_ref(),
                    None,
                    &controls,
                )
            }
        })
        .await
        .unwrap();

        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                let status_code = match map_error_status(&error) {
                    StatusCode::UNAUTHORIZED => 401u16,
                    StatusCode::INTERNAL_SERVER_ERROR => 500u16,
                    status => status.as_u16(),
                };
                let body = format_anthropic_error(status_code, &error.to_string());
                return Err((
                    StatusCode::from_u16(status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(body),
                )
                    .into_response());
            }
        };

        let (sender, receiver) = tokio::sync::mpsc::channel::<Event>(32);
        let worker_sender = sender.clone();
        let worker = task::spawn_blocking(move || {
            let mut adapter = AnthropicStreamAdapter::new(&client_model, &request_id);
            for chunk in adapter.start() {
                worker_sender
                    .blocking_send(anthropic_sse_event(&chunk))
                    .map_err(|_| {
                        ProviderError::Request("Anthropic SSE client disconnected".to_string())
                    })?;
            }

            provider.stream_prepared_chat(prepared, |event| {
                for chunk in adapter.push(&event) {
                    worker_sender
                        .blocking_send(anthropic_sse_event(&chunk))
                        .map_err(|_| {
                            ProviderError::Request("Anthropic SSE client disconnected".to_string())
                        })?;
                }
                Ok(())
            })
        });

        tokio::spawn(async move {
            let error = match worker.await {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(error),
                Err(error) => Some(ProviderError::Request(format!(
                    "Anthropic SSE worker failed: {error}"
                ))),
            };
            if let Some(error) = error {
                let status_code = match map_error_status(&error) {
                    StatusCode::UNAUTHORIZED => 401u16,
                    StatusCode::INTERNAL_SERVER_ERROR => 500u16,
                    status => status.as_u16(),
                };
                let _ = sender
                    .send(anthropic_stream_error_event(
                        status_code,
                        &error.to_string(),
                    ))
                    .await;
            }
        });

        let event_stream = stream::unfold(receiver, |mut receiver| async move {
            receiver
                .recv()
                .await
                .map(|event| (Ok::<Event, std::convert::Infallible>(event), receiver))
        });
        Ok(Sse::new(event_stream).into_response())
    } else {
        let provider = state.provider.clone();
        let request_model_clone = request_model.clone();

        let result = task::spawn_blocking(move || {
            let tools_ref = tools.as_deref();
            let stop_ref = stop.as_deref();
            let tc_ref = tool_choice_val.as_ref();
            let text_ref = text_val.as_ref();

            provider.chat_with_controls(
                &messages,
                tools_ref,
                None,
                reasoning_effort.as_deref(),
                max_tokens,
                stop_ref,
                None,
                subagent.as_deref(),
                memgen_request,
                None,
                Some(request_model_clone.as_str()),
                tc_ref,
                service_tier.as_deref(),
                text_ref,
                None,
                None,
                responses_lite_val.as_ref(),
                None,
                &controls,
            )
        })
        .await
        .unwrap();

        let response = match result {
            Ok(response) => response,
            Err(e) => {
                let status_code = match map_error_status(&e) {
                    StatusCode::UNAUTHORIZED => 401u16,
                    StatusCode::INTERNAL_SERVER_ERROR => 500u16,
                    s => s.as_u16(),
                };
                let body = format_anthropic_error(status_code, &e.to_string());
                return Err((
                    StatusCode::from_u16(status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(body),
                )
                    .into_response());
            }
        };

        let out = internal_response_to_anthropic(&response, &client_model, &request_id);
        Ok(Json(out).into_response())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::messages::{Message, MessageRole};
    use axum::body::{Body, Bytes};
    use axum::http::header::CONTENT_TYPE;
    use futures::StreamExt;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;
    use tokio::sync::Semaphore;
    use tokio::task::JoinHandle;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum RecordedEndpoint {
        Responses,
        Compact,
    }

    #[derive(Clone, Debug)]
    struct RecordedRequest {
        endpoint: RecordedEndpoint,
        headers: HeaderMap,
        body: Value,
    }

    #[derive(Clone)]
    struct RecordingState {
        requests: Arc<Mutex<Vec<RecordedRequest>>>,
        response_output: Arc<Mutex<Vec<Value>>>,
        emit_completed: Arc<Mutex<bool>>,
        trailing_event: Arc<Mutex<Option<Value>>>,
        completed_response_id: Arc<Mutex<Value>>,
        response_usage: Arc<Mutex<Value>>,
    }

    impl Default for RecordingState {
        fn default() -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                response_output: Arc::new(Mutex::new(vec![json!({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "recorded"}]
                })])),
                emit_completed: Arc::new(Mutex::new(true)),
                trailing_event: Arc::new(Mutex::new(None)),
                completed_response_id: Arc::new(Mutex::new(json!("resp-recorded"))),
                response_usage: Arc::new(Mutex::new(json!({
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3
                }))),
            }
        }
    }

    impl RecordingState {
        fn record(&self, endpoint: RecordedEndpoint, headers: HeaderMap, body: Value) {
            self.requests.lock().unwrap().push(RecordedRequest {
                endpoint,
                headers,
                body,
            });
        }

        fn requests(&self) -> Vec<RecordedRequest> {
            self.requests.lock().unwrap().clone()
        }

        fn set_response_output(&self, output: Vec<Value>) {
            *self.response_output.lock().unwrap() = output;
        }

        fn response_output(&self) -> Vec<Value> {
            self.response_output.lock().unwrap().clone()
        }

        fn set_emit_completed(&self, emit_completed: bool) {
            *self.emit_completed.lock().unwrap() = emit_completed;
        }

        fn emit_completed(&self) -> bool {
            *self.emit_completed.lock().unwrap()
        }

        fn set_trailing_event(&self, event: Value) {
            *self.trailing_event.lock().unwrap() = Some(event);
        }

        fn trailing_event(&self) -> Option<Value> {
            self.trailing_event.lock().unwrap().clone()
        }

        fn set_completed_response_id(&self, id: Value) {
            *self.completed_response_id.lock().unwrap() = id;
        }

        fn completed_response_id(&self) -> Value {
            self.completed_response_id.lock().unwrap().clone()
        }

        fn set_response_usage(&self, usage: Value) {
            *self.response_usage.lock().unwrap() = usage;
        }

        fn response_usage(&self) -> Value {
            self.response_usage.lock().unwrap().clone()
        }
    }

    async fn record_responses_request(
        State(state): State<RecordingState>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> impl IntoResponse {
        state.record(RecordedEndpoint::Responses, headers, body);
        let completed = json!({
            "type": "response.completed",
            "response": {
                "id": state.completed_response_id(),
                "output": [],
                "usage": state.response_usage()
            }
        });
        let mut stream_body = String::new();
        for item in state.response_output() {
            let done = json!({"type": "response.output_item.done", "item": item});
            stream_body.push_str(&format!(
                "data: {}\n\n",
                serde_json::to_string(&done).unwrap()
            ));
        }
        if state.emit_completed() {
            stream_body.push_str(&format!(
                "data: {}\n\n",
                serde_json::to_string(&completed).unwrap()
            ));
        }
        if let Some(event) = state.trailing_event() {
            stream_body.push_str(&format!(
                "data: {}\n\n",
                serde_json::to_string(&event).unwrap()
            ));
        }
        ([(CONTENT_TYPE, "text/event-stream")], stream_body)
    }

    async fn record_compact_request(
        State(state): State<RecordingState>,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        state.record(RecordedEndpoint::Compact, headers, body);
        Json(json!({
            "output": [
                {"type": "additional_tools", "role": "developer", "tools": []},
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "compact-only instructions"}]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "<environment_context>\n<cwd>/stale</cwd>\n</environment_context>"}]
                },
                {"type": "reasoning", "id": "reasoning-1", "summary": [], "encrypted_content": "opaque"},
                {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "prior answer"}]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "compacted"}]
                },
                {"type": "compaction_summary", "encrypted_content": "summary"}
            ]
        }))
    }

    async fn start_recording_upstream() -> (String, RecordingState, JoinHandle<()>) {
        let state = RecordingState::default();
        let app = Router::new()
            .route("/responses", post(record_responses_request))
            .route("/responses/compact", post(record_compact_request))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), state, handle)
    }

    #[derive(Clone)]
    struct GatedSseState {
        early_emitted: Arc<Semaphore>,
        release_completion: Arc<Semaphore>,
    }

    async fn gated_sse_response(State(state): State<GatedSseState>) -> impl IntoResponse {
        let body_stream = stream::unfold((0u8, state), |(stage, state)| async move {
            match stage {
                0 => {
                    state.early_emitted.add_permits(1);
                    let event = json!({
                        "type": "response.output_text.delta",
                        "delta": "early"
                    });
                    Some((
                        Ok::<Bytes, std::convert::Infallible>(Bytes::from(format!(
                            "data: {}\n\n",
                            serde_json::to_string(&event).unwrap()
                        ))),
                        (1, state),
                    ))
                }
                1 => {
                    let permit = state.release_completion.acquire().await.unwrap();
                    permit.forget();
                    let output_item = json!({
                        "type": "response.output_item.done",
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "early"}]
                        }
                    });
                    let completed = json!({
                        "type": "response.completed",
                        "response": {
                            "id": "resp-incremental",
                            "output": [],
                            "usage": {
                                "input_tokens": 2,
                                "output_tokens": 1,
                                "total_tokens": 3
                            }
                        }
                    });
                    Some((
                        Ok(Bytes::from(format!(
                            "data: {}\n\ndata: {}\n\n",
                            serde_json::to_string(&output_item).unwrap(),
                            serde_json::to_string(&completed).unwrap()
                        ))),
                        (2, state),
                    ))
                }
                _ => None,
            }
        });
        (
            [(CONTENT_TYPE, "text/event-stream")],
            Body::from_stream(body_stream),
        )
    }

    async fn start_gated_sse_upstream() -> (String, GatedSseState, JoinHandle<()>) {
        let state = GatedSseState {
            early_emitted: Arc::new(Semaphore::new(0)),
            release_completion: Arc::new(Semaphore::new(0)),
        };
        let app = Router::new()
            .route("/responses", post(gated_sse_response))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), state, handle)
    }

    async fn silent_sse_response() -> impl IntoResponse {
        let body_stream = stream::pending::<Result<Bytes, std::convert::Infallible>>();
        (
            [(CONTENT_TYPE, "text/event-stream")],
            Body::from_stream(body_stream),
        )
    }

    async fn start_silent_sse_upstream() -> (String, JoinHandle<()>) {
        let app = Router::new().route("/responses", post(silent_sse_response));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), handle)
    }

    async fn fixed_status_response(State(status): State<StatusCode>) -> impl IntoResponse {
        (status, "upstream rejected request")
    }

    async fn start_fixed_status_upstream(status: u16) -> (String, JoinHandle<()>) {
        let status = StatusCode::from_u16(status).unwrap();
        let app = Router::new()
            .route("/responses", post(fixed_status_response))
            .with_state(status);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), handle)
    }

    #[derive(Clone, Default)]
    struct AuthRetryState {
        authorizations: Arc<Mutex<Vec<String>>>,
        refresh_calls: Arc<AtomicUsize>,
        fail_refresh: Arc<AtomicBool>,
        always_unauthorized: Arc<AtomicBool>,
    }

    async fn auth_retry_responses(
        State(state): State<AuthRetryState>,
        headers: HeaderMap,
    ) -> axum::response::Response {
        let authorization = headers
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("")
            .to_string();
        let request_number = {
            let mut authorizations = state.authorizations.lock().unwrap();
            authorizations.push(authorization);
            authorizations.len()
        };
        if request_number == 1 || state.always_unauthorized.load(Ordering::SeqCst) {
            return (StatusCode::UNAUTHORIZED, "expired access token").into_response();
        }

        let completed = json!({
            "type": "response.completed",
            "response": {
                "id": "resp-auth-retry",
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            }
        });
        (
            [(CONTENT_TYPE, "text/event-stream")],
            format!("data: {}\n\n", serde_json::to_string(&completed).unwrap()),
        )
            .into_response()
    }

    async fn auth_retry_token(State(state): State<AuthRetryState>) -> axum::response::Response {
        state.refresh_calls.fetch_add(1, Ordering::SeqCst);
        if state.fail_refresh.load(Ordering::SeqCst) {
            return (StatusCode::UNAUTHORIZED, "invalid refresh token").into_response();
        }
        Json(json!({
            "access_token": "header.eyJzdWIiOiJyZWZyZXNoZWQifQ.signature"
        }))
        .into_response()
    }

    async fn start_auth_retry_upstream() -> (String, AuthRetryState, JoinHandle<()>) {
        let state = AuthRetryState::default();
        let app = Router::new()
            .route("/responses", post(auth_retry_responses))
            .route("/token", post(auth_retry_token))
            .with_state(state.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), state, handle)
    }

    struct EnvironmentGuard {
        name: &'static str,
        previous: Option<std::ffi::OsString>,
    }

    impl EnvironmentGuard {
        fn set(name: &'static str, value: &str) -> Self {
            let previous = std::env::var_os(name);
            std::env::set_var(name, value);
            Self { name, previous }
        }
    }

    impl Drop for EnvironmentGuard {
        fn drop(&mut self) {
            if let Some(previous) = &self.previous {
                std::env::set_var(self.name, previous);
            } else {
                std::env::remove_var(self.name);
            }
        }
    }

    fn parse_anthropic_sse_blocks(pending: &mut String, parsed: &mut Vec<(String, Value)>) {
        while let Some(end) = pending.find("\n\n") {
            let block = pending[..end].to_string();
            pending.drain(..end + 2);
            if block.is_empty() {
                continue;
            }
            let mut event_type = None;
            let mut data = None;
            for line in block.lines() {
                if let Some(value) = line.strip_prefix("event: ") {
                    event_type = Some(value.to_string());
                } else if let Some(value) = line.strip_prefix("data: ") {
                    data = Some(serde_json::from_str(value).unwrap());
                }
            }
            parsed.push((event_type.unwrap(), data.unwrap()));
        }
    }

    fn parse_sse_json_events(body: &str) -> Vec<Value> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data:"))
            .map(str::trim)
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str(data).unwrap())
            .collect()
    }

    fn test_codex_config(
        model_reasoning_effort: Option<&str>,
        model_context_window: Option<i64>,
        model_auto_compact_token_limit: Option<i64>,
    ) -> CodexConfig {
        CodexConfig {
            codex_home: "/tmp/codex-as-api-test".to_string(),
            config_path: "/tmp/codex-as-api-test/config.toml".to_string(),
            model: None,
            model_reasoning_effort: model_reasoning_effort.map(|value| value.to_string()),
            model_context_window,
            model_auto_compact_token_limit,
        }
    }

    fn write_test_auth_file() -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "codex-as-api-rust-test-auth-{}.json",
            uuid::Uuid::new_v4()
        ));
        let auth = json!({
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": "header.e30.signature",
                "refresh_token": "refresh-token",
                "id_token": "header.e30.signature",
                "account_id": "account-id"
            }
        });
        std::fs::write(&path, serde_json::to_vec(&auth).unwrap()).unwrap();
        path
    }

    async fn start_api_server(
        upstream_base_url: &str,
        state_model: &str,
        codex_config: CodexConfig,
    ) -> (String, PathBuf, JoinHandle<()>) {
        start_api_server_with_timeout(
            upstream_base_url,
            state_model,
            codex_config,
            std::time::Duration::from_secs(5),
        )
        .await
    }

    async fn start_api_server_with_timeout(
        upstream_base_url: &str,
        state_model: &str,
        codex_config: CodexConfig,
        timeout: std::time::Duration,
    ) -> (String, PathBuf, JoinHandle<()>) {
        let auth_path = write_test_auth_file();
        let auth_path_string = auth_path.to_string_lossy().to_string();
        let provider = ChatGPTOAuthProvider::new(
            state_model.to_string(),
            upstream_base_url.to_string(),
            Some(auth_path_string.clone()),
            Some(timeout),
        );
        let app = create_router(AppState {
            model: state_model.to_string(),
            auth_path: Some(auth_path_string),
            codex_config,
            provider: Arc::new(provider),
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), auth_path, handle)
    }

    fn helper_state(model: &str, codex_config: CodexConfig) -> AppState {
        AppState {
            model: model.to_string(),
            auth_path: None,
            codex_config,
            provider: Arc::new(ChatGPTOAuthProvider::new(
                model.to_string(),
                "http://127.0.0.1:1".to_string(),
                None,
                None,
            )),
        }
    }

    #[test]
    fn estimate_input_tokens_uses_o200k_for_long_ascii() {
        let content = "abcd".repeat(1_000);
        let msg = Message::new(MessageRole::User, content, vec![], None, None).unwrap();
        let estimate = estimate_input_tokens(&[msg], None);
        let expected = BASE_PROMPT_TOKENS + MESSAGE_BOUNDARY_TOKENS + 1 + 1_000;
        assert_eq!(estimate, expected as i64);
        assert_eq!(estimate, 1_012);
    }

    #[test]
    fn estimate_input_tokens_uses_o200k_for_unicode() {
        let content = "hello 안녕 👋";
        let msg = Message::new(MessageRole::User, content.to_string(), vec![], None, None).unwrap();
        assert_eq!(count_ordinary(content), 5);
        assert_eq!(estimate_input_tokens(&[msg], None), 17);
    }

    #[test]
    fn estimate_input_tokens_includes_normalized_tools_once() {
        let msg = Message::new(MessageRole::User, "hello".to_string(), vec![], None, None).unwrap();
        let tools = vec![ToolSchema {
            name: "lookup".to_string(),
            description: "Search docs".to_string(),
            parameters: json!({"type": "object"}),
        }];
        let expected_tokens = BASE_PROMPT_TOKENS
            + MESSAGE_BOUNDARY_TOKENS
            + count_ordinary("user")
            + count_ordinary("hello")
            + json_token_count(&tools);
        assert_eq!(
            estimate_input_tokens(&[msg], Some(&tools)),
            expected_tokens as i64
        );
    }

    #[test]
    fn estimate_input_tokens_includes_tool_metadata_and_reasoning() {
        let tool_call = ToolCall {
            id: "call_123".to_string(),
            name: "lookup".to_string(),
            arguments: HashMap::from([("query".to_string(), json!("문서"))]),
        };
        let mut assistant = Message::new(
            MessageRole::Assistant,
            "checking".to_string(),
            vec![tool_call],
            None,
            None,
        )
        .unwrap();
        assistant.reasoning_content = Some("reason carefully".to_string());
        let tool_result = Message::new(
            MessageRole::Tool,
            "result".to_string(),
            vec![],
            Some("call_123".to_string()),
            Some("lookup".to_string()),
        )
        .unwrap();

        let expected_tokens = BASE_PROMPT_TOKENS
            + MESSAGE_BOUNDARY_TOKENS
            + count_ordinary("assistant")
            + count_ordinary("checking")
            + json_token_count(&assistant.tool_calls)
            + count_ordinary("reason carefully")
            + MESSAGE_BOUNDARY_TOKENS
            + count_ordinary("tool")
            + count_ordinary("result")
            + count_ordinary("call_123")
            + count_ordinary("lookup");
        assert_eq!(
            estimate_input_tokens(&[assistant, tool_result], None),
            expected_tokens as i64
        );
    }

    #[test]
    fn estimate_input_tokens_counts_each_image_once() {
        let mut legacy =
            Message::new(MessageRole::User, String::new(), vec![], None, None).unwrap();
        legacy.images = vec!["data:image/png;base64,AAAA".to_string()];
        let mut structured =
            Message::new(MessageRole::User, String::new(), vec![], None, None).unwrap();
        structured.structured_content = Some(vec![json!({
            "type": "input_image",
            "image_url": "data:image/png;base64,AAAA"
        })]);
        let text_tokens =
            (BASE_PROMPT_TOKENS + MESSAGE_BOUNDARY_TOKENS + count_ordinary("user")) as i64;
        let expected = text_tokens + IMAGE_TOKEN_ESTIMATE as i64;
        assert_eq!(estimate_input_tokens(&[legacy], None), expected);
        assert_eq!(estimate_input_tokens(&[structured], None), expected);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_count_tokens_rejects_new_top_level_features_without_provider_calls() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();
        let base_body = json!({
            "model": "claude-sonnet-4-5",
            "messages": [{"role": "user", "content": "Count this."}],
            "context_management": {
                "edits": [{
                    "type": "clear_thinking_20251015",
                    "keep": "all"
                }]
            },
            "multi_agent": null,
            "programmatic_tool_calling": null
        });

        let response = client
            .post(format!("{api_url}/v1/messages/count_tokens"))
            .json(&base_body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        assert!(body["input_tokens"]
            .as_i64()
            .is_some_and(|tokens| tokens > 0));
        assert!(recording.requests().is_empty());

        for (field, value) in [
            ("multi_agent", json!({"enabled": true})),
            ("programmatic_tool_calling", json!({"enabled": true})),
            ("tools", json!([{"type": "programmatic_tool_calling"}])),
            (
                "tools",
                json!([{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "allowed_callers": ["programmatic"]
                    }
                }]),
            ),
            (
                "tools",
                json!([{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "output_schema": {"type": "object"}
                    }
                }]),
            ),
        ] {
            let mut body = base_body.clone();
            body.as_object_mut()
                .unwrap()
                .insert(field.to_string(), value);
            let invalid_response = client
                .post(format!("{api_url}/v1/messages/count_tokens"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(invalid_response.status(), reqwest::StatusCode::BAD_REQUEST);
            let error: Value = invalid_response.json().await.unwrap();
            assert_eq!(error["type"], json!("error"));
            assert_eq!(error["error"]["type"], json!("invalid_request_error"));
        }
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_count_tokens_uses_normalized_input_and_ignores_control_fields() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();
        let base_body = json!({
            "model": "claude-sonnet-4-5",
            "system": "Be precise.",
            "messages": [{"role": "user", "content": "Count this."}],
            "tools": [{
                "name": "lookup",
                "description": "Search documentation",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}}
                }
            }]
        });
        let (messages, tools, _, _, _, _) = anthropic_request_to_internal(&base_body).unwrap();
        let expected = estimate_input_tokens(&messages, tools.as_deref());

        let base_response = client
            .post(format!("{api_url}/v1/messages/count_tokens"))
            .json(&base_body)
            .send()
            .await
            .unwrap();
        assert_eq!(base_response.status(), reqwest::StatusCode::OK);
        let base_result: Value = base_response.json().await.unwrap();
        assert_eq!(base_result["input_tokens"], json!(expected));

        let mut controlled_body = base_body.clone();
        let controlled = controlled_body.as_object_mut().unwrap();
        controlled.insert("max_tokens".to_string(), json!(128_000));
        controlled.insert("stream".to_string(), json!(true));
        controlled.insert("temperature".to_string(), json!(0.1));
        controlled.insert(
            "metadata".to_string(),
            json!({"padding": "x".repeat(4_000)}),
        );
        controlled.insert("output_config".to_string(), json!({"effort": "max"}));
        let tool = controlled
            .get_mut("tools")
            .and_then(Value::as_array_mut)
            .and_then(|tools| tools.first_mut())
            .and_then(Value::as_object_mut)
            .unwrap();
        tool.insert("strict".to_string(), json!(false));
        tool.insert("defer_loading".to_string(), json!(false));

        let controlled_response = client
            .post(format!("{api_url}/v1/messages/count_tokens"))
            .json(&controlled_body)
            .send()
            .await
            .unwrap();
        assert_eq!(controlled_response.status(), reqwest::StatusCode::OK);
        let controlled_result: Value = controlled_response.json().await.unwrap();
        assert_eq!(
            controlled_result["input_tokens"],
            base_result["input_tokens"]
        );
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_count_tokens_reports_the_effective_requested_model_context() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();

        let requested = client
            .post(format!("{api_url}/v1/messages/count_tokens"))
            .json(&json!({
                "model": "gpt-5.6",
                "messages": [{"role": "user", "content": "Count this."}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(requested.status(), reqwest::StatusCode::OK);
        let requested_body: Value = requested.json().await.unwrap();
        assert_eq!(requested_body["context_window"], json!(372_000));
        assert_eq!(requested_body["auto_compact_token_limit"], json!(334_800));

        let fallback = client
            .post(format!("{api_url}/v1/messages/count_tokens"))
            .json(&json!({
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "Count this."}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(fallback.status(), reqwest::StatusCode::OK);
        let fallback_body: Value = fallback.json().await.unwrap();
        assert_eq!(fallback_body["context_window"], json!(272_000));
        assert_eq!(fallback_body["auto_compact_token_limit"], json!(244_800));
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_route_uses_known_gpt_model_and_output_config_effort_then_falls_back_for_claude(
    ) {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();

        let gpt_response = client
            .post(format!("{api_url}/v1/messages"))
            .json(&json!({
                "model": "gpt-5.6",
                "max_tokens": 64,
                "system": "You are precise.",
                "messages": [{"role": "user", "content": "Use the requested GPT model."}],
                "thinking": {"type": "enabled"},
                "output_config": {"effort": "low"},
                "speed": "fast",
                "service_tier": "priority",
                "context_management": {
                    "edits": [{
                        "type": "clear_thinking_20251015",
                        "keep": "all"
                    }]
                }
            }))
            .send()
            .await
            .unwrap();
        let gpt_status = gpt_response.status();
        let gpt_body: Value = gpt_response.json().await.unwrap();
        assert_eq!(gpt_status, reqwest::StatusCode::OK, "{gpt_body}");
        assert_eq!(gpt_body["model"], json!("gpt-5.6"));

        let claude_response = client
            .post(format!("{api_url}/v1/messages"))
            .json(&json!({
                "model": "claude-opus-4-6",
                "max_tokens": 64,
                "system": "You are precise.",
                "messages": [{"role": "user", "content": "Use the configured fallback."}],
                "speed": "standard"
            }))
            .send()
            .await
            .unwrap();
        let claude_status = claude_response.status();
        let claude_body: Value = claude_response.json().await.unwrap();
        assert_eq!(claude_status, reqwest::StatusCode::OK, "{claude_body}");
        assert_eq!(claude_body["model"], json!("claude-opus-4-6"));

        let requests = recording.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[0].endpoint, RecordedEndpoint::Responses);
        assert_eq!(requests[0].body["model"], json!("gpt-5.6-sol"));
        assert_eq!(requests[0].body["reasoning"]["effort"], json!("low"));
        assert_eq!(requests[0].body["service_tier"], json!("priority"));
        assert_eq!(requests[1].endpoint, RecordedEndpoint::Responses);
        assert_eq!(requests[1].body["model"], json!("gpt-5.5"));
        assert!(requests[1].body.get("service_tier").is_none());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_route_preserves_url_images_and_tool_error_results_on_provider_wire() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/messages"))
            .json(&json!({
                "model": "gpt-5.5",
                "max_tokens": 64,
                "system": "You are precise.",
                "tools": [{
                    "name": "lookup",
                    "strict": false,
                    "defer_loading": null,
                    "eager_input_streaming": false,
                    "input_schema": {"type": "object"}
                }],
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "Inspect this image."},
                        {"type": "image", "source": {
                            "type": "url",
                            "url": "https://example.com/direct.png"
                        }}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "tool_use", "id": "call-error", "name": "lookup", "input": {}}
                    ]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "call-error", "is_error": true, "content": [
                            {"type": "text", "text": "command failed"},
                            {"type": "image", "source": {
                                "type": "url",
                                "url": "https://example.com/result.png"
                            }}
                        ]}
                    ]}
                ]
            }))
            .send()
            .await
            .unwrap();
        let status = response.status();
        let response_body: Value = response.json().await.unwrap();
        assert_eq!(status, reqwest::StatusCode::OK, "{response_body}");

        let requests = recording.requests();
        assert_eq!(requests.len(), 1);
        let input = requests[0].body["input"].as_array().unwrap();
        let image_urls: Vec<&str> = input
            .iter()
            .filter_map(|item| item.get("content").and_then(Value::as_array))
            .flatten()
            .filter_map(|content| {
                if content.get("type").and_then(Value::as_str) == Some("input_image") {
                    content.get("image_url").and_then(Value::as_str)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            image_urls,
            vec![
                "https://example.com/direct.png",
                "https://example.com/result.png"
            ]
        );
        let tool_result = input
            .iter()
            .find(|item| item.get("type").and_then(Value::as_str) == Some("function_call_output"))
            .unwrap();
        assert_eq!(tool_result["call_id"], json!("call-error"));
        assert_eq!(tool_result["output"], json!("[tool_error]\ncommand failed"));

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_routes_reject_unknown_context_management_before_upstream() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();
        let invalid_values = [
            json!({}),
            json!({"edits": []}),
            json!({"edits": [{"type": "clear_thinking_20251015", "keep": "last"}]}),
            json!({"edits": [{"type": "clear_thinking_20251015", "keep": "all"}], "extra": true}),
            json!([{"type": "clear_thinking_20251015", "keep": "all"}]),
        ];

        for context_management in invalid_values {
            let response = client
                .post(format!("{api_url}/v1/messages"))
                .json(&json!({
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "Reject this control."}],
                    "context_management": context_management
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
            let error: Value = response.json().await.unwrap();
            assert_eq!(error["type"], json!("error"));
            assert_eq!(error["error"]["type"], json!("invalid_request_error"));
        }

        let count_response = client
            .post(format!("{api_url}/v1/messages/count_tokens"))
            .json(&json!({
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "Reject this control."}],
                "context_management": {"edits": []}
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(count_response.status(), reqwest::StatusCode::BAD_REQUEST);
        let count_error: Value = count_response.json().await.unwrap();
        assert_eq!(count_error["type"], json!("error"));
        assert_eq!(count_error["error"]["type"], json!("invalid_request_error"));
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_route_rejects_unrepresentable_latest_controls_before_upstream() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();
        let invalid_controls = [
            json!({"output_config": {"effort": ""}}),
            json!({"output_config": {"effort": 42}}),
            json!({"output_config": {"effort": "ultra"}}),
            json!({"output_config": {"effort": "low", "experimental": true}}),
            json!({"output_config": {"task_budget": {"type": "tokens", "total": 20_000}}}),
            json!({
                "thinking": {"type": "disabled"},
                "output_config": {"effort": "high"}
            }),
            json!({
                "reasoning_effort": "high",
                "output_config": {"effort": "low"}
            }),
            json!({
                "tools": [{
                    "name": "lookup",
                    "strict": true,
                    "input_schema": {"type": "object"}
                }]
            }),
            json!({
                "messages": [{"role": "user", "content": [{
                    "type": "image",
                    "source": {"type": "url", "url": ""}
                }]}]
            }),
            json!({
                "messages": [{"role": "user", "content": [{"type": "image"}]}]
            }),
            json!({
                "messages": [{"role": "user", "content": [{
                    "type": "image",
                    "source": "not-an-object"
                }]}]
            }),
            json!({
                "messages": [{"role": "user", "content": [{
                    "type": "image",
                    "source": {"type": "file", "file_id": "file-1"}
                }]}]
            }),
        ];

        for controls in invalid_controls {
            let mut body = json!({
                "model": "gpt-5.6-sol",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Reject this effort."}]
            });
            body.as_object_mut()
                .unwrap()
                .extend(controls.as_object().unwrap().clone());
            let response = client
                .post(format!("{api_url}/v1/messages"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
            let error: Value = response.json().await.unwrap();
            assert_eq!(error["type"], json!("error"));
            assert_eq!(error["error"]["type"], json!("invalid_request_error"));
        }
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_endpoints_reject_conflicting_output_formats_before_upstream() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();

        for endpoint in [
            "/v1/messages",
            "/v1/messages/count_tokens",
            "/v1/messages/compact",
        ] {
            let response = client
                .post(format!("{api_url}{endpoint}"))
                .json(&json!({
                    "model": "gpt-5.6-sol",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "Reject this format."}],
                    "output_format": {"type": "json_object"},
                    "output_config": {
                        "format": {"type": "json_schema", "schema": {"type": "object"}}
                    }
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                reqwest::StatusCode::BAD_REQUEST,
                "endpoint {endpoint} accepted conflicting output formats"
            );
        }
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_route_rejects_invalid_or_conflicting_speed_before_upstream() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();
        let invalid_controls = [
            json!({"speed": "turbo"}),
            json!({"speed": true}),
            json!({"speed": "fast", "service_tier": "default"}),
            json!({"speed": "standard", "service_tier": "fast"}),
            json!({"service_tier": ""}),
        ];

        for controls in invalid_controls {
            let mut body = json!({
                "model": "gpt-5.6-sol",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "Reject this speed."}]
            });
            body.as_object_mut()
                .unwrap()
                .extend(controls.as_object().unwrap().clone());
            let response = client
                .post(format!("{api_url}/v1/messages"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
            let error: Value = response.json().await.unwrap();
            assert_eq!(error["type"], json!("error"));
            assert_eq!(error["error"]["type"], json!("invalid_request_error"));
        }
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[test]
    fn catalog_defaults_drive_reasoning_context_and_compaction_threshold() {
        let state = helper_state("gpt-5.6-sol", test_codex_config(None, None, None));

        assert_eq!(
            effective_reasoning_effort(&state, None, &state.model),
            Some("low".to_string())
        );
        assert_eq!(context_window(&state), 372_000);
        assert_eq!(auto_compact_token_limit(&state), 334_800);
    }

    #[test]
    fn unknown_model_uses_legacy_context_and_compaction_fallback() {
        let state = helper_state("unknown-model", test_codex_config(None, None, None));

        assert_eq!(effective_reasoning_effort(&state, None, &state.model), None);
        assert_eq!(context_window(&state), 200_000);
        assert_eq!(auto_compact_token_limit(&state), 160_000);
    }

    #[test]
    fn unknown_model_explicit_compaction_limit_clamps_to_resolved_fallback_context() {
        let state = helper_state(
            "unknown-model",
            test_codex_config(None, None, Some(190_000)),
        );

        assert_eq!(context_window(&state), 200_000);
        assert_eq!(auto_compact_token_limit(&state), 180_000);
    }

    #[test]
    fn request_effort_precedes_config_and_catalog_without_wire_conversion() {
        let state = helper_state("gpt-5.6-sol", test_codex_config(Some("ultra"), None, None));

        assert_eq!(
            effective_reasoning_effort(&state, Some("CustomEffort"), &state.model),
            Some("CustomEffort".to_string())
        );
        assert_eq!(
            effective_reasoning_effort(&state, None, &state.model),
            Some("ultra".to_string())
        );
    }

    #[test]
    fn configured_context_uses_official_ninety_percent_compaction_default() {
        let state = helper_state(
            "unknown-model",
            test_codex_config(None, Some(100_000), None),
        );

        assert_eq!(context_window(&state), 100_000);
        assert_eq!(auto_compact_token_limit(&state), 90_000);
    }

    #[test]
    fn catalog_maximum_clamps_context_and_explicit_compaction_overrides() {
        let state = helper_state(
            "gpt-5.6-sol",
            test_codex_config(None, Some(500_000), Some(450_000)),
        );

        assert_eq!(context_window(&state), 372_000);
        assert_eq!(auto_compact_token_limit(&state), 334_800);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn axum_chat_route_sends_sol_ultra_as_lite_max_and_labels_requested_model() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(Some("medium"), None, None),
        )
        .await;

        let client = reqwest::Client::new();
        let health_response = client
            .get(format!("{api_url}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(health_response.status(), reqwest::StatusCode::OK);
        let health_body: Value = health_response.json().await.unwrap();
        assert_eq!(health_body["reasoning_effort"], json!("medium"));

        let response = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "You are precise."},
                    {"role": "user", "content": "Reply briefly."}
                ],
                "reasoning_effort": "ultra",
                "responses_lite": "auto",
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let response_body: Value = response.json().await.unwrap();
        assert_eq!(response_body["model"], json!("codex-oauth:gpt-5.6-sol"));

        let requests = recording.requests();
        assert_eq!(requests.len(), 1);
        let recorded = &requests[0];
        assert_eq!(recorded.endpoint, RecordedEndpoint::Responses);
        assert_eq!(
            recorded
                .headers
                .get(crate::model_capabilities::LITE_HEADER_NAME)
                .unwrap()
                .to_str()
                .unwrap(),
            crate::model_capabilities::LITE_HEADER_VALUE
        );
        assert_eq!(recorded.body["model"], json!("gpt-5.6-sol"));
        assert_eq!(recorded.body["reasoning"]["effort"], json!("max"));
        assert_eq!(recorded.body["reasoning"]["context"], json!("all_turns"));
        assert_eq!(
            recorded.body["include"],
            json!(["reasoning.encrypted_content"])
        );
        assert_eq!(recorded.body["text"], json!({"verbosity": "low"}));
        assert_eq!(recorded.body["tool_choice"], json!("auto"));
        assert_eq!(recorded.body["parallel_tool_calls"], json!(false));
        assert!(recorded.body.get("instructions").is_none());
        assert!(recorded.body.get("tools").is_none());
        assert_eq!(
            recorded.body["input"][0],
            json!({"type": "additional_tools", "role": "developer", "tools": []})
        );
        assert_eq!(
            recorded.body["input"][1],
            json!({
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "You are precise."}]
            })
        );

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn axum_chat_route_rejects_non_empty_stop_before_upstream() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.5",
                "messages": [
                    {"role": "system", "content": "You are precise."},
                    {"role": "user", "content": "Reply briefly."}
                ],
                "stop": "END",
                "stream": true
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], json!("chatgpt_oauth_error"));
        assert!(body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("stop is not supported"));
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn axum_chat_route_maps_output_item_done_tools_to_tool_finish_reason() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        recording.set_response_output(vec![
            json!({
                "type": "function_call",
                "id": "item-1",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": "{\"query\":\"one\"}"
            }),
            json!({
                "type": "function_call",
                "id": "item-2",
                "call_id": "call-2",
                "name": "lookup",
                "arguments": "{\"query\":\"two\"}"
            }),
        ]);
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "You are precise."},
                    {"role": "user", "content": "Use the lookup tool."}
                ],
                "responses_lite": "auto",
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let response_body: Value = response.json().await.unwrap();
        assert_eq!(response_body["model"], json!("codex-oauth:gpt-5.6-sol"));
        assert_eq!(
            response_body["choices"][0]["finish_reason"],
            json!("tool_calls")
        );
        assert_eq!(
            response_body["choices"][0]["message"]["tool_calls"],
            json!([
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"query\":\"one\"}"}
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{\"query\":\"two\"}"}
                }
            ])
        );
        assert_eq!(
            recording.requests()[0].body["reasoning"]["effort"],
            json!("low")
        );

        let stream_response = reqwest::Client::new()
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "You are precise."},
                    {"role": "user", "content": "Use both lookup calls."}
                ],
                "stream": true,
                "responses_lite": "auto",
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(stream_response.status(), reqwest::StatusCode::OK);
        let events = parse_sse_json_events(&stream_response.text().await.unwrap());
        let tool_chunks: Vec<&Value> = events
            .iter()
            .filter(|event| {
                event["choices"][0]["delta"]["tool_calls"]
                    .as_array()
                    .is_some()
            })
            .collect();
        assert_eq!(tool_chunks.len(), 2);
        assert_eq!(
            tool_chunks[0]["choices"][0]["delta"]["tool_calls"][0]["index"],
            json!(0)
        );
        assert_eq!(
            tool_chunks[1]["choices"][0]["delta"]["tool_calls"][0]["index"],
            json!(1)
        );
        let usage = events
            .iter()
            .find(|event| {
                event["choices"].as_array().is_some_and(Vec::is_empty) && event["usage"].is_object()
            })
            .unwrap();
        assert_eq!(
            usage["usage"],
            json!({
                "prompt_tokens": 2,
                "completion_tokens": 1,
                "total_tokens": 3,
                "prompt_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                }
            })
        );
        let terminal = events
            .iter()
            .find(|event| event["choices"][0]["finish_reason"] == json!("tool_calls"))
            .unwrap();
        assert_eq!(terminal["choices"][0]["delta"], json!({}));

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn all_responses_sse_paths_reject_eof_before_completed() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        recording.set_emit_completed(false);
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.5",
                "messages": [
                    {"role": "system", "content": "Be precise."},
                    {"role": "user", "content": "Answer."}
                ],
                "responses_lite": "auto",
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        );
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], json!("chatgpt_oauth_error"));
        assert_eq!(recording.requests().len(), 1);

        let chained_after_incomplete = reqwest::Client::new()
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.5",
                "messages": [
                    {"role": "system", "content": "Be precise."},
                    {"role": "user", "content": "Must not continue."}
                ],
                "previous_response_id": "resp-recorded",
                "responses_lite": false,
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            chained_after_incomplete.status(),
            reqwest::StatusCode::BAD_REQUEST
        );
        assert_eq!(recording.requests().len(), 1);

        let inspect_response = reqwest::Client::new()
            .post(format!("{api_url}/v1/inspect"))
            .json(&json!({
                "prompt": "Inspect this.",
                "images": [{"image_url": "data:image/png;base64,AAAA"}],
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            inspect_response.status(),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        );
        let inspect_body: Value = inspect_response.json().await.unwrap();
        assert_eq!(inspect_body["error"]["type"], json!("chatgpt_oauth_error"));
        assert_eq!(recording.requests().len(), 2);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn malformed_completed_response_id_fails_before_success() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        recording.set_completed_response_id(json!(""));
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.5",
                "messages": [
                    {"role": "system", "content": "Be precise."},
                    {"role": "user", "content": "Answer."}
                ],
                "responses_lite": false,
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(
            response.status(),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        );
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], json!("chatgpt_oauth_error"));
        assert_eq!(recording.requests().len(), 1);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn responses_paths_stop_at_completed_and_ignore_trailing_events() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        recording.set_trailing_event(json!({
            "type": "response.failed",
            "response": {"error": {"message": "must not be observed"}}
        }));
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;

        let chat_response = reqwest::Client::new()
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.5",
                "messages": [
                    {"role": "system", "content": "Be precise."},
                    {"role": "user", "content": "Answer."}
                ],
                "responses_lite": false,
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(chat_response.status(), reqwest::StatusCode::OK);
        let chat_body: Value = chat_response.json().await.unwrap();
        assert_eq!(
            chat_body["choices"][0]["message"]["content"],
            json!("recorded")
        );

        let inspect_response = reqwest::Client::new()
            .post(format!("{api_url}/v1/inspect"))
            .json(&json!({
                "prompt": "Inspect this.",
                "images": [{"image_url": "data:image/png;base64,AAAA"}],
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(inspect_response.status(), reqwest::StatusCode::OK);
        assert_eq!(
            inspect_response.json::<Value>().await.unwrap(),
            json!({"content": "recorded"})
        );
        assert_eq!(recording.requests().len(), 2);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_stream_rejects_lite_hosted_web_search_before_sse_headers() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/messages"))
            .json(&json!({
                "model": "claude-sonnet-4-5",
                "system": "Use the available tools.",
                "messages": [{"role": "user", "content": "Search the web."}],
                "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                "stream": true,
                "responses_lite": "auto",
                "max_tokens": 100
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        assert_eq!(
            response
                .headers()
                .get(CONTENT_TYPE)
                .unwrap()
                .to_str()
                .unwrap(),
            "application/json"
        );
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["type"], json!("error"));
        assert_eq!(body["error"]["type"], json!("invalid_request_error"));
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_stream_forwards_translated_delta_before_upstream_completion() {
        let (upstream_url, gated, upstream_handle) = start_gated_sse_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/messages"))
            .json(&json!({
                "model": "claude-sonnet-4-6",
                "system": "Be precise.",
                "messages": [{"role": "user", "content": "Stream the answer."}],
                "max_tokens": 64,
                "stream": true,
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        tokio::time::timeout(
            std::time::Duration::from_secs(2),
            gated.early_emitted.acquire(),
        )
        .await
        .expect("upstream did not emit its early event")
        .unwrap()
        .forget();

        let mut body_stream = response.bytes_stream();
        let mut pending = String::new();
        let mut events = Vec::new();
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            loop {
                let chunk = body_stream.next().await.unwrap().unwrap();
                pending.push_str(std::str::from_utf8(&chunk).unwrap());
                parse_anthropic_sse_blocks(&mut pending, &mut events);
                if events.last().is_some_and(|(_, data)| {
                    data.get("type").and_then(Value::as_str) == Some("content_block_delta")
                }) {
                    break;
                }
            }
        })
        .await
        .expect("translated early event was not delivered before upstream completion");

        assert_eq!(
            events
                .iter()
                .map(|(event_type, _)| event_type.as_str())
                .collect::<Vec<_>>(),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta"
            ]
        );
        assert_eq!(events[0].1["type"], json!("message_start"));
        assert_eq!(events[1].1["type"], json!("content_block_start"));
        assert_eq!(events[1].1["content_block"]["type"], json!("text"));
        assert_eq!(events[2].1["type"], json!("content_block_delta"));
        assert_eq!(events[2].1["delta"]["type"], json!("text_delta"));
        assert_eq!(events[2].1["delta"]["text"], json!("early"));

        gated.release_completion.add_permits(1);
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while let Some(chunk) = body_stream.next().await {
                let chunk = chunk.unwrap();
                pending.push_str(std::str::from_utf8(&chunk).unwrap());
                parse_anthropic_sse_blocks(&mut pending, &mut events);
            }
        })
        .await
        .expect("translated stream did not terminate after upstream completion");

        assert!(pending.is_empty());
        assert_eq!(
            events
                .iter()
                .map(|(event_type, _)| event_type.as_str())
                .collect::<Vec<_>>(),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop"
            ]
        );
        assert_eq!(events[3].1["type"], json!("content_block_stop"));
        assert_eq!(events[4].1["type"], json!("message_delta"));
        assert_eq!(events[4].1["delta"]["stop_reason"], json!("end_turn"));
        assert_eq!(events[5].1["type"], json!("message_stop"));

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_stream_times_out_a_silent_upstream_and_closes_the_worker() {
        let (upstream_url, upstream_handle) = start_silent_sse_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server_with_timeout(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
            std::time::Duration::from_millis(100),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/messages"))
            .json(&json!({
                "model": "claude-sonnet-4-6",
                "system": "Be precise.",
                "messages": [{"role": "user", "content": "Wait for the stream."}],
                "max_tokens": 64,
                "stream": true,
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let body = tokio::time::timeout(std::time::Duration::from_secs(2), response.text())
            .await
            .expect("silent upstream did not release its blocking worker")
            .unwrap();
        let mut pending = body;
        let mut events = Vec::new();
        parse_anthropic_sse_blocks(&mut pending, &mut events);
        assert!(pending.is_empty());
        assert_eq!(events[0].0, "message_start");
        assert_eq!(events.last().unwrap().0, "error");
        assert_eq!(
            events.last().unwrap().1["error"]["type"],
            json!("api_error")
        );

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_stream_preserves_upstream_retry_status_errors() {
        for (status, expected_error_type) in [(429, "rate_limit_error"), (529, "overloaded_error")]
        {
            let (upstream_url, upstream_handle) = start_fixed_status_upstream(status).await;
            let (api_url, auth_path, api_handle) = start_api_server(
                &upstream_url,
                "gpt-5.5",
                test_codex_config(None, None, None),
            )
            .await;
            let response = reqwest::Client::new()
                .post(format!("{api_url}/v1/messages"))
                .json(&json!({
                    "model": "claude-sonnet-4-6",
                    "system": "Be precise.",
                    "messages": [{"role": "user", "content": "Trigger the error."}],
                    "max_tokens": 64,
                    "stream": true,
                    "responses_lite": false
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            let mut pending = response.text().await.unwrap();
            let mut events = Vec::new();
            parse_anthropic_sse_blocks(&mut pending, &mut events);
            assert!(pending.is_empty());
            assert_eq!(events[0].0, "message_start");
            assert_eq!(events.last().unwrap().0, "error");
            assert_eq!(
                events.last().unwrap().1["error"]["type"],
                json!(expected_error_type)
            );

            api_handle.abort();
            upstream_handle.abort();
            std::fs::remove_file(auth_path).unwrap();
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_stream_refreshes_once_and_surfaces_refresh_authentication_failures() {
        let (upstream_url, upstream_state, upstream_handle) = start_auth_retry_upstream().await;
        let _refresh_url = EnvironmentGuard::set(
            crate::auth::REFRESH_URL_OVERRIDE_ENV,
            &format!("{upstream_url}/token"),
        );
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();
        let request = || {
            client.post(format!("{api_url}/v1/messages")).json(&json!({
                "model": "claude-sonnet-4-6",
                "system": "Be precise.",
                "messages": [{"role": "user", "content": "Exercise auth retry."}],
                "max_tokens": 64,
                "stream": true,
                "responses_lite": false
            }))
        };

        let refreshed = request().send().await.unwrap();
        assert_eq!(refreshed.status(), reqwest::StatusCode::OK);
        let mut pending = refreshed.text().await.unwrap();
        let mut events = Vec::new();
        parse_anthropic_sse_blocks(&mut pending, &mut events);
        assert!(pending.is_empty());
        assert_eq!(events.last().unwrap().0, "message_stop");
        assert_eq!(upstream_state.refresh_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            upstream_state.authorizations.lock().unwrap().as_slice(),
            [
                "Bearer header.e30.signature",
                "Bearer header.eyJzdWIiOiJyZWZyZXNoZWQifQ.signature"
            ]
        );

        upstream_state
            .always_unauthorized
            .store(true, Ordering::SeqCst);
        let rejected_after_refresh = request().send().await.unwrap();
        assert_eq!(rejected_after_refresh.status(), reqwest::StatusCode::OK);
        let mut pending = rejected_after_refresh.text().await.unwrap();
        let mut events = Vec::new();
        parse_anthropic_sse_blocks(&mut pending, &mut events);
        assert!(pending.is_empty());
        assert_eq!(events.last().unwrap().0, "error");
        assert_eq!(
            events.last().unwrap().1["error"]["type"],
            json!("authentication_error")
        );
        assert_eq!(upstream_state.refresh_calls.load(Ordering::SeqCst), 2);
        assert_eq!(upstream_state.authorizations.lock().unwrap().len(), 4);

        upstream_state.fail_refresh.store(true, Ordering::SeqCst);
        let failed_refresh = request().send().await.unwrap();
        assert_eq!(failed_refresh.status(), reqwest::StatusCode::OK);
        let mut pending = failed_refresh.text().await.unwrap();
        let mut events = Vec::new();
        parse_anthropic_sse_blocks(&mut pending, &mut events);
        assert!(pending.is_empty());
        assert_eq!(events.last().unwrap().0, "error");
        assert_eq!(
            events.last().unwrap().1["error"]["type"],
            json!("authentication_error")
        );
        assert_eq!(upstream_state.refresh_calls.load(Ordering::SeqCst), 3);
        assert_eq!(upstream_state.authorizations.lock().unwrap().len(), 5);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn invalid_responses_lite_request_returns_bad_request_before_upstream() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "Be precise."},
                    {"role": "user", "content": "Answer."}
                ],
                "responses_lite": 42,
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], json!("chatgpt_oauth_error"));
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_rejects_any_malformed_message_before_upstream() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/compact"))
            .json(&json!({
                "messages": [
                    {"role": "user", "content": "valid"},
                    {"role": 42, "content": "must fail"}
                ],
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], json!("chatgpt_oauth_error"));
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn image_route_supports_classic_override_and_rejects_invalid_lite_mode() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        recording.set_response_output(vec![json!({
            "type": "image_generation_call",
            "id": "image-1",
            "status": "completed",
            "result": "https://example.test/image.png"
        })]);
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/images/generations"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "prompt": "A lighthouse",
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        assert_eq!(
            body["data"][0]["url"],
            json!("https://example.test/image.png")
        );
        let requests = recording.requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0]
            .headers
            .get(crate::model_capabilities::LITE_HEADER_NAME)
            .is_none());
        assert_eq!(
            requests[0].body["tools"],
            json!([{"type": "image_generation", "output_format": "png"}])
        );

        let invalid = reqwest::Client::new()
            .post(format!("{api_url}/v1/images/generations"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "prompt": "A lighthouse",
                "responses_lite": 42
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(invalid.status(), reqwest::StatusCode::BAD_REQUEST);
        assert_eq!(recording.requests().len(), 1);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_wires_tool_schemas_in_lite_and_classic_payloads() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;
        let base_body = json!({
            "messages": [
                {"role": "system", "content": "Use lookup."},
                {"role": "user", "content": "Compact this."}
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup",
                    "parameters": {"type": "object"}
                }
            }]
        });

        for mode in [json!("auto"), json!(false)] {
            let mut body = base_body.clone();
            body.as_object_mut()
                .unwrap()
                .insert("responses_lite".to_string(), mode);
            let response = reqwest::Client::new()
                .post(format!("{api_url}/v1/compact"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
        }

        let requests = recording.requests();
        assert_eq!(requests.len(), 2);
        let expected_tool = json!({
            "type": "function",
            "name": "lookup",
            "description": "Lookup",
            "parameters": {"type": "object"},
            "strict": false
        });
        assert_eq!(
            requests[0].body["input"][0],
            json!({
                "type": "additional_tools",
                "role": "developer",
                "tools": [expected_tool.clone()]
            })
        );
        assert!(requests[0].body.get("tools").is_none());
        assert_eq!(requests[1].body["tools"], json!([expected_tool]));
        assert_eq!(requests[1].body["instructions"], json!("Use lookup."));
        assert_eq!(
            requests[1].body["input"],
            json!([{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Compact this."}]
            }])
        );

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn messages_compact_always_uses_anthropic_content_block_conversion() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/messages/compact"))
            .json(&json!({
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "question"},
                        {"type": "tool_result", "tool_use_id": "call-1", "content": "result"}
                    ]
                }],
                "responses_lite": "auto"
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let requests = recording.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].body["input"],
            json!([
                {"type": "additional_tools", "role": "developer", "tools": []},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "question"}]
                },
                {"type": "function_call_output", "call_id": "call-1", "output": "result"}
            ])
        );
        assert!(requests[0].body.get("instructions").is_none());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn messages_compact_routes_known_gpt_model_and_falls_back_for_claude_model() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();

        for model in ["gpt-5.6", "claude-fable-5"] {
            let response = client
                .post(format!("{api_url}/v1/messages/compact"))
                .json(&json!({
                    "model": model,
                    "system": "Compact precisely.",
                    "messages": [{"role": "user", "content": "History"}],
                    "speed": if model == "gpt-5.6" { "fast" } else { "standard" },
                    "responses_lite": "auto"
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
        }

        let native = client
            .post(format!("{api_url}/v1/compact"))
            .json(&json!({
                "messages": [
                    {"role": "system", "content": "Compact precisely."},
                    {"role": "user", "content": "Native history"}
                ],
                "speed": "fast",
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(native.status(), reqwest::StatusCode::OK);

        let requests = recording.requests();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[0].endpoint, RecordedEndpoint::Compact);
        assert_eq!(requests[0].body["model"], json!("gpt-5.6-sol"));
        assert_eq!(requests[0].body["service_tier"], json!("priority"));
        assert_eq!(requests[1].endpoint, RecordedEndpoint::Compact);
        assert_eq!(requests[1].body["model"], json!("gpt-5.5"));
        assert!(requests[1].body.get("service_tier").is_none());
        assert_eq!(requests[2].endpoint, RecordedEndpoint::Compact);
        assert!(requests[2].body.get("service_tier").is_none());

        let conflicting_effort = client
            .post(format!("{api_url}/v1/messages/compact"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "system": "Compact precisely.",
                "messages": [{"role": "user", "content": "History"}],
                "reasoning_effort": "high",
                "output_config": {"effort": "low"},
                "responses_lite": "auto"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            conflicting_effort.status(),
            reqwest::StatusCode::BAD_REQUEST
        );
        assert_eq!(recording.requests().len(), 3);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn messages_compact_wires_converted_output_format_and_rejects_text_conflicts() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();
        let schema = json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"]
        });

        let response = client
            .post(format!("{api_url}/v1/messages/compact"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "system": "Compact precisely.",
                "messages": [{"role": "user", "content": "History"}],
                "output_config": {
                    "format": {"type": "json_schema", "schema": schema.clone()}
                },
                "text": {"verbosity": "high"},
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let requests = recording.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(
            requests[0].body["text"],
            json!({
                "verbosity": "high",
                "format": {
                    "type": "json_schema",
                    "name": "structured_output",
                    "schema": schema
                }
            })
        );

        let conflicting = client
            .post(format!("{api_url}/v1/messages/compact"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "system": "Compact precisely.",
                "messages": [{"role": "user", "content": "History"}],
                "output_config": {
                    "format": {"type": "json_schema", "schema": {"type": "object"}}
                },
                "text": {"format": {"type": "json_object"}},
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(conflicting.status(), reqwest::StatusCode::BAD_REQUEST);
        assert_eq!(recording.requests().len(), 1);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn health_rejects_empty_effective_reasoning_effort() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(Some(""), None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .get(format!("{api_url}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        );
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], json!("chatgpt_oauth_error"));
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn axum_compact_route_sends_catalog_default_in_lite_json_request() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/compact"))
            .json(&json!({
                "messages": [
                    {"role": "system", "content": "Preserve decisions."},
                    {"role": "user", "content": "Compact this."}
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Lookup",
                        "parameters": {"type": "object"}
                    }
                }],
                "responses_lite": "auto"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let response_body: Value = response.json().await.unwrap();
        let (marker, serialized_output) = response_body["checkpoint"]
            .as_str()
            .unwrap()
            .split_once('\n')
            .unwrap();
        assert_eq!(marker, crate::provider::REMOTE_COMPACTION_MARKER);
        assert_eq!(
            serde_json::from_str::<Value>(serialized_output).unwrap(),
            json!([
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "prior answer"}]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "compacted"}]
                },
                {"type": "compaction_summary", "encrypted_content": "summary"}
            ])
        );

        let requests = recording.requests();
        assert_eq!(requests.len(), 1);
        let recorded = &requests[0];
        assert_eq!(recorded.endpoint, RecordedEndpoint::Compact);
        assert_eq!(
            recorded
                .headers
                .get(crate::model_capabilities::LITE_HEADER_NAME)
                .unwrap()
                .to_str()
                .unwrap(),
            crate::model_capabilities::LITE_HEADER_VALUE
        );
        assert_eq!(recorded.body["model"], json!("gpt-5.6-sol"));
        assert_eq!(
            recorded.body["reasoning"],
            json!({"effort": "low", "context": "all_turns"})
        );
        assert_eq!(recorded.body["text"], json!({"verbosity": "low"}));
        assert_eq!(recorded.body["parallel_tool_calls"], json!(false));
        assert!(recorded.body.get("include").is_none());
        assert!(recorded.body.get("instructions").is_none());
        assert!(recorded.body.get("tools").is_none());
        assert_eq!(
            recorded.body["input"][0],
            json!({
                "type": "additional_tools",
                "role": "developer",
                "tools": [{
                    "type": "function",
                    "name": "lookup",
                    "description": "Lookup",
                    "parameters": {"type": "object"},
                    "strict": false
                }]
            })
        );
        assert_eq!(
            recorded.body["input"][1],
            json!({
                "type": "message",
                "role": "developer",
                "content": [{
                    "type": "input_text",
                    "text": "Preserve decisions."
                }]
            })
        );
        assert_eq!(
            recorded.body["input"][2],
            json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Compact this."}]
            })
        );
        assert_eq!(recorded.body["input"].as_array().unwrap().len(), 3);

        let checkpoint = response_body["checkpoint"].as_str().unwrap();
        let continuation_response = reqwest::Client::new()
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "Continue safely."},
                    {"role": "system", "content": checkpoint},
                    {"role": "user", "content": "Continue now."}
                ],
                "responses_lite": "auto",
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(continuation_response.status(), reqwest::StatusCode::OK);

        let requests = recording.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[1].body["input"],
            json!([
                {"type": "additional_tools", "role": "developer", "tools": []},
                {
                    "type": "message",
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Continue safely."}]
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "prior answer"}]
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "compacted"}]
                },
                {"type": "compaction_summary", "encrypted_content": "summary"},
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Continue now."}]
                }
            ])
        );

        let inspect_response = reqwest::Client::new()
            .post(format!("{api_url}/v1/inspect"))
            .json(&json!({
                "prompt": "Inspect this image.",
                "images": [{"image_url": "data:image/png;base64,AAAA"}],
                "responses_lite": "auto"
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(inspect_response.status(), reqwest::StatusCode::OK);
        assert_eq!(
            inspect_response.json::<Value>().await.unwrap(),
            json!({"content": "recorded"})
        );

        let requests = recording.requests();
        assert_eq!(requests.len(), 3);
        let inspect_request = &requests[2];
        assert_eq!(inspect_request.endpoint, RecordedEndpoint::Responses);
        assert_eq!(inspect_request.body["tool_choice"], json!("auto"));
        assert_eq!(inspect_request.body["reasoning"]["effort"], json!("low"));
        assert_eq!(
            inspect_request.body["reasoning"]["context"],
            json!("all_turns")
        );

        let invalid_compact_response = reqwest::Client::new()
            .post(format!("{api_url}/v1/compact"))
            .json(&json!({
                "messages": [{"role": "user", "content": "History"}],
                "reasoning_effort": 42
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            invalid_compact_response.status(),
            reqwest::StatusCode::BAD_REQUEST
        );
        assert_eq!(recording.requests().len(), 3);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lite_compact_wire_sends_effort_and_all_turns_context() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/compact"))
            .json(&json!({
                "messages": [
                    {"role": "system", "content": "Preserve decisions."},
                    {"role": "user", "content": "Compact this."}
                ],
                "reasoning_effort": "high",
                "responses_lite": true
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let requests = recording.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].endpoint, RecordedEndpoint::Compact);
        assert_eq!(
            requests[0].body["reasoning"],
            json!({"effort": "high", "context": "all_turns"})
        );
        assert_eq!(
            requests[0]
                .headers
                .get(crate::model_capabilities::LITE_HEADER_NAME)
                .unwrap()
                .to_str()
                .unwrap(),
            crate::model_capabilities::LITE_HEADER_VALUE
        );

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn compact_resolves_known_history_without_forwarding_public_only_fields() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        recording.set_completed_response_id(json!("resp-prior"));
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();

        let seed_response = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "Preserve decisions."},
                    {"role": "user", "content": "Prior turn"}
                ],
                "responses_lite": false,
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(seed_response.status(), reqwest::StatusCode::OK);

        let response = client
            .post(format!("{api_url}/v1/compact"))
            .json(&json!({
                "messages": [
                    {"role": "system", "content": "Preserve decisions."},
                    {"role": "user", "content": "Compact this."}
                ],
                "previous_response_id": "resp-prior",
                "safety_identifier": null,
                "include": null,
                "prompt_cache_retention": null,
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let requests = recording.requests();
        assert_eq!(requests.len(), 2);
        assert_eq!(requests[1].endpoint, RecordedEndpoint::Compact);
        assert!(requests[1].body.get("previous_response_id").is_none());
        assert!(requests[1].body.get("prompt_cache_options").is_none());
        let mut expected_input = requests[0].body["input"].as_array().unwrap().clone();
        expected_input.extend(recording.response_output());
        expected_input.push(json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Compact this."}]
        }));
        assert_eq!(requests[1].body["input"], json!(expected_input));

        for rejected_field in [
            json!({"previous_response_id": ""}),
            json!({"previous_response_id": "   "}),
            json!({"previous_response_id": "resp-unknown"}),
            json!({"safety_identifier": "stable-user"}),
            json!({"include": []}),
            json!({"prompt_cache_retention": "24h"}),
            json!({"prompt_cache_options": {"mode": "implicit", "ttl": "30m"}}),
            json!({"prompt_cache_options": {"mode": "explicit", "ttl": "30m"}}),
        ] {
            let mut body = json!({
                "messages": [
                    {"role": "system", "content": "Preserve decisions."},
                    {"role": "user", "content": "Compact this."}
                ],
                "responses_lite": false
            });
            body.as_object_mut()
                .unwrap()
                .extend(rejected_field.as_object().unwrap().clone());
            let invalid_response = client
                .post(format!("{api_url}/v1/compact"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(invalid_response.status(), reqwest::StatusCode::BAD_REQUEST);
        }
        assert_eq!(recording.requests().len(), 2);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn gpt_5_6_controls_structured_content_usage_and_response_id_wire_end_to_end() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        recording.set_response_usage(json!({
            "input_tokens": 11,
            "output_tokens": 2,
            "total_tokens": 13,
            "input_tokens_details": {
                "cached_tokens": 4,
                "cache_write_tokens": 7
            }
        }));
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();
        let request_body = json!({
            "model": "gpt-5.6",
            "messages": [
                {"role": "system", "content": "Be exact."},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": "Inspect this."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,AAAA",
                            "detail": "original"
                        }
                    }
                ]}
            ],
            "reasoning": {"mode": "standard", "context": "current_turn"},
            "service_tier": "fast",
            "verbosity": "high",
            "text": {"verbosity": "high"},
            "responses_lite": false,
            "codex_metadata": false
        });

        let response = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&request_body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["response_id"], json!("resp-recorded"));
        assert_eq!(
            body["usage"]["prompt_tokens_details"],
            json!({"cached_tokens": 4, "cache_write_tokens": 7})
        );

        let requests = recording.requests();
        assert_eq!(requests.len(), 1);
        let outbound = &requests[0].body;
        assert_eq!(outbound["model"], json!("gpt-5.6-sol"));
        assert_eq!(
            outbound["reasoning"],
            json!({"effort": "medium", "context": "current_turn"})
        );
        assert!(outbound.get("safety_identifier").is_none());
        assert!(outbound.get("prompt_cache_options").is_none());
        assert_eq!(outbound["service_tier"], json!("priority"));
        assert_eq!(outbound["text"]["verbosity"], json!("high"));
        assert_eq!(
            outbound["input"][0]["content"],
            json!([
                {
                    "type": "input_text",
                    "text": "Inspect this."
                },
                {
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "original"
                }
            ])
        );

        let stream_response = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "stream": true,
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "Be exact."},
                    {"role": "user", "content": "Hello"}
                ],
                "responses_lite": false,
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(stream_response.status(), reqwest::StatusCode::OK);
        let events = parse_sse_json_events(&stream_response.text().await.unwrap());
        let final_chunk = events.last().unwrap();
        assert_eq!(final_chunk["response_id"], json!("resp-recorded"));
        assert_eq!(
            final_chunk["usage"]["prompt_tokens_details"],
            json!({"cached_tokens": 4, "cache_write_tokens": 7})
        );
        assert!(recording.requests()[1].body.get("service_tier").is_none());

        let before_invalid = recording.requests().len();
        for invalid in [
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "system", "content": "x"}],
                "reasoning": {"mode": "pro"}
            }),
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "system", "content": "x"}],
                "safety_identifier": "stable-user"
            }),
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "system", "content": "x"}],
                "prompt_cache_options": {"mode": "implicit", "ttl": "30m"}
            }),
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "system", "content": "x"}],
                "prompt_cache_options": {"mode": "explicit", "ttl": "30m"}
            }),
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "system", "content": "x"}],
                "service_tier": "flex"
            }),
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "system", "content": "x"}],
                "multi_agent": {"enabled": true}
            }),
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "system", "content": "x"}],
                "tools": [{"type": "programmatic_tool_calling"}]
            }),
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "system", "content": "x"}],
                "tools": [{"type": "function", "function": {
                    "name": "x",
                    "parameters": {},
                    "allowed_callers": ["programmatic"]
                }}]
            }),
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "system", "content": "x"}],
                "tools": [{"type": "function", "function": {
                    "name": "x",
                    "parameters": {},
                    "output_schema": {"type": "object"}
                }}]
            }),
            json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "x"},
                    {"role": "user", "content": [{
                        "type": "text",
                        "text": "x",
                        "prompt_cache_breakpoint": {"mode": "explicit"}
                    }]}
                ]
            }),
            json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "system", "content": [{
                    "type": "text",
                    "text": "x",
                    "prompt_cache_breakpoint": {"mode": "explicit"}
                }]}]
            }),
            json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "x"},
                    {"role": "assistant", "content": [{
                        "type": "text",
                        "text": "prior",
                        "prompt_cache_breakpoint": {"mode": "explicit"}
                    }]}
                ]
            }),
            json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "x"},
                    {"role": "user", "content": [{"type": "input_file", "file_id": "f"}]}
                ]
            }),
        ] {
            let response = client
                .post(format!("{api_url}/v1/chat/completions"))
                .json(&invalid)
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                reqwest::StatusCode::BAD_REQUEST,
                "invalid request unexpectedly reached a non-validation path: {invalid}"
            );
        }
        assert_eq!(recording.requests().len(), before_invalid);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn previous_response_id_replays_exact_history_and_supports_branching() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let prior_output = vec![
            json!({
                "type": "reasoning",
                "id": "reasoning-first",
                "summary": [],
                "encrypted_content": "opaque-first"
            }),
            json!({
                "type": "function_call",
                "call_id": "call-first",
                "name": "lookup",
                "arguments": "{\"query\":\"first\"}"
            }),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "recorded"}]
            }),
        ];
        recording.set_response_output(prior_output.clone());
        recording.set_completed_response_id(json!("resp-first"));
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();

        let first_response = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "Answer exactly."},
                    {"role": "user", "content": "First turn"}
                ],
                "responses_lite": false,
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        let first_status = first_response.status();
        let first_text = first_response.text().await.unwrap();
        assert_eq!(
            first_status,
            reqwest::StatusCode::OK,
            "first response: {first_text}"
        );
        let first_body: Value = serde_json::from_str(&first_text).unwrap();
        let first_response_id = first_body["response_id"].as_str().unwrap().to_string();
        assert_eq!(first_response_id, "resp-first");

        recording.set_completed_response_id(json!("resp-second"));
        let second_response = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "Answer exactly."},
                    {"role": "user", "content": "Second turn"}
                ],
                "previous_response_id": first_response_id,
                "responses_lite": false,
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(second_response.status(), reqwest::StatusCode::OK);
        let second_body: Value = second_response.json().await.unwrap();
        assert_eq!(second_body["response_id"], json!("resp-second"));

        let requests = recording.requests();
        assert_eq!(requests.len(), 2);
        assert!(requests[0].body.get("previous_response_id").is_none());
        assert!(requests[1].body.get("previous_response_id").is_none());
        assert!(requests[1].body.get("thread_id").is_none());
        let mut expected_second_input = requests[0].body["input"].as_array().unwrap().clone();
        expected_second_input.extend(prior_output.clone());
        expected_second_input.push(json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Second turn"}]
        }));
        assert_eq!(requests[1].body["input"], json!(expected_second_input));

        let before_unknown = recording.requests().len();
        let unknown_response = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "Answer exactly."},
                    {"role": "user", "content": "Unknown parent"}
                ],
                "previous_response_id": "resp-missing",
                "responses_lite": false,
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(unknown_response.status(), reqwest::StatusCode::BAD_REQUEST);
        assert_eq!(recording.requests().len(), before_unknown);

        recording.set_completed_response_id(json!("resp-branch"));
        let branch_response = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "Answer exactly."},
                    {"role": "user", "content": "Branch turn"}
                ],
                "previous_response_id": "resp-first",
                "responses_lite": false,
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(branch_response.status(), reqwest::StatusCode::OK);
        let requests = recording.requests();
        assert_eq!(requests.len(), 3);
        let mut expected_branch_input = requests[0].body["input"].as_array().unwrap().clone();
        expected_branch_input.extend(prior_output);
        expected_branch_input.push(json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Branch turn"}]
        }));
        assert_eq!(requests[2].body["input"], json!(expected_branch_input));

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn streaming_completion_commits_a_replayable_response_chain() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let output = vec![json!({
            "type": "reasoning",
            "id": "reasoning-stream",
            "summary": [],
            "encrypted_content": "opaque-stream"
        })];
        recording.set_response_output(output.clone());
        recording.set_completed_response_id(json!("resp-stream-root"));
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();

        let streamed = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "stream": true,
                "messages": [
                    {"role": "system", "content": "Answer exactly."},
                    {"role": "user", "content": "Stream root"}
                ],
                "responses_lite": false,
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(streamed.status(), reqwest::StatusCode::OK);
        let stream_events = parse_sse_json_events(&streamed.text().await.unwrap());
        assert_eq!(
            stream_events.last().unwrap()["response_id"],
            json!("resp-stream-root")
        );

        recording.set_completed_response_id(json!("resp-after-stream"));
        let chained = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "messages": [
                    {"role": "system", "content": "Answer exactly."},
                    {"role": "user", "content": "After stream"}
                ],
                "previous_response_id": "resp-stream-root",
                "responses_lite": false,
                "codex_metadata": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(chained.status(), reqwest::StatusCode::OK);
        let requests = recording.requests();
        let mut expected = requests[0].body["input"].as_array().unwrap().clone();
        expected.extend(output);
        expected.push(json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "After stream"}]
        }));
        assert_eq!(requests[1].body["input"], json!(expected));
        assert!(requests[1].body.get("previous_response_id").is_none());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn non_chat_routes_reject_programmatic_tools_and_multi_agent_before_upstream() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();

        let cases = [
            (
                "/v1/images/generations",
                json!({
                    "model": "gpt-5.6-sol",
                    "prompt": "Draw a circle",
                    "tools": [{"type": "programmatic_tool_calling"}]
                }),
            ),
            (
                "/v1/images/generations",
                json!({
                    "model": "gpt-5.6-sol",
                    "prompt": "Draw a circle",
                    "programmatic_tool_calling": {"enabled": true}
                }),
            ),
            (
                "/v1/images/generations",
                json!({
                    "model": "gpt-5.6-sol",
                    "prompt": "Draw a circle",
                    "multi_agent": {"enabled": true}
                }),
            ),
            (
                "/v1/inspect",
                json!({
                    "prompt": "Inspect",
                    "images": [],
                    "tools": [{"type": "programmatic_tool_calling"}]
                }),
            ),
            (
                "/v1/inspect",
                json!({
                    "prompt": "Inspect",
                    "images": [],
                    "programmatic_tool_calling": {"enabled": true}
                }),
            ),
            (
                "/v1/inspect",
                json!({
                    "prompt": "Inspect",
                    "images": [],
                    "multi_agent": {"enabled": true}
                }),
            ),
            (
                "/v1/compact",
                json!({
                    "messages": [{"role": "user", "content": "History"}],
                    "tools": [{"type": "programmatic_tool_calling"}]
                }),
            ),
            (
                "/v1/compact",
                json!({
                    "messages": [{"role": "user", "content": "History"}],
                    "programmatic_tool_calling": {"enabled": true}
                }),
            ),
            (
                "/v1/compact",
                json!({
                    "messages": [{"role": "user", "content": "History"}],
                    "multi_agent": {"enabled": true}
                }),
            ),
        ];

        for (path, body) in cases {
            let response = client
                .post(format!("{api_url}{path}"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        }
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn chat_endpoint_rejects_non_string_image_detail_before_upstream() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/png;base64,AAAA",
                            "detail": 7
                        }
                    }]
                }]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn chat_route_rejects_top_level_programmatic_tool_calling_before_upstream() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "messages": [{"role": "system", "content": "Answer exactly."}],
                "programmatic_tool_calling": {"enabled": true}
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn inspect_anthropic_and_compact_new_controls_wire_end_to_end() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();

        let inspect_response = client
            .post(format!("{api_url}/v1/inspect"))
            .json(&json!({
                "prompt": "Inspect.",
                "images": [{
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "original"
                }],
                "reasoning": {"mode": "standard", "context": "all_turns"},
                "verbosity": "medium",
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(inspect_response.status(), reqwest::StatusCode::OK);
        let inspect_outbound = &recording.requests()[0].body;
        assert_eq!(
            inspect_outbound["input"][0]["content"][1]["detail"],
            "original"
        );
        assert!(inspect_outbound.get("prompt_cache_options").is_none());
        assert!(inspect_outbound["reasoning"].get("mode").is_none());

        let anthropic_response = client
            .post(format!("{api_url}/v1/messages"))
            .json(&json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 128,
                "system": "Be exact.",
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning": {"mode": "standard", "context": "current_turn"},
                "verbosity": "high",
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(anthropic_response.status(), reqwest::StatusCode::OK);
        let anthropic_outbound = &recording.requests()[1].body;
        assert!(anthropic_outbound["reasoning"].get("mode").is_none());
        assert_eq!(anthropic_outbound["reasoning"]["effort"], "medium");
        assert!(anthropic_outbound.get("safety_identifier").is_none());

        let compact_response = client
            .post(format!("{api_url}/v1/compact"))
            .json(&json!({
                "messages": [{"role": "user", "content": "History"}],
                "reasoning": {"effort": "high"},
                "prompt_cache_key": "compact-cache-key",
                "service_tier": "fast",
                "text": {"format": {"type": "text"}},
                "verbosity": "high",
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(compact_response.status(), reqwest::StatusCode::OK);
        let compact_outbound = &recording.requests()[2];
        assert_eq!(compact_outbound.endpoint, RecordedEndpoint::Compact);
        assert_eq!(
            compact_outbound.body["prompt_cache_key"],
            "compact-cache-key"
        );
        assert!(compact_outbound.body.get("prompt_cache_options").is_none());
        assert!(compact_outbound.body["reasoning"].get("mode").is_none());
        assert!(compact_outbound.body["reasoning"].get("context").is_none());
        assert_eq!(compact_outbound.body["reasoning"]["effort"], "high");
        assert_eq!(compact_outbound.body["service_tier"], "priority");
        assert_eq!(
            compact_outbound.body["text"],
            json!({"format": {"type": "text"}, "verbosity": "high"})
        );

        let anthropic_compact_response = client
            .post(format!("{api_url}/v1/messages/compact"))
            .json(&json!({
                "system": "Compact precisely.",
                "messages": [{"role": "user", "content": "Anthropic history"}],
                "reasoning_effort": "medium",
                "prompt_cache_key": "anthropic-compact-key",
                "service_tier": "default",
                "text": {"format": {"type": "text"}},
                "verbosity": "medium",
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(anthropic_compact_response.status(), reqwest::StatusCode::OK);
        let requests = recording.requests();
        let anthropic_compact_outbound = &requests[3];
        assert_eq!(
            anthropic_compact_outbound.endpoint,
            RecordedEndpoint::Compact
        );
        assert_eq!(
            anthropic_compact_outbound.body["prompt_cache_key"],
            "anthropic-compact-key"
        );
        assert!(anthropic_compact_outbound
            .body
            .get("service_tier")
            .is_none());
        assert_eq!(
            anthropic_compact_outbound.body["text"],
            json!({"format": {"type": "text"}, "verbosity": "medium"})
        );

        let before_invalid = recording.requests().len();
        for invalid in [
            json!({
                "messages": [{"role": "user", "content": "History"}],
                "prompt_cache_key": "",
                "responses_lite": false
            }),
            json!({
                "messages": [{"role": "user", "content": "History"}],
                "text": {"verbosity": "low"},
                "verbosity": "high",
                "responses_lite": false
            }),
            json!({
                "messages": [{"role": "user", "content": "History"}],
                "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
                "responses_lite": false
            }),
            json!({
                "messages": [{"role": "user", "content": "History"}],
                "service_tier": "flex",
                "responses_lite": false
            }),
        ] {
            let response = client
                .post(format!("{api_url}/v1/compact"))
                .json(&invalid)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        }
        assert_eq!(recording.requests().len(), before_invalid);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }
}
