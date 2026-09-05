use axum::extract::rejection::JsonRejection;
use axum::extract::{DefaultBodyLimit, OriginalUri, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::task;

use crate::anthropic_adapter::{
    anthropic_request_to_internal, format_anthropic_error, internal_response_to_anthropic,
    validate_anthropic_cache_controls, AnthropicStreamAdapter,
};
use crate::auth;
use crate::codex_config::CodexConfig;
use crate::messages::{Message, MessageRole, ToolCall, ToolSchema};
use crate::model_catalog::ModelInfo;
use crate::o200k_tokenizer::count_ordinary;
use crate::provider::{
    parse_usage, resolve_reasoning_effort, validate_image_content_values, ChatGPTOAuthProvider,
    CompactControls, GenerationControls, ProviderError, ResolvedModel,
};
use crate::strict_json;

pub const REQUEST_BODY_LIMIT_BYTES: usize = 50 * 1024 * 1024;

#[derive(Clone)]
pub struct AppState {
    pub auth_path: Option<String>,
    pub codex_config: CodexConfig,
    pub provider: Arc<ChatGPTOAuthProvider>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatMessage {
    pub role: String,
    pub content: Option<Value>,
    pub name: Option<String>,
    pub tool_calls: Option<Vec<Value>>,
    pub tool_call_id: Option<String>,
    pub audio: Option<Value>,
    pub function_call: Option<Value>,
    pub refusal: Option<Value>,
}

#[derive(Debug, Clone, Default)]
pub struct PresentJsonValue {
    present: bool,
    value: Option<Value>,
}

impl Serialize for PresentJsonValue {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PresentJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        Ok(Self {
            present: true,
            value: (!value.is_null()).then_some(value),
        })
    }
}

impl PresentJsonValue {
    fn reject_explicit_null(&self, field: &str) -> Result<(), ProviderError> {
        if self.present && self.value.is_none() {
            return Err(ProviderError::InvalidRequest(format!(
                "{field} must not be null"
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub max_completion_tokens: Option<i64>,
    pub stop: Option<Value>,
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
    #[serde(default)]
    pub responses_lite: PresentJsonValue,
    pub parallel_tool_calls: Option<bool>,
    pub n: Option<Value>,
    pub logprobs: Option<Value>,
    pub top_logprobs: Option<Value>,
    pub logit_bias: Option<Value>,
    pub seed: Option<Value>,
    pub response_format: Option<Value>,
    pub stream_options: Option<Value>,
    pub modalities: Option<Value>,
    pub audio: Option<Value>,
    pub store: Option<Value>,
    pub metadata: Option<Value>,
    pub prediction: Option<Value>,
    pub function_call: Option<Value>,
    pub functions: Option<Value>,
    pub prompt_cache_retention: Option<Value>,
    pub web_search_options: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageGenerationRequest {
    pub model: Option<String>,
    pub prompt: String,
    pub reference_images: Option<Vec<Value>>,
    pub size: Option<String>,
    pub tools: Option<Vec<Value>>,
    pub programmatic_tool_calling: Option<Value>,
    pub reasoning_effort: Option<String>,
    pub reasoning: Option<Value>,
    pub prompt_cache_options: Option<Value>,
    pub safety_identifier: Option<String>,
    pub verbosity: Option<String>,
    pub multi_agent: Option<Value>,
    #[serde(default)]
    pub responses_lite: PresentJsonValue,
    pub background: Option<Value>,
    pub moderation: Option<Value>,
    pub n: Option<Value>,
    pub output_compression: Option<Value>,
    pub output_format: Option<Value>,
    pub partial_images: Option<Value>,
    pub quality: Option<Value>,
    pub response_format: Option<Value>,
    pub stream: Option<Value>,
    pub style: Option<Value>,
    pub user: Option<Value>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct InspectRequest {
    pub model: Option<String>,
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
    #[serde(default)]
    pub responses_lite: PresentJsonValue,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorDetail,
}

#[derive(Debug, Serialize)]
struct ErrorDetail {
    message: String,
    r#type: String,
    code: String,
}

fn error_response_with_type(
    status: StatusCode,
    message: String,
    error_type: &str,
) -> impl IntoResponse {
    (
        status,
        Json(ErrorResponse {
            error: ErrorDetail {
                message,
                r#type: error_type.to_string(),
                code: error_type.to_string(),
            },
        }),
    )
}

fn provider_error_response(error: &ProviderError) -> axum::response::Response {
    let status = map_error_status(error);
    error_response_with_type(
        status,
        public_error_message(error),
        provider_error_type(error),
    )
    .into_response()
}

fn public_error_message(error: &ProviderError) -> String {
    match error {
        ProviderError::Auth(_) => {
            "ChatGPT OAuth credentials are unavailable; rerun codex login".to_string()
        }
        ProviderError::CatalogUnavailable(_) => {
            "authenticated model catalog is unavailable".to_string()
        }
        ProviderError::UpstreamProtocol(_) => "upstream protocol validation failed".to_string(),
        ProviderError::UpstreamHttp { .. } | ProviderError::UpstreamTransport(_) => {
            "upstream request failed".to_string()
        }
        ProviderError::Request(_) => "internal server error".to_string(),
        _ => error.to_string(),
    }
}

fn provider_error_type(error: &ProviderError) -> &'static str {
    match error {
        ProviderError::Auth(_) => "authentication_error",
        ProviderError::InvalidRequest(_) => "invalid_request_error",
        ProviderError::ModelNotFound(_) => "model_not_found",
        ProviderError::CatalogUnavailable(_) => "catalog_unavailable",
        ProviderError::UpstreamProtocol(_) => "upstream_protocol_error",
        ProviderError::UpstreamHttp { .. } | ProviderError::UpstreamTransport(_) => {
            "upstream_error"
        }
        _ => "server_error",
    }
}

fn health_error_message(error: &ProviderError) -> &'static str {
    match error {
        ProviderError::Auth(_) => "ChatGPT OAuth credentials are unavailable",
        ProviderError::CatalogUnavailable(_) => "authenticated model catalog is unavailable",
        ProviderError::ModelNotFound(_) => {
            "configured model is unavailable in the authenticated catalog"
        }
        ProviderError::UpstreamHttp { .. } | ProviderError::UpstreamTransport(_) => {
            "upstream request failed"
        }
        ProviderError::UpstreamProtocol(_) => "upstream protocol validation failed",
        ProviderError::InvalidRequest(_) => "health configuration is invalid",
        ProviderError::Request(_) => "health preflight failed",
    }
}

fn anthropic_provider_error_response(error: &ProviderError) -> axum::response::Response {
    let status = map_error_status(error);
    (
        status,
        Json(format_anthropic_error(
            status.as_u16(),
            &public_error_message(error),
        )),
    )
        .into_response()
}

fn join_error_response(_error: tokio::task::JoinError) -> axum::response::Response {
    error_response_with_type(
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal server error".to_string(),
        "server_error",
    )
    .into_response()
}

fn openai_usage_value(usage: &crate::messages::Usage) -> Value {
    let mut details = serde_json::Map::new();
    if let Some(cached_tokens) = usage.cached_tokens {
        details.insert("cached_tokens".to_string(), json!(cached_tokens));
    }
    if let Some(cache_write_tokens) = usage.cache_write_tokens {
        details.insert("cache_write_tokens".to_string(), json!(cache_write_tokens));
    }
    let mut value = json!({
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    });
    if !details.is_empty() {
        value
            .as_object_mut()
            .expect("usage literal must be an object")
            .insert("prompt_tokens_details".to_string(), Value::Object(details));
    }
    value
}

#[allow(clippy::result_large_err)]
fn openai_json<T: Serialize>(
    payload: Result<Json<T>, JsonRejection>,
) -> Result<T, axum::response::Response> {
    let value = payload.map(|Json(value)| value).map_err(|error| {
        let status = error.status();
        if matches!(
            status,
            StatusCode::PAYLOAD_TOO_LARGE | StatusCode::UNSUPPORTED_MEDIA_TYPE
        ) {
            let message = if status == StatusCode::PAYLOAD_TOO_LARGE {
                "request body exceeds 50 MiB"
            } else {
                "request Content-Type must be application/json or application/*+json"
            };
            return (
                status,
                Json(json!({
                    "error": {
                        "message": message,
                        "type": "invalid_request_error",
                        "code": "invalid_request_error"
                    }
                })),
            )
                .into_response();
        }
        provider_error_response(&ProviderError::InvalidRequest(format!(
            "invalid JSON request: {error}"
        )))
    })?;
    let parsed = serde_json::to_value(&value).map_err(|_| {
        provider_error_response(&ProviderError::InvalidRequest(
            "invalid JSON request".to_string(),
        ))
    })?;
    strict_json::validate_value(&parsed).map_err(|_| {
        provider_error_response(&ProviderError::InvalidRequest(
            "invalid JSON request".to_string(),
        ))
    })?;
    Ok(value)
}

#[allow(clippy::result_large_err)]
fn parse_openai_request<T: serde::de::DeserializeOwned>(
    value: Value,
) -> Result<T, axum::response::Response> {
    serde_json::from_value(value).map_err(|error| {
        provider_error_response(&ProviderError::InvalidRequest(format!(
            "invalid JSON request: {error}"
        )))
    })
}

fn reject_explicit_null_fields(body: &Value, fields: &[&str]) -> Result<(), ProviderError> {
    let object = body.as_object().ok_or_else(|| {
        ProviderError::InvalidRequest("request body must be a JSON object".to_string())
    })?;
    if let Some(field) = fields
        .iter()
        .find(|field| object.contains_key(**field) && object.get(**field) == Some(&Value::Null))
    {
        return Err(ProviderError::InvalidRequest(format!(
            "{field} must not be null"
        )));
    }
    Ok(())
}

fn validate_openai_chat_message_fields(body: &Value) -> Result<(), ProviderError> {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return Ok(());
    };
    for (index, message) in messages.iter().enumerate() {
        let Some(object) = message.as_object() else {
            continue;
        };
        let Some(role) = object.get("role").and_then(Value::as_str) else {
            continue;
        };
        let allowed: &[&str] = match role {
            "assistant" => &[
                "role",
                "content",
                "tool_calls",
                "audio",
                "function_call",
                "refusal",
            ],
            "tool" => &["role", "content", "tool_call_id"],
            "system" | "developer" | "user" => &["role", "content"],
            _ => continue,
        };
        reject_unknown_object_fields(object, &format!("messages[{index}]"), allowed)?;
        if role == "assistant" {
            if !object.contains_key("content") && !object.contains_key("tool_calls") {
                return Err(ProviderError::InvalidRequest(format!(
                    "messages[{index}] requires content or tool_calls"
                )));
            }
            if object.get("tool_calls") == Some(&Value::Null) {
                return Err(ProviderError::InvalidRequest(format!(
                    "messages[{index}].tool_calls must not be null"
                )));
            }
            for field in ["audio", "function_call", "refusal"] {
                if object.get(field).is_some_and(|value| !value.is_null()) {
                    return Err(ProviderError::InvalidRequest(format!(
                        "messages[{index}].{field} is not supported by the Codex OAuth HTTP transport"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn normalize_assistant_optional_content(body: &mut Value) {
    let Some(messages) = body.get_mut("messages").and_then(Value::as_array_mut) else {
        return;
    };
    for message in messages {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        if object.get("role").and_then(Value::as_str) == Some("assistant")
            && (object.get("content") == Some(&Value::Null)
                || (!object.contains_key("content") && object.contains_key("tool_calls")))
        {
            object.insert("content".to_string(), Value::Array(Vec::new()));
        }
    }
}

#[allow(clippy::result_large_err)]
fn anthropic_json<T: Serialize>(
    payload: Result<Json<T>, JsonRejection>,
) -> Result<T, axum::response::Response> {
    let value = payload.map(|Json(value)| value).map_err(|error| {
        let status = error.status();
        if matches!(
            status,
            StatusCode::PAYLOAD_TOO_LARGE | StatusCode::UNSUPPORTED_MEDIA_TYPE
        ) {
            let message = if status == StatusCode::PAYLOAD_TOO_LARGE {
                "request body exceeds 50 MiB"
            } else {
                "request Content-Type must be application/json or application/*+json"
            };
            return (
                status,
                Json(format_anthropic_error(status.as_u16(), message)),
            )
                .into_response();
        }
        anthropic_provider_error_response(&ProviderError::InvalidRequest(format!(
            "invalid JSON request: {error}"
        )))
    })?;
    let parsed = serde_json::to_value(&value).map_err(|_| {
        anthropic_provider_error_response(&ProviderError::InvalidRequest(
            "invalid JSON request".to_string(),
        ))
    })?;
    strict_json::validate_value(&parsed).map_err(|_| {
        anthropic_provider_error_response(&ProviderError::InvalidRequest(
            "invalid JSON request".to_string(),
        ))
    })?;
    Ok(value)
}

fn reject_unknown_top_level_fields(body: &Value, allowed: &[&str]) -> Result<(), ProviderError> {
    let object = body.as_object().ok_or_else(|| {
        ProviderError::InvalidRequest("request body must be a JSON object".to_string())
    })?;
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(ProviderError::InvalidRequest(format!(
            "unknown request field {field:?}"
        )));
    }
    Ok(())
}

fn reject_unknown_object_fields(
    object: &serde_json::Map<String, Value>,
    path: &str,
    allowed: &[&str],
) -> Result<(), ProviderError> {
    if let Some(field) = object
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(ProviderError::InvalidRequest(format!(
            "{path} contains unknown field {field:?}"
        )));
    }
    Ok(())
}

const OPENAI_COMPACT_FIELDS: &[&str] = &[
    "model",
    "messages",
    "tools",
    "programmatic_tool_calling",
    "safety_identifier",
    "include",
    "prompt_cache_retention",
    "reasoning",
    "multi_agent",
    "reasoning_effort",
    "responses_lite",
    "previous_response_id",
    "service_tier",
    "text",
    "prompt_cache_key",
    "prompt_cache_options",
    "verbosity",
];

const ANTHROPIC_COMPACT_FIELDS: &[&str] = &[
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "stop_sequences",
    "thinking",
    "output_config",
    "output_format",
    "cache_control",
    "context_management",
    "max_tokens",
    "programmatic_tool_calling",
    "safety_identifier",
    "include",
    "prompt_cache_retention",
    "reasoning",
    "multi_agent",
    "reasoning_effort",
    "responses_lite",
    "previous_response_id",
    "service_tier",
    "speed",
    "text",
    "prompt_cache_key",
    "prompt_cache_options",
    "verbosity",
];

fn positive_js_safe_integer(value: &Value) -> Option<i64> {
    value
        .as_number()
        .and_then(crate::strict_json::as_js_safe_integer)
        .filter(|value| *value > 0)
}

fn reject_explicit_null_anthropic_fields(body: &Value) -> Result<(), ProviderError> {
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
        if body.get(field).is_some_and(Value::is_null) {
            return Err(ProviderError::InvalidRequest(format!(
                "{field} must not be null"
            )));
        }
    }
    Ok(())
}

const ANTHROPIC_COUNT_FIELDS: &[&str] = &[
    "model",
    "messages",
    "system",
    "tools",
    "tool_choice",
    "thinking",
    "output_config",
    "output_format",
    "cache_control",
    "context_management",
    "multi_agent",
    "programmatic_tool_calling",
    "temperature",
    "top_p",
    "top_k",
    "stop_sequences",
    "parallel_tool_calls",
    "max_tokens",
];

const ANTHROPIC_MESSAGE_FIELDS: &[&str] = &[
    "model",
    "messages",
    "system",
    "max_tokens",
    "stream",
    "tools",
    "tool_choice",
    "stop_sequences",
    "temperature",
    "top_p",
    "top_k",
    "parallel_tool_calls",
    "thinking",
    "output_config",
    "output_format",
    "cache_control",
    "reasoning_effort",
    "reasoning",
    "prompt_cache_key",
    "prompt_cache_options",
    "safety_identifier",
    "verbosity",
    "multi_agent",
    "programmatic_tool_calling",
    "responses_lite",
    "service_tier",
    "speed",
    "subagent",
    "memgen_request",
    "context_management",
    "previous_response_id",
];

fn anthropic_sse_event(chunk: &str) -> Event {
    let mut lines = chunk.trim_end_matches('\n').splitn(2, '\n');
    let event_type = lines
        .next()
        .and_then(|line| line.strip_prefix("event: "))
        .expect("Anthropic adapter SSE must contain an event line");
    let data = lines
        .next()
        .and_then(|line| line.strip_prefix("data: "))
        .expect("Anthropic adapter SSE must contain a data line");
    Event::default().event(event_type).data(data)
}

fn anthropic_stream_error_event(status: u16, message: &str) -> Event {
    Event::default().event("error").data(
        serde_json::to_string(&format_anthropic_error(status, message))
            .expect("Anthropic error JSON value must serialize"),
    )
}

fn openai_sse_json(payload: &Value) -> Result<Event, ProviderError> {
    serde_json::to_string(payload)
        .map(|data| Event::default().data(data))
        .map_err(|error| {
            ProviderError::Request(format!("failed to serialize OpenAI SSE event: {error}"))
        })
}

fn openai_stream_error_event(error: &ProviderError) -> Event {
    Event::default().data(
        json!({
            "error": {
                "message": public_error_message(error),
                "type": provider_error_type(error),
                "code": provider_error_type(error),
            }
        })
        .to_string(),
    )
}

struct DownstreamEventReceiver {
    receiver: tokio::sync::mpsc::Receiver<Event>,
    cancellation: Arc<AtomicBool>,
}

impl Drop for DownstreamEventReceiver {
    fn drop(&mut self) {
        self.cancellation.store(true, Ordering::Release);
    }
}

fn send_sse_event_with_backpressure(
    sender: &tokio::sync::mpsc::Sender<Event>,
    mut event: Event,
    cancellation: &AtomicBool,
    disconnected_message: &str,
) -> Result<(), ProviderError> {
    loop {
        if cancellation.load(Ordering::Acquire) {
            return Err(ProviderError::Request(disconnected_message.to_string()));
        }
        match sender.try_send(event) {
            Ok(()) => return Ok(()),
            Err(tokio::sync::mpsc::error::TrySendError::Full(returned)) => {
                event = returned;
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                return Err(ProviderError::Request(disconnected_message.to_string()));
            }
        }
    }
}

struct OpenAiStreamState {
    request_id: String,
    created: i64,
    model: String,
    tool_call_index: usize,
    finished: bool,
}

impl OpenAiStreamState {
    fn chunk(&self, delta: Value, finish_reason: Value, response_id: Option<&str>) -> Value {
        let mut chunk = json!({
            "id": self.request_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }],
        });
        if let Some(response_id) = response_id {
            chunk
                .as_object_mut()
                .expect("OpenAI stream chunk is an object")
                .insert(
                    "response_id".to_string(),
                    Value::String(response_id.to_string()),
                );
        }
        chunk
    }

    fn preamble(&self) -> Result<Event, ProviderError> {
        openai_sse_json(&self.chunk(json!({"role": "assistant"}), Value::Null, None))
    }

    fn push(&mut self, event: &Value) -> Result<Vec<Event>, ProviderError> {
        let event_type = event.get("type").and_then(Value::as_str).ok_or_else(|| {
            ProviderError::UpstreamProtocol(
                "normalized response event requires a string type".to_string(),
            )
        })?;
        let chunks = match event_type {
            "content" => {
                let text = event.get("text").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::UpstreamProtocol(
                        "normalized content event requires string text".to_string(),
                    )
                })?;
                vec![openai_sse_json(&self.chunk(
                    json!({"content": text}),
                    Value::Null,
                    None,
                ))?]
            }
            "reasoning_delta" => {
                let text = event.get("text").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::UpstreamProtocol(
                        "normalized reasoning_delta event requires string text".to_string(),
                    )
                })?;
                vec![openai_sse_json(&self.chunk(
                    json!({"reasoning_content": text}),
                    Value::Null,
                    None,
                ))?]
            }
            "reasoning_raw_delta" => {
                let text = event.get("text").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::UpstreamProtocol(
                        "normalized reasoning_raw_delta event requires string text".to_string(),
                    )
                })?;
                vec![openai_sse_json(&self.chunk(
                    json!({"reasoning": text}),
                    Value::Null,
                    None,
                ))?]
            }
            "tool_call" => {
                let id = event.get("id").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::UpstreamProtocol(
                        "normalized tool_call event requires a string id".to_string(),
                    )
                })?;
                let name = event.get("name").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::UpstreamProtocol(
                        "normalized tool_call event requires a string name".to_string(),
                    )
                })?;
                let arguments =
                    event
                        .get("arguments")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            ProviderError::UpstreamProtocol(
                                "normalized tool_call event requires string arguments".to_string(),
                            )
                        })?;
                let tool_call = json!({
                    "index": self.tool_call_index,
                    "id": id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": arguments,
                    },
                });
                self.tool_call_index += 1;
                vec![openai_sse_json(&self.chunk(
                    json!({"tool_calls": [tool_call]}),
                    Value::Null,
                    None,
                ))?]
            }
            "finish" => {
                if self.finished {
                    return Err(ProviderError::UpstreamProtocol(
                        "normalized response stream contains more than one finish event"
                            .to_string(),
                    ));
                }
                let response_id = event
                    .get("response_id")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        ProviderError::UpstreamProtocol(
                            "normalized finish event requires a non-empty response_id".to_string(),
                        )
                    })?;
                let finish_reason = match event.get("finish_reason") {
                    Some(Value::String(value))
                        if matches!(value.as_str(), "stop" | "tool_calls") =>
                    {
                        Value::String(value.clone())
                    }
                    Some(_) => {
                        return Err(ProviderError::UpstreamProtocol(
                            "normalized finish event requires a final finish_reason".to_string(),
                        ));
                    }
                    None => {
                        return Err(ProviderError::UpstreamProtocol(
                            "normalized finish event requires a final finish_reason".to_string(),
                        ));
                    }
                };
                self.finished = true;
                let mut chunks = vec![openai_sse_json(&self.chunk(
                    json!({}),
                    finish_reason,
                    Some(response_id),
                ))?];
                if let Some(usage_value) = event.get("usage").filter(|usage| !usage.is_null()) {
                    let usage = parse_usage(usage_value)?;
                    chunks.push(openai_sse_json(&json!({
                        "id": self.request_id,
                        "response_id": response_id,
                        "object": "chat.completion.chunk",
                        "created": self.created,
                        "model": self.model,
                        "choices": [],
                        "usage": openai_usage_value(&usage),
                    }))?);
                }
                chunks
            }
            "reasoning_section_break" => Vec::new(),
            "web_search_call" => {
                return Err(ProviderError::UpstreamProtocol(
                    "provider web_search_call event cannot be represented by /v1/chat/completions"
                        .to_string(),
                ));
            }
            _ => {
                return Err(ProviderError::UpstreamProtocol(format!(
                    "normalized response stream has unsupported event type {event_type:?}"
                )));
            }
        };
        Ok(chunks)
    }
}

fn map_error_status(e: &ProviderError) -> StatusCode {
    match e {
        ProviderError::Auth(_) => StatusCode::UNAUTHORIZED,
        ProviderError::InvalidRequest(_) => StatusCode::BAD_REQUEST,
        ProviderError::ModelNotFound(_) => StatusCode::NOT_FOUND,
        ProviderError::CatalogUnavailable(_) => StatusCode::SERVICE_UNAVAILABLE,
        ProviderError::UpstreamProtocol(_) => StatusCode::BAD_GATEWAY,
        ProviderError::UpstreamTransport(_) => StatusCode::BAD_GATEWAY,
        ProviderError::UpstreamHttp { status, .. } => {
            StatusCode::from_u16(*status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
        }
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/images/generations", post(images_generations))
        .route("/v1/inspect", post(inspect))
        .route("/v1/compact", post(compact))
        .route("/v1/messages/compact", post(compact))
        .route("/v1/messages/count_tokens", post(anthropic_count_tokens))
        .route("/v1/messages", post(anthropic_messages))
        .layer(DefaultBodyLimit::max(REQUEST_BODY_LIMIT_BYTES))
        .with_state(state)
}

fn request_messages_to_internal(messages: &[ChatMessage]) -> Result<Vec<Message>, ProviderError> {
    if messages.is_empty() {
        return Err(ProviderError::InvalidRequest(
            "messages must be a non-empty array".to_string(),
        ));
    }
    let mut result = Vec::new();
    for (message_index, msg) in messages.iter().enumerate() {
        let role = map_role(&msg.role)?;
        if msg.name.is_some() {
            return Err(ProviderError::InvalidRequest(format!(
                "messages item {message_index} name is not supported"
            )));
        }
        let (content, structured_content) = normalize_content(&msg.content, role, message_index)?;
        let tool_calls = parse_tool_calls(&msg.tool_calls, message_index)?;
        if msg.content.is_none() && role != MessageRole::Assistant {
            return Err(ProviderError::InvalidRequest(format!(
                "messages item {message_index} requires content"
            )));
        }
        let mut m = Message::new(role, content, tool_calls, msg.tool_call_id.clone(), None)
            .map_err(|error| ProviderError::InvalidRequest(error.to_string()))?;
        m.structured_content = structured_content;
        result.push(m);
    }
    Ok(result)
}

fn map_role(role: &str) -> Result<MessageRole, ProviderError> {
    match role {
        "system" => Ok(MessageRole::System),
        "developer" => Ok(MessageRole::Developer),
        "user" => Ok(MessageRole::User),
        "assistant" => Ok(MessageRole::Assistant),
        "tool" => Ok(MessageRole::Tool),
        _ => Err(ProviderError::InvalidRequest(format!(
            "unsupported message role {role:?}"
        ))),
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
                        if (block_type == "output_text" && role != MessageRole::Assistant)
                            || (block_type == "input_text" && role == MessageRole::Assistant)
                        {
                            return Err(ProviderError::InvalidRequest(format!(
                                "messages item {message_index} content item {block_index} type {block_type:?} is not valid for role {role:?}"
                            )));
                        }
                        reject_unknown_object_fields(
                            object,
                            &format!("messages item {message_index} content item {block_index}"),
                            &["type", "text", "prompt_cache_breakpoint"],
                        )?;
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
                        reject_unknown_object_fields(
                            object,
                            &format!("messages item {message_index} content item {block_index}"),
                            &["type", "image_url", "detail", "prompt_cache_breakpoint"],
                        )?;
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
                    "input_audio" => {
                        reject_unknown_object_fields(
                            object,
                            &format!("messages item {message_index} content item {block_index}"),
                            &["type", "input_audio", "prompt_cache_breakpoint"],
                        )?;
                        if role != MessageRole::User {
                            return Err(ProviderError::InvalidRequest(
                                "input_audio content is supported only on user messages"
                                    .to_string(),
                            ));
                        }
                        let input_audio = object
                            .get("input_audio")
                            .and_then(Value::as_object)
                            .ok_or_else(|| {
                                ProviderError::InvalidRequest(format!(
                                    "messages item {message_index} content item {block_index} input_audio must be an object"
                                ))
                            })?;
                        reject_unknown_object_fields(
                            input_audio,
                            &format!(
                                "messages item {message_index} content item {block_index} input_audio"
                            ),
                            &["data", "format"],
                        )?;
                        let data = input_audio
                            .get("data")
                            .and_then(Value::as_str)
                            .ok_or_else(|| {
                                ProviderError::InvalidRequest(format!(
                                    "messages item {message_index} content item {block_index} input_audio.data must be a string"
                                ))
                            })?;
                        let audio_format = input_audio
                            .get("format")
                            .and_then(Value::as_str)
                            .filter(|value| matches!(*value, "wav" | "mp3"))
                            .ok_or_else(|| {
                                ProviderError::InvalidRequest(format!(
                                    "messages item {message_index} content item {block_index} input_audio.format must be wav or mp3"
                                ))
                            })?;
                        let mut normalized = serde_json::Map::new();
                        normalized.insert("type".to_string(), json!("input_audio"));
                        normalized.insert(
                            "audio_url".to_string(),
                            Value::String(format!("data:audio/{audio_format};base64,{data}")),
                        );
                        if let Some(value) = breakpoint {
                            normalized.insert("prompt_cache_breakpoint".to_string(), value);
                        }
                        blocks.push(Value::Object(normalized));
                    }
                    "file" | "input_file" | "audio" => {
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
            Ok((text_parts.join(""), Some(blocks)))
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
        Value::Object(image) => {
            reject_unknown_object_fields(
                image,
                &format!("messages item {message_index} content item {block_index} image_url"),
                &["url", "detail"],
            )?;
            if image.get("detail") == Some(&Value::Null) {
                return Err(ProviderError::InvalidRequest(format!(
                    "messages item {message_index} content item {block_index} image_url.detail must not be null"
                )));
            }
            let url = image
                .get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    ProviderError::InvalidRequest(format!(
                        "messages item {message_index} content item {block_index} image_url.url must be a string"
                    ))
                })?
                .to_string();
            let nested_detail = image.get("detail").filter(|value| !value.is_null());
            let outer_detail = object.get("detail").filter(|value| !value.is_null());
            if nested_detail.is_some() && outer_detail.is_some() && nested_detail != outer_detail {
                return Err(ProviderError::InvalidRequest(format!(
                    "messages item {message_index} content item {block_index} image detail fields conflict"
                )));
            }
            (url, nested_detail.or(outer_detail))
        }
        _ => {
            return Err(ProviderError::InvalidRequest(format!(
                "messages item {message_index} content item {block_index} image_url must be a string or object"
            )));
        }
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

fn parse_tool_calls(
    raw: &Option<Vec<Value>>,
    message_index: usize,
) -> Result<Vec<ToolCall>, ProviderError> {
    let items = match raw {
        Some(v) => v,
        None => return Ok(vec![]),
    };
    let mut calls = Vec::new();
    for (call_index, item) in items.iter().enumerate() {
        let path = format!("messages item {message_index} tool_calls item {call_index}");
        let obj = item
            .as_object()
            .ok_or_else(|| ProviderError::InvalidRequest(format!("{path} must be an object")))?;
        reject_unknown_object_fields(obj, &path, &["type", "id", "function"])?;
        if obj.get("type").and_then(Value::as_str) != Some("function") {
            return Err(ProviderError::InvalidRequest(format!(
                "{path}.type must be function"
            )));
        }
        let call_id = obj
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| ProviderError::InvalidRequest(format!("{path}.id is required")))?
            .to_string();
        let func = obj
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| ProviderError::InvalidRequest(format!("{path}.function is required")))?;
        reject_unknown_object_fields(func, &format!("{path}.function"), &["name", "arguments"])?;
        let name = func.get("name").and_then(Value::as_str).ok_or_else(|| {
            ProviderError::InvalidRequest(format!("{path}.function.name must be a string"))
        })?;
        let arguments = func
            .get("arguments")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProviderError::InvalidRequest(format!("{path}.function.arguments must be a string"))
            })?
            .to_string();
        calls.push(ToolCall {
            id: call_id,
            name: name.to_string(),
            arguments,
        });
    }
    Ok(calls)
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
        if obj.get("type").and_then(Value::as_str) != Some("function") {
            return Err(ProviderError::InvalidRequest(
                "tools entries must have type function".to_string(),
            ));
        }
        if let Some(field) = obj.keys().find(|field| {
            !matches!(
                field.as_str(),
                "type"
                    | "function"
                    | "allowed_callers"
                    | "output_schema"
                    | "defer_loading"
                    | "eager_input_streaming"
            )
        }) {
            return Err(ProviderError::InvalidRequest(format!(
                "unsupported tool field {field:?}"
            )));
        }
        let func = obj
            .get("function")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                ProviderError::InvalidRequest("tools entries require a function object".to_string())
            })?;
        if let Some(field) = func.keys().find(|field| {
            !matches!(
                field.as_str(),
                "name"
                    | "description"
                    | "parameters"
                    | "strict"
                    | "allowed_callers"
                    | "output_schema"
                    | "defer_loading"
                    | "eager_input_streaming"
            )
        }) {
            return Err(ProviderError::InvalidRequest(format!(
                "unsupported tool function field {field:?}"
            )));
        }
        let name = func
            .get("name")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                ProviderError::InvalidRequest("tool function name is required".to_string())
            })?;
        let description = match func.get("description") {
            None => None,
            Some(Value::Null) => {
                return Err(ProviderError::InvalidRequest(
                    "tool function description must not be null".to_string(),
                ));
            }
            Some(Value::String(value)) => Some(value.clone()),
            Some(_) => {
                return Err(ProviderError::InvalidRequest(
                    "tool function description must be a string when provided".to_string(),
                ));
            }
        };
        let params = func
            .get("parameters")
            .cloned()
            .unwrap_or_else(|| Value::Object(serde_json::Map::new()));
        if !params.is_object() {
            return Err(ProviderError::InvalidRequest(
                "tool function parameters must be an object".to_string(),
            ));
        }
        let strict = match func.get("strict") {
            None | Some(Value::Null) => false,
            Some(Value::Bool(value)) => *value,
            Some(_) => {
                return Err(ProviderError::InvalidRequest(
                    "tool function strict must be a boolean when provided".to_string(),
                ));
            }
        };
        schemas.push(ToolSchema {
            name: name.to_string(),
            description,
            parameters: params,
            strict,
        });
    }
    Ok(if schemas.is_empty() {
        None
    } else {
        Some(schemas)
    })
}

fn reject_unsupported_openai_controls(req: &ChatCompletionRequest) -> Result<(), ProviderError> {
    for (name, present) in [
        ("temperature", req.temperature.is_some()),
        ("max_tokens", req.max_tokens.is_some()),
        ("max_completion_tokens", req.max_completion_tokens.is_some()),
        (
            "stop",
            req.stop.as_ref().is_some_and(|value| !value.is_null()),
        ),
        ("top_p", req.top_p.is_some()),
        ("frequency_penalty", req.frequency_penalty.is_some()),
        ("presence_penalty", req.presence_penalty.is_some()),
        ("user", req.user.is_some()),
        ("n", req.n.as_ref().is_some_and(|value| !value.is_null())),
        (
            "logprobs",
            req.logprobs.as_ref().is_some_and(|value| !value.is_null()),
        ),
        (
            "top_logprobs",
            req.top_logprobs
                .as_ref()
                .is_some_and(|value| !value.is_null()),
        ),
        (
            "logit_bias",
            req.logit_bias
                .as_ref()
                .is_some_and(|value| !value.is_null()),
        ),
        (
            "seed",
            req.seed.as_ref().is_some_and(|value| !value.is_null()),
        ),
        (
            "response_format",
            req.response_format
                .as_ref()
                .is_some_and(|value| !value.is_null()),
        ),
        (
            "stream_options",
            req.stream_options
                .as_ref()
                .is_some_and(|value| !value.is_null()),
        ),
        (
            "modalities",
            req.modalities
                .as_ref()
                .is_some_and(|value| !value.is_null()),
        ),
        (
            "audio",
            req.audio.as_ref().is_some_and(|value| !value.is_null()),
        ),
        (
            "store",
            req.store.as_ref().is_some_and(|value| !value.is_null()),
        ),
        (
            "metadata",
            req.metadata.as_ref().is_some_and(|value| !value.is_null()),
        ),
        (
            "prediction",
            req.prediction
                .as_ref()
                .is_some_and(|value| !value.is_null()),
        ),
        (
            "function_call",
            req.function_call
                .as_ref()
                .is_some_and(|value| !value.is_null()),
        ),
        (
            "functions",
            req.functions.as_ref().is_some_and(|value| !value.is_null()),
        ),
        (
            "prompt_cache_retention",
            req.prompt_cache_retention
                .as_ref()
                .is_some_and(|value| !value.is_null()),
        ),
        (
            "web_search_options",
            req.web_search_options
                .as_ref()
                .is_some_and(|value| !value.is_null()),
        ),
    ] {
        if present {
            return Err(ProviderError::InvalidRequest(format!(
                "{name} is not supported by the Codex OAuth HTTP transport"
            )));
        }
    }
    Ok(())
}

fn parse_openai_tool_choice(value: Option<&Value>) -> Result<Option<Value>, ProviderError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    if let Some(choice) = value.as_str() {
        if matches!(choice, "auto" | "none" | "required") {
            return Ok(Some(Value::String(choice.to_string())));
        }
        return Err(ProviderError::InvalidRequest(
            "tool_choice must be one of: auto, none, required, or a function choice object"
                .to_string(),
        ));
    }
    let object = value.as_object().ok_or_else(|| {
        ProviderError::InvalidRequest("tool_choice must be a string or object".to_string())
    })?;
    if let Some(field) = object
        .keys()
        .find(|field| !matches!(field.as_str(), "type" | "function"))
    {
        return Err(ProviderError::InvalidRequest(format!(
            "tool_choice.{field} is not supported"
        )));
    }
    if object.get("type").and_then(Value::as_str) != Some("function") {
        return Err(ProviderError::InvalidRequest(
            "tool_choice.type must be function".to_string(),
        ));
    }
    let function = object
        .get("function")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ProviderError::InvalidRequest("tool_choice.function must be an object".to_string())
        })?;
    if let Some(field) = function.keys().find(|field| field.as_str() != "name") {
        return Err(ProviderError::InvalidRequest(format!(
            "tool_choice.function.{field} is not supported"
        )));
    }
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.is_empty())
        .ok_or_else(|| {
            ProviderError::InvalidRequest(
                "tool_choice.function.name must be a non-empty string".to_string(),
            )
        })?;
    Ok(Some(json!({"type": "function", "name": name})))
}

fn reject_unsupported_anthropic_controls(body: &Value) -> Result<(), ProviderError> {
    for field in [
        "temperature",
        "top_p",
        "top_k",
        "stop_sequences",
        "parallel_tool_calls",
    ] {
        if body.get(field).is_some() {
            return Err(ProviderError::InvalidRequest(format!(
                "{field} is not supported by the Codex OAuth HTTP transport"
            )));
        }
    }
    Ok(())
}

fn raw_context_window_for_model(
    state: &AppState,
    model: &ModelInfo,
) -> Result<Option<i64>, ProviderError> {
    let resolved = if let Some(configured) = state.codex_config.model_context_window {
        if model.max_context_window.is_some_and(|maximum| maximum <= 0) {
            return Err(ProviderError::CatalogUnavailable(format!(
                "model {} reports a non-positive max context window",
                model.slug
            )));
        }
        Some(
            model
                .max_context_window
                .map(|maximum| configured.min(maximum))
                .unwrap_or(configured),
        )
    } else {
        model.resolved_context_window()
    };
    if resolved.is_some_and(|value| value <= 0) {
        return Err(ProviderError::CatalogUnavailable(format!(
            "model {} reports a non-positive context window",
            model.slug
        )));
    }
    Ok(resolved)
}

fn context_window_for_model(
    state: &AppState,
    model: &ModelInfo,
) -> Result<Option<i64>, ProviderError> {
    let Some(resolved) = raw_context_window_for_model(state, model)? else {
        return Ok(None);
    };
    let effective = resolved.saturating_mul(model.effective_context_window_percent) / 100;
    if effective <= 0 || effective > 9_007_199_254_740_991 {
        return Err(ProviderError::CatalogUnavailable(format!(
            "model {} reports an unusable effective context window",
            model.slug
        )));
    }
    Ok(Some(effective))
}

fn auto_compact_token_limit_for_model(
    state: &AppState,
    model: &ModelInfo,
) -> Result<Option<i64>, ProviderError> {
    let context_window = raw_context_window_for_model(state, model)?;
    let maximum = context_window.map(|value| value.saturating_mul(9) / 10);
    if let Some(limit) = state.codex_config.model_auto_compact_token_limit {
        return Ok(Some(
            maximum.map(|maximum| limit.min(maximum)).unwrap_or(limit),
        ));
    }
    Ok(match (model.auto_compact_token_limit, maximum) {
        (Some(limit), Some(maximum)) => Some(limit.min(maximum)),
        (Some(limit), None) => Some(limit),
        (None, maximum) => maximum,
    })
}

fn effective_reasoning_effort_with_options(
    state: &AppState,
    requested: Option<&str>,
    reasoning: Option<&Value>,
    model: &ModelInfo,
) -> Result<Option<String>, ProviderError> {
    if requested.is_some_and(|value| value.is_empty() || value != value.trim()) {
        return Err(ProviderError::InvalidRequest(
            "reasoning_effort must be a non-empty string when provided".to_string(),
        ));
    }
    let mut nested_effort: Option<&str> = None;
    if let Some(reasoning) = reasoning {
        let object = reasoning.as_object().ok_or_else(|| {
            ProviderError::InvalidRequest("reasoning must be an object".to_string())
        })?;
        if let Some(value) = object.get("effort") {
            if !value.is_null() {
                nested_effort = Some(
                    value
                        .as_str()
                        .filter(|value| !value.is_empty() && *value == value.trim())
                        .ok_or_else(|| {
                            ProviderError::InvalidRequest(
                                "reasoning.effort must be a non-empty string when provided"
                                    .to_string(),
                            )
                        })?,
                );
            }
        }
        if object.get("mode").is_some_and(|value| !value.is_null()) {
            return Err(ProviderError::InvalidRequest(
                "reasoning.mode is not supported by the Codex OAuth HTTP transport".to_string(),
            ));
        }
    }
    if let (Some(top_level), Some(nested)) = (requested, nested_effort) {
        if top_level != nested {
            return Err(ProviderError::InvalidRequest(
                "reasoning_effort conflicts with reasoning.effort".to_string(),
            ));
        }
    }
    let configured = (requested.is_none() && nested_effort.is_none())
        .then_some(state.codex_config.model_reasoning_effort.as_deref())
        .flatten();
    if configured.is_some_and(|value| value.is_empty() || value != value.trim()) {
        return Err(ProviderError::Request(
            "configured reasoning_effort is invalid".to_string(),
        ));
    }
    let selected = requested
        .or(nested_effort)
        .map(str::to_string)
        .or_else(|| configured.map(str::to_string))
        .or_else(|| model.default_reasoning_level.clone());
    let explicit = requested.is_some() || nested_effort.is_some();
    match selected.map(|effort| resolve_reasoning_effort(model, &effort)) {
        Some(Err(_)) if configured.is_some() => Err(ProviderError::Request(
            "configured reasoning_effort is not supported by the live model".to_string(),
        )),
        Some(Err(_)) if !explicit => Err(ProviderError::CatalogUnavailable(
            "selected model publishes an unsupported default reasoning effort".to_string(),
        )),
        Some(result) => result.map(Some),
        None => Ok(None),
    }
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
        reasoning: reasoning.clone(),
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

fn optional_header_string(
    headers: &HeaderMap,
    name: &str,
) -> Result<Option<String>, ProviderError> {
    headers
        .get(name)
        .map(|value| {
            let value = value.to_str().map_err(|_| {
                ProviderError::InvalidRequest(format!("{name} must be valid UTF-8"))
            })?;
            if value.trim().is_empty() {
                return Err(ProviderError::InvalidRequest(format!(
                    "{name} must be non-empty when provided"
                )));
            }
            Ok(value.to_string())
        })
        .transpose()
}

fn merge_optional_nonempty_string(
    body_value: Option<String>,
    header_value: Option<String>,
    field: &str,
) -> Result<Option<String>, ProviderError> {
    if body_value
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(ProviderError::InvalidRequest(format!(
            "{field} must be non-empty when provided"
        )));
    }
    if matches!((&body_value, &header_value), (Some(body), Some(header)) if body != header) {
        return Err(ProviderError::InvalidRequest(format!(
            "{field} conflicts with its request header"
        )));
    }
    Ok(body_value.or(header_value))
}

fn merge_optional_bool(
    body_value: Option<bool>,
    header_value: Option<bool>,
    field: &str,
) -> Result<Option<bool>, ProviderError> {
    if matches!((body_value, header_value), (Some(body), Some(header)) if body != header) {
        return Err(ProviderError::InvalidRequest(format!(
            "{field} conflicts with its request header"
        )));
    }
    Ok(body_value.or(header_value))
}

fn optional_header_bool(headers: &HeaderMap, name: &str) -> Result<Option<bool>, ProviderError> {
    optional_header_string(headers, name)?
        .map(|value| match value.as_str() {
            "true" => Ok(true),
            "false" => Ok(false),
            _ => Err(ProviderError::InvalidRequest(format!(
                "{name} must be exactly true or false"
            ))),
        })
        .transpose()
}

fn anthropic_prompt_cache_key(
    body: &Value,
    headers: &HeaderMap,
) -> Result<Option<String>, ProviderError> {
    if let Some(explicit) = optional_string_field(body, "prompt_cache_key")? {
        if explicit.trim().is_empty() {
            return Err(ProviderError::InvalidRequest(
                "prompt_cache_key must be a non-empty string when provided".to_string(),
            ));
        }
        return Ok(Some(explicit));
    }

    let Some(session_id) = claude_code_session_id(headers)? else {
        return Ok(None);
    };

    let digest =
        Sha256::digest(format!("codex-as-api:claude-code-session:{session_id}").as_bytes());
    let mut key = String::with_capacity(digest.len() * 2);
    for byte in digest {
        write!(&mut key, "{byte:02x}").expect("writing to a String cannot fail");
    }
    Ok(Some(key))
}

fn claude_code_session_id(headers: &HeaderMap) -> Result<Option<String>, ProviderError> {
    let mut values = headers.get_all("x-claude-code-session-id").iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(ProviderError::InvalidRequest(
            "x-claude-code-session-id must be provided at most once".to_string(),
        ));
    }
    let value = value.to_str().map_err(|_| {
        ProviderError::InvalidRequest("x-claude-code-session-id must be valid UTF-8".to_string())
    })?;
    if value.trim().is_empty() {
        return Err(ProviderError::InvalidRequest(
            "x-claude-code-session-id must be a non-empty string".to_string(),
        ));
    }
    Ok(Some(value.to_string()))
}

fn validate_anthropic_compatibility_scope(
    body: &Value,
    headers: &HeaderMap,
) -> Result<bool, ProviderError> {
    let is_claude_code = claude_code_session_id(headers)?.is_some();
    validate_anthropic_cache_controls(body, is_claude_code)
        .map_err(ProviderError::InvalidRequest)?;
    if body.get("max_tokens").is_some_and(|value| !value.is_null()) && !is_claude_code {
        return Err(ProviderError::InvalidRequest(
            "max_tokens is accepted without forwarding only for Claude Code requests".to_string(),
        ));
    }
    Ok(is_claude_code)
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

fn reject_unsupported_proxy_extension_presence(
    body: &Value,
    _anthropic: bool,
) -> Result<(), ProviderError> {
    for field in [
        "multi_agent",
        "programmatic_tool_calling",
        "safety_identifier",
    ] {
        if body.get(field).is_some() {
            return Err(ProviderError::InvalidRequest(format!(
                "{field} is not supported by this compatibility API"
            )));
        }
    }
    Ok(())
}

fn reject_unsupported_tool_features(body: &Value, anthropic: bool) -> Result<(), ProviderError> {
    reject_unsupported_generation_tool_features(
        body.get("tools")
            .and_then(Value::as_array)
            .map(Vec::as_slice),
        body.get("programmatic_tool_calling"),
        anthropic,
    )
}

fn reject_unsupported_generation_tool_features(
    tools: Option<&[Value]>,
    programmatic_tool_calling: Option<&Value>,
    anthropic: bool,
) -> Result<(), ProviderError> {
    if programmatic_tool_calling.is_some() {
        return Err(ProviderError::InvalidRequest(
            "programmatic_tool_calling is not supported by this compatibility API".to_string(),
        ));
    }
    let Some(tools) = tools else {
        return Ok(());
    };
    for tool in tools {
        let object = tool.as_object().ok_or_else(|| {
            ProviderError::InvalidRequest("tools entries must be objects".to_string())
        })?;
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
        for field in ["defer_loading", "eager_input_streaming"] {
            for container in [object, function] {
                match container.get(field) {
                    None => {}
                    Some(Value::Null) if anthropic && field == "eager_input_streaming" => {}
                    Some(Value::Bool(_)) => {
                        return Err(ProviderError::InvalidRequest(format!(
                            "tool {field} is not supported by the Codex OAuth backend"
                        )));
                    }
                    Some(_) => {
                        return Err(ProviderError::InvalidRequest(format!(
                            "tool {field} must be a boolean when provided"
                        )));
                    }
                }
            }
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
            MessageRole::Developer => "developer",
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

type CompactRequestParts = (
    Vec<Message>,
    Option<Vec<ToolSchema>>,
    Option<String>,
    Option<Value>,
    Option<bool>,
);

fn messages_from_compact_body(
    body: &Value,
    force_anthropic: bool,
) -> Result<CompactRequestParts, ProviderError> {
    if force_anthropic
        || body.get("system").is_some()
        || body.get("thinking").is_some()
        || body.get("tool_choice").is_some()
        || body.get("stop_sequences").is_some()
    {
        let (messages, tools, tool_choice, _stop, reasoning_effort, text, parallel_tool_calls) =
            anthropic_request_to_internal(body).map_err(ProviderError::InvalidRequest)?;
        if tool_choice
            .as_ref()
            .is_some_and(|choice| choice != &json!("auto"))
        {
            return Err(ProviderError::InvalidRequest(
                "compact supports only Anthropic tool_choice.type=auto".to_string(),
            ));
        }
        return Ok((messages, tools, reasoning_effort, text, parallel_tool_calls));
    }

    let raw_items = match body.get("messages") {
        None | Some(Value::Null) => {
            return Err(ProviderError::InvalidRequest(
                "messages must be a non-empty array".to_string(),
            ));
        }
        Some(Value::Array(items)) => items.as_slice(),
        Some(_) => {
            return Err(ProviderError::InvalidRequest(
                "messages must be an array".to_string(),
            ));
        }
    };
    if raw_items.is_empty() {
        return Err(ProviderError::InvalidRequest(
            "messages must be a non-empty array".to_string(),
        ));
    }
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

async fn resolve_route_model(
    state: &AppState,
    requested: Option<String>,
) -> Result<ResolvedModel, axum::response::Response> {
    resolve_route_model_typed(state, requested)
        .await
        .map_err(|error| provider_error_response(&error))
}

async fn resolve_route_model_typed(
    state: &AppState,
    requested: Option<String>,
) -> Result<ResolvedModel, ProviderError> {
    let provider = state.provider.clone();
    task::spawn_blocking(move || match requested {
        Some(model) => provider.resolve_model(Some(&model)),
        None => provider.configured_or_default_model(),
    })
    .await
    .map_err(|error| ProviderError::Request(format!("blocking provider task failed: {error}")))?
}

async fn resolve_anthropic_route_model(
    state: &AppState,
    requested: &str,
) -> Result<ResolvedModel, ProviderError> {
    if requested.trim().is_empty() || requested != requested.trim() {
        return Err(ProviderError::InvalidRequest(
            "model must be a non-empty string without surrounding whitespace".to_string(),
        ));
    }
    let provider = state.provider.clone();
    let requested = requested.to_string();
    task::spawn_blocking(move || {
        if requested.starts_with("claude-") {
            if provider.model.trim().is_empty() {
                return Err(ProviderError::InvalidRequest(
                    "claude-* model facades require CODEX_AS_API_MODEL or config.toml model"
                        .to_string(),
                ));
            }
            let snapshot = provider.model_catalog_snapshot()?;
            provider.configured_or_default_model_from_snapshot(snapshot)
        } else {
            provider.resolve_model(Some(&requested))
        }
    })
    .await
    .map_err(|error| ProviderError::Request(format!("blocking provider task failed: {error}")))?
}

async fn models(State(state): State<AppState>) -> axum::response::Response {
    let provider = state.provider.clone();
    let snapshot = match task::spawn_blocking(move || provider.model_catalog_snapshot()).await {
        Ok(Ok(snapshot)) => snapshot,
        Ok(Err(error)) => return provider_error_response(&error),
        Err(error) => return join_error_response(error),
    };
    let data: Vec<Value> = snapshot
        .models
        .iter()
        .map(|model| {
            json!({
                "id": model.slug,
                "object": "model",
                "owned_by": "openai",
                "display_name": model.display_name,
                "description": model.description,
                "priority": model.priority,
                "visibility": model.visibility,
                "supported_in_api": model.supported_in_api,
                "default_reasoning_level": model.default_reasoning_level,
                "supported_reasoning_levels": model.supported_reasoning_levels,
                "multi_agent_reasoning_effort": model.multi_agent_reasoning_effort,
                "supports_reasoning_summary_parameter": model.supports_reasoning_summary_parameter,
                "default_reasoning_summary": model.default_reasoning_summary,
                "comp_hash": model.comp_hash,
                "service_tiers": model.service_tiers,
                "default_service_tier": model.default_service_tier,
                "support_verbosity": model.support_verbosity,
                "default_verbosity": model.default_verbosity,
                "supports_image_detail_original": model.supports_image_detail_original,
                "context_window": model.context_window,
                "max_context_window": model.max_context_window,
                "auto_compact_token_limit": model.auto_compact_token_limit,
                "effective_context_window_percent": model.effective_context_window_percent,
                "input_modalities": model.input_modalities,
                "use_responses_lite": model.use_responses_lite,
            })
        })
        .collect();
    Json(json!({"object": "list", "data": data})).into_response()
}

async fn health(State(state): State<AppState>) -> axum::response::Response {
    let auth_available = auth::is_auth_locally_available(state.auth_path.as_deref());
    let provider = state.provider.clone();
    let snapshot = match task::spawn_blocking(move || provider.model_catalog_snapshot()).await {
        Ok(Ok(snapshot)) => snapshot,
        Ok(Err(error)) => {
            let auth_available = auth_available && !matches!(error, ProviderError::Auth(_));
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "status": "error",
                    "auth_available": auth_available,
                    "catalog_status": "unavailable",
                    "catalog_fetched_at": null,
                    "catalog_expires_at": null,
                    "model": null,
                    "reasoning_effort": null,
                    "context_window": null,
                    "auto_compact_token_limit": null,
                    "error": {
                        "type": provider_error_type(&error),
                        "message": health_error_message(&error),
                    },
                })),
            )
                .into_response();
        }
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "status": "error",
                    "auth_available": auth_available,
                    "catalog_status": "unavailable",
                    "catalog_fetched_at": null,
                    "catalog_expires_at": null,
                    "model": null,
                    "reasoning_effort": null,
                    "context_window": null,
                    "auto_compact_token_limit": null,
                    "error": {
                        "type": "server_error",
                        "message": "health preflight failed",
                    },
                })),
            )
                .into_response();
        }
    };
    let provider = state.provider.clone();
    let diagnostic_snapshot = snapshot.clone();
    let resolved = match task::spawn_blocking(move || {
        provider.configured_or_default_model_from_snapshot(snapshot)
    })
    .await
    {
        Ok(Ok(resolved)) => resolved,
        Ok(Err(error)) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "status": "error",
                    "auth_available": auth_available,
                    "catalog_status": "fresh",
                    "catalog_fetched_at": diagnostic_snapshot.fetched_at,
                    "catalog_expires_at": diagnostic_snapshot.expires_at,
                    "model": null,
                    "reasoning_effort": null,
                    "context_window": null,
                    "auto_compact_token_limit": null,
                    "error": {
                        "type": provider_error_type(&error),
                        "message": health_error_message(&error),
                    },
                })),
            )
                .into_response();
        }
        Err(_) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "status": "error",
                    "auth_available": auth_available,
                    "catalog_status": "fresh",
                    "catalog_fetched_at": diagnostic_snapshot.fetched_at,
                    "catalog_expires_at": diagnostic_snapshot.expires_at,
                    "model": null,
                    "reasoning_effort": null,
                    "context_window": null,
                    "auto_compact_token_limit": null,
                    "error": {
                        "type": "server_error",
                        "message": "health preflight failed",
                    },
                })),
            )
                .into_response();
        }
    };
    let context_window = match context_window_for_model(&state, &resolved.model) {
        Ok(value) => value,
        Err(error) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "status": "error",
                    "auth_available": auth_available,
                    "catalog_status": "fresh",
                    "catalog_fetched_at": resolved.snapshot.fetched_at,
                    "catalog_expires_at": resolved.snapshot.expires_at,
                    "model": null,
                    "reasoning_effort": null,
                    "context_window": null,
                    "auto_compact_token_limit": null,
                    "error": {
                        "type": provider_error_type(&error),
                        "message": health_error_message(&error),
                    },
                })),
            )
                .into_response();
        }
    };
    let auto_compact_token_limit = match auto_compact_token_limit_for_model(&state, &resolved.model)
    {
        Ok(value) => value,
        Err(error) => {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({
                    "status": "error",
                    "auth_available": auth_available,
                    "catalog_status": "fresh",
                    "catalog_fetched_at": resolved.snapshot.fetched_at,
                    "catalog_expires_at": resolved.snapshot.expires_at,
                    "model": null,
                    "reasoning_effort": null,
                    "context_window": null,
                    "auto_compact_token_limit": null,
                    "error": {
                        "type": provider_error_type(&error),
                        "message": health_error_message(&error),
                    },
                })),
            )
                .into_response();
        }
    };
    let reasoning_effort =
        match effective_reasoning_effort_with_options(&state, None, None, &resolved.model) {
            Ok(value) => value,
            Err(error) => {
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(json!({
                        "status": "error",
                        "auth_available": auth_available,
                        "catalog_status": "fresh",
                        "catalog_fetched_at": resolved.snapshot.fetched_at,
                        "catalog_expires_at": resolved.snapshot.expires_at,
                        "model": null,
                        "reasoning_effort": null,
                        "context_window": null,
                        "auto_compact_token_limit": null,
                        "error": {
                            "type": provider_error_type(&error),
                            "message": health_error_message(&error),
                        },
                    })),
                )
                    .into_response()
            }
        };
    Json(json!({
        "status": "ok",
        "auth_available": auth_available,
        "model": resolved.model.slug,
        "reasoning_effort": reasoning_effort,
        "context_window": context_window,
        "auto_compact_token_limit": auto_compact_token_limit,
        "catalog_status": "fresh",
        "catalog_fetched_at": resolved.snapshot.fetched_at,
        "catalog_expires_at": resolved.snapshot.expires_at,
    }))
    .into_response()
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<axum::response::Response, axum::response::Response> {
    let mut body = openai_json(payload)?;
    reject_unsupported_proxy_extension_presence(&body, false)
        .map_err(|error| provider_error_response(&error))?;
    reject_explicit_null_fields(
        &body,
        &[
            "messages",
            "model",
            "function_call",
            "functions",
            "parallel_tool_calls",
            "prompt_cache_key",
            "response_format",
            "safety_identifier",
            "tool_choice",
            "tools",
            "user",
            "web_search_options",
        ],
    )
    .map_err(|error| provider_error_response(&error))?;
    validate_openai_chat_message_fields(&body).map_err(|error| provider_error_response(&error))?;
    normalize_assistant_optional_content(&mut body);
    let request: ChatCompletionRequest = parse_openai_request(body)?;
    request
        .responses_lite
        .reject_explicit_null("responses_lite")
        .map_err(|error| provider_error_response(&error))?;
    let responses_lite = request.responses_lite.value.clone();
    reject_unsupported_openai_controls(&request)
        .map_err(|error| provider_error_response(&error))?;
    reject_unsupported_generation_tool_features(
        request.tools.as_deref(),
        request.programmatic_tool_calling.as_ref(),
        false,
    )
    .map_err(|error| provider_error_response(&error))?;
    let tool_choice = parse_openai_tool_choice(request.tool_choice.as_ref())
        .map_err(|error| provider_error_response(&error))?;
    let messages = request_messages_to_internal(&request.messages)
        .map_err(|error| provider_error_response(&error))?;
    let tools = parse_tools(&request.tools).map_err(|error| provider_error_response(&error))?;
    let stop: Option<Vec<String>> = None;
    let max_tokens = None;

    let header_subagent = optional_header_string(&headers, "x-openai-subagent")
        .map_err(|error| provider_error_response(&error))?;
    let subagent =
        merge_optional_nonempty_string(request.subagent.clone(), header_subagent, "subagent")
            .map_err(|error| provider_error_response(&error))?;
    let header_memgen = optional_header_bool(&headers, "x-openai-memgen-request")
        .map_err(|error| provider_error_response(&error))?;
    let memgen_request =
        merge_optional_bool(request.memgen_request, header_memgen, "memgen_request")
            .map_err(|error| provider_error_response(&error))?;

    let previous_response_id = request.previous_response_id.clone();
    let controls = generation_controls(
        request.reasoning.clone(),
        request.safety_identifier.clone(),
        request.prompt_cache_options.clone(),
        request.verbosity.clone(),
        request.multi_agent.as_ref(),
    )
    .map_err(|error| provider_error_response(&error))?;
    let resolved = resolve_route_model(&state, request.model.clone()).await?;
    let request_model = resolved.model.slug.clone();
    let reasoning_effort = effective_reasoning_effort_with_options(
        &state,
        request.reasoning_effort.as_deref(),
        request.reasoning.as_ref(),
        &resolved.model,
    )
    .map_err(|error| provider_error_response(&error))?;

    if request.stream.unwrap_or(false) {
        let provider = state.provider.clone();
        let model_id = request_model;
        let request_id = format!(
            "chatcmpl-{}",
            &uuid::Uuid::new_v4().simple().to_string()[..24]
        );
        let created = chrono::Utc::now().timestamp();
        let temperature = request.temperature;
        let prompt_cache_key = request.prompt_cache_key.clone();
        let tool_choice = tool_choice.clone();
        let service_tier = request.service_tier.clone();
        let text = request.text.clone();
        let client_metadata = request.client_metadata.clone();
        let codex_metadata = request.codex_metadata;
        let responses_lite = responses_lite.clone();
        let parallel_tool_calls = request.parallel_tool_calls;

        let provider_for_prepare = provider.clone();
        let prepared = task::spawn_blocking(move || {
            provider_for_prepare.prepare_chat_stream_for_resolved_model_with_controls(
                &messages,
                tools.as_deref(),
                temperature,
                reasoning_effort.as_deref(),
                max_tokens,
                stop.as_deref(),
                prompt_cache_key.as_deref(),
                subagent.as_deref(),
                memgen_request,
                previous_response_id.as_deref(),
                resolved,
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
        .map_err(join_error_response)?;
        let prepared = prepared.map_err(|error| provider_error_response(&error))?;
        let cancellation = Arc::new(AtomicBool::new(false));
        let prepared = prepared.cancel_when(cancellation.clone());

        let (sender, receiver) = tokio::sync::mpsc::channel::<Event>(32);
        let mut stream_state = OpenAiStreamState {
            request_id,
            created,
            model: model_id,
            tool_call_index: 0,
            finished: false,
        };
        sender
            .send(
                stream_state
                    .preamble()
                    .map_err(|error| provider_error_response(&error))?,
            )
            .await
            .map_err(|error| {
                provider_error_response(&ProviderError::Request(format!(
                    "failed to initialize OpenAI SSE stream: {error}"
                )))
            })?;

        let worker_sender = sender.clone();
        let worker_cancellation = cancellation.clone();
        let worker = task::spawn_blocking(move || {
            provider.stream_prepared_chat(prepared, |event| {
                for chunk in stream_state.push(&event)? {
                    send_sse_event_with_backpressure(
                        &worker_sender,
                        chunk,
                        &worker_cancellation,
                        "OpenAI SSE client disconnected",
                    )?;
                }
                Ok(())
            })?;
            if !stream_state.finished {
                return Err(ProviderError::UpstreamProtocol(
                    "normalized response stream ended without a finish event".to_string(),
                ));
            }
            send_sse_event_with_backpressure(
                &worker_sender,
                Event::default().data("[DONE]"),
                &worker_cancellation,
                "OpenAI SSE client disconnected",
            )
        });

        tokio::spawn(async move {
            let error = match worker.await {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some(error),
                Err(error) => Some(ProviderError::Request(format!(
                    "OpenAI SSE worker failed: {error}"
                ))),
            };
            if let Some(error) = error {
                let _ = sender.send(openai_stream_error_event(&error)).await;
            }
        });

        let event_stream = stream::unfold(
            DownstreamEventReceiver {
                receiver,
                cancellation,
            },
            |mut downstream| async move {
                downstream
                    .receiver
                    .recv()
                    .await
                    .map(|event| (Ok::<Event, std::convert::Infallible>(event), downstream))
            },
        );
        Ok(Sse::new(event_stream).into_response())
    } else {
        let provider = state.provider.clone();
        let model_id = request_model.clone();
        let temperature = request.temperature;
        let prompt_cache_key = request.prompt_cache_key.clone();
        let service_tier = request.service_tier.clone();
        let text = request.text.clone();
        let client_metadata = request.client_metadata.clone();
        let codex_metadata = request.codex_metadata;
        let responses_lite = responses_lite.clone();
        let parallel_tool_calls = request.parallel_tool_calls;

        let result = task::spawn_blocking(move || {
            let tools_ref = tools.as_deref();
            let stop_ref: Option<Vec<String>> = stop;
            let stop_slice = stop_ref.as_deref();

            let prepared = provider.prepare_chat_stream_for_resolved_model_with_controls(
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
                resolved,
                tool_choice.as_ref(),
                service_tier.as_deref(),
                text.as_ref(),
                client_metadata.as_ref(),
                codex_metadata,
                responses_lite.as_ref(),
                parallel_tool_calls,
                &controls,
            )?;
            provider.chat_prepared(prepared)
        })
        .await
        .map_err(join_error_response)?;

        let response = match result {
            Ok(resp) => resp,
            Err(e) => {
                return Err(provider_error_response(&e));
            }
        };
        if response
            .raw
            .as_ref()
            .and_then(|raw| raw.get("events"))
            .and_then(Value::as_array)
            .is_some_and(|events| {
                events.iter().any(|event| {
                    event.get("type").and_then(Value::as_str) == Some("web_search_call")
                })
            })
        {
            return Err(provider_error_response(&ProviderError::UpstreamProtocol(
                "provider web_search_call event cannot be represented by /v1/chat/completions"
                    .to_string(),
            )));
        }

        let mut message_obj = json!({
            "role": "assistant",
            "content": response.content,
            "refusal": null,
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
                            "arguments": tc.arguments,
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

        let response_id = response.response_id.as_deref().ok_or_else(|| {
            provider_error_response(&ProviderError::UpstreamProtocol(
                "normalized completion response requires response_id".to_string(),
            ))
        })?;
        if !matches!(
            response.finish_reason.as_deref(),
            Some("stop" | "tool_calls")
        ) {
            return Err(provider_error_response(&ProviderError::UpstreamProtocol(
                "normalized completion requires a final finish_reason".to_string(),
            )));
        }
        let mut result_obj = json!({
            "id": format!("chatcmpl-{}", &uuid::Uuid::new_v4().simple().to_string()[..24]),
            "object": "chat.completion",
            "created": chrono::Utc::now().timestamp(),
            "model": model_id,
            "response_id": response_id,
            "choices": [{
                "index": 0,
                "message": message_obj,
                "finish_reason": response.finish_reason,
                "logprobs": null,
            }],
        });

        if let Some(usage) = response.usage.as_ref() {
            result_obj
                .as_object_mut()
                .expect("completion response literal must be an object")
                .insert("usage".to_string(), openai_usage_value(usage));
        }

        Ok(Json(result_obj).into_response())
    }
}

async fn images_generations(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, axum::response::Response> {
    let body = openai_json(payload)?;
    reject_unsupported_proxy_extension_presence(&body, false)
        .map_err(|error| provider_error_response(&error))?;
    reject_explicit_null_fields(&body, &["prompt", "user"])
        .map_err(|error| provider_error_response(&error))?;
    if body
        .as_object()
        .is_some_and(|object| object.contains_key("tools"))
    {
        return Err(provider_error_response(&ProviderError::InvalidRequest(
            "tools are not supported by the image generation endpoint".to_string(),
        )));
    }
    let request: ImageGenerationRequest = parse_openai_request(body)?;
    request
        .responses_lite
        .reject_explicit_null("responses_lite")
        .map_err(|error| provider_error_response(&error))?;
    for (name, present) in [
        ("background", request.background.is_some()),
        ("moderation", request.moderation.is_some()),
        ("n", request.n.is_some()),
        ("output_compression", request.output_compression.is_some()),
        ("output_format", request.output_format.is_some()),
        ("partial_images", request.partial_images.is_some()),
        ("quality", request.quality.is_some()),
        ("response_format", request.response_format.is_some()),
        ("stream", request.stream.is_some()),
        ("style", request.style.is_some()),
        ("user", request.user.is_some()),
    ] {
        if present {
            return Err(provider_error_response(&ProviderError::InvalidRequest(
                format!("{name} is not supported by the Codex OAuth HTTP transport"),
            )));
        }
    }
    if request.prompt.trim().is_empty() {
        return Err(provider_error_response(&ProviderError::InvalidRequest(
            "prompt must be a non-empty string".to_string(),
        )));
    }
    if request.size.as_deref().is_some_and(|size| size != "auto") {
        return Err(provider_error_response(&ProviderError::InvalidRequest(
            "size and tools are not supported by the image generation facade".to_string(),
        )));
    }
    reject_unsupported_generation_tool_features(
        request.tools.as_deref(),
        request.programmatic_tool_calling.as_ref(),
        false,
    )
    .map_err(|error| provider_error_response(&error))?;
    let reference_images = request.reference_images.clone().unwrap_or_default();
    validate_image_content_values(&reference_images)
        .map_err(|error| provider_error_response(&error))?;
    let prompt = request.prompt.clone();
    let controls = generation_controls(
        request.reasoning.clone(),
        request.safety_identifier.clone(),
        request.prompt_cache_options.clone(),
        request.verbosity.clone(),
        request.multi_agent.as_ref(),
    )
    .map_err(|error| provider_error_response(&error))?;
    let resolved = resolve_route_model(&state, request.model.clone()).await?;
    let reasoning_effort = effective_reasoning_effort_with_options(
        &state,
        request.reasoning_effort.as_deref(),
        request.reasoning.as_ref(),
        &resolved.model,
    )
    .map_err(|error| provider_error_response(&error))?;
    let provider = state.provider.clone();
    let responses_lite = request.responses_lite.value.clone();

    let result = task::spawn_blocking(move || {
        provider.generate_image_for_resolved_model_with_controls(
            &prompt,
            &reference_images,
            None,
            reasoning_effort.as_deref(),
            resolved,
            responses_lite.as_ref(),
            &controls,
        )
    })
    .await
    .map_err(join_error_response)?;

    let images = result.map_err(|error| provider_error_response(&error))?;

    let mut data = Vec::with_capacity(images.len());
    for image in &images {
        let result_url = image.get("result").and_then(Value::as_str).ok_or_else(|| {
            provider_error_response(&ProviderError::UpstreamProtocol(
                "image generation output requires a string result".to_string(),
            ))
        })?;
        let mut generated = serde_json::Map::from_iter([(
            "url".to_string(),
            Value::String(result_url.to_string()),
        )]);
        if let Some(revised_prompt) = image.get("revised_prompt") {
            let revised_prompt = revised_prompt.as_str().ok_or_else(|| {
                provider_error_response(&ProviderError::UpstreamProtocol(
                    "image generation output revised_prompt must be a string when present"
                        .to_string(),
                ))
            })?;
            generated.insert(
                "revised_prompt".to_string(),
                Value::String(revised_prompt.to_string()),
            );
        }
        data.push(Value::Object(generated));
    }

    Ok(Json(json!({
        "created": chrono::Utc::now().timestamp(),
        "data": data,
    })))
}

async fn inspect(
    State(state): State<AppState>,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, axum::response::Response> {
    let body = openai_json(payload)?;
    reject_unsupported_proxy_extension_presence(&body, false)
        .map_err(|error| provider_error_response(&error))?;
    let request: InspectRequest = parse_openai_request(body)?;
    request
        .responses_lite
        .reject_explicit_null("responses_lite")
        .map_err(|error| provider_error_response(&error))?;
    if request
        .tools
        .as_ref()
        .is_some_and(|tools| !tools.is_empty())
    {
        return Err(provider_error_response(&ProviderError::InvalidRequest(
            "tools are not supported by the image inspection facade".to_string(),
        )));
    }
    reject_unsupported_generation_tool_features(
        request.tools.as_deref(),
        request.programmatic_tool_calling.as_ref(),
        false,
    )
    .map_err(|error| provider_error_response(&error))?;
    let prompt = request.prompt.clone().ok_or_else(|| {
        provider_error_response(&ProviderError::InvalidRequest(
            "inspect requires prompt".to_string(),
        ))
    })?;
    if prompt.trim().is_empty() {
        return Err(provider_error_response(&ProviderError::InvalidRequest(
            "inspect prompt must be a non-empty string".to_string(),
        )));
    }
    let images = request.images.clone().ok_or_else(|| {
        provider_error_response(&ProviderError::InvalidRequest(
            "inspect requires images".to_string(),
        ))
    })?;
    if images.is_empty() {
        return Err(provider_error_response(&ProviderError::InvalidRequest(
            "inspect images must be a non-empty array".to_string(),
        )));
    }
    validate_image_content_values(&images).map_err(|error| provider_error_response(&error))?;
    let controls = generation_controls(
        request.reasoning.clone(),
        request.safety_identifier.clone(),
        request.prompt_cache_options.clone(),
        request.verbosity.clone(),
        request.multi_agent.as_ref(),
    )
    .map_err(|error| provider_error_response(&error))?;
    let resolved = resolve_route_model(&state, request.model.clone()).await?;
    let reasoning_effort = effective_reasoning_effort_with_options(
        &state,
        request.reasoning_effort.as_deref(),
        request.reasoning.as_ref(),
        &resolved.model,
    )
    .map_err(|error| provider_error_response(&error))?;
    let provider = state.provider.clone();
    let responses_lite = request.responses_lite.value.clone();

    let result = task::spawn_blocking(move || {
        provider.inspect_image_values_for_resolved_model_with_controls(
            &prompt,
            &images,
            reasoning_effort.as_deref(),
            resolved,
            responses_lite.as_ref(),
            &controls,
        )
    })
    .await
    .map_err(join_error_response)?;

    let content = result.map_err(|error| provider_error_response(&error))?;

    Ok(Json(json!({"content": content})))
}

async fn compact(
    State(state): State<AppState>,
    OriginalUri(uri): OriginalUri,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, axum::response::Response> {
    let force_anthropic = uri.path() == "/v1/messages/compact";
    let body = if force_anthropic {
        anthropic_json(payload)?
    } else {
        openai_json(payload)?
    };
    let respond = |error: &ProviderError| {
        if force_anthropic {
            anthropic_provider_error_response(error)
        } else {
            provider_error_response(error)
        }
    };
    reject_unsupported_proxy_extension_presence(&body, force_anthropic)
        .map_err(|error| respond(&error))?;
    reject_unknown_top_level_fields(
        &body,
        if force_anthropic {
            ANTHROPIC_COMPACT_FIELDS
        } else {
            OPENAI_COMPACT_FIELDS
        },
    )
    .map_err(|error| respond(&error))?;
    if force_anthropic {
        reject_explicit_null_anthropic_fields(&body).map_err(|error| respond(&error))?;
        if body
            .get("max_tokens")
            .and_then(positive_js_safe_integer)
            .is_none()
        {
            return Err(respond(&ProviderError::InvalidRequest(
                "max_tokens must be a positive integer".to_string(),
            )));
        }
        validate_anthropic_compatibility_scope(&body, &headers).map_err(|error| respond(&error))?;
        validate_anthropic_context_management(&body).map_err(|error| respond(&error))?;
        if body
            .get("stop_sequences")
            .is_some_and(|value| !value.is_null())
        {
            return Err(respond(&ProviderError::InvalidRequest(
                "stop_sequences is not supported by the private Codex OAuth compact transport"
                    .to_string(),
            )));
        }
    }
    reject_unsupported_tool_features(&body, force_anthropic).map_err(|error| respond(&error))?;
    for field in ["safety_identifier", "include", "prompt_cache_retention"] {
        if body.get(field).is_some_and(|value| !value.is_null()) {
            let error = ProviderError::InvalidRequest(format!(
                "{field} is not supported by the compact facade"
            ));
            return Err(respond(&error));
        }
    }
    let nested_reasoning_effort = match body.get("reasoning") {
        None | Some(Value::Null) => None,
        Some(Value::Object(reasoning)) => {
            if reasoning.contains_key("mode") || reasoning.contains_key("context") {
                let error = ProviderError::InvalidRequest(
                    "reasoning.mode and reasoning.context are not supported by compact".to_string(),
                );
                return Err(respond(&error));
            }
            if reasoning.keys().any(|key| key != "effort") {
                let error = ProviderError::InvalidRequest(
                    "compact reasoning supports only effort".to_string(),
                );
                return Err(respond(&error));
            }
            match reasoning.get("effort") {
                None | Some(Value::Null) => None,
                Some(Value::String(value)) if !value.is_empty() => Some(value.clone()),
                Some(_) => {
                    let error = ProviderError::InvalidRequest(
                        "reasoning.effort must be a non-empty string when provided".to_string(),
                    );
                    return Err(respond(&error));
                }
            }
        }
        Some(_) => {
            let error = ProviderError::InvalidRequest("reasoning must be an object".to_string());
            return Err(respond(&error));
        }
    };
    if body
        .get("multi_agent")
        .is_some_and(|value| !value.is_null())
    {
        let error = ProviderError::InvalidRequest(
            "multi_agent is not supported by the compact facade".to_string(),
        );
        return Err(respond(&error));
    }
    let (messages, tools, converted_reasoning_effort, converted_text, parallel_tool_calls) =
        messages_from_compact_body(&body, force_anthropic).map_err(|error| respond(&error))?;
    if force_anthropic && parallel_tool_calls == Some(true) {
        let error = ProviderError::InvalidRequest(
            "tool_choice.disable_parallel_tool_use=false cannot be represented by the compact endpoint"
                .to_string(),
        );
        return Err(respond(&error));
    }
    let top_level_reasoning_effort = match body.get("reasoning_effort") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) if !value.is_empty() && value == value.trim() => {
            Some(value.clone())
        }
        Some(_) => {
            let error = ProviderError::InvalidRequest(
                "reasoning_effort must be a non-empty string when provided".to_string(),
            );
            return Err(respond(&error));
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
        return Err(respond(&error));
    }
    let requested_reasoning_effort = first_effort.map(str::to_string);
    let responses_lite = body.get("responses_lite").cloned();
    let previous_response_id =
        optional_string_field(&body, "previous_response_id").map_err(|error| respond(&error))?;
    if previous_response_id.as_deref().is_some_and(str::is_empty) {
        let error = ProviderError::InvalidRequest(
            "previous_response_id must be a non-empty string".to_string(),
        );
        return Err(respond(&error));
    }
    let compact_service_tier = if force_anthropic {
        anthropic_service_tier(&body)
    } else {
        optional_string_field(&body, "service_tier")
    }
    .map_err(|error| respond(&error))?;
    let compact_text = if force_anthropic {
        merge_anthropic_compact_text(body.get("text"), converted_text)
            .map_err(|error| respond(&error))?
    } else {
        body.get("text").filter(|value| !value.is_null()).cloned()
    };
    let compact_controls = CompactControls {
        previous_response_id,
        prompt_cache_key: if force_anthropic {
            anthropic_prompt_cache_key(&body, &headers).map_err(|error| respond(&error))?
        } else {
            optional_string_field(&body, "prompt_cache_key").map_err(|error| respond(&error))?
        },
        prompt_cache_options: body
            .get("prompt_cache_options")
            .filter(|value| !value.is_null())
            .cloned(),
        service_tier: compact_service_tier,
        text: compact_text,
        verbosity: optional_string_field(&body, "verbosity").map_err(|error| respond(&error))?,
    };
    let requested_model = optional_string_field(&body, "model").map_err(|error| respond(&error))?;
    let resolved = if force_anthropic {
        let model = requested_model.as_deref().ok_or_else(|| {
            respond(&ProviderError::InvalidRequest(
                "model is required".to_string(),
            ))
        })?;
        resolve_anthropic_route_model(&state, model)
            .await
            .map_err(|error| respond(&error))?
    } else {
        resolve_route_model(&state, requested_model).await?
    };
    let reasoning_effort = effective_reasoning_effort_with_options(
        &state,
        requested_reasoning_effort.as_deref(),
        None,
        &resolved.model,
    )
    .map_err(|error| respond(&error))?;
    let provider = state.provider.clone();

    let result = task::spawn_blocking(move || {
        provider.compact_messages_for_resolved_model_with_controls(
            &messages,
            tools.as_deref(),
            reasoning_effort.as_deref(),
            resolved,
            responses_lite.as_ref(),
            &compact_controls,
        )
    })
    .await
    .map_err(|error| {
        respond(&ProviderError::Request(format!(
            "blocking provider task failed: {error}"
        )))
    })?;

    let checkpoint = result.map_err(|error| respond(&error))?;

    Ok(Json(json!({"checkpoint": checkpoint})))
}

async fn anthropic_count_tokens(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<Json<Value>, axum::response::Response> {
    let body = anthropic_json(payload)?;
    reject_unsupported_proxy_extension_presence(&body, true)
        .map_err(|error| anthropic_provider_error_response(&error))?;
    reject_explicit_null_anthropic_fields(&body)
        .map_err(|error| anthropic_provider_error_response(&error))?;
    reject_unknown_top_level_fields(&body, ANTHROPIC_COUNT_FIELDS)
        .map_err(|error| anthropic_provider_error_response(&error))?;
    reject_unsupported_anthropic_controls(&body)
        .map_err(|error| anthropic_provider_error_response(&error))?;
    if let Some(value) = body.get("max_tokens") {
        if positive_js_safe_integer(value).is_none() {
            return Err(anthropic_provider_error_response(
                &ProviderError::InvalidRequest("max_tokens must be a positive integer".to_string()),
            ));
        }
    }
    validate_anthropic_compatibility_scope(&body, &headers)
        .map_err(|error| anthropic_provider_error_response(&error))?;
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
    if let Err(error) = reject_unsupported_tool_features(&body, true) {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(format_anthropic_error(400, &error.to_string())),
        )
            .into_response());
    }
    let (messages, tools, _tool_choice, _stop, _reasoning_effort, _text, _parallel_tool_calls) =
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
        .filter(|model| !model.trim().is_empty())
        .ok_or_else(|| {
            anthropic_provider_error_response(&ProviderError::InvalidRequest(
                "model is required".to_string(),
            ))
        })?;
    let resolved = resolve_anthropic_route_model(&state, request_model)
        .await
        .map_err(|error| anthropic_provider_error_response(&error))?;
    let context_window = context_window_for_model(&state, &resolved.model)
        .map_err(|error| anthropic_provider_error_response(&error))?;
    let auto_compact_token_limit = if context_window.is_some() {
        auto_compact_token_limit_for_model(&state, &resolved.model)
            .map_err(|error| anthropic_provider_error_response(&error))?
    } else {
        None
    };
    Ok(Json(json!({
        "input_tokens": input_tokens,
        "context_window": context_window,
        "auto_compact_token_limit": auto_compact_token_limit,
    })))
}

async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Result<Json<Value>, JsonRejection>,
) -> Result<axum::response::Response, axum::response::Response> {
    let body = anthropic_json(payload)?;
    reject_unsupported_proxy_extension_presence(&body, true)
        .map_err(|error| anthropic_provider_error_response(&error))?;
    reject_explicit_null_anthropic_fields(&body)
        .map_err(|error| anthropic_provider_error_response(&error))?;
    reject_unknown_top_level_fields(&body, ANTHROPIC_MESSAGE_FIELDS)
        .map_err(|error| anthropic_provider_error_response(&error))?;
    let request_id = format!("msg_{}", &uuid::Uuid::new_v4().simple().to_string()[..24]);
    let client_model = body
        .get("model")
        .and_then(Value::as_str)
        .filter(|model| !model.trim().is_empty())
        .ok_or_else(|| {
            anthropic_provider_error_response(&ProviderError::InvalidRequest(
                "model is required".to_string(),
            ))
        })?
        .to_string();
    let stream = match body.get("stream") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(_) => {
            return Err(anthropic_provider_error_response(
                &ProviderError::InvalidRequest("stream must be a boolean".to_string()),
            ));
        }
    };
    match body.get("max_tokens") {
        Some(value) if positive_js_safe_integer(value).is_some() => {}
        _ => {
            return Err(anthropic_provider_error_response(
                &ProviderError::InvalidRequest("max_tokens must be a positive integer".to_string()),
            ));
        }
    }
    validate_anthropic_compatibility_scope(&body, &headers)
        .map_err(|error| anthropic_provider_error_response(&error))?;
    reject_unsupported_anthropic_controls(&body)
        .map_err(|error| anthropic_provider_error_response(&error))?;
    validate_anthropic_context_management(&body).map_err(|error| {
        (
            StatusCode::BAD_REQUEST,
            Json(format_anthropic_error(400, &error.to_string())),
        )
            .into_response()
    })?;
    if body
        .get("previous_response_id")
        .is_some_and(|value| !value.is_null())
    {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(format_anthropic_error(
                400,
                "previous_response_id is not supported by /v1/messages",
            )),
        )
            .into_response());
    }
    reject_unsupported_tool_features(&body, true)
        .map_err(|error| anthropic_provider_error_response(&error))?;

    let body_subagent = optional_string_field(&body, "subagent")
        .map_err(|error| anthropic_provider_error_response(&error))?;
    let header_subagent = optional_header_string(&headers, "x-openai-subagent")
        .map_err(|error| anthropic_provider_error_response(&error))?;
    let subagent = merge_optional_nonempty_string(body_subagent, header_subagent, "subagent")
        .map_err(|error| anthropic_provider_error_response(&error))?;
    let body_memgen = match body.get("memgen_request") {
        None | Some(Value::Null) => None,
        Some(Value::Bool(value)) => Some(*value),
        Some(_) => {
            return Err(anthropic_provider_error_response(
                &ProviderError::InvalidRequest("memgen_request must be a boolean".to_string()),
            ));
        }
    };
    let header_memgen = optional_header_bool(&headers, "x-openai-memgen-request")
        .map_err(|error| anthropic_provider_error_response(&error))?;
    let memgen_request = merge_optional_bool(body_memgen, header_memgen, "memgen_request")
        .map_err(|error| anthropic_provider_error_response(&error))?;

    let (messages, tools, tool_choice, stop, converted_reasoning_effort, text, parallel_tool_calls) =
        anthropic_request_to_internal(&body).map_err(|message| {
            (
                StatusCode::BAD_REQUEST,
                Json(format_anthropic_error(400, &message)),
            )
                .into_response()
        })?;

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
    let service_tier = anthropic_service_tier(&body).map_err(|error| {
        (
            StatusCode::BAD_REQUEST,
            Json(format_anthropic_error(400, &error.to_string())),
        )
            .into_response()
    })?;
    let prompt_cache_key = anthropic_prompt_cache_key(&body, &headers).map_err(|error| {
        (
            StatusCode::BAD_REQUEST,
            Json(format_anthropic_error(400, &error.to_string())),
        )
            .into_response()
    })?;
    let controls = generation_controls(
        reasoning.clone(),
        optional_string_field(&body, "safety_identifier")
            .map_err(|error| anthropic_provider_error_response(&error))?,
        body.get("prompt_cache_options")
            .filter(|value| !value.is_null())
            .cloned(),
        optional_string_field(&body, "verbosity")
            .map_err(|error| anthropic_provider_error_response(&error))?,
        body.get("multi_agent"),
    )
    .map_err(|error| anthropic_provider_error_response(&error))?;

    let resolved = resolve_anthropic_route_model(&state, &client_model)
        .await
        .map_err(|error| anthropic_provider_error_response(&error))?;
    let reasoning_effort = effective_reasoning_effort_with_options(
        &state,
        requested_reasoning_effort.as_deref(),
        reasoning.as_ref(),
        &resolved.model,
    )
    .map_err(|error| anthropic_provider_error_response(&error))?;

    let max_tokens = None;

    let tool_choice_val: Option<Value> = tool_choice;
    let text_val: Option<Value> = text;
    let responses_lite_val = body.get("responses_lite").cloned();

    if stream {
        let provider = state.provider.clone();

        let prepared = task::spawn_blocking({
            let provider = provider.clone();
            move || {
                let tools_ref = tools.as_deref();
                let stop_ref = stop.as_deref();
                let tc_ref = tool_choice_val.as_ref();
                let text_ref = text_val.as_ref();

                provider.prepare_chat_stream_for_resolved_model_with_controls(
                    &messages,
                    tools_ref,
                    None,
                    reasoning_effort.as_deref(),
                    max_tokens,
                    stop_ref,
                    prompt_cache_key.as_deref(),
                    subagent.as_deref(),
                    memgen_request,
                    None,
                    resolved,
                    tc_ref,
                    service_tier.as_deref(),
                    text_ref,
                    None,
                    Some(false),
                    responses_lite_val.as_ref(),
                    parallel_tool_calls,
                    &controls,
                )
            }
        })
        .await
        .map_err(join_error_response)?;

        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                let status_code = match map_error_status(&error) {
                    StatusCode::UNAUTHORIZED => 401u16,
                    StatusCode::INTERNAL_SERVER_ERROR => 500u16,
                    status => status.as_u16(),
                };
                let body = format_anthropic_error(status_code, &public_error_message(&error));
                return Err((
                    StatusCode::from_u16(status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(body),
                )
                    .into_response());
            }
        };

        let cancellation = Arc::new(AtomicBool::new(false));
        let prepared = prepared.cancel_when(cancellation.clone());
        let (sender, receiver) = tokio::sync::mpsc::channel::<Event>(32);
        let worker_sender = sender.clone();
        let worker_cancellation = cancellation.clone();
        let worker = task::spawn_blocking(move || {
            let mut adapter = AnthropicStreamAdapter::new(&client_model, &request_id);
            for chunk in adapter.start() {
                send_sse_event_with_backpressure(
                    &worker_sender,
                    anthropic_sse_event(&chunk),
                    &worker_cancellation,
                    "Anthropic SSE client disconnected",
                )?;
            }

            provider.stream_prepared_chat(prepared, |event| {
                let chunks = adapter
                    .push(&event)
                    .map_err(ProviderError::UpstreamProtocol)?;
                for chunk in chunks {
                    send_sse_event_with_backpressure(
                        &worker_sender,
                        anthropic_sse_event(&chunk),
                        &worker_cancellation,
                        "Anthropic SSE client disconnected",
                    )?;
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
                        &public_error_message(&error),
                    ))
                    .await;
            }
        });

        let event_stream = stream::unfold(
            DownstreamEventReceiver {
                receiver,
                cancellation,
            },
            |mut downstream| async move {
                downstream
                    .receiver
                    .recv()
                    .await
                    .map(|event| (Ok::<Event, std::convert::Infallible>(event), downstream))
            },
        );
        Ok(Sse::new(event_stream).into_response())
    } else {
        let provider = state.provider.clone();
        let result = task::spawn_blocking(move || {
            let tools_ref = tools.as_deref();
            let stop_ref = stop.as_deref();
            let tc_ref = tool_choice_val.as_ref();
            let text_ref = text_val.as_ref();

            let prepared = provider.prepare_chat_stream_for_resolved_model_with_controls(
                &messages,
                tools_ref,
                None,
                reasoning_effort.as_deref(),
                max_tokens,
                stop_ref,
                prompt_cache_key.as_deref(),
                subagent.as_deref(),
                memgen_request,
                None,
                resolved,
                tc_ref,
                service_tier.as_deref(),
                text_ref,
                None,
                Some(false),
                responses_lite_val.as_ref(),
                parallel_tool_calls,
                &controls,
            )?;
            provider.chat_prepared(prepared)
        })
        .await
        .map_err(join_error_response)?;

        let response = match result {
            Ok(response) => response,
            Err(e) => {
                let status_code = match map_error_status(&e) {
                    StatusCode::UNAUTHORIZED => 401u16,
                    StatusCode::INTERNAL_SERVER_ERROR => 500u16,
                    s => s.as_u16(),
                };
                let body = format_anthropic_error(status_code, &public_error_message(&e));
                return Err((
                    StatusCode::from_u16(status_code).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
                    Json(body),
                )
                    .into_response());
            }
        };

        let out = internal_response_to_anthropic(&response, &client_model, &request_id).map_err(
            |message| anthropic_provider_error_response(&ProviderError::UpstreamProtocol(message)),
        )?;
        Ok(Json(out).into_response())
    }
}

#[cfg(test)]
mod live_catalog_tests {
    use super::*;
    use axum::extract::Query;
    use axum::http::header::CONTENT_TYPE;
    use base64::Engine;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicU16, AtomicU64, Ordering};
    use std::sync::Mutex;
    use tokio::task::JoinHandle;

    type RecordedModelRequests = Vec<(HeaderMap, HashMap<String, String>)>;

    fn upstream_contract() -> Value {
        serde_json::from_str(include_str!("../../config/codex-upstream-contract.json"))
            .expect("Codex upstream contract fixture must be valid JSON")
    }

    #[test]
    fn openai_function_tool_preserves_strict_mode() {
        let tools = parse_tools(&Some(vec![json!({
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup",
                "parameters": {"type": "object", "properties": {}},
                "strict": true
            }
        })]))
        .unwrap()
        .unwrap();
        assert!(tools[0].strict);
    }

    #[test]
    fn assistant_tool_call_arguments_preserve_raw_strings() {
        let calls = parse_tool_calls(
            &Some(vec![json!({
                "type": "function",
                "id": "call_1",
                "function": {
                    "name": "lookup",
                    "arguments": " not-yet-valid "
                }
            })]),
            0,
        )
        .unwrap();
        assert_eq!(calls[0].arguments, " not-yet-valid ");

        let calls = parse_tool_calls(
            &Some(vec![json!({
                "type": "function",
                "id": "call_1",
                "function": {
                    "name": "lookup",
                    "arguments": "{\"b\":2, \"a\":1}"
                }
            })]),
            0,
        )
        .unwrap();
        assert_eq!(calls[0].arguments, "{\"b\":2, \"a\":1}");
    }

    #[test]
    fn health_error_messages_never_expose_internal_error_details() {
        let cases = [
            (
                ProviderError::Auth(crate::auth::AuthError::OAuth(
                    "/secret/auth.json".to_string(),
                )),
                "ChatGPT OAuth credentials are unavailable",
            ),
            (
                ProviderError::CatalogUnavailable("secret catalog body".to_string()),
                "authenticated model catalog is unavailable",
            ),
            (
                ProviderError::ModelNotFound("private-model".to_string()),
                "configured model is unavailable in the authenticated catalog",
            ),
            (
                ProviderError::UpstreamHttp {
                    status: 500,
                    message: "secret upstream body".to_string(),
                },
                "upstream request failed",
            ),
            (
                ProviderError::UpstreamTransport("secret transport detail".to_string()),
                "upstream request failed",
            ),
            (
                ProviderError::UpstreamProtocol("secret payload".to_string()),
                "upstream protocol validation failed",
            ),
            (
                ProviderError::InvalidRequest("secret config".to_string()),
                "health configuration is invalid",
            ),
            (
                ProviderError::Request("secret task detail".to_string()),
                "health preflight failed",
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(health_error_message(&error), expected);
        }
    }

    #[derive(Clone)]
    struct MockState {
        catalog: Arc<Mutex<Value>>,
        catalog_etag: Arc<Mutex<String>>,
        malformed_response: bool,
        model_delay_millis: Arc<AtomicU64>,
        response_models_etag: Arc<Mutex<Option<String>>>,
        unauthorized_models_once: Arc<AtomicBool>,
        unauthorized_models_always: Arc<AtomicBool>,
        models_status: Arc<AtomicU16>,
        replacement_auth_path: Arc<Mutex<Option<PathBuf>>>,
        model_requests: Arc<Mutex<RecordedModelRequests>>,
        response_requests: Arc<Mutex<Vec<Value>>>,
    }

    impl MockState {
        fn new(catalog: Value) -> Self {
            Self {
                catalog: Arc::new(Mutex::new(catalog)),
                catalog_etag: Arc::new(Mutex::new("catalog-v1".to_string())),
                malformed_response: false,
                model_delay_millis: Arc::new(AtomicU64::new(0)),
                response_models_etag: Arc::new(Mutex::new(None)),
                unauthorized_models_once: Arc::new(AtomicBool::new(false)),
                unauthorized_models_always: Arc::new(AtomicBool::new(false)),
                models_status: Arc::new(AtomicU16::new(200)),
                replacement_auth_path: Arc::new(Mutex::new(None)),
                model_requests: Arc::new(Mutex::new(Vec::new())),
                response_requests: Arc::new(Mutex::new(Vec::new())),
            }
        }
    }

    fn live_model(slug: &str, priority: i64, supported_in_api: bool) -> Value {
        json!({
            "slug": slug,
            "display_name": slug,
            "description": "live test model",
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [
                {"effort": "low", "description": "low"},
                {"effort": "medium", "description": "medium"},
                {"effort": "high", "description": "high"}
            ],
            "visibility": "list",
            "supported_in_api": supported_in_api,
            "priority": priority,
            "service_tiers": [{"id":"priority","name":"Priority","description":"Faster service"}],
            "default_service_tier": "priority",
            "support_verbosity": true,
            "default_verbosity": "medium",
            "supports_image_detail_original": true,
            "context_window": 100000,
            "max_context_window": 120000,
            "auto_compact_token_limit": null,
            "input_modalities": ["text", "image"],
            "use_responses_lite": false,
            "supports_reasoning_summaries": true,
            "available_in_plans": ["plus"],
            "prefer_websockets": false,
            "requires_sandboxed_review": false,
            "minimal_client_version": "0.153.3",
            "base_instructions": "Never expose this.",
            "model_messages": {"input": "Never expose this either."}
        })
    }

    async fn mock_models(
        State(state): State<MockState>,
        headers: HeaderMap,
        Query(query): Query<HashMap<String, String>>,
    ) -> axum::response::Response {
        state.model_requests.lock().unwrap().push((headers, query));
        let delay = state.model_delay_millis.load(Ordering::SeqCst);
        if delay > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
        }
        if state.unauthorized_models_once.swap(false, Ordering::SeqCst) {
            let path = state
                .replacement_auth_path
                .lock()
                .unwrap()
                .clone()
                .expect("refresh test must configure an auth path");
            write_auth_at(&path, "account-id", "refreshed-access-token");
            return StatusCode::UNAUTHORIZED.into_response();
        }
        if state.unauthorized_models_always.load(Ordering::SeqCst) {
            return StatusCode::UNAUTHORIZED.into_response();
        }
        let status = state.models_status.load(Ordering::SeqCst);
        if status != 200 {
            return (
                StatusCode::from_u16(status).expect("test model status must be valid"),
                "upstream catalog detail",
            )
                .into_response();
        }
        let catalog = state.catalog.lock().unwrap().clone();
        let etag = state.catalog_etag.lock().unwrap().clone();
        let mut response = Json(catalog).into_response();
        let contract = upstream_contract();
        let etag_header = axum::http::HeaderName::from_bytes(
            contract["models_request"]["etag_header"]
                .as_str()
                .expect("models_request.etag_header must be a string")
                .as_bytes(),
        )
        .expect("models_request.etag_header must be a valid header name");
        response.headers_mut().insert(
            etag_header,
            axum::http::HeaderValue::from_str(&etag).expect("test ETag must be valid"),
        );
        response
    }

    async fn mock_responses(
        State(state): State<MockState>,
        Json(body): Json<Value>,
    ) -> axum::response::Response {
        state.response_requests.lock().unwrap().push(body);
        let output = if state.malformed_response {
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": 42}]
                }
            })
        } else {
            json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "ok"}]
                }
            })
        };
        let completed = json!({
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "end_turn": true,
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0}
                }
            }
        });
        let mut response = (
            [(CONTENT_TYPE, "text/event-stream")],
            format!("data: {output}\n\ndata: {completed}\n\n"),
        )
            .into_response();
        if let Some(etag) = state.response_models_etag.lock().unwrap().as_deref() {
            let contract = upstream_contract();
            let responses_etag_header = axum::http::HeaderName::from_bytes(
                contract["models_request"]["responses_etag_header"]
                    .as_str()
                    .expect("models_request.responses_etag_header must be a string")
                    .as_bytes(),
            )
            .expect("models_request.responses_etag_header must be a valid header name");
            response.headers_mut().insert(
                responses_etag_header,
                axum::http::HeaderValue::from_str(etag).expect("test ETag must be valid"),
            );
        }
        response
    }

    async fn start_mock_upstream(state: MockState) -> (String, JoinHandle<()>) {
        let contract = upstream_contract();
        let models_contract = &contract["models_request"];
        assert_eq!(models_contract["method"], "GET");
        let models_path = models_contract["path"]
            .as_str()
            .expect("models_request.path must be a string");
        let app = Router::new()
            .route(models_path, get(mock_models))
            .route("/responses", post(mock_responses))
            .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), handle)
    }

    fn test_access_token(account_id: &str, marker: &str) -> String {
        let claims = json!({
            "exp": 9999999999i64,
            "marker": marker,
            "https://api.openai.com/auth": {"chatgpt_account_id": account_id}
        });
        format!(
            "header.{}.signature",
            base64::engine::general_purpose::URL_SAFE_NO_PAD
                .encode(serde_json::to_vec(&claims).unwrap())
        )
    }

    fn write_auth_at(path: &std::path::Path, account_id: &str, access_marker: &str) {
        let access_token = test_access_token(account_id, access_marker);
        let auth = json!({
            "auth_mode": "chatgpt",
            "tokens": {
                "access_token": access_token,
                "refresh_token": "refresh-token",
                "id_token": "header.e30.signature",
                "account_id": account_id
            }
        });
        std::fs::write(path, serde_json::to_vec(&auth).unwrap()).unwrap();
    }

    fn write_auth_file(account_id: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "codex-as-api-live-catalog-{}.json",
            uuid::Uuid::new_v4()
        ));
        write_auth_at(&path, account_id, "header.e30.signature");
        path
    }

    async fn start_api(
        upstream: &str,
        configured_model: &str,
    ) -> (String, PathBuf, JoinHandle<()>) {
        start_api_with_catalog_ttl(
            upstream,
            configured_model,
            crate::model_catalog::DEFAULT_CATALOG_TTL,
        )
        .await
    }

    async fn start_api_with_catalog_ttl(
        upstream: &str,
        configured_model: &str,
        catalog_ttl: std::time::Duration,
    ) -> (String, PathBuf, JoinHandle<()>) {
        let auth_path = write_auth_file("account-id");
        let auth_path_string = auth_path.to_string_lossy().to_string();
        let provider = Arc::new(
            ChatGPTOAuthProvider::new_with_catalog_ttl(
                configured_model.to_string(),
                upstream.to_string(),
                Some(auth_path_string.clone()),
                Some(std::time::Duration::from_secs(5)),
                catalog_ttl,
            )
            .unwrap(),
        );
        let config = CodexConfig {
            model: None,
            model_reasoning_effort: None,
            model_context_window: None,
            model_auto_compact_token_limit: None,
        };
        let app = create_router(AppState {
            auth_path: Some(auth_path_string),
            codex_config: config,
            provider,
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), auth_path, handle)
    }

    #[tokio::test]
    async fn models_and_chat_use_authenticated_live_catalog_and_raw_slug() {
        let mut hidden_model = live_model("hidden-api-model", 2, false);
        hidden_model["comp_hash"] = json!("compatibility-family");
        let state = MockState::new(json!({
            "models": [hidden_model, live_model("live-model", 1, true)]
        }));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let client = reqwest::Client::new();

        let models_response = client.get(format!("{api}/v1/models")).send().await.unwrap();
        assert_eq!(models_response.status(), reqwest::StatusCode::OK);
        assert!(models_response.headers().get("etag").is_none());
        let models_body: Value = models_response.json().await.unwrap();
        assert_eq!(models_body["data"].as_array().unwrap().len(), 2);
        assert!(models_body["data"][0].get("created").is_none());
        assert!(models_body["data"][0].get("base_instructions").is_none());
        assert!(models_body["data"][0].get("model_messages").is_none());
        assert!(models_body["data"][0]
            .get("supports_reasoning_summaries")
            .is_none());
        assert_eq!(
            models_body["data"][0]["supports_reasoning_summary_parameter"],
            true
        );
        assert_eq!(models_body["data"][0]["default_reasoning_summary"], "auto");
        assert_eq!(models_body["data"][0]["comp_hash"], "compatibility-family");
        assert!(models_body["data"][0]["supported_reasoning_levels"][0]
            .get("description")
            .is_some());
        assert_eq!(models_body["data"][0]["service_tiers"][0]["id"], "priority");
        assert_eq!(
            models_body["data"][0]["service_tiers"][0]["name"],
            "Priority"
        );

        let health = client.get(format!("{api}/health")).send().await.unwrap();
        assert_eq!(health.status(), reqwest::StatusCode::OK);
        let health_body: Value = health.json().await.unwrap();
        assert_eq!(health_body["status"], "ok");
        assert_eq!(health_body["catalog_status"], "fresh");
        assert!(health_body.get("catalog_etag").is_none());
        assert_eq!(health_body["model"], "live-model");
        assert_eq!(health_body["reasoning_effort"], "medium");
        assert_eq!(health_body["context_window"], 95000);
        assert_eq!(health_body["auto_compact_token_limit"], 90000);
        assert!(health_body["catalog_fetched_at"].as_str().is_some());
        assert!(health_body["catalog_expires_at"].as_str().is_some());
        assert!(health_body.get("error").is_none());
        assert!(health_body.get("account_id").is_none());
        assert!(health_body.get("token").is_none());
        assert!(health_body.get("effective_context_window").is_none());

        let chat = client
            .post(format!("{api}/v1/chat/completions"))
            .json(&json!({
                "model": "live-model",
                "messages": [
                    {"role": "system", "content": "Answer concisely."},
                    {"role": "user", "content": "hello"}
                ]
            }))
            .send()
            .await
            .unwrap();
        let chat_status = chat.status();
        let chat_text = chat.text().await.unwrap();
        assert_eq!(chat_status, reqwest::StatusCode::OK, "{chat_text}");
        let chat_body: Value = serde_json::from_str(&chat_text).unwrap();
        assert_eq!(chat_body["model"], "live-model");

        let default_chat = client
            .post(format!("{api}/v1/chat/completions"))
            .json(&json!({
                "messages": [
                    {"role": "system", "content": "Answer concisely."},
                    {"role": "user", "content": "hello again"}
                ]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(default_chat.status(), reqwest::StatusCode::OK);
        let default_body: Value = default_chat.json().await.unwrap();
        assert_eq!(default_body["model"], "live-model");

        let model_requests = state.model_requests.lock().unwrap();
        assert_eq!(model_requests.len(), 1);
        let contract = upstream_contract();
        let models_contract = &contract["models_request"];
        let query_name = models_contract["client_version_query"]
            .as_str()
            .expect("models_request.client_version_query must be a string");
        let client_version = contract["upstream"]["version"]
            .as_str()
            .expect("upstream.version must be a string");
        assert_eq!(
            model_requests[0].1.get(query_name).map(String::as_str),
            Some(client_version)
        );
        assert_eq!(
            crate::model_catalog::DEFAULT_CATALOG_TTL.as_secs(),
            models_contract["cache_ttl_seconds"]
                .as_u64()
                .expect("models_request.cache_ttl_seconds must be unsigned")
        );
        assert_eq!(
            crate::provider::MODEL_CATALOG_TIMEOUT.as_secs(),
            models_contract["request_timeout_seconds"]
                .as_u64()
                .expect("models_request.request_timeout_seconds must be unsigned")
        );
        let actual_key = crate::model_catalog::CatalogKey {
            account_id: "account-id".to_string(),
            base_url: upstream.clone(),
            client_version: client_version.to_string(),
        };
        let actual_scope = json!({
            "account_id": actual_key.account_id,
            "base_url": actual_key.base_url,
            "client_version": actual_key.client_version,
        });
        let actual_scope_names: Vec<Value> = actual_scope
            .as_object()
            .expect("actual catalog scope must be an object")
            .keys()
            .cloned()
            .map(Value::String)
            .collect();
        assert_eq!(
            models_contract["cache_scope"],
            Value::Array(actual_scope_names)
        );
        assert_eq!(
            model_requests[0].0.get("chatgpt-account-id").unwrap(),
            "account-id"
        );
        assert_eq!(
            model_requests[0].0.get("originator").unwrap(),
            "codex_cli_rs"
        );
        assert_eq!(
            model_requests[0].0.get("accept").unwrap(),
            "application/json"
        );
        assert!(model_requests[0].0.get("user-agent").is_some());
        drop(model_requests);
        let response_requests = state.response_requests.lock().unwrap();
        assert_eq!(response_requests.len(), 2);
        assert_eq!(response_requests[0]["model"], "live-model");
        assert_eq!(response_requests[1]["model"], "live-model");

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn models_exposes_an_empty_live_catalog_without_default_fallback() {
        let state = MockState::new(json!({"models": []}));
        let (upstream, upstream_handle) = start_mock_upstream(state).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let client = reqwest::Client::new();

        let models = client.get(format!("{api}/v1/models")).send().await.unwrap();
        assert_eq!(models.status(), reqwest::StatusCode::OK);
        assert_eq!(
            models.json::<Value>().await.unwrap(),
            json!({"object": "list", "data": []})
        );
        let health = client.get(format!("{api}/health")).send().await.unwrap();
        assert_eq!(health.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn catalog_unauthorized_refreshes_once_and_retries_with_new_token() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        state.unauthorized_models_once.store(true, Ordering::SeqCst);
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        *state.replacement_auth_path.lock().unwrap() = Some(auth_path.clone());

        let response = reqwest::Client::new()
            .get(format!("{api}/v1/models"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let requests = state.model_requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].0.get("authorization").unwrap(),
            &format!(
                "Bearer {}",
                test_access_token("account-id", "header.e30.signature")
            )
        );
        assert_eq!(
            requests[1].0.get("authorization").unwrap(),
            &format!(
                "Bearer {}",
                test_access_token("account-id", "refreshed-access-token")
            )
        );
        drop(requests);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn catalog_terminal_unauthorized_preserves_upstream_error() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        state.unauthorized_models_once.store(true, Ordering::SeqCst);
        state
            .unauthorized_models_always
            .store(true, Ordering::SeqCst);
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        *state.replacement_auth_path.lock().unwrap() = Some(auth_path.clone());
        let client = reqwest::Client::new();

        let response = client.get(format!("{api}/v1/models")).send().await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], "upstream_error");

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();

        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        state.unauthorized_models_once.store(true, Ordering::SeqCst);
        state
            .unauthorized_models_always
            .store(true, Ordering::SeqCst);
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        *state.replacement_auth_path.lock().unwrap() = Some(auth_path.clone());
        let health = client.get(format!("{api}/health")).send().await.unwrap();
        assert_eq!(health.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let health_body: Value = health.json().await.unwrap();
        assert_eq!(health_body["auth_available"], true);
        assert_eq!(health_body["status"], "error");
        assert_eq!(health_body["catalog_status"], "unavailable");
        assert_eq!(health_body["error"]["type"], "upstream_error");
        assert_eq!(health_body["error"]["message"], "upstream request failed");
        for field in [
            "model",
            "reasoning_effort",
            "context_window",
            "auto_compact_token_limit",
        ] {
            assert!(health_body[field].is_null(), "{field}");
        }

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn catalog_non_auth_http_status_is_preserved_without_leaking_health_details() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        state.models_status.store(429, Ordering::SeqCst);
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let client = reqwest::Client::new();

        let response = client.get(format!("{api}/v1/models")).send().await.unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::TOO_MANY_REQUESTS);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], "upstream_error");

        let health = client.get(format!("{api}/health")).send().await.unwrap();
        assert_eq!(health.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let health_body: Value = health.json().await.unwrap();
        assert_eq!(health_body["status"], "error");
        assert_eq!(health_body["auth_available"], true);
        assert_eq!(health_body["catalog_status"], "unavailable");
        assert_eq!(health_body["error"]["type"], "upstream_error");
        assert_eq!(health_body["error"]["message"], "upstream request failed");
        assert_eq!(state.model_requests.lock().unwrap().len(), 2);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn malformed_input_and_unknown_models_never_reach_responses() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let client = reqwest::Client::new();

        let malformed_json = client
            .post(format!("{api}/v1/chat/completions"))
            .header(CONTENT_TYPE, "application/json")
            .body(r#"{"model":42,"messages":[]}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(malformed_json.status(), reqwest::StatusCode::BAD_REQUEST);
        let malformed_json_body: Value = malformed_json.json().await.unwrap();
        assert_eq!(
            malformed_json_body["error"]["type"],
            "invalid_request_error"
        );
        assert!(state.model_requests.lock().unwrap().is_empty());

        let malformed = client
            .post(format!("{api}/v1/chat/completions"))
            .json(&json!({"model": "live-model", "messages": [{"role": "typo", "content": "x"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(malformed.status(), reqwest::StatusCode::BAD_REQUEST);
        assert!(state.model_requests.lock().unwrap().is_empty());

        let empty_model = client
            .post(format!("{api}/v1/chat/completions"))
            .json(&json!({"model": " ", "messages": [{"role": "user", "content": "x"}]}))
            .send()
            .await
            .unwrap();
        assert_eq!(empty_model.status(), reqwest::StatusCode::BAD_REQUEST);
        assert!(state.model_requests.lock().unwrap().is_empty());

        let unsupported = client
            .post(format!("{api}/v1/chat/completions"))
            .json(&json!({
                "model": "live-model",
                "messages": [{"role": "user", "content": "x"}],
                "n": 2
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(unsupported.status(), reqwest::StatusCode::BAD_REQUEST);
        assert!(state.model_requests.lock().unwrap().is_empty());

        for control in [json!({"temperature": 0.0}), json!({"max_tokens": 128})] {
            let mut request = json!({
                "model": "live-model",
                "messages": [{"role": "user", "content": "x"}]
            });
            request
                .as_object_mut()
                .unwrap()
                .extend(control.as_object().unwrap().clone());
            let response = client
                .post(format!("{api}/v1/chat/completions"))
                .json(&request)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        }
        assert!(state.model_requests.lock().unwrap().is_empty());

        let invalid_header = client
            .post(format!("{api}/v1/chat/completions"))
            .header("x-openai-memgen-request", "1")
            .json(&json!({
                "model": "live-model",
                "messages": [{"role": "user", "content": "x"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(invalid_header.status(), reqwest::StatusCode::BAD_REQUEST);
        assert!(state.model_requests.lock().unwrap().is_empty());

        let anthropic_typo = client
            .post(format!("{api}/v1/messages"))
            .json(&json!({
                "model": "live-model",
                "max_tokens": 128,
                "system": "Answer concisely.",
                "messages": [{"role": "user", "content": "x"}],
                "typo": true
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(anthropic_typo.status(), reqwest::StatusCode::BAD_REQUEST);
        assert!(state.model_requests.lock().unwrap().is_empty());

        let compact_typo = client
            .post(format!("{api}/v1/compact"))
            .json(&json!({"messages": [], "typo": true}))
            .send()
            .await
            .unwrap();
        assert_eq!(compact_typo.status(), reqwest::StatusCode::BAD_REQUEST);
        assert!(state.model_requests.lock().unwrap().is_empty());

        let unconfigured_claude = client
            .post(format!("{api}/v1/messages"))
            .json(&json!({
                "model": "claude-sonnet-current",
                "max_tokens": 128,
                "system": "Answer concisely.",
                "messages": [{"role": "user", "content": "x"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            unconfigured_claude.status(),
            reqwest::StatusCode::BAD_REQUEST
        );
        let unconfigured_body: Value = unconfigured_claude.json().await.unwrap();
        assert_eq!(unconfigured_body["error"]["type"], "invalid_request_error");
        assert!(state.model_requests.lock().unwrap().is_empty());

        let unknown = client
            .post(format!("{api}/v1/chat/completions"))
            .json(&json!({
                "model": "live-model-typo",
                "messages": [
                    {"role": "system", "content": "Answer concisely."},
                    {"role": "user", "content": "x"}
                ]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(unknown.status(), reqwest::StatusCode::NOT_FOUND);
        let body: Value = unknown.json().await.unwrap();
        assert_eq!(body["error"]["type"], "model_not_found");
        assert!(state.response_requests.lock().unwrap().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn anthropic_model_whitespace_is_rejected_before_configured_backend_resolution() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "live-model").await;

        let response = reqwest::Client::new()
            .post(format!("{api}/v1/messages"))
            .json(&json!({
                "model": "claude-sonnet-current ",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "x"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], "invalid_request_error");
        assert!(state.model_requests.lock().unwrap().is_empty());
        assert!(state.response_requests.lock().unwrap().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn anthropic_compact_requires_a_valid_model_before_catalog_fetch() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "live-model").await;
        let client = reqwest::Client::new();

        for invalid_model in [
            None,
            Some(json!(null)),
            Some(json!("")),
            Some(json!("live-model ")),
        ] {
            let mut body = json!({
                "system": "Compact precisely.",
                "messages": [{"role": "user", "content": "History"}]
            });
            if let Some(invalid_model) = invalid_model {
                body.as_object_mut()
                    .unwrap()
                    .insert("model".to_string(), invalid_model);
            }
            let response = client
                .post(format!("{api}/v1/messages/compact"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
            let response_body: Value = response.json().await.unwrap();
            assert_eq!(response_body["error"]["type"], "invalid_request_error");
        }
        assert!(state.model_requests.lock().unwrap().is_empty());
        assert!(state.response_requests.lock().unwrap().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn anthropic_count_tokens_rejects_ignored_envelope_fields_before_catalog_fetch() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "live-model").await;
        let client = reqwest::Client::new();

        for unsupported in [json!({"stream": false}), json!({"metadata": {}})] {
            let mut body = json!({
                "model": "live-model",
                "messages": [{"role": "user", "content": "Count this"}]
            });
            body.as_object_mut()
                .unwrap()
                .extend(unsupported.as_object().unwrap().clone());
            let response = client
                .post(format!("{api}/v1/messages/count_tokens"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
            let response_body: Value = response.json().await.unwrap();
            assert_eq!(response_body["error"]["type"], "invalid_request_error");
        }

        assert!(state.model_requests.lock().unwrap().is_empty());
        assert!(state.response_requests.lock().unwrap().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn stream_catalog_failure_is_json_503_before_sse_headers() {
        let state = MockState::new(json!({"models": []}));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let response = reqwest::Client::new()
            .post(format!("{api}/v1/chat/completions"))
            .json(&json!({
                "stream": true,
                "messages": [
                    {"role": "system", "content": "Answer concisely."},
                    {"role": "user", "content": "hello"}
                ]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        assert!(response
            .headers()
            .get(CONTENT_TYPE)
            .unwrap()
            .to_str()
            .unwrap()
            .starts_with("application/json"));
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], "catalog_unavailable");
        assert!(state.response_requests.lock().unwrap().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn unsupported_catalog_default_reasoning_is_http_503() {
        let mut model = live_model("live-model", 1, true);
        model["default_reasoning_level"] = json!("not-listed");
        let state = MockState::new(json!({"models": [model]}));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let client = reqwest::Client::new();

        let health = client.get(format!("{api}/health")).send().await.unwrap();
        assert_eq!(health.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let health_body: Value = health.json().await.unwrap();
        assert_eq!(health_body["error"]["type"], "catalog_unavailable");

        let chat = client
            .post(format!("{api}/v1/chat/completions"))
            .json(&json!({
                "model": "live-model",
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(chat.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let chat_body: Value = chat.json().await.unwrap();
        assert_eq!(chat_body["error"]["type"], "catalog_unavailable");
        assert!(state.response_requests.lock().unwrap().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn default_ultra_without_wire_mapping_is_http_503() {
        let mut model = live_model("live-model", 1, true);
        model["supported_reasoning_levels"] = json!([{"effort": "ultra", "description": "ultra"}]);
        model["default_reasoning_level"] = json!("ultra");
        model["multi_agent_reasoning_effort"] = json!("ultra");
        let state = MockState::new(json!({"models": [model]}));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let client = reqwest::Client::new();

        let health = client.get(format!("{api}/health")).send().await.unwrap();
        assert_eq!(health.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let health_body: Value = health.json().await.unwrap();
        assert_eq!(health_body["error"]["type"], "catalog_unavailable");

        let chat = client
            .post(format!("{api}/v1/chat/completions"))
            .json(&json!({
                "model": "live-model",
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(chat.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        assert!(state.response_requests.lock().unwrap().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn zero_effective_context_is_catalog_unavailable() {
        let mut model = live_model("live-model", 1, true);
        model["context_window"] = json!(1);
        model["max_context_window"] = json!(1);
        model["auto_compact_token_limit"] = Value::Null;
        let state = MockState::new(json!({"models": [model]}));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;

        let health = reqwest::Client::new()
            .get(format!("{api}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(health.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let body: Value = health.json().await.unwrap();
        assert_eq!(body["error"]["type"], "catalog_unavailable");
        assert!(state.response_requests.lock().unwrap().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn direct_zero_auto_compact_limit_is_exposed() {
        let mut model = live_model("live-model", 1, true);
        model["auto_compact_token_limit"] = json!(0);
        let state = MockState::new(json!({"models": [model]}));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let client = reqwest::Client::new();

        let health = client.get(format!("{api}/health")).send().await.unwrap();
        assert_eq!(health.status(), reqwest::StatusCode::OK);
        let health_body: Value = health.json().await.unwrap();
        assert_eq!(health_body["auto_compact_token_limit"], 0);

        let count = client
            .post(format!("{api}/v1/messages/count_tokens"))
            .json(&json!({
                "model": "live-model",
                "messages": [{"role": "user", "content": "Count this"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(count.status(), reqwest::StatusCode::OK);
        let count_body: Value = count.json().await.unwrap();
        assert_eq!(count_body["auto_compact_token_limit"], 0);
        assert!(state.response_requests.lock().unwrap().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn malformed_upstream_completion_is_502() {
        let mut state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        state.malformed_response = true;
        let (upstream, upstream_handle) = start_mock_upstream(state).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let response = reqwest::Client::new()
            .post(format!("{api}/v1/chat/completions"))
            .json(&json!({
                "model": "live-model",
                "messages": [
                    {"role": "system", "content": "Answer concisely."},
                    {"role": "user", "content": "x"}
                ]
            }))
            .send()
            .await
            .unwrap();
        let response_status = response.status();
        let response_text = response.text().await.unwrap();
        assert_eq!(
            response_status,
            reqwest::StatusCode::BAD_GATEWAY,
            "{response_text}"
        );
        let body: Value = serde_json::from_str(&response_text).unwrap();
        assert_eq!(body["error"]["type"], "upstream_protocol_error");

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn malformed_stream_output_is_typed_sse_error_after_headers() {
        let mut state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        state.malformed_response = true;
        let (upstream, upstream_handle) = start_mock_upstream(state).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let response = reqwest::Client::new()
            .post(format!("{api}/v1/chat/completions"))
            .json(&json!({
                "model": "live-model",
                "stream": true,
                "messages": [
                    {"role": "system", "content": "Answer concisely."},
                    {"role": "user", "content": "x"}
                ]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        assert!(response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| value.starts_with("text/event-stream")));
        let response_body = response.text().await.unwrap();
        let data_frames: Vec<&str> = response_body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .collect();
        assert_ne!(data_frames.last().copied(), Some("[DONE]"));
        let parsed_frames: Vec<Value> = data_frames
            .iter()
            .filter(|frame| **frame != "[DONE]")
            .map(|frame| serde_json::from_str(frame).unwrap())
            .collect();
        let errors: Vec<&Value> = parsed_frames
            .iter()
            .filter_map(|frame| frame.get("error"))
            .collect();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0]["type"], "upstream_protocol_error");
        assert_eq!(errors[0]["code"], "upstream_protocol_error");

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn catalog_ttl_expiry_never_serves_stale_data_after_schema_failure() {
        assert_eq!(
            upstream_contract()["models_request"]["allow_stale_on_refresh_error"],
            false
        );
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) =
            start_api_with_catalog_ttl(&upstream, "", std::time::Duration::from_millis(20)).await;
        let client = reqwest::Client::new();

        let first = client.get(format!("{api}/v1/models")).send().await.unwrap();
        assert_eq!(first.status(), reqwest::StatusCode::OK);
        *state.catalog.lock().unwrap() = json!({"models": "invalid"});
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;

        let expired = client.get(format!("{api}/v1/models")).send().await.unwrap();
        assert_eq!(expired.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let body: Value = expired.json().await.unwrap();
        assert_eq!(body["error"]["type"], "catalog_unavailable");
        assert_eq!(state.model_requests.lock().unwrap().len(), 2);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn concurrent_model_routes_share_one_catalog_fetch() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        state.model_delay_millis.store(50, Ordering::SeqCst);
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let client = reqwest::Client::new();
        let mut requests = Vec::new();
        for _ in 0..8 {
            let client = client.clone();
            let url = format!("{api}/v1/models");
            requests.push(tokio::spawn(async move {
                client.get(url).send().await.unwrap()
            }));
        }
        for request in requests {
            assert_eq!(request.await.unwrap().status(), reqwest::StatusCode::OK);
        }
        assert_eq!(state.model_requests.lock().unwrap().len(), 1);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn catalog_cache_is_scoped_by_authenticated_account() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let client = reqwest::Client::new();

        assert_eq!(
            client
                .get(format!("{api}/v1/models"))
                .send()
                .await
                .unwrap()
                .status(),
            reqwest::StatusCode::OK
        );
        write_auth_at(&auth_path, "second-account", "second-access-token");
        assert_eq!(
            client
                .get(format!("{api}/v1/models"))
                .send()
                .await
                .unwrap()
                .status(),
            reqwest::StatusCode::OK
        );
        let requests = state.model_requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        assert_eq!(
            requests[0].0.get("chatgpt-account-id").unwrap(),
            "account-id"
        );
        assert_eq!(
            requests[1].0.get("chatgpt-account-id").unwrap(),
            "second-account"
        );
        drop(requests);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn response_models_etag_invalidates_the_next_catalog_request() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        *state.response_models_etag.lock().unwrap() = Some("catalog-v2".to_string());
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let client = reqwest::Client::new();

        assert_eq!(
            client
                .get(format!("{api}/v1/models"))
                .send()
                .await
                .unwrap()
                .status(),
            reqwest::StatusCode::OK
        );
        let chat = client
            .post(format!("{api}/v1/chat/completions"))
            .json(&json!({
                "model": "live-model",
                "messages": [
                    {"role": "system", "content": "Answer concisely."},
                    {"role": "user", "content": "hello"}
                ]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(chat.status(), reqwest::StatusCode::OK);

        *state.catalog_etag.lock().unwrap() = "catalog-v2".to_string();
        let refreshed = client.get(format!("{api}/v1/models")).send().await.unwrap();
        assert_eq!(refreshed.status(), reqwest::StatusCode::OK);
        assert!(refreshed.headers().get("etag").is_none());
        assert_eq!(state.model_requests.lock().unwrap().len(), 2);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn blank_catalog_etags_are_treated_as_absent() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        *state.catalog_etag.lock().unwrap() = String::new();
        *state.response_models_etag.lock().unwrap() = Some(String::new());
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let client = reqwest::Client::new();

        assert_eq!(
            client
                .get(format!("{api}/v1/models"))
                .send()
                .await
                .unwrap()
                .status(),
            reqwest::StatusCode::OK
        );
        assert_eq!(
            client
                .post(format!("{api}/v1/chat/completions"))
                .json(&json!({
                    "model": "live-model",
                    "messages": [
                        {"role": "system", "content": "Answer concisely."},
                        {"role": "user", "content": "hello"}
                    ]
                }))
                .send()
                .await
                .unwrap()
                .status(),
            reqwest::StatusCode::OK
        );
        assert_eq!(
            client
                .get(format!("{api}/v1/models"))
                .send()
                .await
                .unwrap()
                .status(),
            reqwest::StatusCode::OK
        );
        assert_eq!(state.model_requests.lock().unwrap().len(), 1);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn health_never_exposes_the_upstream_catalog_etag() {
        let secret = "access-token-sentinel";
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        *state.catalog_etag.lock().unwrap() = secret.to_string();
        let (upstream, upstream_handle) = start_mock_upstream(state).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;

        let response = reqwest::Client::new()
            .get(format!("{api}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);
        let body_text = response.text().await.unwrap();
        let body: Value = serde_json::from_str(&body_text).unwrap();
        assert!(body.get("catalog_etag").is_none());
        assert!(!body_text.contains(secret));

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn an_empty_catalog_is_exposed_but_no_model_dependent_route_reaches_responses() {
        let state = MockState::new(json!({"models": []}));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        let client = reqwest::Client::new();

        let models = client.get(format!("{api}/v1/models")).send().await.unwrap();
        assert_eq!(models.status(), reqwest::StatusCode::OK);
        let models_body: Value = models.json().await.unwrap();
        assert_eq!(models_body["data"], json!([]));
        let health = client.get(format!("{api}/health")).send().await.unwrap();
        assert_eq!(health.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let health_body: Value = health.json().await.unwrap();
        assert_eq!(health_body["status"], "error");
        assert_eq!(health_body["catalog_status"], "fresh");
        assert_eq!(health_body["error"]["type"], "catalog_unavailable");
        assert_eq!(
            health_body["error"]["message"],
            "authenticated model catalog is unavailable"
        );
        assert!(health_body["catalog_fetched_at"].as_str().is_some());
        assert!(health_body["catalog_expires_at"].as_str().is_some());
        for field in [
            "model",
            "reasoning_effort",
            "context_window",
            "auto_compact_token_limit",
        ] {
            assert!(health_body[field].is_null(), "{field}");
        }
        assert!(health_body.get("account_id").is_none());
        assert!(health_body.get("token").is_none());

        for (path, body, expected_status) in [
            (
                "/v1/images/generations",
                json!({"prompt": "draw"}),
                reqwest::StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                "/v1/inspect",
                json!({
                    "prompt": "inspect",
                    "images": [{"image_url": "data:image/png;base64,AA=="}]
                }),
                reqwest::StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                "/v1/compact",
                json!({
                    "messages": [
                        {"role": "system", "content": "Compact."},
                        {"role": "user", "content": "history"}
                    ]
                }),
                reqwest::StatusCode::SERVICE_UNAVAILABLE,
            ),
            (
                "/v1/messages/compact",
                json!({
                    "model": "live-model",
                    "max_tokens": 128,
                    "system": "Compact.",
                    "messages": [{"role": "user", "content": "history"}]
                }),
                reqwest::StatusCode::NOT_FOUND,
            ),
            (
                "/v1/messages/count_tokens",
                json!({
                    "model": "live-model",
                    "system": "Count.",
                    "messages": [{"role": "user", "content": "history"}]
                }),
                reqwest::StatusCode::NOT_FOUND,
            ),
            (
                "/v1/messages",
                json!({
                    "model": "live-model",
                    "max_tokens": 128,
                    "system": "Answer.",
                    "messages": [{"role": "user", "content": "hello"}]
                }),
                reqwest::StatusCode::NOT_FOUND,
            ),
        ] {
            let response = client
                .post(format!("{api}{path}"))
                .header("x-claude-code-session-id", "test-session")
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), expected_status, "{path}");
        }
        assert!(state.response_requests.lock().unwrap().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn health_reports_fresh_catalog_when_configured_model_is_missing() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        let (upstream, upstream_handle) = start_mock_upstream(state).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "missing-model").await;

        let health = reqwest::Client::new()
            .get(format!("{api}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(health.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let body: Value = health.json().await.unwrap();
        assert_eq!(body["catalog_status"], "fresh");
        assert_eq!(body["error"]["type"], "catalog_unavailable");
        assert_eq!(
            body["error"]["message"],
            "authenticated model catalog is unavailable"
        );
        assert!(body["catalog_fetched_at"].as_str().is_some());
        assert!(body["catalog_expires_at"].as_str().is_some());
        for field in [
            "model",
            "reasoning_effort",
            "context_window",
            "auto_compact_token_limit",
        ] {
            assert!(body[field].is_null(), "{field}");
        }

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn health_accepts_a_model_without_optional_default_reasoning() {
        let mut model = live_model("live-model", 1, true);
        model["default_reasoning_level"] = Value::Null;
        model["context_window"] = Value::Null;
        model["max_context_window"] = Value::Null;
        model["auto_compact_token_limit"] = Value::Null;
        let state = MockState::new(json!({"models": [model]}));
        let (upstream, upstream_handle) = start_mock_upstream(state).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;

        let health = reqwest::Client::new()
            .get(format!("{api}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(health.status(), reqwest::StatusCode::OK);
        let body: Value = health.json().await.unwrap();
        assert_eq!(body["status"], "ok");
        assert_eq!(body["catalog_status"], "fresh");
        assert_eq!(body["model"], "live-model");
        assert!(body["reasoning_effort"].is_null());
        assert!(body["context_window"].is_null());
        assert!(body["auto_compact_token_limit"].is_null());
        assert!(body.get("error").is_none());

        let count = reqwest::Client::new()
            .post(format!("{api}/v1/messages/count_tokens"))
            .json(&json!({
                "model": "live-model",
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(count.status(), reqwest::StatusCode::OK);
        let count_body: Value = count.json().await.unwrap();
        assert!(count_body["input_tokens"].as_u64().is_some());
        assert!(count_body["context_window"].is_null());
        assert!(count_body["auto_compact_token_limit"].is_null());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test]
    async fn missing_auth_is_401_instead_of_a_catalog_fallback() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        let (upstream, upstream_handle) = start_mock_upstream(state).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "").await;
        std::fs::remove_file(&auth_path).unwrap();

        let response = reqwest::Client::new()
            .get(format!("{api}/v1/models"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], "authentication_error");
        assert_eq!(
            body["error"]["message"],
            "ChatGPT OAuth credentials are unavailable; rerun codex login"
        );
        assert!(!body.to_string().contains(auth_path.to_str().unwrap()));

        let health = reqwest::Client::new()
            .get(format!("{api}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(health.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let health_body: Value = health.json().await.unwrap();
        assert_eq!(health_body["status"], "error");
        assert_eq!(health_body["auth_available"], false);
        assert_eq!(
            health_body["error"]["message"],
            "ChatGPT OAuth credentials are unavailable"
        );

        api_handle.abort();
        upstream_handle.abort();
    }

    #[tokio::test]
    async fn claude_code_anthropic_models_use_live_backend_without_forwarding_max_tokens() {
        let state = MockState::new(json!({"models": [live_model("live-model", 1, true)]}));
        let (upstream, upstream_handle) = start_mock_upstream(state.clone()).await;
        let (api, auth_path, api_handle) = start_api(&upstream, "live-model").await;
        let client = reqwest::Client::new();

        for requested_model in ["live-model", "claude-sonnet-current"] {
            let response = client
                .post(format!("{api}/v1/messages"))
                .header("x-claude-code-session-id", "test-session")
                .json(&json!({
                    "model": requested_model,
                    "max_tokens": 128,
                    "system": "Answer.",
                    "messages": [{"role": "user", "content": "hello"}]
                }))
                .send()
                .await
                .unwrap();
            let status = response.status();
            let body: Value = response.json().await.unwrap();
            assert_eq!(status, reqwest::StatusCode::OK, "{body}");
            assert_eq!(body["model"], requested_model);
        }

        let requests = state.response_requests.lock().unwrap();
        assert_eq!(requests.len(), 2);
        for request in requests.iter() {
            assert_eq!(request["model"], "live-model");
            assert!(request.get("max_tokens").is_none());
            assert!(request.get("max_output_tokens").is_none());
        }
        drop(requests);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::AuthError;
    use crate::messages::{Message, MessageRole};
    use axum::body::{Body, Bytes};
    use axum::http::{header::CONTENT_TYPE, Method, Uri};
    use futures::StreamExt;
    use serde_json::json;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Mutex;
    use tokio::sync::Semaphore;
    use tokio::task::JoinHandle;

    #[test]
    fn positive_integer_controls_accept_integral_json_number_forms() {
        for value in [
            json!(1),
            json!(1.0),
            json!(1e0),
            json!(9_007_199_254_740_991.0),
        ] {
            assert!(positive_js_safe_integer(&value).is_some());
        }
        for value in [
            json!(null),
            json!(true),
            json!(0),
            json!(-1),
            json!(1.5),
            json!(9_007_199_254_740_992.0),
        ] {
            assert!(positive_js_safe_integer(&value).is_none());
        }
    }

    #[test]
    fn openai_tool_choice_is_strictly_parsed_and_normalized() {
        for choice in [json!("auto"), json!("none"), json!("required")] {
            assert_eq!(
                parse_openai_tool_choice(Some(&choice)).unwrap(),
                Some(choice)
            );
        }
        assert_eq!(
            parse_openai_tool_choice(Some(&json!({
                "type": "function",
                "function": {"name": "lookup"}
            })))
            .unwrap(),
            Some(json!({"type": "function", "name": "lookup"}))
        );
        for choice in [
            json!(true),
            json!(1),
            json!("future"),
            json!({}),
            json!({"type": "function"}),
            json!({"type": "function", "function": null}),
            json!({"type": "function", "function": {"name": ""}}),
            json!({"type": "function", "function": {"name": "lookup", "extra": true}}),
            json!({"type": "function", "function": {"name": "lookup"}, "extra": true}),
        ] {
            assert!(parse_openai_tool_choice(Some(&choice)).is_err());
        }
    }

    #[test]
    fn chat_messages_reject_role_mismatched_and_named_content() {
        for value in [
            json!({"role": "user", "content": [{"type": "output_text", "text": "wrong"}]}),
            json!({"role": "system", "content": [{"type": "output_text", "text": "wrong"}]}),
            json!({
                "role": "tool",
                "tool_call_id": "call-1",
                "content": [{"type": "output_text", "text": "wrong"}]
            }),
            json!({"role": "assistant", "content": [{"type": "input_text", "text": "wrong"}]}),
            json!({
                "role": "tool",
                "tool_call_id": "call-1",
                "name": "lookup",
                "content": "result"
            }),
        ] {
            let message: ChatMessage = serde_json::from_value(value).unwrap();
            assert!(matches!(
                request_messages_to_internal(&[message]),
                Err(ProviderError::InvalidRequest(_))
            ));
        }
    }

    #[test]
    fn chat_messages_preserve_empty_content_arrays() {
        for role in ["user", "developer", "assistant"] {
            let message: ChatMessage =
                serde_json::from_value(json!({"role": role, "content": []})).unwrap();
            let normalized = request_messages_to_internal(&[message]).unwrap();
            assert_eq!(normalized[0].structured_content, Some(vec![]));
            let wire = crate::provider::messages_to_response_items(&normalized).unwrap();
            assert_eq!(wire[0]["role"], role);
            assert_eq!(wire[0]["content"], json!([]));
        }
    }

    #[test]
    fn chat_messages_accept_assistant_null_content_and_reject_missing_payload() {
        let value = json!({"role": "assistant", "content": null});
        validate_openai_chat_message_fields(&json!({"messages": [value.clone()]})).unwrap();
        let message: ChatMessage = serde_json::from_value(value).unwrap();
        let normalized = request_messages_to_internal(&[message]).unwrap();
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].role, MessageRole::Assistant);
        assert!(normalized[0].content.is_empty());
        assert!(normalized[0].structured_content.is_none());
        assert!(normalized[0].tool_calls.is_empty());

        assert!(matches!(
            validate_openai_chat_message_fields(&json!({
                "messages": [{"role": "assistant"}]
            })),
            Err(ProviderError::InvalidRequest(_))
        ));

        for role in ["system", "developer", "user", "tool"] {
            let message: ChatMessage = serde_json::from_value(json!({"role": role})).unwrap();
            assert!(matches!(
                request_messages_to_internal(&[message]),
                Err(ProviderError::InvalidRequest(_))
            ));
        }
    }

    #[test]
    fn chat_messages_normalize_input_audio_and_reject_invalid_shapes() {
        let message: ChatMessage = serde_json::from_value(json!({
            "role": "user",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": "AAAA", "format": "wav"}
            }]
        }))
        .unwrap();
        let normalized = request_messages_to_internal(&[message]).unwrap();
        assert_eq!(
            normalized[0].structured_content,
            Some(vec![json!({
                "type": "input_audio",
                "audio_url": "data:audio/wav;base64,AAAA"
            })])
        );

        for content in [
            json!({"type": "input_audio"}),
            json!({"type": "input_audio", "input_audio": null}),
            json!({"type": "input_audio", "input_audio": {"format": "wav"}}),
            json!({"type": "input_audio", "input_audio": {"data": "AAAA"}}),
            json!({"type": "input_audio", "input_audio": {"data": "AAAA", "format": "flac"}}),
            json!({"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav", "extra": true}}),
        ] {
            let message: ChatMessage = serde_json::from_value(json!({
                "role": "user",
                "content": [content]
            }))
            .unwrap();
            assert!(matches!(
                request_messages_to_internal(&[message]),
                Err(ProviderError::InvalidRequest(_))
            ));
        }

        let assistant: ChatMessage = serde_json::from_value(json!({
            "role": "assistant",
            "content": [{
                "type": "input_audio",
                "input_audio": {"data": "AAAA", "format": "mp3"}
            }]
        }))
        .unwrap();
        assert!(matches!(
            request_messages_to_internal(&[assistant]),
            Err(ProviderError::InvalidRequest(_))
        ));
    }

    #[test]
    fn openai_stream_rejects_normalized_web_search_events() {
        let mut state = OpenAiStreamState {
            request_id: "chatcmpl-test".to_string(),
            created: 0,
            model: "gpt-5.5".to_string(),
            tool_call_index: 0,
            finished: false,
        };
        assert!(matches!(
            state.push(&json!({
                "type": "web_search_call",
                "id": "search-1",
                "input": {"query": "q"},
                "content": []
            })),
            Err(ProviderError::UpstreamProtocol(_))
        ));
    }

    #[test]
    fn auth_status_mapping_preserves_authentication_failures() {
        assert_eq!(
            map_error_status(&ProviderError::Auth(AuthError::Missing(
                "missing".to_string()
            ))),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            map_error_status(&ProviderError::Auth(AuthError::Refresh(
                "failed".to_string()
            ))),
            StatusCode::UNAUTHORIZED
        );
        assert_eq!(
            map_error_status(&ProviderError::Auth(AuthError::OAuth(
                "invalid".to_string()
            ))),
            StatusCode::UNAUTHORIZED
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sse_event_bridge_applies_bounded_backpressure_and_cancels_on_drop() {
        let cancellation = Arc::new(AtomicBool::new(false));
        let (sender, receiver) = tokio::sync::mpsc::channel(1);
        sender.send(Event::default().data("first")).await.unwrap();
        assert_eq!(sender.max_capacity(), 1);

        let worker_sender = sender.clone();
        let worker_cancellation = cancellation.clone();
        let worker = task::spawn_blocking(move || {
            send_sse_event_with_backpressure(
                &worker_sender,
                Event::default().data("second"),
                &worker_cancellation,
                "test downstream disconnected",
            )
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!worker.is_finished());

        let mut downstream = DownstreamEventReceiver {
            receiver,
            cancellation: cancellation.clone(),
        };
        assert!(downstream.receiver.recv().await.is_some());
        tokio::time::timeout(Duration::from_secs(1), worker)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        assert!(downstream.receiver.recv().await.is_some());
        drop(downstream);
        assert!(cancellation.load(Ordering::Acquire));

        assert!(send_sse_event_with_backpressure(
            &sender,
            Event::default().data("third"),
            &cancellation,
            "test downstream disconnected",
        )
        .is_err());
    }

    fn has_nested_key(value: &Value, key: &str) -> bool {
        match value {
            Value::Array(items) => items.iter().any(|item| has_nested_key(item, key)),
            Value::Object(object) => {
                object.contains_key(key) || object.values().any(|item| has_nested_key(item, key))
            }
            _ => false,
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum RecordedEndpoint {
        Responses,
        Compact,
    }

    #[derive(Clone, Debug)]
    struct RecordedRequest {
        endpoint: RecordedEndpoint,
        method: Method,
        path: String,
        headers: HeaderMap,
        body: Value,
    }

    #[derive(Clone)]
    struct RecordingState {
        requests: Arc<Mutex<Vec<RecordedRequest>>>,
        response_prefix_events: Arc<Mutex<Vec<Value>>>,
        response_output: Arc<Mutex<Vec<Value>>>,
        emit_completed: Arc<Mutex<bool>>,
        trailing_event: Arc<Mutex<Option<Value>>>,
        completed_response_id: Arc<Mutex<Value>>,
        response_usage: Arc<Mutex<Value>>,
        completed_end_turn: Arc<Mutex<Option<Value>>>,
    }

    impl Default for RecordingState {
        fn default() -> Self {
            Self {
                requests: Arc::new(Mutex::new(Vec::new())),
                response_prefix_events: Arc::new(Mutex::new(Vec::new())),
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
                    "total_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 0}
                }))),
                completed_end_turn: Arc::new(Mutex::new(Some(json!(true)))),
            }
        }
    }

    impl RecordingState {
        fn record(
            &self,
            endpoint: RecordedEndpoint,
            method: Method,
            path: String,
            headers: HeaderMap,
            body: Value,
        ) {
            self.requests.lock().unwrap().push(RecordedRequest {
                endpoint,
                method,
                path,
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

        fn set_response_prefix_events(&self, events: Vec<Value>) {
            *self.response_prefix_events.lock().unwrap() = events;
        }

        fn response_prefix_events(&self) -> Vec<Value> {
            self.response_prefix_events.lock().unwrap().clone()
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

        fn set_completed_end_turn(&self, end_turn: Option<bool>) {
            *self.completed_end_turn.lock().unwrap() = end_turn.map(Value::Bool);
        }

        fn completed_end_turn(&self) -> Option<Value> {
            self.completed_end_turn.lock().unwrap().clone()
        }
    }

    async fn record_responses_request(
        State(state): State<RecordingState>,
        method: Method,
        uri: Uri,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> impl IntoResponse {
        state.record(
            RecordedEndpoint::Responses,
            method,
            uri.path().to_string(),
            headers,
            body,
        );
        let mut completed = json!({
            "type": "response.completed",
            "response": {
                "id": state.completed_response_id(),
                "output": [],
                "usage": state.response_usage()
            }
        });
        if let Some(end_turn) = state.completed_end_turn() {
            completed["response"]["end_turn"] = end_turn;
        }
        let mut stream_body = String::new();
        for event in state.response_prefix_events() {
            stream_body.push_str(&format!(
                "data: {}\n\n",
                serde_json::to_string(&event).unwrap()
            ));
        }
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
        method: Method,
        uri: Uri,
        headers: HeaderMap,
        Json(body): Json<Value>,
    ) -> Json<Value> {
        state.record(
            RecordedEndpoint::Compact,
            method,
            uri.path().to_string(),
            headers,
            body,
        );
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

    fn live_model(slug: &str, priority: i32, use_responses_lite: bool) -> Value {
        json!({
            "slug": slug,
            "display_name": slug,
            "description": "live catalog test fixture",
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [
                {"effort": "none", "description": "none"},
                {"effort": "low", "description": "low"},
                {"effort": "medium", "description": "medium"},
                {"effort": "high", "description": "high"},
                {"effort": "max", "description": "max"}
            ],
            "multi_agent_reasoning_effort": "max",
            "visibility": "list",
            "supported_in_api": true,
            "priority": priority,
            "service_tiers": [{"id": "priority", "name": "Priority", "description": "Priority"}],
            "default_service_tier": null,
            "support_verbosity": true,
            "default_verbosity": "low",
            "supports_image_detail_original": true,
            "context_window": 272000,
            "max_context_window": 272000,
            "auto_compact_token_limit": null,
            "input_modalities": ["text", "image"],
            "use_responses_lite": use_responses_lite
        })
    }

    async fn test_models() -> Json<Value> {
        Json(json!({
            "models": [
                live_model("gpt-5.5", 1, false),
                live_model("gpt-5.6-sol", 2, true)
            ]
        }))
    }

    async fn effective_context_overflow_models() -> Json<Value> {
        let mut model = live_model("gpt-effective-context-overflow", 1, false);
        model["context_window"] = json!(9_007_199_254_740_991i64);
        model["max_context_window"] = json!(9_007_199_254_740_991i64);
        model["effective_context_window_percent"] = json!(9_007_199_254_740_991i64);
        Json(json!({"models": [model]}))
    }

    async fn start_effective_context_overflow_upstream() -> (String, JoinHandle<()>) {
        let app = Router::new().route("/models", get(effective_context_overflow_models));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), handle)
    }

    async fn start_recording_upstream() -> (String, RecordingState, JoinHandle<()>) {
        let state = RecordingState::default();
        let app = Router::new()
            .route("/models", get(test_models))
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
                            "end_turn": true,
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
            .route("/models", get(test_models))
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
        let app = Router::new()
            .route("/models", get(test_models))
            .route("/responses", post(silent_sse_response));
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
            .route("/models", get(test_models))
            .route("/responses", post(fixed_status_response))
            .with_state(status);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), handle)
    }

    async fn slow_error_body_response() -> impl IntoResponse {
        let body_stream = stream::unfold(0usize, |index| async move {
            if index >= 100 {
                return None;
            }
            if index > 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
            Some((
                Ok::<Bytes, std::convert::Infallible>(Bytes::from_static(b"x")),
                index + 1,
            ))
        });
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Body::from_stream(body_stream),
        )
    }

    async fn start_slow_error_body_upstream() -> (String, JoinHandle<()>) {
        let app = Router::new()
            .route("/models", get(test_models))
            .route("/responses", post(slow_error_body_response));
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
                "end_turn": true,
                "output": [],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            }
        });
        let output = json!({
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "refreshed"}]
            }
        });
        (
            [(CONTENT_TYPE, "text/event-stream")],
            format!(
                "data: {}\n\ndata: {}\n\n",
                serde_json::to_string(&output).unwrap(),
                serde_json::to_string(&completed).unwrap()
            ),
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
            .route("/models", get(test_models))
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
        )
        .unwrap();
        let app = create_router(AppState {
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
            auth_path: None,
            codex_config,
            provider: Arc::new(
                ChatGPTOAuthProvider::new(
                    model.to_string(),
                    "http://127.0.0.1:1".to_string(),
                    None,
                    None,
                )
                .unwrap(),
            ),
        }
    }

    fn model_for_settings(
        context_window: Option<i64>,
        max_context_window: Option<i64>,
        auto_compact_token_limit: Option<i64>,
        default_reasoning_level: Option<&str>,
    ) -> Arc<ModelInfo> {
        let mut model = live_model("live-settings-model", 0, false);
        model["context_window"] = context_window.map_or(Value::Null, Value::from);
        model["max_context_window"] = max_context_window.map_or(Value::Null, Value::from);
        model["auto_compact_token_limit"] =
            auto_compact_token_limit.map_or(Value::Null, Value::from);
        model["default_reasoning_level"] = default_reasoning_level.map_or(Value::Null, Value::from);
        crate::model_catalog::parse_models_response(
            &serde_json::to_vec(&json!({"models": [model]})).unwrap(),
        )
        .unwrap()
        .remove(0)
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
            description: Some("Search docs".to_string()),
            parameters: json!({"type": "object"}),
            strict: false,
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
            arguments: "{\"query\":\"문서\"}".to_string(),
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
            }
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
            ("multi_agent", Value::Null),
            ("programmatic_tool_calling", Value::Null),
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
    async fn anthropic_count_tokens_uses_normalized_input_and_ignores_supported_control_fields() {
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
        let (messages, tools, _, _, _, _, _) = anthropic_request_to_internal(&base_body).unwrap();
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
        controlled.insert("output_config".to_string(), json!({"effort": "max"}));
        let tool = controlled
            .get_mut("tools")
            .and_then(Value::as_array_mut)
            .and_then(|tools| tools.first_mut())
            .and_then(Value::as_object_mut)
            .unwrap();
        tool.insert("strict".to_string(), json!(false));

        let controlled_response = client
            .post(format!("{api_url}/v1/messages/count_tokens"))
            .header("x-claude-code-session-id", "test-session")
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
                "model": "gpt-5.6-sol",
                "messages": [{"role": "user", "content": "Count this."}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(requested.status(), reqwest::StatusCode::OK);
        let requested_body: Value = requested.json().await.unwrap();
        assert_eq!(requested_body["context_window"], json!(258_400));
        assert_eq!(requested_body["auto_compact_token_limit"], json!(244_800));

        let configured_backend = client
            .post(format!("{api_url}/v1/messages/count_tokens"))
            .json(&json!({
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "Count this."}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(configured_backend.status(), reqwest::StatusCode::OK);
        let configured_backend_body: Value = configured_backend.json().await.unwrap();
        assert_eq!(configured_backend_body["context_window"], json!(258_400));
        assert_eq!(
            configured_backend_body["auto_compact_token_limit"],
            json!(244_800)
        );
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_route_uses_live_gpt_model_and_explicit_backend_for_claude() {
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
            .header("x-claude-code-session-id", "test-session")
            .json(&json!({
                "model": "gpt-5.6-sol",
                "max_tokens": 2048,
                "system": "You are precise.",
                "messages": [{"role": "user", "content": "Use the requested GPT model."}],
                "thinking": {"type": "enabled", "budget_tokens": 1024},
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
        assert_eq!(gpt_body["model"], json!("gpt-5.6-sol"));

        let claude_response = client
            .post(format!("{api_url}/v1/messages"))
            .header("x-claude-code-session-id", "test-session")
            .json(&json!({
                "model": "claude-opus-4-6",
                "max_tokens": 64,
                "system": "You are precise.",
                "messages": [{"role": "user", "content": "Use the configured backend."}],
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
    async fn anthropic_stream_custom_tool_uses_disabled_thinking_over_ambient_effort() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        recording.set_response_output(vec![json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Lookup complete."}]
        })]);
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(Some("high"), None, None),
        )
        .await;

        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/messages"))
            .header("x-claude-code-session-id", "test-session")
            .json(&json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 100,
                "system": "Use the available tools.",
                "messages": [{"role": "user", "content": "Look up the docs."}],
                "tools": [{
                    "name": "lookup",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }],
                "stream": true,
                "thinking": {"type": "disabled"},
                "output_config": {"effort": "high"},
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        let status = response.status();
        let content_type = response
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        let mut pending = response.text().await.unwrap();
        assert_eq!(status, reqwest::StatusCode::OK, "{pending}");
        assert_eq!(content_type.as_deref(), Some("text/event-stream"));
        let mut events = Vec::new();
        parse_anthropic_sse_blocks(&mut pending, &mut events);
        assert!(pending.is_empty());
        assert_eq!(events.first().unwrap().0, "message_start");
        assert_eq!(events.last().unwrap().0, "message_stop");
        assert_eq!(events.last().unwrap().1["type"], json!("message_stop"));

        let requests = recording.requests();
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].endpoint, RecordedEndpoint::Responses);
        assert_eq!(requests[0].body["model"], json!("gpt-5.6-sol"));
        assert_eq!(requests[0].body["reasoning"]["effort"], json!("none"));
        assert_eq!(requests[0].body["tools"][0]["type"], json!("function"));
        assert_eq!(requests[0].body["tools"][0]["name"], json!("lookup"));
        assert!(requests[0]
            .headers
            .get(crate::model_capabilities::LITE_HEADER_NAME)
            .is_none());

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
            .header("x-claude-code-session-id", "test-session")
            .json(&json!({
                "model": "gpt-5.5",
                "max_tokens": 64,
                "system": "You are precise.",
                "tools": [{
                    "name": "lookup",
                    "strict": false,
                    "eager_input_streaming": null,
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
                "reasoning_effort": "high",
                "output_config": {"effort": "low"}
            }),
            json!({
                "tools": [{
                    "name": "lookup",
                    "defer_loading": true,
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
    fn live_catalog_defaults_drive_reasoning_context_and_compaction_threshold() {
        let state = helper_state("live-settings-model", test_codex_config(None, None, None));
        let model = model_for_settings(Some(272_000), Some(272_000), None, Some("low"));

        assert_eq!(
            effective_reasoning_effort_with_options(&state, None, None, &model).unwrap(),
            Some("low".to_string())
        );
        assert_eq!(
            context_window_for_model(&state, &model).unwrap(),
            Some(258_400)
        );
        assert_eq!(
            auto_compact_token_limit_for_model(&state, &model).unwrap(),
            Some(244_800)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn saturated_effective_context_overflow_fails_health_and_count() {
        let (upstream_url, upstream_handle) = start_effective_context_overflow_upstream().await;
        let model = "gpt-effective-context-overflow";
        let (api_url, auth_path, api_handle) =
            start_api_server(&upstream_url, model, test_codex_config(None, None, None)).await;

        let client = reqwest::Client::new();
        let health = client
            .get(format!("{api_url}/health"))
            .send()
            .await
            .unwrap();
        assert_eq!(health.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let health_body: Value = health.json().await.unwrap();
        assert_eq!(health_body["error"]["type"], json!("catalog_unavailable"));

        let count = client
            .post(format!("{api_url}/v1/messages/count_tokens"))
            .json(&json!({
                "model": model,
                "messages": [{"role": "user", "content": "hello"}]
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(count.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let count_body: Value = count.json().await.unwrap();
        assert_eq!(count_body["error"]["type"], json!("api_error"));

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[test]
    fn missing_live_context_remains_absent_without_a_config_override() {
        let state = helper_state("live-settings-model", test_codex_config(None, None, None));
        let model = model_for_settings(None, None, None, None);

        assert_eq!(context_window_for_model(&state, &model).unwrap(), None);
        assert_eq!(
            auto_compact_token_limit_for_model(&state, &model).unwrap(),
            None
        );
    }

    #[test]
    fn configured_context_applies_when_live_context_is_absent() {
        let state = helper_state(
            "live-settings-model",
            test_codex_config(None, Some(100_000), None),
        );
        let model = model_for_settings(None, None, None, None);

        assert_eq!(
            context_window_for_model(&state, &model).unwrap(),
            Some(95_000)
        );
        assert_eq!(
            auto_compact_token_limit_for_model(&state, &model).unwrap(),
            Some(90_000)
        );
    }

    #[test]
    fn explicit_compaction_is_clamped_to_ninety_percent_of_context() {
        let state = helper_state(
            "live-settings-model",
            test_codex_config(None, None, Some(190_000)),
        );
        let model = model_for_settings(Some(100_000), Some(120_000), None, None);

        assert_eq!(
            auto_compact_token_limit_for_model(&state, &model).unwrap(),
            Some(90_000)
        );
    }

    #[test]
    fn request_effort_precedes_config_and_live_catalog() {
        let state = helper_state(
            "live-settings-model",
            test_codex_config(Some("low"), None, None),
        );
        let model = model_for_settings(Some(100_000), Some(120_000), None, Some("medium"));

        assert_eq!(
            effective_reasoning_effort_with_options(&state, Some("high"), None, &model).unwrap(),
            Some("high".to_string())
        );
        assert_eq!(
            effective_reasoning_effort_with_options(&state, None, None, &model).unwrap(),
            Some("low".to_string())
        );
    }

    #[test]
    fn configured_effort_errors_are_internal_while_request_errors_are_caller_errors() {
        let model = model_for_settings(Some(100_000), Some(120_000), None, Some("medium"));
        for configured in ["", "not-supported"] {
            let state = helper_state(
                "live-settings-model",
                test_codex_config(Some(configured), None, None),
            );
            assert!(matches!(
                effective_reasoning_effort_with_options(&state, None, None, &model),
                Err(ProviderError::Request(_))
            ));
        }

        let state = helper_state("live-settings-model", test_codex_config(None, None, None));
        assert!(matches!(
            effective_reasoning_effort_with_options(&state, Some("not-supported"), None, &model),
            Err(ProviderError::InvalidRequest(_))
        ));
    }

    #[test]
    fn configured_context_uses_official_ninety_percent_compaction_default() {
        let state = helper_state(
            "live-settings-model",
            test_codex_config(None, Some(100_000), None),
        );
        let model = model_for_settings(Some(80_000), Some(120_000), None, None);

        assert_eq!(
            context_window_for_model(&state, &model).unwrap(),
            Some(95_000)
        );
        assert_eq!(
            auto_compact_token_limit_for_model(&state, &model).unwrap(),
            Some(90_000)
        );
    }

    #[test]
    fn configured_context_without_a_live_max_is_not_clamped_to_live_context() {
        let state = helper_state(
            "live-settings-model",
            test_codex_config(None, Some(100_000), None),
        );
        let model = model_for_settings(Some(80_000), None, None, None);

        assert_eq!(
            context_window_for_model(&state, &model).unwrap(),
            Some(95_000)
        );
        assert_eq!(
            auto_compact_token_limit_for_model(&state, &model).unwrap(),
            Some(90_000)
        );
    }

    #[test]
    fn configured_context_and_compaction_are_clamped_to_live_limits() {
        let state = helper_state(
            "live-settings-model",
            test_codex_config(None, Some(500_000), Some(450_000)),
        );
        let model = model_for_settings(Some(272_000), Some(272_000), None, None);

        assert_eq!(
            context_window_for_model(&state, &model).unwrap(),
            Some(258_400)
        );
        assert_eq!(
            auto_compact_token_limit_for_model(&state, &model).unwrap(),
            Some(244_800)
        );
    }

    #[test]
    fn nonpositive_live_catalog_limits_fail_when_consumed() {
        let state = helper_state("live-settings-model", test_codex_config(None, None, None));
        assert!(context_window_for_model(
            &state,
            &model_for_settings(Some(0), Some(1), None, None)
        )
        .is_err());
        assert!(
            context_window_for_model(&state, &model_for_settings(None, Some(-1), None, None))
                .is_err()
        );
        assert_eq!(
            auto_compact_token_limit_for_model(
                &state,
                &model_for_settings(Some(100_000), Some(100_000), Some(0), None),
            )
            .unwrap(),
            Some(0)
        );
        assert_eq!(
            auto_compact_token_limit_for_model(
                &state,
                &model_for_settings(Some(100_000), Some(100_000), Some(-1), None),
            )
            .unwrap(),
            Some(-1)
        );
        assert_eq!(
            auto_compact_token_limit_for_model(
                &state,
                &model_for_settings(Some(1), Some(1), None, None),
            )
            .unwrap(),
            Some(0)
        );

        let configured = helper_state(
            "live-settings-model",
            test_codex_config(None, Some(50_000), None),
        );
        assert_eq!(
            context_window_for_model(
                &configured,
                &model_for_settings(Some(0), Some(100_000), None, None),
            )
            .unwrap(),
            Some(47_500)
        );
        assert!(context_window_for_model(
            &configured,
            &model_for_settings(Some(100_000), Some(0), None, None),
        )
        .is_err());
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
        let status = response.status();
        let response_body = response.text().await.unwrap();
        assert_eq!(status, reqwest::StatusCode::OK, "{response_body}");
        let response_body: Value = serde_json::from_str(&response_body).unwrap();
        assert_eq!(response_body["model"], json!("gpt-5.6-sol"));
        assert!(response_body["choices"][0]["logprobs"].is_null());
        assert!(response_body["choices"][0]["message"]["refusal"].is_null());

        let requests = recording.requests();
        assert_eq!(requests.len(), 1);
        let recorded = &requests[0];
        let contract: Value =
            serde_json::from_str(include_str!("../../config/codex-upstream-contract.json"))
                .unwrap();
        let request_contract = &contract["responses_request"];
        let lite_contract = &contract["responses_lite"];
        let originator_contract = &contract["headers"]["originator"];
        assert_eq!(recorded.endpoint, RecordedEndpoint::Responses);
        assert_eq!(
            recorded.method.as_str(),
            request_contract["method"].as_str().unwrap()
        );
        assert_eq!(recorded.path, request_contract["path"].as_str().unwrap());
        assert_eq!(
            recorded.headers.get("accept").unwrap().to_str().unwrap(),
            request_contract["streaming_accept"].as_str().unwrap()
        );
        assert_eq!(
            recorded
                .headers
                .get(originator_contract["name"].as_str().unwrap())
                .unwrap()
                .to_str()
                .unwrap(),
            originator_contract["value"].as_str().unwrap()
        );
        assert_eq!(
            recorded
                .headers
                .get(lite_contract["header"]["name"].as_str().unwrap())
                .unwrap()
                .to_str()
                .unwrap(),
            lite_contract["header"]["value"].as_str().unwrap()
        );
        assert_eq!(recorded.body["model"], json!("gpt-5.6-sol"));
        assert_eq!(recorded.body["reasoning"]["effort"], json!("max"));
        assert_eq!(
            recorded.body["reasoning"]["context"],
            lite_contract["reasoning_context"]
        );
        assert!(recorded.body["include"]
            .as_array()
            .unwrap()
            .contains(&request_contract["reasoning_encrypted_content_include"]));
        assert_eq!(recorded.body["text"], json!({"verbosity": "low"}));
        assert_eq!(recorded.body["tool_choice"], json!("auto"));
        assert_eq!(
            recorded.body["parallel_tool_calls"],
            lite_contract["parallel_tool_calls"]
        );
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
        assert_eq!(body["error"]["type"], json!("invalid_request_error"));
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
        assert_eq!(response_body["model"], json!("gpt-5.6-sol"));
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
            json!("medium")
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
    async fn chat_route_handles_completion_terminal_semantics_and_reasoning_mismatch() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        recording.set_response_output(vec![]);
        recording.set_completed_end_turn(None);
        recording.set_response_usage(json!({
            "input_tokens": 1,
            "output_tokens": 0,
            "total_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0}
        }));
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();
        let request = json!({
            "model": "gpt-5.6-sol",
            "messages": [
                {"role": "system", "content": "Follow the request."},
                {"role": "user", "content": "Return nothing."}
            ],
            "responses_lite": false
        });

        let terminal_without_end_turn = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&request)
            .send()
            .await
            .unwrap();
        let terminal_status = terminal_without_end_turn.status();
        let terminal_body: Value = terminal_without_end_turn.json().await.unwrap();
        assert_eq!(terminal_status, reqwest::StatusCode::OK, "{terminal_body}");
        assert_eq!(terminal_body["choices"][0]["finish_reason"], "stop");

        recording.set_completed_end_turn(Some(false));

        let empty = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&request)
            .send()
            .await
            .unwrap();
        let empty_status = empty.status();
        let empty_body: Value = empty.json().await.unwrap();
        assert_eq!(
            empty_status,
            reqwest::StatusCode::BAD_GATEWAY,
            "{empty_body}"
        );
        assert_eq!(empty_body["error"]["type"], "upstream_protocol_error");

        let mut stream_request = request.clone();
        stream_request["stream"] = json!(true);
        let stream = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&stream_request)
            .send()
            .await
            .unwrap();
        assert_eq!(stream.status(), reqwest::StatusCode::OK);
        let stream_body = stream.text().await.unwrap();
        assert!(stream_body.contains("upstream_protocol_error"));
        assert!(!stream_body.contains("[DONE]"));

        recording.set_response_output(vec![json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "not terminal"}]
        })]);
        let not_terminal = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&request)
            .send()
            .await
            .unwrap();
        assert_eq!(not_terminal.status(), reqwest::StatusCode::BAD_GATEWAY);
        let not_terminal_body: Value = not_terminal.json().await.unwrap();
        assert_eq!(
            not_terminal_body["error"]["type"],
            "upstream_protocol_error"
        );

        recording.set_completed_end_turn(Some(true));
        recording.set_response_prefix_events(vec![json!({
            "type": "response.reasoning_summary_text.delta",
            "delta": "streamed summary",
            "summary_index": 0
        })]);
        recording.set_response_output(vec![json!({
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "different final summary"}]
        })]);
        let reasoning = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&request)
            .send()
            .await
            .unwrap();
        assert_eq!(reasoning.status(), reqwest::StatusCode::BAD_GATEWAY);
        let reasoning_body: Value = reasoning.json().await.unwrap();
        assert_eq!(reasoning_body["error"]["type"], "upstream_protocol_error");

        recording.set_response_prefix_events(vec![json!({
            "type": "response.reasoning_text.delta",
            "delta": "streamed raw",
            "content_index": 0
        })]);
        recording.set_response_output(vec![json!({
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "different raw"}]
        })]);
        let raw_reasoning = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&request)
            .send()
            .await
            .unwrap();
        assert_eq!(raw_reasoning.status(), reqwest::StatusCode::BAD_GATEWAY);
        let raw_reasoning_body: Value = raw_reasoning.json().await.unwrap();
        assert_eq!(
            raw_reasoning_body["error"]["type"],
            "upstream_protocol_error"
        );

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn openai_chat_rejects_web_search_and_image_generation_provider_items() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();
        let request = json!({
            "model": "gpt-5.5",
            "messages": [{"role": "user", "content": "hello"}],
            "responses_lite": false,
            "codex_metadata": false
        });

        for output in [
            json!({
                "type": "web_search_call",
                "id": "search-1",
                "status": "completed",
                "action": {"type": "search", "query": "q", "sources": []}
            }),
            json!({
                "type": "image_generation_call",
                "id": "image-1",
                "status": "completed",
                "result": "data:image/png;base64,AAAA"
            }),
        ] {
            recording.set_response_output(vec![output]);

            let response = client
                .post(format!("{api_url}/v1/chat/completions"))
                .json(&request)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::BAD_GATEWAY);
            let body: Value = response.json().await.unwrap();
            assert_eq!(body["error"]["type"], "upstream_protocol_error");

            let mut streaming_request = request.clone();
            streaming_request["stream"] = json!(true);
            let stream_response = client
                .post(format!("{api_url}/v1/chat/completions"))
                .json(&streaming_request)
                .send()
                .await
                .unwrap();
            assert_eq!(stream_response.status(), reqwest::StatusCode::OK);
            let events = parse_sse_json_events(&stream_response.text().await.unwrap());
            let errors: Vec<&Value> = events
                .iter()
                .filter(|event| event.get("error").is_some())
                .collect();
            assert_eq!(errors.len(), 1);
            assert_eq!(errors[0]["error"]["type"], "upstream_protocol_error");
        }

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

        assert_eq!(response.status(), reqwest::StatusCode::BAD_GATEWAY);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], json!("upstream_protocol_error"));
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
        assert_eq!(inspect_response.status(), reqwest::StatusCode::BAD_GATEWAY);
        let inspect_body: Value = inspect_response.json().await.unwrap();
        assert_eq!(
            inspect_body["error"]["type"],
            json!("upstream_protocol_error")
        );
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

        assert_eq!(response.status(), reqwest::StatusCode::BAD_GATEWAY);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], json!("upstream_protocol_error"));
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
    async fn anthropic_routes_reject_hosted_web_search_before_provider_work() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;

        for responses_lite in ["auto", "on", "off"] {
            for route in [
                "/v1/messages",
                "/v1/messages/count_tokens",
                "/v1/messages/compact",
            ] {
                let mut body = json!({
                    "model": "claude-sonnet-4-5",
                    "system": "Use the available tools.",
                    "messages": [{"role": "user", "content": "Search the web."}],
                    "tools": [{"type": "web_search_20250305", "name": "web_search"}],
                    "max_tokens": 100
                });
                if route == "/v1/messages" {
                    body["stream"] = json!(true);
                    body["responses_lite"] = json!(responses_lite);
                } else if route == "/v1/messages/compact" {
                    body["responses_lite"] = json!(responses_lite);
                }
                let response = reqwest::Client::new()
                    .post(format!("{api_url}{route}"))
                    .header("x-claude-code-session-id", "test-session")
                    .json(&body)
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
                assert!(body["error"]["message"]
                    .as_str()
                    .unwrap()
                    .contains("cannot be represented losslessly"));
            }
        }
        assert!(recording.requests().is_empty());

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_nonstream_requires_authoritative_final_usage_and_finish_reason() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;

        for missing in ["usage", "finish_reason"] {
            recording.set_response_usage(if missing == "usage" {
                Value::Null
            } else {
                json!({
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 0}
                })
            });
            recording.set_completed_end_turn(if missing == "finish_reason" {
                Some(false)
            } else {
                Some(true)
            });
            let response = reqwest::Client::new()
                .post(format!("{api_url}/v1/messages"))
                .header("x-claude-code-session-id", "test-session")
                .json(&json!({
                    "model": "gpt-5.5",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hello"}]
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::BAD_GATEWAY);
            let body: Value = response.json().await.unwrap();
            assert_eq!(body["error"]["type"], json!("api_error"));
        }

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_stream_final_contract_failures_end_with_error_not_message_stop() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;

        for missing in ["usage", "finish_reason"] {
            recording.set_response_usage(if missing == "usage" {
                Value::Null
            } else {
                json!({
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 0}
                })
            });
            recording.set_completed_end_turn(if missing == "finish_reason" {
                Some(false)
            } else {
                Some(true)
            });
            let response = reqwest::Client::new()
                .post(format!("{api_url}/v1/messages"))
                .header("x-claude-code-session-id", "test-session")
                .json(&json!({
                    "model": "gpt-5.5",
                    "max_tokens": 64,
                    "stream": true,
                    "messages": [{"role": "user", "content": "hello"}]
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::OK);
            let mut pending = response.text().await.unwrap();
            let mut events = Vec::new();
            parse_anthropic_sse_blocks(&mut pending, &mut events);
            assert!(pending.is_empty());
            assert_eq!(events.first().unwrap().0, "message_start");
            assert_eq!(events.last().unwrap().0, "error");
            assert!(events.iter().all(|(event, _)| event != "message_stop"));
        }

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
            .header("x-claude-code-session-id", "test-session")
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
            .header("x-claude-code-session-id", "test-session")
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
                .header("x-claude-code-session-id", "test-session")
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
    async fn anthropic_stream_bounds_the_total_upstream_error_body_read() {
        let (upstream_url, upstream_handle) = start_slow_error_body_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server_with_timeout(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
            Duration::from_millis(50),
        )
        .await;
        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/messages"))
            .header("x-claude-code-session-id", "test-session")
            .json(&json!({
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "Trigger the error."}],
                "max_tokens": 64,
                "stream": true,
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::OK);

        let mut pending = tokio::time::timeout(Duration::from_millis(500), response.text())
            .await
            .expect("slow upstream error body exceeded the configured total timeout")
            .unwrap();
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
    async fn anthropic_nonstream_masks_upstream_error_details() {
        let (upstream_url, upstream_handle) = start_fixed_status_upstream(500).await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let response = reqwest::Client::new()
            .post(format!("{api_url}/v1/messages"))
            .header("x-claude-code-session-id", "test-session")
            .json(&json!({
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "Trigger the error."}],
                "max_tokens": 64,
                "stream": false,
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.status(),
            reqwest::StatusCode::INTERNAL_SERVER_ERROR
        );
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["message"], json!("upstream request failed"));
        assert!(!body.to_string().contains("upstream rejected request"));

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
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
            client
                .post(format!("{api_url}/v1/messages"))
                .header("x-claude-code-session-id", "test-session")
                .json(&json!({
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
        assert_eq!(body["error"]["type"], json!("invalid_request_error"));
        assert!(recording.requests().is_empty());

        for (path, body) in [
            (
                "/v1/chat/completions",
                json!({
                    "model": "gpt-5.6-sol",
                    "messages": [{"role": "user", "content": "Answer."}],
                    "responses_lite": null
                }),
            ),
            (
                "/v1/images/generations",
                json!({"prompt": "draw", "responses_lite": null}),
            ),
            (
                "/v1/inspect",
                json!({
                    "prompt": "inspect",
                    "images": [{"image_url": "data:image/png;base64,AAAA"}],
                    "responses_lite": null
                }),
            ),
        ] {
            let response = reqwest::Client::new()
                .post(format!("{api_url}{path}"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                reqwest::StatusCode::BAD_REQUEST,
                "{path}"
            );
            let error: Value = response.json().await.unwrap();
            assert_eq!(error["error"]["type"], json!("invalid_request_error"));
        }
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
        assert_eq!(body["error"]["type"], json!("invalid_request_error"));
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
            "result": "https://example.test/image.png",
            "revised_prompt": "A lighthouse"
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
                "reference_images": [{
                    "image_url": "data:image/png;base64,BBBB",
                    "detail": "high",
                    "prompt_cache_breakpoint": null
                }],
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
        assert_eq!(
            requests[0].body["input"][0]["content"][1],
            json!({
                "type": "input_image",
                "image_url": "data:image/png;base64,BBBB",
                "detail": "high"
            })
        );

        for optional_fields in [json!({}), json!({"id": null, "revised_prompt": null})] {
            let mut output = json!({
                "type": "image_generation_call",
                "status": "completed",
                "result": "https://example.test/image.png"
            });
            output
                .as_object_mut()
                .unwrap()
                .extend(optional_fields.as_object().unwrap().clone());
            recording.set_response_output(vec![output]);
            let optional = reqwest::Client::new()
                .post(format!("{api_url}/v1/images/generations"))
                .json(&json!({
                    "model": "gpt-5.6-sol",
                    "prompt": "A lighthouse",
                    "responses_lite": false
                }))
                .send()
                .await
                .unwrap();
            assert_eq!(optional.status(), reqwest::StatusCode::OK);
            let optional_body: Value = optional.json().await.unwrap();
            assert_eq!(
                optional_body["data"],
                json!([{"url": "https://example.test/image.png"}])
            );
        }

        let auto_size = reqwest::Client::new()
            .post(format!("{api_url}/v1/images/generations"))
            .json(&json!({
                "model": "gpt-5.6-sol",
                "prompt": "A lighthouse",
                "size": "auto",
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(auto_size.status(), reqwest::StatusCode::OK);

        for invalid_body in [
            json!({
                "model": "gpt-5.6-sol",
                "prompt": "A lighthouse",
                "size": "1024x1024"
            }),
            json!({
                "model": "gpt-5.6-sol",
                "prompt": "A lighthouse",
                "tools": []
            }),
            json!({
                "model": "gpt-5.6-sol",
                "prompt": "A lighthouse",
                "reference_images": "not-an-array"
            }),
            json!({
                "model": "gpt-5.6-sol",
                "prompt": "A lighthouse",
                "reference_images": [{
                    "image_url": "data:image/png;base64,BBBB",
                    "unknown": true
                }]
            }),
            json!({
                "model": "gpt-5.6-sol",
                "prompt": "A lighthouse",
                "reference_images": [{
                    "image_url": "data:image/png;base64,BBBB",
                    "detail": "full"
                }]
            }),
            json!({
                "model": "gpt-5.6-sol",
                "prompt": "A lighthouse",
                "reference_images": [{
                    "image_url": "data:image/png;base64,BBBB",
                    "prompt_cache_breakpoint": true
                }]
            }),
        ] {
            let invalid = reqwest::Client::new()
                .post(format!("{api_url}/v1/images/generations"))
                .json(&invalid_body)
                .send()
                .await
                .unwrap();
            assert_eq!(invalid.status(), reqwest::StatusCode::BAD_REQUEST);
        }

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
        assert_eq!(recording.requests().len(), 4);

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
            .header("x-claude-code-session-id", "test-session")
            .json(&json!({
                "model": "gpt-5.6-sol",
                "max_tokens": 128,
                "system": "Compact precisely.",
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
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "Compact precisely."}]
                },
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
    async fn messages_compact_rejects_stop_sequences_and_non_auto_tool_choices_before_upstream() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();

        for unsupported in [
            json!({"stop_sequences": ["stop"]}),
            json!({"tool_choice": {"type": "any"}}),
            json!({"tool_choice": {"type": "tool", "name": "lookup"}}),
            json!({"tool_choice": {"type": "none"}}),
        ] {
            let mut body = json!({
                "model": "gpt-5.5",
                "max_tokens": 128,
                "messages": [{"role": "user", "content": "History"}],
            });
            body.as_object_mut()
                .unwrap()
                .extend(unsupported.as_object().unwrap().clone());
            let response = client
                .post(format!("{api_url}/v1/messages/compact"))
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
    async fn messages_compact_requires_a_positive_safe_integral_max_tokens_before_upstream() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();

        for value in [
            None,
            Some(Value::Null),
            Some(json!(0)),
            Some(json!(-1)),
            Some(json!(1.5)),
            Some(json!(9_007_199_254_740_992_i64)),
        ] {
            let mut body = json!({
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": "History"}]
            });
            if let Some(value) = value {
                body.as_object_mut()
                    .unwrap()
                    .insert("max_tokens".to_string(), value);
            }
            let response = client
                .post(format!("{api_url}/v1/messages/compact"))
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
    async fn messages_compact_routes_live_gpt_and_explicit_claude_backend_models() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();

        for model in ["gpt-5.6-sol", "claude-fable-5"] {
            let response = client
                .post(format!("{api_url}/v1/messages/compact"))
                .header("x-claude-code-session-id", "test-session")
                .json(&json!({
                    "model": model,
                    "max_tokens": 128,
                    "system": "Compact precisely.",
                    "messages": [{"role": "user", "content": "History"}],
                    "speed": if model == "gpt-5.6-sol" { "fast" } else { "standard" },
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
            .header("x-claude-code-session-id", "test-session")
            .json(&json!({
                "model": "gpt-5.6-sol",
                "max_tokens": 128,
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
            .header("x-claude-code-session-id", "test-session")
            .json(&json!({
                "model": "gpt-5.6-sol",
                "max_tokens": 128,
                "system": "Compact precisely.",
                "messages": [{"role": "user", "content": "History"}],
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "name": "structured_output",
                        "schema": schema.clone()
                    }
                },
                "text": {"verbosity": "high"},
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        let status = response.status();
        let response_body = response.text().await.unwrap();
        assert_eq!(status, reqwest::StatusCode::OK, "{response_body}");
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
            .header("x-claude-code-session-id", "test-session")
            .json(&json!({
                "model": "gpt-5.6-sol",
                "max_tokens": 128,
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
        assert_eq!(response.status(), reqwest::StatusCode::SERVICE_UNAVAILABLE);
        let body: Value = response.json().await.unwrap();
        assert_eq!(body["error"]["type"], json!("server_error"));
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
            json!({"effort": "medium", "summary": "auto", "context": "all_turns"})
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
        assert_eq!(inspect_request.body["reasoning"]["effort"], json!("medium"));
        assert_eq!(
            inspect_request.body["reasoning"]["context"],
            json!("all_turns")
        );

        let invalid_compact_response = reqwest::Client::new()
            .post(format!("{api_url}/v1/compact"))
            .json(&json!({
                "messages": [
                    {"role": "system", "content": "Compact precisely."},
                    {"role": "user", "content": "History"}
                ],
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
            json!({"effort": "high", "summary": "auto", "context": "all_turns"})
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
            json!({"safety_identifier": null}),
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
    async fn safety_identifier_null_is_rejected_before_upstream_on_every_accepting_facade() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.6-sol",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();

        for (route, body) in [
            (
                "/v1/images/generations",
                json!({"prompt": "draw", "safety_identifier": null}),
            ),
            (
                "/v1/inspect",
                json!({
                    "prompt": "inspect",
                    "images": [{"image_url": "data:image/png;base64,AAAA"}],
                    "safety_identifier": null
                }),
            ),
            (
                "/v1/compact",
                json!({
                    "messages": [{"role": "user", "content": "hello"}],
                    "safety_identifier": null
                }),
            ),
            (
                "/v1/messages",
                json!({
                    "model": "claude-sonnet-4-5",
                    "max_tokens": 128,
                    "messages": [{"role": "user", "content": "hello"}],
                    "safety_identifier": null
                }),
            ),
        ] {
            let response = client
                .post(format!("{api_url}{route}"))
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(
                response.status(),
                reqwest::StatusCode::BAD_REQUEST,
                "{route}"
            );
        }
        assert!(recording.requests().is_empty());

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
            "model": "gpt-5.6-sol",
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
            "reasoning": {"effort": "medium", "context": "current_turn"},
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
            json!({"effort": "medium", "summary": "auto", "context": "current_turn"})
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
            "summary": [{"type": "summary_text", "text": "actual reasoning"}],
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
                "reasoning": {"effort": "medium", "context": "all_turns"},
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
            .header("x-claude-code-session-id", "test-session")
            .json(&json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 128,
                "system": "Be exact.",
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning": {"effort": "medium", "context": "current_turn"},
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
                "messages": [
                    {"role": "system", "content": "Compact precisely."},
                    {"role": "user", "content": "History"}
                ],
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
            .header("x-claude-code-session-id", "test-session")
            .json(&json!({
                "model": "gpt-5.6-sol",
                "max_tokens": 128,
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn chat_identity_drives_codex_metadata_and_prompt_cache_affinity() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();
        let messages = json!([
            {"role": "system", "content": "Be precise."},
            {"role": "user", "content": "Answer."}
        ]);

        let missing_identity = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.5",
                "messages": messages,
                "codex_metadata": true,
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(missing_identity.status(), reqwest::StatusCode::BAD_REQUEST);
        assert!(recording.requests().is_empty());

        let root = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.5",
                "messages": messages,
                "client_metadata": {
                    "session_id": "session-root",
                    "caller": "test"
                },
                "codex_metadata": true,
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(root.status(), reqwest::StatusCode::OK);

        let child = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.5",
                "messages": messages,
                "prompt_cache_key": "explicit-cache-key",
                "client_metadata": {
                    "session_id": "session-root",
                    "thread_id": "thread-child"
                },
                "codex_metadata": true,
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(child.status(), reqwest::StatusCode::OK);

        let anonymous = client
            .post(format!("{api_url}/v1/chat/completions"))
            .json(&json!({
                "model": "gpt-5.5",
                "messages": messages,
                "codex_metadata": false,
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(anonymous.status(), reqwest::StatusCode::OK);

        let requests = recording.requests();
        assert_eq!(requests.len(), 3);
        assert_eq!(requests[0].body["prompt_cache_key"], "session-root");
        assert_eq!(requests[1].body["prompt_cache_key"], "explicit-cache-key");
        assert!(requests[2].body.get("prompt_cache_key").is_none());

        let root_metadata = requests[0].body["client_metadata"].as_object().unwrap();
        let child_metadata = requests[1].body["client_metadata"].as_object().unwrap();
        assert_eq!(root_metadata["session_id"], "session-root");
        assert_eq!(root_metadata["thread_id"], "session-root");
        assert_eq!(root_metadata["caller"], "test");
        assert_eq!(child_metadata["session_id"], "session-root");
        assert_eq!(child_metadata["thread_id"], "thread-child");
        assert_eq!(
            root_metadata["x-codex-installation-id"],
            child_metadata["x-codex-installation-id"]
        );
        assert_eq!(
            root_metadata["x-codex-window-id"],
            child_metadata["x-codex-window-id"]
        );
        assert_ne!(
            root_metadata["turn_id"], child_metadata["turn_id"],
            "turn_id must be fresh for each request"
        );

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn anthropic_session_cache_affinity_and_cache_controls_wire_end_to_end() {
        let (upstream_url, recording, upstream_handle) = start_recording_upstream().await;
        let (api_url, auth_path, api_handle) = start_api_server(
            &upstream_url,
            "gpt-5.5",
            test_codex_config(None, None, None),
        )
        .await;
        let client = reqwest::Client::new();
        let controlled_body = json!({
            "model": "claude-sonnet-4-5",
            "max_tokens": 64,
            "cache_control": {"type": "ephemeral"},
            "system": [{
                "type": "text",
                "text": "Be precise.",
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }],
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Answer.",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"}
                }]
            }],
            "tools": [{
                "name": "lookup",
                "description": "Look something up",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"}
            }],
            "responses_lite": false
        });

        let first = client
            .post(format!("{api_url}/v1/messages"))
            .header("x-claude-code-session-id", "session-a")
            .json(&controlled_body)
            .send()
            .await
            .unwrap();
        assert_eq!(first.status(), reqwest::StatusCode::OK);

        let mut streaming_body = controlled_body.clone();
        streaming_body["stream"] = json!(true);
        let second = client
            .post(format!("{api_url}/v1/messages"))
            .header("x-claude-code-session-id", "session-a")
            .json(&streaming_body)
            .send()
            .await
            .unwrap();
        assert_eq!(second.status(), reqwest::StatusCode::OK);
        let _ = second.text().await.unwrap();

        let third = client
            .post(format!("{api_url}/v1/messages"))
            .header("x-claude-code-session-id", "session-b")
            .json(&controlled_body)
            .send()
            .await
            .unwrap();
        assert_eq!(third.status(), reqwest::StatusCode::OK);

        let mut explicit_body = controlled_body.clone();
        explicit_body["prompt_cache_key"] = json!("caller-cache-key");
        let explicit = client
            .post(format!("{api_url}/v1/messages"))
            .header("x-claude-code-session-id", "session-a")
            .json(&explicit_body)
            .send()
            .await
            .unwrap();
        assert_eq!(explicit.status(), reqwest::StatusCode::OK);

        let mut no_session_body = controlled_body.clone();
        no_session_body["previous_response_id"] = Value::Null;
        let no_session = client
            .post(format!("{api_url}/v1/messages"))
            .json(&no_session_body)
            .send()
            .await
            .unwrap();
        assert_eq!(no_session.status(), reqwest::StatusCode::BAD_REQUEST);

        let requests = recording.requests();
        assert_eq!(requests.len(), 4);
        let session_a_key = "fed3534a8ed1887fdc51648477597f2a98ec264b033271334062e9cba3c23fae";
        let session_b_key = "00d0e78bab5b2ca951064120f1c2d86d481250a6814637aeec71f12b6aed398b";
        assert_eq!(requests[0].body["prompt_cache_key"], session_a_key);
        assert_eq!(requests[1].body["prompt_cache_key"], session_a_key);
        assert_eq!(requests[2].body["prompt_cache_key"], session_b_key);
        assert_eq!(requests[3].body["prompt_cache_key"], "caller-cache-key");
        for request in &requests {
            assert!(request.body.get("client_metadata").is_none());
            assert!(request.body.get("previous_response_id").is_none());
        }
        assert!(!has_nested_key(&requests[0].body, "cache_control"));

        let before_invalid = recording.requests().len();
        let invalid_requests = [
            json!({"previous_response_id": "resp-prior"}),
            json!({"prompt_cache_key": ""}),
            json!({"prompt_cache_key": 42}),
            json!({"cache_control": {"type": "persistent"}}),
            json!({
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "text",
                        "text": "Answer.",
                        "cache_control": {"type": "ephemeral", "ttl": "2h"}
                    }]
                }]
            }),
        ];
        for fields in invalid_requests {
            let mut body = json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "system": "Be precise.",
                "messages": [{"role": "user", "content": "Answer."}],
                "responses_lite": false
            });
            body.as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            let response = client
                .post(format!("{api_url}/v1/messages"))
                .header("x-claude-code-session-id", "session-a")
                .json(&body)
                .send()
                .await
                .unwrap();
            assert_eq!(response.status(), reqwest::StatusCode::BAD_REQUEST);
            let error: Value = response.json().await.unwrap();
            assert_eq!(error["type"], "error");
            assert_eq!(error["error"]["type"], "invalid_request_error");
        }

        let empty_session = client
            .post(format!("{api_url}/v1/messages"))
            .header("x-claude-code-session-id", " ")
            .json(&json!({
                "model": "claude-sonnet-4-5",
                "max_tokens": 64,
                "system": "Be precise.",
                "messages": [{"role": "user", "content": "Answer."}],
                "responses_lite": false
            }))
            .send()
            .await
            .unwrap();
        assert_eq!(empty_session.status(), reqwest::StatusCode::BAD_REQUEST);
        assert_eq!(recording.requests().len(), before_invalid);

        api_handle.abort();
        upstream_handle.abort();
        std::fs::remove_file(auth_path).unwrap();
    }
}
