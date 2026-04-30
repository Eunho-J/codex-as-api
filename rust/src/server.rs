use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use futures::stream;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::task;

use crate::anthropic_adapter::{
    anthropic_request_to_internal, anthropic_stream_adapter, format_anthropic_error,
    internal_response_to_anthropic,
};
use crate::auth::{self, AuthError};
use crate::messages::{Message, MessageRole, ToolCall, ToolSchema};
use crate::provider::{ChatGPTOAuthProvider, ProviderError};

#[derive(Clone)]
pub struct AppState {
    pub model: String,
    pub auth_path: Option<String>,
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
    pub prompt_cache_key: Option<String>,
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
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct InspectRequest {
    pub prompt: Option<String>,
    pub images: Option<Vec<HashMap<String, String>>>,
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CompactRequest {
    pub messages: Option<Vec<ChatMessage>>,
    pub reasoning_effort: Option<String>,
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

fn map_error_status(e: &ProviderError) -> StatusCode {
    match e {
        ProviderError::Auth(AuthError::Missing(_)) => StatusCode::UNAUTHORIZED,
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
        .route("/v1/messages", post(anthropic_messages))
        .with_state(state)
}

fn openai_model_id(model: &str) -> String {
    format!("codex-oauth:{}", model)
}

fn request_messages_to_internal(messages: &[ChatMessage]) -> Vec<Message> {
    let mut result = Vec::new();
    for msg in messages {
        let role = map_role(&msg.role);
        let content = normalize_content(&msg.content);
        let tool_calls = parse_tool_calls(&msg.tool_calls);
        let m = Message {
            role,
            content,
            tool_calls,
            tool_call_id: msg.tool_call_id.clone(),
            name: msg.name.clone(),
            reasoning_content: None,
        };
        result.push(m);
    }
    result
}

fn map_role(role: &str) -> MessageRole {
    match role.to_lowercase().as_str() {
        "system" => MessageRole::System,
        "assistant" => MessageRole::Assistant,
        "tool" => MessageRole::Tool,
        _ => MessageRole::User,
    }
}

fn normalize_content(content: &Option<Value>) -> String {
    match content {
        None => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Array(arr)) => {
            let mut parts = Vec::new();
            for item in arr {
                if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                    parts.push(text.to_string());
                }
            }
            parts.join("")
        }
        Some(other) => other.to_string(),
    }
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

fn parse_tools(raw: &Option<Vec<Value>>) -> Option<Vec<ToolSchema>> {
    let items = match raw {
        Some(v) if !v.is_empty() => v,
        _ => return None,
    };
    let mut schemas = Vec::new();
    for item in items {
        let obj = match item.as_object() {
            Some(o) => o,
            None => continue,
        };
        let func = obj.get("function").and_then(|v| v.as_object()).unwrap_or(obj);
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
    if schemas.is_empty() {
        None
    } else {
        Some(schemas)
    }
}

fn max_tokens_from_request(req: &ChatCompletionRequest) -> Option<i64> {
    req.max_completion_tokens.or(req.max_tokens)
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    let auth_available = auth::is_auth_locally_available(state.auth_path.as_deref());
    Json(json!({
        "status": "ok",
        "auth_available": auth_available,
        "model": state.model,
    }))
}

async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<axum::response::Response, axum::response::Response> {
    let messages = request_messages_to_internal(&request.messages);
    let tools = parse_tools(&request.tools);
    let stop = request.stop.as_ref().map(|s| s.to_vec());
    let max_tokens = max_tokens_from_request(&request);

    let subagent = request
        .subagent
        .clone()
        .or_else(|| {
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

    if request.stream {
        let provider = state.provider.clone();
        let model_id = openai_model_id(&state.model);
        let request_id = format!("chatcmpl-{}", &uuid::Uuid::new_v4().simple().to_string()[..24]);
        let created = chrono::Utc::now().timestamp();
        let temperature = request.temperature;
        let reasoning_effort = request.reasoning_effort.clone();
        let prompt_cache_key = request.prompt_cache_key.clone();
        let request_model = request.model.clone();
        let tool_choice = request.tool_choice.clone();

        let service_tier = request.service_tier.clone();
        let text = request.text.clone();
        let client_metadata = request.client_metadata.clone();

        let result = task::spawn_blocking(move || {
            let tools_ref = tools.as_deref();
            let stop_ref: Option<Vec<String>> = stop;
            let stop_slice = stop_ref.as_deref();

            provider.chat_stream(
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
        sse_events.push(Ok(Event::default().data(serde_json::to_string(&preamble).unwrap())));

        let mut usage_dict: Option<Value> = None;

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
                    sse_events.push(Ok(Event::default().data(serde_json::to_string(&chunk).unwrap())));
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
                    sse_events.push(Ok(Event::default().data(serde_json::to_string(&chunk).unwrap())));
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
                    sse_events.push(Ok(Event::default().data(serde_json::to_string(&chunk).unwrap())));
                }
                "tool_call" => {
                    let tc = json!({
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
                    sse_events.push(Ok(Event::default().data(serde_json::to_string(&chunk).unwrap())));
                }
                "finish" => {
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
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason,
                        }],
                    });
                    sse_events.push(Ok(Event::default().data(serde_json::to_string(&chunk).unwrap())));
                }
                _ => {}
            }
        }

        if let Some(u) = &usage_dict {
            let finish_chunk = json!({
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_id,
                "choices": [],
                "usage": {
                    "prompt_tokens": u.get("prompt_tokens").and_then(|v| v.as_i64()).unwrap_or(0),
                    "completion_tokens": u.get("completion_tokens").and_then(|v| v.as_i64()).unwrap_or(0),
                    "total_tokens": u.get("total_tokens").and_then(|v| v.as_i64()).unwrap_or(0),
                },
            });
            sse_events.push(Ok(Event::default().data(serde_json::to_string(&finish_chunk).unwrap())));
        }

        sse_events.push(Ok(Event::default().data("[DONE]")));

        let sse = Sse::new(stream::iter(sse_events));
        Ok(sse.into_response())
    } else {
        let provider = state.provider.clone();
        let model_id = openai_model_id(&state.model);
        let temperature = request.temperature;
        let reasoning_effort = request.reasoning_effort.clone();
        let prompt_cache_key = request.prompt_cache_key.clone();
        let request_model = request.model.clone();
        let tool_choice = request.tool_choice.clone();
        let service_tier = request.service_tier.clone();
        let text = request.text.clone();
        let client_metadata = request.client_metadata.clone();

        let result = task::spawn_blocking(move || {
            let tools_ref = tools.as_deref();
            let stop_ref: Option<Vec<String>> = stop;
            let stop_slice = stop_ref.as_deref();

            provider.chat(
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
    let provider = state.provider.clone();
    let prompt = request.prompt.clone();
    let size = request.size.clone();
    let reasoning_effort = request.reasoning_effort.clone();
    let request_model = request.model.clone();

    let result = task::spawn_blocking(move || {
        provider.generate_image(
            &prompt,
            &[],
            size.as_deref(),
            reasoning_effort.as_deref(),
            Some(request_model.as_str()),
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
    let provider = state.provider.clone();
    let prompt = request.prompt.clone().unwrap_or_default();
    let images = request.images.clone().unwrap_or_default();
    let reasoning_effort = request.reasoning_effort.clone();

    let result = task::spawn_blocking(move || {
        provider.inspect_images(&prompt, &images, reasoning_effort.as_deref(), None)
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
    Json(request): Json<CompactRequest>,
) -> Result<Json<Value>, axum::response::Response> {
    let provider = state.provider.clone();
    let raw_messages = request.messages.unwrap_or_default();
    let messages = request_messages_to_internal(&raw_messages);
    let reasoning_effort = request.reasoning_effort.clone();

    let result = task::spawn_blocking(move || {
        provider.compact_messages(&messages, reasoning_effort.as_deref(), None)
    })
    .await
    .unwrap();

    let checkpoint = result.map_err(|e| {
        let status = map_error_status(&e);
        error_response(status, e.to_string()).into_response()
    })?;

    Ok(Json(json!({"checkpoint": checkpoint})))
}

async fn anthropic_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, axum::response::Response> {
    let request_id = format!(
        "msg_{}",
        &uuid::Uuid::new_v4().simple().to_string()[..24]
    );

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

    let (messages, tools, tool_choice, stop, reasoning_effort) =
        anthropic_request_to_internal(&body);

    let stream = body
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let backend_model = state.model.clone();
    let model_for_response = body
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or(&backend_model)
        .to_string();

    let max_tokens = body
        .get("max_tokens")
        .and_then(|v| v.as_i64());

    let tool_choice_val: Option<Value> = tool_choice;

    if stream {
        let provider = state.provider.clone();
        let backend_model_clone = backend_model.clone();

        let result = task::spawn_blocking(move || {
            let tools_ref = tools.as_deref();
            let stop_ref = stop.as_deref();
            let tc_ref = tool_choice_val.as_ref();

            provider.chat_stream(
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
                Some(backend_model_clone.as_str()),
                tc_ref,
                None,
                None,
                None,
            )
        })
        .await
        .unwrap();

        let events = match result {
            Ok(evts) => evts,
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

        let sse_strings = anthropic_stream_adapter(&events, &model_for_response, &request_id);

        let sse_events: Vec<Result<Event, std::convert::Infallible>> = sse_strings
            .into_iter()
            .map(|chunk| {
                let mut lines = chunk.trim_end_matches('\n').splitn(2, '\n');
                let event_type = lines
                    .next()
                    .and_then(|l| l.strip_prefix("event: "))
                    .unwrap_or("message")
                    .to_string();
                let data = lines
                    .next()
                    .and_then(|l| l.strip_prefix("data: "))
                    .unwrap_or("")
                    .to_string();
                Ok(Event::default().event(event_type).data(data))
            })
            .collect();

        let sse = Sse::new(stream::iter(sse_events));
        Ok(sse.into_response())
    } else {
        let provider = state.provider.clone();
        let backend_model_clone = backend_model.clone();

        let result = task::spawn_blocking(move || {
            let tools_ref = tools.as_deref();
            let stop_ref = stop.as_deref();
            let tc_ref = tool_choice_val.as_ref();

            provider.chat(
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
                Some(backend_model_clone.as_str()),
                tc_ref,
                None,
                None,
                None,
            )
        })
        .await
        .unwrap();

        let response = match result {
            Ok(resp) => resp,
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

        let out = internal_response_to_anthropic(&response, &model_for_response, &request_id);
        Ok(Json(out).into_response())
    }
}
