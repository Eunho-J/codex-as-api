use crate::auth::{self, AuthError, ChatGPTTokenData};
use crate::messages::{AssistantResponse, Message, MessageRole, ToolCall, ToolSchema, Usage};
use crate::model_capabilities::{
    apply_model_capability_fields, build_codex_client_metadata, capability_for_model,
    resolve_codex_metadata_enabled, should_enable_parallel_tool_calls, strip_image_detail_fields,
    use_responses_lite, LITE_HEADER_NAME, LITE_HEADER_VALUE, SESSION_ID_KEY,
};
use crate::protocol::{reasoning_from_response_items, response_failure_message};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::io::{BufRead, Read};
use std::process::{Command, Stdio};
use std::sync::{Mutex, OnceLock};

pub const CHATGPT_OAUTH_DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
pub const CHATGPT_OAUTH_DEFAULT_MODEL: &str = "gpt-5.5";
// This matches reqwest::blocking's existing default operation timeout. Response reads apply the
// duration per read, so long-lived active SSE streams do not acquire a new total-time cutoff.
pub const CHATGPT_OAUTH_DEFAULT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
pub const REMOTE_COMPACTION_MARKER: &str = "[Remote Responses compacted history]";
const CODEX_CLI_ORIGINATOR: &str = "codex_cli_rs";
const CODEX_CLI_VERSION_ENV: &str = "CODEX_AS_API_CODEX_CLI_VERSION";
const CODEX_CLI_NPM_PACKAGE: &str = "@openai/codex";
static CODEX_CLI_VERSION_CACHE: OnceLock<Option<String>> = OnceLock::new();
const RESPONSE_CHAIN_CAPACITY: usize = 256;

fn resolve_prompt_cache_key(
    explicit: Option<&str>,
    client_metadata: Option<&HashMap<String, String>>,
) -> Result<Option<String>, ProviderError> {
    if let Some(key) = explicit {
        if key.trim().is_empty() {
            return Err(ProviderError::InvalidRequest(
                "prompt_cache_key must be a non-empty string when provided".to_string(),
            ));
        }
        return Ok(Some(key.to_string()));
    }

    Ok(client_metadata
        .and_then(|metadata| metadata.get(SESSION_ID_KEY))
        .filter(|session_id| !session_id.trim().is_empty())
        .cloned())
}

pub fn prime_codex_cli_version_cache() {
    let _ = resolve_codex_cli_version();
}

fn resolve_codex_cli_version() -> Option<String> {
    if let Ok(value) = std::env::var(CODEX_CLI_VERSION_ENV) {
        if let Some(version) = normalize_codex_cli_version(&value) {
            return Some(version);
        }
    }
    CODEX_CLI_VERSION_CACHE
        .get_or_init(fetch_latest_codex_cli_version_from_npm)
        .clone()
}

fn fetch_latest_codex_cli_version_from_npm() -> Option<String> {
    let mut child = Command::new("npm")
        .args(["view", CODEX_CLI_NPM_PACKAGE, "version"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let started = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut stdout = String::new();
                if let Some(mut pipe) = child.stdout.take() {
                    let _ = pipe.read_to_string(&mut stdout);
                }
                return if status.success() {
                    normalize_codex_cli_version(&stdout)
                } else {
                    None
                };
            }
            Ok(None) => {
                if started.elapsed() >= std::time::Duration::from_secs(3) {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
}

fn normalize_codex_cli_version(value: &str) -> Option<String> {
    let version = value.trim();
    if version.is_empty() {
        return None;
    }
    let mut chars = version.chars().peekable();
    let mut numeric_parts = 0usize;
    loop {
        let mut digits = 0usize;
        while matches!(chars.peek(), Some(ch) if ch.is_ascii_digit()) {
            chars.next();
            digits += 1;
        }
        if digits == 0 {
            return None;
        }
        numeric_parts += 1;
        if !matches!(chars.peek(), Some('.')) {
            break;
        }
        chars.next();
    }
    if !(2..=4).contains(&numeric_parts) {
        return None;
    }
    let rest: String = chars.collect();
    if !rest.is_empty() {
        let mut rest_chars = rest.chars();
        if !matches!(rest_chars.next(), Some('-' | '+')) {
            return None;
        }
        let mut suffix = 0usize;
        for ch in rest_chars {
            if !(ch.is_ascii_alphanumeric() || ch == '.' || ch == '-') {
                return None;
            }
            suffix += 1;
        }
        if suffix == 0 {
            return None;
        }
    }
    Some(version.to_string())
}

fn codex_cli_headers() -> HashMap<String, String> {
    codex_cli_headers_for_version(resolve_codex_cli_version().as_deref())
}

fn codex_cli_headers_for_version(version: Option<&str>) -> HashMap<String, String> {
    let mut headers = HashMap::new();
    headers.insert("originator".to_string(), CODEX_CLI_ORIGINATOR.to_string());
    if let Some(version) = version.and_then(normalize_codex_cli_version) {
        headers.insert(
            "User-Agent".to_string(),
            sanitize_header_value(&format!(
                "{CODEX_CLI_ORIGINATOR}/{version} ({}) codex-as-api",
                codex_os_info()
            )),
        );
    }
    headers
}

fn codex_os_info() -> String {
    format!("{} unknown; {}", codex_os_name(), std::env::consts::ARCH)
}

fn codex_os_name() -> &'static str {
    match std::env::consts::OS {
        "macos" => "Mac OS",
        "windows" => "Windows",
        "linux" => "Linux",
        other => other,
    }
}

fn sanitize_header_value(value: &str) -> String {
    value
        .chars()
        .map(|ch| if matches!(ch, ' '..='~') { ch } else { '_' })
        .collect()
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("{0}")]
    Auth(#[from] AuthError),
    #[error("{0}")]
    InvalidRequest(String),
    #[error("{0}")]
    Request(String),
    #[error("{message}")]
    UpstreamHttp { status: u16, message: String },
}

#[derive(Clone, Copy)]
enum ResponsesEndpointKind {
    Standard,
    Compact,
}

#[derive(Debug)]
struct FinalizedResponsesRequest {
    payload: Value,
    use_responses_lite: bool,
    conversation_input: Vec<Value>,
}

pub struct PreparedChatStream {
    request: FinalizedResponsesRequest,
    extra_headers: HashMap<String, String>,
}

#[derive(Default)]
struct ChatStreamState {
    final_output: Vec<Value>,
    reasoning_parts: Vec<String>,
    emitted_tool_call: bool,
    saw_text_delta: bool,
    saw_reasoning_delta: bool,
    saw_completed: bool,
    completed_response_id: Option<String>,
    pending_finish: Option<Value>,
}

#[derive(Default)]
struct ResponseChainStoreInner {
    chains: HashMap<String, Vec<Value>>,
    lru: VecDeque<String>,
}

struct ResponseChainStore {
    capacity: usize,
    inner: Mutex<ResponseChainStoreInner>,
}

impl ResponseChainStore {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            inner: Mutex::new(ResponseChainStoreInner::default()),
        }
    }

    fn resolve(&self, response_id: &str) -> Result<Vec<Value>, ProviderError> {
        if response_id.trim().is_empty() {
            return Err(ProviderError::InvalidRequest(
                "previous_response_id must be a non-empty string".to_string(),
            ));
        }

        let mut inner = self.inner.lock().map_err(|_| {
            ProviderError::Request("response chain store lock is poisoned".to_string())
        })?;
        let history = inner.chains.get(response_id).cloned().ok_or_else(|| {
            ProviderError::InvalidRequest(format!(
                "previous_response_id {response_id:?} is unknown or has been evicted"
            ))
        })?;
        inner.lru.retain(|id| id != response_id);
        inner.lru.push_back(response_id.to_string());
        Ok(history)
    }

    fn commit(
        &self,
        response_id: &str,
        conversation_input: &[Value],
        response_output: &[Value],
    ) -> Result<(), ProviderError> {
        if response_id.is_empty() || self.capacity == 0 {
            return Ok(());
        }

        let mut history = Vec::with_capacity(conversation_input.len() + response_output.len());
        history.extend_from_slice(conversation_input);
        history.extend_from_slice(response_output);

        let mut inner = self.inner.lock().map_err(|_| {
            ProviderError::Request("response chain store lock is poisoned".to_string())
        })?;
        inner.lru.retain(|id| id != response_id);
        inner.chains.insert(response_id.to_string(), history);
        inner.lru.push_back(response_id.to_string());
        while inner.chains.len() > self.capacity {
            let Some(evicted_id) = inner.lru.pop_front() else {
                break;
            };
            inner.chains.remove(&evicted_id);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default)]
pub struct GenerationControls {
    pub reasoning: Option<Value>,
    pub safety_identifier: Option<String>,
    pub prompt_cache_options: Option<Value>,
    pub verbosity: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct CompactControls {
    pub previous_response_id: Option<String>,
    pub prompt_cache_key: Option<String>,
    pub prompt_cache_options: Option<Value>,
    pub service_tier: Option<String>,
    pub text: Option<Value>,
    pub verbosity: Option<String>,
}

pub struct ChatGPTOAuthProvider {
    pub model: String,
    pub base_url: String,
    pub auth_json_path: Option<String>,
    pub timeout: Option<std::time::Duration>,
    response_chains: ResponseChainStore,
}

impl ChatGPTOAuthProvider {
    pub fn new(
        model: String,
        base_url: String,
        auth_json_path: Option<String>,
        timeout: Option<std::time::Duration>,
    ) -> Self {
        Self {
            model,
            base_url: base_url.trim_end_matches('/').to_string(),
            auth_json_path,
            timeout: Some(timeout.unwrap_or(CHATGPT_OAUTH_DEFAULT_TIMEOUT)),
            response_chains: ResponseChainStore::new(RESPONSE_CHAIN_CAPACITY),
        }
    }

    pub fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        temperature: Option<f64>,
        reasoning_effort: Option<&str>,
        max_tokens: Option<i64>,
        stop: Option<&[String]>,
        prompt_cache_key: Option<&str>,
        subagent: Option<&str>,
        memgen_request: Option<bool>,
        previous_response_id: Option<&str>,
        model: Option<&str>,
        tool_choice: Option<&Value>,
        service_tier: Option<&str>,
        text: Option<&Value>,
        client_metadata: Option<&HashMap<String, String>>,
        codex_metadata: Option<bool>,
        responses_lite: Option<&Value>,
        parallel_tool_calls: Option<bool>,
    ) -> Result<AssistantResponse, ProviderError> {
        self.chat_with_controls(
            messages,
            tools,
            temperature,
            reasoning_effort,
            max_tokens,
            stop,
            prompt_cache_key,
            subagent,
            memgen_request,
            previous_response_id,
            model,
            tool_choice,
            service_tier,
            text,
            client_metadata,
            codex_metadata,
            responses_lite,
            parallel_tool_calls,
            &GenerationControls::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chat_with_controls(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        temperature: Option<f64>,
        reasoning_effort: Option<&str>,
        max_tokens: Option<i64>,
        stop: Option<&[String]>,
        prompt_cache_key: Option<&str>,
        subagent: Option<&str>,
        memgen_request: Option<bool>,
        previous_response_id: Option<&str>,
        model: Option<&str>,
        tool_choice: Option<&Value>,
        service_tier: Option<&str>,
        text: Option<&Value>,
        client_metadata: Option<&HashMap<String, String>>,
        codex_metadata: Option<bool>,
        responses_lite: Option<&Value>,
        parallel_tool_calls: Option<bool>,
        controls: &GenerationControls,
    ) -> Result<AssistantResponse, ProviderError> {
        let mut content_parts: Vec<String> = Vec::new();
        let mut reasoning_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut finish_reason = "stop".to_string();
        let mut raw_events: Vec<Value> = Vec::new();
        let mut usage: Option<Usage> = None;
        let mut response_id: Option<String> = None;

        let events = self.chat_stream_with_controls(
            messages,
            tools,
            temperature,
            reasoning_effort,
            max_tokens,
            stop,
            prompt_cache_key,
            subagent,
            memgen_request,
            previous_response_id,
            model,
            tool_choice,
            service_tier,
            text,
            client_metadata,
            codex_metadata,
            responses_lite,
            parallel_tool_calls,
            controls,
        )?;

        for event in events {
            raw_events.push(event.clone());
            let typ = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match typ {
                "content" => {
                    if let Some(text) = event.get("text").and_then(|v| v.as_str()) {
                        content_parts.push(text.to_string());
                    }
                }
                "reasoning_delta" | "reasoning_raw_delta" => {
                    if let Some(text) = event.get("text").and_then(|v| v.as_str()) {
                        reasoning_parts.push(text.to_string());
                    }
                }
                "tool_call" => {
                    let id = event
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = event
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments = event
                        .get("arguments")
                        .and_then(|v| v.as_object())
                        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                        .unwrap_or_default();
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
                "finish" => {
                    if let Some(fr) = event.get("finish_reason").and_then(|v| v.as_str()) {
                        finish_reason = fr.to_string();
                    }
                    if let Some(rc) = event.get("reasoning_content").and_then(|v| v.as_str()) {
                        reasoning_parts = vec![rc.to_string()];
                    }
                    if let Some(u) = usage_from_response(event.get("usage").unwrap_or(&Value::Null))
                    {
                        usage = Some(u);
                    }
                    response_id = event
                        .get("response_id")
                        .and_then(Value::as_str)
                        .map(str::to_string);
                }
                _ => {}
            }
        }

        let reasoning_content = {
            let joined = reasoning_parts.join("");
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        };

        let tail_events = compact_raw_events(&raw_events);

        Ok(AssistantResponse {
            content: content_parts.join(""),
            tool_calls,
            finish_reason,
            usage,
            reasoning_content,
            raw: Some(json!({"events": tail_events})),
            response_id,
        })
    }

    pub fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        temperature: Option<f64>,
        reasoning_effort: Option<&str>,
        max_tokens: Option<i64>,
        stop: Option<&[String]>,
        prompt_cache_key: Option<&str>,
        subagent: Option<&str>,
        memgen_request: Option<bool>,
        previous_response_id: Option<&str>,
        model: Option<&str>,
        tool_choice: Option<&Value>,
        service_tier: Option<&str>,
        text: Option<&Value>,
        client_metadata: Option<&HashMap<String, String>>,
        codex_metadata: Option<bool>,
        responses_lite: Option<&Value>,
        parallel_tool_calls: Option<bool>,
    ) -> Result<Vec<Value>, ProviderError> {
        self.chat_stream_with_controls(
            messages,
            tools,
            temperature,
            reasoning_effort,
            max_tokens,
            stop,
            prompt_cache_key,
            subagent,
            memgen_request,
            previous_response_id,
            model,
            tool_choice,
            service_tier,
            text,
            client_metadata,
            codex_metadata,
            responses_lite,
            parallel_tool_calls,
            &GenerationControls::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn chat_stream_with_controls(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        temperature: Option<f64>,
        reasoning_effort: Option<&str>,
        max_tokens: Option<i64>,
        stop: Option<&[String]>,
        prompt_cache_key: Option<&str>,
        subagent: Option<&str>,
        memgen_request: Option<bool>,
        previous_response_id: Option<&str>,
        model: Option<&str>,
        tool_choice: Option<&Value>,
        service_tier: Option<&str>,
        text: Option<&Value>,
        client_metadata: Option<&HashMap<String, String>>,
        codex_metadata: Option<bool>,
        responses_lite: Option<&Value>,
        parallel_tool_calls: Option<bool>,
        controls: &GenerationControls,
    ) -> Result<Vec<Value>, ProviderError> {
        let prepared = self.prepare_chat_stream_with_controls(
            messages,
            tools,
            temperature,
            reasoning_effort,
            max_tokens,
            stop,
            prompt_cache_key,
            subagent,
            memgen_request,
            previous_response_id,
            model,
            tool_choice,
            service_tier,
            text,
            client_metadata,
            codex_metadata,
            responses_lite,
            parallel_tool_calls,
            controls,
        )?;
        let mut result_events = Vec::new();
        self.stream_prepared_chat(prepared, |event| {
            result_events.push(event);
            Ok(())
        })?;
        Ok(result_events)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_chat_stream_with_controls(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        temperature: Option<f64>,
        reasoning_effort: Option<&str>,
        max_tokens: Option<i64>,
        stop: Option<&[String]>,
        prompt_cache_key: Option<&str>,
        subagent: Option<&str>,
        memgen_request: Option<bool>,
        previous_response_id: Option<&str>,
        model: Option<&str>,
        tool_choice: Option<&Value>,
        service_tier: Option<&str>,
        text: Option<&Value>,
        client_metadata: Option<&HashMap<String, String>>,
        codex_metadata: Option<bool>,
        responses_lite: Option<&Value>,
        parallel_tool_calls: Option<bool>,
        controls: &GenerationControls,
    ) -> Result<PreparedChatStream, ProviderError> {
        let _ = temperature;
        let request = self.responses_payload_with_controls(
            messages,
            tools,
            reasoning_effort,
            stop,
            prompt_cache_key,
            max_tokens,
            previous_response_id,
            model,
            tool_choice,
            service_tier,
            text,
            client_metadata,
            codex_metadata,
            responses_lite,
            parallel_tool_calls,
            controls,
        )?;

        let mut extra_headers: HashMap<String, String> = HashMap::new();
        if request.use_responses_lite {
            extra_headers.insert(LITE_HEADER_NAME.to_string(), LITE_HEADER_VALUE.to_string());
        }
        if let Some(sa) = subagent {
            extra_headers.insert("x-openai-subagent".to_string(), sa.to_string());
        }
        if let Some(mg) = memgen_request {
            extra_headers.insert(
                "x-openai-memgen-request".to_string(),
                if mg { "true" } else { "false" }.to_string(),
            );
        }
        let _ = self.headers()?;

        Ok(PreparedChatStream {
            request,
            extra_headers,
        })
    }

    pub fn stream_prepared_chat<F>(
        &self,
        prepared: PreparedChatStream,
        mut emit: F,
    ) -> Result<(), ProviderError>
    where
        F: FnMut(Value) -> Result<(), ProviderError>,
    {
        let conversation_input = prepared.request.conversation_input.clone();
        let mut state = ChatStreamState::default();

        self.request_sse_each(
            "/responses",
            &prepared.request.payload,
            Some(&prepared.extra_headers),
            |event| {
                let typ = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match typ {
                    "response.output_text.delta" => {
                        if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                            if !delta.is_empty() {
                                state.saw_text_delta = true;
                                emit(json!({"type": "content", "text": delta}))?;
                            }
                        }
                    }
                    "response.output_item.done" => {
                        let item = event.get("item").ok_or_else(|| {
                            ProviderError::Request(
                                "response.output_item.done must contain an object item".to_string(),
                            )
                        })?;
                        state.final_output.push(item.clone());
                        if let Some(tc) = tool_call_from_response_item(item) {
                            state.emitted_tool_call = true;
                            emit(json!({
                                "type": "tool_call",
                                "id": tc.id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                            }))?;
                        }
                        if let Some(web_search) = web_search_event_from_response_item(item, &[]) {
                            emit(web_search)?;
                        }
                    }
                    "response.reasoning_summary_part.added" => {
                        emit(json!({
                            "type": "reasoning_section_break",
                            "summary_index": event.get("summary_index"),
                            "part_index": event.get("part_index"),
                        }))?;
                    }
                    "response.reasoning_summary_text.delta" => {
                        if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                            if !delta.is_empty() {
                                state.saw_reasoning_delta = true;
                                state.reasoning_parts.push(delta.to_string());
                                emit(json!({
                                    "type": "reasoning_delta",
                                    "text": delta,
                                    "summary_index": event.get("summary_index"),
                                }))?;
                            }
                        }
                    }
                    "response.reasoning_text.delta" => {
                        if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                            if !delta.is_empty() {
                                state.saw_reasoning_delta = true;
                                state.reasoning_parts.push(delta.to_string());
                                emit(json!({
                                    "type": "reasoning_raw_delta",
                                    "text": delta,
                                    "summary_index": event.get("summary_index"),
                                }))?;
                            }
                        }
                    }
                    "response.failed" => {
                        return Err(ProviderError::Request(response_failure_message(
                            &event, "failed",
                        )));
                    }
                    "response.incomplete" => {
                        return Err(ProviderError::Request(response_failure_message(
                            &event,
                            "incomplete",
                        )));
                    }
                    "response.completed" => {
                        state.saw_completed = true;
                        let mut usage_val = Value::Null;
                        let mut response_id = Value::Null;
                        if let Some(response) = event.get("response").and_then(|v| v.as_object()) {
                            response_id = response.get("id").cloned().unwrap_or(Value::Null);
                            state.completed_response_id = response
                                .get("id")
                                .and_then(Value::as_str)
                                .filter(|id| !id.is_empty())
                                .map(str::to_string);
                            usage_val = response.get("usage").cloned().unwrap_or(Value::Null);
                            if !state.saw_text_delta {
                                let final_text = text_from_response_items(&state.final_output);
                                if !final_text.is_empty() {
                                    state.saw_text_delta = true;
                                    emit(json!({"type": "content", "text": final_text}))?;
                                }
                            }
                            if !state.saw_reasoning_delta {
                                let completed_reasoning = reasoning_from_response_items(
                                    &state
                                        .final_output
                                        .iter()
                                        .filter(|item| item.is_object())
                                        .cloned()
                                        .collect::<Vec<_>>(),
                                );
                                if !completed_reasoning.is_empty() {
                                    state.saw_reasoning_delta = true;
                                    state.reasoning_parts.push(completed_reasoning.clone());
                                    emit(json!({
                                        "type": "reasoning_delta",
                                        "text": completed_reasoning,
                                    }))?;
                                }
                            }
                        }
                        let reasoning_joined = state.reasoning_parts.join("");
                        state.pending_finish = Some(json!({
                            "type": "finish",
                            "finish_reason": if state.emitted_tool_call { "tool_calls" } else { "stop" },
                            "usage": usage_val,
                            "reasoning_content": if reasoning_joined.is_empty() { Value::Null } else { Value::String(reasoning_joined) },
                            "response_id": response_id,
                        }));
                    }
                    _ => {}
                }
                Ok(())
            },
        )?;

        if !state.saw_completed {
            return Err(ProviderError::Request(
                "ChatGPT OAuth response stream ended before response.completed".to_string(),
            ));
        }

        if let Some(response_id) = state.completed_response_id {
            self.response_chains
                .commit(&response_id, &conversation_input, &state.final_output)?;
        }
        emit(state.pending_finish.ok_or_else(|| {
            ProviderError::Request("response.completed did not produce a finish event".to_string())
        })?)?;

        Ok(())
    }

    pub fn generate_image(
        &self,
        prompt: &str,
        reference_images: &[HashMap<String, String>],
        size: Option<&str>,
        reasoning_effort: Option<&str>,
        model: Option<&str>,
        responses_lite: Option<&Value>,
    ) -> Result<Vec<Value>, ProviderError> {
        self.generate_image_with_controls(
            prompt,
            reference_images,
            size,
            reasoning_effort,
            model,
            responses_lite,
            &GenerationControls::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_image_with_controls(
        &self,
        prompt: &str,
        reference_images: &[HashMap<String, String>],
        size: Option<&str>,
        reasoning_effort: Option<&str>,
        model: Option<&str>,
        responses_lite: Option<&Value>,
        controls: &GenerationControls,
    ) -> Result<Vec<Value>, ProviderError> {
        if prompt.trim().is_empty() {
            return Err(ProviderError::Request(
                "image generation prompt is required".to_string(),
            ));
        }

        let mut content: Vec<Value> = vec![json!({"type": "input_text", "text": prompt})];
        let validated = validate_image_content_items(reference_images)?;
        content.extend(validated);

        if let Some(s) = size {
            if s != "auto" {
                let new_text = format!("{}\n\nRequested output size/aspect: {}", prompt, s);
                content[0] = json!({"type": "input_text", "text": new_text});
            }
        }

        let mut payload = json!({
            "model": model.unwrap_or(&self.model),
            "instructions": "Use the image_generation tool to create the requested image. Return the generated image through an image_generation_call result.",
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [{"type": "image_generation", "output_format": "png"}],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "stream": true,
            "store": false,
            "include": [],
            "prompt_cache_key": uuid::Uuid::new_v4().to_string(),
        });

        set_reasoning_payload_with_options(
            payload.as_object_mut().unwrap(),
            reasoning_effort,
            controls.reasoning.as_ref(),
        )?;
        let request_model = model.unwrap_or(&self.model);
        apply_generation_controls(payload.as_object_mut().unwrap(), request_model, controls)?;
        let merged_text = merge_text_verbosity(None, controls.verbosity.as_deref())?;
        let request = finalize_responses_request(
            payload,
            request_model,
            responses_lite,
            merged_text.as_ref(),
            None,
            ResponsesEndpointKind::Standard,
        )?;
        let output_items = self.collect_response_output_items(request)?;

        let mut generated: Vec<Value> = Vec::new();
        for item in &output_items {
            if let Some(img) = image_generation_from_item(item)? {
                generated.push(img);
            }
        }

        if generated.is_empty() {
            return Err(ProviderError::Request(
                "image generation response returned no image_generation_call".to_string(),
            ));
        }

        Ok(generated)
    }

    pub fn inspect_images(
        &self,
        prompt: &str,
        images: &[HashMap<String, String>],
        reasoning_effort: Option<&str>,
        model: Option<&str>,
        responses_lite: Option<&Value>,
    ) -> Result<String, ProviderError> {
        self.inspect_images_with_controls(
            prompt,
            images,
            reasoning_effort,
            model,
            responses_lite,
            &GenerationControls::default(),
        )
    }

    pub fn inspect_images_with_controls(
        &self,
        prompt: &str,
        images: &[HashMap<String, String>],
        reasoning_effort: Option<&str>,
        model: Option<&str>,
        responses_lite: Option<&Value>,
        controls: &GenerationControls,
    ) -> Result<String, ProviderError> {
        if prompt.trim().is_empty() {
            return Err(ProviderError::Request(
                "image inspection prompt is required".to_string(),
            ));
        }

        let mut content: Vec<Value> = vec![json!({"type": "input_text", "text": prompt})];
        let validated = validate_image_content_items(images)?;
        content.extend(validated);

        let mut payload = json!({
            "model": model.unwrap_or(&self.model),
            "instructions": "Inspect the attached image(s) and answer the user's review prompt directly.",
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "stream": true,
            "store": false,
            "include": [],
            "prompt_cache_key": uuid::Uuid::new_v4().to_string(),
        });

        set_reasoning_payload_with_options(
            payload.as_object_mut().unwrap(),
            reasoning_effort,
            controls.reasoning.as_ref(),
        )?;
        let request_model = model.unwrap_or(&self.model);
        apply_generation_controls(payload.as_object_mut().unwrap(), request_model, controls)?;
        let merged_text = merge_text_verbosity(None, controls.verbosity.as_deref())?;
        let request = finalize_responses_request(
            payload,
            request_model,
            responses_lite,
            merged_text.as_ref(),
            None,
            ResponsesEndpointKind::Standard,
        )?;
        let output_items = self.collect_response_output_items(request)?;
        let text = text_from_response_items(&output_items).trim().to_string();

        if text.is_empty() {
            return Err(ProviderError::Request(
                "image inspection response returned empty content".to_string(),
            ));
        }

        Ok(text)
    }

    pub fn inspect_image_values_with_controls(
        &self,
        prompt: &str,
        images: &[Value],
        reasoning_effort: Option<&str>,
        model: Option<&str>,
        responses_lite: Option<&Value>,
        controls: &GenerationControls,
    ) -> Result<String, ProviderError> {
        if prompt.trim().is_empty() {
            return Err(ProviderError::Request(
                "image inspection prompt is required".to_string(),
            ));
        }
        let mut content: Vec<Value> = vec![json!({"type": "input_text", "text": prompt})];
        content.extend(validate_image_content_values(images)?);
        let mut payload = json!({
            "model": model.unwrap_or(&self.model),
            "instructions": "Inspect the attached image(s) and answer the user's review prompt directly.",
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "stream": true,
            "store": false,
            "include": [],
            "prompt_cache_key": uuid::Uuid::new_v4().to_string(),
        });
        set_reasoning_payload_with_options(
            payload.as_object_mut().unwrap(),
            reasoning_effort,
            controls.reasoning.as_ref(),
        )?;
        let request_model = model.unwrap_or(&self.model);
        apply_generation_controls(payload.as_object_mut().unwrap(), request_model, controls)?;
        let merged_text = merge_text_verbosity(None, controls.verbosity.as_deref())?;
        let request = finalize_responses_request(
            payload,
            request_model,
            responses_lite,
            merged_text.as_ref(),
            None,
            ResponsesEndpointKind::Standard,
        )?;
        let output_items = self.collect_response_output_items(request)?;
        let text = text_from_response_items(&output_items).trim().to_string();
        if text.is_empty() {
            return Err(ProviderError::Request(
                "image inspection response returned empty content".to_string(),
            ));
        }
        Ok(text)
    }

    pub fn compact_messages(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        reasoning_effort: Option<&str>,
        model: Option<&str>,
        responses_lite: Option<&Value>,
    ) -> Result<String, ProviderError> {
        self.compact_messages_with_prompt_cache_options(
            messages,
            tools,
            reasoning_effort,
            model,
            responses_lite,
            None,
        )
    }

    pub fn compact_messages_with_prompt_cache_options(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        reasoning_effort: Option<&str>,
        model: Option<&str>,
        responses_lite: Option<&Value>,
        prompt_cache_options: Option<&Value>,
    ) -> Result<String, ProviderError> {
        let controls = CompactControls {
            prompt_cache_options: prompt_cache_options.cloned(),
            ..CompactControls::default()
        };
        self.compact_messages_with_controls(
            messages,
            tools,
            reasoning_effort,
            model,
            responses_lite,
            &controls,
        )
    }

    pub fn compact_messages_with_controls(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        reasoning_effort: Option<&str>,
        model: Option<&str>,
        responses_lite: Option<&Value>,
        controls: &CompactControls,
    ) -> Result<String, ProviderError> {
        let (base_instructions, mut input_items) = split_instructions_and_input(messages)?;
        if let Some(tools) = tools {
            for tool in tools {
                if let Some(error) = tool
                    .parameters
                    .get("__codex_as_api_error")
                    .and_then(Value::as_str)
                {
                    return Err(ProviderError::InvalidRequest(error.to_string()));
                }
            }
        }
        let tools_payload: Vec<Value> = tools
            .map(|tools| tools.iter().map(tool_schema_to_response_dict).collect())
            .unwrap_or_default();
        if let Some(previous_response_id) = &controls.previous_response_id {
            let mut history = self.response_chains.resolve(previous_response_id)?;
            history.append(&mut input_items);
            input_items = history;
        }
        let mut payload = json!({
            "model": model.unwrap_or(&self.model),
            "input": input_items,
            "tools": tools_payload,
            "parallel_tool_calls": false,
        });
        if !base_instructions.is_empty() {
            payload
                .as_object_mut()
                .unwrap()
                .insert("instructions".to_string(), Value::String(base_instructions));
        }

        set_reasoning_payload(payload.as_object_mut().unwrap(), reasoning_effort)?;
        let request_model = model.unwrap_or(&self.model);
        if let Some(key) = &controls.prompt_cache_key {
            if key.is_empty() {
                return Err(ProviderError::InvalidRequest(
                    "prompt_cache_key must be a non-empty string".to_string(),
                ));
            }
            payload
                .as_object_mut()
                .unwrap()
                .insert("prompt_cache_key".to_string(), Value::String(key.clone()));
        }
        let generation_controls = GenerationControls {
            prompt_cache_options: controls.prompt_cache_options.clone(),
            ..GenerationControls::default()
        };
        apply_generation_controls(
            payload.as_object_mut().unwrap(),
            request_model,
            &generation_controls,
        )?;
        let merged_text =
            merge_text_verbosity(controls.text.as_ref(), controls.verbosity.as_deref())?;
        let request = finalize_responses_request(
            payload,
            request_model,
            responses_lite,
            merged_text.as_ref(),
            controls.service_tier.as_deref(),
            ResponsesEndpointKind::Compact,
        )?;
        let extra_headers = responses_lite_headers(request.use_responses_lite);
        let data = self.post_json("/responses/compact", &request.payload, Some(&extra_headers))?;

        let output = data
            .get("output")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                ProviderError::Request("remote compact response missing output array".to_string())
            })?;

        let compacted_history = filter_compacted_history_items(output)?;
        let serialized = serde_json::to_string(&compacted_history).unwrap();
        Ok(format!("{}\n{}", REMOTE_COMPACTION_MARKER, serialized))
    }

    fn collect_response_output_items(
        &self,
        request: FinalizedResponsesRequest,
    ) -> Result<Vec<Value>, ProviderError> {
        let mut output_items: Vec<Value> = Vec::new();
        let mut saw_completed = false;
        let mut seen_keys: HashSet<String> = HashSet::new();

        let append_item = |item: &Value, items: &mut Vec<Value>, seen: &mut HashSet<String>| {
            let mut key_parts = vec![item
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()];

            let mut found_id = false;
            for field in &["id", "call_id"] {
                if let Some(s) = item.get(field).and_then(|v| v.as_str()) {
                    if !s.is_empty() {
                        key_parts.push(s.to_string());
                        found_id = true;
                        break;
                    }
                }
            }
            if !found_id {
                key_parts.push(serde_json::to_string(item).unwrap_or_default());
            }

            let key = key_parts.join("\x1f");
            if seen.contains(&key) {
                return;
            }
            seen.insert(key);
            items.push(item.clone());
        };

        let extra_headers = responses_lite_headers(request.use_responses_lite);
        let events = self.post_sse("/responses", &request.payload, Some(&extra_headers))?;
        for event in &events {
            let typ = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match typ {
                "response.output_item.done" => {
                    if let Some(item) = event.get("item") {
                        if item.is_object() {
                            append_item(item, &mut output_items, &mut seen_keys);
                        }
                    }
                }
                "response.failed" => {
                    return Err(ProviderError::Request(response_failure_message(
                        event, "failed",
                    )));
                }
                "response.incomplete" => {
                    return Err(ProviderError::Request(response_failure_message(
                        event,
                        "incomplete",
                    )));
                }
                "response.completed" => {
                    saw_completed = true;
                }
                _ => {}
            }
        }

        if !saw_completed {
            return Err(ProviderError::Request(
                "ChatGPT OAuth response stream ended before response.completed".to_string(),
            ));
        }

        Ok(output_items)
    }

    fn responses_payload(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        reasoning_effort: Option<&str>,
        stop: Option<&[String]>,
        prompt_cache_key: Option<&str>,
        max_tokens: Option<i64>,
        previous_response_id: Option<&str>,
        model: Option<&str>,
        tool_choice: Option<&Value>,
        service_tier: Option<&str>,
        text: Option<&Value>,
        client_metadata: Option<&HashMap<String, String>>,
        codex_metadata: Option<bool>,
        responses_lite: Option<&Value>,
        parallel_tool_calls: Option<bool>,
    ) -> Result<FinalizedResponsesRequest, ProviderError> {
        self.responses_payload_with_controls(
            messages,
            tools,
            reasoning_effort,
            stop,
            prompt_cache_key,
            max_tokens,
            previous_response_id,
            model,
            tool_choice,
            service_tier,
            text,
            client_metadata,
            codex_metadata,
            responses_lite,
            parallel_tool_calls,
            &GenerationControls::default(),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn responses_payload_with_controls(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        reasoning_effort: Option<&str>,
        stop: Option<&[String]>,
        prompt_cache_key: Option<&str>,
        max_tokens: Option<i64>,
        previous_response_id: Option<&str>,
        model: Option<&str>,
        tool_choice: Option<&Value>,
        service_tier: Option<&str>,
        text: Option<&Value>,
        client_metadata: Option<&HashMap<String, String>>,
        codex_metadata: Option<bool>,
        responses_lite: Option<&Value>,
        parallel_tool_calls: Option<bool>,
        controls: &GenerationControls,
    ) -> Result<FinalizedResponsesRequest, ProviderError> {
        if let Some(stop_values) = stop {
            if stop_values.iter().any(|value| !value.is_empty()) {
                return Err(ProviderError::InvalidRequest(
                    "stop is not supported by the private Codex OAuth HTTP transport".to_string(),
                ));
            }
        }
        let (instructions, mut input_items) = split_instructions_and_input(messages)?;

        if instructions.is_empty() {
            return Err(ProviderError::Request(
                "ChatGPT OAuth Responses request requires system instructions".to_string(),
            ));
        }

        if let Some(ts) = tools {
            for tool in ts {
                if let Some(error) = tool
                    .parameters
                    .get("__codex_as_api_error")
                    .and_then(|v| v.as_str())
                {
                    return Err(ProviderError::InvalidRequest(error.to_string()));
                }
            }
        }

        let tools_array: Vec<Value> = match tools {
            Some(ts) => ts.iter().map(tool_schema_to_response_dict).collect(),
            None => vec![],
        };
        let request_model = model.unwrap_or(&self.model);

        if let Some(previous_response_id) = previous_response_id {
            let mut history = self.response_chains.resolve(previous_response_id)?;
            history.append(&mut input_items);
            input_items = history;
        }

        let mut payload = json!({
            "model": request_model,
            "instructions": instructions,
            "input": input_items,
            "tools": tools_array,
            "tool_choice": tool_choice.cloned().unwrap_or(json!("auto")),
            "parallel_tool_calls": should_enable_parallel_tool_calls(request_model, parallel_tool_calls, false),
            "stream": true,
            "store": false,
            "include": [],
        });
        if tools_array
            .iter()
            .any(|tool| tool.get("type").and_then(|v| v.as_str()) == Some("web_search"))
        {
            payload.as_object_mut().unwrap().insert(
                "include".to_string(),
                json!(["web_search_call.action.sources"]),
            );
        }

        if let Some(key) = resolve_prompt_cache_key(prompt_cache_key, client_metadata)? {
            payload
                .as_object_mut()
                .unwrap()
                .insert("prompt_cache_key".to_string(), Value::String(key));
        }
        let _ = max_tokens; // ChatGPT Codex backend rejects max_output_tokens for this endpoint.
        if let Some(cm) = client_metadata {
            let map: serde_json::Map<String, Value> = cm
                .iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect();
            payload
                .as_object_mut()
                .unwrap()
                .insert("client_metadata".to_string(), Value::Object(map));
        }
        if resolve_codex_metadata_enabled(codex_metadata).map_err(ProviderError::Request)? {
            let merged =
                build_codex_client_metadata(self.auth_json_path.as_deref(), client_metadata)
                    .map_err(ProviderError::InvalidRequest)?;
            let map: serde_json::Map<String, Value> = merged
                .into_iter()
                .map(|(k, v)| (k, Value::String(v)))
                .collect();
            payload
                .as_object_mut()
                .unwrap()
                .insert("client_metadata".to_string(), Value::Object(map));
        }

        set_reasoning_payload_with_options(
            payload.as_object_mut().unwrap(),
            reasoning_effort,
            controls.reasoning.as_ref(),
        )?;
        apply_generation_controls(payload.as_object_mut().unwrap(), request_model, controls)?;
        let merged_text = merge_text_verbosity(text, controls.verbosity.as_deref())?;
        finalize_responses_request(
            payload,
            request_model,
            responses_lite,
            merged_text.as_ref(),
            service_tier,
            ResponsesEndpointKind::Standard,
        )
    }

    fn headers(&self) -> Result<(HashMap<String, String>, ChatGPTTokenData), ProviderError> {
        let token = auth::load_token_data(self.auth_json_path.as_deref())?;
        let mut headers = HashMap::new();
        headers.extend(codex_cli_headers());
        headers.insert(
            "Authorization".to_string(),
            format!("Bearer {}", token.access_token),
        );
        headers.insert("ChatGPT-Account-Id".to_string(), token.account_id.clone());
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        if token.fedramp {
            headers.insert("X-OpenAI-Fedramp".to_string(), "true".to_string());
        }
        Ok((headers, token))
    }

    fn post_json(
        &self,
        path: &str,
        payload: &Value,
        extra_headers: Option<&HashMap<String, String>>,
    ) -> Result<Value, ProviderError> {
        let raw = self.request_json(path, payload, extra_headers)?;
        let data: Value = serde_json::from_slice(&raw).map_err(|_| {
            ProviderError::Request("ChatGPT OAuth response must be a JSON object".to_string())
        })?;
        if !data.is_object() {
            return Err(ProviderError::Request(
                "ChatGPT OAuth response must be a JSON object".to_string(),
            ));
        }
        Ok(data)
    }

    fn post_sse(
        &self,
        path: &str,
        payload: &Value,
        extra_headers: Option<&HashMap<String, String>>,
    ) -> Result<Vec<Value>, ProviderError> {
        self.request_sse(path, payload, extra_headers)
    }

    fn request_sse(
        &self,
        path: &str,
        payload: &Value,
        extra_headers: Option<&HashMap<String, String>>,
    ) -> Result<Vec<Value>, ProviderError> {
        let mut events = Vec::new();
        self.request_sse_each(path, payload, extra_headers, |event| {
            events.push(event);
            Ok(())
        })?;
        Ok(events)
    }

    fn request_sse_each<F>(
        &self,
        path: &str,
        payload: &Value,
        extra_headers: Option<&HashMap<String, String>>,
        mut on_event: F,
    ) -> Result<(), ProviderError>
    where
        F: FnMut(Value) -> Result<(), ProviderError>,
    {
        for attempt in 0..2 {
            let (mut headers, token) = self.headers()?;
            headers.insert("Accept".to_string(), "text/event-stream".to_string());
            if let Some(eh) = extra_headers {
                for (k, v) in eh {
                    headers.insert(k.clone(), v.clone());
                }
            }
            let token_values = [
                token.access_token.as_str(),
                token.refresh_token.as_str(),
                token.id_token.as_str(),
                token.account_id.as_str(),
            ];

            let url = format!("{}{}", self.base_url, path);
            let body = serde_json::to_vec(payload).unwrap();

            let mut builder = reqwest::blocking::Client::new().post(&url);
            for (k, v) in &headers {
                builder = builder.header(k.as_str(), v.as_str());
            }
            if let Some(t) = self.timeout {
                builder = builder.timeout(t);
            }
            builder = builder.body(body);

            match builder.send() {
                Ok(response) => {
                    let status = response.status();
                    if status == reqwest::StatusCode::UNAUTHORIZED && attempt == 0 {
                        auth::do_refresh_token(self.auth_json_path.as_deref())?;
                        continue;
                    }
                    if !status.is_success() {
                        let body_text = response.text().unwrap_or_default();
                        let redacted = auth::redact_text(&body_text, &token_values);
                        return Err(ProviderError::UpstreamHttp {
                            status: status.as_u16(),
                            message: format!(
                                "ChatGPT OAuth request failed: HTTP {}: {}",
                                status.as_u16(),
                                redacted
                            ),
                        });
                    }

                    let reader = std::io::BufReader::new(response);
                    let mut block: Vec<String> = Vec::new();

                    for line_result in reader.lines() {
                        let line = line_result.map_err(|error| {
                            ProviderError::Request(format!(
                                "ChatGPT OAuth SSE read failed: {}",
                                auth::redact_text(&error.to_string(), &token_values)
                            ))
                        })?;
                        if line.is_empty() {
                            if let Some(event) = decode_sse_block(&block)? {
                                validate_response_event(&event)?;
                                let terminal = is_terminal_response_event(&event);
                                on_event(event)?;
                                if terminal {
                                    return Ok(());
                                }
                            }
                            block.clear();
                            continue;
                        }
                        block.push(line);
                    }
                    if !block.is_empty() {
                        if let Some(event) = decode_sse_block(&block)? {
                            validate_response_event(&event)?;
                            let terminal = is_terminal_response_event(&event);
                            on_event(event)?;
                            if terminal {
                                return Ok(());
                            }
                        }
                    }

                    return Ok(());
                }
                Err(e) => {
                    let msg = auth::redact_text(&e.to_string(), &token_values);
                    return Err(ProviderError::Request(format!(
                        "ChatGPT OAuth request failed: {}",
                        msg
                    )));
                }
            }
        }

        unreachable!("ChatGPT OAuth request retry state")
    }

    fn request_json(
        &self,
        path: &str,
        payload: &Value,
        extra_headers: Option<&HashMap<String, String>>,
    ) -> Result<Vec<u8>, ProviderError> {
        for attempt in 0..2 {
            let (mut headers, token) = self.headers()?;
            if let Some(eh) = extra_headers {
                for (k, v) in eh {
                    headers.insert(k.clone(), v.clone());
                }
            }
            let token_values = [
                token.access_token.as_str(),
                token.refresh_token.as_str(),
                token.id_token.as_str(),
                token.account_id.as_str(),
            ];

            let url = format!("{}{}", self.base_url, path);
            let body = serde_json::to_vec(payload).unwrap();

            let mut builder = reqwest::blocking::Client::new().post(&url);
            for (k, v) in &headers {
                builder = builder.header(k.as_str(), v.as_str());
            }
            if let Some(t) = self.timeout {
                builder = builder.timeout(t);
            }
            builder = builder.body(body);

            match builder.send() {
                Ok(response) => {
                    let status = response.status();
                    if status == reqwest::StatusCode::UNAUTHORIZED && attempt == 0 {
                        auth::do_refresh_token(self.auth_json_path.as_deref())?;
                        continue;
                    }
                    if !status.is_success() {
                        let body_text = response.text().unwrap_or_default();
                        let redacted = auth::redact_text(&body_text, &token_values);
                        return Err(ProviderError::UpstreamHttp {
                            status: status.as_u16(),
                            message: format!(
                                "ChatGPT OAuth request failed: HTTP {}: {}",
                                status.as_u16(),
                                redacted
                            ),
                        });
                    }
                    let bytes = response.bytes().map_err(|e| {
                        ProviderError::Request(format!(
                            "ChatGPT OAuth request failed: {}",
                            auth::redact_text(&e.to_string(), &token_values)
                        ))
                    })?;
                    return Ok(bytes.to_vec());
                }
                Err(e) => {
                    let msg = auth::redact_text(&e.to_string(), &token_values);
                    return Err(ProviderError::Request(format!(
                        "ChatGPT OAuth request failed: {}",
                        msg
                    )));
                }
            }
        }

        unreachable!("ChatGPT OAuth request retry state")
    }
}

fn validate_image_content_items(
    images: &[HashMap<String, String>],
) -> Result<Vec<Value>, ProviderError> {
    let mut items: Vec<Value> = Vec::new();
    for (index, image) in images.iter().enumerate() {
        let image_url = image.get("image_url").map(|s| s.as_str()).unwrap_or("");
        if image_url.trim().is_empty() {
            return Err(ProviderError::Request(format!(
                "image reference {} requires image_url",
                index
            )));
        }
        if !image_url.starts_with("data:image/") {
            return Err(ProviderError::Request(format!(
                "image reference {} must be a data:image URL",
                index
            )));
        }
        let mut item = json!({"type": "input_image", "image_url": image_url});
        if let Some(detail) = image.get("detail") {
            if !matches!(detail.as_str(), "auto" | "low" | "high" | "original") {
                return Err(ProviderError::InvalidRequest(
                    "image detail must be one of: auto, low, high, original".to_string(),
                ));
            }
            item["detail"] = Value::String(detail.clone());
        }
        items.push(item);
    }
    Ok(items)
}

fn validate_image_content_values(images: &[Value]) -> Result<Vec<Value>, ProviderError> {
    let mut items = Vec::new();
    for (index, image) in images.iter().enumerate() {
        let object = image.as_object().ok_or_else(|| {
            ProviderError::InvalidRequest(format!("image reference {index} must be an object"))
        })?;
        let image_url = object
            .get("image_url")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if image_url.trim().is_empty() {
            return Err(ProviderError::InvalidRequest(format!(
                "image reference {index} requires image_url"
            )));
        }
        if !image_url.starts_with("data:image/") {
            return Err(ProviderError::InvalidRequest(format!(
                "image reference {index} must be a data:image URL"
            )));
        }
        let mut normalized = serde_json::Map::new();
        normalized.insert("type".to_string(), json!("input_image"));
        normalized.insert("image_url".to_string(), json!(image_url));
        if let Some(detail) = object.get("detail") {
            if !detail.is_null() {
                if !matches!(detail.as_str(), Some("auto" | "low" | "high" | "original")) {
                    return Err(ProviderError::InvalidRequest(
                        "image detail must be one of: auto, low, high, original".to_string(),
                    ));
                }
                normalized.insert("detail".to_string(), detail.clone());
            }
        }
        if let Some(breakpoint) = object.get("prompt_cache_breakpoint") {
            if breakpoint.is_null() {
                items.push(Value::Object(normalized));
                continue;
            }
            let valid = breakpoint.as_object().is_some_and(|object| {
                object.len() == 1 && object.get("mode").and_then(Value::as_str) == Some("explicit")
            });
            if !valid {
                return Err(ProviderError::InvalidRequest(
                    "image prompt_cache_breakpoint must be {\"mode\":\"explicit\"}".to_string(),
                ));
            }
            normalized.insert("prompt_cache_breakpoint".to_string(), breakpoint.clone());
        }
        items.push(Value::Object(normalized));
    }
    Ok(items)
}

fn image_generation_from_item(item: &Value) -> Result<Option<Value>, ProviderError> {
    if item.get("type").and_then(|v| v.as_str()) != Some("image_generation_call") {
        return Ok(None);
    }
    let result = item.get("result").and_then(|v| v.as_str()).unwrap_or("");
    if result.trim().is_empty() {
        return Err(ProviderError::Request(
            "image_generation_call returned empty result".to_string(),
        ));
    }
    let id = item
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
    let status = item
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("completed");
    let revised_prompt = item
        .get("revised_prompt")
        .and_then(|v| v.as_str())
        .map(|s| Value::String(s.to_string()))
        .unwrap_or(Value::Null);

    Ok(Some(json!({
        "id": id,
        "status": status,
        "revised_prompt": revised_prompt,
        "result": result,
    })))
}

fn decode_sse_block(lines: &[String]) -> Result<Option<Value>, ProviderError> {
    let data_lines: Vec<&str> = lines
        .iter()
        .filter_map(|line| {
            if line.starts_with("data:") {
                Some(line[5..].trim())
            } else {
                None
            }
        })
        .collect();

    if data_lines.is_empty() {
        return Ok(None);
    }

    let joined = data_lines.join("\n");
    if joined == "[DONE]" {
        return Ok(None);
    }

    let event: Value = serde_json::from_str(&joined).map_err(|error| {
        ProviderError::Request(format!(
            "ChatGPT OAuth SSE event must be valid JSON: {error}"
        ))
    })?;
    if event.is_object() {
        Ok(Some(event))
    } else {
        Err(ProviderError::Request(
            "ChatGPT OAuth SSE event must be a JSON object".to_string(),
        ))
    }
}

fn is_terminal_response_event(event: &Value) -> bool {
    matches!(
        event.get("type").and_then(Value::as_str),
        Some("response.completed" | "response.failed" | "response.incomplete")
    )
}

fn validate_response_event(event: &Value) -> Result<(), ProviderError> {
    if event.get("type").and_then(Value::as_str) == Some("response.output_item.done") {
        if !event.get("item").is_some_and(Value::is_object) {
            return Err(ProviderError::Request(
                "response.output_item.done must contain an object item".to_string(),
            ));
        }
        return Ok(());
    }
    if event.get("type").and_then(Value::as_str) != Some("response.completed") {
        return Ok(());
    }
    let response = event
        .get("response")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ProviderError::Request(
                "ChatGPT OAuth response.completed requires a response object".to_string(),
            )
        })?;
    if response
        .get("id")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        return Err(ProviderError::Request(
            "ChatGPT OAuth response.completed requires a non-empty response id".to_string(),
        ));
    }
    Ok(())
}

pub fn split_instructions_and_input(
    messages: &[Message],
) -> Result<(String, Vec<Value>), ProviderError> {
    let mut instructions: Vec<String> = Vec::new();
    let mut input_messages: Vec<&Message> = Vec::new();

    for msg in messages {
        if msg.role == MessageRole::System && !msg.content.starts_with(REMOTE_COMPACTION_MARKER) {
            if msg.structured_content.as_ref().is_some_and(|parts| {
                parts.iter().any(|part| {
                    part.get("prompt_cache_breakpoint")
                        .is_some_and(|v| !v.is_null())
                })
            }) {
                return Err(ProviderError::InvalidRequest(
                    "system message content cannot use prompt_cache_breakpoint".to_string(),
                ));
            }
            instructions.push(msg.content.clone());
        } else {
            input_messages.push(msg);
        }
    }

    Ok((
        instructions.join("\n\n"),
        messages_to_response_items_refs(&input_messages)?,
    ))
}

pub fn messages_to_response_items(messages: &[Message]) -> Result<Vec<Value>, ProviderError> {
    let refs: Vec<&Message> = messages.iter().collect();
    messages_to_response_items_refs(&refs)
}

fn messages_to_response_items_refs(messages: &[&Message]) -> Result<Vec<Value>, ProviderError> {
    let mut items: Vec<Value> = Vec::new();

    for message in messages {
        if message.role != MessageRole::User
            && message.structured_content.as_ref().is_some_and(|parts| {
                parts.iter().any(|part| {
                    part.get("prompt_cache_breakpoint")
                        .is_some_and(|value| !value.is_null())
                })
            })
        {
            return Err(ProviderError::InvalidRequest(
                "prompt_cache_breakpoint is supported only on user text/image content".to_string(),
            ));
        }
        if message.role == MessageRole::System
            && message.content.starts_with(REMOTE_COMPACTION_MARKER)
        {
            let raw = message.content[REMOTE_COMPACTION_MARKER.len()..].trim();
            let parsed: Value = serde_json::from_str(raw).map_err(|error| {
                ProviderError::Request(format!(
                    "remote compaction marker must contain valid JSON: {error}"
                ))
            })?;
            let compacted = parsed.as_array().ok_or_else(|| {
                ProviderError::Request(
                    "remote compaction marker must contain a JSON array".to_string(),
                )
            })?;
            items.extend(filter_compacted_history_items(compacted)?);
            continue;
        }

        if message.role == MessageRole::Tool {
            let call_id = message
                .tool_call_id
                .as_deref()
                .or(message.name.as_deref())
                .unwrap_or("tool-call");
            items.push(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": message.content,
            }));
            continue;
        }

        if message.role == MessageRole::Assistant && !message.tool_calls.is_empty() {
            if !message.content.is_empty() || message.structured_content.is_some() {
                items.push(message_item(
                    "assistant",
                    &message.content,
                    &[],
                    message.structured_content.as_deref(),
                ));
            }
            for tc in &message.tool_calls {
                items.push(json!({
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".to_string()),
                }));
            }
            continue;
        }

        let role = if message.role == MessageRole::Assistant {
            "assistant"
        } else {
            "user"
        };
        items.push(message_item(
            role,
            &message.content,
            &message.images,
            message.structured_content.as_deref(),
        ));
    }

    Ok(items)
}

fn message_item(
    role: &str,
    content: &str,
    images: &[String],
    structured_content: Option<&[Value]>,
) -> Value {
    if let Some(structured_content) = structured_content {
        let normalized_content: Vec<Value> = structured_content
            .iter()
            .map(|part| {
                let mut normalized = part.clone();
                if let Some(object) = normalized.as_object_mut() {
                    if object
                        .get("prompt_cache_breakpoint")
                        .is_some_and(Value::is_null)
                    {
                        object.remove("prompt_cache_breakpoint");
                    }
                    if object.get("type").and_then(Value::as_str) == Some("input_image")
                        && object.get("detail").is_some_and(Value::is_null)
                    {
                        object.remove("detail");
                    }
                }
                normalized
            })
            .collect();
        return json!({
            "type": "message",
            "role": role,
            "content": normalized_content,
        });
    }
    let typ = if role == "assistant" {
        "output_text"
    } else {
        "input_text"
    };
    let mut content_items = vec![json!({"type": typ, "text": content})];
    for image_url in images {
        content_items.push(json!({"type": "input_image", "image_url": image_url}));
    }
    content_items[0] = json!({"type": typ, "text": if content.is_empty() { "" } else { content }});
    json!({
        "type": "message",
        "role": role,
        "content": content_items,
    })
}

fn tool_schema_to_response_dict(tool: &ToolSchema) -> Value {
    if tool
        .parameters
        .get("__codex_as_api_tool_type")
        .and_then(|v| v.as_str())
        == Some("web_search")
    {
        return tool
            .parameters
            .get("openai_tool")
            .filter(|v| v.is_object())
            .cloned()
            .unwrap_or_else(|| json!({"type": "web_search", "external_web_access": true}));
    }
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": false,
    })
}

fn finalize_responses_request(
    mut payload: Value,
    model: &str,
    responses_lite: Option<&Value>,
    text: Option<&Value>,
    service_tier: Option<&str>,
    endpoint: ResponsesEndpointKind,
) -> Result<FinalizedResponsesRequest, ProviderError> {
    if !capability_for_model(model).supports_image_detail_original
        && value_has_original_image_detail(&payload)
    {
        return Err(ProviderError::InvalidRequest(format!(
            "image detail 'original' is not supported for model {model:?}"
        )));
    }
    let payload_object = payload.as_object_mut().ok_or_else(|| {
        ProviderError::InvalidRequest("Responses request payload must be an object".to_string())
    })?;
    for field in [
        "previous_response_id",
        "prompt_cache_options",
        "safety_identifier",
    ] {
        if payload_object
            .get(field)
            .is_some_and(|value| !value.is_null())
        {
            return Err(ProviderError::InvalidRequest(format!(
                "{field} is not supported by the Codex OAuth HTTP transport"
            )));
        }
    }
    if payload_object
        .get("input")
        .is_some_and(value_has_prompt_cache_breakpoint)
    {
        return Err(ProviderError::InvalidRequest(
            "prompt_cache_breakpoint is not supported by the Codex OAuth HTTP transport"
                .to_string(),
        ));
    }
    if let Some(mode) = payload_object
        .get("reasoning")
        .and_then(Value::as_object)
        .and_then(|reasoning| reasoning.get("mode"))
        .cloned()
    {
        match mode.as_str() {
            Some("standard") => {
                if let Some(reasoning) = payload_object
                    .get_mut("reasoning")
                    .and_then(Value::as_object_mut)
                {
                    reasoning.remove("mode");
                }
            }
            None if mode.is_null() => {
                if let Some(reasoning) = payload_object
                    .get_mut("reasoning")
                    .and_then(Value::as_object_mut)
                {
                    reasoning.remove("mode");
                }
            }
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
    let conversation_input = payload_object
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    payload_object.insert(
        "model".to_string(),
        Value::String(wire_model_id(model).to_string()),
    );
    apply_model_capability_fields(payload_object, model, text, service_tier)
        .map_err(ProviderError::InvalidRequest)?;
    if payload_object
        .get("reasoning")
        .and_then(Value::as_object)
        .and_then(|reasoning| reasoning.get("effort"))
        .is_none()
    {
        if let Some(default_effort) = capability_for_model(model).default_reasoning_effort {
            set_reasoning_payload_with_options(payload_object, Some(&default_effort), None)?;
        }
    }

    match endpoint {
        ResponsesEndpointKind::Standard => {
            if payload_object.get("reasoning").is_some() {
                let include = payload_object
                    .entry("include".to_string())
                    .or_insert_with(|| json!([]));
                if !include.is_array() {
                    *include = json!([]);
                }
                let include_items = include.as_array_mut().unwrap();
                if !include_items
                    .iter()
                    .any(|value| value.as_str() == Some("reasoning.encrypted_content"))
                {
                    include_items.push(json!("reasoning.encrypted_content"));
                }
            }
        }
        ResponsesEndpointKind::Compact => {
            payload_object.remove("include");
        }
    }

    let lite = use_responses_lite(model, responses_lite).map_err(ProviderError::InvalidRequest)?;
    if lite {
        match endpoint {
            ResponsesEndpointKind::Standard => match payload_object.get("tool_choice") {
                None => {
                    payload_object.insert("tool_choice".to_string(), json!("auto"));
                }
                Some(Value::String(value)) if value == "auto" => {}
                Some(_) => {
                    return Err(ProviderError::InvalidRequest(
                        "Responses Lite tool_choice must be the exact string auto".to_string(),
                    ));
                }
            },
            ResponsesEndpointKind::Compact => {
                payload_object.remove("tool_choice");
            }
        }
        apply_responses_lite_payload(payload_object)?;
    }

    Ok(FinalizedResponsesRequest {
        payload,
        use_responses_lite: lite,
        conversation_input,
    })
}

fn value_has_original_image_detail(value: &Value) -> bool {
    match value {
        Value::Array(items) => items.iter().any(value_has_original_image_detail),
        Value::Object(object) => {
            (object.get("type").and_then(Value::as_str) == Some("input_image")
                && object.get("detail").and_then(Value::as_str) == Some("original"))
                || object.values().any(value_has_original_image_detail)
        }
        _ => false,
    }
}

fn responses_lite_headers(use_responses_lite: bool) -> HashMap<String, String> {
    if use_responses_lite {
        HashMap::from([(LITE_HEADER_NAME.to_string(), LITE_HEADER_VALUE.to_string())])
    } else {
        HashMap::new()
    }
}

fn apply_responses_lite_payload(
    payload: &mut serde_json::Map<String, Value>,
) -> Result<(), ProviderError> {
    let tools_payload = payload
        .get("tools")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    if let Some(tool_type) = tools_payload.iter().find_map(|tool| {
        let tool_type = tool.get("type").and_then(|v| v.as_str())?;
        matches!(tool_type, "web_search" | "image_generation").then_some(tool_type)
    }) {
        return Err(ProviderError::InvalidRequest(format!(
            "Responses Lite cannot use hosted {tool_type} without a standalone executor"
        )));
    }

    let instructions = payload
        .remove("instructions")
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .unwrap_or_default();
    payload.remove("tools");
    payload.insert("parallel_tool_calls".to_string(), Value::Bool(false));
    let input = payload
        .remove("input")
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();
    let mut items = vec![json!({
        "type": "additional_tools",
        "role": "developer",
        "tools": tools_payload,
    })];
    if !instructions.is_empty() {
        items.push(json!({
            "type": "message",
            "role": "developer",
            "content": [{"type": "input_text", "text": instructions}],
        }));
    }
    items.extend(input);
    payload.insert(
        "input".to_string(),
        strip_image_detail_fields(Value::Array(items)),
    );
    let reasoning = payload
        .entry("reasoning".to_string())
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .ok_or_else(|| ProviderError::InvalidRequest("reasoning must be an object".to_string()))?;
    if reasoning
        .get("context")
        .and_then(Value::as_str)
        .is_some_and(|context| context != "all_turns")
    {
        return Err(ProviderError::InvalidRequest(
            "Responses Lite reasoning.context must be all_turns".to_string(),
        ));
    }
    reasoning.insert(
        "context".to_string(),
        Value::String("all_turns".to_string()),
    );

    Ok(())
}

fn compact_raw_events(events: &[Value]) -> Vec<Value> {
    let mut keep: Vec<Value> = events
        .iter()
        .filter(|event| event.get("type").and_then(|v| v.as_str()) == Some("web_search_call"))
        .cloned()
        .collect();
    let start = events.len().saturating_sub(20);
    for event in &events[start..] {
        if !keep.iter().any(|kept| kept == event) {
            keep.push(event.clone());
        }
    }
    keep
}

fn filter_compacted_history_items(items: &[Value]) -> Result<Vec<Value>, ProviderError> {
    let mut compacted = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let object = item.as_object().ok_or_else(|| {
            ProviderError::Request(format!(
                "remote compact output item {index} must be an object"
            ))
        })?;
        validate_compacted_history_item(object, index)?;
        let keep = should_keep_compacted_history_item(object);
        if keep {
            compacted.push(item.clone());
        }
    }
    Ok(compacted)
}

fn validate_compacted_history_item(
    object: &serde_json::Map<String, Value>,
    index: usize,
) -> Result<(), ProviderError> {
    let item_type = object.get("type").and_then(Value::as_str).ok_or_else(|| {
        ProviderError::Request(format!(
            "remote compact output item {index} requires a string type"
        ))
    })?;
    match item_type {
        "message" => validate_compacted_message(object, index),
        "agent_message" => validate_compacted_agent_message(object, index),
        "compaction" | "compaction_summary" => {
            if object
                .get("encrypted_content")
                .and_then(Value::as_str)
                .is_none()
            {
                return Err(ProviderError::Request(format!(
                    "remote compact output {item_type} item {index} requires encrypted_content"
                )));
            }
            Ok(())
        }
        "context_compaction" => match object.get("encrypted_content") {
            None | Some(Value::Null | Value::String(_)) => Ok(()),
            Some(_) => Err(ProviderError::Request(format!(
                "remote compact output context_compaction item {index} encrypted_content must be a string"
            ))),
        },
        _ => Ok(()),
    }
}

fn validate_compacted_message(
    object: &serde_json::Map<String, Value>,
    index: usize,
) -> Result<(), ProviderError> {
    if object.get("role").and_then(Value::as_str).is_none() {
        return Err(ProviderError::Request(format!(
            "remote compact output message {index} requires a string role"
        )));
    }
    let content = object
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ProviderError::Request(format!(
                "remote compact output message {index} requires a content array"
            ))
        })?;
    for (content_index, content_item) in content.iter().enumerate() {
        let content_object = content_item.as_object().ok_or_else(|| {
            ProviderError::Request(format!(
                "remote compact output message {index} content item {content_index} must be an object"
            ))
        })?;
        match content_object.get("type").and_then(Value::as_str) {
            Some("input_text" | "output_text")
                if content_object.get("text").and_then(Value::as_str).is_some() => {}
            Some("input_image")
                if content_object
                    .get("image_url")
                    .and_then(Value::as_str)
                    .is_some()
                    && match content_object.get("detail") {
                        None | Some(Value::Null) => true,
                        Some(Value::String(detail)) => {
                            matches!(detail.as_str(), "auto" | "low" | "high" | "original")
                        }
                        Some(_) => false,
                    } => {}
            _ => {
                return Err(ProviderError::Request(format!(
                    "remote compact output message {index} content item {content_index} is invalid"
                )));
            }
        }
    }
    Ok(())
}

fn validate_compacted_agent_message(
    object: &serde_json::Map<String, Value>,
    index: usize,
) -> Result<(), ProviderError> {
    for field in ["author", "recipient"] {
        if object.get(field).and_then(Value::as_str).is_none() {
            return Err(ProviderError::Request(format!(
                "remote compact output agent_message item {index} requires string {field}"
            )));
        }
    }
    let content = object
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ProviderError::Request(format!(
                "remote compact output agent_message item {index} requires a content array"
            ))
        })?;
    for (content_index, content_item) in content.iter().enumerate() {
        let valid = content_item.as_object().is_some_and(|content| {
            matches!(
                content.get("type").and_then(Value::as_str),
                Some("input_text")
            ) && content.get("text").and_then(Value::as_str).is_some()
                || matches!(
                    content.get("type").and_then(Value::as_str),
                    Some("encrypted_content")
                ) && content
                    .get("encrypted_content")
                    .and_then(Value::as_str)
                    .is_some()
        });
        if !valid {
            return Err(ProviderError::Request(format!(
                "remote compact output agent_message item {index} content item {content_index} is invalid"
            )));
        }
    }
    Ok(())
}

fn should_keep_compacted_history_item(object: &serde_json::Map<String, Value>) -> bool {
    match object.get("type").and_then(Value::as_str) {
        Some("message") => match object.get("role").and_then(Value::as_str) {
            Some("assistant") => true,
            Some("user") => is_real_user_or_hook_message(object.get("content")),
            _ => false,
        },
        Some("agent_message" | "compaction" | "compaction_summary" | "context_compaction") => true,
        _ => false,
    }
}

fn is_real_user_or_hook_message(content: Option<&Value>) -> bool {
    let Some(content) = content.and_then(Value::as_array) else {
        return true;
    };
    let has_visible_hook = content.iter().any(|item| {
        item.get("type").and_then(Value::as_str) == Some("input_text")
            && item
                .get("text")
                .and_then(Value::as_str)
                .is_some_and(is_hook_prompt_text)
    });
    if has_visible_hook
        && content.iter().all(|item| {
            item.get("type").and_then(Value::as_str) == Some("input_text")
                && item
                    .get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| is_hook_prompt_text(text) || is_contextual_user_text(text))
        })
    {
        return true;
    }
    !content.iter().any(|item| {
        item.get("type").and_then(Value::as_str) == Some("input_text")
            && item
                .get("text")
                .and_then(Value::as_str)
                .is_some_and(|text| is_hook_prompt_text(text) || is_contextual_user_text(text))
    })
}

fn is_hook_prompt_text(text: &str) -> bool {
    let trimmed = text.trim();
    let Some(start_tag_end) = trimmed.find('>') else {
        return false;
    };
    let start_tag = &trimmed[..=start_tag_end];
    if !start_tag
        .get(.."<hook_prompt ".len())
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("<hook_prompt "))
        || !trimmed.ends_with("</hook_prompt>")
    {
        return false;
    }
    let Some(attribute_start) = start_tag.find("hook_run_id=\"") else {
        return false;
    };
    let value = &start_tag[attribute_start + "hook_run_id=\"".len()..];
    value
        .find('"')
        .is_some_and(|end| !value[..end].trim().is_empty())
}

fn is_contextual_user_text(text: &str) -> bool {
    const MARKER_PAIRS: &[(&str, &str)] = &[
        ("# AGENTS.md instructions", "</INSTRUCTIONS>"),
        ("<environment_context>", "</environment_context>"),
        ("<skill>", "</skill>"),
        ("<user_shell_command>", "</user_shell_command>"),
        ("<turn_aborted>", "</turn_aborted>"),
        ("<subagent_notification>", "</subagent_notification>"),
        ("<recommended_plugins>", "</recommended_plugins>"),
    ];
    let trimmed = text.trim();
    if MARKER_PAIRS.iter().any(|(start, end)| {
        starts_with_ignore_ascii_case(trimmed, start) && ends_with_ignore_ascii_case(trimmed, end)
    }) {
        return true;
    }
    if is_external_context_wrapper(trimmed) {
        return true;
    }
    if let Some(source) = trimmed.strip_prefix("<codex_internal_context source=\"") {
        if let Some((source, remainder)) = source.split_once("\">") {
            let mut source_chars = source.chars();
            if !source.is_empty()
                && source_chars
                    .next()
                    .is_some_and(|ch| ch.is_ascii_lowercase())
                && source_chars
                    .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
                && remainder.ends_with("</codex_internal_context>")
            {
                return true;
            }
        }
    }
    if starts_with_ignore_ascii_case(trimmed, "<goal_context>")
        && ends_with_ignore_ascii_case(trimmed, "</goal_context>")
    {
        return true;
    }
    trimmed
        .starts_with("Warning: The maximum number of unified exec processes you can keep open is")
        || (trimmed.starts_with("Warning: apply_patch was requested via ")
            && trimmed.ends_with("Use the apply_patch tool instead of exec_command."))
        || trimmed.starts_with(
            "Warning: Your account was flagged for potentially high-risk cyber activity",
        )
}

fn starts_with_ignore_ascii_case(value: &str, prefix: &str) -> bool {
    value
        .get(..prefix.len())
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(prefix))
}

fn ends_with_ignore_ascii_case(value: &str, suffix: &str) -> bool {
    value
        .get(value.len().saturating_sub(suffix.len())..)
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(suffix))
}

fn is_external_context_wrapper(text: &str) -> bool {
    let Some(start_tag_end) = text.find('>') else {
        return false;
    };
    let start_tag = &text[..start_tag_end];
    let Some(key) = start_tag.strip_prefix("<external_") else {
        return false;
    };
    !key.is_empty() && text.ends_with(&format!("</external_{key}>"))
}

fn web_search_event_from_response_item(item: &Value, all_items: &[Value]) -> Option<Value> {
    if item.get("type").and_then(|v| v.as_str()) != Some("web_search_call") {
        return None;
    }
    let raw_id = item
        .get("id")
        .or_else(|| item.get("call_id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
    let tool_id = if raw_id.starts_with("srvtoolu_") {
        raw_id
    } else {
        format!(
            "srvtoolu_{}",
            raw_id
                .chars()
                .filter(|ch| ch.is_ascii_alphanumeric() || *ch == '_')
                .collect::<String>()
        )
    };
    let action = item
        .get("action")
        .filter(|v| v.is_object())
        .cloned()
        .unwrap_or_else(|| json!({}));
    let mut sources = web_search_sources_from_action(&action);
    if sources.is_empty() && !all_items.is_empty() {
        sources.extend(web_search_sources_from_annotations(all_items));
    }
    Some(json!({
        "type": "web_search_call",
        "id": tool_id,
        "input": {"query": web_search_query_from_action(&action)},
        "content": sources,
    }))
}

fn web_search_query_from_action(action: &Value) -> String {
    if let Some(query) = action.get("query").and_then(|v| v.as_str()) {
        return query.to_string();
    }
    if let Some(queries) = action.get("queries").and_then(|v| v.as_array()) {
        if let Some(first) = queries
            .iter()
            .filter_map(|q| q.as_str())
            .find(|q| !q.is_empty())
        {
            return first.to_string();
        }
    }
    action
        .get("url")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string()
}

fn web_search_sources_from_action(action: &Value) -> Vec<Value> {
    normalize_web_search_sources(action.get("sources").unwrap_or(&Value::Null))
}

fn web_search_sources_from_annotations(items: &[Value]) -> Vec<Value> {
    let mut raw = Vec::new();
    for item in items {
        if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
            for part in content {
                if let Some(annotations) = part.get("annotations").and_then(|v| v.as_array()) {
                    for ann in annotations {
                        if ann.get("type").and_then(|v| v.as_str()) == Some("url_citation") {
                            raw.push(ann.clone());
                        }
                    }
                }
            }
        }
    }
    normalize_web_search_sources(&Value::Array(raw))
}

fn normalize_web_search_sources(value: &Value) -> Vec<Value> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let Some(sources) = value.as_array() else {
        return out;
    };
    for source in sources {
        let Some(url) = source.get("url").and_then(|v| v.as_str()) else {
            continue;
        };
        if url.is_empty() || seen.contains(url) {
            continue;
        }
        seen.insert(url.to_string());
        let mut result = serde_json::Map::new();
        result.insert("type".to_string(), json!("web_search_result"));
        result.insert("url".to_string(), json!(url));
        result.insert(
            "title".to_string(),
            json!(source.get("title").and_then(|v| v.as_str()).unwrap_or(url)),
        );
        if let Some(page_age) = source.get("page_age").and_then(|v| v.as_str()) {
            result.insert("page_age".to_string(), json!(page_age));
        }
        out.push(Value::Object(result));
    }
    out
}

fn set_reasoning_payload(
    payload: &mut serde_json::Map<String, Value>,
    reasoning_effort: Option<&str>,
) -> Result<(), ProviderError> {
    let effort = match reasoning_effort {
        Some(e) if !e.is_empty() => e,
        Some(_) => {
            return Err(ProviderError::InvalidRequest(
                "reasoning_effort must be a non-empty string when provided".to_string(),
            ));
        }
        None => return Ok(()),
    };

    let wire_effort = match effort {
        "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" => effort.to_string(),
        "ultra" => "max".to_string(),
        _ => effort.to_string(),
    };

    payload.insert("reasoning".to_string(), json!({"effort": wire_effort}));

    Ok(())
}

fn set_reasoning_payload_with_options(
    payload: &mut serde_json::Map<String, Value>,
    reasoning_effort: Option<&str>,
    reasoning: Option<&Value>,
) -> Result<(), ProviderError> {
    let mut merged = payload
        .get("reasoning")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();

    let mut nested_effort: Option<&str> = None;
    let mut standard_mode = false;
    if let Some(reasoning) = reasoning {
        let object = reasoning.as_object().ok_or_else(|| {
            ProviderError::InvalidRequest("reasoning must be an object".to_string())
        })?;
        for key in object.keys() {
            if !matches!(key.as_str(), "effort" | "mode" | "context") {
                return Err(ProviderError::InvalidRequest(format!(
                    "reasoning.{key} is not supported"
                )));
            }
        }
        if let Some(value) = object.get("effort") {
            if !value.is_null() {
                let effort = value
                    .as_str()
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| {
                        ProviderError::InvalidRequest(
                            "reasoning.effort must be a non-empty string when provided".to_string(),
                        )
                    })?;
                nested_effort = Some(effort);
            }
        }
        if let Some(value) = object.get("mode") {
            if !value.is_null() {
                let mode = value.as_str().ok_or_else(|| {
                    ProviderError::InvalidRequest(
                        "reasoning.mode must be one of: standard, pro".to_string(),
                    )
                })?;
                if !matches!(mode, "standard" | "pro") {
                    return Err(ProviderError::InvalidRequest(
                        "reasoning.mode must be one of: standard, pro".to_string(),
                    ));
                }
                if mode == "pro" {
                    return Err(ProviderError::InvalidRequest(
                        "reasoning.mode pro is not supported by the Codex OAuth HTTP transport"
                            .to_string(),
                    ));
                }
                standard_mode = true;
            }
        }
        if let Some(value) = object.get("context") {
            if !value.is_null() {
                let context = value.as_str().ok_or_else(|| {
                    ProviderError::InvalidRequest(
                        "reasoning.context must be one of: auto, current_turn, all_turns"
                            .to_string(),
                    )
                })?;
                if !matches!(context, "auto" | "current_turn" | "all_turns") {
                    return Err(ProviderError::InvalidRequest(
                        "reasoning.context must be one of: auto, current_turn, all_turns"
                            .to_string(),
                    ));
                }
                merged.insert("context".to_string(), Value::String(context.to_string()));
            }
        }
    }

    if let (Some(top_level), Some(nested)) = (reasoning_effort, nested_effort) {
        if top_level != nested {
            return Err(ProviderError::InvalidRequest(
                "reasoning_effort conflicts with reasoning.effort".to_string(),
            ));
        }
    }
    merged.remove("mode");
    let effort = reasoning_effort
        .or(nested_effort)
        .or_else(|| standard_mode.then_some("medium"));
    if let Some(effort) = effort {
        if effort.is_empty() {
            return Err(ProviderError::InvalidRequest(
                "reasoning_effort must be a non-empty string when provided".to_string(),
            ));
        }
        let wire_effort = if effort == "ultra" { "max" } else { effort };
        merged.insert("effort".to_string(), Value::String(wire_effort.to_string()));
    }

    if !merged.is_empty() {
        payload.insert("reasoning".to_string(), Value::Object(merged));
    }
    Ok(())
}

fn is_gpt_5_6_model(model: &str) -> bool {
    model == "gpt-5.6" || model.starts_with("gpt-5.6-")
}

fn wire_model_id(model: &str) -> &str {
    if model == "gpt-5.6" {
        "gpt-5.6-sol"
    } else {
        model
    }
}

fn apply_generation_controls(
    payload: &mut serde_json::Map<String, Value>,
    model: &str,
    controls: &GenerationControls,
) -> Result<(), ProviderError> {
    if controls
        .reasoning
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|reasoning| reasoning.get("mode"))
        .is_some_and(|mode| !mode.is_null())
        && !is_gpt_5_6_model(model)
    {
        return Err(ProviderError::InvalidRequest(
            "reasoning.mode is supported only for GPT-5.6 models".to_string(),
        ));
    }
    if controls.safety_identifier.is_some() {
        return Err(ProviderError::InvalidRequest(
            "safety_identifier is not supported by the Codex OAuth HTTP transport".to_string(),
        ));
    }
    if let Some(options) = &controls.prompt_cache_options {
        if !options.is_null() {
            return Err(ProviderError::InvalidRequest(
                "prompt_cache_options is not supported by the Codex OAuth HTTP transport"
                    .to_string(),
            ));
        }
    }
    if payload
        .get("input")
        .is_some_and(value_has_prompt_cache_breakpoint)
    {
        return Err(ProviderError::InvalidRequest(
            "prompt_cache_breakpoint is not supported by the Codex OAuth HTTP transport"
                .to_string(),
        ));
    }
    Ok(())
}

fn value_has_prompt_cache_breakpoint(value: &Value) -> bool {
    match value {
        Value::Object(object) => {
            object
                .get("prompt_cache_breakpoint")
                .is_some_and(|value| !value.is_null())
                || object.values().any(value_has_prompt_cache_breakpoint)
        }
        Value::Array(items) => items.iter().any(value_has_prompt_cache_breakpoint),
        _ => false,
    }
}

fn merge_text_verbosity(
    text: Option<&Value>,
    verbosity: Option<&str>,
) -> Result<Option<Value>, ProviderError> {
    let mut object = match text {
        None | Some(Value::Null) => serde_json::Map::new(),
        Some(Value::Object(object)) => object.clone(),
        Some(_) => {
            return Err(ProviderError::InvalidRequest(
                "text must be an object when provided".to_string(),
            ));
        }
    };
    if let Some(existing) = object.get("verbosity") {
        if !existing.is_null() && !matches!(existing.as_str(), Some("low" | "medium" | "high")) {
            return Err(ProviderError::InvalidRequest(
                "text.verbosity must be one of: low, medium, high".to_string(),
            ));
        }
    }
    let Some(verbosity) = verbosity else {
        return Ok(text
            .filter(|value| !value.is_null())
            .map(|_| Value::Object(object)));
    };
    if !matches!(verbosity, "low" | "medium" | "high") {
        return Err(ProviderError::InvalidRequest(
            "verbosity must be one of: low, medium, high".to_string(),
        ));
    }
    if let Some(existing) = object.get("verbosity") {
        if !existing.is_null() && existing.as_str() != Some(verbosity) {
            return Err(ProviderError::InvalidRequest(
                "verbosity conflicts with text.verbosity".to_string(),
            ));
        }
    }
    object.insert(
        "verbosity".to_string(),
        Value::String(verbosity.to_string()),
    );
    Ok(Some(Value::Object(object)))
}

fn tool_call_from_response_item(item: &Value) -> Option<ToolCall> {
    let typ = item.get("type").and_then(|v| v.as_str())?;
    if typ != "function_call" && typ != "custom_tool_call" {
        return None;
    }
    let name = item.get("name").and_then(|v| v.as_str())?;
    if name.is_empty() {
        return None;
    }
    let raw_args = item
        .get("arguments")
        .or_else(|| item.get("input"))
        .cloned()
        .unwrap_or(Value::String("{}".to_string()));

    let args: HashMap<String, Value> = match &raw_args {
        Value::String(s) => {
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
        Value::Object(map) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        _ => HashMap::new(),
    };

    let call_id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());

    Some(ToolCall {
        id: call_id,
        name: name.to_string(),
        arguments: args,
    })
}

pub fn text_from_response_items(items: &[Value]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for item in items {
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if item_type == "output_text" || item_type == "text" {
            if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    parts.push(text.to_string());
                }
            }
            continue;
        }
        if item_type != "message" {
            continue;
        }
        if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
            for part in content {
                match part {
                    Value::String(s) if !s.is_empty() => {
                        parts.push(s.clone());
                    }
                    Value::Object(map) => {
                        let part_type = map.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if part_type == "output_text" || part_type == "text" {
                            if let Some(text) = map.get("text").and_then(|v| v.as_str()) {
                                if !text.is_empty() {
                                    parts.push(text.to_string());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    parts.join("")
}

pub fn usage_from_response(value: &Value) -> Option<Usage> {
    let obj = value.as_object()?;
    let prompt = obj
        .get("input_tokens")
        .or_else(|| obj.get("prompt_tokens"))
        .and_then(|v| v.as_i64())?;
    let completion = obj
        .get("output_tokens")
        .or_else(|| obj.get("completion_tokens"))
        .and_then(|v| v.as_i64())?;
    let total = obj.get("total_tokens").and_then(|v| v.as_i64());

    let mut cached_tokens: i64 = 0;
    let mut cache_write_tokens: i64 = 0;
    let token_details = obj
        .get("input_tokens_details")
        .or_else(|| obj.get("prompt_tokens_details"));
    if let Some(details) = token_details.and_then(|v| v.as_object()) {
        if let Some(ct) = details.get("cached_tokens").and_then(|v| v.as_i64()) {
            cached_tokens = ct;
        }
        if let Some(tokens) = details.get("cache_write_tokens").and_then(Value::as_i64) {
            cache_write_tokens = tokens;
        }
    } else if let Some(ct) = obj.get("cached_input_tokens").and_then(|v| v.as_i64()) {
        cached_tokens = ct;
    } else if let Some(ct) = obj.get("cache_read_input_tokens").and_then(|v| v.as_i64()) {
        cached_tokens = ct;
    }
    if cache_write_tokens == 0 {
        cache_write_tokens = obj
            .get("cache_write_input_tokens")
            .and_then(Value::as_i64)
            .unwrap_or(0);
    }

    let mut usage = Usage::new(prompt, completion, total, cached_tokens);
    usage.cache_write_tokens = cache_write_tokens;
    Some(usage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    #[test]
    fn provider_without_an_override_uses_a_finite_request_timeout() {
        let provider = ChatGPTOAuthProvider::new(
            "gpt-5.5".to_string(),
            "http://127.0.0.1:1".to_string(),
            None,
            None,
        );
        assert_eq!(provider.timeout, Some(CHATGPT_OAUTH_DEFAULT_TIMEOUT));
    }

    #[test]
    fn response_chain_store_clones_items_and_supports_branches() {
        let store = ResponseChainStore::new(256);
        let initial_input = vec![json!({"type": "message", "role": "user", "content": []})];
        let initial_output = vec![json!({
            "type": "reasoning",
            "id": "reasoning-root",
            "encrypted_content": "opaque-root",
            "summary": []
        })];
        store
            .commit("resp-root", &initial_input, &initial_output)
            .unwrap();

        let expected_root: Vec<Value> = initial_input
            .iter()
            .chain(initial_output.iter())
            .cloned()
            .collect();
        let mut caller_copy = store.resolve("resp-root").unwrap();
        caller_copy[1]["encrypted_content"] = json!("mutated");
        assert_eq!(store.resolve("resp-root").unwrap(), expected_root);

        let mut branch_a_input = expected_root.clone();
        branch_a_input.push(json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "A"}]}));
        let mut branch_b_input = expected_root.clone();
        branch_b_input.push(json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "B"}]}));
        store
            .commit(
                "resp-a",
                &branch_a_input,
                &[json!({"type": "function_call", "call_id": "call-a", "name": "a", "arguments": "{}"})],
            )
            .unwrap();
        store
            .commit(
                "resp-b",
                &branch_b_input,
                &[json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "B"}]})],
            )
            .unwrap();

        assert_eq!(store.resolve("resp-a").unwrap()[2], branch_a_input[2]);
        assert_eq!(store.resolve("resp-b").unwrap()[2], branch_b_input[2]);
    }

    #[test]
    fn response_chain_store_uses_lru_eviction() {
        assert_eq!(RESPONSE_CHAIN_CAPACITY, 256);
        let store = ResponseChainStore::new(2);
        let input = [json!({"type": "message", "role": "user", "content": []})];
        store.commit("resp-a", &input, &[]).unwrap();
        store.commit("resp-b", &input, &[]).unwrap();
        store.resolve("resp-a").unwrap();
        store.commit("resp-c", &input, &[]).unwrap();

        assert!(matches!(
            store.resolve("resp-b"),
            Err(ProviderError::InvalidRequest(_))
        ));
        assert!(store.resolve("resp-a").is_ok());
        assert!(store.resolve("resp-c").is_ok());
    }

    #[test]
    fn response_chain_store_allows_concurrent_branches() {
        let store = Arc::new(ResponseChainStore::new(256));
        store
            .commit(
                "resp-root",
                &[json!({"type": "message", "role": "user", "content": []})],
                &[json!({"type": "reasoning", "encrypted_content": "opaque"})],
            )
            .unwrap();
        let barrier = Arc::new(Barrier::new(8));
        let handles: Vec<_> = (0..8)
            .map(|index| {
                let store = Arc::clone(&store);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    let mut input = store.resolve("resp-root").unwrap();
                    input.push(json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": index.to_string()}]}));
                    store
                        .commit(&format!("resp-{index}"), &input, &[])
                        .unwrap();
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        for index in 0..8 {
            let history = store.resolve(&format!("resp-{index}")).unwrap();
            assert_eq!(history[2]["content"][0]["text"], index.to_string());
        }
    }

    #[test]
    fn lite_replay_keeps_only_the_current_developer_prefix() {
        let provider = ChatGPTOAuthProvider::new(
            "gpt-5.6-sol".to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        );
        let root_input = vec![json!({
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Root"}]
        })];
        let root_output = vec![json!({
            "type": "reasoning",
            "id": "reasoning-root",
            "summary": [],
            "encrypted_content": "opaque-root"
        })];
        provider
            .response_chains
            .commit("resp-root", &root_input, &root_output)
            .unwrap();
        let messages = vec![
            Message::new(
                MessageRole::System,
                "Current instructions".to_string(),
                vec![],
                None,
                None,
            )
            .unwrap(),
            Message::new(MessageRole::User, "Next".to_string(), vec![], None, None).unwrap(),
        ];

        let request = provider
            .responses_payload(
                &messages,
                None,
                None,
                None,
                None,
                None,
                Some("resp-root"),
                Some("gpt-5.6-sol"),
                None,
                None,
                None,
                None,
                None,
                Some(&json!(true)),
                None,
            )
            .unwrap();
        let semantic_types: Vec<_> = request
            .conversation_input
            .iter()
            .map(|item| item["type"].as_str().unwrap())
            .collect();
        assert_eq!(semantic_types, vec!["message", "reasoning", "message"]);
        let wire_input = request.payload["input"].as_array().unwrap();
        assert_eq!(
            wire_input
                .iter()
                .filter(|item| item["type"] == "additional_tools")
                .count(),
            1
        );
        assert_eq!(
            wire_input
                .iter()
                .filter(|item| item["role"] == "developer")
                .count(),
            2
        );
        assert_eq!(wire_input.len(), 5);
    }

    #[test]
    fn test_responses_payload_omits_max_output_tokens_when_max_tokens_is_set() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        );
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are helpful.".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
            Message {
                role: MessageRole::User,
                content: "Hello".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
        ];

        let payload = provider
            .responses_payload(
                &messages,
                None,
                None,
                None,
                None,
                Some(1024),
                None,
                Some("gpt-5.5"),
                None,
                None,
                None,
                None,
                None,
                Some(&json!(false)),
                None,
            )
            .unwrap()
            .payload;

        assert!(payload.get("max_output_tokens").is_none());

        assert!(provider
            .responses_payload(
                &messages,
                None,
                None,
                None,
                Some(""),
                None,
                None,
                Some("gpt-5.6-sol"),
                None,
                None,
                None,
                None,
                None,
                Some(&json!(false)),
                None,
            )
            .is_err());
    }

    #[test]
    fn empty_stop_values_are_omitted_from_private_requests() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        );
        let messages = vec![
            Message::new(
                MessageRole::System,
                "You are helpful.".to_string(),
                vec![],
                None,
                None,
            )
            .unwrap(),
            Message::new(MessageRole::User, "Hello".to_string(), vec![], None, None).unwrap(),
        ];
        let empty_stops = [
            vec![],
            vec!["".to_string()],
            vec!["".to_string(), "".to_string()],
        ];

        for stop in &empty_stops {
            let request = provider
                .responses_payload(
                    &messages,
                    None,
                    None,
                    Some(stop),
                    None,
                    None,
                    None,
                    Some("gpt-5.5"),
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(&json!(false)),
                    None,
                )
                .unwrap();
            assert!(request.payload.get("stop").is_none());
        }

        let omitted = provider
            .responses_payload(
                &messages,
                None,
                None,
                None,
                None,
                None,
                None,
                Some("gpt-5.5"),
                None,
                None,
                None,
                None,
                None,
                Some(&json!(false)),
                None,
            )
            .unwrap();
        assert!(omitted.payload.get("stop").is_none());
    }

    #[test]
    fn non_empty_stop_fails_before_auth_or_transport() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            "http://127.0.0.1:9".to_string(),
            Some("/definitely/missing/codex-as-api-auth.json".to_string()),
            Some(std::time::Duration::from_millis(50)),
        );
        let messages = vec![
            Message::new(
                MessageRole::System,
                "You are helpful.".to_string(),
                vec![],
                None,
                None,
            )
            .unwrap(),
            Message::new(MessageRole::User, "Hello".to_string(), vec![], None, None).unwrap(),
        ];
        let stop = vec!["END".to_string()];

        let error = provider
            .chat_stream(
                &messages,
                None,
                None,
                None,
                None,
                Some(&stop),
                None,
                None,
                None,
                None,
                Some("gpt-5.5"),
                None,
                None,
                None,
                None,
                None,
                Some(&json!(false)),
                None,
            )
            .unwrap_err();

        assert!(matches!(
            error,
            ProviderError::InvalidRequest(message) if message.contains("stop is not supported")
        ));
    }

    #[test]
    fn test_responses_payload_includes_web_search_sources() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        );
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are helpful.".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
            Message {
                role: MessageRole::User,
                content: "Hello".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
        ];
        let tools = vec![ToolSchema {
            name: "web_search".to_string(),
            description: "Web search".to_string(),
            parameters: json!({
                "__codex_as_api_tool_type": "web_search",
                "openai_tool": {"type": "web_search", "external_web_access": true},
            }),
        }];

        let payload = provider
            .responses_payload(
                &messages,
                Some(&tools),
                None,
                None,
                None,
                None,
                None,
                Some("gpt-5.5"),
                Some(&json!({"type": "web_search"})),
                None,
                None,
                None,
                None,
                Some(&json!(false)),
                None,
            )
            .unwrap()
            .payload;

        assert_eq!(
            payload["tools"],
            json!([{"type": "web_search", "external_web_access": true}])
        );
        assert_eq!(payload["tool_choice"], json!({"type": "web_search"}));
        assert_eq!(
            payload["include"],
            json!([
                "web_search_call.action.sources",
                "reasoning.encrypted_content"
            ])
        );
    }

    #[test]
    fn test_responses_payload_reasoning_includes_encrypted_content() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        );
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are helpful.".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
            Message {
                role: MessageRole::User,
                content: "Hello".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
        ];

        let payload = provider
            .responses_payload(
                &messages,
                None,
                Some("high"),
                None,
                None,
                None,
                None,
                Some("gpt-5.5"),
                None,
                None,
                None,
                None,
                None,
                Some(&json!(false)),
                None,
            )
            .unwrap()
            .payload;

        assert_eq!(payload["reasoning"], json!({"effort": "high"}));
        assert_eq!(payload["include"], json!(["reasoning.encrypted_content"]));
    }

    #[test]
    fn test_responses_payload_forces_responses_lite_shape() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        );
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are helpful.".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
            Message {
                role: MessageRole::User,
                content: "Hello".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
        ];
        let tools = vec![ToolSchema {
            name: "lookup".to_string(),
            description: "Lookup".to_string(),
            parameters: json!({"type": "object"}),
        }];

        let payload = provider
            .responses_payload(
                &messages,
                Some(&tools),
                Some("low"),
                None,
                None,
                None,
                None,
                Some("gpt-5.5"),
                None,
                None,
                None,
                None,
                None,
                Some(&json!(true)),
                None,
            )
            .unwrap()
            .payload;

        assert!(payload.get("tools").is_none());
        assert!(payload.get("instructions").is_none());
        assert_eq!(payload["tool_choice"], json!("auto"));
        assert_eq!(payload["parallel_tool_calls"], json!(false));
        assert_eq!(payload["reasoning"]["context"], json!("all_turns"));
        assert_eq!(
            payload["input"][0],
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
            payload["input"][1],
            json!({
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": "You are helpful."}]
            })
        );
    }

    #[test]
    fn test_responses_payload_responses_lite_auto_uses_capability_table() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        );
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are helpful.".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
            Message {
                role: MessageRole::User,
                content: "Hello".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
        ];

        let payload = provider
            .responses_payload(
                &messages,
                None,
                None,
                None,
                None,
                None,
                None,
                Some("gpt-5.5"),
                None,
                Some("priority"),
                None,
                None,
                None,
                Some(&json!("auto")),
                None,
            )
            .unwrap()
            .payload;
        let unknown_error = provider
            .responses_payload(
                &messages,
                None,
                None,
                None,
                None,
                None,
                None,
                Some("unknown-model"),
                None,
                Some("priority"),
                None,
                None,
                None,
                Some(&json!("auto")),
                None,
            )
            .unwrap_err();

        assert!(payload.get("tools").is_some());
        assert_eq!(payload["text"], json!({"verbosity": "low"}));
        assert_eq!(payload["service_tier"], json!("priority"));
        assert!(matches!(unknown_error, ProviderError::InvalidRequest(_)));
    }

    #[test]
    fn test_responses_payload_parallel_tool_calls_uses_capability_table() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        );
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are helpful.".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
            Message {
                role: MessageRole::User,
                content: "Hello".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
        ];

        let payload = provider
            .responses_payload(
                &messages,
                None,
                None,
                None,
                None,
                None,
                None,
                Some("gpt-5.5"),
                None,
                None,
                None,
                None,
                None,
                Some(&json!(false)),
                Some(true),
            )
            .unwrap()
            .payload;
        let spark_payload = provider
            .responses_payload(
                &messages,
                None,
                None,
                None,
                None,
                None,
                None,
                Some("gpt-5.3-codex-spark"),
                None,
                None,
                None,
                None,
                None,
                Some(&json!(false)),
                Some(true),
            )
            .unwrap()
            .payload;
        let lite_payload = provider
            .responses_payload(
                &messages,
                None,
                None,
                None,
                None,
                None,
                None,
                Some("gpt-5.5"),
                None,
                None,
                None,
                None,
                None,
                Some(&json!(true)),
                Some(true),
            )
            .unwrap()
            .payload;

        assert_eq!(payload["parallel_tool_calls"], json!(true));
        assert_eq!(spark_payload["parallel_tool_calls"], json!(false));
        assert_eq!(lite_payload["parallel_tool_calls"], json!(false));
    }

    #[test]
    fn test_codex_cli_headers_include_official_originator_and_versioned_user_agent() {
        let headers = codex_cli_headers_for_version(Some("1.2.3\n"));

        assert_eq!(headers.get("originator").unwrap(), "codex_cli_rs");
        let user_agent = headers.get("User-Agent").unwrap();
        assert!(user_agent.starts_with("codex_cli_rs/1.2.3 ("));
        assert!(user_agent.ends_with(") codex-as-api"));
    }

    #[test]
    fn test_codex_cli_headers_omit_user_agent_for_invalid_version() {
        let headers = codex_cli_headers_for_version(Some("not-a-version"));

        assert_eq!(headers.get("originator").unwrap(), "codex_cli_rs");
        assert!(!headers.contains_key("User-Agent"));
    }

    #[test]
    fn test_split_instructions_and_input() {
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are helpful.".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
            Message {
                role: MessageRole::User,
                content: "Hello".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            },
        ];
        let (instructions, input) = split_instructions_and_input(&messages).unwrap();
        assert_eq!(instructions, "You are helpful.");
        assert_eq!(input.len(), 1);
    }

    #[test]
    fn compact_history_filter_matches_official_replacement_history_shape() {
        let items = json!([
            {"type": "additional_tools", "role": "developer", "tools": []},
            {"type": "message", "role": "developer", "content": []},
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "<environment_context>\n<cwd>/tmp</cwd>\n</environment_context>"}]
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "<skill>\nname: review\n</skill>"}]
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "<external_design>\ncontext\n</external_design>"}]
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "<hook_prompt hook_run_id=\"run-1\">retry</hook_prompt>"}]
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "real question"}]
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "answer"}]
            },
            {"type": "reasoning", "id": "r1", "summary": [], "encrypted_content": null},
            {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
            {"type": "compaction_summary", "encrypted_content": "summary"},
            {"type": "context_compaction", "id": "context-1"}
        ]);
        let filtered = filter_compacted_history_items(items.as_array().unwrap()).unwrap();

        assert_eq!(
            filtered,
            vec![
                items[5].clone(),
                items[6].clone(),
                items[7].clone(),
                items[10].clone(),
                items[11].clone()
            ]
        );
    }

    #[test]
    fn malformed_remote_compaction_markers_fail_loudly() {
        for history in ["not-json", "{}", "[42]"] {
            let marker = Message {
                role: MessageRole::System,
                content: format!("{REMOTE_COMPACTION_MARKER}\n{history}"),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
                structured_content: None,
            };
            assert!(messages_to_response_items(&[marker]).is_err());
        }
    }

    #[test]
    fn malformed_compacted_message_objects_fail_before_filtering() {
        for item in [
            json!({"type": "message", "content": []}),
            json!({"type": "message", "role": "developer", "content": "stale"}),
            json!({"type": "message", "role": "user", "content": [42]}),
            json!({"type": "message", "role": "assistant", "content": [{"type": "output_text"}]}),
            json!({"role": "user", "content": []}),
            json!({"type": "agent_message", "author": "worker", "content": []}),
            json!({"type": "agent_message", "author": "worker", "recipient": "parent", "content": [{"type": "input_text"}]}),
            json!({"type": "compaction"}),
            json!({"type": "compaction_summary", "encrypted_content": 42}),
            json!({"type": "context_compaction", "encrypted_content": 42}),
        ] {
            assert!(filter_compacted_history_items(&[item]).is_err());
        }

        let valid = json!([
            {
                "type": "agent_message",
                "author": "worker",
                "recipient": "parent",
                "content": [
                    {"type": "input_text", "text": "done"},
                    {"type": "encrypted_content", "encrypted_content": "opaque"}
                ]
            },
            {"type": "compaction", "encrypted_content": "summary"},
            {"type": "context_compaction", "encrypted_content": null}
        ]);
        assert_eq!(
            filter_compacted_history_items(valid.as_array().unwrap()).unwrap(),
            valid.as_array().unwrap().clone()
        );
    }

    #[test]
    fn visible_hook_messages_are_kept_only_when_every_part_is_hook_context() {
        let hook = json!({
            "type": "input_text",
            "text": "<hook_prompt hook_run_id=\"run-1\">retry</hook_prompt>"
        });
        assert!(is_real_user_or_hook_message(Some(&json!([hook.clone()]))));
        assert!(!is_real_user_or_hook_message(Some(&json!([
            hook,
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        ]))));
        assert!(is_real_user_or_hook_message(Some(&json!([
            {"type": "input_text", "text": "real question"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"}
        ]))));
    }

    #[test]
    fn test_message_item_user() {
        let item = message_item("user", "hello", &[], None);
        assert_eq!(item["type"], "message");
        assert_eq!(item["role"], "user");
        let content = item["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[0]["text"], "hello");
    }

    #[test]
    fn test_message_item_assistant() {
        let item = message_item("assistant", "response", &[], None);
        let content = item["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "output_text");
    }

    #[test]
    fn test_tool_schema_to_response_dict() {
        let schema = ToolSchema {
            name: "my_tool".to_string(),
            description: "Does stuff".to_string(),
            parameters: json!({"type": "object"}),
        };
        let result = tool_schema_to_response_dict(&schema);
        assert_eq!(result["type"], "function");
        assert_eq!(result["name"], "my_tool");
        assert_eq!(result["strict"], false);
    }

    #[test]
    fn test_set_reasoning_payload_valid() {
        let mut payload = serde_json::Map::new();
        set_reasoning_payload(&mut payload, Some("high")).unwrap();
        assert_eq!(payload["reasoning"], json!({"effort": "high"}));
        assert!(payload.get("include").is_none());
    }

    #[test]
    fn test_set_reasoning_payload_rejects_empty_effort() {
        let mut payload = serde_json::Map::new();
        let result = set_reasoning_payload(&mut payload, Some(""));
        assert!(result.is_err());
    }

    #[test]
    fn test_set_reasoning_payload_maps_exact_ultra_and_preserves_custom_efforts() {
        let cases = [
            ("HIGH", "HIGH"),
            ("MaX", "MaX"),
            ("ultra", "max"),
            ("ULTRA", "ULTRA"),
            ("FutureEffort", "FutureEffort"),
        ];

        for (provided, expected_wire) in cases {
            let mut payload = serde_json::Map::new();
            set_reasoning_payload(&mut payload, Some(provided)).unwrap();
            assert_eq!(payload["reasoning"]["effort"], json!(expected_wire));
        }
    }

    #[test]
    fn test_lite_finalizer_rejects_hosted_tools_without_an_executor() {
        for tool_type in ["web_search", "image_generation"] {
            let payload = json!({
                "model": "gpt-5.6-sol",
                "instructions": "Use the tool.",
                "input": [],
                "tools": [{"type": tool_type}],
                "tool_choice": "auto",
                "parallel_tool_calls": true,
            });

            let result = finalize_responses_request(
                payload,
                "gpt-5.6-sol",
                Some(&json!(true)),
                None,
                None,
                ResponsesEndpointKind::Standard,
            );

            assert!(matches!(result, Err(ProviderError::InvalidRequest(_))));
        }
    }

    #[test]
    fn standard_lite_rejects_non_auto_tool_choice_and_defaults_missing_choice() {
        for tool_choice in [
            json!("required"),
            json!({"type": "function", "name": "lookup"}),
        ] {
            let payload = json!({
                "model": "gpt-5.6-sol",
                "instructions": "Use tools.",
                "input": [],
                "tools": [],
                "tool_choice": tool_choice,
                "parallel_tool_calls": false,
            });
            assert!(finalize_responses_request(
                payload,
                "gpt-5.6-sol",
                Some(&json!(true)),
                None,
                None,
                ResponsesEndpointKind::Standard,
            )
            .is_err());
        }

        let request = finalize_responses_request(
            json!({
                "model": "gpt-5.6-sol",
                "instructions": "No tools.",
                "input": [],
                "tools": [],
                "parallel_tool_calls": false,
            }),
            "gpt-5.6-sol",
            Some(&json!(true)),
            None,
            None,
            ResponsesEndpointKind::Standard,
        )
        .unwrap();
        assert_eq!(request.payload["tool_choice"], json!("auto"));
    }

    #[test]
    fn test_lite_can_be_disabled_for_catalog_lite_model() {
        let payload = json!({
            "model": "gpt-5.6-sol",
            "instructions": "Stay classic.",
            "input": [],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
        });

        let request = finalize_responses_request(
            payload,
            "gpt-5.6-sol",
            Some(&json!(false)),
            None,
            None,
            ResponsesEndpointKind::Standard,
        )
        .unwrap();

        assert!(!request.use_responses_lite);
        assert_eq!(request.payload["instructions"], json!("Stay classic."));
        assert_eq!(request.payload["tools"], json!([]));
        assert_eq!(request.payload["parallel_tool_calls"], json!(true));
    }

    #[test]
    fn original_image_detail_is_gated_by_the_effective_model() {
        let payload = || {
            json!({
                "model": "placeholder",
                "instructions": "Inspect the image.",
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AAAA",
                        "detail": "original"
                    }]
                }],
                "tools": [],
                "tool_choice": "auto",
                "parallel_tool_calls": false,
            })
        };

        for model in [
            "gpt-5.6",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
        ] {
            let request = finalize_responses_request(
                payload(),
                model,
                Some(&json!(false)),
                None,
                None,
                ResponsesEndpointKind::Standard,
            )
            .unwrap();
            assert_eq!(
                request.payload["input"][0]["content"][0]["detail"],
                "original"
            );
        }

        for model in [
            "gpt-5.2",
            "gpt-5.3-codex",
            "gpt-5.3-codex-spark",
            "future-model",
        ] {
            assert!(matches!(
                finalize_responses_request(
                    payload(),
                    model,
                    Some(&json!(false)),
                    None,
                    None,
                    ResponsesEndpointKind::Standard,
                ),
                Err(ProviderError::InvalidRequest(_))
            ));
        }
    }

    #[test]
    fn gpt_5_2_keeps_non_original_image_detail_modes() {
        for detail in ["auto", "low", "high"] {
            let request = finalize_responses_request(
                json!({
                    "model": "gpt-5.2",
                    "instructions": "Inspect the image.",
                    "input": [{
                        "type": "message",
                        "role": "user",
                        "content": [{
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AAAA",
                            "detail": detail
                        }]
                    }],
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": false,
                }),
                "gpt-5.2",
                Some(&json!(false)),
                None,
                None,
                ResponsesEndpointKind::Standard,
            )
            .unwrap();
            assert_eq!(request.payload["input"][0]["content"][0]["detail"], detail);
        }
    }

    #[test]
    fn test_finalizer_applies_catalog_reasoning_and_verbosity_defaults() {
        let payload = json!({
            "model": "gpt-5.6-terra",
            "instructions": "Use catalog defaults.",
            "input": [],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "include": [],
        });

        let request = finalize_responses_request(
            payload,
            "gpt-5.6-terra",
            Some(&json!("auto")),
            None,
            None,
            ResponsesEndpointKind::Standard,
        )
        .unwrap();

        assert!(request.use_responses_lite);
        assert_eq!(request.payload["reasoning"]["effort"], json!("medium"));
        assert_eq!(request.payload["reasoning"]["context"], json!("all_turns"));
        assert_eq!(request.payload["text"], json!({"verbosity": "low"}));
        assert_eq!(
            request.payload["include"],
            json!(["reasoning.encrypted_content"])
        );
    }

    #[test]
    fn public_gpt_5_6_alias_uses_sol_on_the_wire() {
        let request = finalize_responses_request(
            json!({
                "model": "gpt-5.6",
                "instructions": "Resolve the public alias.",
                "input": [],
                "tools": [],
                "tool_choice": "auto",
                "parallel_tool_calls": true,
                "include": [],
            }),
            "gpt-5.6",
            Some(&json!(false)),
            None,
            None,
            ResponsesEndpointKind::Standard,
        )
        .unwrap();

        assert_eq!(request.payload["model"], json!("gpt-5.6-sol"));
        assert_eq!(request.payload["reasoning"]["effort"], json!("low"));
    }

    #[test]
    fn test_forced_lite_unknown_model_does_not_invent_reasoning() {
        let payload = json!({
            "model": "unknown-model",
            "instructions": "No reasoning metadata exists.",
            "input": [],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "include": [],
        });

        let request = finalize_responses_request(
            payload,
            "unknown-model",
            Some(&json!(true)),
            None,
            None,
            ResponsesEndpointKind::Standard,
        )
        .unwrap();

        assert!(request.use_responses_lite);
        assert_eq!(
            request.payload["reasoning"],
            json!({"context": "all_turns"})
        );
        assert_eq!(request.payload["include"], json!([]));
    }

    #[test]
    fn test_compact_finalizer_omits_include_and_uses_lite_shape() {
        let mut payload = json!({
            "model": "gpt-5.6-sol",
            "instructions": "Compact the conversation.",
            "input": [],
            "tools": [],
            "tool_choice": "required",
            "parallel_tool_calls": true,
        });
        set_reasoning_payload(payload.as_object_mut().unwrap(), Some("low")).unwrap();

        let request = finalize_responses_request(
            payload,
            "gpt-5.6-sol",
            Some(&json!(true)),
            None,
            None,
            ResponsesEndpointKind::Compact,
        )
        .unwrap();

        assert!(request.use_responses_lite);
        assert!(request.payload.get("include").is_none());
        assert!(request.payload.get("tool_choice").is_none());
        assert_eq!(
            request.payload["reasoning"],
            json!({"effort": "low", "context": "all_turns"})
        );
        assert_eq!(
            request.payload["input"][0]["type"],
            json!("additional_tools")
        );
    }

    #[test]
    fn test_set_reasoning_payload_none() {
        let mut payload = serde_json::Map::new();
        set_reasoning_payload(&mut payload, None).unwrap();
        assert!(!payload.contains_key("reasoning"));
    }

    #[test]
    fn test_text_from_response_items() {
        let items = vec![
            json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello "}]}),
            json!({"type": "output_text", "text": "world"}),
        ];
        assert_eq!(text_from_response_items(&items), "Hello world");
    }

    #[test]
    fn test_text_from_response_items_empty() {
        let items: Vec<Value> = vec![];
        assert_eq!(text_from_response_items(&items), "");
    }

    #[test]
    fn test_usage_from_response_valid() {
        let val = json!({
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_tokens_details": {"cached_tokens": 30}
        });
        let usage = usage_from_response(&val).unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.cached_tokens, 30);
    }

    #[test]
    fn test_usage_from_response_alternative_keys() {
        let val = json!({
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "cached_input_tokens": 50
        });
        let usage = usage_from_response(&val).unwrap();
        assert_eq!(usage.prompt_tokens, 200);
        assert_eq!(usage.cached_tokens, 50);
    }

    #[test]
    fn test_usage_from_response_invalid() {
        let val = json!({"foo": "bar"});
        assert!(usage_from_response(&val).is_none());
    }

    #[test]
    fn test_decode_sse_block_valid() {
        let lines = vec!["data: {\"type\": \"response.completed\"}".to_string()];
        let event = decode_sse_block(&lines).unwrap().unwrap();
        assert_eq!(event["type"], "response.completed");
    }

    #[test]
    fn test_decode_sse_block_done() {
        let lines = vec!["data: [DONE]".to_string()];
        assert!(decode_sse_block(&lines).unwrap().is_none());
    }

    #[test]
    fn test_decode_sse_block_no_data() {
        let lines = vec!["event: ping".to_string()];
        assert!(decode_sse_block(&lines).unwrap().is_none());
    }

    #[test]
    fn test_decode_sse_block_rejects_invalid_json_and_nonobject_data() {
        for line in ["data: {", "data: []"] {
            assert!(decode_sse_block(&[line.to_string()]).is_err());
        }
    }

    #[test]
    fn response_events_require_output_item_objects_and_completed_ids() {
        assert!(validate_response_event(&json!({
            "type": "response.completed",
            "response": {"id": "resp-1"}
        }))
        .is_ok());
        assert!(validate_response_event(&json!({
            "type": "response.output_item.done",
            "item": {"type": "message", "role": "assistant", "content": []}
        }))
        .is_ok());
        assert!(validate_response_event(&json!({
            "type": "response.completed",
            "response": {"id": "resp-1", "output": ["ignored-extra-field"]}
        }))
        .is_ok());
        for event in [
            json!({"type": "response.completed"}),
            json!({"type": "response.completed", "response": []}),
            json!({"type": "response.completed", "response": {}}),
            json!({"type": "response.completed", "response": {"id": ""}}),
            json!({"type": "response.completed", "response": {"id": 42}}),
            json!({"type": "response.output_item.done"}),
            json!({"type": "response.output_item.done", "item": []}),
        ] {
            assert!(validate_response_event(&event).is_err());
        }
    }

    #[test]
    fn test_tool_call_from_response_item_function() {
        let item = json!({
            "type": "function_call",
            "name": "read_file",
            "call_id": "call-1",
            "arguments": "{\"path\": \"/tmp/test\"}"
        });
        let tc = tool_call_from_response_item(&item).unwrap();
        assert_eq!(tc.name, "read_file");
        assert_eq!(tc.id, "call-1");
        assert_eq!(tc.arguments["path"], "/tmp/test");
    }

    #[test]
    fn test_tool_call_from_response_item_not_function() {
        let item = json!({"type": "message"});
        assert!(tool_call_from_response_item(&item).is_none());
    }

    #[test]
    fn test_messages_to_response_items_tool() {
        let msg = Message {
            role: MessageRole::Tool,
            content: "result".to_string(),
            tool_calls: vec![],
            tool_call_id: Some("call-1".to_string()),
            name: Some("my_tool".to_string()),
            reasoning_content: None,
            images: vec![],
            structured_content: None,
        };
        let items = messages_to_response_items(&[msg]).unwrap();
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "function_call_output");
        assert_eq!(items[0]["call_id"], "call-1");
    }

    #[test]
    fn test_validate_image_content_items_valid() {
        let mut img = HashMap::new();
        img.insert(
            "image_url".to_string(),
            "data:image/png;base64,abc".to_string(),
        );
        img.insert("detail".to_string(), "high".to_string());
        let result = validate_image_content_items(&[img]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "input_image");
        assert_eq!(result[0]["detail"], "high");
    }

    #[test]
    fn test_validate_image_content_items_missing_url() {
        let img = HashMap::new();
        let result = validate_image_content_items(&[img]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_image_content_items_wrong_prefix() {
        let mut img = HashMap::new();
        img.insert(
            "image_url".to_string(),
            "https://example.com/img.png".to_string(),
        );
        let result = validate_image_content_items(&[img]);
        assert!(result.is_err());
    }

    #[test]
    fn reasoning_standard_mode_supplies_medium_without_a_wire_mode() {
        let mut payload = json!({"reasoning": {"summary": "auto"}});
        set_reasoning_payload_with_options(
            payload.as_object_mut().unwrap(),
            None,
            Some(&json!({"mode": "standard", "context": "current_turn"})),
        )
        .unwrap();
        assert_eq!(
            payload["reasoning"],
            json!({
                "summary": "auto",
                "effort": "medium",
                "context": "current_turn"
            })
        );
        let mut pro_payload = json!({});
        assert!(matches!(
            set_reasoning_payload_with_options(
                pro_payload.as_object_mut().unwrap(),
                None,
                Some(&json!({"mode": "pro"})),
            ),
            Err(ProviderError::InvalidRequest(_))
        ));
    }

    #[test]
    fn reasoning_options_reject_conflicts_and_invalid_values() {
        let cases = [
            (Some("high"), json!({"effort": "low"})),
            (None, json!({"mode": "turbo"})),
            (None, json!({"context": "forever"})),
            (None, json!({"effort": ""})),
        ];
        for (effort, reasoning) in cases {
            let mut payload = json!({});
            assert!(set_reasoning_payload_with_options(
                payload.as_object_mut().unwrap(),
                effort,
                Some(&reasoning),
            )
            .is_err());
        }
    }

    #[test]
    fn lite_rejects_explicit_non_all_turns_context() {
        let mut payload = json!({
            "input": [],
            "tools": [],
            "reasoning": {"context": "current_turn"}
        });
        let error = apply_responses_lite_payload(payload.as_object_mut().unwrap()).unwrap_err();
        assert!(error.to_string().contains("must be all_turns"));
    }

    #[test]
    fn lite_strips_image_detail_but_preserves_explicit_cache_breakpoint() {
        let mut payload = json!({
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "original",
                    "prompt_cache_breakpoint": {"mode": "explicit"}
                }]
            }],
            "tools": [],
            "reasoning": {"context": "all_turns"}
        });
        apply_responses_lite_payload(payload.as_object_mut().unwrap()).unwrap();
        let image = &payload["input"][1]["content"][0];
        assert!(image.get("detail").is_none());
        assert_eq!(
            image["prompt_cache_breakpoint"],
            json!({"mode": "explicit"})
        );
    }

    #[test]
    fn generation_controls_accept_standard_mode_only_for_gpt_5_6() {
        let controls = GenerationControls {
            reasoning: Some(json!({"mode": "standard"})),
            verbosity: Some("high".to_string()),
            ..GenerationControls::default()
        };
        let mut payload = json!({});
        apply_generation_controls(payload.as_object_mut().unwrap(), "gpt-5.6-sol", &controls)
            .unwrap();
        let mut other = json!({});
        assert!(matches!(
            apply_generation_controls(other.as_object_mut().unwrap(), "gpt-5.5", &controls),
            Err(ProviderError::InvalidRequest(_))
        ));
    }

    #[test]
    fn generation_controls_reject_all_safety_and_prompt_cache_options() {
        for controls in [
            GenerationControls {
                safety_identifier: Some(" ".to_string()),
                ..GenerationControls::default()
            },
            GenerationControls {
                safety_identifier: Some("x".repeat(65)),
                ..GenerationControls::default()
            },
            GenerationControls {
                prompt_cache_options: Some(json!({"mode": "manual"})),
                ..GenerationControls::default()
            },
            GenerationControls {
                prompt_cache_options: Some(json!({"ttl": "1h"})),
                ..GenerationControls::default()
            },
            GenerationControls {
                prompt_cache_options: Some(json!({"mode": "implicit", "ttl": "30m"})),
                ..GenerationControls::default()
            },
            GenerationControls {
                prompt_cache_options: Some(json!({"mode": "explicit", "ttl": "30m"})),
                ..GenerationControls::default()
            },
        ] {
            let mut payload = json!({});
            assert!(matches!(
                apply_generation_controls(
                    payload.as_object_mut().unwrap(),
                    "gpt-5.6-sol",
                    &controls,
                ),
                Err(ProviderError::InvalidRequest(_))
            ));
        }
    }

    #[test]
    fn cache_breakpoints_are_always_rejected() {
        let mut payload = json!({
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "cache here",
                    "prompt_cache_breakpoint": {"mode": "explicit"}
                }]
            }]
        });
        let controls = GenerationControls {
            prompt_cache_options: Some(json!({"mode": "implicit"})),
            ..GenerationControls::default()
        };
        assert!(matches!(
            apply_generation_controls(payload.as_object_mut().unwrap(), "gpt-5.6-sol", &controls,),
            Err(ProviderError::InvalidRequest(_))
        ));

        let mut default_implicit = json!({
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_text",
                    "text": "cache here",
                    "prompt_cache_breakpoint": {"mode": "explicit"}
                }]
            }]
        });
        assert!(matches!(
            apply_generation_controls(
                default_implicit.as_object_mut().unwrap(),
                "gpt-5.6-sol",
                &GenerationControls::default(),
            ),
            Err(ProviderError::InvalidRequest(_))
        ));
        assert!(matches!(
            apply_generation_controls(
                default_implicit.as_object_mut().unwrap(),
                "gpt-5.5",
                &GenerationControls::default(),
            ),
            Err(ProviderError::InvalidRequest(_))
        ));
    }

    #[test]
    fn tool_schema_property_named_prompt_cache_breakpoint_is_preserved() {
        let request = finalize_responses_request(
            json!({
                "model": "gpt-5.6-sol",
                "instructions": "Use tools.",
                "input": [],
                "tools": [{
                    "type": "function",
                    "name": "lookup",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt_cache_breakpoint": {"type": "string"}
                        }
                    }
                }],
                "tool_choice": "auto",
                "parallel_tool_calls": true
            }),
            "gpt-5.6-sol",
            Some(&json!(false)),
            None,
            None,
            ResponsesEndpointKind::Standard,
        )
        .unwrap();

        assert_eq!(
            request.payload["tools"][0]["parameters"]["properties"]["prompt_cache_breakpoint"],
            json!({"type": "string"})
        );
    }

    #[test]
    fn merge_text_verbosity_rejects_conflict() {
        assert_eq!(
            merge_text_verbosity(Some(&json!({"format": {"type": "text"}})), Some("high")).unwrap(),
            Some(json!({"format": {"type": "text"}, "verbosity": "high"}))
        );
        assert!(merge_text_verbosity(Some(&json!({"verbosity": "low"})), Some("high")).is_err());
        assert!(merge_text_verbosity(Some(&json!(42)), None).is_err());
        assert!(merge_text_verbosity(Some(&json!({"verbosity": "verbose"})), None).is_err());
    }

    #[test]
    fn structured_message_content_is_preserved() {
        let message = Message {
            role: MessageRole::User,
            content: "look".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            images: vec![],
            structured_content: Some(vec![
                json!({
                    "type": "input_text",
                    "text": "look",
                    "prompt_cache_breakpoint": {"mode": "explicit"}
                }),
                json!({
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA",
                    "detail": "original"
                }),
            ]),
        };
        let items = messages_to_response_items(&[message]).unwrap();
        assert_eq!(
            items[0]["content"][0]["prompt_cache_breakpoint"]["mode"],
            "explicit"
        );
        assert_eq!(items[0]["content"][1]["detail"], "original");
    }

    #[test]
    fn null_prompt_cache_breakpoint_is_omitted_from_structured_content() {
        let message = Message {
            role: MessageRole::User,
            content: "hello".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            images: vec![],
            structured_content: Some(vec![json!({
                "type": "input_text",
                "text": "hello",
                "prompt_cache_breakpoint": null
            })]),
        };

        let items = messages_to_response_items(&[message]).unwrap();
        assert_eq!(
            items[0]["content"],
            json!([{"type": "input_text", "text": "hello"}])
        );
    }

    #[test]
    fn direct_non_user_message_cache_breakpoints_are_rejected() {
        let system_message = Message {
            role: MessageRole::System,
            content: "system".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            images: vec![],
            structured_content: Some(vec![json!({
                "type": "input_text",
                "text": "system",
                "prompt_cache_breakpoint": {"mode": "explicit"}
            })]),
        };
        assert!(split_instructions_and_input(&[system_message]).is_err());

        let assistant_message = Message {
            role: MessageRole::Assistant,
            content: "prior".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            images: vec![],
            structured_content: Some(vec![json!({
                "type": "output_text",
                "text": "prior",
                "prompt_cache_breakpoint": {"mode": "explicit"}
            })]),
        };
        assert!(messages_to_response_items(&[assistant_message]).is_err());
    }

    #[test]
    fn usage_parses_cache_write_tokens() {
        let usage = usage_from_response(&json!({
            "input_tokens": 10,
            "output_tokens": 2,
            "input_tokens_details": {
                "cached_tokens": 3,
                "cache_write_tokens": 7
            }
        }))
        .unwrap();
        assert_eq!(usage.cached_tokens, 3);
        assert_eq!(usage.cache_write_tokens, 7);
    }
}
