use crate::auth::{self, AuthError, ChatGPTTokenData};
use crate::messages::{AssistantResponse, Message, MessageRole, ToolCall, ToolSchema, Usage};
use crate::model_capabilities::{
    apply_model_capability_fields, build_codex_client_metadata, resolve_codex_metadata_enabled,
    should_enable_parallel_tool_calls, use_responses_lite, LITE_HEADER_NAME, LITE_HEADER_VALUE,
    SESSION_ID_KEY, THREAD_ID_KEY,
};
use crate::model_catalog::{
    CatalogError, CatalogKey, ModelCatalogCache, ModelCatalogSnapshot, ModelInfo,
    DEFAULT_CATALOG_TTL,
};
use crate::protocol::{
    reasoning_from_response_items, reasoning_parts_from_response_items, response_failure_message,
};
use crate::strict_json;
use futures::StreamExt;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

pub const CHATGPT_OAUTH_DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
// The pinned Codex provider applies a 300-second idle timeout to each stream read rather than a
// total lifetime cutoff, so active long-lived SSE responses remain valid.
pub const CHATGPT_OAUTH_DEFAULT_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(5 * 60);
pub(crate) const MODEL_CATALOG_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);
pub const REMOTE_COMPACTION_MARKER: &str = "[Remote Responses compacted history]";
const CODEX_CLI_ORIGINATOR: &str = "codex_cli_rs";
const CODEX_UPSTREAM_CONTRACT_JSON: &str =
    include_str!("../../config/codex-upstream-contract.json");
static CODEX_COMPATIBILITY_VERSION: OnceLock<String> = OnceLock::new();
const RESPONSE_CHAIN_CAPACITY: usize = 256;

fn has_malformed_percent_escape(value: &str) -> bool {
    let bytes = value.as_bytes();
    (0..bytes.len()).any(|index| {
        bytes[index] == b'%'
            && (index + 2 >= bytes.len()
                || !bytes[index + 1].is_ascii_hexdigit()
                || !bytes[index + 2].is_ascii_hexdigit())
    })
}

fn ensure_catalog_account_unchanged(
    initial_account_id: &str,
    refreshed_account_id: &str,
) -> Result<(), CatalogError> {
    if initial_account_id != refreshed_account_id {
        return Err(CatalogError::Auth(
            "authenticated account changed during model catalog refresh".to_string(),
        ));
    }
    Ok(())
}

fn catalog_error_from_refresh(error: AuthError) -> CatalogError {
    match error {
        AuthError::RefreshUpstreamHttp { status, message } => {
            CatalogError::RefreshUpstreamHttp { status, message }
        }
        AuthError::RefreshTransport(message) => CatalogError::RefreshTransport(message),
        AuthError::RefreshProtocol(message) => CatalogError::RefreshProtocol(message),
        AuthError::Internal(message) => CatalogError::Internal(message),
        error @ AuthError::WriteCleanup { .. } => CatalogError::Internal(error.to_string()),
        error => CatalogError::Auth(error.to_string()),
    }
}

fn redact_failure_event(value: &mut Value, secrets: &[&str]) {
    match value {
        Value::String(text) => *text = auth::redact_text(text, secrets),
        Value::Array(items) => {
            for item in items {
                redact_failure_event(item, secrets);
            }
        }
        Value::Object(fields) => {
            for item in fields.values_mut() {
                redact_failure_event(item, secrets);
            }
        }
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn redact_upstream_failure_event(mut event: Value, secrets: &[&str]) -> Value {
    if matches!(
        event.get("type").and_then(Value::as_str),
        Some("error" | "response.failed" | "response.incomplete")
    ) {
        redact_failure_event(&mut event, secrets);
    }
    event
}

fn optional_header_text<'a>(
    headers: &'a reqwest::header::HeaderMap,
    name: &str,
) -> Option<&'a str> {
    headers
        .get(name)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
}

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

fn validate_client_metadata_reserved_keys(
    client_metadata: Option<&HashMap<String, String>>,
) -> Result<(), ProviderError> {
    let Some(metadata) = client_metadata else {
        return Ok(());
    };
    for key in [SESSION_ID_KEY, THREAD_ID_KEY] {
        if metadata
            .get(key)
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(ProviderError::InvalidRequest(format!(
                "client_metadata.{key} must be a non-empty string when provided"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod live_catalog_tests {
    use super::*;

    fn model_with_levels(levels: &[&str], multi_agent: Option<&str>) -> Arc<ModelInfo> {
        let default_reasoning = levels.iter().copied().find(|effort| *effort != "ultra");
        let levels: Vec<Value> = levels
            .iter()
            .map(|effort| json!({"effort": effort, "description": effort}))
            .collect();
        let body = serde_json::to_vec(&json!({
            "models": [{
                "slug": "live-model",
                "display_name": "Live Model",
                "description": "test",
                "default_reasoning_level": default_reasoning,
                "supported_reasoning_levels": levels,
                "multi_agent_reasoning_effort": multi_agent,
                "visibility": "list",
                "supported_in_api": true,
                "priority": 1,
                "service_tiers": [],
                "default_service_tier": null,
                "support_verbosity": true,
                "default_verbosity": "medium",
                "supports_image_detail_original": true,
                "context_window": 100000,
                "max_context_window": 120000,
                "auto_compact_token_limit": null,
                "input_modalities": ["text", "image"],
                "use_responses_lite": false
            }]
        }))
        .unwrap();
        crate::model_catalog::parse_models_response(&body).unwrap()[0].clone()
    }

    #[test]
    fn response_history_is_scoped_by_account() {
        let store = ResponseChainStore::new(4);
        store
            .commit("account-a", "response-1", &[json!({"a": 1})], &[])
            .unwrap();
        assert!(store.resolve("account-a", "response-1").is_ok());
        assert!(store.resolve("account-b", "response-1").is_err());
    }

    #[test]
    fn ultra_uses_live_multi_agent_then_max_then_last_supported() {
        let explicit = model_with_levels(&["low", "max", "xhigh"], Some("xhigh"));
        assert_eq!(
            resolve_reasoning_effort(&explicit, "ultra").unwrap(),
            "xhigh"
        );
        let max = model_with_levels(&["low", "max"], None);
        assert_eq!(resolve_reasoning_effort(&max, "ultra").unwrap(), "max");
        let last = model_with_levels(&["low", "high"], None);
        assert_eq!(resolve_reasoning_effort(&last, "ultra").unwrap(), "high");
        let none = model_with_levels(&["ultra"], None);
        assert!(resolve_reasoning_effort(&none, "ultra").is_err());
        let self_mapping = model_with_levels(&["ultra"], Some("ultra"));
        assert!(resolve_reasoning_effort(&self_mapping, "ultra").is_err());
    }

    #[test]
    fn persistent_reasoning_maps_to_disabled() {
        let model = model_with_levels(&["persistent"], None);
        assert_eq!(
            resolve_reasoning_effort(&model, "persistent").unwrap(),
            "disabled"
        );
    }

    #[test]
    fn malformed_upstream_tool_image_and_web_items_fail() {
        assert!(tool_call_from_response_item(&json!({
            "type": "function_call",
            "name": "tool",
            "arguments": "{}"
        }))
        .is_err());
        assert!(tool_call_from_response_item(&json!({
            "type": "function_call",
            "name": "tool",
            "id": "item-id",
            "arguments": {}
        }))
        .is_err());
        assert!(tool_call_from_response_item(&json!({
            "type": "custom_tool_call",
            "name": "tool",
            "call_id": "call-id",
            "arguments": "{}"
        }))
        .is_err());
        assert!(image_generation_from_item(&json!({
            "type": "image_generation_call",
            "result": "data"
        }))
        .is_err());
        assert!(web_search_event_from_response_item(
            &json!({"type": "web_search_call", "id": "call", "action": {}}),
        )
        .is_err());
    }

    #[test]
    fn image_generation_optional_fields_are_validated_without_fabrication() {
        let missing = image_generation_from_item(&json!({
            "type": "image_generation_call",
            "status": "completed",
            "result": "data"
        }))
        .unwrap()
        .unwrap();
        assert_eq!(missing, json!({"status": "completed", "result": "data"}));

        let null = image_generation_from_item(&json!({
            "type": "image_generation_call",
            "id": null,
            "status": "completed",
            "result": "data",
            "revised_prompt": null
        }))
        .unwrap()
        .unwrap();
        assert_eq!(null, json!({"status": "completed", "result": "data"}));

        let present = image_generation_from_item(&json!({
            "type": "image_generation_call",
            "id": "image-1",
            "status": "completed",
            "result": "data",
            "revised_prompt": ""
        }))
        .unwrap()
        .unwrap();
        assert_eq!(present["id"], "image-1");
        assert_eq!(present["revised_prompt"], "");

        let empty = image_generation_from_item(&json!({
            "type": "image_generation_call",
            "id": "",
            "status": "",
            "result": ""
        }))
        .unwrap()
        .unwrap();
        assert_eq!(empty, json!({"id": "", "status": "", "result": ""}));

        for invalid_id in [json!(1), json!(false)] {
            assert!(image_generation_from_item(&json!({
                "type": "image_generation_call",
                "id": invalid_id,
                "status": "completed",
                "result": "data"
            }))
            .is_err());
        }
        assert!(image_generation_from_item(&json!({
            "type": "image_generation_call",
            "status": "completed",
            "result": "data",
            "revised_prompt": 1
        }))
        .is_err());
    }

    #[test]
    fn response_item_optional_fields_follow_pinned_nullable_and_additive_contract() {
        let valid = [
            json!({
                "type": "message",
                "id": "",
                "role": "assistant",
                "content": [],
                "phase": "final_answer",
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": null,
                    "create_time": 1.5,
                    "content_item_kinds": {"future": true}
                },
                "future": true
            }),
            json!({
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "text", "text": "raw", "future": true}],
                "encrypted_content": null,
                "future": true
            }),
            json!({
                "type": "function_call",
                "name": "",
                "call_id": "",
                "arguments": "not-json",
                "namespace": null,
                "encrypted_function_args": ["a", ""],
                "future": true
            }),
            json!({
                "type": "web_search_call",
                "id": "",
                "status": null,
                "action": {"type": "search", "query": "q", "sources": []},
                "future": true
            }),
            json!({
                "type": "image_generation_call",
                "id": "",
                "status": "",
                "result": "",
                "revised_prompt": null,
                "future": true
            }),
        ];
        for item in &valid {
            validate_added_response_output_item(item).unwrap();
        }

        let invalid = [
            json!({"type": "message", "id": 1, "role": "assistant", "content": []}),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [],
                "internal_chat_message_metadata_passthrough": []
            }),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [],
                "internal_chat_message_metadata_passthrough": {"turn_id": 1}
            }),
            json!({
                "type": "message",
                "role": "assistant",
                "content": [],
                "internal_chat_message_metadata_passthrough": {"create_time": "now"}
            }),
            json!({"type": "message", "role": "assistant", "content": [], "phase": "future"}),
            json!({
                "type": "function_call",
                "name": "f",
                "call_id": "c",
                "arguments": "{}",
                "namespace": 1
            }),
            json!({
                "type": "function_call",
                "name": "f",
                "call_id": "c",
                "arguments": "{}",
                "encrypted_function_args": [1]
            }),
            json!({"type": "custom_tool_call", "name": "f", "call_id": "c", "input": "", "status": 1}),
            json!({"type": "web_search_call", "status": 1}),
            json!({"type": "image_generation_call", "status": "", "result": "", "revised_prompt": 1}),
        ];
        for item in &invalid {
            assert!(validate_added_response_output_item(item).is_err());
        }
    }

    #[test]
    fn image_generation_and_inspection_reject_endpoint_incompatible_items() {
        let image = json!({
            "type": "image_generation_call",
            "status": "completed",
            "result": "encoded-image"
        });
        let message = json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "visible"}]
        });
        let function = json!({
            "type": "function_call",
            "name": "lookup",
            "call_id": "call-1",
            "arguments": "{}"
        });
        let web = json!({
            "type": "web_search_call",
            "id": "search-1",
            "action": {"type": "search", "query": "q", "sources": []}
        });

        for unexpected in [&message, &function, &web] {
            assert!(matches!(
                image_generations_from_response_items(&[image.clone(), unexpected.clone()]),
                Err(ProviderError::UpstreamProtocol(_))
            ));
        }
        for unexpected in [&function, &image, &web] {
            assert!(matches!(
                inspection_text_from_response_items(&[message.clone(), unexpected.clone()]),
                Err(ProviderError::UpstreamProtocol(_))
            ));
        }
        assert_eq!(
            inspection_text_from_response_items(&[
                json!({"type": "reasoning", "summary": [], "content": []}),
                message
            ])
            .unwrap(),
            "visible"
        );
    }

    #[test]
    fn unknown_upstream_event_types_are_ignored() {
        assert!(validate_response_event(&json!({"type": "response.future_event"})).is_ok());
        assert!(validate_response_event(&json!({"type": ""})).is_ok());

        let mut block = vec!["data: {\"type\":\"response.future_telemetry\"}".to_string()];
        let mut delivered = 0;
        assert!(!emit_sse_block(&mut block, &[], &mut |_event| {
            delivered += 1;
            Ok(())
        })
        .unwrap());
        assert_eq!(delivered, 0);
    }

    #[test]
    fn known_unsupported_semantic_event_types_fail_immediately() {
        for event_type in [
            "response.file_search_call.in_progress",
            "response.file_search_call.searching",
            "response.file_search_call.completed",
            "response.code_interpreter_call.in_progress",
            "response.code_interpreter_call.interpreting",
            "response.code_interpreter_call_code.delta",
            "response.code_interpreter_call_code.done",
            "response.code_interpreter_call.completed",
            "response.mcp_call.in_progress",
            "response.mcp_call_arguments.delta",
            "response.mcp_call_arguments.done",
            "response.mcp_call.completed",
            "response.mcp_call.failed",
            "response.mcp_list_tools.in_progress",
            "response.mcp_list_tools.completed",
            "response.mcp_list_tools.failed",
            "response.shell_call_command.added",
            "response.shell_call_command.delta",
            "response.shell_call_command.done",
            "response.shell_call_output_content.delta",
            "response.shell_call_output_content.done",
            "response.audio.delta",
            "response.audio.done",
            "response.audio.transcript.delta",
            "response.audio.transcript.done",
            "response.refusal.delta",
            "response.refusal.done",
            "response.output_text.annotation.added",
            "response.custom_tool_call_input.delta",
            "response.custom_tool_call_input.done",
        ] {
            assert!(matches!(
                validate_response_event(&json!({"type": event_type})),
                Err(ProviderError::UpstreamProtocol(_))
            ));
        }
    }

    #[test]
    fn content_part_events_accept_output_text_and_reject_unrepresentable_parts() {
        for event_type in ["response.content_part.added", "response.content_part.done"] {
            assert!(validate_response_event(&json!({
                "type": event_type,
                "part": {
                    "type": "output_text",
                    "text": "",
                    "annotations": [],
                    "logprobs": []
                }
            }))
            .is_ok());

            for part in [
                Value::Null,
                json!({"type": "refusal", "refusal": "blocked"}),
                json!({"type": "future_content", "value": "opaque"}),
                json!({"type": "output_text"}),
                json!({"type": "output_text", "text": "ok", "annotations": {}}),
                json!({"type": "output_text", "text": "ok", "logprobs": {}}),
            ] {
                assert!(matches!(
                    validate_response_event(&json!({"type": event_type, "part": part})),
                    Err(ProviderError::UpstreamProtocol(_))
                ));
            }
        }
    }

    #[test]
    fn unknown_added_semantic_items_fail_immediately() {
        for item_type in [
            "tool_search_call",
            "computer_call",
            "file_search_call",
            "code_interpreter_call",
            "mcp_call",
            "local_shell_call",
        ] {
            assert!(matches!(
                validate_response_event(&json!({
                    "type": "response.output_item.added",
                    "item": {"type": item_type}
                })),
                Err(ProviderError::UpstreamProtocol(_))
            ));
        }
    }

    #[test]
    fn openai_function_tool_strict_mode_is_forwarded() {
        let tool = ToolSchema {
            name: "lookup".to_string(),
            description: Some("Lookup".to_string()),
            parameters: json!({"type": "object", "properties": {}}),
            strict: true,
        };
        assert_eq!(tool_schema_to_response_dict(&tool).unwrap()["strict"], true);
    }

    #[test]
    fn web_search_event_preserves_the_upstream_id_and_query() {
        let event = web_search_event_from_response_item(&json!({
            "type": "web_search_call",
            "id": "raw-web.id-1",
            "action": {"type": "search", "query": "live query", "sources": []}
        }))
        .unwrap()
        .unwrap();
        assert_eq!(event["id"], "raw-web.id-1");
        assert_eq!(event["input"]["query"], "live query");
    }

    #[test]
    fn web_search_event_validates_queries_and_preserves_duplicate_sources() {
        let event = web_search_event_from_response_item(&json!({
            "type": "web_search_call",
            "id": "raw-web.id-2",
            "action": {
                "type": "search",
                "query": "preferred query",
                "queries": ["preferred query"],
                "sources": [
                    {"url": "https://example.test", "title": "", "page_age": null},
                    {"url": "https://example.test", "title": "", "page_age": null}
                ]
            }
        }))
        .unwrap()
        .unwrap();
        assert_eq!(event["input"]["query"], "preferred query");
        let content = event["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0], content[1]);
        assert_eq!(content[0]["title"], "");
        assert!(content[0].get("page_age").is_none());

        assert!(matches!(
            web_search_event_from_response_item(&json!({
                "type": "web_search_call",
                "id": "raw-web.id-3",
                "action": {"type": "search", "query": "query", "sources": null}
            })),
            Err(ProviderError::UpstreamProtocol(_))
        ));

        for (action, expected) in [
            (
                json!({"query": "direct", "queries": null, "sources": []}),
                "direct",
            ),
            (
                json!({"query": null, "queries": ["fallback"], "sources": []}),
                "fallback",
            ),
            (json!({"query": "", "sources": []}), ""),
            (
                json!({"query": "same", "queries": ["same"], "sources": []}),
                "same",
            ),
        ] {
            let event = web_search_event_from_response_item(&json!({
                "type": "web_search_call",
                "id": "",
                "action": {
                    "type": "search",
                    "query": action.get("query"),
                    "queries": action.get("queries"),
                    "sources": action.get("sources"),
                },
            }))
            .unwrap()
            .unwrap();
            assert_eq!(event["id"], "");
            assert_eq!(event["input"]["query"], expected);
        }

        for id in [None, Some(Value::Null)] {
            let mut item = json!({
                "type": "web_search_call",
                "action": {"type": "search", "query": "q", "sources": []},
            });
            if let Some(id) = id {
                item["id"] = id;
            }
            assert!(matches!(
                web_search_event_from_response_item(&item),
                Err(ProviderError::UpstreamProtocol(_))
            ));
        }

        for action in [
            json!({"query": "query", "sources": []}),
            json!({"type": "search", "queries": ["first", "second"], "sources": []}),
            json!({"type": "search", "query": "first", "queries": ["second"], "sources": []}),
            json!({"type": "search", "sources": []}),
            json!({"type": "search", "queries": [], "sources": []}),
            json!({"type": "open_page", "url": "https://example.test", "sources": []}),
            json!({"type": "find_in_page", "pattern": "needle", "sources": []}),
            json!({"type": "future_action", "sources": []}),
        ] {
            assert!(matches!(
                web_search_event_from_response_item(&json!({
                    "type": "web_search_call",
                    "id": "raw-web.id-invalid",
                    "action": action,
                })),
                Err(ProviderError::UpstreamProtocol(_))
            ));
        }
    }

    #[test]
    fn completed_event_ignores_additive_output_without_done_items() {
        let event = json!({
            "type": "response.completed",
            "response": {
                "id": "resp",
                "output": [{"type": "message", "role": "assistant", "content": []}]
            }
        });
        assert!(validate_response_event(&event).is_ok());
    }

    #[test]
    fn provider_timeout_override_must_be_positive() {
        assert!(matches!(
            ChatGPTOAuthProvider::new(
                String::new(),
                "https://example.test/backend-api/codex".to_string(),
                None,
                Some(std::time::Duration::ZERO),
            ),
            Err(ProviderError::InvalidRequest(_))
        ));
    }
}

fn pinned_codex_compatibility_version() -> &'static str {
    CODEX_COMPATIBILITY_VERSION.get_or_init(|| {
        let contract: Value = serde_json::from_str(CODEX_UPSTREAM_CONTRACT_JSON)
            .expect("embedded Codex upstream contract must be valid JSON");
        let version = contract
            .get("upstream")
            .and_then(|value| value.get("version"))
            .and_then(Value::as_str)
            .expect("embedded Codex upstream contract must contain upstream.version");
        normalize_codex_cli_version(version)
            .expect("embedded Codex upstream contract version must be a semantic version")
    })
}

fn normalize_codex_cli_version(value: &str) -> Option<String> {
    if value != value.trim() {
        return None;
    }
    let version = value;
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
    let version = pinned_codex_compatibility_version();
    let mut headers = HashMap::new();
    headers.insert("originator".to_string(), CODEX_CLI_ORIGINATOR.to_string());
    headers.insert(
        "User-Agent".to_string(),
        sanitize_header_value(&format!(
            "{CODEX_CLI_ORIGINATOR}/{version} ({}) codex-as-api/{}",
            codex_os_info(),
            env!("CARGO_PKG_VERSION")
        )),
    );
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
    Auth(AuthError),
    #[error("{0}")]
    InvalidRequest(String),
    #[error("{0}")]
    Request(String),
    #[error("{message}")]
    UpstreamHttp { status: u16, message: String },
    #[error("{0}")]
    UpstreamTransport(String),
    #[error("{0}")]
    CatalogUnavailable(String),
    #[error("requested model is not present in the authenticated model catalog")]
    ModelNotFound(String),
    #[error("{0}")]
    UpstreamProtocol(String),
}

impl From<AuthError> for ProviderError {
    fn from(error: AuthError) -> Self {
        match error {
            AuthError::RefreshUpstreamHttp { status, message } => {
                Self::UpstreamHttp { status, message }
            }
            AuthError::RefreshTransport(message) => Self::UpstreamTransport(message),
            AuthError::RefreshProtocol(message) => Self::UpstreamProtocol(message),
            AuthError::Internal(message) => Self::Request(message),
            error @ AuthError::WriteCleanup { .. } => Self::Request(error.to_string()),
            error => Self::Auth(error),
        }
    }
}

pub struct ResolvedModel {
    pub snapshot: Arc<ModelCatalogSnapshot>,
    pub model: Arc<ModelInfo>,
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
    account_id: String,
    comp_hash: Option<String>,
}

pub struct PreparedChatStream {
    request: FinalizedResponsesRequest,
    extra_headers: HashMap<String, String>,
    cancellation: Option<Arc<AtomicBool>>,
}

impl PreparedChatStream {
    pub fn cancel_when(mut self, cancellation: Arc<AtomicBool>) -> Self {
        self.cancellation = Some(cancellation);
        self
    }
}

#[derive(Default)]
struct ChatStreamState {
    final_output: Vec<Value>,
    tool_call_ids: HashSet<String>,
    text_parts: Vec<String>,
    reasoning_summary_parts: Vec<String>,
    reasoning_raw_parts: Vec<String>,
    emitted_tool_call: bool,
    emitted_web_search: bool,
    saw_text_delta: bool,
    saw_reasoning_summary_delta: bool,
    saw_reasoning_raw_delta: bool,
    saw_completed: bool,
    completed_response_id: Option<String>,
    pending_finish: Option<Value>,
}

#[derive(Default)]
struct ResponseChainStoreInner {
    chains: HashMap<(String, String), ResponseChain>,
    lru: VecDeque<(String, String)>,
}

#[derive(Clone)]
struct ResponseChain {
    history: Vec<Value>,
    comp_hash: Option<String>,
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

    #[cfg(test)]
    fn resolve(&self, account_id: &str, response_id: &str) -> Result<Vec<Value>, ProviderError> {
        self.resolve_for_model(account_id, response_id, None)
    }

    fn resolve_for_model(
        &self,
        account_id: &str,
        response_id: &str,
        current_comp_hash: Option<&str>,
    ) -> Result<Vec<Value>, ProviderError> {
        if response_id.trim().is_empty() {
            return Err(ProviderError::InvalidRequest(
                "previous_response_id must be a non-empty string".to_string(),
            ));
        }

        let mut inner = self.inner.lock().map_err(|_| {
            ProviderError::Request("response chain store lock is poisoned".to_string())
        })?;
        let key = (account_id.to_string(), response_id.to_string());
        let chain = inner.chains.get(&key).cloned().ok_or_else(|| {
            ProviderError::InvalidRequest(
                "previous_response_id is unknown or has been evicted".to_string(),
            )
        })?;
        if chain.comp_hash.as_deref().is_some()
            && current_comp_hash.is_some()
            && chain.comp_hash.as_deref() != current_comp_hash
        {
            return Err(ProviderError::InvalidRequest(
                "previous_response_id requires compaction because the model compatibility hash changed"
                    .to_string(),
            ));
        }
        inner.lru.retain(|candidate| candidate != &key);
        inner.lru.push_back(key);
        Ok(chain.history)
    }

    #[cfg(test)]
    fn commit(
        &self,
        account_id: &str,
        response_id: &str,
        conversation_input: &[Value],
        response_output: &[Value],
    ) -> Result<(), ProviderError> {
        self.commit_for_model(
            account_id,
            response_id,
            conversation_input,
            response_output,
            None,
        )
    }

    fn commit_for_model(
        &self,
        account_id: &str,
        response_id: &str,
        conversation_input: &[Value],
        response_output: &[Value],
        comp_hash: Option<&str>,
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
        let key = (account_id.to_string(), response_id.to_string());
        inner.lru.retain(|candidate| candidate != &key);
        inner.chains.insert(
            key.clone(),
            ResponseChain {
                history,
                comp_hash: comp_hash.map(str::to_string),
            },
        );
        inner.lru.push_back(key);
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
    pub timeout: std::time::Duration,
    response_chains: ResponseChainStore,
    model_catalog: ModelCatalogCache,
}

impl ChatGPTOAuthProvider {
    pub fn new(
        model: String,
        base_url: String,
        auth_json_path: Option<String>,
        timeout: Option<std::time::Duration>,
    ) -> Result<Self, ProviderError> {
        Self::with_catalog_ttl(
            model,
            base_url,
            auth_json_path,
            timeout,
            DEFAULT_CATALOG_TTL,
        )
    }

    fn with_catalog_ttl(
        model: String,
        base_url: String,
        auth_json_path: Option<String>,
        timeout: Option<std::time::Duration>,
        catalog_ttl: std::time::Duration,
    ) -> Result<Self, ProviderError> {
        let timeout = timeout.unwrap_or(CHATGPT_OAUTH_DEFAULT_TIMEOUT);
        if timeout.is_zero() {
            return Err(ProviderError::InvalidRequest(
                "provider timeout must be greater than zero".to_string(),
            ));
        }
        if catalog_ttl.is_zero() {
            return Err(ProviderError::InvalidRequest(
                "model catalog TTL must be greater than zero".to_string(),
            ));
        }
        if !model.is_empty() && model != model.trim() {
            return Err(ProviderError::InvalidRequest(
                "model must not contain surrounding whitespace".to_string(),
            ));
        }
        if base_url != base_url.trim() {
            return Err(ProviderError::InvalidRequest(
                "provider base URL must not contain surrounding whitespace".to_string(),
            ));
        }
        if base_url
            .chars()
            .any(|character| character.is_whitespace() || character.is_control())
        {
            return Err(ProviderError::InvalidRequest(
                "provider base URL must not contain whitespace or control characters".to_string(),
            ));
        }
        if has_malformed_percent_escape(&base_url) {
            return Err(ProviderError::InvalidRequest(
                "provider base URL contains a malformed percent escape".to_string(),
            ));
        }
        let base_url = base_url.trim_end_matches('/').to_string();
        let has_raw_authority = base_url
            .split_once("://")
            .is_some_and(|(_, rest)| !rest.split('/').next().unwrap_or("").is_empty());
        let parsed_base_url = reqwest::Url::parse(&base_url).map_err(|_| {
            ProviderError::InvalidRequest("provider base URL is invalid".to_string())
        })?;
        if !matches!(parsed_base_url.scheme(), "http" | "https")
            || parsed_base_url.host_str().is_none()
            || !has_raw_authority
        {
            return Err(ProviderError::InvalidRequest(
                "provider base URL must be an absolute HTTP(S) URL with a hostname".to_string(),
            ));
        }
        if !parsed_base_url.username().is_empty()
            || parsed_base_url.password().is_some()
            || parsed_base_url.query().is_some()
            || parsed_base_url.fragment().is_some()
        {
            return Err(ProviderError::InvalidRequest(
                "provider base URL must not contain credentials, query, or fragment".to_string(),
            ));
        }
        Ok(Self {
            model,
            base_url,
            auth_json_path,
            timeout,
            response_chains: ResponseChainStore::new(RESPONSE_CHAIN_CAPACITY),
            model_catalog: ModelCatalogCache::new(catalog_ttl),
        })
    }

    #[cfg(test)]
    pub fn new_with_catalog_ttl(
        model: String,
        base_url: String,
        auth_json_path: Option<String>,
        timeout: Option<std::time::Duration>,
        catalog_ttl: std::time::Duration,
    ) -> Result<Self, ProviderError> {
        Self::with_catalog_ttl(model, base_url, auth_json_path, timeout, catalog_ttl)
    }

    pub fn model_catalog_snapshot(&self) -> Result<Arc<ModelCatalogSnapshot>, ProviderError> {
        let token = auth::token_for_request(self.auth_json_path.as_deref())?;
        let client_version = pinned_codex_compatibility_version().to_string();
        let key = CatalogKey {
            account_id: token.account_id.clone(),
            base_url: self.base_url.clone(),
            client_version: client_version.clone(),
        };
        self.model_catalog
            .snapshot(key, || self.fetch_models(token, &client_version))
            .map_err(|error| match error {
                CatalogError::Invalid(message) => ProviderError::CatalogUnavailable(format!(
                    "model catalog response is invalid: {message}"
                )),
                CatalogError::Auth(message) => ProviderError::Auth(AuthError::OAuth(message)),
                CatalogError::RefreshUpstreamHttp { status, message } => {
                    ProviderError::UpstreamHttp { status, message }
                }
                CatalogError::RefreshTransport(message) => {
                    ProviderError::UpstreamTransport(message)
                }
                CatalogError::RefreshProtocol(message) => ProviderError::UpstreamProtocol(message),
                CatalogError::Internal(message) => ProviderError::Request(message),
                CatalogError::UpstreamHttp { status, message } => {
                    ProviderError::UpstreamHttp { status, message }
                }
                CatalogError::Request(message) => ProviderError::CatalogUnavailable(format!(
                    "model catalog request failed: {message}"
                )),
            })
    }

    pub fn resolve_model(&self, requested: Option<&str>) -> Result<ResolvedModel, ProviderError> {
        if requested
            .is_some_and(|requested| requested.trim().is_empty() || requested != requested.trim())
        {
            return Err(ProviderError::InvalidRequest(
                "model must be a non-empty string".to_string(),
            ));
        }
        let snapshot = self.model_catalog_snapshot()?;
        Self::resolve_model_from_snapshot(snapshot, requested)
    }

    fn resolve_model_from_snapshot(
        snapshot: Arc<ModelCatalogSnapshot>,
        requested: Option<&str>,
    ) -> Result<ResolvedModel, ProviderError> {
        let model = match requested {
            Some(requested) => snapshot.model(requested),
            None => snapshot.default_model(),
        }
        .ok_or_else(|| {
            requested.map_or_else(
                || {
                    ProviderError::CatalogUnavailable(
                        "upstream model catalog has no model with list visibility for default selection"
                            .to_string(),
                    )
                },
                |requested| ProviderError::ModelNotFound(requested.to_string()),
            )
        })?;
        if requested.is_none() && (model.slug.trim().is_empty() || model.slug != model.slug.trim())
        {
            return Err(ProviderError::CatalogUnavailable(
                "default model publishes an unusable slug".to_string(),
            ));
        }
        Ok(ResolvedModel { snapshot, model })
    }

    pub(crate) fn configured_or_default_model_from_snapshot(
        &self,
        snapshot: Arc<ModelCatalogSnapshot>,
    ) -> Result<ResolvedModel, ProviderError> {
        if !self.model.is_empty()
            && (self.model.trim().is_empty() || self.model != self.model.trim())
        {
            return Err(ProviderError::CatalogUnavailable(
                "configured model is unusable".to_string(),
            ));
        }
        let configured = (!self.model.is_empty()).then_some(self.model.as_str());
        Self::resolve_model_from_snapshot(snapshot, configured).map_err(|error| match error {
            ProviderError::ModelNotFound(_) if configured.is_some() => {
                ProviderError::CatalogUnavailable(
                    "configured model is unavailable in the authenticated catalog".to_string(),
                )
            }
            error => error,
        })
    }

    pub fn configured_or_default_model(&self) -> Result<ResolvedModel, ProviderError> {
        let snapshot = self.model_catalog_snapshot()?;
        self.configured_or_default_model_from_snapshot(snapshot)
    }

    fn fetch_models(
        &self,
        initial_token: ChatGPTTokenData,
        client_version: &str,
    ) -> Result<(Vec<u8>, Option<String>), CatalogError> {
        let initial_account_id = initial_token.account_id.clone();
        let mut token = initial_token;
        for attempt in 0..2 {
            let headers = headers_for_token(&token)
                .map_err(|error| CatalogError::Request(error.to_string()))?;
            let url = format!("{}/models?client_version={client_version}", self.base_url);
            let client = reqwest::blocking::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|error| CatalogError::Request(error.to_string()))?;
            let mut builder = client.get(url);
            for (name, value) in &headers {
                builder = builder.header(name.as_str(), value.as_str());
            }
            builder = builder.header(reqwest::header::ACCEPT, "application/json");
            builder = builder.timeout(MODEL_CATALOG_TIMEOUT);
            let response = builder.send().map_err(|error| {
                CatalogError::Request(auth::redact_text(
                    &error.to_string(),
                    &[
                        token.access_token.as_str(),
                        token.refresh_token.as_str(),
                        token.id_token.as_str(),
                        token.account_id.as_str(),
                    ],
                ))
            })?;
            let status = response.status();
            if status == reqwest::StatusCode::UNAUTHORIZED && attempt == 0 {
                token =
                    auth::refresh_after_unauthorized(&token).map_err(catalog_error_from_refresh)?;
                ensure_catalog_account_unchanged(&initial_account_id, &token.account_id)?;
                continue;
            }
            if status == reqwest::StatusCode::UNAUTHORIZED {
                let body = response
                    .text()
                    .map_err(|error| CatalogError::UpstreamHttp {
                        status: 401,
                        message: format!("HTTP 401 and failed to read error body: {error}"),
                    })?;
                return Err(CatalogError::UpstreamHttp {
                    status: 401,
                    message: format!(
                        "ChatGPT OAuth upstream returned HTTP 401 after one refresh: {}",
                        auth::redact_text(
                            &body,
                            &[
                                token.access_token.as_str(),
                                token.refresh_token.as_str(),
                                token.id_token.as_str(),
                                token.account_id.as_str(),
                            ],
                        )
                    ),
                });
            }
            if !status.is_success() {
                let body = response.text().map_err(|error| {
                    CatalogError::UpstreamHttp {
                        status: status.as_u16(),
                        message: format!(
                            "ChatGPT OAuth model catalog request failed: HTTP {} and could not read the error body: {error}",
                            status.as_u16()
                        ),
                    }
                })?;
                return Err(CatalogError::UpstreamHttp {
                    status: status.as_u16(),
                    message: format!(
                        "ChatGPT OAuth model catalog request failed: HTTP {}: {}",
                        status.as_u16(),
                        auth::redact_text(
                            &body,
                            &[
                                token.access_token.as_str(),
                                token.refresh_token.as_str(),
                                token.id_token.as_str(),
                                token.account_id.as_str(),
                            ],
                        )
                    ),
                });
            }
            let etag = optional_header_text(response.headers(), "etag").map(str::to_string);
            let bytes = response
                .bytes()
                .map_err(|error| CatalogError::Request(error.to_string()))?;
            return Ok((bytes.to_vec(), etag));
        }
        Err(CatalogError::Request(
            "authentication retry did not complete".to_string(),
        ))
    }

    pub fn chat_prepared(
        &self,
        prepared: PreparedChatStream,
    ) -> Result<AssistantResponse, ProviderError> {
        let mut content_parts: Vec<String> = Vec::new();
        let mut reasoning_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut finish_reason: Option<String> = None;
        let mut saw_finish = false;
        let mut raw_events: Vec<Value> = Vec::new();
        let mut usage: Option<Usage> = None;
        let mut response_id: Option<String> = None;
        let mut tool_call_ids: HashSet<String> = HashSet::new();

        let mut events = Vec::new();
        self.stream_prepared_chat(prepared, |event| {
            events.push(event);
            Ok(())
        })?;

        for event in events {
            raw_events.push(event.clone());
            let typ = event.get("type").and_then(Value::as_str).ok_or_else(|| {
                ProviderError::UpstreamProtocol(
                    "normalized response event requires a string type".to_string(),
                )
            })?;
            match typ {
                "content" => {
                    let text = event.get("text").and_then(Value::as_str).ok_or_else(|| {
                        ProviderError::UpstreamProtocol(
                            "content event requires a string text".to_string(),
                        )
                    })?;
                    content_parts.push(text.to_string());
                }
                "reasoning_delta" | "reasoning_raw_delta" => {
                    let text = event.get("text").and_then(Value::as_str).ok_or_else(|| {
                        ProviderError::UpstreamProtocol(
                            "reasoning event requires a string text".to_string(),
                        )
                    })?;
                    reasoning_parts.push(text.to_string());
                }
                "tool_call" => {
                    let id = event
                        .get("id")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            ProviderError::UpstreamProtocol(
                                "tool_call event requires a string id".to_string(),
                            )
                        })?
                        .to_string();
                    let name = event
                        .get("name")
                        .and_then(Value::as_str)
                        .ok_or_else(|| {
                            ProviderError::UpstreamProtocol(
                                "tool_call event requires a string name".to_string(),
                            )
                        })?
                        .to_string();
                    let arguments = event
                        .get("arguments")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .ok_or_else(|| {
                            ProviderError::UpstreamProtocol(
                                "tool_call event requires string arguments".to_string(),
                            )
                        })?;
                    if !tool_call_ids.insert(id.clone()) {
                        return Err(ProviderError::UpstreamProtocol(format!(
                            "provider response contains duplicate call_id {id:?}"
                        )));
                    }
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
                "finish" => {
                    if saw_finish {
                        return Err(ProviderError::UpstreamProtocol(
                            "normalized response contains more than one finish event".to_string(),
                        ));
                    }
                    saw_finish = true;
                    finish_reason = match event.get("finish_reason") {
                        None | Some(Value::Null) => None,
                        Some(Value::String(value)) if !value.is_empty() => Some(value.clone()),
                        Some(_) => {
                            return Err(ProviderError::UpstreamProtocol(
                                "finish event finish_reason must be non-empty or null".to_string(),
                            ));
                        }
                    };
                    match event.get("reasoning_content") {
                        None | Some(Value::Null) => {}
                        Some(Value::String(reasoning)) => {
                            reasoning_parts = vec![reasoning.clone()];
                        }
                        Some(_) => {
                            return Err(ProviderError::UpstreamProtocol(
                                "finish event reasoning_content must be a string or null"
                                    .to_string(),
                            ));
                        }
                    }
                    usage = match event.get("usage") {
                        None | Some(Value::Null) => None,
                        Some(value) => Some(parse_usage(value)?),
                    };
                    response_id = Some(
                        event
                            .get("response_id")
                            .and_then(Value::as_str)
                            .filter(|value| !value.is_empty())
                            .ok_or_else(|| {
                                ProviderError::UpstreamProtocol(
                                    "finish event requires a non-empty response_id".to_string(),
                                )
                            })?
                            .to_string(),
                    );
                }
                "web_search_call" | "reasoning_section_break" => {}
                _ => {
                    return Err(ProviderError::UpstreamProtocol(format!(
                        "unsupported normalized response event type {typ:?}"
                    )));
                }
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
        if !saw_finish {
            return Err(ProviderError::UpstreamProtocol(
                "normalized response ended before a finish event".to_string(),
            ));
        }
        let response_id = response_id.ok_or_else(|| {
            ProviderError::UpstreamProtocol(
                "normalized response finish event did not provide a response id".to_string(),
            )
        })?;

        Ok(AssistantResponse {
            content: content_parts.join(""),
            tool_calls,
            finish_reason,
            usage,
            reasoning_content,
            raw: Some(json!({"events": tail_events})),
            response_id: Some(response_id),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare_chat_stream_for_resolved_model_with_controls(
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
        resolved: ResolvedModel,
        tool_choice: Option<&Value>,
        service_tier: Option<&str>,
        text: Option<&Value>,
        client_metadata: Option<&HashMap<String, String>>,
        codex_metadata: Option<bool>,
        responses_lite: Option<&Value>,
        parallel_tool_calls: Option<bool>,
        controls: &GenerationControls,
    ) -> Result<PreparedChatStream, ProviderError> {
        validate_private_request_controls(temperature, max_tokens, stop)?;
        let request = self.responses_payload_with_controls(
            messages,
            tools,
            reasoning_effort,
            stop,
            prompt_cache_key,
            max_tokens,
            previous_response_id,
            &resolved,
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
            if sa.is_empty() || !sa.bytes().all(|byte| (0x21..=0x7e).contains(&byte)) {
                return Err(ProviderError::InvalidRequest(
                    "subagent must contain only visible ASCII characters without spaces"
                        .to_string(),
                ));
            }
            extra_headers.insert("x-openai-subagent".to_string(), sa.to_string());
        }
        if let Some(mg) = memgen_request {
            extra_headers.insert(
                "x-openai-memgen-request".to_string(),
                if mg { "true" } else { "false" }.to_string(),
            );
        }
        let _ = self.headers_for_account(&request.account_id)?;

        Ok(PreparedChatStream {
            request,
            extra_headers,
            cancellation: None,
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
        let account_id = prepared.request.account_id.clone();
        let comp_hash = prepared.request.comp_hash.clone();
        let mut state = ChatStreamState::default();

        self.request_sse_each(
            "/responses",
            &prepared.request.payload,
            Some(&prepared.extra_headers),
            &account_id,
            prepared.cancellation,
            |event| {
                let typ = event
                    .get("type")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        ProviderError::UpstreamProtocol(
                            "ChatGPT OAuth event requires a string type".to_string(),
                        )
                    })?;
                match typ {
                    "response.output_text.delta" => {
                        let delta = event.get("delta").and_then(Value::as_str).ok_or_else(|| {
                            ProviderError::UpstreamProtocol(
                                "response.output_text.delta requires a string delta".to_string(),
                            )
                        })?;
                        if !delta.is_empty() {
                            state.saw_text_delta = true;
                            state.text_parts.push(delta.to_string());
                            emit(json!({"type": "content", "text": delta}))?;
                        }
                    }
                    "response.output_item.done" => {
                        let item = event.get("item").ok_or_else(|| {
                            ProviderError::UpstreamProtocol(
                                "response.output_item.done must contain an object item".to_string(),
                            )
                        })?;
                        if item.get("type").and_then(Value::as_str)
                            == Some("image_generation_call")
                        {
                            return Err(ProviderError::UpstreamProtocol(
                                "image_generation_call cannot be represented by normal chat"
                                    .to_string(),
                            ));
                        }
                        let tool_call = tool_call_from_response_item(item)?;
                        if let Some(tc) = &tool_call {
                            if !state.tool_call_ids.insert(tc.id.clone()) {
                                return Err(ProviderError::UpstreamProtocol(format!(
                                    "provider response contains duplicate call_id {:?}",
                                    tc.id
                                )));
                            }
                        }
                        state.final_output.push(item.clone());
                        if let Some(tc) = tool_call {
                            state.emitted_tool_call = true;
                            emit(json!({
                                "type": "tool_call",
                                "id": tc.id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                            }))?;
                        }
                        if let Some(web_search) = web_search_event_from_response_item(item)? {
                            state.emitted_web_search = true;
                            emit(web_search)?;
                        }
                    }
                    "response.reasoning_summary_part.added" => {
                        emit(json!({
                            "type": "reasoning_section_break",
                            "summary_index": event.get("summary_index"),
                        }))?;
                    }
                    "response.reasoning_summary_text.delta" => {
                        let delta = event.get("delta").and_then(Value::as_str).ok_or_else(|| {
                            ProviderError::UpstreamProtocol(
                                "response.reasoning_summary_text.delta requires a string delta"
                                    .to_string(),
                            )
                        })?;
                        state.saw_reasoning_summary_delta = true;
                        state.reasoning_summary_parts.push(delta.to_string());
                        if !delta.is_empty() {
                            emit(json!({
                                "type": "reasoning_delta",
                                "text": delta,
                                "summary_index": event.get("summary_index"),
                            }))?;
                        }
                    }
                    "response.reasoning_text.delta" => {
                        let delta = event.get("delta").and_then(Value::as_str).ok_or_else(|| {
                            ProviderError::UpstreamProtocol(
                                "response.reasoning_text.delta requires a string delta".to_string(),
                            )
                        })?;
                        state.saw_reasoning_raw_delta = true;
                        state.reasoning_raw_parts.push(delta.to_string());
                        if !delta.is_empty() {
                            emit(json!({
                                "type": "reasoning_raw_delta",
                                "text": delta,
                                "content_index": event.get("content_index"),
                            }))?;
                        }
                    }
                    "response.failed" => {
                        return Err(ProviderError::UpstreamTransport(response_failure_message(
                            &event, "failed",
                        )));
                    }
                    "response.incomplete" => {
                        return Err(ProviderError::UpstreamTransport(response_failure_message(
                            &event,
                            "incomplete",
                        )));
                    }
                    "response.completed" => {
                        state.saw_completed = true;
                        let response = event
                            .get("response")
                            .and_then(Value::as_object)
                            .ok_or_else(|| {
                                ProviderError::UpstreamProtocol(
                                    "response.completed requires a response object".to_string(),
                                )
                            })?;
                        let response_id = response
                            .get("id")
                            .and_then(Value::as_str)
                            .filter(|id| !id.is_empty())
                            .ok_or_else(|| {
                                ProviderError::UpstreamProtocol(
                                    "response.completed requires a non-empty response id"
                                        .to_string(),
                                )
                            })?
                            .to_string();
                        state.completed_response_id = Some(response_id.clone());
                        let usage_val = match response.get("usage") {
                            None | Some(Value::Null) => None,
                            Some(value) => {
                                parse_usage(value)?;
                                Some(value.clone())
                            }
                        };
                        let end_turn = match response.get("end_turn") {
                            None | Some(Value::Null) => None,
                            Some(Value::Bool(value)) => Some(*value),
                            Some(_) => {
                                return Err(ProviderError::UpstreamProtocol(
                                    "response.completed response.end_turn must be a boolean or null"
                                        .to_string(),
                                ));
                            }
                        };
                        let final_text = text_from_response_items(&state.final_output)?;
                        if state.saw_text_delta {
                            if state.text_parts.join("") != final_text {
                                return Err(ProviderError::UpstreamProtocol(
                                    "response.completed output text does not match streamed output text"
                                        .to_string(),
                                ));
                            }
                        } else {
                            if !final_text.is_empty() {
                                state.saw_text_delta = true;
                                state.text_parts.push(final_text.clone());
                                emit(json!({"type": "content", "text": final_text}))?;
                            }
                        }
                        let (completed_summary, completed_raw) =
                            reasoning_parts_from_response_items(&state.final_output)
                                .map_err(ProviderError::UpstreamProtocol)?;
                        if state.saw_reasoning_summary_delta {
                            if state.reasoning_summary_parts.join("") != completed_summary {
                                return Err(ProviderError::UpstreamProtocol(
                                    "response.completed reasoning summary does not match streamed reasoning summary"
                                        .to_string(),
                                ));
                            }
                        } else if !completed_summary.is_empty() {
                            state.reasoning_summary_parts.push(completed_summary.clone());
                            emit(json!({
                                "type": "reasoning_delta",
                                "text": completed_summary,
                            }))?;
                        }
                        if state.saw_reasoning_raw_delta {
                            if state.reasoning_raw_parts.join("") != completed_raw {
                                return Err(ProviderError::UpstreamProtocol(
                                    "response.completed reasoning content does not match streamed reasoning content"
                                        .to_string(),
                                ));
                            }
                        } else if !completed_raw.is_empty() {
                            state.reasoning_raw_parts.push(completed_raw.clone());
                            emit(json!({
                                "type": "reasoning_raw_delta",
                                "text": completed_raw,
                            }))?;
                        }
                        let reasoning_joined = state.reasoning_summary_parts.join("")
                            + &state.reasoning_raw_parts.join("");
                        let finish_reason = if state.emitted_tool_call {
                            Some("tool_calls")
                        } else if end_turn == Some(false) {
                            None
                        } else {
                            Some("stop")
                        };
                        let mut finish = json!({
                            "type": "finish",
                            "finish_reason": finish_reason,
                            "reasoning_content": if reasoning_joined.is_empty() { Value::Null } else { Value::String(reasoning_joined) },
                            "response_id": response_id,
                        });
                        if let Some(usage) = usage_val {
                            finish
                                .as_object_mut()
                                .expect("finish literal must be an object")
                                .insert("usage".to_string(), usage);
                        }
                        state.pending_finish = Some(finish);
                    }
                    _ => {}
                }
                Ok(())
            },
        )?;

        if !state.saw_completed {
            return Err(ProviderError::UpstreamProtocol(
                "ChatGPT OAuth response stream ended before response.completed".to_string(),
            ));
        }

        if let Some(response_id) = state.completed_response_id {
            self.response_chains.commit_for_model(
                &account_id,
                &response_id,
                &conversation_input,
                &state.final_output,
                comp_hash.as_deref(),
            )?;
        }
        emit(state.pending_finish.ok_or_else(|| {
            ProviderError::UpstreamProtocol(
                "response.completed did not produce a finish event".to_string(),
            )
        })?)?;

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_image_for_resolved_model_with_controls(
        &self,
        prompt: &str,
        reference_images: &[Value],
        size: Option<&str>,
        reasoning_effort: Option<&str>,
        resolved: ResolvedModel,
        responses_lite: Option<&Value>,
        controls: &GenerationControls,
    ) -> Result<Vec<Value>, ProviderError> {
        validate_generate_image_input(prompt, reference_images, size)?;
        if !resolved
            .model
            .input_modalities
            .iter()
            .any(|modality| modality == "image")
        {
            return Err(ProviderError::InvalidRequest(
                "image generation is not supported by the requested model".to_string(),
            ));
        }

        let mut content: Vec<Value> = vec![json!({"type": "input_text", "text": prompt})];
        let validated = validate_image_content_values(reference_images)?;
        content.extend(validated);

        let mut payload = json!({
            "model": resolved.model.slug,
            "instructions": "Use the image_generation tool to create the requested image. Return the generated image through an image_generation_call result.",
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [{"type": "image_generation", "output_format": "png"}],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "stream": true,
            "store": false,
            "include": [],
        });

        set_reasoning_payload_with_options(
            payload.as_object_mut().unwrap(),
            reasoning_effort,
            controls.reasoning.as_ref(),
        )?;
        apply_generation_controls(
            payload.as_object_mut().unwrap(),
            &resolved.model.slug,
            controls,
        )?;
        let merged_text = merge_text_verbosity(None, controls.verbosity.as_deref())?;
        let request = finalize_responses_request(
            payload,
            &resolved,
            responses_lite,
            merged_text.as_ref(),
            None,
            ResponsesEndpointKind::Standard,
        )?;
        let output_items = self.collect_response_output_items(request)?;

        let generated = image_generations_from_response_items(&output_items)?;

        if generated.is_empty() {
            return Err(ProviderError::UpstreamProtocol(
                "image generation response returned no image_generation_call".to_string(),
            ));
        }

        Ok(generated)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn inspect_image_values_for_resolved_model_with_controls(
        &self,
        prompt: &str,
        images: &[Value],
        reasoning_effort: Option<&str>,
        resolved: ResolvedModel,
        responses_lite: Option<&Value>,
        controls: &GenerationControls,
    ) -> Result<String, ProviderError> {
        validate_inspect_image_values_input(prompt, images)?;
        let mut content: Vec<Value> = vec![json!({"type": "input_text", "text": prompt})];
        content.extend(validate_image_content_values(images)?);
        let mut payload = json!({
            "model": resolved.model.slug,
            "instructions": "Inspect the attached image(s) and answer the user's review prompt directly.",
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "stream": true,
            "store": false,
            "include": [],
        });
        set_reasoning_payload_with_options(
            payload.as_object_mut().unwrap(),
            reasoning_effort,
            controls.reasoning.as_ref(),
        )?;
        apply_generation_controls(
            payload.as_object_mut().unwrap(),
            &resolved.model.slug,
            controls,
        )?;
        let merged_text = merge_text_verbosity(None, controls.verbosity.as_deref())?;
        let request = finalize_responses_request(
            payload,
            &resolved,
            responses_lite,
            merged_text.as_ref(),
            None,
            ResponsesEndpointKind::Standard,
        )?;
        let output_items = self.collect_response_output_items(request)?;
        let text = inspection_text_from_response_items(&output_items)?;
        if text.is_empty() {
            return Err(ProviderError::UpstreamProtocol(
                "image inspection response returned empty content".to_string(),
            ));
        }
        Ok(text)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compact_messages_for_resolved_model_with_controls(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        reasoning_effort: Option<&str>,
        resolved: ResolvedModel,
        responses_lite: Option<&Value>,
        controls: &CompactControls,
    ) -> Result<String, ProviderError> {
        let (base_instructions, mut input_items) = split_instructions_and_input(messages)?;
        let tools_payload: Vec<Value> = match tools {
            Some(tools) => tools
                .iter()
                .map(tool_schema_to_response_dict)
                .collect::<Result<_, _>>()?,
            None => Vec::new(),
        };
        if let Some(previous_response_id) = &controls.previous_response_id {
            let mut history = self.response_chains.resolve_for_model(
                &resolved.snapshot.key.account_id,
                previous_response_id,
                resolved.model.comp_hash.as_deref(),
            )?;
            history.append(&mut input_items);
            input_items = history;
        }
        let mut payload = json!({
            "model": resolved.model.slug,
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
            &resolved.model.slug,
            &generation_controls,
        )?;
        let merged_text =
            merge_text_verbosity(controls.text.as_ref(), controls.verbosity.as_deref())?;
        let request = finalize_responses_request(
            payload,
            &resolved,
            responses_lite,
            merged_text.as_ref(),
            controls.service_tier.as_deref(),
            ResponsesEndpointKind::Compact,
        )?;
        let extra_headers = responses_lite_headers(request.use_responses_lite);
        let data = self.post_json(
            "/responses/compact",
            &request.payload,
            Some(&extra_headers),
            &request.account_id,
        )?;

        let output = data
            .get("output")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                ProviderError::UpstreamProtocol(
                    "remote compact response missing output array".to_string(),
                )
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
        let mut completed_response: Option<&Value> = None;

        let extra_headers = responses_lite_headers(request.use_responses_lite);
        let events = self.post_sse(
            "/responses",
            &request.payload,
            Some(&extra_headers),
            &request.account_id,
        )?;
        for event in &events {
            let typ = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match typ {
                "response.output_item.done" => {
                    if let Some(item) = event.get("item") {
                        if item.is_object() {
                            output_items.push(item.clone());
                        }
                    }
                }
                "response.failed" => {
                    return Err(ProviderError::UpstreamTransport(response_failure_message(
                        event, "failed",
                    )));
                }
                "response.incomplete" => {
                    return Err(ProviderError::UpstreamTransport(response_failure_message(
                        event,
                        "incomplete",
                    )));
                }
                "response.completed" => {
                    completed_response = event.get("response");
                }
                _ => {}
            }
        }

        if completed_response.is_none() {
            return Err(ProviderError::UpstreamProtocol(
                "ChatGPT OAuth response stream ended before response.completed".to_string(),
            ));
        }
        Ok(output_items)
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
        resolved: &ResolvedModel,
        tool_choice: Option<&Value>,
        service_tier: Option<&str>,
        text: Option<&Value>,
        client_metadata: Option<&HashMap<String, String>>,
        codex_metadata: Option<bool>,
        responses_lite: Option<&Value>,
        parallel_tool_calls: Option<bool>,
        controls: &GenerationControls,
    ) -> Result<FinalizedResponsesRequest, ProviderError> {
        validate_private_request_controls(None, max_tokens, stop)?;
        validate_normalized_tool_choice(tool_choice)?;
        validate_client_metadata_reserved_keys(client_metadata)?;
        let (instructions, mut input_items) = split_instructions_and_input(messages)?;

        let tools_array: Vec<Value> = match tools {
            Some(ts) => ts
                .iter()
                .map(tool_schema_to_response_dict)
                .collect::<Result<_, _>>()?,
            None => vec![],
        };
        if let Some(previous_response_id) = previous_response_id {
            let mut history = self.response_chains.resolve_for_model(
                &resolved.snapshot.key.account_id,
                previous_response_id,
                resolved.model.comp_hash.as_deref(),
            )?;
            history.append(&mut input_items);
            input_items = history;
        }

        let lite = use_responses_lite(&resolved.model, responses_lite)
            .map_err(ProviderError::InvalidRequest)?;
        let parallel_tool_calls = should_enable_parallel_tool_calls(parallel_tool_calls, lite)
            .map_err(ProviderError::InvalidRequest)?;

        let mut payload = json!({
            "model": resolved.model.slug,
            "input": input_items,
            "tools": tools_array,
            "tool_choice": tool_choice.cloned().unwrap_or(json!("auto")),
            "parallel_tool_calls": parallel_tool_calls,
            "stream": true,
            "store": false,
            "include": [],
        });
        if !instructions.is_empty() {
            payload
                .as_object_mut()
                .unwrap()
                .insert("instructions".to_string(), Value::String(instructions));
        }
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
        apply_generation_controls(
            payload.as_object_mut().unwrap(),
            &resolved.model.slug,
            controls,
        )?;
        let merged_text = merge_text_verbosity(text, controls.verbosity.as_deref())?;
        finalize_responses_request(
            payload,
            resolved,
            responses_lite,
            merged_text.as_ref(),
            service_tier,
            ResponsesEndpointKind::Standard,
        )
    }

    fn headers(&self) -> Result<(HashMap<String, String>, ChatGPTTokenData), ProviderError> {
        let token = auth::token_for_request(self.auth_json_path.as_deref())?;
        let headers = headers_for_token(&token)?;
        Ok((headers, token))
    }

    fn headers_for_account(
        &self,
        expected_account_id: &str,
    ) -> Result<(HashMap<String, String>, ChatGPTTokenData), ProviderError> {
        let (headers, token) = self.headers()?;
        if token.account_id != expected_account_id {
            return Err(ProviderError::Auth(AuthError::Refresh(
                "ChatGPT OAuth account changed after model resolution".to_string(),
            )));
        }
        Ok((headers, token))
    }

    fn observe_models_etag(
        &self,
        token: &ChatGPTTokenData,
        headers: &reqwest::header::HeaderMap,
    ) -> Result<(), ProviderError> {
        let Some(etag) = optional_header_text(headers, "x-models-etag") else {
            return Ok(());
        };
        let client_version = pinned_codex_compatibility_version().to_string();
        self.model_catalog
            .observe_etag(
                &CatalogKey {
                    account_id: token.account_id.clone(),
                    base_url: self.base_url.clone(),
                    client_version,
                },
                Some(etag),
            )
            .map_err(|error| ProviderError::CatalogUnavailable(error.to_string()))
    }

    fn post_json(
        &self,
        path: &str,
        payload: &Value,
        extra_headers: Option<&HashMap<String, String>>,
        expected_account_id: &str,
    ) -> Result<Value, ProviderError> {
        let raw = self.request_json(path, payload, extra_headers, expected_account_id)?;
        let data = strict_json::parse_slice(&raw).map_err(|error| {
            ProviderError::UpstreamProtocol(format!(
                "ChatGPT OAuth response must be valid JSON: {error}"
            ))
        })?;
        if !data.is_object() {
            return Err(ProviderError::UpstreamProtocol(
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
        expected_account_id: &str,
    ) -> Result<Vec<Value>, ProviderError> {
        self.request_sse(path, payload, extra_headers, expected_account_id)
    }

    fn request_sse(
        &self,
        path: &str,
        payload: &Value,
        extra_headers: Option<&HashMap<String, String>>,
        expected_account_id: &str,
    ) -> Result<Vec<Value>, ProviderError> {
        let mut events = Vec::new();
        self.request_sse_each(
            path,
            payload,
            extra_headers,
            expected_account_id,
            None,
            |event| {
                events.push(event);
                Ok(())
            },
        )?;
        Ok(events)
    }

    fn request_sse_each<F>(
        &self,
        path: &str,
        payload: &Value,
        extra_headers: Option<&HashMap<String, String>>,
        expected_account_id: &str,
        cancellation: Option<Arc<AtomicBool>>,
        mut on_event: F,
    ) -> Result<(), ProviderError>
    where
        F: FnMut(Value) -> Result<(), ProviderError>,
    {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| ProviderError::Request(error.to_string()))?;
        runtime.block_on(async {
            for attempt in 0..2 {
                let auth_json_path = self.auth_json_path.clone();
                let expected_account_id = expected_account_id.to_string();
                let (mut headers, token) = tokio::task::spawn_blocking(move || {
                    let token = auth::token_for_request(auth_json_path.as_deref())?;
                    if token.account_id != expected_account_id {
                        return Err(ProviderError::Auth(AuthError::Refresh(
                            "ChatGPT OAuth account changed after model resolution".to_string(),
                        )));
                    }
                    let headers = headers_for_token(&token)?;
                    Ok((headers, token))
                })
                .await
                .map_err(|error| {
                    ProviderError::Request(format!("ChatGPT OAuth credential worker failed: {error}"))
                })??;
                headers.insert("Accept".to_string(), "text/event-stream".to_string());
                if let Some(eh) = extra_headers {
                    for (key, value) in eh {
                        headers.insert(key.clone(), value.clone());
                    }
                }
                let token_values = [
                    token.access_token.as_str(),
                    token.refresh_token.as_str(),
                    token.id_token.as_str(),
                    token.account_id.as_str(),
                ];
                let body = serde_json::to_vec(payload).map_err(|error| {
                    ProviderError::InvalidRequest(format!(
                        "failed to serialize ChatGPT OAuth request: {error}"
                    ))
                })?;
                let client = reqwest::Client::builder()
                    .redirect(reqwest::redirect::Policy::none())
                    .connect_timeout(self.timeout)
                    .read_timeout(self.timeout)
                    .build()
                    .map_err(|error| ProviderError::Request(error.to_string()))?;
                let mut builder = client.post(format!("{}{}", self.base_url, path));
                for (key, value) in &headers {
                    builder = builder.header(key.as_str(), value.as_str());
                }
                let response = tokio::select! {
                    response = builder.body(body).send() => response,
                    () = wait_for_cancellation(cancellation.clone()) => {
                        return Err(ProviderError::Request(
                            "SSE downstream client disconnected".to_string(),
                        ));
                    }
                }.map_err(|error| {
                    ProviderError::UpstreamTransport(format!(
                        "ChatGPT OAuth request failed: {}",
                        auth::redact_text(&error.to_string(), &token_values)
                    ))
                })?;
                let status = response.status();
                if status == reqwest::StatusCode::UNAUTHORIZED && attempt == 0 {
                    tokio::task::spawn_blocking(move || auth::refresh_after_unauthorized(&token))
                        .await
                        .map_err(|error| {
                            ProviderError::Request(format!(
                                "ChatGPT OAuth refresh worker failed: {error}"
                            ))
                        })??;
                    continue;
                }
                if !status.is_success() {
                    let body_text = tokio::select! {
                        body = tokio::time::timeout(self.timeout, response.text()) => body,
                        () = wait_for_cancellation(cancellation.clone()) => {
                            return Err(ProviderError::Request(
                                "SSE downstream client disconnected".to_string(),
                            ));
                        }
                    }
                    .map_err(|_| ProviderError::UpstreamHttp {
                        status: status.as_u16(),
                        message: format!(
                            "ChatGPT OAuth request failed: HTTP {} and the error body exceeded its total timeout",
                            status.as_u16(),
                        ),
                    })?
                    .map_err(|error| {
                        ProviderError::UpstreamHttp {
                            status: status.as_u16(),
                            message: format!(
                                "ChatGPT OAuth request failed: HTTP {} and could not read the error body: {}",
                                status.as_u16(),
                                auth::redact_text(&error.to_string(), &token_values)
                            ),
                        }
                    })?;
                    return Err(ProviderError::UpstreamHttp {
                        status: status.as_u16(),
                        message: format!(
                            "ChatGPT OAuth request failed: HTTP {}: {}",
                            status.as_u16(),
                            auth::redact_text(&body_text, &token_values)
                        ),
                    });
                }

                self.observe_models_etag(&token, response.headers())?;
                let mut bytes_stream = response.bytes_stream();
                let mut raw_buffer = Vec::<u8>::new();
                let mut block = Vec::<String>::new();
                loop {
                    let next = tokio::select! {
                        next = tokio::time::timeout(self.timeout, bytes_stream.next()) => next,
                        () = wait_for_cancellation(cancellation.clone()) => {
                            return Err(ProviderError::Request(
                                "SSE downstream client disconnected".to_string(),
                            ));
                        }
                    }
                        .map_err(|_| {
                            ProviderError::UpstreamTransport(
                                "ChatGPT OAuth SSE stream exceeded its idle timeout".to_string(),
                            )
                        })?;
                    let Some(chunk) = next else {
                        if !raw_buffer.is_empty()
                            && consume_sse_line(
                                &raw_buffer,
                                &mut block,
                                &token_values,
                                &mut on_event,
                            )?
                        {
                            return Ok(());
                        }
                        if !block.is_empty()
                            && emit_sse_block(&mut block, &token_values, &mut on_event)?
                        {
                            return Ok(());
                        }
                        return Ok(());
                    };
                    let chunk = chunk.map_err(|error| {
                        ProviderError::UpstreamTransport(format!(
                            "ChatGPT OAuth SSE read failed: {}",
                            auth::redact_text(&error.to_string(), &token_values)
                        ))
                    })?;
                    raw_buffer.extend_from_slice(&chunk);
                    while let Some(newline) = raw_buffer.iter().position(|byte| *byte == b'\n') {
                        let mut remainder = raw_buffer.split_off(newline + 1);
                        std::mem::swap(&mut raw_buffer, &mut remainder);
                        remainder.truncate(newline);
                        if consume_sse_line(
                            &remainder,
                            &mut block,
                            &token_values,
                            &mut on_event,
                        )? {
                            return Ok(());
                        }
                    }
                }
            }
            unreachable!("ChatGPT OAuth request retry state")
        })
    }

    fn request_json(
        &self,
        path: &str,
        payload: &Value,
        extra_headers: Option<&HashMap<String, String>>,
        expected_account_id: &str,
    ) -> Result<Vec<u8>, ProviderError> {
        for attempt in 0..2 {
            let (mut headers, token) = self.headers_for_account(expected_account_id)?;
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
            let body = serde_json::to_vec(payload).map_err(|error| {
                ProviderError::InvalidRequest(format!(
                    "failed to serialize ChatGPT OAuth request: {error}"
                ))
            })?;

            let client = reqwest::blocking::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .map_err(|error| ProviderError::Request(error.to_string()))?;
            let mut builder = client.post(&url);
            for (k, v) in &headers {
                builder = builder.header(k.as_str(), v.as_str());
            }
            builder = builder.timeout(self.timeout);
            builder = builder.body(body);

            match builder.send() {
                Ok(response) => {
                    let status = response.status();
                    if status == reqwest::StatusCode::UNAUTHORIZED && attempt == 0 {
                        auth::refresh_after_unauthorized(&token)?;
                        continue;
                    }
                    if !status.is_success() {
                        let body_text = response.text().map_err(|error| {
                            ProviderError::UpstreamHttp {
                                status: status.as_u16(),
                                message: format!(
                                    "ChatGPT OAuth request failed: HTTP {} and could not read the error body: {}",
                                    status.as_u16(),
                                    auth::redact_text(&error.to_string(), &token_values)
                                ),
                            }
                        })?;
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
                    self.observe_models_etag(&token, response.headers())?;
                    let bytes = response.bytes().map_err(|e| {
                        ProviderError::UpstreamTransport(format!(
                            "ChatGPT OAuth request failed: {}",
                            auth::redact_text(&e.to_string(), &token_values)
                        ))
                    })?;
                    return Ok(bytes.to_vec());
                }
                Err(e) => {
                    let msg = auth::redact_text(&e.to_string(), &token_values);
                    return Err(ProviderError::UpstreamTransport(format!(
                        "ChatGPT OAuth request failed: {}",
                        msg
                    )));
                }
            }
        }

        unreachable!("ChatGPT OAuth request retry state")
    }
}

async fn wait_for_cancellation(cancellation: Option<Arc<AtomicBool>>) {
    let Some(cancellation) = cancellation else {
        std::future::pending::<()>().await;
        return;
    };
    while !cancellation.load(Ordering::Acquire) {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    }
}

fn headers_for_token(token: &ChatGPTTokenData) -> Result<HashMap<String, String>, ProviderError> {
    let mut headers = codex_cli_headers();
    headers.insert(
        "Authorization".to_string(),
        format!("Bearer {}", token.access_token),
    );
    headers.insert("ChatGPT-Account-Id".to_string(), token.account_id.clone());
    headers.insert("Content-Type".to_string(), "application/json".to_string());
    if token.fedramp {
        headers.insert("X-OpenAI-Fedramp".to_string(), "true".to_string());
    }
    Ok(headers)
}

fn validate_generate_image_input(
    prompt: &str,
    reference_images: &[Value],
    size: Option<&str>,
) -> Result<(), ProviderError> {
    if prompt.trim().is_empty() {
        return Err(ProviderError::InvalidRequest(
            "image generation prompt is required".to_string(),
        ));
    }
    validate_image_content_values(reference_images)?;
    if size.is_some_and(|size| size != "auto") {
        return Err(ProviderError::InvalidRequest(
            "image size is not supported by the Codex OAuth HTTP transport".to_string(),
        ));
    }
    Ok(())
}

fn validate_inspect_image_values_input(
    prompt: &str,
    images: &[Value],
) -> Result<(), ProviderError> {
    if prompt.trim().is_empty() {
        return Err(ProviderError::InvalidRequest(
            "image inspection prompt is required".to_string(),
        ));
    }
    validate_image_content_values(images)?;
    Ok(())
}

pub(crate) fn validate_image_content_values(images: &[Value]) -> Result<Vec<Value>, ProviderError> {
    let mut items = Vec::new();
    for (index, image) in images.iter().enumerate() {
        let object = image.as_object().ok_or_else(|| {
            ProviderError::InvalidRequest(format!("image reference {index} must be an object"))
        })?;
        if let Some(field) = object.keys().find(|field| {
            !matches!(
                field.as_str(),
                "image_url" | "detail" | "prompt_cache_breakpoint"
            )
        }) {
            return Err(ProviderError::InvalidRequest(format!(
                "image reference {index} contains unknown field {field:?}"
            )));
        }
        let image_url = object
            .get("image_url")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ProviderError::InvalidRequest(format!(
                    "image reference {index} requires a string image_url"
                ))
            })?;
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
        if object
            .get("prompt_cache_breakpoint")
            .is_some_and(|value| !value.is_null())
        {
            return Err(ProviderError::InvalidRequest(
                "prompt_cache_breakpoint is not supported by the Codex OAuth HTTP transport"
                    .to_string(),
            ));
        }
        items.push(Value::Object(normalized));
    }
    Ok(items)
}

fn image_generation_from_item(item: &Value) -> Result<Option<Value>, ProviderError> {
    if item.get("type").and_then(|v| v.as_str()) != Some("image_generation_call") {
        return Ok(None);
    }
    let result = item.get("result").and_then(Value::as_str).ok_or_else(|| {
        ProviderError::UpstreamProtocol(
            "image_generation_call requires a string result".to_string(),
        )
    })?;
    let id = match item.get("id") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.as_str()),
        Some(_) => {
            return Err(ProviderError::UpstreamProtocol(
                "image_generation_call id must be a string or null".to_string(),
            ));
        }
    };
    let status = item.get("status").and_then(Value::as_str).ok_or_else(|| {
        ProviderError::UpstreamProtocol(
            "image_generation_call requires a string status".to_string(),
        )
    })?;
    let revised_prompt = match item.get("revised_prompt") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.as_str()),
        Some(_) => {
            return Err(ProviderError::UpstreamProtocol(
                "image_generation_call revised_prompt must be a string or null".to_string(),
            ));
        }
    };

    let mut image = serde_json::Map::from_iter([
        ("status".to_string(), Value::String(status.to_string())),
        ("result".to_string(), Value::String(result.to_string())),
    ]);
    if let Some(id) = id {
        image.insert("id".to_string(), Value::String(id.to_string()));
    }
    if let Some(revised_prompt) = revised_prompt {
        image.insert(
            "revised_prompt".to_string(),
            Value::String(revised_prompt.to_string()),
        );
    }
    Ok(Some(Value::Object(image)))
}

fn image_generations_from_response_items(items: &[Value]) -> Result<Vec<Value>, ProviderError> {
    let mut generated = Vec::new();
    for item in items {
        match item.get("type").and_then(Value::as_str) {
            Some("reasoning") => validate_response_output_item(item)?,
            Some("image_generation_call") => {
                if let Some(image) = image_generation_from_item(item)? {
                    generated.push(image);
                }
            }
            _ => {
                return Err(ProviderError::UpstreamProtocol(
                    "image generation response contains an unsupported output item".to_string(),
                ));
            }
        }
    }
    Ok(generated)
}

fn inspection_text_from_response_items(items: &[Value]) -> Result<String, ProviderError> {
    for item in items {
        if !matches!(
            item.get("type").and_then(Value::as_str),
            Some("reasoning" | "message")
        ) {
            return Err(ProviderError::UpstreamProtocol(
                "image inspection response contains an unsupported output item".to_string(),
            ));
        }
        validate_response_output_item(item)?;
    }
    Ok(text_from_response_items(items)?.trim().to_string())
}

fn decode_sse_block(lines: &[String]) -> Result<Option<Value>, ProviderError> {
    let data_lines: Vec<&str> = lines
        .iter()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim))
        .collect();

    if data_lines.is_empty() {
        return Ok(None);
    }

    let joined = data_lines.join("\n");
    if joined == "[DONE]" {
        return Ok(None);
    }

    let event = strict_json::parse_str(&joined).map_err(|error| {
        ProviderError::UpstreamProtocol(format!(
            "ChatGPT OAuth SSE event must be valid JSON: {error}"
        ))
    })?;
    if event.is_object() {
        Ok(Some(event))
    } else {
        Err(ProviderError::UpstreamProtocol(
            "ChatGPT OAuth SSE event must be a JSON object".to_string(),
        ))
    }
}

fn emit_sse_block<F>(
    block: &mut Vec<String>,
    token_values: &[&str],
    on_event: &mut F,
) -> Result<bool, ProviderError>
where
    F: FnMut(Value) -> Result<(), ProviderError>,
{
    let Some(event) = decode_sse_block(block)? else {
        block.clear();
        return Ok(false);
    };
    block.clear();
    let event = redact_upstream_failure_event(event, token_values);
    validate_response_event(&event)?;
    if !event
        .get("type")
        .and_then(Value::as_str)
        .is_some_and(is_known_response_event_type)
    {
        return Ok(false);
    }
    let terminal = is_terminal_response_event(&event);
    on_event(event)?;
    Ok(terminal)
}

fn consume_sse_line<F>(
    raw_line: &[u8],
    block: &mut Vec<String>,
    token_values: &[&str],
    on_event: &mut F,
) -> Result<bool, ProviderError>
where
    F: FnMut(Value) -> Result<(), ProviderError>,
{
    let raw_line = raw_line.strip_suffix(b"\r").unwrap_or(raw_line);
    let line = std::str::from_utf8(raw_line).map_err(|_| {
        ProviderError::UpstreamProtocol(
            "ChatGPT OAuth SSE stream contains invalid UTF-8".to_string(),
        )
    })?;
    if line.is_empty() {
        return emit_sse_block(block, token_values, on_event);
    }
    block.push(line.to_string());
    Ok(false)
}

#[cfg(test)]
fn classify_sse_line_read_error(error: std::io::Error, secrets: &[&str]) -> ProviderError {
    let message = auth::redact_text(&error.to_string(), secrets);
    if error.kind() == std::io::ErrorKind::InvalidData {
        ProviderError::UpstreamProtocol(format!(
            "ChatGPT OAuth SSE stream contains invalid text data: {message}"
        ))
    } else {
        ProviderError::UpstreamTransport(format!("ChatGPT OAuth SSE read failed: {message}"))
    }
}

fn is_terminal_response_event(event: &Value) -> bool {
    matches!(
        event.get("type").and_then(Value::as_str),
        Some("response.completed" | "response.failed" | "response.incomplete")
    )
}

fn is_known_response_event_type(event_type: &str) -> bool {
    matches!(
        event_type,
        "response.created"
            | "response.metadata"
            | "codex.response.metadata"
            | "responsesapi.websocket_timing"
            | "response.in_progress"
            | "response.queued"
            | "response.output_item.added"
            | "response.output_item.done"
            | "response.content_part.added"
            | "response.content_part.done"
            | "response.output_text.delta"
            | "response.output_text.done"
            | "response.function_call_arguments.delta"
            | "response.function_call_arguments.done"
            | "response.reasoning_summary_part.added"
            | "response.reasoning_summary_part.done"
            | "response.reasoning_summary_text.delta"
            | "response.reasoning_summary_text.done"
            | "response.reasoning_text.delta"
            | "response.reasoning_text.done"
            | "response.web_search_call.in_progress"
            | "response.web_search_call.searching"
            | "response.web_search_call.completed"
            | "response.image_generation_call.in_progress"
            | "response.image_generation_call.generating"
            | "response.image_generation_call.partial_image"
            | "response.image_generation_call.completed"
            | "response.failed"
            | "response.incomplete"
            | "response.completed"
    )
}

fn is_unsupported_response_event_type(event_type: &str) -> bool {
    matches!(
        event_type,
        "response.file_search_call.in_progress"
            | "response.file_search_call.searching"
            | "response.file_search_call.completed"
            | "response.code_interpreter_call.in_progress"
            | "response.code_interpreter_call.interpreting"
            | "response.code_interpreter_call_code.delta"
            | "response.code_interpreter_call_code.done"
            | "response.code_interpreter_call.completed"
            | "response.mcp_call.in_progress"
            | "response.mcp_call_arguments.delta"
            | "response.mcp_call_arguments.done"
            | "response.mcp_call.completed"
            | "response.mcp_call.failed"
            | "response.mcp_list_tools.in_progress"
            | "response.mcp_list_tools.completed"
            | "response.mcp_list_tools.failed"
            | "response.shell_call_command.added"
            | "response.shell_call_command.delta"
            | "response.shell_call_command.done"
            | "response.shell_call_output_content.delta"
            | "response.shell_call_output_content.done"
            | "response.audio.delta"
            | "response.audio.done"
            | "response.audio.transcript.delta"
            | "response.audio.transcript.done"
            | "response.refusal.delta"
            | "response.refusal.done"
            | "response.output_text.annotation.added"
            | "response.custom_tool_call_input.delta"
            | "response.custom_tool_call_input.done"
    )
}

fn validate_response_event(event: &Value) -> Result<(), ProviderError> {
    let event_type = event.get("type").and_then(Value::as_str).ok_or_else(|| {
        ProviderError::UpstreamProtocol(
            "ChatGPT OAuth SSE event requires a string type".to_string(),
        )
    })?;
    if event_type == "error" {
        return Err(ProviderError::UpstreamTransport(response_failure_message(
            event, "error",
        )));
    }
    if is_unsupported_response_event_type(event_type) {
        return Err(ProviderError::UpstreamProtocol(
            "ChatGPT OAuth SSE event has an unsupported semantic type".to_string(),
        ));
    }
    if !is_known_response_event_type(event_type) {
        return Ok(());
    }
    if event_type == "response.created" {
        if !event.get("response").is_some_and(Value::is_object) {
            return Err(ProviderError::UpstreamProtocol(
                "response.created must contain an object response".to_string(),
            ));
        }
        return Ok(());
    }
    if event_type == "response.output_item.added" {
        let item = event
            .get("item")
            .filter(|item| item.is_object())
            .ok_or_else(|| {
                ProviderError::UpstreamProtocol(
                    "response.output_item.added must contain an object item".to_string(),
                )
            })?;
        validate_added_response_output_item(item)?;
        return Ok(());
    }
    if event_type == "response.output_item.done" {
        let item = event
            .get("item")
            .filter(|item| item.is_object())
            .ok_or_else(|| {
                ProviderError::UpstreamProtocol(format!("{event_type} must contain an object item"))
            })?;
        validate_response_output_item(item)?;
        return Ok(());
    }
    if matches!(
        event_type,
        "response.content_part.added" | "response.content_part.done"
    ) {
        let part = event
            .get("part")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                ProviderError::UpstreamProtocol(format!("{event_type} must contain an object part"))
            })?;
        if part.get("type").and_then(Value::as_str) != Some("output_text") {
            return Err(ProviderError::UpstreamProtocol(format!(
                "{event_type} has an unsupported semantic part type"
            )));
        }
        if !part.get("text").is_some_and(Value::is_string) {
            return Err(ProviderError::UpstreamProtocol(format!(
                "{event_type} output_text part requires a text string"
            )));
        }
        if part
            .get("annotations")
            .is_some_and(|value| !value.is_array())
        {
            return Err(ProviderError::UpstreamProtocol(format!(
                "{event_type} output_text annotations must be an array"
            )));
        }
        if part.get("logprobs").is_some_and(|value| !value.is_array()) {
            return Err(ProviderError::UpstreamProtocol(format!(
                "{event_type} output_text logprobs must be an array"
            )));
        }
        return Ok(());
    }
    if event_type == "response.output_text.delta"
        && !event.get("delta").is_some_and(Value::is_string)
    {
        return Err(ProviderError::UpstreamProtocol(format!(
            "{event_type} requires a string delta"
        )));
    }
    if event_type == "response.reasoning_summary_text.delta"
        && (event.get("delta").is_none_or(|value| !value.is_string())
            || !event.get("summary_index").is_some_and(Value::is_i64))
    {
        return Err(ProviderError::UpstreamProtocol(
            "response.reasoning_summary_text.delta requires a string delta and integer summary_index"
                .to_string(),
        ));
    }
    if event_type == "response.reasoning_summary_text.done"
        && (event.get("item_id").is_none_or(|value| !value.is_string())
            || !event.get("text").is_some_and(Value::is_string)
            || !event.get("summary_index").is_some_and(Value::is_i64))
    {
        return Err(ProviderError::UpstreamProtocol(
            "response.reasoning_summary_text.done requires string item_id/text and integer summary_index"
                .to_string(),
        ));
    }
    if event_type == "response.reasoning_text.delta"
        && (event.get("delta").is_none_or(|value| !value.is_string())
            || !event.get("content_index").is_some_and(Value::is_i64))
    {
        return Err(ProviderError::UpstreamProtocol(
            "response.reasoning_text.delta requires a string delta and integer content_index"
                .to_string(),
        ));
    }
    if event_type == "response.reasoning_summary_part.added"
        && !event.get("summary_index").is_some_and(Value::is_i64)
    {
        return Err(ProviderError::UpstreamProtocol(
            "response.reasoning_summary_part.added requires integer summary_index".to_string(),
        ));
    }
    if event_type != "response.completed" {
        return Ok(());
    }
    let response = event
        .get("response")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ProviderError::UpstreamProtocol(
                "ChatGPT OAuth response.completed requires a response object".to_string(),
            )
        })?;
    if response
        .get("id")
        .and_then(Value::as_str)
        .is_none_or(str::is_empty)
    {
        return Err(ProviderError::UpstreamProtocol(
            "ChatGPT OAuth response.completed requires a non-empty response id".to_string(),
        ));
    }
    if let Some(usage) = response.get("usage").filter(|usage| !usage.is_null()) {
        parse_usage(usage)?;
    }
    if response
        .get("end_turn")
        .is_some_and(|value| !value.is_null() && !value.is_boolean())
    {
        return Err(ProviderError::UpstreamProtocol(
            "ChatGPT OAuth response.completed response.end_turn must be a boolean or null"
                .to_string(),
        ));
    }
    Ok(())
}

fn validate_added_response_output_item(item: &Value) -> Result<(), ProviderError> {
    let item_type = item
        .get("type")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            ProviderError::UpstreamProtocol(
                "response.output_item.added item requires a non-empty string type".to_string(),
            )
        })?;
    if item_type == "custom_tool_call" {
        return Err(ProviderError::UpstreamProtocol(
            "custom_tool_call is not supported by the public tool contract".to_string(),
        ));
    }
    validate_response_item_optional_fields(item, item_type)?;
    match item_type {
        "function_call" => {
            for field in ["name", "arguments", "call_id"] {
                if !item.get(field).is_some_and(Value::is_string) {
                    return Err(ProviderError::UpstreamProtocol(format!(
                        "response.output_item.added {item_type} requires string {field}"
                    )));
                }
            }
        }
        "message" => {
            if !item.get("role").is_some_and(Value::is_string) {
                return Err(ProviderError::UpstreamProtocol(
                    "response.output_item.added message requires string role and content array"
                        .to_string(),
                ));
            }
            let content = item
                .get("content")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    ProviderError::UpstreamProtocol(
                        "response.output_item.added message requires string role and content array"
                            .to_string(),
                    )
                })?;
            for (index, part) in content.iter().enumerate() {
                let Some(object) = part.as_object() else {
                    return Err(ProviderError::UpstreamProtocol(format!(
                        "response.output_item.added message content[{index}] must be an object"
                    )));
                };
                let value_field = match object.get("type").and_then(Value::as_str) {
                    Some("input_text" | "output_text") => "text",
                    Some("input_image") => "image_url",
                    Some("input_audio") => "audio_url",
                    _ => {
                        return Err(ProviderError::UpstreamProtocol(format!(
                            "response.output_item.added message content[{index}] is invalid"
                        )));
                    }
                };
                if !object.get(value_field).is_some_and(Value::is_string) {
                    return Err(ProviderError::UpstreamProtocol(format!(
                        "response.output_item.added message content[{index}] is invalid"
                    )));
                }
            }
        }
        "reasoning" => {
            let summary = item
                .get("summary")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    ProviderError::UpstreamProtocol(
                        "response.output_item.added reasoning requires a summary array".to_string(),
                    )
                })?;
            for (index, part) in summary.iter().enumerate() {
                let valid = part.as_object().is_some_and(|object| {
                    object.get("type").and_then(Value::as_str) == Some("summary_text")
                        && object.get("text").is_some_and(Value::is_string)
                });
                if !valid {
                    return Err(ProviderError::UpstreamProtocol(format!(
                        "response.output_item.added reasoning summary[{index}] is invalid"
                    )));
                }
            }
            if let Some(content) = item.get("content").filter(|value| !value.is_null()) {
                let content = content.as_array().ok_or_else(|| {
                    ProviderError::UpstreamProtocol(
                        "response.output_item.added reasoning content must be an array or null"
                            .to_string(),
                    )
                })?;
                for (index, part) in content.iter().enumerate() {
                    let valid = part.as_object().is_some_and(|object| {
                        matches!(
                            object.get("type").and_then(Value::as_str),
                            Some("reasoning_text" | "text")
                        ) && object.get("text").is_some_and(Value::is_string)
                    });
                    if !valid {
                        return Err(ProviderError::UpstreamProtocol(format!(
                            "response.output_item.added reasoning content[{index}] is invalid"
                        )));
                    }
                }
            }
            if item
                .get("encrypted_content")
                .is_some_and(|value| !value.is_null() && !value.is_string())
            {
                return Err(ProviderError::UpstreamProtocol(
                    "response.output_item.added reasoning encrypted_content must be a string or null"
                        .to_string(),
                ));
            }
        }
        "web_search_call" => {
            if item
                .get("status")
                .is_some_and(|value| !value.is_null() && !value.is_string())
            {
                return Err(ProviderError::UpstreamProtocol(
                    "response.output_item.added web_search_call status must be a string or null"
                        .to_string(),
                ));
            }
            if item
                .get("action")
                .is_some_and(|value| !value.is_null() && !value.is_object())
            {
                return Err(ProviderError::UpstreamProtocol(
                    "response.output_item.added web_search_call action must be an object or null"
                        .to_string(),
                ));
            }
        }
        "image_generation_call" => {
            image_generation_from_item(item)?;
        }
        _ => {
            return Err(ProviderError::UpstreamProtocol(
                "response.output_item.added item has an unsupported type".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_response_output_item(item: &Value) -> Result<(), ProviderError> {
    let item_type = item.get("type").and_then(Value::as_str).ok_or_else(|| {
        ProviderError::UpstreamProtocol("response output item requires a string type".to_string())
    })?;
    if item_type == "custom_tool_call" {
        return Err(ProviderError::UpstreamProtocol(
            "custom_tool_call is not supported by the public tool contract".to_string(),
        ));
    }
    validate_response_item_optional_fields(item, item_type)?;
    match item_type {
        "function_call" => {
            tool_call_from_response_item(item)?;
        }
        "image_generation_call" => {
            image_generation_from_item(item)?;
        }
        "web_search_call" => {
            web_search_event_from_response_item(item)?;
        }
        "message" => {
            if item.get("role").and_then(Value::as_str) != Some("assistant") {
                return Err(ProviderError::UpstreamProtocol(
                    "response message item role must be 'assistant'".to_string(),
                ));
            }
            let content = item
                .get("content")
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    ProviderError::UpstreamProtocol(
                        "response message item requires a content array".to_string(),
                    )
                })?;
            for (index, part) in content.iter().enumerate() {
                let object = part.as_object().ok_or_else(|| {
                    ProviderError::UpstreamProtocol(format!(
                        "response message content item {index} must be an object"
                    ))
                })?;
                let part_type = object.get("type").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::UpstreamProtocol(format!(
                        "response message content item {index} requires a string type"
                    ))
                })?;
                match part_type {
                    "output_text" => {
                        if !object.get("text").is_some_and(Value::is_string) {
                            return Err(ProviderError::UpstreamProtocol(format!(
                                "response message content item {index} requires string text"
                            )));
                        }
                    }
                    _ => {
                        return Err(ProviderError::UpstreamProtocol(format!(
                            "response message content item {index} has an unsupported type"
                        )));
                    }
                }
            }
        }
        "reasoning" => {
            reasoning_from_response_items(std::slice::from_ref(item))
                .map_err(ProviderError::UpstreamProtocol)?;
            if !matches!(
                item.get("encrypted_content"),
                None | Some(Value::Null | Value::String(_))
            ) {
                return Err(ProviderError::UpstreamProtocol(
                    "reasoning encrypted_content must be a string or null".to_string(),
                ));
            }
        }
        _ => {
            return Err(ProviderError::UpstreamProtocol(
                "response output item has an unsupported type".to_string(),
            ));
        }
    }
    Ok(())
}

fn validate_response_item_optional_fields(
    item: &Value,
    item_type: &str,
) -> Result<(), ProviderError> {
    let nullable_string = |field: &str| -> Result<(), ProviderError> {
        if !matches!(item.get(field), None | Some(Value::Null | Value::String(_))) {
            return Err(ProviderError::UpstreamProtocol(format!(
                "{item_type} {field} must be a string or null"
            )));
        }
        Ok(())
    };

    nullable_string("id")?;
    if let Some(metadata) = item
        .get("internal_chat_message_metadata_passthrough")
        .filter(|value| !value.is_null())
    {
        let metadata = metadata.as_object().ok_or_else(|| {
            ProviderError::UpstreamProtocol(format!(
                "{item_type} internal_chat_message_metadata_passthrough must be an object or null"
            ))
        })?;
        if !matches!(
            metadata.get("turn_id"),
            None | Some(Value::Null | Value::String(_))
        ) {
            return Err(ProviderError::UpstreamProtocol(format!(
                "{item_type} internal_chat_message_metadata_passthrough.turn_id must be a string or null"
            )));
        }
        if !matches!(
            metadata.get("create_time"),
            None | Some(Value::Null | Value::Number(_))
        ) {
            return Err(ProviderError::UpstreamProtocol(format!(
                "{item_type} internal_chat_message_metadata_passthrough.create_time must be a JSON number or null"
            )));
        }
    }

    match item_type {
        "message" => match item.get("phase") {
            None | Some(Value::Null) => {}
            Some(Value::String(value)) if value == "commentary" || value == "final_answer" => {}
            _ => {
                return Err(ProviderError::UpstreamProtocol(
                    "message phase must be commentary, final_answer, or null".to_string(),
                ));
            }
        },
        "reasoning" => nullable_string("encrypted_content")?,
        "function_call" => {
            nullable_string("namespace")?;
            if let Some(encrypted_args) = item
                .get("encrypted_function_args")
                .filter(|value| !value.is_null())
            {
                let encrypted_args = encrypted_args.as_array().ok_or_else(|| {
                    ProviderError::UpstreamProtocol(
                        "function_call encrypted_function_args must be a string array or null"
                            .to_string(),
                    )
                })?;
                if encrypted_args.iter().any(|value| !value.is_string()) {
                    return Err(ProviderError::UpstreamProtocol(
                        "function_call encrypted_function_args must be a string array or null"
                            .to_string(),
                    ));
                }
            }
        }
        "web_search_call" => nullable_string("status")?,
        "image_generation_call" => nullable_string("revised_prompt")?,
        _ => {}
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

#[cfg(test)]
pub fn messages_to_response_items(messages: &[Message]) -> Result<Vec<Value>, ProviderError> {
    let refs: Vec<&Message> = messages.iter().collect();
    messages_to_response_items_refs(&refs)
}

fn messages_to_response_items_refs(messages: &[&Message]) -> Result<Vec<Value>, ProviderError> {
    let mut items: Vec<Value> = Vec::new();

    for message in messages {
        if message.structured_content.as_ref().is_some_and(|parts| {
            parts.iter().any(|part| {
                part.get("prompt_cache_breakpoint")
                    .is_some_and(|value| !value.is_null())
            })
        }) {
            return Err(ProviderError::InvalidRequest(
                "prompt_cache_breakpoint is not supported by the Codex OAuth HTTP transport"
                    .to_string(),
            ));
        }
        if message.role == MessageRole::System
            && message.content.starts_with(REMOTE_COMPACTION_MARKER)
        {
            let raw = message.content[REMOTE_COMPACTION_MARKER.len()..].trim();
            let parsed = strict_json::parse_str(raw).map_err(|error| {
                ProviderError::InvalidRequest(format!(
                    "remote compaction marker must contain valid JSON: {error}"
                ))
            })?;
            let compacted = parsed.as_array().ok_or_else(|| {
                ProviderError::InvalidRequest(
                    "remote compaction marker must contain a JSON array".to_string(),
                )
            })?;
            items.extend(filter_compacted_history_items(compacted).map_err(
                |error| match error {
                    ProviderError::UpstreamProtocol(message) => {
                        ProviderError::InvalidRequest(message)
                    }
                    other => other,
                },
            )?);
            continue;
        }

        if message.role == MessageRole::Tool {
            let call_id = message.tool_call_id.as_deref().ok_or_else(|| {
                ProviderError::InvalidRequest(
                    "tool messages require a string tool_call_id".to_string(),
                )
            })?;
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
                    "arguments": tc.arguments,
                }));
            }
            continue;
        }

        let role = match message.role {
            MessageRole::Assistant => "assistant",
            MessageRole::Developer => "developer",
            MessageRole::User => "user",
            MessageRole::System | MessageRole::Tool => {
                return Err(ProviderError::InvalidRequest(format!(
                    "unsupported internal message role {:?}",
                    message.role
                )));
            }
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
    json!({
        "type": "message",
        "role": role,
        "content": content_items,
    })
}

fn tool_schema_to_response_dict(tool: &ToolSchema) -> Result<Value, ProviderError> {
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
            .ok_or_else(|| {
                ProviderError::InvalidRequest(
                    "web_search tool metadata requires an openai_tool object".to_string(),
                )
            });
    }
    let mut output = json!({
        "type": "function",
        "name": tool.name,
        "parameters": tool.parameters,
        "strict": tool.strict,
    });
    if let Some(description) = tool.description.as_ref() {
        output
            .as_object_mut()
            .expect("function tool payload is an object")
            .insert(
                "description".to_string(),
                Value::String(description.clone()),
            );
    }
    Ok(output)
}

fn finalize_responses_request(
    mut payload: Value,
    resolved: &ResolvedModel,
    responses_lite: Option<&Value>,
    text: Option<&Value>,
    service_tier: Option<&str>,
    endpoint: ResponsesEndpointKind,
) -> Result<FinalizedResponsesRequest, ProviderError> {
    let model = &resolved.model;
    for modality in required_input_modalities(&payload) {
        if !model
            .input_modalities
            .iter()
            .any(|supported| supported == modality)
        {
            return Err(ProviderError::InvalidRequest(format!(
                "the requested model does not support {modality} input"
            )));
        }
    }
    let has_image_detail = value_has_image_detail(&payload);
    if !model.supports_image_detail_original && value_has_original_image_detail(&payload) {
        return Err(ProviderError::InvalidRequest(
            "image detail 'original' is not supported for the requested model".to_string(),
        ));
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
    if let Some(mode) = payload_object
        .get("reasoning")
        .and_then(Value::as_object)
        .and_then(|reasoning| reasoning.get("mode"))
        .cloned()
    {
        match mode {
            Value::Null => {
                if let Some(reasoning) = payload_object
                    .get_mut("reasoning")
                    .and_then(Value::as_object_mut)
                {
                    reasoning.remove("mode");
                }
            }
            _ => {
                return Err(ProviderError::InvalidRequest(
                    "reasoning.mode is not supported by the Codex OAuth HTTP transport".to_string(),
                ));
            }
        }
    }
    let conversation_input = payload_object
        .get("input")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| ProviderError::InvalidRequest("Responses input must be an array".into()))?;
    payload_object.insert("model".to_string(), Value::String(model.slug.clone()));
    apply_model_capability_fields(payload_object, model, text, service_tier)
        .map_err(ProviderError::InvalidRequest)?;
    if payload_object
        .get("reasoning")
        .and_then(Value::as_object)
        .and_then(|reasoning| reasoning.get("effort"))
        .is_none()
    {
        if let Some(default_reasoning_level) = model.default_reasoning_level.as_deref() {
            set_reasoning_payload_with_options(
                payload_object,
                Some(default_reasoning_level),
                None,
            )?;
        }
    }
    if let Some(effort) = payload_object
        .get("reasoning")
        .and_then(Value::as_object)
        .and_then(|reasoning| reasoning.get("effort"))
        .and_then(Value::as_str)
        .map(str::to_string)
    {
        let wire_effort = resolve_reasoning_effort(model, &effort)?;
        payload_object
            .get_mut("reasoning")
            .and_then(Value::as_object_mut)
            .expect("reasoning was read as an object")
            .insert("effort".to_string(), Value::String(wire_effort));
    }
    let reasoning_summary = (model.supports_reasoning_summary_parameter
        && model.default_reasoning_summary != "none")
        .then_some(model.default_reasoning_summary.as_str());
    if let Some(summary) = reasoning_summary {
        let reasoning = payload_object
            .entry("reasoning".to_string())
            .or_insert_with(|| json!({}))
            .as_object_mut()
            .ok_or_else(|| {
                ProviderError::InvalidRequest("reasoning must be an object".to_string())
            })?;
        reasoning.insert("summary".to_string(), Value::String(summary.to_string()));
    } else if let Some(reasoning) = payload_object
        .get_mut("reasoning")
        .and_then(Value::as_object_mut)
    {
        reasoning.remove("summary");
    }

    match endpoint {
        ResponsesEndpointKind::Standard => {
            let include = payload_object
                .entry("include".to_string())
                .or_insert_with(|| json!([]));
            let include_items = include.as_array_mut().ok_or_else(|| {
                ProviderError::InvalidRequest("include must be an array".to_string())
            })?;
            if !include_items
                .iter()
                .any(|value| value.as_str() == Some("reasoning.encrypted_content"))
            {
                include_items.push(json!("reasoning.encrypted_content"));
            }
        }
        ResponsesEndpointKind::Compact => {
            payload_object.remove("include");
        }
    }

    let lite = use_responses_lite(model, responses_lite).map_err(ProviderError::InvalidRequest)?;
    if lite {
        if has_image_detail {
            return Err(ProviderError::InvalidRequest(
                "image detail is not supported by Responses Lite".to_string(),
            ));
        }
        if payload_object.get("parallel_tool_calls") == Some(&Value::Bool(true)) {
            return Err(ProviderError::InvalidRequest(
                "parallel_tool_calls=true is not supported by Responses Lite".to_string(),
            ));
        }
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
        account_id: resolved.snapshot.key.account_id.clone(),
        comp_hash: resolved.model.comp_hash.clone(),
    })
}

pub(crate) fn resolve_reasoning_effort(
    model: &ModelInfo,
    requested: &str,
) -> Result<String, ProviderError> {
    if requested.is_empty() || requested != requested.trim() {
        return Err(ProviderError::InvalidRequest(
            "reasoning_effort must be a non-empty string without surrounding whitespace"
                .to_string(),
        ));
    }
    if requested == "ultra" {
        let candidate = model
            .multi_agent_reasoning_effort
            .as_ref()
            .filter(|effort| {
                effort.as_str() != "ultra"
                    && model
                        .supported_reasoning_levels
                        .iter()
                        .any(|level| level.effort.as_str() == effort.as_str())
            })
            .cloned()
            .or_else(|| {
                model
                    .supported_reasoning_levels
                    .iter()
                    .find(|level| level.effort == "max")
                    .map(|level| level.effort.clone())
            })
            .or_else(|| {
                model
                    .supported_reasoning_levels
                    .iter()
                    .rev()
                    .find(|level| level.effort != "ultra")
                    .map(|level| level.effort.clone())
            });
        return candidate.ok_or_else(|| {
            ProviderError::InvalidRequest(
                "the requested model has no wire reasoning effort for ultra".to_string(),
            )
        });
    }
    if !model
        .supported_reasoning_levels
        .iter()
        .any(|level| level.effort == requested)
    {
        return Err(ProviderError::InvalidRequest(
            "reasoning effort is not supported for the requested model".to_string(),
        ));
    }
    Ok(if requested == "persistent" {
        "disabled".to_string()
    } else {
        requested.to_string()
    })
}

fn value_has_original_image_detail(value: &Value) -> bool {
    input_has_image_matching(value, |image| {
        image.get("detail").and_then(Value::as_str) == Some("original")
    })
}

fn value_has_image_detail(value: &Value) -> bool {
    input_has_image_matching(value, |image| {
        image.get("detail").is_some_and(|detail| !detail.is_null())
    })
}

#[cfg(test)]
fn value_has_input_image(value: &Value) -> bool {
    input_has_image_matching(value, |_| true)
}

fn required_input_modalities(value: &Value) -> Vec<&'static str> {
    fn collect_parts(parts: &[Value], text: &mut bool, image: &mut bool, audio: &mut bool) {
        for part in parts {
            match part.get("type").and_then(Value::as_str) {
                Some("input_text" | "output_text") => *text = true,
                Some("input_image") => *image = true,
                Some("input_audio") => *audio = true,
                _ => {}
            }
        }
    }

    let mut text = value
        .get("instructions")
        .and_then(Value::as_str)
        .is_some_and(|instructions| !instructions.is_empty());
    let mut image = false;
    let mut audio = false;
    if let Some(items) = value.get("input").and_then(Value::as_array) {
        for item in items {
            let Some(message) = item.as_object() else {
                continue;
            };
            match message.get("type").and_then(Value::as_str) {
                Some("message") => {
                    if let Some(parts) = message.get("content").and_then(Value::as_array) {
                        collect_parts(parts, &mut text, &mut image, &mut audio);
                    }
                }
                Some("function_call_output" | "custom_tool_call_output") => {
                    match message.get("output") {
                        Some(Value::String(_)) => text = true,
                        Some(Value::Array(parts)) => {
                            collect_parts(parts, &mut text, &mut image, &mut audio)
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }
    let mut required = Vec::new();
    if text {
        required.push("text");
    }
    if image {
        required.push("image");
    }
    if audio {
        required.push("audio");
    }
    required
}

fn input_has_image_matching<F>(value: &Value, predicate: F) -> bool
where
    F: Fn(&serde_json::Map<String, Value>) -> bool,
{
    let input = value.get("input").unwrap_or(value);
    let Some(items) = input.as_array() else {
        return false;
    };
    for item in items {
        let Some(message) = item.as_object() else {
            continue;
        };
        if message.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let Some(content) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        if content.iter().any(|part| {
            part.as_object().is_some_and(|part| {
                part.get("type").and_then(Value::as_str) == Some("input_image") && predicate(part)
            })
        }) {
            return true;
        }
    }
    false
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
        .ok_or_else(|| {
            ProviderError::Request(
                "internal Responses Lite payload requires a tools array".to_string(),
            )
        })?;
    if let Some(tool_type) = tools_payload.iter().find_map(|tool| {
        let tool_type = tool.get("type").and_then(|v| v.as_str())?;
        matches!(tool_type, "web_search" | "image_generation").then_some(tool_type)
    }) {
        return Err(ProviderError::InvalidRequest(format!(
            "Responses Lite cannot use hosted {tool_type} without a standalone executor"
        )));
    }

    let instructions = match payload.remove("instructions") {
        None => String::new(),
        Some(Value::String(instructions)) => instructions,
        Some(_) => {
            return Err(ProviderError::Request(
                "internal Responses Lite payload requires string instructions".to_string(),
            ));
        }
    };
    payload.remove("tools");
    payload.insert("parallel_tool_calls".to_string(), Value::Bool(false));
    let input = payload
        .remove("input")
        .and_then(|v| v.as_array().cloned())
        .ok_or_else(|| {
            ProviderError::Request(
                "internal Responses Lite payload requires an input array".to_string(),
            )
        })?;
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
    payload.insert("input".to_string(), Value::Array(items));
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
    let start = events.len().saturating_sub(20);
    events
        .iter()
        .enumerate()
        .filter(|(index, event)| {
            *index >= start || event.get("type").and_then(Value::as_str) == Some("web_search_call")
        })
        .map(|(_, event)| event.clone())
        .collect()
}

fn filter_compacted_history_items(items: &[Value]) -> Result<Vec<Value>, ProviderError> {
    let mut compacted = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let object = item.as_object().ok_or_else(|| {
            ProviderError::UpstreamProtocol(format!(
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
        ProviderError::UpstreamProtocol(format!(
            "remote compact output item {index} requires a string type"
        ))
    })?;
    validate_response_item_optional_fields(&Value::Object(object.clone()), item_type)?;
    match item_type {
        "additional_tools" => {
            if object.get("role").and_then(Value::as_str) != Some("developer") {
                return Err(ProviderError::UpstreamProtocol(format!(
                    "remote compact output additional_tools item {index} requires developer role"
                )));
            }
            let tools = object.get("tools").and_then(Value::as_array).ok_or_else(|| {
                ProviderError::UpstreamProtocol(format!(
                    "remote compact output additional_tools item {index} requires a tools array"
                ))
            })?;
            if tools.iter().any(|tool| !tool.is_object()) {
                return Err(ProviderError::UpstreamProtocol(format!(
                    "remote compact output additional_tools item {index} tools must be objects"
                )));
            }
            Ok(())
        }
        "message" => validate_compacted_message(object, index),
        "agent_message" => validate_compacted_agent_message(object, index),
        "reasoning" => validate_compacted_reasoning(object, index),
        "function_call" => validate_compacted_function_call(object, index),
        "compaction" | "compaction_summary" => {
            if object
                .get("encrypted_content")
                .and_then(Value::as_str)
                .is_none()
            {
                return Err(ProviderError::UpstreamProtocol(format!(
                    "remote compact output {item_type} item {index} requires encrypted_content"
                )));
            }
            Ok(())
        }
        "context_compaction" => match object.get("encrypted_content") {
            None | Some(Value::Null | Value::String(_)) => Ok(()),
            Some(_) => Err(ProviderError::UpstreamProtocol(format!(
                "remote compact output context_compaction item {index} encrypted_content must be a string"
            ))),
        },
        _ => Err(ProviderError::UpstreamProtocol(format!(
            "remote compact output item {index} has an unsupported type"
        ))),
    }
}

fn validate_compacted_message(
    object: &serde_json::Map<String, Value>,
    index: usize,
) -> Result<(), ProviderError> {
    match object.get("role").and_then(Value::as_str) {
        Some("user" | "assistant" | "developer") => {}
        Some(_) => {
            return Err(ProviderError::UpstreamProtocol(format!(
                "remote compact output message {index} has an unsupported role"
            )));
        }
        None => {
            return Err(ProviderError::UpstreamProtocol(format!(
                "remote compact output message {index} requires a string role"
            )));
        }
    }
    let content = object
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ProviderError::UpstreamProtocol(format!(
                "remote compact output message {index} requires a content array"
            ))
        })?;
    for (content_index, content_item) in content.iter().enumerate() {
        let content_object = content_item.as_object().ok_or_else(|| {
            ProviderError::UpstreamProtocol(format!(
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
            Some("input_audio")
                if content_object
                    .get("audio_url")
                    .and_then(Value::as_str)
                    .is_some() => {}
            _ => {
                return Err(ProviderError::UpstreamProtocol(format!(
                    "remote compact output message {index} content item {content_index} is invalid"
                )));
            }
        }
    }
    Ok(())
}

fn validate_compacted_reasoning(
    object: &serde_json::Map<String, Value>,
    index: usize,
) -> Result<(), ProviderError> {
    if !matches!(
        object.get("id"),
        None | Some(Value::Null | Value::String(_))
    ) {
        return Err(ProviderError::UpstreamProtocol(format!(
            "remote compact output reasoning item {index} id must be a string or null"
        )));
    }
    match object.get("encrypted_content") {
        None | Some(Value::Null | Value::String(_)) => {}
        Some(_) => {
            return Err(ProviderError::UpstreamProtocol(format!(
                "remote compact output reasoning item {index} encrypted_content must be a string or null"
            )));
        }
    }
    let summary = object
        .get("summary")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ProviderError::UpstreamProtocol(format!(
                "remote compact output reasoning item {index} requires a summary array"
            ))
        })?;
    for (summary_index, part) in summary.iter().enumerate() {
        let valid = part.as_object().is_some_and(|part| {
            part.get("type").and_then(Value::as_str) == Some("summary_text")
                && part.get("text").is_some_and(Value::is_string)
        });
        if !valid {
            return Err(ProviderError::UpstreamProtocol(format!(
                "remote compact output reasoning item {index} summary item {summary_index} is invalid"
            )));
        }
    }
    if let Some(content) = object.get("content").filter(|value| !value.is_null()) {
        let content = content.as_array().ok_or_else(|| {
            ProviderError::UpstreamProtocol(format!(
                "remote compact output reasoning item {index} content must be an array or null"
            ))
        })?;
        for (content_index, part) in content.iter().enumerate() {
            let valid = part.as_object().is_some_and(|part| {
                matches!(
                    part.get("type").and_then(Value::as_str),
                    Some("reasoning_text" | "text")
                ) && part.get("text").is_some_and(Value::is_string)
            });
            if !valid {
                return Err(ProviderError::UpstreamProtocol(format!(
                    "remote compact output reasoning item {index} content item {content_index} is invalid"
                )));
            }
        }
    }
    Ok(())
}

fn validate_compacted_function_call(
    object: &serde_json::Map<String, Value>,
    index: usize,
) -> Result<(), ProviderError> {
    for field in ["call_id", "name"] {
        if object.get(field).and_then(Value::as_str).is_none() {
            return Err(ProviderError::UpstreamProtocol(format!(
                "remote compact output function_call item {index} requires string {field}"
            )));
        }
    }
    object
        .get("arguments")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ProviderError::UpstreamProtocol(format!(
                "remote compact output function_call item {index} requires string arguments"
            ))
        })?;
    Ok(())
}

fn validate_compacted_agent_message(
    object: &serde_json::Map<String, Value>,
    index: usize,
) -> Result<(), ProviderError> {
    for field in ["author", "recipient"] {
        if object.get(field).and_then(Value::as_str).is_none() {
            return Err(ProviderError::UpstreamProtocol(format!(
                "remote compact output agent_message item {index} requires string {field}"
            )));
        }
    }
    let content = object
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ProviderError::UpstreamProtocol(format!(
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
            return Err(ProviderError::UpstreamProtocol(format!(
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

fn web_search_event_from_response_item(item: &Value) -> Result<Option<Value>, ProviderError> {
    let item_type = item.get("type").and_then(Value::as_str).ok_or_else(|| {
        ProviderError::UpstreamProtocol("response output item requires a string type".to_string())
    })?;
    if item_type != "web_search_call" {
        return Ok(None);
    }
    let tool_id = item.get("id").and_then(Value::as_str).ok_or_else(|| {
        ProviderError::UpstreamProtocol("web_search_call requires a string id".to_string())
    })?;
    let action = item
        .get("action")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ProviderError::UpstreamProtocol("web_search_call requires an action object".to_string())
        })?;
    let query = web_search_query_from_action(action)?;
    let sources = web_search_sources_from_action(action)?;
    Ok(Some(json!({
        "type": "web_search_call",
        "id": tool_id,
        "input": {"query": query},
        "content": sources,
    })))
}

fn web_search_query_from_action(
    action: &serde_json::Map<String, Value>,
) -> Result<String, ProviderError> {
    let action_type = match action.get("type") {
        Some(Value::String(action_type)) => action_type.as_str(),
        Some(_) => {
            return Err(ProviderError::UpstreamProtocol(
                "web search action type must be a string".to_string(),
            ));
        }
        None => {
            return Err(ProviderError::UpstreamProtocol(
                "web search action type must be a string".to_string(),
            ));
        }
    };
    if action_type != "search" {
        return Err(ProviderError::UpstreamProtocol(format!(
            "web search action type {action_type:?} cannot be represented by this facade"
        )));
    }

    let query = match action.get("query") {
        None | Some(Value::Null) => None,
        Some(Value::String(query)) => Some(query.as_str()),
        Some(_) => {
            return Err(ProviderError::UpstreamProtocol(
                "web search action query must be a string".to_string(),
            ));
        }
    };
    let queries = if let Some(queries) = action.get("queries").filter(|value| !value.is_null()) {
        let queries = queries.as_array().ok_or_else(|| {
            ProviderError::UpstreamProtocol(
                "web search action queries must be an array".to_string(),
            )
        })?;
        if queries.iter().any(|query| query.as_str().is_none()) {
            return Err(ProviderError::UpstreamProtocol(
                "web search action queries must contain strings".to_string(),
            ));
        }
        Some(queries)
    } else {
        None
    };
    if queries.is_some_and(|queries| queries.len() > 1) {
        return Err(ProviderError::UpstreamProtocol(
            "web search action contains multiple queries that cannot be represented by this facade"
                .to_string(),
        ));
    }
    if let Some(query) = query {
        if queries
            .and_then(|queries| queries.first())
            .and_then(Value::as_str)
            .is_some_and(|alternate| alternate != query)
        {
            return Err(ProviderError::UpstreamProtocol(
                "web search action query conflicts with queries".to_string(),
            ));
        }
        return Ok(query.to_string());
    }
    queries
        .and_then(|queries| queries.first())
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| {
            ProviderError::UpstreamProtocol("web search action requires a query".to_string())
        })
}

fn web_search_sources_from_action(
    action: &serde_json::Map<String, Value>,
) -> Result<Vec<Value>, ProviderError> {
    let sources = action.get("sources").ok_or_else(|| {
        ProviderError::UpstreamProtocol(
            "web search action sources are required when sources were requested".to_string(),
        )
    })?;
    normalize_web_search_sources(sources)
}

fn normalize_web_search_sources(value: &Value) -> Result<Vec<Value>, ProviderError> {
    let mut out = Vec::new();
    let sources = value.as_array().ok_or_else(|| {
        ProviderError::UpstreamProtocol("web search sources must be an array".to_string())
    })?;
    for (index, source) in sources.iter().enumerate() {
        let object = source.as_object().ok_or_else(|| {
            ProviderError::UpstreamProtocol(format!("web search source {index} must be an object"))
        })?;
        let url = object
            .get("url")
            .and_then(Value::as_str)
            .filter(|url| !url.is_empty())
            .ok_or_else(|| {
                ProviderError::UpstreamProtocol(format!(
                    "web search source {index} requires a non-empty url"
                ))
            })?;
        let mut result = serde_json::Map::new();
        result.insert("type".to_string(), json!("web_search_result"));
        result.insert("url".to_string(), json!(url));
        if let Some(title) = object.get("title") {
            if !title.is_null() {
                let title = title.as_str().ok_or_else(|| {
                    ProviderError::UpstreamProtocol(format!(
                        "web search source {index} title must be a string"
                    ))
                })?;
                result.insert("title".to_string(), json!(title));
            }
        }
        if let Some(page_age) = object.get("page_age") {
            if !page_age.is_null() {
                let page_age = page_age.as_str().ok_or_else(|| {
                    ProviderError::UpstreamProtocol(format!(
                        "web search source {index} page_age must be a string"
                    ))
                })?;
                result.insert("page_age".to_string(), json!(page_age));
            }
        }
        out.push(Value::Object(result));
    }
    Ok(out)
}

fn set_reasoning_payload(
    payload: &mut serde_json::Map<String, Value>,
    reasoning_effort: Option<&str>,
) -> Result<(), ProviderError> {
    let effort = match reasoning_effort {
        Some(e) if !e.is_empty() && e == e.trim() => e,
        Some(_) => {
            return Err(ProviderError::InvalidRequest(
                "reasoning_effort must be a non-empty string when provided".to_string(),
            ));
        }
        None => return Ok(()),
    };

    payload.insert("reasoning".to_string(), json!({"effort": effort}));

    Ok(())
}

fn set_reasoning_payload_with_options(
    payload: &mut serde_json::Map<String, Value>,
    reasoning_effort: Option<&str>,
    reasoning: Option<&Value>,
) -> Result<(), ProviderError> {
    let mut merged = match payload.get("reasoning") {
        None => serde_json::Map::new(),
        Some(Value::Object(reasoning)) => reasoning.clone(),
        Some(_) => {
            return Err(ProviderError::InvalidRequest(
                "request payload reasoning must be an object".to_string(),
            ));
        }
    };
    if merged.get("mode").is_some_and(|mode| !mode.is_null()) {
        return Err(ProviderError::InvalidRequest(
            "reasoning.mode is not supported by the Codex OAuth HTTP transport".to_string(),
        ));
    }
    merged.remove("mode");

    let mut nested_effort: Option<&str> = None;
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
                    .filter(|value| !value.is_empty() && *value == value.trim())
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
                return Err(ProviderError::InvalidRequest(
                    "reasoning.mode is not supported by the Codex OAuth HTTP transport".to_string(),
                ));
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
    let effort = reasoning_effort.or(nested_effort);
    if let Some(effort) = effort {
        if effort.is_empty() || effort != effort.trim() {
            return Err(ProviderError::InvalidRequest(
                "reasoning_effort must be a non-empty string when provided".to_string(),
            ));
        }
        merged.insert("effort".to_string(), Value::String(effort.to_string()));
    }

    if !merged.is_empty() {
        payload.insert("reasoning".to_string(), Value::Object(merged));
    }
    Ok(())
}

fn apply_generation_controls(
    _payload: &mut serde_json::Map<String, Value>,
    _model: &str,
    controls: &GenerationControls,
) -> Result<(), ProviderError> {
    if controls
        .reasoning
        .as_ref()
        .and_then(Value::as_object)
        .and_then(|reasoning| reasoning.get("mode"))
        .is_some_and(|mode| !mode.is_null())
    {
        return Err(ProviderError::InvalidRequest(
            "reasoning.mode is not supported by the Codex OAuth HTTP transport".to_string(),
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
    Ok(())
}

fn validate_private_request_controls(
    temperature: Option<f64>,
    max_tokens: Option<i64>,
    stop: Option<&[String]>,
) -> Result<(), ProviderError> {
    if temperature.is_some() {
        return Err(ProviderError::InvalidRequest(
            "temperature is not supported by the private Codex OAuth HTTP transport".to_string(),
        ));
    }
    if max_tokens.is_some() {
        return Err(ProviderError::InvalidRequest(
            "max_tokens is not supported by the private Codex OAuth HTTP transport".to_string(),
        ));
    }
    if stop.is_some() {
        return Err(ProviderError::InvalidRequest(
            "stop is not supported by the private Codex OAuth HTTP transport".to_string(),
        ));
    }
    Ok(())
}

fn validate_normalized_tool_choice(tool_choice: Option<&Value>) -> Result<(), ProviderError> {
    let Some(tool_choice) = tool_choice else {
        return Ok(());
    };
    match tool_choice {
        Value::String(choice) if matches!(choice.as_str(), "auto" | "none" | "required") => Ok(()),
        Value::Object(object)
            if object.len() == 2
                && object.get("type").and_then(Value::as_str) == Some("function")
                && object
                    .get("name")
                    .and_then(Value::as_str)
                    .is_some_and(|name| !name.is_empty()) =>
        {
            Ok(())
        }
        Value::Object(object)
            if object.len() == 1
                && object.get("type").and_then(Value::as_str) == Some("web_search") =>
        {
            Ok(())
        }
        _ => Err(ProviderError::InvalidRequest(
            "tool_choice is not a valid normalized private Responses tool choice".to_string(),
        )),
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
    if object.get("verbosity").is_some_and(Value::is_null) {
        object.remove("verbosity");
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

fn tool_call_from_response_item(item: &Value) -> Result<Option<ToolCall>, ProviderError> {
    let typ = item.get("type").and_then(Value::as_str).ok_or_else(|| {
        ProviderError::UpstreamProtocol("response output item requires a string type".to_string())
    })?;
    if typ == "custom_tool_call" {
        return Err(ProviderError::UpstreamProtocol(
            "custom_tool_call is not supported by the public tool contract".to_string(),
        ));
    }
    if typ != "function_call" {
        return Ok(None);
    }
    let name = item
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| ProviderError::UpstreamProtocol(format!("{typ} requires a string name")))?;
    let argument_field = "arguments";
    let raw_args = item
        .get(argument_field)
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ProviderError::UpstreamProtocol(format!("{typ} {argument_field} must be a string"))
        })?;

    let call_id = item
        .get("call_id")
        .and_then(Value::as_str)
        .ok_or_else(|| ProviderError::UpstreamProtocol(format!("{typ} requires a string call_id")))?
        .to_string();

    Ok(Some(ToolCall {
        id: call_id,
        name: name.to_string(),
        arguments: raw_args.to_string(),
    }))
}

pub fn text_from_response_items(items: &[Value]) -> Result<String, ProviderError> {
    let mut parts: Vec<String> = Vec::new();
    for item in items {
        let item_type = item.get("type").and_then(Value::as_str).ok_or_else(|| {
            ProviderError::UpstreamProtocol(
                "response output item requires a string type".to_string(),
            )
        })?;
        if item_type != "message" {
            if !matches!(
                item_type,
                "function_call" | "image_generation_call" | "reasoning" | "web_search_call"
            ) {
                return Err(ProviderError::UpstreamProtocol(
                    "response output item has an unsupported type".to_string(),
                ));
            }
            continue;
        }
        if item.get("role").and_then(Value::as_str) != Some("assistant") {
            return Err(ProviderError::UpstreamProtocol(
                "response message item role must be 'assistant'".to_string(),
            ));
        }
        let content = item
            .get("content")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                ProviderError::UpstreamProtocol(
                    "response message item requires a content array".to_string(),
                )
            })?;
        for (index, part) in content.iter().enumerate() {
            let map = part.as_object().ok_or_else(|| {
                ProviderError::UpstreamProtocol(format!(
                    "response message content[{index}] must be an object"
                ))
            })?;
            let part_type = map.get("type").and_then(Value::as_str);
            if part_type != Some("output_text") {
                return Err(ProviderError::UpstreamProtocol(
                    "response message content has an unsupported type".to_string(),
                ));
            }
            let text = map.get("text").and_then(Value::as_str).ok_or_else(|| {
                ProviderError::UpstreamProtocol(format!(
                    "response message content[{index}] requires string text"
                ))
            })?;
            if !text.is_empty() {
                parts.push(text.to_string());
            }
        }
    }
    Ok(parts.join(""))
}

pub(crate) fn parse_usage(value: &Value) -> Result<Usage, ProviderError> {
    let invalid = |message: String| ProviderError::UpstreamProtocol(message);
    let obj = value
        .as_object()
        .ok_or_else(|| invalid("usage must be an object".to_string()))?;
    if let Some(field) = [
        "prompt_tokens",
        "completion_tokens",
        "prompt_tokens_details",
        "cached_input_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ]
    .into_iter()
    .find(|field| obj.contains_key(*field))
    {
        return Err(invalid(format!(
            "usage contains unsupported public alias field {field:?}"
        )));
    }
    let required_count = |key: &str| -> Result<i64, ProviderError> {
        obj.get(key)
            .and_then(Value::as_i64)
            .filter(|value| *value >= 0)
            .ok_or_else(|| invalid(format!("usage requires non-negative {key}")))
    };
    let prompt = required_count("input_tokens")?;
    let completion = required_count("output_tokens")?;
    let total = required_count("total_tokens")?;
    if prompt.checked_add(completion) != Some(total) {
        return Err(invalid(
            "usage total_tokens must equal input_tokens plus output_tokens".to_string(),
        ));
    }

    let (cached_tokens, cache_write_tokens) = match obj.get("input_tokens_details") {
        None | Some(Value::Null) => (None, None),
        Some(value) => {
            let details = value.as_object().ok_or_else(|| {
                invalid("usage input_tokens_details must be an object or null".to_string())
            })?;
            let cached_tokens = details
                .get("cached_tokens")
                .and_then(Value::as_i64)
                .filter(|value| *value >= 0)
                .ok_or_else(|| {
                    invalid("usage cached_tokens must be a non-negative integer".to_string())
                })?;
            let cache_write_tokens = match details.get("cache_write_tokens") {
                None | Some(Value::Null) => None,
                Some(value) => {
                    Some(value.as_i64().filter(|value| *value >= 0).ok_or_else(|| {
                        invalid(
                            "usage cache_write_tokens must be a non-negative integer".to_string(),
                        )
                    })?)
                }
            };
            (Some(cached_tokens), cache_write_tokens)
        }
    };

    Ok(Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        total_tokens: total,
        cached_tokens,
        cache_write_tokens,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};

    const CHATGPT_OAUTH_DEFAULT_MODEL: &str = "test-model";
    const TEST_ACCOUNT_ID: &str = "test-account";

    fn resolved_model_for_test(
        slug: &str,
        use_responses_lite: bool,
        supports_image_detail_original: bool,
    ) -> ResolvedModel {
        resolved_model_from_row_for_test(live_model_row_for_test(
            slug,
            use_responses_lite,
            supports_image_detail_original,
        ))
    }

    fn live_model_row_for_test(
        slug: &str,
        use_responses_lite: bool,
        supports_image_detail_original: bool,
    ) -> Value {
        json!({
            "slug": slug,
            "display_name": "Test Model",
            "description": "live catalog fixture",
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [
                {"effort": "low", "description": "low"},
                {"effort": "medium", "description": "medium"},
                {"effort": "high", "description": "high"},
                {"effort": "xhigh", "description": "xhigh"},
                {"effort": "max", "description": "max"}
            ],
            "visibility": "list",
            "supported_in_api": true,
            "priority": 0,
            "service_tiers": [{"id": "priority", "name": "Priority", "description": "Priority"}],
            "default_service_tier": null,
            "support_verbosity": true,
            "default_verbosity": "medium",
            "supports_image_detail_original": supports_image_detail_original,
            "context_window": 100000,
            "max_context_window": 120000,
            "auto_compact_token_limit": null,
            "input_modalities": ["text", "image"],
            "use_responses_lite": use_responses_lite
        })
    }

    fn resolved_model_from_row_for_test(row: Value) -> ResolvedModel {
        let slug = row["slug"].as_str().unwrap().to_string();
        let snapshot = snapshot_from_rows_for_test(vec![row]);
        let model = snapshot.model(&slug).unwrap();
        ResolvedModel { snapshot, model }
    }

    fn snapshot_from_rows_for_test(rows: Vec<Value>) -> Arc<ModelCatalogSnapshot> {
        let cache = ModelCatalogCache::new(std::time::Duration::from_secs(60));
        let key = CatalogKey {
            account_id: TEST_ACCOUNT_ID.to_string(),
            base_url: "https://example.invalid".to_string(),
            client_version: pinned_codex_compatibility_version().to_string(),
        };
        let body = serde_json::to_vec(&json!({"models": rows})).unwrap();
        cache
            .snapshot(key, || Ok((body, Some("test-etag".to_string()))))
            .unwrap()
    }

    #[test]
    fn provider_rejects_unsafe_base_urls() {
        for base_url in [
            "not-a-url",
            "ftp://example.com/codex",
            "https://user:secret@example.com/codex",
            "https://example.com/codex?mode=test",
            "https://example.com/codex#fragment",
            " https://example.com/codex",
            "https://example.com/codex ",
            "https:///codex",
            "http://example.com\n.evil/codex",
            "https://example.com/a path",
            "https://example.com/%",
            "https://example.com/%zz",
            "https://example.com/%0G",
        ] {
            assert!(
                ChatGPTOAuthProvider::new(String::new(), base_url.to_string(), None, None,)
                    .is_err()
            );
        }
        assert!(ChatGPTOAuthProvider::new(
            String::new(),
            "https://example.com/codex%20api".to_string(),
            None,
            None,
        )
        .is_ok());
    }

    #[test]
    fn image_capability_scans_only_responses_message_content() {
        let opaque_image_shape = json!({"type": "input_image", "detail": "original"});
        let mut payload = json!({
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}]
            }],
            "tools": [{"type": "function", "parameters": {"example": opaque_image_shape}}],
            "client_metadata": {"example": {"type": "input_image", "detail": "original"}}
        });
        assert!(!value_has_input_image(&payload));
        assert!(!value_has_image_detail(&payload));
        assert!(!value_has_original_image_detail(&payload));

        payload["input"][0]["content"] = json!([{
            "type": "input_image",
            "image_url": "data:image/png;base64,AAAA",
            "detail": "original"
        }]);
        assert!(value_has_input_image(&payload));
        assert!(value_has_image_detail(&payload));
        assert!(value_has_original_image_detail(&payload));
    }

    trait LegacyResponsesPayload {
        #[allow(clippy::too_many_arguments)]
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
        ) -> Result<FinalizedResponsesRequest, ProviderError>;
    }

    impl LegacyResponsesPayload for ChatGPTOAuthProvider {
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
            let slug = model.unwrap_or("test-model");
            let explicit_lite = responses_lite.and_then(Value::as_bool).unwrap_or(false);
            let resolved = resolved_model_for_test(slug, explicit_lite, true);
            self.responses_payload_with_controls(
                messages,
                tools,
                reasoning_effort,
                stop,
                prompt_cache_key,
                max_tokens,
                previous_response_id,
                &resolved,
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
    }

    fn finalize_responses_request(
        payload: Value,
        model: &str,
        responses_lite: Option<&Value>,
        text: Option<&Value>,
        service_tier: Option<&str>,
        endpoint: ResponsesEndpointKind,
    ) -> Result<FinalizedResponsesRequest, ProviderError> {
        let resolved = resolved_model_for_test(model, false, true);
        super::finalize_responses_request(
            payload,
            &resolved,
            responses_lite,
            text,
            service_tier,
            endpoint,
        )
    }

    fn usage_from_response(value: &Value) -> Option<Usage> {
        parse_usage(value).ok()
    }

    #[test]
    fn provider_without_an_override_uses_a_finite_request_timeout() {
        let provider = ChatGPTOAuthProvider::new(
            "gpt-5.5".to_string(),
            "http://127.0.0.1:1".to_string(),
            None,
            None,
        )
        .unwrap();
        assert_eq!(provider.timeout, CHATGPT_OAUTH_DEFAULT_TIMEOUT);
    }

    #[test]
    fn provider_rejects_zero_model_catalog_ttl() {
        let result = ChatGPTOAuthProvider::new_with_catalog_ttl(
            "gpt-5.5".to_string(),
            "http://127.0.0.1:1".to_string(),
            None,
            None,
            std::time::Duration::ZERO,
        );
        assert!(
            matches!(result, Err(ProviderError::InvalidRequest(message)) if message.contains("TTL"))
        );
    }

    #[test]
    fn optional_model_etag_headers_ignore_invalid_or_empty_metadata() {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            "etag",
            reqwest::header::HeaderValue::from_bytes(b"\xff").unwrap(),
        );
        assert_eq!(optional_header_text(&headers, "etag"), None);

        headers.insert("etag", reqwest::header::HeaderValue::from_static("   "));
        assert_eq!(optional_header_text(&headers, "etag"), None);

        headers.insert(
            "x-models-etag",
            reqwest::header::HeaderValue::from_static("  live-etag  "),
        );
        assert_eq!(
            optional_header_text(&headers, "x-models-etag"),
            Some("live-etag")
        );
    }

    #[test]
    fn authenticated_sse_failure_events_redact_request_credentials_before_aggregation() {
        use std::io::{Read, Write};

        let auth_path = std::env::temp_dir().join(format!(
            "codex-provider-sse-redaction-{}.json",
            uuid::Uuid::new_v4()
        ));
        let secrets = [
            "header.e30.signature",
            "refresh-token",
            "header.e30.id-signature",
            TEST_ACCOUNT_ID,
        ];
        std::fs::write(
            &auth_path,
            serde_json::to_vec(&json!({
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": secrets[0],
                    "refresh_token": secrets[1],
                    "id_token": secrets[2],
                    "account_id": secrets[3]
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let failure_message = secrets.join(" ");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 8192];
            let _ = stream.read(&mut request).unwrap();
            let body = format!(
                "data: {}\n\n",
                serde_json::to_string(&json!({
                    "type": "response.failed",
                    "response": {
                        "error": {
                            "message": failure_message,
                            "metadata": {"credential_echo": failure_message}
                        }
                    }
                }))
                .unwrap()
            );
            write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
        .unwrap();
            stream.flush().unwrap();
        });

        let provider = ChatGPTOAuthProvider::new(
            String::new(),
            format!("http://{address}"),
            Some(auth_path.to_string_lossy().to_string()),
            Some(std::time::Duration::from_secs(2)),
        )
        .unwrap();
        let error = provider
            .collect_response_output_items(FinalizedResponsesRequest {
                payload: json!({}),
                use_responses_lite: false,
                conversation_input: Vec::new(),
                account_id: TEST_ACCOUNT_ID.to_string(),
                comp_hash: None,
            })
            .unwrap_err()
            .to_string();
        assert!(error.contains("***"));
        for secret in secrets {
            assert!(!error.contains(secret));
        }

        server.join().unwrap();
        std::fs::remove_file(auth_path).unwrap();
    }

    fn response_events_error(events: Vec<Value>, nonstream: bool) -> ProviderError {
        use std::io::{Read, Write};

        let auth_path = std::env::temp_dir().join(format!(
            "codex-provider-response-error-{}.json",
            uuid::Uuid::new_v4()
        ));
        std::fs::write(
            &auth_path,
            serde_json::to_vec(&json!({
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "header.e30.signature",
                    "refresh_token": "refresh-token",
                    "id_token": "header.e30.signature",
                    "account_id": TEST_ACCOUNT_ID
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 8192];
            let _ = stream.read(&mut request).unwrap();
            let body = events
                .into_iter()
                .map(|event| format!("data: {}\n\n", serde_json::to_string(&event).unwrap()))
                .collect::<String>();
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            )
            .unwrap();
            stream.flush().unwrap();
        });

        let provider = ChatGPTOAuthProvider::new(
            String::new(),
            format!("http://{address}"),
            Some(auth_path.to_string_lossy().to_string()),
            Some(std::time::Duration::from_secs(2)),
        )
        .unwrap();
        let prepared = PreparedChatStream {
            request: FinalizedResponsesRequest {
                payload: json!({}),
                use_responses_lite: false,
                conversation_input: Vec::new(),
                account_id: TEST_ACCOUNT_ID.to_string(),
                comp_hash: None,
            },
            extra_headers: HashMap::new(),
            cancellation: None,
        };
        let result = if nonstream {
            provider.chat_prepared(prepared).map(|_| ())
        } else {
            provider.stream_prepared_chat(prepared, |_| Ok(()))
        };

        server.join().unwrap();
        std::fs::remove_file(auth_path).unwrap();
        result.unwrap_err()
    }

    fn duplicate_tool_call_response_error(nonstream: bool) -> ProviderError {
        response_events_error(
            vec![
                json!({
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "duplicate-call",
                        "name": "first",
                        "arguments": "{}"
                    }
                }),
                json!({
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "call_id": "duplicate-call",
                        "name": "second",
                        "arguments": "{}"
                    }
                }),
            ],
            nonstream,
        )
    }

    #[test]
    fn streaming_response_rejects_duplicate_function_call_ids() {
        let error = duplicate_tool_call_response_error(false);
        assert!(matches!(
            error,
            ProviderError::UpstreamProtocol(message) if message.contains("duplicate call_id")
        ));
    }

    #[test]
    fn nonstream_response_rejects_duplicate_function_call_ids() {
        let error = duplicate_tool_call_response_error(true);
        assert!(matches!(
            error,
            ProviderError::UpstreamProtocol(message) if message.contains("duplicate call_id")
        ));
    }

    #[test]
    fn streaming_response_rejects_custom_tool_calls() {
        let error = response_events_error(
            vec![json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "custom_tool_call",
                    "call_id": "custom-call",
                    "name": "shell",
                    "input": "{\"command\":\"pwd\"}"
                }
            })],
            false,
        );
        assert!(matches!(
            error,
            ProviderError::UpstreamProtocol(message) if message.contains("custom_tool_call")
        ));
    }

    #[test]
    fn nonstream_response_rejects_custom_tool_calls() {
        let error = response_events_error(
            vec![json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "custom_tool_call",
                    "call_id": "custom-call",
                    "name": "shell",
                    "input": "{\"command\":\"pwd\"}"
                }
            })],
            true,
        );
        assert!(matches!(
            error,
            ProviderError::UpstreamProtocol(message) if message.contains("custom_tool_call")
        ));
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
            .commit(
                TEST_ACCOUNT_ID,
                "resp-root",
                &initial_input,
                &initial_output,
            )
            .unwrap();

        let expected_root: Vec<Value> = initial_input
            .iter()
            .chain(initial_output.iter())
            .cloned()
            .collect();
        let mut caller_copy = store.resolve(TEST_ACCOUNT_ID, "resp-root").unwrap();
        caller_copy[1]["encrypted_content"] = json!("mutated");
        assert_eq!(
            store.resolve(TEST_ACCOUNT_ID, "resp-root").unwrap(),
            expected_root
        );

        let mut branch_a_input = expected_root.clone();
        branch_a_input.push(json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "A"}]}));
        let mut branch_b_input = expected_root.clone();
        branch_b_input.push(json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": "B"}]}));
        store
            .commit(
                TEST_ACCOUNT_ID,
                "resp-a",
                &branch_a_input,
                &[json!({"type": "function_call", "call_id": "call-a", "name": "a", "arguments": "{}"})],
            )
            .unwrap();
        store
            .commit(
                TEST_ACCOUNT_ID,
                "resp-b",
                &branch_b_input,
                &[json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "B"}]})],
            )
            .unwrap();

        assert_eq!(
            store.resolve(TEST_ACCOUNT_ID, "resp-a").unwrap()[2],
            branch_a_input[2]
        );
        assert_eq!(
            store.resolve(TEST_ACCOUNT_ID, "resp-b").unwrap()[2],
            branch_b_input[2]
        );
    }

    #[test]
    fn response_chain_store_requires_compaction_only_for_known_hash_mismatches() {
        let store = ResponseChainStore::new(4);
        store
            .commit_for_model(
                TEST_ACCOUNT_ID,
                "resp-hashed",
                &[json!({"type": "message", "role": "user", "content": []})],
                &[],
                Some("family-a"),
            )
            .unwrap();

        assert!(store
            .resolve_for_model(TEST_ACCOUNT_ID, "resp-hashed", Some("family-a"))
            .is_ok());
        assert!(store
            .resolve_for_model(TEST_ACCOUNT_ID, "resp-hashed", None)
            .is_ok());
        assert!(matches!(
            store.resolve_for_model(TEST_ACCOUNT_ID, "resp-hashed", Some("family-b")),
            Err(ProviderError::InvalidRequest(message)) if message.contains("requires compaction")
        ));
    }

    #[test]
    fn prepared_request_rejects_an_authenticated_account_switch_before_transport() {
        fn write_auth(path: &std::path::Path, account_id: &str) {
            std::fs::write(
                path,
                serde_json::to_vec(&json!({
                    "auth_mode": "chatgpt",
                    "tokens": {
                        "access_token": "header.e30.signature",
                        "refresh_token": "refresh-token",
                        "id_token": "header.e30.signature",
                        "account_id": account_id
                    }
                }))
                .unwrap(),
            )
            .unwrap();
        }

        let auth_path = std::env::temp_dir().join(format!(
            "codex-provider-account-switch-{}.json",
            uuid::Uuid::new_v4()
        ));
        write_auth(&auth_path, TEST_ACCOUNT_ID);
        let provider = ChatGPTOAuthProvider::new(
            String::new(),
            "http://127.0.0.1:1".to_string(),
            Some(auth_path.to_string_lossy().to_string()),
            Some(std::time::Duration::from_millis(100)),
        )
        .unwrap();
        let messages = vec![
            Message::new(
                MessageRole::System,
                "Answer.".to_string(),
                vec![],
                None,
                None,
            )
            .unwrap(),
            Message::new(MessageRole::User, "Hello".to_string(), vec![], None, None).unwrap(),
        ];
        let prepared = provider
            .prepare_chat_stream_for_resolved_model_with_controls(
                &messages,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                resolved_model_for_test("live-model", false, true),
                None,
                None,
                None,
                None,
                None,
                None,
                Some(false),
                &GenerationControls::default(),
            )
            .unwrap();

        write_auth(&auth_path, "other-account");
        let error = provider
            .stream_prepared_chat(prepared, |_| Ok(()))
            .unwrap_err();
        assert!(matches!(error, ProviderError::Auth(_)));
        std::fs::remove_file(auth_path).unwrap();
    }

    #[test]
    fn response_chain_store_uses_lru_eviction() {
        assert_eq!(RESPONSE_CHAIN_CAPACITY, 256);
        let store = ResponseChainStore::new(2);
        let input = [json!({"type": "message", "role": "user", "content": []})];
        store
            .commit(TEST_ACCOUNT_ID, "resp-a", &input, &[])
            .unwrap();
        store
            .commit(TEST_ACCOUNT_ID, "resp-b", &input, &[])
            .unwrap();
        store.resolve(TEST_ACCOUNT_ID, "resp-a").unwrap();
        store
            .commit(TEST_ACCOUNT_ID, "resp-c", &input, &[])
            .unwrap();

        assert!(matches!(
            store.resolve(TEST_ACCOUNT_ID, "resp-b"),
            Err(ProviderError::InvalidRequest(_))
        ));
        assert!(store.resolve(TEST_ACCOUNT_ID, "resp-a").is_ok());
        assert!(store.resolve(TEST_ACCOUNT_ID, "resp-c").is_ok());
    }

    #[test]
    fn response_chain_store_allows_concurrent_branches() {
        let store = Arc::new(ResponseChainStore::new(256));
        store
            .commit(
                TEST_ACCOUNT_ID,
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
                    let mut input = store.resolve(TEST_ACCOUNT_ID, "resp-root").unwrap();
                    input.push(json!({"type": "message", "role": "user", "content": [{"type": "input_text", "text": index.to_string()}]}));
                    store
                        .commit(TEST_ACCOUNT_ID, &format!("resp-{index}"), &input, &[])
                        .unwrap();
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap();
        }
        for index in 0..8 {
            let history = store
                .resolve(TEST_ACCOUNT_ID, &format!("resp-{index}"))
                .unwrap();
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
        )
        .unwrap();
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
            .commit(TEST_ACCOUNT_ID, "resp-root", &root_input, &root_output)
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
    fn test_responses_payload_rejects_max_tokens() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        )
        .unwrap();
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

        let error = provider
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
            .unwrap_err();

        assert!(matches!(error, ProviderError::InvalidRequest(_)));

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
    fn present_empty_stop_values_are_rejected_by_private_requests() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        )
        .unwrap();
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
            let error = provider
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
                .unwrap_err();
            assert!(matches!(
                error,
                ProviderError::InvalidRequest(message) if message.contains("stop is not supported")
            ));
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
    fn non_empty_stop_is_rejected() {
        let stop = vec!["END".to_string()];
        let error = validate_private_request_controls(None, None, Some(&stop)).unwrap_err();

        assert!(matches!(
            error,
            ProviderError::InvalidRequest(message) if message.contains("stop is not supported")
        ));
    }

    #[test]
    fn provider_rejects_non_normalized_tool_choices() {
        for choice in [
            json!(true),
            json!(1),
            json!("future"),
            json!({}),
            json!({"type": "function", "name": ""}),
            json!({"type": "function", "function": {"name": "lookup"}}),
            json!({"type": "function", "name": "lookup", "extra": true}),
        ] {
            assert!(validate_normalized_tool_choice(Some(&choice)).is_err());
        }
        assert!(validate_normalized_tool_choice(Some(&json!({
            "type": "function",
            "name": "lookup"
        })))
        .is_ok());
    }

    #[test]
    fn reserved_client_metadata_identity_must_not_be_blank() {
        for key in [SESSION_ID_KEY, THREAD_ID_KEY] {
            let metadata = HashMap::from([
                (key.to_string(), "   ".to_string()),
                ("opaque".to_string(), String::new()),
            ]);
            assert!(matches!(
                validate_client_metadata_reserved_keys(Some(&metadata)),
                Err(ProviderError::InvalidRequest(message)) if message.contains(key)
            ));
        }
        let opaque = HashMap::from([("opaque".to_string(), String::new())]);
        assert!(validate_client_metadata_reserved_keys(Some(&opaque)).is_ok());
    }

    #[test]
    fn test_responses_payload_includes_web_search_sources() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        )
        .unwrap();
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
            description: Some("Web search".to_string()),
            parameters: json!({
                "__codex_as_api_tool_type": "web_search",
                "openai_tool": {"type": "web_search", "external_web_access": true},
            }),
            strict: false,
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
        )
        .unwrap();
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

        assert_eq!(
            payload["reasoning"],
            json!({"effort": "high", "summary": "auto"})
        );
        assert_eq!(payload["include"], json!(["reasoning.encrypted_content"]));
    }

    #[test]
    fn test_responses_payload_forces_responses_lite_shape() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        )
        .unwrap();
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
            description: Some("Lookup".to_string()),
            parameters: json!({"type": "object"}),
            strict: false,
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
        )
        .unwrap();
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

        let resolved = resolved_model_for_test("live-lite-model", true, true);
        let request = provider
            .responses_payload_with_controls(
                &messages,
                None,
                None,
                None,
                None,
                None,
                None,
                &resolved,
                None,
                Some("priority"),
                None,
                None,
                None,
                Some(&json!("auto")),
                None,
                &GenerationControls::default(),
            )
            .unwrap();

        assert!(request.use_responses_lite);
        let payload = request.payload;
        assert!(payload.get("tools").is_none());
        assert_eq!(payload["text"], json!({"verbosity": "medium"}));
        assert_eq!(payload["service_tier"], json!("priority"));
    }

    #[test]
    fn responses_payload_accepts_user_input_without_system_instructions() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        )
        .unwrap();
        let messages = vec![Message {
            role: MessageRole::User,
            content: "Hello".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            images: vec![],
            structured_content: None,
        }];

        for responses_lite in [json!(false), json!(true)] {
            let resolved = resolved_model_for_test("live-model", false, true);
            let request = provider
                .responses_payload_with_controls(
                    &messages,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    &resolved,
                    None,
                    None,
                    None,
                    None,
                    None,
                    Some(&responses_lite),
                    None,
                    &GenerationControls::default(),
                )
                .unwrap();

            assert!(request.payload.get("instructions").is_none());
            assert!(request.payload["input"]
                .as_array()
                .unwrap()
                .iter()
                .any(|item| item.get("role") == Some(&json!("user"))));
        }
    }

    #[test]
    fn prepared_chat_rejects_header_unsafe_subagent_values() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        )
        .unwrap();
        let messages = vec![Message {
            role: MessageRole::User,
            content: "Hello".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            images: vec![],
            structured_content: None,
        }];
        for subagent in ["has space", "line\nbreak", "tab\tvalue", "nonascii-é"] {
            let result = provider.prepare_chat_stream_for_resolved_model_with_controls(
                &messages,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(subagent),
                None,
                None,
                resolved_model_for_test("live-model", false, true),
                None,
                None,
                None,
                None,
                None,
                Some(&json!(false)),
                None,
                &GenerationControls::default(),
            );
            let error = match result {
                Err(error) => error,
                Ok(_) => panic!("unsafe subagent was accepted"),
            };
            assert!(error.to_string().contains("visible ASCII"));
            assert!(!error.to_string().contains(subagent));
        }
    }

    #[test]
    fn test_responses_payload_parallel_tool_calls_respects_lite_mode() {
        let provider = ChatGPTOAuthProvider::new(
            CHATGPT_OAUTH_DEFAULT_MODEL.to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        )
        .unwrap();
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

        let build = |resolved: &ResolvedModel,
                     requested: Option<bool>,
                     lite: Value|
         -> Result<FinalizedResponsesRequest, ProviderError> {
            provider.responses_payload_with_controls(
                &messages,
                None,
                None,
                None,
                None,
                None,
                None,
                resolved,
                None,
                None,
                None,
                None,
                None,
                Some(&lite),
                requested,
                &GenerationControls::default(),
            )
        };
        let classic = resolved_model_for_test("parallel-model", false, true);
        let lite = resolved_model_for_test("lite-model", true, true);

        assert_eq!(
            build(&classic, Some(true), json!(false)).unwrap().payload["parallel_tool_calls"],
            json!(true)
        );
        assert!(matches!(
            build(&lite, Some(true), json!("auto")),
            Err(ProviderError::InvalidRequest(_))
        ));
        assert_eq!(
            build(&classic, Some(false), json!(false)).unwrap().payload["parallel_tool_calls"],
            json!(false)
        );
    }

    #[test]
    fn test_codex_cli_headers_include_official_originator_and_pinned_user_agent() {
        let headers = codex_cli_headers();

        assert_eq!(headers.get("originator").unwrap(), "codex_cli_rs");
        let user_agent = headers.get("User-Agent").unwrap();
        assert!(user_agent.starts_with("codex_cli_rs/0.153.3 ("));
        assert!(user_agent.ends_with(") codex-as-api/0.7.0"));
    }

    #[test]
    fn test_codex_cli_version_parser_rejects_invalid_version() {
        assert!(normalize_codex_cli_version("not-a-version").is_none());
        assert!(normalize_codex_cli_version(" 0.153.3").is_none());
        assert!(normalize_codex_cli_version("0.153.3 ").is_none());
    }

    #[test]
    fn test_codex_cli_version_defaults_to_pinned_upstream_contract() {
        assert_eq!(pinned_codex_compatibility_version(), "0.153.3");
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
            assert!(matches!(
                messages_to_response_items(&[marker]),
                Err(ProviderError::InvalidRequest(_))
            ));
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
            json!({"type": "message", "role": "system", "content": []}),
        ] {
            assert!(matches!(
                filter_compacted_history_items(&[item]),
                Err(ProviderError::UpstreamProtocol(_))
            ));
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
    fn compact_variants_validate_common_response_item_fields() {
        let valid = json!([
            {
                "type": "message",
                "id": "",
                "role": "assistant",
                "content": [],
                "phase": "commentary",
                "internal_chat_message_metadata_passthrough": {
                    "turn_id": null,
                    "create_time": 1.5,
                    "content_item_kinds": "default-on-error"
                },
                "future": true
            },
            {"type": "agent_message", "id": null, "author": "agent", "recipient": "parent", "content": []},
            {"type": "compaction", "id": "", "encrypted_content": "opaque"},
            {"type": "context_compaction", "id": null}
        ]);
        assert_eq!(
            filter_compacted_history_items(valid.as_array().unwrap()).unwrap(),
            valid.as_array().unwrap().clone()
        );

        for item in [
            json!({"type": "message", "role": "assistant", "content": [], "phase": "future"}),
            json!({"type": "agent_message", "id": 42, "author": "agent", "recipient": "parent", "content": []}),
            json!({"type": "additional_tools", "role": "developer", "tools": [], "id": 42}),
            json!({"type": "compaction", "encrypted_content": "opaque", "id": 42}),
            json!({"type": "context_compaction", "internal_chat_message_metadata_passthrough": "bad"}),
        ] {
            assert!(matches!(
                filter_compacted_history_items(&[item]),
                Err(ProviderError::UpstreamProtocol(_))
            ));
        }
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
            description: Some("Does stuff".to_string()),
            parameters: json!({"type": "object"}),
            strict: false,
        };
        let result = tool_schema_to_response_dict(&schema).unwrap();
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
    fn test_set_reasoning_payload_preserves_efforts_until_live_resolution() {
        let cases = [
            ("HIGH", "HIGH"),
            ("MaX", "MaX"),
            ("ultra", "ultra"),
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
    fn catalog_default_persistent_reasoning_uses_disabled_wire_value() {
        let mut row = live_model_row_for_test("persistent-model", false, true);
        row["default_reasoning_level"] = json!("persistent");
        row["supported_reasoning_levels"] = json!([
            {"effort": "persistent", "description": "persistent"}
        ]);
        let resolved = resolved_model_from_row_for_test(row);

        let request = super::finalize_responses_request(
            json!({"input": [], "tools": []}),
            &resolved,
            None,
            None,
            None,
            ResponsesEndpointKind::Standard,
        )
        .unwrap();

        assert_eq!(request.payload["reasoning"]["effort"], json!("disabled"));
        assert_eq!(request.payload["reasoning"]["summary"], json!("auto"));
    }

    #[test]
    fn live_reasoning_summary_controls_determine_private_wire_payload() {
        for (supported, summary, expected) in [
            (true, "detailed", Some("detailed")),
            (true, "none", None),
            (false, "concise", None),
        ] {
            let mut row = live_model_row_for_test("summary-model", false, true);
            row["default_reasoning_level"] = Value::Null;
            row["supported_reasoning_levels"] = json!([]);
            row["supports_reasoning_summary_parameter"] = json!(supported);
            row["default_reasoning_summary"] = json!(summary);
            let resolved = resolved_model_from_row_for_test(row);

            let request = super::finalize_responses_request(
                json!({"input": [], "tools": []}),
                &resolved,
                None,
                None,
                None,
                ResponsesEndpointKind::Standard,
            )
            .unwrap();
            assert_eq!(
                request
                    .payload
                    .get("reasoning")
                    .and_then(|reasoning| reasoning.get("summary")),
                expected.map(Value::from).as_ref()
            );
            assert!(request.payload["include"]
                .as_array()
                .is_some_and(|include| include
                    .iter()
                    .any(|value| { value.as_str() == Some("reasoning.encrypted_content") })));
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
    fn original_image_detail_is_gated_by_live_model_capability() {
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

        let supported = resolved_model_for_test("image-model", false, true);
        let unsupported = resolved_model_for_test("text-model", false, false);
        let request = super::finalize_responses_request(
            payload(),
            &supported,
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

        assert!(matches!(
            super::finalize_responses_request(
                payload(),
                &unsupported,
                Some(&json!(false)),
                None,
                None,
                ResponsesEndpointKind::Standard,
            ),
            Err(ProviderError::InvalidRequest(_))
        ));
    }

    #[test]
    fn non_original_image_detail_modes_do_not_require_original_capability() {
        let resolved = resolved_model_for_test("text-model", false, false);
        for detail in ["auto", "low", "high"] {
            let request = super::finalize_responses_request(
                json!({
                    "model": "placeholder",
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
                &resolved,
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
    fn live_input_modalities_gate_only_present_content() {
        let mut text_only_row = live_model_row_for_test("text-only", false, false);
        text_only_row["input_modalities"] = json!(["text"]);
        let text_only = resolved_model_from_row_for_test(text_only_row);
        let image_payload = json!({
            "model": "placeholder",
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{
                    "type": "input_image",
                    "image_url": "data:image/png;base64,AAAA"
                }]
            }],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": false
        });
        assert!(matches!(
            super::finalize_responses_request(
                image_payload.clone(),
                &text_only,
                Some(&json!(false)),
                None,
                None,
                ResponsesEndpointKind::Standard,
            ),
            Err(ProviderError::InvalidRequest(_))
        ));

        let mut image_only_row = live_model_row_for_test("image-only", false, true);
        image_only_row["input_modalities"] = json!(["image"]);
        let image_only = resolved_model_from_row_for_test(image_only_row);
        assert!(super::finalize_responses_request(
            image_payload,
            &image_only,
            Some(&json!(false)),
            None,
            None,
            ResponsesEndpointKind::Standard,
        )
        .is_ok());
        assert!(matches!(
            super::finalize_responses_request(
                json!({
                    "model": "placeholder",
                    "input": [{
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}]
                    }],
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": false
                }),
                &image_only,
                Some(&json!(false)),
                None,
                None,
                ResponsesEndpointKind::Standard,
            ),
            Err(ProviderError::InvalidRequest(_))
        ));

        let mut audio_only_row = live_model_row_for_test("audio-only", false, false);
        audio_only_row["input_modalities"] = json!(["audio"]);
        let audio_only = resolved_model_from_row_for_test(audio_only_row);
        assert!(super::finalize_responses_request(
            json!({
                "model": "placeholder",
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_audio",
                        "audio_url": "data:audio/wav;base64,AAAA"
                    }]
                }],
                "tools": [],
                "tool_choice": "auto",
                "parallel_tool_calls": false
            }),
            &audio_only,
            Some(&json!(false)),
            None,
            None,
            ResponsesEndpointKind::Standard,
        )
        .is_ok());

        assert!(matches!(
            super::finalize_responses_request(
                json!({
                    "model": "placeholder",
                    "input": [{
                        "type": "function_call_output",
                        "call_id": "call-1",
                        "output": "result"
                    }],
                    "tools": [],
                    "tool_choice": "auto",
                    "parallel_tool_calls": false
                }),
                &audio_only,
                Some(&json!(false)),
                None,
                None,
                ResponsesEndpointKind::Standard,
            ),
            Err(ProviderError::InvalidRequest(_))
        ));
    }

    #[test]
    fn image_generation_requires_live_image_input_capability_before_transport() {
        let provider = ChatGPTOAuthProvider::new(
            "text-only".to_string(),
            "http://127.0.0.1:1".to_string(),
            None,
            None,
        )
        .unwrap();
        let mut row = live_model_row_for_test("text-only", false, false);
        row["input_modalities"] = json!(["text"]);
        let result = provider.generate_image_for_resolved_model_with_controls(
            "draw a square",
            &[],
            None,
            None,
            resolved_model_from_row_for_test(row),
            Some(&json!(false)),
            &GenerationControls::default(),
        );
        assert!(matches!(result, Err(ProviderError::InvalidRequest(_))));
    }

    #[test]
    fn test_finalizer_applies_catalog_reasoning_and_verbosity_defaults() {
        let resolved = resolved_model_for_test("live-lite-model", true, true);
        let payload = json!({
            "model": "placeholder",
            "instructions": "Use catalog defaults.",
            "input": [],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "include": [],
        });

        let request = super::finalize_responses_request(
            payload,
            &resolved,
            Some(&json!("auto")),
            None,
            None,
            ResponsesEndpointKind::Standard,
        )
        .unwrap();

        assert!(request.use_responses_lite);
        assert_eq!(request.payload["reasoning"]["effort"], json!("medium"));
        assert_eq!(request.payload["reasoning"]["context"], json!("all_turns"));
        assert_eq!(request.payload["text"], json!({"verbosity": "medium"}));
        assert_eq!(
            request.payload["include"],
            json!(["reasoning.encrypted_content"])
        );
    }

    #[test]
    fn requested_model_uses_the_exact_live_slug() {
        let snapshot = snapshot_from_rows_for_test(vec![
            live_model_row_for_test("gpt-5.6", false, true),
            live_model_row_for_test("gpt-5.6-sol", false, true),
        ]);
        let resolved =
            ChatGPTOAuthProvider::resolve_model_from_snapshot(snapshot, Some("gpt-5.6")).unwrap();
        assert_eq!(resolved.model.slug, "gpt-5.6");
    }

    #[test]
    fn opaque_live_slug_is_preserved_but_unusable_implicit_default_fails() {
        let snapshot = snapshot_from_rows_for_test(vec![live_model_row_for_test(" ", false, true)]);

        assert!(matches!(
            ChatGPTOAuthProvider::resolve_model_from_snapshot(snapshot.clone(), None),
            Err(ProviderError::CatalogUnavailable(_))
        ));
        let provider = ChatGPTOAuthProvider::new(
            String::new(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        )
        .unwrap();
        assert!(matches!(
            provider.resolve_model(Some(" ")),
            Err(ProviderError::InvalidRequest(_))
        ));
    }

    #[test]
    fn implicit_hidden_only_and_removed_configured_models_are_catalog_unavailable() {
        let mut hidden = live_model_row_for_test("hidden-model", false, true);
        hidden["visibility"] = json!("hide");
        let snapshot = snapshot_from_rows_for_test(vec![hidden]);
        assert_eq!(
            ChatGPTOAuthProvider::resolve_model_from_snapshot(
                snapshot.clone(),
                Some("hidden-model")
            )
            .unwrap()
            .model
            .slug,
            "hidden-model"
        );
        assert!(matches!(
            ChatGPTOAuthProvider::resolve_model_from_snapshot(snapshot.clone(), None),
            Err(ProviderError::CatalogUnavailable(_))
        ));

        let provider = ChatGPTOAuthProvider::new(
            "removed-model".to_string(),
            CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            None,
            None,
        )
        .unwrap();
        assert!(matches!(
            provider.configured_or_default_model_from_snapshot(snapshot.clone()),
            Err(ProviderError::CatalogUnavailable(_))
        ));
        assert!(matches!(
            ChatGPTOAuthProvider::resolve_model_from_snapshot(snapshot, Some("removed-model")),
            Err(ProviderError::ModelNotFound(_))
        ));
    }

    #[test]
    fn invalid_model_control_diagnostics_do_not_reflect_values() {
        let secret = "access-token-sentinel";
        let mut row = live_model_row_for_test(secret, false, false);
        row["service_tiers"] = json!([]);
        let resolved = resolved_model_from_row_for_test(row);

        let errors = [
            should_enable_parallel_tool_calls(Some(true), true).unwrap_err(),
            apply_model_capability_fields(
                &mut serde_json::Map::new(),
                &resolved.model,
                None,
                Some(secret),
            )
            .unwrap_err(),
            resolve_reasoning_effort(&resolved.model, secret)
                .unwrap_err()
                .to_string(),
            ResponseChainStore::new(1)
                .resolve(TEST_ACCOUNT_ID, secret)
                .unwrap_err()
                .to_string(),
        ];
        for error in errors {
            assert!(!error.contains(secret));
        }
    }

    #[test]
    fn catalog_refresh_account_switch_fails_before_snapshot_publication() {
        let cache = ModelCatalogCache::new(std::time::Duration::from_secs(60));
        let catalog_key = CatalogKey {
            account_id: "account-a".to_string(),
            base_url: "https://example.invalid".to_string(),
            client_version: "0.153.3".to_string(),
        };
        let result = cache.snapshot(catalog_key.clone(), || {
            ensure_catalog_account_unchanged("account-a", "account-b")?;
            Ok((
                serde_json::to_vec(&json!({"models": [live_model_row_for_test(
                    "model-b", false, true
                )]}))
                .unwrap(),
                Some("etag-b".to_string()),
            ))
        });
        assert!(matches!(result, Err(CatalogError::Auth(_))));
        let recovered = cache
            .snapshot(catalog_key, || {
                Ok((
                    serde_json::to_vec(&json!({"models": [live_model_row_for_test(
                        "model-a", false, true
                    )]}))
                    .unwrap(),
                    Some("etag-a".to_string()),
                ))
            })
            .unwrap();
        assert_eq!(recovered.models[0].slug, "model-a");
    }

    #[test]
    fn test_forced_lite_live_model_without_reasoning_does_not_invent_reasoning() {
        let resolved = resolved_model_from_row_for_test(json!({
            "slug": "live-no-reasoning-model",
            "display_name": "No reasoning model",
            "description": null,
            "default_reasoning_level": null,
            "supported_reasoning_levels": [],
            "visibility": "list",
            "supported_in_api": true,
            "priority": 0,
            "service_tiers": [],
            "default_service_tier": null,
            "support_verbosity": false,
            "default_verbosity": null,
            "supports_image_detail_original": false,
            "context_window": null,
            "max_context_window": null,
            "auto_compact_token_limit": null,
            "input_modalities": ["text"],
            "use_responses_lite": false
        }));
        let payload = json!({
            "model": "placeholder",
            "instructions": "No reasoning metadata exists.",
            "input": [],
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "include": [],
        });

        let request = super::finalize_responses_request(
            payload,
            &resolved,
            Some(&json!(true)),
            None,
            None,
            ResponsesEndpointKind::Standard,
        )
        .unwrap();

        assert!(request.use_responses_lite);
        assert_eq!(
            request.payload["reasoning"],
            json!({"summary": "auto", "context": "all_turns"})
        );
        assert_eq!(
            request.payload["include"],
            json!(["reasoning.encrypted_content"])
        );
    }

    #[test]
    fn test_compact_finalizer_omits_include_and_uses_lite_shape() {
        let mut payload = json!({
            "model": "gpt-5.6-sol",
            "instructions": "Compact the conversation.",
            "input": [],
            "tools": [],
            "tool_choice": "required",
            "parallel_tool_calls": false,
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
            json!({"effort": "low", "summary": "auto", "context": "all_turns"})
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
        let items = vec![json!({"type": "message", "role": "assistant", "content": [
            {"type": "output_text", "text": "Hello "},
            {"type": "output_text", "text": "world"}
        ]})];
        assert_eq!(text_from_response_items(&items).unwrap(), "Hello world");
    }

    #[test]
    fn test_text_from_response_items_empty() {
        let items: Vec<Value> = vec![];
        assert_eq!(text_from_response_items(&items).unwrap(), "");
    }

    #[test]
    fn text_from_response_items_rejects_unrepresentable_message_content() {
        for item in [
            json!({"type": "message", "role": "user", "content": []}),
            json!({"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "no"}]}),
            json!({"type": "message", "role": "assistant", "content": [{"type": "output_text"}]}),
        ] {
            assert!(matches!(
                text_from_response_items(&[item]),
                Err(ProviderError::UpstreamProtocol(_))
            ));
        }
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
        assert_eq!(usage.cached_tokens, Some(30));
    }

    #[test]
    fn test_usage_from_response_rejects_public_alias_keys() {
        let val = json!({
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "total_tokens": 300,
            "prompt_tokens_details": {"cached_tokens": 50}
        });
        assert!(usage_from_response(&val).is_none());
        assert!(matches!(
            parse_usage(&val),
            Err(ProviderError::UpstreamProtocol(_))
        ));
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
    fn sse_line_decode_errors_are_protocol_failures_but_io_errors_are_transport_failures() {
        let secret = "access-token-sentinel";
        let invalid_text = classify_sse_line_read_error(
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid UTF-8 near {secret}"),
            ),
            &[secret],
        );
        assert!(matches!(
            invalid_text,
            ProviderError::UpstreamProtocol(ref message)
                if message.contains("***") && !message.contains(secret)
        ));

        let read_failure = classify_sse_line_read_error(
            std::io::Error::new(std::io::ErrorKind::ConnectionReset, "connection reset"),
            &[],
        );
        assert!(matches!(
            read_failure,
            ProviderError::UpstreamTransport(ref message)
                if message.contains("connection reset")
        ));
    }

    #[test]
    fn upstream_protocol_diagnostics_do_not_reflect_unknown_values() {
        let secret = "access-token-sentinel";
        let errors = [
            decode_sse_block(&[format!("data: {secret}")]).unwrap_err(),
            validate_response_event(&json!({
                "type": "response.output_item.done",
                "item": {"type": secret}
            }))
            .unwrap_err(),
            validate_response_event(&json!({
                "type": "response.output_item.done",
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": secret, "text": "ignored"}]
                }
            }))
            .unwrap_err(),
            text_from_response_items(&[json!({"type": secret})]).unwrap_err(),
            filter_compacted_history_items(&[json!({"type": secret})]).unwrap_err(),
            filter_compacted_history_items(&[json!({
                "type": "message",
                "role": secret,
                "content": []
            })])
            .unwrap_err(),
        ];
        for error in errors {
            assert!(!error.to_string().contains(secret));
        }
    }

    #[test]
    fn added_reasoning_items_accept_pinned_typed_content() {
        let official = json!({
            "type": "response.output_item.added",
            "item": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "summary"}],
                "content": [{"type": "text", "text": "raw"}]
            }
        });
        assert!(validate_response_event(&official).is_ok());

        let mut text_variant = official.clone();
        text_variant["item"]["content"][0]["type"] = json!("text");
        assert!(validate_response_event(&text_variant).is_ok());

        let mut unsupported = official;
        unsupported["item"]["content"][0]["type"] = json!("future");
        assert!(matches!(
            validate_response_event(&unsupported),
            Err(ProviderError::UpstreamProtocol(_))
        ));
    }

    #[test]
    fn response_events_require_output_item_objects_and_completed_ids() {
        assert!(validate_response_event(&json!({
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "output": ["ignored by the private completed event contract"]
            }
        }))
        .is_ok());
        assert!(validate_response_event(&json!({
            "type": "response.output_item.done",
            "item": {"type": "message", "role": "assistant", "content": []}
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
    fn consumed_lifecycle_events_require_the_official_fields() {
        for event in [
            json!({"type": "response.metadata", "metadata": {"trace": "ignored"}}),
            json!({"type": "codex.response.metadata", "headers": {}}),
            json!({"type": "responsesapi.websocket_timing", "elapsed_ms": 1}),
            json!({"type": "response.created", "response": {}}),
            json!({
                "type": "response.output_item.added",
                "item": {"type": "message", "role": "assistant", "content": []}
            }),
            json!({
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "apply_patch",
                    "arguments": ""
                }
            }),
            json!({
                "type": "response.output_item.added",
                "item": {"type": "web_search_call", "id": "search-1", "status": "searching"}
            }),
            json!({"type": "response.output_text.delta", "delta": ""}),
            json!({
                "type": "response.reasoning_summary_text.delta",
                "delta": "",
                "summary_index": 0
            }),
            json!({
                "type": "response.reasoning_summary_text.done",
                "item_id": "",
                "text": "",
                "summary_index": 0
            }),
            json!({
                "type": "response.reasoning_text.delta",
                "delta": "",
                "content_index": 0
            }),
            json!({
                "type": "response.reasoning_summary_part.added",
                "summary_index": 0
            }),
        ] {
            assert!(validate_response_event(&event).is_ok(), "{event}");
        }

        for event in [
            json!({"type": "response.created"}),
            json!({"type": "response.output_item.added", "item": null}),
            json!({"type": "response.custom_tool_call_input.delta", "delta": ""}),
            json!({
                "type": "response.output_item.added",
                "item": {"type": "custom_tool_call", "call_id": "call-1", "name": "apply_patch"}
            }),
            json!({
                "type": "response.reasoning_summary_text.delta",
                "delta": "summary"
            }),
            json!({
                "type": "response.reasoning_summary_text.done",
                "item_id": "reasoning-1",
                "text": "summary"
            }),
            json!({
                "type": "response.reasoning_text.delta",
                "delta": "raw",
                "summary_index": 0
            }),
            json!({
                "type": "response.reasoning_summary_part.added",
                "part_index": 0
            }),
        ] {
            assert!(validate_response_event(&event).is_err(), "{event}");
        }

        assert!(validate_response_event(&json!({
            "type": "response.completed",
            "response": {"id": "response-1", "end_turn": "true"}
        }))
        .is_err());
        for response in [
            json!({"id": "response-1"}),
            json!({"id": "response-1", "end_turn": false}),
            json!({"id": "response-1", "end_turn": true}),
        ] {
            assert!(validate_response_event(&json!({
                "type": "response.completed",
                "response": response
            }))
            .is_ok());
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
        let tc = tool_call_from_response_item(&item).unwrap().unwrap();
        assert_eq!(tc.name, "read_file");
        assert_eq!(tc.id, "call-1");
        assert_eq!(tc.arguments, "{\"path\": \"/tmp/test\"}");
    }

    #[test]
    fn function_calls_ignore_additive_input_fields_and_custom_calls_fail() {
        let function = tool_call_from_response_item(&json!({
            "type": "function_call",
            "call_id": "call-1",
            "name": "read_file",
            "arguments": "{\"path\":\"/tmp/test\"}",
            "input": "additive"
        }))
        .unwrap()
        .unwrap();
        assert_eq!(function.arguments, "{\"path\":\"/tmp/test\"}");

        let custom_error = tool_call_from_response_item(&json!({
            "type": "custom_tool_call",
            "call_id": "call-2",
            "name": "apply_patch",
            "input": "{\"patch\":\"content\"}",
            "arguments": "additive"
        }))
        .unwrap_err();
        assert!(matches!(
            custom_error,
            ProviderError::UpstreamProtocol(message) if message.contains("custom_tool_call")
        ));
    }

    #[test]
    fn test_tool_call_from_response_item_not_function() {
        let item = json!({"type": "message"});
        assert!(tool_call_from_response_item(&item).unwrap().is_none());
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
        let result = validate_image_content_values(&[json!({
            "image_url": "data:image/png;base64,abc",
            "detail": "high"
        })])
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "input_image");
        assert_eq!(result[0]["detail"], "high");
    }

    #[test]
    fn test_validate_image_content_items_missing_url() {
        let result = validate_image_content_values(&[json!({})]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_image_content_items_wrong_prefix() {
        let result = validate_image_content_values(&[json!({
            "image_url": "https://example.com/img.png"
        })]);
        assert!(result.is_err());
    }

    #[test]
    fn reasoning_modes_and_non_object_existing_payloads_fail_loudly() {
        for mode in ["standard", "pro", "future"] {
            let mut payload = json!({"reasoning": {"summary": "auto"}});
            assert!(matches!(
                set_reasoning_payload_with_options(
                    payload.as_object_mut().unwrap(),
                    None,
                    Some(&json!({"mode": mode})),
                ),
                Err(ProviderError::InvalidRequest(_))
            ));
        }

        for existing in [json!(null), json!("medium"), json!(42)] {
            let mut payload = json!({"reasoning": existing});
            assert!(matches!(
                set_reasoning_payload_with_options(payload.as_object_mut().unwrap(), None, None),
                Err(ProviderError::InvalidRequest(_))
            ));
        }

        let mut payload = json!({"reasoning": {"mode": "standard"}});
        assert!(matches!(
            set_reasoning_payload_with_options(payload.as_object_mut().unwrap(), None, None),
            Err(ProviderError::InvalidRequest(_))
        ));

        let mut payload = json!({"reasoning": {"summary": "auto"}});
        set_reasoning_payload_with_options(
            payload.as_object_mut().unwrap(),
            Some("high"),
            Some(&json!({"mode": null, "context": "current_turn"})),
        )
        .unwrap();
        assert_eq!(
            payload["reasoning"],
            json!({"summary": "auto", "effort": "high", "context": "current_turn"})
        );
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
            "instructions": "test",
            "input": [],
            "tools": [],
            "reasoning": {"context": "current_turn"}
        });
        let error = apply_responses_lite_payload(payload.as_object_mut().unwrap()).unwrap_err();
        assert!(matches!(error, ProviderError::InvalidRequest(_)));
    }

    #[test]
    fn explicit_cache_breakpoint_is_rejected_before_lite_rewrite() {
        let images = vec![json!({
            "image_url": "data:image/png;base64,AAAA",
            "detail": "original",
            "prompt_cache_breakpoint": {"mode": "explicit"}
        })];
        assert!(matches!(
            validate_image_content_values(&images),
            Err(ProviderError::InvalidRequest(_))
        ));
    }

    #[test]
    fn generation_controls_reject_standard_mode_for_every_model() {
        let controls = GenerationControls {
            reasoning: Some(json!({"mode": "standard"})),
            verbosity: Some("high".to_string()),
            ..GenerationControls::default()
        };
        for model in ["live-model-a", "live-model-b"] {
            let mut payload = json!({});
            assert!(matches!(
                apply_generation_controls(payload.as_object_mut().unwrap(), model, &controls),
                Err(ProviderError::InvalidRequest(_))
            ));
        }
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
        let message = Message {
            role: MessageRole::User,
            content: "cache here".to_string(),
            tool_calls: vec![],
            tool_call_id: None,
            name: None,
            reasoning_content: None,
            images: vec![],
            structured_content: Some(vec![json!({
                "type": "input_text",
                "text": "cache here",
                "prompt_cache_breakpoint": {"mode": "explicit"}
            })]),
        };

        assert!(matches!(
            messages_to_response_items(&[message]),
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
    fn structured_message_cache_breakpoint_is_rejected() {
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
        assert!(matches!(
            messages_to_response_items(&[message]),
            Err(ProviderError::InvalidRequest(_))
        ));
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
            "total_tokens": 12,
            "input_tokens_details": {
                "cached_tokens": 3,
                "cache_write_tokens": 7
            }
        }))
        .unwrap();
        assert_eq!(usage.cached_tokens, Some(3));
        assert_eq!(usage.cache_write_tokens, Some(7));
    }

    #[test]
    fn usage_ignores_additive_telemetry_without_inventing_cache_counts() {
        let usage = parse_usage(&json!({
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "codex_rollout_budget_units": 4.5,
            "input_tokens_details": null
        }))
        .unwrap();
        assert_eq!(usage.cached_tokens, None);
        assert_eq!(usage.cache_write_tokens, None);

        let usage = parse_usage(&json!({
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "input_tokens_details": {
                "cached_tokens": 3,
                "future_telemetry": {"value": 1}
            }
        }))
        .unwrap();
        assert_eq!(usage.cached_tokens, Some(3));
        assert_eq!(usage.cache_write_tokens, None);
    }

    #[test]
    fn usage_rejects_public_alias_fields_in_private_upstream_payloads() {
        for alias in [
            "prompt_tokens",
            "completion_tokens",
            "prompt_tokens_details",
            "cached_input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ] {
            let mut value = json!({
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12
            });
            value
                .as_object_mut()
                .unwrap()
                .insert(alias.to_string(), json!(0));
            assert!(matches!(
                parse_usage(&value),
                Err(ProviderError::UpstreamProtocol(_))
            ));
        }
    }

    #[test]
    fn compact_raw_events_preserves_identical_web_search_calls_by_position() {
        let duplicate = json!({
            "type": "web_search_call",
            "id": "web-1",
            "input": {"query": "same"},
            "content": []
        });
        let mut events = vec![duplicate.clone(), duplicate];
        events.extend((0..25).map(|index| json!({"type": "content", "text": index.to_string()})));

        let compacted = compact_raw_events(&events);
        assert_eq!(
            compacted
                .iter()
                .filter(|event| event.get("type") == Some(&json!("web_search_call")))
                .count(),
            2
        );
        assert_eq!(compacted.len(), 22);
    }
}
