use crate::model_catalog::ModelInfo;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::OnceLock;

pub const RESPONSES_LITE_ENV: &str = "CODEX_AS_API_RESPONSES_LITE";
pub const CODEX_METADATA_ENV: &str = "CODEX_AS_API_CODEX_METADATA";
pub const LITE_HEADER_NAME: &str = "x-openai-internal-codex-responses-lite";
pub const LITE_HEADER_VALUE: &str = "true";
pub const TURN_METADATA_KEY: &str = "x-codex-turn-metadata";
pub const INSTALLATION_ID_KEY: &str = "x-codex-installation-id";
pub const WINDOW_ID_KEY: &str = "x-codex-window-id";
pub const SESSION_ID_KEY: &str = "session_id";
pub const THREAD_ID_KEY: &str = "thread_id";
pub const TURN_ID_KEY: &str = "turn_id";

static WINDOW_ID: OnceLock<String> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponsesLiteMode {
    Off,
    On,
    Auto,
}

pub fn use_responses_lite(model: &ModelInfo, value: Option<&Value>) -> Result<bool, String> {
    let mode = resolve_responses_lite_mode(value)?;
    Ok(match mode {
        ResponsesLiteMode::On => true,
        ResponsesLiteMode::Off => false,
        ResponsesLiteMode::Auto => model.use_responses_lite,
    })
}

pub fn should_enable_parallel_tool_calls(
    requested: Option<bool>,
    responses_lite: bool,
) -> Result<bool, String> {
    if requested != Some(true) {
        return Ok(false);
    }
    if responses_lite {
        return Err("parallel_tool_calls=true is not supported by Responses Lite".to_string());
    }
    Ok(true)
}

pub fn resolve_codex_metadata_enabled(value: Option<bool>) -> Result<bool, String> {
    if let Some(v) = value {
        return Ok(v);
    }
    let raw = environment_value(CODEX_METADATA_ENV)?.unwrap_or_else(|| "off".to_string());
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err("codex_metadata must be on or off".to_string()),
    }
}

pub fn validate_model_capability_environment() -> Result<(), String> {
    resolve_codex_metadata_enabled(None)?;
    resolve_responses_lite_mode(None)?;
    Ok(())
}

pub fn apply_model_capability_fields(
    payload: &mut serde_json::Map<String, Value>,
    model: &ModelInfo,
    text: Option<&Value>,
    service_tier: Option<&str>,
) -> Result<(), String> {
    let mut text = match text {
        None | Some(Value::Null) => None,
        Some(Value::Object(object)) => Some(object.clone()),
        Some(_) => return Err("text must be an object when provided".to_string()),
    };
    if let Some(object) = text.as_mut() {
        if object.get("verbosity").is_some_and(Value::is_null) {
            object.remove("verbosity");
        }
        if let Some(verbosity) = object.get("verbosity") {
            if !model.support_verbosity {
                return Err("text.verbosity is not supported for the requested model".to_string());
            }
            if !matches!(verbosity.as_str(), Some("low" | "medium" | "high")) {
                return Err("text.verbosity must be one of: low, medium, high".to_string());
            }
        }
    }

    if model.support_verbosity {
        let mut merged = text.unwrap_or_default();
        if !merged.contains_key("verbosity") {
            if let Some(default_verbosity) = model.default_verbosity.as_ref() {
                merged.insert(
                    "verbosity".to_string(),
                    Value::String(default_verbosity.clone()),
                );
            }
        }
        if !merged.is_empty() {
            payload.insert("text".to_string(), Value::Object(merged));
        }
    } else if let Some(text) = text {
        payload.insert("text".to_string(), Value::Object(text));
    }

    if let Some(requested_tier) = service_tier {
        let wire_tier = match requested_tier {
            "default" => None,
            "fast" => Some("priority"),
            value => Some(value),
        };
        if let Some(wire_tier) = wire_tier {
            if !model
                .service_tiers
                .iter()
                .any(|value| value.id == wire_tier)
            {
                return Err("service_tier is not supported for the requested model".to_string());
            }
            payload.insert(
                "service_tier".to_string(),
                Value::String(wire_tier.to_string()),
            );
        }
    }
    Ok(())
}

pub fn build_codex_client_metadata(
    auth_json_path: Option<&str>,
    existing: Option<&HashMap<String, String>>,
) -> Result<HashMap<String, String>, String> {
    let mut metadata = existing.cloned().unwrap_or_default();
    let session_id = metadata
        .get(SESSION_ID_KEY)
        .filter(|value| !value.trim().is_empty())
        .cloned()
        .ok_or_else(|| {
            "codex_metadata requires a non-empty client_metadata.session_id".to_string()
        })?;
    let thread_id = match metadata.get(THREAD_ID_KEY) {
        Some(value) if value.trim().is_empty() => {
            return Err(
                "client_metadata.thread_id must be a non-empty string when provided".to_string(),
            );
        }
        Some(value) => value.clone(),
        None => session_id.clone(),
    };
    let expanded =
        crate::auth::resolve_auth_path(auth_json_path).map_err(|error| error.to_string())?;
    let absolute = if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir()
            .map_err(|error| format!("failed to resolve current directory: {error}"))?
            .join(expanded)
    };
    let namespace = uuid::Uuid::parse_str("d2c81270-8f15-5e8d-a5c4-4cdbf2c21fd0")
        .expect("constant installation namespace must be a UUID");
    let installation_id = uuid::Uuid::new_v5(
        &namespace,
        format!("codex-as-api:{}", absolute.display()).as_bytes(),
    )
    .to_string();
    let window_id = WINDOW_ID
        .get_or_init(|| uuid::Uuid::new_v4().to_string())
        .clone();
    let turn_id = uuid::Uuid::new_v4().to_string();
    let turn_metadata = json!({
        "installation_id": installation_id,
        "session_id": session_id,
        "thread_id": thread_id,
        "turn_id": turn_id,
        "window_id": window_id,
        "source": "codex-as-api",
    });
    metadata.insert(INSTALLATION_ID_KEY.to_string(), installation_id);
    metadata.insert(SESSION_ID_KEY.to_string(), session_id);
    metadata.insert(THREAD_ID_KEY.to_string(), thread_id);
    metadata.insert(TURN_ID_KEY.to_string(), turn_id);
    metadata.insert(WINDOW_ID_KEY.to_string(), window_id);
    metadata.insert(TURN_METADATA_KEY.to_string(), turn_metadata.to_string());
    Ok(metadata)
}

fn resolve_responses_lite_mode(value: Option<&Value>) -> Result<ResponsesLiteMode, String> {
    let raw = match value {
        Some(Value::Bool(v)) => {
            return Ok(if *v {
                ResponsesLiteMode::On
            } else {
                ResponsesLiteMode::Off
            });
        }
        Some(Value::String(value)) => value.clone(),
        Some(_) => return Err("responses_lite must be one of: off, on, auto".to_string()),
        None => environment_value(RESPONSES_LITE_ENV)?.unwrap_or_else(|| "auto".to_string()),
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(ResponsesLiteMode::On),
        "false" | "0" | "no" | "off" => Ok(ResponsesLiteMode::Off),
        "auto" => Ok(ResponsesLiteMode::Auto),
        _ => Err("responses_lite must be one of: off, on, auto".to_string()),
    }
}

fn environment_value(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(value) if value.trim().is_empty() => {
            Err(format!("{name} must not be empty when provided"))
        }
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!("{name} must contain valid Unicode")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_catalog::parse_models_response;

    fn parsed_live_model() -> std::sync::Arc<ModelInfo> {
        let body = json!({"models":[{
            "slug":"live-model","display_name":"Live","description":null,
            "default_reasoning_level":"medium",
            "supported_reasoning_levels":[{"effort":"medium","description":"Medium"}],
            "visibility":"list","supported_in_api":true,"priority":1,
            "service_tiers":[{"id":"priority","name":"Fast","description":"Fast"}],
            "default_service_tier":null,"support_verbosity":true,"default_verbosity":"low",
            "supports_image_detail_original":true,
            "context_window":100000,"max_context_window":100000,
            "auto_compact_token_limit":90000,"effective_context_window_percent":95,
            "input_modalities":["text","image"],"use_responses_lite":false
        }]});
        parse_models_response(&serde_json::to_vec(&body).unwrap())
            .unwrap()
            .remove(0)
    }

    fn live_model(use_responses_lite: bool, support_verbosity: bool) -> ModelInfo {
        ModelInfo {
            slug: "live-model".to_string(),
            display_name: "Live Model".to_string(),
            description: Some("test fixture".to_string()),
            default_reasoning_level: Some("medium".to_string()),
            supported_reasoning_levels: vec![crate::model_catalog::ReasoningLevel {
                effort: "medium".to_string(),
                description: "Medium".to_string(),
            }],
            multi_agent_reasoning_effort: None,
            visibility: "list".to_string(),
            supported_in_api: true,
            priority: 0,
            service_tiers: vec![crate::model_catalog::ServiceTier {
                id: "priority".to_string(),
                name: "Priority".to_string(),
                description: "Priority".to_string(),
            }],
            default_service_tier: None,
            support_verbosity,
            default_verbosity: support_verbosity.then(|| "low".to_string()),
            supports_reasoning_summary_parameter: true,
            default_reasoning_summary: "auto".to_string(),
            comp_hash: None,
            supports_image_detail_original: true,
            context_window: Some(100_000),
            max_context_window: Some(120_000),
            auto_compact_token_limit: Some(90_000),
            effective_context_window_percent: 95,
            input_modalities: vec!["text".to_string(), "image".to_string()],
            use_responses_lite,
        }
    }

    #[test]
    fn live_model_controls_are_applied_without_static_fallback() {
        let model = parsed_live_model();
        let mut payload = serde_json::Map::new();
        apply_model_capability_fields(&mut payload, &model, None, Some("fast")).unwrap();
        assert_eq!(payload["service_tier"], "priority");
        assert_eq!(payload["text"]["verbosity"], "low");
        assert!(should_enable_parallel_tool_calls(Some(true), false).unwrap());
    }

    #[test]
    fn explicit_invalid_responses_lite_type_fails() {
        assert!(use_responses_lite(&live_model(false, true), Some(&json!(42))).is_err());
    }

    #[test]
    fn responses_lite_auto_uses_the_live_capability() {
        assert!(use_responses_lite(&live_model(true, true), Some(&json!("auto")),).unwrap());
        assert!(!use_responses_lite(&live_model(false, true), Some(&json!("auto")),).unwrap());
    }

    #[test]
    fn classic_responses_preserve_explicit_parallel_tools() {
        assert!(should_enable_parallel_tool_calls(Some(true), false).unwrap());
    }

    #[test]
    fn omitted_or_false_parallel_tools_stay_disabled() {
        assert!(!should_enable_parallel_tool_calls(None, false).unwrap());
        assert!(!should_enable_parallel_tool_calls(Some(false), false).unwrap());
    }

    #[test]
    fn responses_lite_rejects_parallel_tools() {
        assert!(should_enable_parallel_tool_calls(Some(true), true).is_err());
    }

    #[test]
    fn service_tier_fast_maps_to_live_priority() {
        let model = live_model(false, true);
        let mut payload = serde_json::Map::new();
        apply_model_capability_fields(&mut payload, &model, None, Some("fast")).unwrap();
        assert_eq!(payload.get("service_tier"), Some(&json!("priority")));
    }

    #[test]
    fn default_service_tier_is_omitted() {
        let model = live_model(false, true);
        let mut payload = serde_json::Map::new();
        apply_model_capability_fields(&mut payload, &model, None, Some("default")).unwrap();
        assert!(!payload.contains_key("service_tier"));
    }

    #[test]
    fn service_tier_not_in_live_catalog_is_rejected() {
        let model = live_model(false, true);
        let mut payload = serde_json::Map::new();
        assert!(apply_model_capability_fields(&mut payload, &model, None, Some("flex")).is_err());
        assert!(!payload.contains_key("service_tier"));
    }

    #[test]
    fn live_default_verbosity_is_applied() {
        let model = live_model(false, true);
        let mut payload = serde_json::Map::new();
        apply_model_capability_fields(&mut payload, &model, None, None).unwrap();
        assert_eq!(payload["text"]["verbosity"], "low");
    }

    #[test]
    fn unsupported_explicit_verbosity_is_rejected() {
        let model = live_model(false, false);
        let mut payload = serde_json::Map::new();
        assert!(apply_model_capability_fields(
            &mut payload,
            &model,
            Some(&json!({"verbosity": "high"})),
            None,
        )
        .is_err());
    }

    #[test]
    fn models_without_verbosity_support_preserve_other_text_controls() {
        let model = live_model(false, false);
        let mut payload = serde_json::Map::new();
        apply_model_capability_fields(
            &mut payload,
            &model,
            Some(&json!({
                "format": {"type": "json_object"},
                "verbosity": null
            })),
            None,
        )
        .unwrap();
        assert_eq!(payload["text"], json!({"format": {"type": "json_object"}}));
    }

    #[test]
    fn null_verbosity_is_omitted_before_applying_a_live_default() {
        let model = live_model(false, true);
        let mut payload = serde_json::Map::new();
        apply_model_capability_fields(
            &mut payload,
            &model,
            Some(&json!({
                "format": {"type": "text"},
                "verbosity": null
            })),
            None,
        )
        .unwrap();
        assert_eq!(
            payload["text"],
            json!({"format": {"type": "text"}, "verbosity": "low"})
        );
    }

    #[test]
    fn non_object_text_controls_are_rejected() {
        let model = live_model(false, true);
        let mut payload = serde_json::Map::new();
        assert!(
            apply_model_capability_fields(&mut payload, &model, Some(&json!("high")), None,)
                .is_err()
        );
    }

    #[test]
    fn codex_metadata_requires_explicit_session_identity() {
        assert!(build_codex_client_metadata(None, None).is_err());
        let metadata = HashMap::from([(SESSION_ID_KEY.to_string(), "   ".to_string())]);
        assert!(build_codex_client_metadata(None, Some(&metadata)).is_err());
    }

    #[test]
    fn codex_metadata_preserves_session_and_thread_identity() {
        let existing = HashMap::from([
            (SESSION_ID_KEY.to_string(), "session-root".to_string()),
            (THREAD_ID_KEY.to_string(), "thread-child".to_string()),
            ("custom".to_string(), "preserved".to_string()),
        ]);
        let first =
            build_codex_client_metadata(Some("/tmp/codex-auth.json"), Some(&existing)).unwrap();
        let second =
            build_codex_client_metadata(Some("/tmp/codex-auth.json"), Some(&existing)).unwrap();
        assert_eq!(first[SESSION_ID_KEY], "session-root");
        assert_eq!(first[THREAD_ID_KEY], "thread-child");
        assert_eq!(first["custom"], "preserved");
        assert_eq!(first[INSTALLATION_ID_KEY], second[INSTALLATION_ID_KEY]);
        assert_eq!(first[WINDOW_ID_KEY], second[WINDOW_ID_KEY]);
        assert_ne!(first[TURN_ID_KEY], second[TURN_ID_KEY]);
    }
}
