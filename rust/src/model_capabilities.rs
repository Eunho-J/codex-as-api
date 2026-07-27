use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
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

static CAPABILITIES_JSON: &str = include_str!("model-capabilities.json");
static WINDOW_ID: OnceLock<String> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct ModelCapability {
    pub use_responses_lite: bool,
    pub supports_parallel_tool_calls: bool,
    pub supports_image_detail_original: bool,
    pub support_verbosity: bool,
    pub default_verbosity: Option<String>,
    pub default_reasoning_effort: Option<String>,
    pub context_window: Option<i64>,
    pub max_context_window: Option<i64>,
    pub service_tiers: Vec<String>,
    #[allow(dead_code)]
    pub default_service_tier: Option<String>,
    #[allow(dead_code)]
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResponsesLiteMode {
    Off,
    On,
    Auto,
}

pub fn capability_for_model(model: &str) -> ModelCapability {
    let parsed: Value = serde_json::from_str(CAPABILITIES_JSON)
        .expect("embedded model-capabilities.json must be valid");
    parsed
        .get("models")
        .and_then(|v| v.as_object())
        .and_then(|models| models.get(model))
        .and_then(capability_from_value)
        .unwrap_or_else(unknown_capability)
}

pub fn use_responses_lite(model: &str, value: Option<&Value>) -> Result<bool, String> {
    let mode = resolve_responses_lite_mode(value)?;
    Ok(match mode {
        ResponsesLiteMode::On => true,
        ResponsesLiteMode::Off => false,
        ResponsesLiteMode::Auto => capability_for_model(model).use_responses_lite,
    })
}

pub fn should_enable_parallel_tool_calls(
    model: &str,
    requested: Option<bool>,
    responses_lite: bool,
) -> bool {
    if responses_lite || requested != Some(true) {
        return false;
    }
    capability_for_model(model).supports_parallel_tool_calls
}

pub fn resolve_codex_metadata_enabled(value: Option<bool>) -> Result<bool, String> {
    if let Some(v) = value {
        return Ok(v);
    }
    let raw = std::env::var(CODEX_METADATA_ENV).unwrap_or_else(|_| "off".to_string());
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" | "" => Ok(false),
        _ => Err("codex_metadata must be on or off".to_string()),
    }
}

pub fn apply_model_capability_fields(
    payload: &mut serde_json::Map<String, Value>,
    model: &str,
    text: Option<&Value>,
    service_tier: Option<&str>,
) -> Result<(), String> {
    let capability = capability_for_model(model);
    if capability.support_verbosity {
        let mut merged = text
            .and_then(|v| v.as_object())
            .cloned()
            .unwrap_or_default();
        if !merged.contains_key("verbosity") {
            if let Some(default_verbosity) = capability.default_verbosity {
                merged.insert("verbosity".to_string(), Value::String(default_verbosity));
            }
        }
        if !merged.is_empty() {
            payload.insert("text".to_string(), Value::Object(merged));
        }
    } else if let Some(t) = text {
        payload.insert("text".to_string(), t.clone());
    }

    if let Some(requested_tier) = service_tier {
        let wire_tier = match requested_tier {
            "default" => None,
            "fast" => Some("priority"),
            value => Some(value),
        };
        if let Some(wire_tier) = wire_tier {
            if !capability
                .service_tiers
                .iter()
                .any(|value| value == wire_tier)
            {
                return Err(format!(
                    "service_tier {requested_tier:?} is not supported for model {model}"
                ));
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
    let raw_path = auth_json_path.unwrap_or("~/.codex/auth.json");
    let expanded = if let Some(rest) = raw_path.strip_prefix("~/") {
        std::env::var("HOME")
            .map(|home| PathBuf::from(home).join(rest))
            .unwrap_or_else(|_| PathBuf::from(raw_path))
    } else {
        PathBuf::from(raw_path)
    };
    let absolute = if expanded.is_absolute() {
        expanded
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(expanded)
    };
    let namespace = uuid::Uuid::parse_str("d2c81270-8f15-5e8d-a5c4-4cdbf2c21fd0").unwrap();
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

pub fn strip_image_detail_fields(value: Value) -> Value {
    match value {
        Value::Array(items) => {
            Value::Array(items.into_iter().map(strip_image_detail_fields).collect())
        }
        Value::Object(map) => {
            let is_image = map.get("type").and_then(|v| v.as_str()) == Some("input_image");
            Value::Object(
                map.into_iter()
                    .filter_map(|(key, child)| {
                        if is_image && key == "detail" {
                            None
                        } else {
                            Some((key, strip_image_detail_fields(child)))
                        }
                    })
                    .collect(),
            )
        }
        other => other,
    }
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
        None => std::env::var(RESPONSES_LITE_ENV).unwrap_or_else(|_| "auto".to_string()),
    };
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(ResponsesLiteMode::On),
        "false" | "0" | "no" | "off" => Ok(ResponsesLiteMode::Off),
        "auto" => Ok(ResponsesLiteMode::Auto),
        _ => Err("responses_lite must be one of: off, on, auto".to_string()),
    }
}

fn capability_from_value(value: &Value) -> Option<ModelCapability> {
    let obj = value.as_object()?;
    let tiers = obj
        .get("service_tiers")
        .and_then(|v| v.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();
    Some(ModelCapability {
        use_responses_lite: obj
            .get("use_responses_lite")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        supports_parallel_tool_calls: obj
            .get("supports_parallel_tool_calls")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        supports_image_detail_original: obj
            .get("supports_image_detail_original")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        support_verbosity: obj
            .get("support_verbosity")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        default_verbosity: obj
            .get("default_verbosity")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        default_reasoning_effort: obj
            .get("default_reasoning_effort")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        context_window: obj.get("context_window").and_then(|v| v.as_i64()),
        max_context_window: obj.get("max_context_window").and_then(|v| v.as_i64()),
        service_tiers: tiers,
        default_service_tier: obj
            .get("default_service_tier")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
        source: obj
            .get("source")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string(),
    })
}

fn unknown_capability() -> ModelCapability {
    ModelCapability {
        use_responses_lite: false,
        supports_parallel_tool_calls: false,
        supports_image_detail_original: false,
        support_verbosity: false,
        default_verbosity: None,
        default_reasoning_effort: None,
        context_window: None,
        max_context_window: None,
        service_tiers: vec![],
        default_service_tier: None,
        source: "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_capability_catalog_is_structurally_valid() {
        let catalog: Value = serde_json::from_str(CAPABILITIES_JSON).unwrap();
        let models = catalog.get("models").and_then(Value::as_object).unwrap();

        assert_eq!(models.len(), 10);
        for model in ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"] {
            assert!(models.get(model).is_some_and(Value::is_object));
        }
    }

    #[test]
    fn parses_gpt_5_6_model_capabilities() {
        let cases = [
            ("gpt-5.6", "low"),
            ("gpt-5.6-sol", "low"),
            ("gpt-5.6-terra", "medium"),
            ("gpt-5.6-luna", "medium"),
        ];

        for (model, default_effort) in cases {
            let capability = capability_for_model(model);

            assert!(capability.use_responses_lite);
            assert!(capability.supports_parallel_tool_calls);
            assert!(capability.support_verbosity);
            assert_eq!(capability.default_verbosity.as_deref(), Some("low"));
            assert_eq!(
                capability.default_reasoning_effort.as_deref(),
                Some(default_effort)
            );
            assert_eq!(capability.context_window, Some(372_000));
            assert_eq!(capability.max_context_window, Some(372_000));
            assert_eq!(capability.service_tiers, vec!["priority".to_string()]);
        }
    }

    #[test]
    fn unknown_model_has_no_catalog_reasoning_or_context_defaults() {
        let capability = capability_for_model("unknown-model");

        assert_eq!(capability.default_reasoning_effort, None);
        assert_eq!(capability.context_window, None);
        assert!(!capability.supports_image_detail_original);
    }

    #[test]
    fn original_image_detail_support_matches_the_official_catalog() {
        for model in [
            "gpt-5.6",
            "gpt-5.6-sol",
            "gpt-5.6-terra",
            "gpt-5.6-luna",
            "gpt-5.5",
            "gpt-5.4",
            "gpt-5.4-mini",
        ] {
            assert!(capability_for_model(model).supports_image_detail_original);
        }
        for model in [
            "gpt-5.2",
            "gpt-5.3-codex",
            "gpt-5.3-codex-spark",
            "unknown-model",
        ] {
            assert!(!capability_for_model(model).supports_image_detail_original);
        }
    }

    #[test]
    fn explicit_invalid_responses_lite_type_fails_instead_of_using_environment() {
        assert!(use_responses_lite("gpt-5.6-sol", Some(&json!(42))).is_err());
    }

    #[test]
    fn service_tier_fast_maps_to_priority_and_default_is_omitted() {
        let mut fast_payload = serde_json::Map::new();
        apply_model_capability_fields(&mut fast_payload, "gpt-5.6-sol", None, Some("fast"))
            .unwrap();
        assert_eq!(fast_payload.get("service_tier"), Some(&json!("priority")));

        let mut default_payload = serde_json::Map::new();
        apply_model_capability_fields(&mut default_payload, "gpt-5.6-sol", None, Some("default"))
            .unwrap();
        assert!(!default_payload.contains_key("service_tier"));
    }

    #[test]
    fn service_tier_rejects_unknown_or_model_unsupported_values() {
        for (model, tier) in [
            ("gpt-5.6-sol", "flex"),
            ("gpt-5.4-mini", "priority"),
            ("unknown-model", "fast"),
        ] {
            let mut payload = serde_json::Map::new();
            assert!(apply_model_capability_fields(&mut payload, model, None, Some(tier)).is_err());
            assert!(!payload.contains_key("service_tier"));
        }
    }

    #[test]
    fn current_existing_models_have_official_context_and_effort_defaults() {
        let cases = [
            ("gpt-5.5", 272_000),
            ("gpt-5.4", 1_000_000),
            ("gpt-5.4-mini", 272_000),
            ("gpt-5.2", 272_000),
        ];

        for (model, maximum) in cases {
            let capability = capability_for_model(model);
            assert_eq!(
                capability.default_reasoning_effort.as_deref(),
                Some("medium")
            );
            assert_eq!(capability.context_window, Some(272_000));
            assert_eq!(capability.max_context_window, Some(maximum));
        }
    }

    #[test]
    fn codex_metadata_requires_explicit_session_identity() {
        assert!(build_codex_client_metadata(None, None).is_err());

        let metadata = HashMap::from([(SESSION_ID_KEY.to_string(), "   ".to_string())]);
        assert!(build_codex_client_metadata(None, Some(&metadata)).is_err());
    }

    #[test]
    fn codex_metadata_defaults_root_thread_and_refreshes_only_turn_identity() {
        let existing = HashMap::from([
            (SESSION_ID_KEY.to_string(), "session-root".to_string()),
            ("custom".to_string(), "preserved".to_string()),
        ]);

        let first =
            build_codex_client_metadata(Some("/tmp/codex-auth.json"), Some(&existing)).unwrap();
        let second =
            build_codex_client_metadata(Some("/tmp/codex-auth.json"), Some(&existing)).unwrap();

        assert_eq!(first[SESSION_ID_KEY], "session-root");
        assert_eq!(first[THREAD_ID_KEY], "session-root");
        assert_eq!(first["custom"], "preserved");
        assert_eq!(first[INSTALLATION_ID_KEY], second[INSTALLATION_ID_KEY]);
        assert_eq!(first[WINDOW_ID_KEY], second[WINDOW_ID_KEY]);
        assert_ne!(first[TURN_ID_KEY], second[TURN_ID_KEY]);

        let turn_metadata: Value = serde_json::from_str(&first[TURN_METADATA_KEY]).unwrap();
        assert_eq!(turn_metadata["session_id"], "session-root");
        assert_eq!(turn_metadata["thread_id"], "session-root");
        assert_eq!(turn_metadata["turn_id"], first[TURN_ID_KEY]);
    }

    #[test]
    fn codex_metadata_preserves_explicit_subagent_thread() {
        let existing = HashMap::from([
            (SESSION_ID_KEY.to_string(), "session-root".to_string()),
            (THREAD_ID_KEY.to_string(), "thread-child".to_string()),
        ]);

        let metadata = build_codex_client_metadata(None, Some(&existing)).unwrap();
        assert_eq!(metadata[SESSION_ID_KEY], "session-root");
        assert_eq!(metadata[THREAD_ID_KEY], "thread-child");

        let invalid = HashMap::from([
            (SESSION_ID_KEY.to_string(), "session-root".to_string()),
            (THREAD_ID_KEY.to_string(), " ".to_string()),
        ]);
        assert!(build_codex_client_metadata(None, Some(&invalid)).is_err());
    }
}
