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
static SESSION_ID: OnceLock<String> = OnceLock::new();
static THREAD_ID: OnceLock<String> = OnceLock::new();
static WINDOW_ID: OnceLock<String> = OnceLock::new();

#[derive(Debug, Clone)]
pub struct ModelCapability {
    pub use_responses_lite: bool,
    pub supports_parallel_tool_calls: bool,
    pub support_verbosity: bool,
    pub default_verbosity: Option<String>,
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
) {
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

    if let Some(tier) = service_tier {
        if tier != "default" && capability.service_tiers.iter().any(|value| value == tier) {
            payload.insert("service_tier".to_string(), Value::String(tier.to_string()));
        }
    }
}

pub fn build_codex_client_metadata(
    auth_json_path: Option<&str>,
    existing: Option<&HashMap<String, String>>,
) -> HashMap<String, String> {
    let mut metadata = existing.cloned().unwrap_or_default();
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
    let session_id = SESSION_ID
        .get_or_init(|| uuid::Uuid::new_v4().to_string())
        .clone();
    let thread_id = THREAD_ID
        .get_or_init(|| uuid::Uuid::new_v4().to_string())
        .clone();
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
    metadata
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
    if let Some(Value::Bool(v)) = value {
        return Ok(if *v {
            ResponsesLiteMode::On
        } else {
            ResponsesLiteMode::Off
        });
    }
    let raw = value
        .and_then(|v| v.as_str().map(|s| s.to_string()))
        .or_else(|| std::env::var(RESPONSES_LITE_ENV).ok())
        .unwrap_or_else(|| "auto".to_string());
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
        support_verbosity: obj
            .get("support_verbosity")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        default_verbosity: obj
            .get("default_verbosity")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string()),
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
        support_verbosity: false,
        default_verbosity: None,
        service_tiers: vec![],
        default_service_tier: None,
        source: "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packaged_capability_json_matches_repo_source() {
        let repo_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../config/model-capabilities.json");
        if !repo_path.exists() {
            return;
        }
        let repo_json = std::fs::read_to_string(repo_path).unwrap();
        let packaged: Value = serde_json::from_str(CAPABILITIES_JSON).unwrap();
        let repo: Value = serde_json::from_str(&repo_json).unwrap();

        assert_eq!(packaged, repo);
    }
}
