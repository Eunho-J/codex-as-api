use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::strict_json;

pub const DEFAULT_CATALOG_TTL: Duration = Duration::from_secs(5 * 60);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CatalogKey {
    pub account_id: String,
    pub base_url: String,
    pub client_version: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReasoningLevel {
    pub effort: String,
    pub description: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ServiceTier {
    pub id: String,
    pub name: String,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub slug: String,
    pub display_name: String,
    pub description: Option<String>,
    pub default_reasoning_level: Option<String>,
    pub supported_reasoning_levels: Vec<ReasoningLevel>,
    pub multi_agent_reasoning_effort: Option<String>,
    pub visibility: String,
    pub supported_in_api: bool,
    pub priority: i32,
    pub service_tiers: Vec<ServiceTier>,
    pub default_service_tier: Option<String>,
    pub support_verbosity: bool,
    pub default_verbosity: Option<String>,
    pub supports_reasoning_summary_parameter: bool,
    pub default_reasoning_summary: String,
    pub comp_hash: Option<String>,
    pub supports_image_detail_original: bool,
    pub context_window: Option<i64>,
    pub max_context_window: Option<i64>,
    pub auto_compact_token_limit: Option<i64>,
    pub effective_context_window_percent: i64,
    pub input_modalities: Vec<String>,
    pub use_responses_lite: bool,
}

impl ModelInfo {
    pub fn resolved_context_window(&self) -> Option<i64> {
        self.context_window.or(self.max_context_window)
    }
}

#[derive(Debug)]
pub struct ModelCatalogSnapshot {
    pub key: CatalogKey,
    pub models: Vec<Arc<ModelInfo>>,
    by_slug: HashMap<String, Arc<ModelInfo>>,
    pub etag: Option<String>,
    pub fetched_at: DateTime<Utc>,
    pub expires_at: DateTime<Utc>,
    deadline: Instant,
}

impl ModelCatalogSnapshot {
    pub fn model(&self, slug: &str) -> Option<Arc<ModelInfo>> {
        self.by_slug.get(slug).cloned()
    }

    pub fn default_model(&self) -> Option<Arc<ModelInfo>> {
        self.models
            .iter()
            .filter(|model| model.visibility == "list")
            .min_by_key(|model| model.priority)
            .cloned()
    }

    fn is_fresh(&self, now: Instant) -> bool {
        now < self.deadline
    }
}

#[derive(Debug, Clone, thiserror::Error)]
pub enum CatalogError {
    #[error("model catalog response is invalid: {0}")]
    Invalid(String),
    #[error("model catalog authentication failed: {0}")]
    Auth(String),
    #[error("{message}")]
    RefreshUpstreamHttp { status: u16, message: String },
    #[error("{0}")]
    RefreshTransport(String),
    #[error("{0}")]
    RefreshProtocol(String),
    #[error("{0}")]
    Internal(String),
    #[error("{message}")]
    UpstreamHttp { status: u16, message: String },
    #[error("model catalog request failed: {0}")]
    Request(String),
}

#[derive(Default)]
struct CacheState {
    snapshots: HashMap<CatalogKey, Arc<ModelCatalogSnapshot>>,
    loading: HashSet<CatalogKey>,
    revisions: HashMap<CatalogKey, u64>,
    failures: HashMap<CatalogKey, (u64, CatalogError)>,
}

pub struct ModelCatalogCache {
    state: Mutex<CacheState>,
    ready: Condvar,
    ttl: Duration,
}

impl ModelCatalogCache {
    pub fn new(ttl: Duration) -> Self {
        Self {
            state: Mutex::new(CacheState::default()),
            ready: Condvar::new(),
            ttl,
        }
    }

    pub fn snapshot<F>(
        &self,
        key: CatalogKey,
        fetch: F,
    ) -> Result<Arc<ModelCatalogSnapshot>, CatalogError>
    where
        F: FnOnce() -> Result<(Vec<u8>, Option<String>), CatalogError>,
    {
        let mut fetch = Some(fetch);
        loop {
            let now = Instant::now();
            let mut state = self.state.lock().map_err(|_| {
                CatalogError::Request("model catalog cache lock is poisoned".into())
            })?;

            if let Some(snapshot) = state.snapshots.get(&key) {
                if snapshot.is_fresh(now) {
                    return Ok(snapshot.clone());
                }
            }

            if state.loading.contains(&key) {
                let observed_revision = state.revisions.get(&key).copied().unwrap_or(0);
                while state.loading.contains(&key) {
                    state = self.ready.wait(state).map_err(|_| {
                        CatalogError::Request("model catalog cache lock is poisoned".into())
                    })?;
                }
                if let Some(snapshot) = state.snapshots.get(&key) {
                    if snapshot.is_fresh(Instant::now()) {
                        return Ok(snapshot.clone());
                    }
                }
                if let Some((revision, error)) = state.failures.get(&key) {
                    if *revision > observed_revision {
                        return Err(error.clone());
                    }
                }
                continue;
            }

            state.loading.insert(key.clone());
            let load_revision = state.revisions.get(&key).copied().unwrap_or(0);
            drop(state);

            let fetched = match fetch.take() {
                Some(fetch_once) => fetch_once(),
                None => Err(CatalogError::Request(
                    "model catalog single-flight fetch closure was already consumed".into(),
                )),
            };
            let parsed = fetched.and_then(|(body, etag)| {
                let etag = etag
                    .as_deref()
                    .map(str::trim)
                    .filter(|etag| !etag.is_empty())
                    .map(str::to_string);
                parse_models_response(&body).map(|models| {
                    let fetched_at = Utc::now();
                    let by_slug = models
                        .iter()
                        .map(|model| (model.slug.clone(), model.clone()))
                        .collect();
                    Arc::new(ModelCatalogSnapshot {
                        key: key.clone(),
                        models,
                        by_slug,
                        etag,
                        fetched_at,
                        expires_at: fetched_at + self.ttl,
                        deadline: Instant::now() + self.ttl,
                    })
                })
            });

            let mut state = self.state.lock().map_err(|_| {
                CatalogError::Request("model catalog cache lock is poisoned".into())
            })?;
            state.loading.remove(&key);
            if state.revisions.get(&key).copied().unwrap_or(0) != load_revision {
                let error = CatalogError::Request(
                    "model catalog refresh was invalidated while in flight".to_string(),
                );
                let revision = state.revisions.get(&key).copied().unwrap_or(0) + 1;
                state.revisions.insert(key.clone(), revision);
                state
                    .failures
                    .insert(key.clone(), (revision, error.clone()));
                self.ready.notify_all();
                return Err(error);
            }
            let revision = state.revisions.get(&key).copied().unwrap_or(0) + 1;
            state.revisions.insert(key.clone(), revision);
            match parsed {
                Ok(snapshot) => {
                    state.failures.remove(&key);
                    state.snapshots.insert(key.clone(), snapshot.clone());
                    self.ready.notify_all();
                    return Ok(snapshot);
                }
                Err(error) => {
                    state.snapshots.remove(&key);
                    state
                        .failures
                        .insert(key.clone(), (revision, error.clone()));
                    self.ready.notify_all();
                    return Err(error);
                }
            }
        }
    }

    #[cfg(test)]
    fn cached_snapshot(
        &self,
        key: &CatalogKey,
    ) -> Result<Option<Arc<ModelCatalogSnapshot>>, CatalogError> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| CatalogError::Request("model catalog cache lock is poisoned".into()))?;
        if state
            .snapshots
            .get(key)
            .is_some_and(|snapshot| !snapshot.is_fresh(Instant::now()))
        {
            state.snapshots.remove(key);
        }
        Ok(state.snapshots.get(key).cloned())
    }

    pub fn observe_etag(&self, key: &CatalogKey, etag: Option<&str>) -> Result<(), CatalogError> {
        let Some(etag) = etag.map(str::trim).filter(|etag| !etag.is_empty()) else {
            return Ok(());
        };
        let mut state = self
            .state
            .lock()
            .map_err(|_| CatalogError::Request("model catalog cache lock is poisoned".into()))?;
        let snapshot_mismatch = state
            .snapshots
            .get(key)
            .is_some_and(|snapshot| snapshot.etag.as_deref() != Some(etag));
        if snapshot_mismatch {
            state.snapshots.remove(key);
            let revision = state.revisions.get(key).copied().unwrap_or(0) + 1;
            state.revisions.insert(key.clone(), revision);
        }
        Ok(())
    }
}

fn required_object<'a>(
    value: &'a Value,
    path: &str,
) -> Result<&'a serde_json::Map<String, Value>, CatalogError> {
    value
        .as_object()
        .ok_or_else(|| CatalogError::Invalid(format!("{path} must be an object")))
}

fn required_string(
    object: &serde_json::Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<String, CatalogError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .map(str::to_string)
        .ok_or_else(|| CatalogError::Invalid(format!("{path}.{key} must be a non-empty string")))
}

fn required_preserved_string(
    object: &serde_json::Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<String, CatalogError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::to_string)
        .ok_or_else(|| CatalogError::Invalid(format!("{path}.{key} must be a string")))
}

fn required_reasoning_effort(
    object: &serde_json::Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<String, CatalogError> {
    match object.get(key) {
        Some(Value::String(value)) if !value.is_empty() => Ok(value.clone()),
        _ => Err(CatalogError::Invalid(format!(
            "{path}.{key} must be a non-empty string"
        ))),
    }
}

fn optional_string(
    object: &serde_json::Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<Option<String>, CatalogError> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.clone())),
        Some(_) => Err(CatalogError::Invalid(format!(
            "{path}.{key} must be a non-empty string or null"
        ))),
    }
}

fn optional_reasoning_effort(
    object: &serde_json::Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<Option<String>, CatalogError> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if !value.is_empty() => Ok(Some(value.clone())),
        Some(_) => Err(CatalogError::Invalid(format!(
            "{path}.{key} must be a non-empty string or null"
        ))),
    }
}

fn optional_preserved_string(
    object: &serde_json::Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<Option<String>, CatalogError> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(CatalogError::Invalid(format!(
            "{path}.{key} must be a string or null"
        ))),
    }
}

fn required_bool(
    object: &serde_json::Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<bool, CatalogError> {
    object
        .get(key)
        .and_then(Value::as_bool)
        .ok_or_else(|| CatalogError::Invalid(format!("{path}.{key} must be a boolean")))
}

fn optional_safe_integer(
    object: &serde_json::Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<Option<i64>, CatalogError> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => strict_json::as_js_safe_integer(number)
            .map(Some)
            .ok_or_else(|| {
                CatalogError::Invalid(format!(
                    "{path}.{key} must be a JavaScript-safe integer or null"
                ))
            }),
        Some(_) => Err(CatalogError::Invalid(format!(
            "{path}.{key} must be a JavaScript-safe integer or null"
        ))),
    }
}

fn string_array(
    object: &serde_json::Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<Vec<String>, CatalogError> {
    let values = object
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| CatalogError::Invalid(format!("{path}.{key} must be an array")))?;
    let mut result = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let value = value
            .as_str()
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                CatalogError::Invalid(format!("{path}.{key}[{index}] must be a non-empty string"))
            })?;
        result.push(value.to_string());
    }
    Ok(result)
}

pub fn parse_models_response(body: &[u8]) -> Result<Vec<Arc<ModelInfo>>, CatalogError> {
    let root = strict_json::parse_slice(body)
        .map_err(|error| CatalogError::Invalid(format!("root must be valid JSON: {error}")))?;
    let root = root
        .as_object()
        .ok_or_else(|| CatalogError::Invalid("root must be an object".into()))?;
    let models = root
        .get("models")
        .and_then(Value::as_array)
        .ok_or_else(|| CatalogError::Invalid("root.models must be an array".into()))?;
    let mut parsed = Vec::with_capacity(models.len());
    let mut slugs = HashSet::new();
    for (index, value) in models.iter().enumerate() {
        let path = format!("root.models[{index}]");
        let object = required_object(value, &path)?;
        let slug = required_preserved_string(object, "slug", &path)?;
        if !slugs.insert(slug.clone()) {
            return Err(CatalogError::Invalid(
                "model catalog contains a duplicate model slug".to_string(),
            ));
        }
        let display_name = required_preserved_string(object, "display_name", &path)?;
        let description = optional_preserved_string(object, "description", &path)?;
        let default_reasoning_level =
            optional_reasoning_effort(object, "default_reasoning_level", &path)?;
        let multi_agent_reasoning_effort =
            optional_reasoning_effort(object, "multi_agent_reasoning_effort", &path)?;

        let level_values = object
            .get("supported_reasoning_levels")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                CatalogError::Invalid(format!(
                    "{path}.supported_reasoning_levels must be an array"
                ))
            })?;
        let mut supported_reasoning_levels = Vec::with_capacity(level_values.len());
        for (level_index, level) in level_values.iter().enumerate() {
            let level_path = format!("{path}.supported_reasoning_levels[{level_index}]");
            let level = required_object(level, &level_path)?;
            let effort = required_reasoning_effort(level, "effort", &level_path)?;
            let description = level
                .get("description")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    CatalogError::Invalid(format!("{level_path}.description must be a string"))
                })?
                .to_string();
            supported_reasoning_levels.push(ReasoningLevel {
                effort,
                description,
            });
        }
        let visibility = required_string(object, "visibility", &path)?;
        if !matches!(visibility.as_str(), "list" | "hide" | "none") {
            return Err(CatalogError::Invalid(format!(
                "{path}.visibility must be one of: list, hide, none"
            )));
        }
        let supported_in_api = required_bool(object, "supported_in_api", &path)?;
        let priority = object
            .get("priority")
            .and_then(Value::as_number)
            .and_then(strict_json::as_js_safe_integer)
            .and_then(|value| i32::try_from(value).ok())
            .ok_or_else(|| {
                CatalogError::Invalid(format!("{path}.priority must be a 32-bit integer"))
            })?;

        let tier_values = match object.get("service_tiers") {
            None => &[][..],
            Some(Value::Array(values)) => values.as_slice(),
            Some(_) => {
                return Err(CatalogError::Invalid(format!(
                    "{path}.service_tiers must be an array"
                )));
            }
        };
        let mut service_tiers = Vec::with_capacity(tier_values.len());
        for (tier_index, tier) in tier_values.iter().enumerate() {
            let tier_path = format!("{path}.service_tiers[{tier_index}]");
            let tier = required_object(tier, &tier_path)?;
            let id = required_preserved_string(tier, "id", &tier_path)?;
            let name = required_preserved_string(tier, "name", &tier_path)?;
            let description = tier
                .get("description")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    CatalogError::Invalid(format!("{tier_path}.description must be a string"))
                })?
                .to_string();
            service_tiers.push(ServiceTier {
                id,
                name,
                description,
            });
        }
        let default_service_tier =
            optional_preserved_string(object, "default_service_tier", &path)?;
        let support_verbosity = required_bool(object, "support_verbosity", &path)?;
        let default_verbosity = optional_string(object, "default_verbosity", &path)?;
        if default_verbosity
            .as_deref()
            .is_some_and(|value| !matches!(value, "low" | "medium" | "high"))
        {
            return Err(CatalogError::Invalid(format!(
                "{path}.default_verbosity must be one of: low, medium, high"
            )));
        }
        let supports_reasoning_summary_parameter =
            match object.get("supports_reasoning_summary_parameter") {
                None => true,
                Some(Value::Bool(value)) => *value,
                Some(_) => {
                    return Err(CatalogError::Invalid(format!(
                        "{path}.supports_reasoning_summary_parameter must be a boolean"
                    )));
                }
            };
        let default_reasoning_summary = match object.get("default_reasoning_summary") {
            None => "auto".to_string(),
            Some(Value::String(value))
                if matches!(value.as_str(), "auto" | "concise" | "detailed" | "none") =>
            {
                value.clone()
            }
            Some(_) => {
                return Err(CatalogError::Invalid(format!(
                    "{path}.default_reasoning_summary must be one of: auto, concise, detailed, none"
                )));
            }
        };
        let comp_hash = optional_preserved_string(object, "comp_hash", &path)?;
        let supports_image_detail_original = match object.get("supports_image_detail_original") {
            None => false,
            Some(Value::Bool(value)) => *value,
            Some(_) => {
                return Err(CatalogError::Invalid(format!(
                    "{path}.supports_image_detail_original must be a boolean"
                )));
            }
        };
        let context_window = optional_safe_integer(object, "context_window", &path)?;
        let max_context_window = optional_safe_integer(object, "max_context_window", &path)?;
        let auto_compact_token_limit =
            optional_safe_integer(object, "auto_compact_token_limit", &path)?;
        let effective_context_window_percent = match object.get("effective_context_window_percent")
        {
            None => 95,
            Some(Value::Number(number)) => {
                strict_json::as_js_safe_integer(number).ok_or_else(|| {
                    CatalogError::Invalid(format!(
                        "{path}.effective_context_window_percent must be a JavaScript-safe integer"
                    ))
                })?
            }
            Some(_) => {
                return Err(CatalogError::Invalid(format!(
                    "{path}.effective_context_window_percent must be a JavaScript-safe integer"
                )));
            }
        };
        let input_modalities = match object.get("input_modalities") {
            None => vec!["text".to_string(), "image".to_string()],
            Some(Value::Array(_)) => string_array(object, "input_modalities", &path)?,
            Some(_) => {
                return Err(CatalogError::Invalid(format!(
                    "{path}.input_modalities must be an array"
                )));
            }
        };
        if input_modalities
            .iter()
            .find(|modality| !matches!(modality.as_str(), "text" | "image" | "audio"))
            .is_some()
        {
            return Err(CatalogError::Invalid(format!(
                "{path}.input_modalities contains an unsupported value"
            )));
        }
        let use_responses_lite = match object.get("use_responses_lite") {
            None => false,
            Some(Value::Bool(value)) => *value,
            Some(_) => {
                return Err(CatalogError::Invalid(format!(
                    "{path}.use_responses_lite must be a boolean"
                )));
            }
        };
        parsed.push(Arc::new(ModelInfo {
            slug,
            display_name,
            description,
            default_reasoning_level,
            supported_reasoning_levels,
            multi_agent_reasoning_effort,
            visibility,
            supported_in_api,
            priority,
            service_tiers,
            default_service_tier,
            support_verbosity,
            default_verbosity,
            supports_reasoning_summary_parameter,
            default_reasoning_summary,
            comp_hash,
            supports_image_detail_original,
            context_window,
            max_context_window,
            auto_compact_token_limit,
            effective_context_window_percent,
            input_modalities,
            use_responses_lite,
        }));
    }

    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Barrier;
    use std::thread;

    fn model(slug: &str, priority: i32) -> Value {
        json!({
            "slug": slug,
            "display_name": slug,
            "description": "test",
            "default_reasoning_level": "medium",
            "supported_reasoning_levels": [{"effort":"low","description":"low"},{"effort":"medium","description":"medium"}],
            "shell_type": "shell_command",
            "visibility": "list",
            "supported_in_api": true,
            "priority": priority,
            "service_tiers": [{"id":"priority","name":"Fast","description":"Fast"}],
            "default_service_tier": null,
            "support_verbosity": true,
            "default_verbosity": "low",
            "supports_image_detail_original": true,
            "context_window": 100000,
            "max_context_window": 120000,
            "auto_compact_token_limit": 80000,
            "effective_context_window_percent": 95,
            "input_modalities": ["text", "image"],
            "use_responses_lite": false,
            "supports_reasoning_summaries": true,
            "available_in_plans": ["plus", "pro"],
            "prefer_websockets": false,
            "requires_sandboxed_review": false,
            "minimal_client_version": "0.153.3",
            "base_instructions": "Do useful work.",
            "model_messages": {"input": "message"}
        })
    }

    #[test]
    fn strict_parser_preserves_upstream_order_and_wire_metadata() {
        let body =
            serde_json::to_vec(&json!({"models":[model("second", 2), model("first", 1)]})).unwrap();
        let models = parse_models_response(&body).unwrap();
        assert_eq!(models[0].slug, "second");
        assert_eq!(models[1].slug, "first");
        assert_eq!(models[0].service_tiers[0].id, "priority");
        assert_eq!(models[0].service_tiers[0].name, "Fast");
        assert!(models[0].supports_reasoning_summary_parameter);
        assert_eq!(models[0].default_reasoning_summary, "auto");
        assert_eq!(models[1].supported_reasoning_levels[1].effort, "medium");
        assert_eq!(
            models[1].supported_reasoning_levels[1].description,
            "medium"
        );
    }

    #[test]
    fn parser_diagnostics_do_not_reflect_slug_or_modality_values() {
        let secret = "access-token-sentinel";
        let duplicate = json!({"models": [model(secret, 1), model(secret, 2)]});
        let error = parse_models_response(&serde_json::to_vec(&duplicate).unwrap()).unwrap_err();
        assert!(!error.to_string().contains(secret));

        let mut unsupported_modality = model("normal-model", 1);
        unsupported_modality["input_modalities"] = json!([secret]);
        let error = parse_models_response(
            &serde_json::to_vec(&json!({"models": [unsupported_modality]})).unwrap(),
        )
        .unwrap_err();
        assert!(!error.to_string().contains(secret));

        let mut valid = model("preserved-model", 1);
        valid["input_modalities"] = json!(["audio", "text"]);
        let parsed =
            parse_models_response(&serde_json::to_vec(&json!({"models": [valid]})).unwrap())
                .unwrap();
        assert_eq!(parsed[0].slug, "preserved-model");
        assert_eq!(parsed[0].input_modalities, ["audio", "text"]);
    }

    #[test]
    fn malformed_and_duplicate_catalogs_fail_atomically() {
        for body in [
            json!([model("bare-array", 1)]),
            json!({"models": [model("same", 1), model("same", 2)]}),
            json!({"models": [model("ok", 1), {"slug":"broken"}]}),
        ] {
            assert!(parse_models_response(&serde_json::to_vec(&body).unwrap()).is_err());
        }
    }

    #[test]
    fn preserves_empty_catalogs_opaque_slugs_comp_hash_and_custom_reasoning_values() {
        assert!(parse_models_response(br#"{"models":[]}"#)
            .unwrap()
            .is_empty());

        let mut candidate = model(" ", 1);
        candidate["comp_hash"] = json!(" compatibility family ");
        candidate["default_reasoning_level"] = json!(" ");
        candidate["multi_agent_reasoning_effort"] = json!(" custom ");
        candidate["supported_reasoning_levels"] = json!([{"effort": " ", "description": "Custom"}]);
        let parsed =
            parse_models_response(&serde_json::to_vec(&json!({"models": [candidate]})).unwrap())
                .unwrap();

        assert_eq!(parsed[0].slug, " ");
        assert_eq!(
            parsed[0].comp_hash.as_deref(),
            Some(" compatibility family ")
        );
        assert_eq!(parsed[0].default_reasoning_level.as_deref(), Some(" "));
        assert_eq!(
            parsed[0].multi_agent_reasoning_effort.as_deref(),
            Some(" custom ")
        );
        assert_eq!(parsed[0].supported_reasoning_levels[0].effort, " ");

        for field in ["default_reasoning_level", "multi_agent_reasoning_effort"] {
            let mut invalid = model("bad-reasoning", 1);
            invalid[field] = json!("");
            assert!(parse_models_response(
                &serde_json::to_vec(&json!({"models": [invalid]})).unwrap()
            )
            .is_err());
        }
        let mut invalid = model("bad-reasoning-level", 1);
        invalid["supported_reasoning_levels"][0]["effort"] = json!("");
        assert!(
            parse_models_response(&serde_json::to_vec(&json!({"models": [invalid]})).unwrap())
                .is_err()
        );
    }

    #[test]
    fn official_optional_fields_use_only_documented_wire_defaults() {
        let mut candidate = model("optional-fields", 1);
        for field in [
            "description",
            "default_reasoning_level",
            "context_window",
            "max_context_window",
            "default_verbosity",
            "auto_compact_token_limit",
            "default_service_tier",
            "multi_agent_reasoning_effort",
            "service_tiers",
            "use_responses_lite",
            "supports_image_detail_original",
            "effective_context_window_percent",
            "input_modalities",
            "supports_reasoning_summary_parameter",
            "default_reasoning_summary",
            "comp_hash",
        ] {
            candidate
                .as_object_mut()
                .expect("test model must be an object")
                .remove(field);
        }
        candidate["supported_reasoning_levels"] = json!([]);
        let parsed =
            parse_models_response(&serde_json::to_vec(&json!({"models": [candidate]})).unwrap())
                .unwrap();
        let model = &parsed[0];
        assert!(model.description.is_none());
        assert!(model.default_reasoning_level.is_none());
        assert!(model.context_window.is_none());
        assert!(model.max_context_window.is_none());
        assert!(model.default_verbosity.is_none());
        assert!(model.auto_compact_token_limit.is_none());
        assert!(model.default_service_tier.is_none());
        assert!(model.service_tiers.is_empty());
        assert!(!model.use_responses_lite);
        assert!(!model.supports_image_detail_original);
        assert_eq!(model.effective_context_window_percent, 95);
        assert_eq!(model.input_modalities, ["text", "image"]);
        assert!(model.supports_reasoning_summary_parameter);
        assert_eq!(model.default_reasoning_summary, "auto");
        assert!(model.comp_hash.is_none());
    }

    #[test]
    fn preserves_and_validates_reasoning_summary_controls() {
        let mut candidate = model("summary-controls", 1);
        candidate["supports_reasoning_summary_parameter"] = json!(false);
        candidate["default_reasoning_summary"] = json!("detailed");
        let parsed =
            parse_models_response(&serde_json::to_vec(&json!({"models": [candidate]})).unwrap())
                .unwrap();
        assert!(!parsed[0].supports_reasoning_summary_parameter);
        assert_eq!(parsed[0].default_reasoning_summary, "detailed");

        for (field, value) in [
            ("supports_reasoning_summary_parameter", Value::Null),
            ("supports_reasoning_summary_parameter", json!("true")),
            ("default_reasoning_summary", Value::Null),
            ("default_reasoning_summary", json!("future")),
        ] {
            let mut invalid = model("bad-summary", 1);
            invalid[field] = value;
            assert!(parse_models_response(
                &serde_json::to_vec(&json!({"models": [invalid]})).unwrap()
            )
            .is_err());
        }
    }

    #[test]
    fn optional_description_and_input_modality_order_are_preserved_verbatim() {
        let mut candidate = model("preserved-fields", 1);
        candidate["description"] = json!("");
        candidate["input_modalities"] = json!(["image", "image", "text"]);

        let parsed =
            parse_models_response(&serde_json::to_vec(&json!({"models": [candidate]})).unwrap())
                .unwrap();

        assert_eq!(parsed[0].description.as_deref(), Some(""));
        assert_eq!(parsed[0].input_modalities, ["image", "image", "text"]);
    }

    #[test]
    fn cosmetic_catalog_strings_may_be_empty() {
        let mut candidate = model("cosmetic-empty", 1);
        candidate["display_name"] = json!("");
        candidate["description"] = json!("");
        candidate["supported_reasoning_levels"][0]["description"] = json!("");
        candidate["service_tiers"][0]["name"] = json!("");
        candidate["service_tiers"][0]["description"] = json!("");
        candidate["service_tiers"][0]["id"] = json!("");
        candidate["default_service_tier"] = json!("");

        let parsed =
            parse_models_response(&serde_json::to_vec(&json!({"models": [candidate]})).unwrap())
                .unwrap();
        let model = &parsed[0];
        assert_eq!(model.display_name, "");
        assert_eq!(model.description.as_deref(), Some(""));
        assert_eq!(model.supported_reasoning_levels[0].description, "");
        assert_eq!(model.service_tiers[0].name, "");
        assert_eq!(model.service_tiers[0].description, "");
        assert_eq!(model.service_tiers[0].id, "");
        assert_eq!(model.default_service_tier.as_deref(), Some(""));
    }

    #[test]
    fn remaining_officially_unconstrained_catalog_values_preserve_order_and_value() {
        let mut candidate = model("unconstrained", 1);
        candidate["supported_reasoning_levels"] = json!([
            {"effort": "low", "description": "first"},
            {"effort": "low", "description": "second"}
        ]);
        candidate["default_reasoning_level"] = json!("low");
        candidate["multi_agent_reasoning_effort"] = json!("also-not-listed");
        candidate["service_tiers"] = json!([
            {"id": "same", "name": "first", "description": ""},
            {"id": "same", "name": "second", "description": ""}
        ]);
        candidate["default_service_tier"] = json!("not-listed");
        candidate["context_window"] = json!(-1);
        candidate["max_context_window"] = json!(-2);
        candidate["auto_compact_token_limit"] = json!(0);
        candidate["effective_context_window_percent"] = json!(-100);

        let parsed =
            parse_models_response(&serde_json::to_vec(&json!({"models": [candidate]})).unwrap())
                .unwrap();
        let model = &parsed[0];
        assert_eq!(
            model
                .supported_reasoning_levels
                .iter()
                .map(|level| level.description.as_str())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert_eq!(
            model
                .service_tiers
                .iter()
                .map(|tier| tier.name.as_str())
                .collect::<Vec<_>>(),
            ["first", "second"]
        );
        assert_eq!(model.default_reasoning_level.as_deref(), Some("low"));
        assert_eq!(
            model.multi_agent_reasoning_effort.as_deref(),
            Some("also-not-listed")
        );
        assert_eq!(model.default_service_tier.as_deref(), Some("not-listed"));
        assert_eq!(model.context_window, Some(-1));
        assert_eq!(model.max_context_window, Some(-2));
        assert_eq!(model.auto_compact_token_limit, Some(0));
        assert_eq!(model.effective_context_window_percent, -100);
    }

    #[test]
    fn default_reasoning_level_outside_supported_levels_is_preserved() {
        let mut candidate = model("invalid-default-reasoning", 1);
        candidate["default_reasoning_level"] = json!("not-listed");

        let parsed =
            parse_models_response(&serde_json::to_vec(&json!({"models": [candidate]})).unwrap())
                .unwrap();

        assert_eq!(
            parsed[0].default_reasoning_level.as_deref(),
            Some("not-listed")
        );
    }

    #[test]
    fn default_ultra_without_a_wire_mapping_is_preserved() {
        let mut candidate = model("invalid-ultra-default", 1);
        candidate["supported_reasoning_levels"] =
            json!([{"effort": "ultra", "description": "ultra"}]);
        candidate["default_reasoning_level"] = json!("ultra");
        candidate["multi_agent_reasoning_effort"] = json!("ultra");

        let parsed =
            parse_models_response(&serde_json::to_vec(&json!({"models": [candidate]})).unwrap())
                .unwrap();

        assert_eq!(parsed[0].default_reasoning_level.as_deref(), Some("ultra"));
        assert_eq!(parsed[0].supported_reasoning_levels[0].effort, "ultra");
    }

    #[test]
    fn cross_runtime_unsafe_integers_are_rejected() {
        for (field, value) in [
            ("context_window", strict_json::JS_SAFE_INTEGER as i64 + 1),
            (
                "max_context_window",
                -(strict_json::JS_SAFE_INTEGER as i64) - 1,
            ),
            (
                "auto_compact_token_limit",
                strict_json::JS_SAFE_INTEGER as i64 + 1,
            ),
            (
                "effective_context_window_percent",
                -(strict_json::JS_SAFE_INTEGER as i64) - 1,
            ),
        ] {
            let mut candidate = model("unsafe-integer", 1);
            candidate[field] = json!(value);
            assert!(parse_models_response(
                &serde_json::to_vec(&json!({"models": [candidate]})).unwrap()
            )
            .is_err());
        }
    }

    #[test]
    fn integral_json_numbers_are_accepted_for_catalog_integer_fields() {
        let mut candidate = model("integral-numbers", 1);
        candidate["priority"] = json!(2.0);
        candidate["context_window"] = json!(100_000.0);
        candidate["max_context_window"] = serde_json::from_str("120000e0").unwrap();
        candidate["auto_compact_token_limit"] = json!(80_000.0);
        candidate["effective_context_window_percent"] = serde_json::from_str("95e0").unwrap();

        let parsed =
            parse_models_response(&serde_json::to_vec(&json!({"models": [candidate]})).unwrap())
                .unwrap();
        assert_eq!(parsed[0].priority, 2);
        assert_eq!(parsed[0].context_window, Some(100_000));
        assert_eq!(parsed[0].max_context_window, Some(120_000));
        assert_eq!(parsed[0].auto_compact_token_limit, Some(80_000));
        assert_eq!(parsed[0].effective_context_window_percent, 95);
    }

    #[test]
    fn fractional_boolean_and_out_of_range_catalog_integers_are_rejected() {
        for (field, value) in [
            ("priority", json!(1.5)),
            ("priority", json!(true)),
            ("priority", json!(i64::from(i32::MAX) + 1)),
            ("context_window", json!(1.5)),
            ("context_window", json!(true)),
            ("effective_context_window_percent", json!(1.5)),
            ("effective_context_window_percent", json!(true)),
        ] {
            let mut candidate = model("invalid-number", 1);
            candidate[field] = value;
            assert!(parse_models_response(
                &serde_json::to_vec(&json!({"models": [candidate]})).unwrap()
            )
            .is_err());
        }
    }

    fn key(account_id: &str) -> CatalogKey {
        CatalogKey {
            account_id: account_id.to_string(),
            base_url: "https://example.test/backend-api/codex".to_string(),
            client_version: "0.153.3".to_string(),
        }
    }

    fn body(slug: &str) -> Vec<u8> {
        serde_json::to_vec(&json!({"models": [model(slug, 1)]})).unwrap()
    }

    #[test]
    fn default_prefers_first_listed_visibility_after_priority_sort() {
        let mut hidden = model("hidden", 0);
        hidden["visibility"] = json!("hide");
        let parsed = parse_models_response(
            &serde_json::to_vec(&json!({"models": [model("visible", 2), hidden]})).unwrap(),
        )
        .unwrap();
        let by_slug = parsed
            .iter()
            .map(|model| (model.slug.clone(), model.clone()))
            .collect();
        let snapshot = ModelCatalogSnapshot {
            key: key("account"),
            models: parsed,
            by_slug,
            etag: None,
            fetched_at: Utc::now(),
            expires_at: Utc::now() + Duration::from_secs(60),
            deadline: Instant::now() + Duration::from_secs(60),
        };
        assert_eq!(snapshot.default_model().unwrap().slug, "visible");
    }

    #[test]
    fn default_selection_is_priority_ordered_and_stable_for_ties() {
        let mut hidden = model("hidden-lower-priority", 0);
        hidden["visibility"] = json!("hide");
        let parsed = parse_models_response(
            &serde_json::to_vec(&json!({
                "models": [
                    model("first-visible-tie", 2),
                    hidden,
                    model("second-visible-tie", 2)
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        let by_slug = parsed
            .iter()
            .map(|model| (model.slug.clone(), model.clone()))
            .collect();
        let snapshot = ModelCatalogSnapshot {
            key: key("account"),
            models: parsed,
            by_slug,
            etag: None,
            fetched_at: Utc::now(),
            expires_at: Utc::now() + Duration::from_secs(60),
            deadline: Instant::now() + Duration::from_secs(60),
        };

        assert_eq!(snapshot.default_model().unwrap().slug, "first-visible-tie");
    }

    #[test]
    fn hidden_only_catalog_has_no_implicit_default() {
        let mut hidden = model("hidden", 0);
        hidden["visibility"] = json!("hide");
        let parsed =
            parse_models_response(&serde_json::to_vec(&json!({"models": [hidden]})).unwrap())
                .unwrap();
        let by_slug = parsed
            .iter()
            .map(|model| (model.slug.clone(), model.clone()))
            .collect();
        let snapshot = ModelCatalogSnapshot {
            key: key("account"),
            models: parsed,
            by_slug,
            etag: None,
            fetched_at: Utc::now(),
            expires_at: Utc::now() + Duration::from_secs(60),
            deadline: Instant::now() + Duration::from_secs(60),
        };
        assert!(snapshot.default_model().is_none());
        assert_eq!(snapshot.model("hidden").unwrap().slug, "hidden");
    }

    #[test]
    fn cache_is_single_flight_for_the_same_key() {
        let cache = Arc::new(ModelCatalogCache::new(Duration::from_secs(60)));
        let calls = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = cache.clone();
            let calls = calls.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                cache
                    .snapshot(key("account"), || {
                        calls.fetch_add(1, Ordering::SeqCst);
                        thread::sleep(Duration::from_millis(20));
                        Ok((body("live"), Some("etag-1".to_string())))
                    })
                    .unwrap()
                    .models[0]
                    .slug
                    .clone()
            }));
        }
        for handle in handles {
            assert_eq!(handle.join().unwrap(), "live");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn expired_snapshot_is_never_returned_after_refresh_failure() {
        let cache = ModelCatalogCache::new(Duration::from_millis(1));
        cache
            .snapshot(key("account"), || Ok((body("first"), None)))
            .unwrap();
        thread::sleep(Duration::from_millis(5));
        let result = cache.snapshot(key("account"), || {
            Err(CatalogError::Request("offline".to_string()))
        });
        assert!(result.is_err());
        assert!(cache.cached_snapshot(&key("account")).unwrap().is_none());
    }

    #[test]
    fn cache_key_includes_account_and_etag_mismatch_invalidates() {
        let cache = ModelCatalogCache::new(Duration::from_secs(60));
        let first = cache
            .snapshot(key("account-a"), || {
                Ok((body("first"), Some("etag-1".to_string())))
            })
            .unwrap();
        assert_eq!(first.key.account_id, "account-a");
        let second = cache
            .snapshot(key("account-b"), || {
                Ok((body("second"), Some("etag-2".to_string())))
            })
            .unwrap();
        assert_eq!(second.key.account_id, "account-b");
        assert_eq!(
            cache
                .cached_snapshot(&key("account-a"))
                .unwrap()
                .unwrap()
                .models[0]
                .slug,
            "first"
        );
        cache
            .observe_etag(&key("account-b"), Some("etag-3"))
            .unwrap();
        let refreshed = cache
            .snapshot(key("account-b"), || {
                Ok((body("third"), Some("etag-3".to_string())))
            })
            .unwrap();
        assert_eq!(refreshed.models[0].slug, "third");
    }

    #[test]
    fn etag_invalidation_prevents_inflight_snapshot_publication() {
        let cache = Arc::new(ModelCatalogCache::new(Duration::from_millis(1)));
        cache
            .snapshot(key("account"), || {
                Ok((body("seed"), Some("stale".to_string())))
            })
            .unwrap();
        thread::sleep(Duration::from_millis(5));
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker_cache = cache.clone();
        let worker = thread::spawn(move || {
            worker_cache.snapshot(key("account"), || {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                Ok((body("stale"), Some("stale".to_string())))
            })
        });

        started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        cache.observe_etag(&key("account"), Some("new")).unwrap();
        release_tx.send(()).unwrap();
        assert!(matches!(
            worker.join().unwrap(),
            Err(CatalogError::Request(_))
        ));

        let refreshed = cache
            .snapshot(key("account"), || {
                Ok((body("fresh"), Some("new".to_string())))
            })
            .unwrap();
        assert_eq!(refreshed.models[0].slug, "fresh");
        assert_eq!(refreshed.etag.as_deref(), Some("new"));
    }

    #[test]
    fn etag_observation_does_not_invalidate_initial_or_same_etag_refresh() {
        for seed in [false, true] {
            let cache = Arc::new(ModelCatalogCache::new(Duration::from_millis(1)));
            if seed {
                cache
                    .snapshot(key("account"), || {
                        Ok((body("seed"), Some("same".to_string())))
                    })
                    .unwrap();
                thread::sleep(Duration::from_millis(5));
            }
            let (started_tx, started_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let worker_cache = cache.clone();
            let worker = thread::spawn(move || {
                worker_cache.snapshot(key("account"), || {
                    started_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok((body("loaded"), Some("same".to_string())))
                })
            });

            started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            cache.observe_etag(&key("account"), Some("same")).unwrap();
            release_tx.send(()).unwrap();
            assert_eq!(
                worker.join().unwrap().unwrap().etag.as_deref(),
                Some("same")
            );
        }
    }

    #[test]
    fn blank_etags_are_treated_as_absent() {
        let cache = ModelCatalogCache::new(Duration::from_secs(60));
        let snapshot = cache
            .snapshot(key("account"), || {
                Ok((body("first"), Some("   ".to_string())))
            })
            .unwrap();
        assert_eq!(snapshot.etag, None);

        cache.observe_etag(&key("account"), Some(" \t ")).unwrap();
        let cached = cache
            .snapshot(key("account"), || {
                panic!("a blank ETag must not invalidate a fresh snapshot")
            })
            .unwrap();
        assert!(Arc::ptr_eq(&snapshot, &cached));
    }

    #[test]
    fn failed_refresh_is_single_flight_and_shared_with_waiters() {
        let cache = Arc::new(ModelCatalogCache::new(Duration::from_secs(60)));
        let calls = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(8));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let cache = cache.clone();
            let calls = calls.clone();
            let barrier = barrier.clone();
            handles.push(thread::spawn(move || {
                barrier.wait();
                cache.snapshot(key("account"), || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    thread::sleep(Duration::from_millis(20));
                    Err(CatalogError::Request("offline".to_string()))
                })
            }));
        }
        for handle in handles {
            assert!(handle.join().unwrap().is_err());
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
