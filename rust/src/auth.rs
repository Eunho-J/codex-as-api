use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::{DateTime, Duration, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use thiserror::Error;

pub const CHATGPT_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const DEFAULT_AUTH_PATH: &str = "~/.codex/auth.json";
pub const DEFAULT_REFRESH_URL: &str = "https://auth.openai.com/oauth/token";
pub const REFRESH_URL_OVERRIDE_ENV: &str = "CODEX_REFRESH_TOKEN_URL_OVERRIDE";
const REFRESH_WINDOW_MINUTES: i64 = 5;

static REFRESH_LOCKS: std::sync::LazyLock<Mutex<HashMap<PathBuf, std::sync::Arc<Mutex<()>>>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

#[derive(Debug, Error)]
pub enum AuthError {
    #[error("{0}")]
    OAuth(String),
    #[error("{0}")]
    Missing(String),
    #[error("{0}")]
    Refresh(String),
}

#[derive(Debug, Clone)]
pub struct ChatGPTTokenData {
    pub auth_path: PathBuf,
    pub access_token: String,
    pub refresh_token: String,
    pub id_token: String,
    pub account_id: String,
    pub plan_type: Option<String>,
    pub user_id: Option<String>,
    pub fedramp: bool,
    pub access_expires_at: Option<DateTime<Utc>>,
}

impl ChatGPTTokenData {
    pub fn expired(&self) -> bool {
        match self.access_expires_at {
            Some(exp) => exp <= Utc::now(),
            None => false,
        }
    }

    pub fn expires_within_refresh_window(&self) -> bool {
        match self.access_expires_at {
            Some(exp) => exp <= Utc::now() + Duration::minutes(REFRESH_WINDOW_MINUTES),
            None => false,
        }
    }
}

pub fn resolve_auth_path(raw: Option<&str>) -> PathBuf {
    let value = raw.or_else(|| std::env::var("CODEX_HOME").ok().as_deref().map(|_| ""));
    match (raw, value) {
        (None, _) => {
            let codex_home = std::env::var("CODEX_HOME").ok();
            if let Some(home) = codex_home {
                expand_tilde(&home).join("auth.json")
            } else {
                expand_tilde(DEFAULT_AUTH_PATH)
            }
        }
        (Some(r), _) => expand_tilde(r),
    }
}

fn expand_tilde(path: &str) -> PathBuf {
    if path.starts_with("~/") {
        if let Some(home) = dirs_home() {
            return PathBuf::from(home).join(&path[2..]);
        }
    }
    PathBuf::from(path)
}

fn dirs_home() -> Option<String> {
    std::env::var("HOME").ok()
}

pub fn jwt_claims(jwt: &str) -> Result<serde_json::Map<String, Value>, AuthError> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() < 2 || parts[1].is_empty() {
        return Ok(serde_json::Map::new());
    }
    let padded = pad_base64(parts[1]);
    let decoded = URL_SAFE_NO_PAD
        .decode(padded.trim_end_matches('='))
        .map_err(|_| AuthError::OAuth("invalid ChatGPT OAuth JWT payload".to_string()))?;
    let value: Value = serde_json::from_slice(&decoded)
        .map_err(|_| AuthError::OAuth("invalid ChatGPT OAuth JWT payload".to_string()))?;
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(AuthError::OAuth(
            "invalid ChatGPT OAuth JWT claims".to_string(),
        )),
    }
}

fn pad_base64(input: &str) -> String {
    let pad = (4 - input.len() % 4) % 4;
    let mut s = input.to_string();
    for _ in 0..pad {
        s.push('=');
    }
    s
}

fn expiration(jwt: &str) -> Result<Option<DateTime<Utc>>, AuthError> {
    let claims = jwt_claims(jwt)?;
    match claims.get("exp") {
        Some(Value::Number(n)) => {
            if let Some(ts) = n.as_i64() {
                Ok(DateTime::from_timestamp(ts, 0))
            } else {
                Ok(None)
            }
        }
        _ => Ok(None),
    }
}

fn auth_claims(jwt: &str) -> Result<serde_json::Map<String, Value>, AuthError> {
    let claims = jwt_claims(jwt)?;
    match claims.get("https://api.openai.com/auth") {
        Some(Value::Object(map)) => Ok(map.clone()),
        _ => Ok(serde_json::Map::new()),
    }
}

pub fn redact_text(text: &str, values: &[&str]) -> String {
    let mut sorted: Vec<&str> = values.iter().filter(|v| !v.is_empty()).copied().collect();
    sorted.sort_by(|a, b| b.len().cmp(&a.len()));
    let mut redacted = text.to_string();
    for v in sorted {
        redacted = redacted.replace(v, "***");
    }
    redacted
}

pub fn load_token_data(auth_json_path: Option<&str>) -> Result<ChatGPTTokenData, AuthError> {
    let path = resolve_auth_path(auth_json_path);
    let raw = fs::read_to_string(&path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            AuthError::Missing(format!(
                "ChatGPT OAuth auth file not found: {}",
                path.display()
            ))
        } else {
            AuthError::OAuth(format!(
                "ChatGPT OAuth auth file read error: {}",
                path.display()
            ))
        }
    })?;
    let data: Value = serde_json::from_str(&raw).map_err(|_| {
        AuthError::OAuth(format!(
            "ChatGPT OAuth auth file is invalid JSON: {}",
            path.display()
        ))
    })?;
    let obj = data.as_object().ok_or_else(|| {
        AuthError::OAuth("ChatGPT OAuth auth file root must be an object".to_string())
    })?;

    if let Some(mode) = obj.get("auth_mode") {
        let valid_modes = [
            "chatgpt",
            "Chatgpt",
            "chatgpt_auth_tokens",
            "ChatgptAuthTokens",
        ];
        if let Some(mode_str) = mode.as_str() {
            if !valid_modes.contains(&mode_str) {
                return Err(AuthError::OAuth(format!(
                    "ChatGPT OAuth auth_mode required, got {:?}",
                    mode_str
                )));
            }
        } else if !mode.is_null() {
            return Err(AuthError::OAuth(format!(
                "ChatGPT OAuth auth_mode required, got {:?}",
                mode
            )));
        }
    }

    let nested_tokens = match obj.get("tokens") {
        Some(Value::Object(tokens)) => Some(tokens),
        None | Some(Value::Null) => None,
        Some(_) => {
            return Err(AuthError::OAuth(
                "ChatGPT OAuth auth file tokens must be an object".to_string(),
            ));
        }
    };
    let root_tokens = root_tokens_from_latest_auth(obj);
    let tokens = nested_tokens
        .or(root_tokens.as_ref())
        .ok_or_else(|| AuthError::OAuth(unsupported_auth_schema_message(obj)))?;

    let access_token = extract_required_string(tokens, "access_token")?;
    let refresh_token_val = extract_required_string(tokens, "refresh_token")?;
    let id_token = extract_required_string(tokens, "id_token")?;

    let id_auth = auth_claims(&id_token)?;
    let access_auth = auth_claims(&access_token)?;

    let account_sources = [
        ("account_id", tokens.get("account_id")),
        ("id_token", id_auth.get("chatgpt_account_id")),
        ("access_token", access_auth.get("chatgpt_account_id")),
    ];
    let mut account_id: Option<&str> = None;
    for (source, value) in account_sources {
        let candidate = match value {
            None | Some(Value::Null) => continue,
            Some(Value::String(value)) if !value.is_empty() => value.as_str(),
            _ => {
                return Err(AuthError::OAuth(format!(
                    "ChatGPT OAuth {} account id is invalid",
                    source
                )))
            }
        };
        if account_id.is_some_and(|existing| existing != candidate) {
            return Err(AuthError::OAuth(
                "ChatGPT OAuth token account ids do not match".to_string(),
            ));
        }
        account_id = Some(candidate);
    }
    let account_id = account_id
        .ok_or_else(|| {
            AuthError::OAuth(
                "ChatGPT OAuth account id not available; rerun codex login".to_string(),
            )
        })?
        .to_string();

    let plan_type = id_auth
        .get("chatgpt_plan_type")
        .or_else(|| access_auth.get("chatgpt_plan_type"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let user_id = id_auth
        .get("chatgpt_user_id")
        .or_else(|| id_auth.get("user_id"))
        .or_else(|| access_auth.get("chatgpt_user_id"))
        .or_else(|| access_auth.get("user_id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    let fedramp = id_auth
        .get("chatgpt_account_is_fedramp")
        .or_else(|| access_auth.get("chatgpt_account_is_fedramp"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let access_expires_at = expiration(&access_token)?;

    Ok(ChatGPTTokenData {
        auth_path: path,
        access_token,
        refresh_token: refresh_token_val,
        id_token,
        account_id,
        plan_type,
        user_id,
        fedramp,
        access_expires_at,
    })
}

fn root_tokens_from_latest_auth(
    data: &serde_json::Map<String, Value>,
) -> Option<serde_json::Map<String, Value>> {
    let names = ["access_token", "refresh_token", "id_token", "account_id"];
    if !names
        .iter()
        .any(|name| data.get(*name).and_then(|v| v.as_str()).is_some())
    {
        return None;
    }
    Some(
        names
            .iter()
            .filter_map(|name| {
                data.get(*name)
                    .map(|value| ((*name).to_string(), value.clone()))
            })
            .collect(),
    )
}

fn unsupported_auth_schema_message(data: &serde_json::Map<String, Value>) -> String {
    let has_file_tokens = ["tokens", "access_token", "refresh_token", "id_token"]
        .iter()
        .any(|key| data.contains_key(*key));
    if data.contains_key("personal_access_token") && !has_file_tokens {
        return "ChatGPT OAuth personal_access_token-only auth is not supported; rerun codex login to create file-backed tokens".to_string();
    }
    if data.contains_key("agent_identity") && !has_file_tokens {
        return "ChatGPT OAuth agent_identity-only auth is not supported; rerun codex login to create file-backed tokens".to_string();
    }
    if data.contains_key("bedrock_api_key") && !has_file_tokens {
        return "ChatGPT OAuth bedrock_api_key-only auth is not supported by the ChatGPT OAuth backend".to_string();
    }
    "ChatGPT OAuth file-backed ChatGPT OAuth tokens are required; rerun codex login".to_string()
}

fn extract_required_string(
    tokens: &serde_json::Map<String, Value>,
    name: &str,
) -> Result<String, AuthError> {
    match tokens.get(name) {
        Some(Value::String(s)) if !s.is_empty() => Ok(s.clone()),
        _ => Err(AuthError::OAuth(format!(
            "ChatGPT OAuth {} is missing",
            name
        ))),
    }
}

pub fn is_auth_locally_available(auth_json_path: Option<&str>) -> bool {
    match load_token_data(auth_json_path) {
        Ok(data) => !data.access_token.is_empty() && !data.account_id.is_empty(),
        Err(_) => false,
    }
}

fn refresh_lock(path: &Path) -> std::sync::Arc<Mutex<()>> {
    let resolved = path.to_path_buf();
    let mut locks = REFRESH_LOCKS.lock().unwrap();
    locks
        .entry(resolved)
        .or_insert_with(|| std::sync::Arc::new(Mutex::new(())))
        .clone()
}

fn write_auth_json(path: &Path, data: &Value) -> Result<(), AuthError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| AuthError::OAuth(format!("failed to create auth directory: {}", e)))?;
    }
    let tmp = path.with_file_name(format!(
        ".{}.tmp-{}",
        path.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    let payload = serde_json::to_string_pretty(data)
        .map_err(|e| AuthError::OAuth(format!("failed to serialize auth data: {}", e)))?
        + "\n";

    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    options.mode(0o600);
    let file = options
        .open(&tmp)
        .map_err(|e| AuthError::OAuth(format!("failed to write temp auth file: {}", e)))?;
    let mut writer = std::io::BufWriter::new(file);
    writer
        .write_all(payload.as_bytes())
        .map_err(|e| AuthError::OAuth(format!("failed to write auth data: {}", e)))?;
    writer
        .flush()
        .map_err(|e| AuthError::OAuth(format!("failed to flush auth data: {}", e)))?;
    drop(writer);

    fs::rename(&tmp, path)
        .map_err(|e| AuthError::OAuth(format!("failed to rename auth file: {}", e)))?;

    let _ = fs::remove_file(&tmp);

    Ok(())
}

pub fn token_for_request(auth_json_path: Option<&str>) -> Result<ChatGPTTokenData, AuthError> {
    let current = load_token_data(auth_json_path)?;
    if !current.expires_within_refresh_window() {
        return Ok(current);
    }
    refresh_from_observed(current, true)
}

pub fn refresh_after_unauthorized(
    current: &ChatGPTTokenData,
) -> Result<ChatGPTTokenData, AuthError> {
    refresh_from_observed(current.clone(), false)
}

fn fail_if_account_changed(
    current: &ChatGPTTokenData,
    latest: &ChatGPTTokenData,
) -> Result<(), AuthError> {
    if current.account_id != latest.account_id {
        return Err(AuthError::Refresh(
            "ChatGPT OAuth account changed while refreshing credentials".to_string(),
        ));
    }
    Ok(())
}

fn access_token_changed(current: &ChatGPTTokenData, latest: &ChatGPTTokenData) -> bool {
    current.access_token != latest.access_token
}

fn other_credentials_changed(current: &ChatGPTTokenData, latest: &ChatGPTTokenData) -> bool {
    current.refresh_token != latest.refresh_token || current.id_token != latest.id_token
}

fn validate_refreshed_token_accounts(payload: &Value, account_id: &str) -> Result<(), AuthError> {
    for name in ["access_token", "id_token"] {
        let Some(value) = payload.get(name).and_then(Value::as_str) else {
            continue;
        };
        let claims = auth_claims(value)?;
        if let Some(claim_account) = claims.get("chatgpt_account_id") {
            if claim_account.as_str() != Some(account_id) {
                return Err(AuthError::Refresh(format!(
                    "ChatGPT OAuth refreshed {} account id does not match current account",
                    name
                )));
            }
        }
    }
    Ok(())
}

fn refresh_from_observed(
    current: ChatGPTTokenData,
    refresh_if_expiring: bool,
) -> Result<ChatGPTTokenData, AuthError> {
    let lock = refresh_lock(&current.auth_path);
    let _guard = lock.lock().unwrap();

    let latest = load_token_data(current.auth_path.to_str())?;
    fail_if_account_changed(&current, &latest)?;
    if access_token_changed(&current, &latest) {
        return Ok(latest);
    }
    if refresh_if_expiring && !latest.expires_within_refresh_window() {
        return Ok(latest);
    }
    let current = latest;
    let endpoint =
        std::env::var(REFRESH_URL_OVERRIDE_ENV).unwrap_or(DEFAULT_REFRESH_URL.to_string());

    let body = serde_json::json!({
        "client_id": CHATGPT_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": current.refresh_token,
    });

    let client = reqwest::blocking::Client::new();
    let response = client
        .post(&endpoint)
        .header("Content-Type", "application/json")
        .timeout(std::time::Duration::from_secs(30))
        .body(serde_json::to_vec(&body).unwrap())
        .send();

    let response = match response {
        Ok(resp) => resp,
        Err(e) => {
            let redacted = redact_text(
                &e.to_string(),
                &[
                    &current.access_token,
                    &current.refresh_token,
                    &current.id_token,
                    &current.account_id,
                ],
            );
            return Err(AuthError::Refresh(format!(
                "ChatGPT OAuth token refresh failed: {}",
                redacted
            )));
        }
    };

    let status = response.status();
    if !status.is_success() {
        let body_text = response.text().unwrap_or_default();
        let redacted = redact_text(
            &body_text,
            &[
                &current.access_token,
                &current.refresh_token,
                &current.id_token,
                &current.account_id,
            ],
        );
        if status.as_u16() == 401 {
            return Err(AuthError::Refresh(format!(
                "ChatGPT OAuth refresh token is invalid; rerun codex login: {}",
                redacted
            )));
        }
        return Err(AuthError::Refresh(format!(
            "ChatGPT OAuth token refresh failed: HTTP {}: {}",
            status.as_u16(),
            redacted
        )));
    }

    let payload: Value = response.json().map_err(|_| {
        AuthError::Refresh("ChatGPT OAuth token refresh returned invalid JSON".to_string())
    })?;

    if !payload.is_object() {
        return Err(AuthError::Refresh(
            "ChatGPT OAuth token refresh returned invalid JSON".to_string(),
        ));
    }
    let access_token = payload
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            AuthError::Refresh(
                "ChatGPT OAuth token refresh response is missing access_token".to_string(),
            )
        })?;
    for name in ["id_token", "refresh_token"] {
        if payload.get(name).is_some()
            && payload
                .get(name)
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .is_none()
        {
            return Err(AuthError::Refresh(format!(
                "ChatGPT OAuth token refresh response has invalid {}",
                name
            )));
        }
    }
    persist_refresh_payload(&current, &payload, access_token)
}

fn persist_refresh_payload(
    current: &ChatGPTTokenData,
    payload: &Value,
    access_token: &str,
) -> Result<ChatGPTTokenData, AuthError> {
    validate_refreshed_token_accounts(payload, &current.account_id)?;

    let latest = load_token_data(current.auth_path.to_str())?;
    fail_if_account_changed(current, &latest)?;
    if access_token_changed(current, &latest) {
        return Ok(latest);
    }
    if other_credentials_changed(current, &latest) {
        return Err(AuthError::Refresh(
            "ChatGPT OAuth credentials changed while token refresh was in flight".to_string(),
        ));
    }

    let auth_raw = fs::read_to_string(&current.auth_path)
        .map_err(|e| AuthError::Refresh(format!("failed to re-read auth file: {}", e)))?;
    let mut data: Value = serde_json::from_str(&auth_raw)
        .map_err(|e| AuthError::Refresh(format!("failed to parse auth file: {}", e)))?;

    apply_refresh_payload(&mut data, payload)?;
    debug_assert_eq!(
        data.get("tokens")
            .and_then(Value::as_object)
            .unwrap_or_else(|| data
                .as_object()
                .expect("auth data was validated as an object"))
            .get("access_token")
            .and_then(Value::as_str),
        Some(access_token)
    );

    let now = Utc::now().format("%Y-%m-%dT%H:%M:%S%.fZ").to_string();
    if let Some(obj) = data.as_object_mut() {
        obj.insert("last_refresh".to_string(), Value::String(now));
    }

    write_auth_json(&current.auth_path, &data)?;
    load_token_data(current.auth_path.to_str())
}

fn apply_refresh_payload(data: &mut Value, payload: &Value) -> Result<(), AuthError> {
    let obj = data
        .as_object_mut()
        .ok_or_else(|| AuthError::Refresh("auth file root must be an object".to_string()))?;
    let has_nested_tokens = matches!(obj.get("tokens"), Some(Value::Object(_)));
    let has_invalid_nested_tokens = !matches!(
        obj.get("tokens"),
        None | Some(Value::Null | Value::Object(_))
    );
    if has_invalid_nested_tokens {
        return Err(AuthError::Refresh(
            "auth file tokens must be an object".to_string(),
        ));
    }
    let tokens = if has_nested_tokens {
        obj.get_mut("tokens")
            .and_then(Value::as_object_mut)
            .expect("nested tokens were validated as an object")
    } else {
        obj
    };

    for name in ["id_token", "access_token", "refresh_token"] {
        if let Some(value) = payload.get(name).and_then(|v| v.as_str()) {
            if !value.is_empty() {
                tokens.insert(name.to_string(), Value::String(value.to_string()));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_jwt_claims_valid() {
        let payload = serde_json::json!({"sub": "user123", "exp": 1700000000});
        let payload_str = serde_json::to_string(&payload).unwrap();
        let encoded = URL_SAFE_NO_PAD.encode(payload_str.as_bytes());
        let jwt = format!("header.{}.signature", encoded);
        let claims = jwt_claims(&jwt).unwrap();
        assert_eq!(claims.get("sub").unwrap().as_str().unwrap(), "user123");
        assert_eq!(claims.get("exp").unwrap().as_i64().unwrap(), 1700000000);
    }

    #[test]
    fn test_jwt_claims_too_few_parts() {
        let claims = jwt_claims("onlyonepart").unwrap();
        assert!(claims.is_empty());
    }

    #[test]
    fn test_jwt_claims_empty_payload() {
        let claims = jwt_claims("header..signature").unwrap();
        assert!(claims.is_empty());
    }

    #[test]
    fn test_jwt_claims_invalid_base64_content() {
        let encoded = URL_SAFE_NO_PAD.encode(b"not json at all {{{");
        let jwt = format!("header.{}.signature", encoded);
        let result = jwt_claims(&jwt);
        assert!(result.is_err());
    }

    #[test]
    fn test_jwt_claims_non_object() {
        let encoded = URL_SAFE_NO_PAD.encode(b"[1,2,3]");
        let jwt = format!("header.{}.signature", encoded);
        let result = jwt_claims(&jwt);
        assert!(result.is_err());
    }

    #[test]
    fn test_redact_text() {
        let text = "token=abc123 and secret=xyz789";
        let redacted = redact_text(text, &["abc123", "xyz789"]);
        assert_eq!(redacted, "token=*** and secret=***");
    }

    #[test]
    fn test_redact_text_empty_values() {
        let text = "nothing to redact";
        let redacted = redact_text(text, &["", ""]);
        assert_eq!(redacted, "nothing to redact");
    }

    #[test]
    fn test_redact_text_longer_first() {
        let text = "abc abcdef";
        let redacted = redact_text(text, &["abc", "abcdef"]);
        assert_eq!(redacted, "*** ***");
    }

    #[cfg(unix)]
    #[test]
    fn test_write_auth_json_sets_owner_only_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = std::env::temp_dir().join(format!("codex_auth_mode_{}", uuid::Uuid::new_v4()));
        let auth_path = dir.join("auth.json");

        write_auth_json(&auth_path, &serde_json::json!({"tokens": {}})).unwrap();

        let mode = fs::metadata(&auth_path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_load_token_data_from_file() {
        let dir = std::env::temp_dir().join(format!("codex_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();

        let auth_claims_payload = serde_json::json!({
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acc-123",
                "chatgpt_plan_type": "pro",
                "chatgpt_user_id": "usr-456",
            },
            "exp": 9999999999i64,
        });
        let encoded_payload = URL_SAFE_NO_PAD.encode(
            serde_json::to_string(&auth_claims_payload)
                .unwrap()
                .as_bytes(),
        );
        let fake_jwt = format!("hdr.{}.sig", encoded_payload);

        let auth_data = serde_json::json!({
            "tokens": {
                "access_token": fake_jwt,
                "refresh_token": "rt_test",
                "id_token": fake_jwt,
            }
        });

        let auth_path = dir.join("auth.json");
        let mut f = fs::File::create(&auth_path).unwrap();
        f.write_all(serde_json::to_string(&auth_data).unwrap().as_bytes())
            .unwrap();

        let token = load_token_data(Some(auth_path.to_str().unwrap())).unwrap();
        assert_eq!(token.account_id, "acc-123");
        assert_eq!(token.plan_type.as_deref(), Some("pro"));
        assert_eq!(token.user_id.as_deref(), Some("usr-456"));
        assert!(!token.expired());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_token_data_latest_root_fields() {
        let dir = std::env::temp_dir().join(format!("codex_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_claims_payload = serde_json::json!({
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acc-root",
                "chatgpt_plan_type": "plus",
                "chatgpt_user_id": "usr-root",
            },
            "exp": 9999999999i64,
        });
        let encoded_payload = URL_SAFE_NO_PAD.encode(
            serde_json::to_string(&auth_claims_payload)
                .unwrap()
                .as_bytes(),
        );
        let fake_jwt = format!("hdr.{}.sig", encoded_payload);
        let auth_data = serde_json::json!({
            "access_token": fake_jwt,
            "refresh_token": "rt-root",
            "id_token": fake_jwt,
            "personal_access_token": "pat-present-but-not-primary",
            "agent_identity": {"id": "agent"},
        });
        let auth_path = dir.join("auth.json");
        fs::write(&auth_path, serde_json::to_string(&auth_data).unwrap()).unwrap();

        let token = load_token_data(Some(auth_path.to_str().unwrap())).unwrap();

        assert_eq!(token.account_id, "acc-root");
        assert_eq!(token.refresh_token, "rt-root");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_root_auth_rejects_present_non_object_tokens_field() {
        let dir = std::env::temp_dir().join(format!("codex_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let claims = serde_json::json!({
            "https://api.openai.com/auth": {"chatgpt_account_id": "acc-root"},
            "exp": 9999999999i64,
        });
        let token = format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap())
        );
        let auth_path = dir.join("auth.json");
        fs::write(
            &auth_path,
            serde_json::to_vec(&serde_json::json!({
                "tokens": ["invalid"],
                "access_token": token,
                "refresh_token": "refresh-root",
                "id_token": token,
                "account_id": "acc-root",
            }))
            .unwrap(),
        )
        .unwrap();

        let error = load_token_data(auth_path.to_str()).unwrap_err();

        assert!(error.to_string().contains("tokens must be an object"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_apply_refresh_payload_preserves_root_auth_layout() {
        let auth_claims_payload = serde_json::json!({
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "acc-new",
            },
            "exp": 9999999999i64,
        });
        let encoded_payload = URL_SAFE_NO_PAD.encode(
            serde_json::to_string(&auth_claims_payload)
                .unwrap()
                .as_bytes(),
        );
        let refreshed = format!("hdr.{}.sig", encoded_payload);
        let mut data = serde_json::json!({
            "access_token": "old-access",
            "refresh_token": "old-refresh",
            "id_token": "old-id",
        });
        let payload = serde_json::json!({
            "access_token": refreshed,
            "refresh_token": "new-refresh",
            "id_token": refreshed,
        });

        apply_refresh_payload(&mut data, &payload).unwrap();
        let tokens = data.as_object().unwrap();

        assert_eq!(
            tokens.get("access_token").unwrap().as_str(),
            Some(refreshed.as_str())
        );
        assert_eq!(
            tokens.get("refresh_token").unwrap().as_str(),
            Some("new-refresh")
        );
        assert_eq!(
            tokens.get("id_token").unwrap().as_str(),
            Some(refreshed.as_str())
        );
        assert!(!tokens.contains_key("tokens"));
    }

    #[test]
    fn test_partial_refresh_preserves_root_auth_credentials() {
        let mut data = serde_json::json!({
            "access_token": "old-access",
            "refresh_token": "old-refresh",
            "id_token": "old-id",
        });
        apply_refresh_payload(
            &mut data,
            &serde_json::json!({"access_token": "new-access"}),
        )
        .unwrap();
        let tokens = data.as_object().unwrap();

        assert_eq!(tokens["access_token"], "new-access");
        assert_eq!(tokens["refresh_token"], "old-refresh");
        assert_eq!(tokens["id_token"], "old-id");
        assert!(!tokens.contains_key("tokens"));
    }

    #[test]
    fn test_load_rejects_mismatched_account_claims() {
        let dir = std::env::temp_dir().join(format!("codex_accounts_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        let access = format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD.encode(
                serde_json::to_vec(&serde_json::json!({
                    "exp": 9999999999i64,
                    "https://api.openai.com/auth": {"chatgpt_account_id": "acc-new"}
                }))
                .unwrap()
            )
        );
        let id = format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD.encode(
                serde_json::to_vec(&serde_json::json!({
                    "https://api.openai.com/auth": {"chatgpt_account_id": "acc-old"}
                }))
                .unwrap()
            )
        );
        fs::write(
            &auth_path,
            serde_json::to_vec(&serde_json::json!({
                "tokens": {"access_token": access, "refresh_token": "refresh", "id_token": id}
            }))
            .unwrap(),
        )
        .unwrap();

        let error = load_token_data(auth_path.to_str()).unwrap_err();

        assert!(error.to_string().contains("account ids do not match"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_persist_refresh_payload_compare_and_set_and_account_validation() {
        let dir = std::env::temp_dir().join(format!("codex_cas_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        let claims = serde_json::json!({
            "https://api.openai.com/auth": {"chatgpt_account_id": "acc-shared"}
        });
        let id = format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&claims).unwrap())
        );
        let access = |version: &str, account: &str| {
            format!(
                "header.{}.signature",
                URL_SAFE_NO_PAD.encode(
                    serde_json::to_vec(&serde_json::json!({
                        "exp": 9999999999i64,
                        "version": version,
                        "https://api.openai.com/auth": {"chatgpt_account_id": account}
                    }))
                    .unwrap()
                )
            )
        };
        let old_access = access("old", "acc-shared");
        fs::write(
            &auth_path,
            serde_json::to_vec(&serde_json::json!({
                "tokens": {
                    "access_token": old_access,
                    "refresh_token": "refresh-old",
                    "id_token": id
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let observed = load_token_data(auth_path.to_str()).unwrap();

        let external_access = access("external", "acc-shared");
        fs::write(
            &auth_path,
            serde_json::to_vec(&serde_json::json!({
                "tokens": {
                    "access_token": external_access,
                    "refresh_token": observed.refresh_token,
                    "id_token": observed.id_token
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let response_access = access("response", "acc-shared");
        let reused = persist_refresh_payload(
            &observed,
            &serde_json::json!({"access_token": response_access}),
            &response_access,
        )
        .unwrap();
        assert_eq!(reused.access_token, external_access);

        let mismatched_access = access("wrong", "acc-other");
        let error = persist_refresh_payload(
            &reused,
            &serde_json::json!({"access_token": mismatched_access}),
            &mismatched_access,
        )
        .unwrap_err();
        assert!(error.to_string().contains("does not match current account"));

        fs::write(
            &auth_path,
            serde_json::to_vec(&serde_json::json!({
                "tokens": {
                    "access_token": reused.access_token,
                    "refresh_token": "refresh-raced",
                    "id_token": reused.id_token
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let error = persist_refresh_payload(
            &reused,
            &serde_json::json!({"access_token": response_access}),
            &response_access,
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("changed while token refresh was in flight"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_account_switch_and_unchanged_access_token_changes_are_distinct() {
        let observed = ChatGPTTokenData {
            auth_path: PathBuf::from("auth.json"),
            access_token: "access".to_string(),
            refresh_token: "refresh-old".to_string(),
            id_token: "id-old".to_string(),
            account_id: "acc-old".to_string(),
            plan_type: None,
            user_id: None,
            fedramp: false,
            access_expires_at: None,
        };
        let mut latest = observed.clone();
        latest.refresh_token = "refresh-latest".to_string();
        latest.id_token = "id-latest".to_string();
        assert!(!access_token_changed(&observed, &latest));
        assert!(other_credentials_changed(&observed, &latest));
        assert!(fail_if_account_changed(&observed, &latest).is_ok());
        latest.account_id = "acc-new".to_string();
        assert!(fail_if_account_changed(&observed, &latest).is_err());
    }

    #[test]
    fn test_refresh_error_redaction_includes_account_id() {
        let redacted = redact_text(
            "request failed for account acc-secret",
            &["access", "refresh", "id", "acc-secret"],
        );
        assert_eq!(redacted, "request failed for account ***");
    }

    #[test]
    fn test_refresh_window_and_matching_account_change_detection() {
        let mut observed = ChatGPTTokenData {
            auth_path: PathBuf::from("auth.json"),
            access_token: "old-access".to_string(),
            refresh_token: "old-refresh".to_string(),
            id_token: "old-id".to_string(),
            account_id: "acc-shared".to_string(),
            plan_type: None,
            user_id: None,
            fedramp: false,
            access_expires_at: Some(Utc::now() + Duration::minutes(4)),
        };
        assert!(observed.expires_within_refresh_window());
        observed.access_expires_at = Some(Utc::now() + Duration::minutes(6));
        assert!(!observed.expires_within_refresh_window());

        let mut latest = observed.clone();
        latest.access_token = "new-access".to_string();
        assert!(access_token_changed(&observed, &latest));
        latest.account_id = "different-account".to_string();
        assert!(fail_if_account_changed(&observed, &latest).is_err());
    }

    #[test]
    fn test_refresh_after_unauthorized_reloads_coalesced_matching_file_update() {
        let dir = std::env::temp_dir().join(format!("codex_refresh_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        let id_claims = serde_json::json!({
            "https://api.openai.com/auth": {"chatgpt_account_id": "acc-shared"},
        });
        let id_token = format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD.encode(serde_json::to_vec(&id_claims).unwrap())
        );
        let old_access = format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD
                .encode(serde_json::to_vec(&serde_json::json!({"exp": 9999999999i64})).unwrap())
        );
        fs::write(
            &auth_path,
            serde_json::to_vec(&serde_json::json!({
                "tokens": {
                    "access_token": old_access,
                    "refresh_token": "refresh-old",
                    "id_token": id_token.clone(),
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let observed = load_token_data(auth_path.to_str()).unwrap();
        let lock = refresh_lock(&auth_path);
        let guard = lock.lock().unwrap();
        let worker = std::thread::spawn(move || refresh_after_unauthorized(&observed).unwrap());
        let new_access = format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD
                .encode(serde_json::to_vec(&serde_json::json!({"exp": 9999999999i64})).unwrap())
        ) + "-new";
        fs::write(
            &auth_path,
            serde_json::to_vec(&serde_json::json!({
                "tokens": {
                    "access_token": new_access,
                    "refresh_token": "refresh-from-file",
                    "id_token": id_token,
                }
            }))
            .unwrap(),
        )
        .unwrap();
        drop(guard);

        let refreshed = worker.join().unwrap();
        assert_eq!(refreshed.refresh_token, "refresh-from-file");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_load_token_data_pat_only_error() {
        let dir = std::env::temp_dir().join(format!("codex_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        fs::write(&auth_path, r#"{"personal_access_token":"pat-only"}"#).unwrap();

        let result = load_token_data(Some(auth_path.to_str().unwrap()));

        assert!(result
            .unwrap_err()
            .to_string()
            .contains("personal_access_token-only auth is not supported"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_token_data_agent_only_error() {
        let dir = std::env::temp_dir().join(format!("codex_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        fs::write(&auth_path, r#"{"agent_identity":{"id":"agent"}}"#).unwrap();

        let result = load_token_data(Some(auth_path.to_str().unwrap()));

        assert!(result
            .unwrap_err()
            .to_string()
            .contains("agent_identity-only auth is not supported"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_token_data_bedrock_only_error() {
        let dir = std::env::temp_dir().join(format!("codex_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        fs::write(&auth_path, r#"{"bedrock_api_key":"bedrock-only"}"#).unwrap();

        let result = load_token_data(Some(auth_path.to_str().unwrap()));

        assert!(result
            .unwrap_err()
            .to_string()
            .contains("bedrock_api_key-only auth is not supported"));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_load_token_data_missing_file() {
        let result = load_token_data(Some("/tmp/nonexistent_codex_auth_test_42.json"));
        assert!(matches!(result, Err(AuthError::Missing(_))));
    }

    #[test]
    fn test_resolve_auth_path_default() {
        std::env::remove_var("CODEX_HOME");
        let path = resolve_auth_path(None);
        assert!(path.to_str().unwrap().ends_with(".codex/auth.json"));
    }

    #[test]
    fn test_resolve_auth_path_explicit() {
        let path = resolve_auth_path(Some("/tmp/custom/auth.json"));
        assert_eq!(path, PathBuf::from("/tmp/custom/auth.json"));
    }
}
