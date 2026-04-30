use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::{DateTime, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use thiserror::Error;

pub const CHATGPT_OAUTH_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
pub const DEFAULT_AUTH_PATH: &str = "~/.codex/auth.json";
pub const DEFAULT_REFRESH_URL: &str = "https://auth.openai.com/oauth/token";
pub const REFRESH_URL_OVERRIDE_ENV: &str = "CODEX_REFRESH_TOKEN_URL_OVERRIDE";

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

    let tokens = obj
        .get("tokens")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            AuthError::OAuth("ChatGPT OAuth token data is not available".to_string())
        })?;

    let access_token = extract_required_string(tokens, "access_token")?;
    let refresh_token_val = extract_required_string(tokens, "refresh_token")?;
    let id_token = extract_required_string(tokens, "id_token")?;

    let id_auth = auth_claims(&id_token)?;
    let access_auth = auth_claims(&access_token)?;

    let account_id = tokens
        .get("account_id")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .or_else(|| {
            id_auth
                .get("chatgpt_account_id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
        })
        .or_else(|| {
            access_auth
                .get("chatgpt_account_id")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
        })
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
        fs::create_dir_all(parent).map_err(|e| {
            AuthError::OAuth(format!("failed to create auth directory: {}", e))
        })?;
    }
    let tmp = path.with_file_name(format!(
        ".{}.tmp-{}",
        path.file_name().unwrap().to_string_lossy(),
        std::process::id()
    ));
    let payload = serde_json::to_string_pretty(data)
        .map_err(|e| AuthError::OAuth(format!("failed to serialize auth data: {}", e)))?
        + "\n";

    let file = fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
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

pub fn do_refresh_token(auth_json_path: Option<&str>) -> Result<ChatGPTTokenData, AuthError> {
    let current = load_token_data(auth_json_path)?;
    let lock = refresh_lock(&current.auth_path);
    let _guard = lock.lock().unwrap();

    let current = load_token_data(auth_json_path)?;
    let endpoint = std::env::var(REFRESH_URL_OVERRIDE_ENV).unwrap_or(DEFAULT_REFRESH_URL.to_string());

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
            return Err(AuthError::Refresh(format!(
                "ChatGPT OAuth token refresh failed: {}",
                e
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

    let payload: Value = response
        .json()
        .map_err(|_| AuthError::Refresh("ChatGPT OAuth token refresh returned invalid JSON".to_string()))?;

    if !payload.is_object() {
        return Err(AuthError::Refresh(
            "ChatGPT OAuth token refresh returned invalid JSON".to_string(),
        ));
    }

    let auth_raw = fs::read_to_string(&current.auth_path).map_err(|e| {
        AuthError::Refresh(format!("failed to re-read auth file: {}", e))
    })?;
    let mut data: Value = serde_json::from_str(&auth_raw).map_err(|e| {
        AuthError::Refresh(format!("failed to parse auth file: {}", e))
    })?;

    if let Some(tokens) = data.get_mut("tokens").and_then(|v| v.as_object_mut()) {
        if let Some(id_tok) = payload.get("id_token").and_then(|v| v.as_str()) {
            if !id_tok.is_empty() {
                tokens.insert("id_token".to_string(), Value::String(id_tok.to_string()));
            }
        }
        if let Some(acc_tok) = payload.get("access_token").and_then(|v| v.as_str()) {
            if !acc_tok.is_empty() {
                tokens.insert(
                    "access_token".to_string(),
                    Value::String(acc_tok.to_string()),
                );
            }
        }
        if let Some(ref_tok) = payload.get("refresh_token").and_then(|v| v.as_str()) {
            if !ref_tok.is_empty() {
                tokens.insert(
                    "refresh_token".to_string(),
                    Value::String(ref_tok.to_string()),
                );
            }
        }
    }

    let now = Utc::now().format("%Y-%m-%dT%H:%M:%S%.fZ").to_string();
    if let Some(obj) = data.as_object_mut() {
        obj.insert("last_refresh".to_string(), Value::String(now));
    }

    write_auth_json(&current.auth_path, &data)?;
    load_token_data(auth_json_path)
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
        let encoded_payload =
            URL_SAFE_NO_PAD.encode(serde_json::to_string(&auth_claims_payload).unwrap().as_bytes());
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
