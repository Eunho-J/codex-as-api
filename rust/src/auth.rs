use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;
use chrono::{DateTime, Duration, Utc};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
#[cfg(windows)]
use std::os::windows::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use thiserror::Error;

use crate::strict_json;

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
    #[error("{message}")]
    RefreshUpstreamHttp { status: u16, message: String },
    #[error("{0}")]
    RefreshTransport(String),
    #[error("{0}")]
    RefreshProtocol(String),
    #[error("{0}")]
    Internal(String),
    #[error("{primary}; temporary auth file cleanup also failed: {cleanup}")]
    WriteCleanup {
        primary: Box<AuthError>,
        cleanup: std::io::Error,
    },
}

#[derive(Debug, Clone)]
pub struct ChatGPTTokenData {
    pub auth_path: PathBuf,
    pub access_token: String,
    pub refresh_token: String,
    pub id_token: String,
    pub account_id: String,
    pub fedramp: bool,
    pub access_expires_at: Option<DateTime<Utc>>,
}

fn is_ascii_visible_token(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| (0x21..=0x7e).contains(&byte))
}

impl ChatGPTTokenData {
    pub fn expires_within_refresh_window(&self) -> bool {
        match self.access_expires_at {
            Some(exp) => exp <= Utc::now() + Duration::minutes(REFRESH_WINDOW_MINUTES),
            None => false,
        }
    }
}

pub fn resolve_auth_path(raw: Option<&str>) -> Result<PathBuf, AuthError> {
    if let Some(path) = raw {
        if path.trim().is_empty() {
            return Err(AuthError::Missing(
                "ChatGPT OAuth auth path must not be empty when provided".to_string(),
            ));
        }
        return expand_tilde(path);
    }

    match std::env::var("CODEX_HOME") {
        Ok(path) if path.trim().is_empty() => Err(AuthError::Missing(
            "CODEX_HOME must not be empty when provided".to_string(),
        )),
        Ok(path) => Ok(expand_tilde(&path)?.join("auth.json")),
        Err(std::env::VarError::NotPresent) => expand_tilde(DEFAULT_AUTH_PATH),
        Err(std::env::VarError::NotUnicode(_)) => Err(AuthError::Missing(
            "CODEX_HOME must contain valid Unicode".to_string(),
        )),
    }
}

fn expand_tilde(path: &str) -> Result<PathBuf, AuthError> {
    let tilde_suffix = if path == "~" {
        Some(None)
    } else {
        path.strip_prefix("~/").map(Some)
    };
    if let Some(rest) = tilde_suffix {
        let home = home_directory_from_environment()?;
        return Ok(match rest {
            Some(rest) => home.join(rest),
            None => home,
        });
    }
    Ok(PathBuf::from(path))
}

#[cfg(not(windows))]
fn home_directory_from_environment() -> Result<PathBuf, AuthError> {
    let home = match std::env::var("HOME") {
        Ok(home) if !home.trim().is_empty() => home,
        Ok(_) => {
            return Err(AuthError::Missing(
                "HOME must not be empty when resolving the ChatGPT OAuth auth path".to_string(),
            ));
        }
        Err(std::env::VarError::NotPresent) => {
            return Err(AuthError::Missing(
                "HOME is required to resolve the ChatGPT OAuth auth path".to_string(),
            ));
        }
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(AuthError::Missing(
                "HOME must contain valid Unicode when resolving the ChatGPT OAuth auth path"
                    .to_string(),
            ));
        }
    };
    Ok(PathBuf::from(home))
}

#[cfg(windows)]
fn home_directory_from_environment() -> Result<PathBuf, AuthError> {
    for name in ["HOME", "USERPROFILE"] {
        match std::env::var(name) {
            Ok(home) if !home.trim().is_empty() => return Ok(PathBuf::from(home)),
            Ok(_) | Err(std::env::VarError::NotPresent) => continue,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err(AuthError::Missing(format!(
                    "{name} must contain valid Unicode when resolving the ChatGPT OAuth auth path"
                )));
            }
        }
    }
    Err(AuthError::Missing(
        "HOME or USERPROFILE is required to resolve the ChatGPT OAuth auth path".to_string(),
    ))
}

pub fn jwt_claims(jwt: &str) -> Result<serde_json::Map<String, Value>, AuthError> {
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 || parts.iter().any(|part| part.is_empty()) {
        return Err(AuthError::OAuth(
            "invalid ChatGPT OAuth JWT structure".to_string(),
        ));
    }
    let payload = parts[1];
    if payload.len() % 4 == 1
        || !payload
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
    {
        return Err(AuthError::OAuth(
            "invalid ChatGPT OAuth JWT payload".to_string(),
        ));
    }
    let decoded = URL_SAFE_NO_PAD
        .decode(payload)
        .map_err(|_| AuthError::OAuth("invalid ChatGPT OAuth JWT payload".to_string()))?;
    if URL_SAFE_NO_PAD.encode(&decoded) != payload {
        return Err(AuthError::OAuth(
            "invalid ChatGPT OAuth JWT payload".to_string(),
        ));
    }
    let value = strict_json::parse_slice(&decoded)
        .map_err(|_| AuthError::OAuth("invalid ChatGPT OAuth JWT payload".to_string()))?;
    match value {
        Value::Object(map) => Ok(map),
        _ => Err(AuthError::OAuth(
            "invalid ChatGPT OAuth JWT claims".to_string(),
        )),
    }
}

fn expiration(jwt: &str) -> Result<Option<DateTime<Utc>>, AuthError> {
    let claims = jwt_claims(jwt)?;
    match claims.get("exp") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => {
            let timestamp = strict_json::as_js_safe_integer(number).ok_or_else(|| {
                AuthError::OAuth(
                    "ChatGPT OAuth JWT exp claim must be an integral number".to_string(),
                )
            })?;
            DateTime::from_timestamp(timestamp, 0)
                .map(Some)
                .ok_or_else(|| {
                    AuthError::OAuth("ChatGPT OAuth JWT exp claim is out of range".to_string())
                })
        }
        Some(_) => Err(AuthError::OAuth(
            "ChatGPT OAuth JWT exp claim must be an integral number".to_string(),
        )),
    }
}

fn auth_claims(jwt: &str) -> Result<serde_json::Map<String, Value>, AuthError> {
    let claims = jwt_claims(jwt)?;
    match claims.get("https://api.openai.com/auth") {
        Some(Value::Object(map)) => Ok(map.clone()),
        None => Ok(serde_json::Map::new()),
        Some(_) => Err(AuthError::OAuth(
            "ChatGPT OAuth JWT auth claim must be an object".to_string(),
        )),
    }
}

pub fn redact_text(text: &str, values: &[&str]) -> String {
    let mut sorted: Vec<&str> = values.iter().filter(|v| !v.is_empty()).copied().collect();
    sorted.sort_by_key(|value| std::cmp::Reverse(value.len()));
    let mut redacted = text.to_string();
    let marker = if sorted.iter().any(|value| "***".contains(value)) {
        ""
    } else {
        "***"
    };
    for v in sorted {
        redacted = redacted.replace(v, marker);
    }
    while values
        .iter()
        .filter(|value| !value.is_empty())
        .any(|value| redacted.contains(value))
    {
        for value in values.iter().filter(|value| !value.is_empty()) {
            redacted = redacted.replace(value, "");
        }
    }
    redacted
}

pub fn load_token_data(auth_json_path: Option<&str>) -> Result<ChatGPTTokenData, AuthError> {
    let path = resolve_auth_path(auth_json_path)?;
    let raw = fs::read_to_string(&path).map_err(|e| {
        if e.kind() == std::io::ErrorKind::NotFound {
            AuthError::Missing("ChatGPT OAuth auth file not found".to_string())
        } else {
            AuthError::Missing("ChatGPT OAuth auth file is unavailable".to_string())
        }
    })?;
    let data = strict_json::parse_str(&raw)
        .map_err(|_| AuthError::OAuth("ChatGPT OAuth auth file is invalid JSON".to_string()))?;
    token_data_from_document(&data, path)
}

fn token_data_from_document(data: &Value, path: PathBuf) -> Result<ChatGPTTokenData, AuthError> {
    let obj = data.as_object().ok_or_else(|| {
        AuthError::OAuth("ChatGPT OAuth auth file root must be an object".to_string())
    })?;

    match obj.get("auth_mode") {
        None | Some(Value::Null) => None,
        Some(Value::String(mode)) if mode == "chatgpt" => Some(mode),
        Some(_) => {
            return Err(AuthError::OAuth(
                "ChatGPT OAuth auth_mode is unsupported".to_string(),
            ));
        }
    };

    let nested_tokens = match obj.get("tokens") {
        Some(Value::Object(tokens)) => Some(tokens),
        None => None,
        Some(_) => {
            return Err(AuthError::OAuth(
                "ChatGPT OAuth auth file tokens must be an object".to_string(),
            ));
        }
    };
    let tokens =
        nested_tokens.ok_or_else(|| AuthError::OAuth(unsupported_auth_schema_message(obj)))?;
    if [
        "access_token",
        "refresh_token",
        "id_token",
        "chatgptAuthTokens",
    ]
    .iter()
    .any(|name| obj.contains_key(*name))
    {
        return Err(AuthError::OAuth(
            "ChatGPT OAuth auth file mixes canonical tokens with unsupported root credentials"
                .to_string(),
        ));
    }

    let access_token = extract_required_string(tokens, "access_token")?;
    if !is_ascii_visible_token(&access_token) {
        return Err(AuthError::OAuth(
            "ChatGPT OAuth access_token is invalid".to_string(),
        ));
    }
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
            Some(Value::String(value)) if !value.trim().is_empty() => value.as_str(),
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
    if !is_ascii_visible_token(&account_id) {
        return Err(AuthError::OAuth(
            "ChatGPT OAuth account id is invalid".to_string(),
        ));
    }

    consistent_optional_string_claim(
        &[
            (&id_auth, "chatgpt_plan_type"),
            (&access_auth, "chatgpt_plan_type"),
        ],
        "plan type",
    )?;
    consistent_optional_string_claim(
        &[
            (&id_auth, "chatgpt_user_id"),
            (&id_auth, "user_id"),
            (&access_auth, "chatgpt_user_id"),
            (&access_auth, "user_id"),
        ],
        "user id",
    )?;

    let fedramp_claim =
        |claims: &serde_json::Map<String, Value>| match claims.get("chatgpt_account_is_fedramp") {
            None => Ok(None),
            Some(Value::Bool(value)) => Ok(Some(*value)),
            Some(_) => Err(AuthError::OAuth(
                "ChatGPT OAuth chatgpt_account_is_fedramp claim must be a boolean".to_string(),
            )),
        };
    let id_fedramp = fedramp_claim(&id_auth)?;
    let access_fedramp = fedramp_claim(&access_auth)?;
    if matches!((id_fedramp, access_fedramp), (Some(id), Some(access)) if id != access) {
        return Err(AuthError::OAuth(
            "ChatGPT OAuth fedramp claims do not match".to_string(),
        ));
    }
    let fedramp = id_fedramp.or(access_fedramp).unwrap_or(false);

    let access_expires_at = expiration(&access_token)?;

    Ok(ChatGPTTokenData {
        auth_path: path,
        access_token,
        refresh_token: refresh_token_val,
        id_token,
        account_id,
        fedramp,
        access_expires_at,
    })
}

fn consistent_optional_string_claim(
    sources: &[(&serde_json::Map<String, Value>, &str)],
    field: &str,
) -> Result<Option<String>, AuthError> {
    let mut selected: Option<&str> = None;
    for (claims, key) in sources {
        let Some(value) = claims.get(*key) else {
            continue;
        };
        let Value::String(value) = value else {
            return Err(AuthError::OAuth(format!(
                "ChatGPT OAuth {field} claim must be a non-empty string"
            )));
        };
        if value.trim().is_empty() {
            return Err(AuthError::OAuth(format!(
                "ChatGPT OAuth {field} claim must be a non-empty string"
            )));
        }
        if selected.is_some_and(|existing| existing != value) {
            return Err(AuthError::OAuth(format!(
                "ChatGPT OAuth {field} claims do not match"
            )));
        }
        selected = Some(value);
    }
    Ok(selected.map(str::to_string))
}

fn unsupported_auth_schema_message(data: &serde_json::Map<String, Value>) -> String {
    let has_file_tokens = data.contains_key("tokens");
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
        Some(Value::String(s)) if !s.trim().is_empty() => Ok(s.clone()),
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

fn refresh_lock(path: &Path) -> Result<std::sync::Arc<Mutex<()>>, AuthError> {
    let resolved = path.to_path_buf();
    let mut locks = REFRESH_LOCKS
        .lock()
        .map_err(|_| AuthError::Refresh("OAuth refresh lock registry is poisoned".to_string()))?;
    Ok(locks
        .entry(resolved)
        .or_insert_with(|| std::sync::Arc::new(Mutex::new(())))
        .clone())
}

fn write_auth_json(path: &Path, data: &Value) -> Result<(), AuthError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| AuthError::OAuth(format!("failed to create auth directory: {}", e)))?;
    }
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| AuthError::OAuth("auth path must name a UTF-8 file".to_string()))?;
    let tmp = path.with_file_name(format!(
        ".{file_name}.tmp-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    let payload = serde_json::to_string_pretty(data)
        .map_err(|e| AuthError::OAuth(format!("failed to serialize auth data: {}", e)))?
        + "\n";

    let mut options = fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    options.mode(0o600);
    let write_result = (|| {
        let mut file = options
            .open(&tmp)
            .map_err(|e| AuthError::OAuth(format!("failed to write temp auth file: {e}")))?;
        file.write_all(payload.as_bytes())
            .map_err(|e| AuthError::OAuth(format!("failed to write auth data: {e}")))?;
        file.sync_all()
            .map_err(|e| AuthError::OAuth(format!("failed to sync auth data: {e}")))?;
        drop(file);

        replace_auth_file(&tmp, path)?;
        sync_auth_directory(path)?;
        Ok(())
    })();

    finish_auth_write(write_result, fs::remove_file(&tmp))
}

#[cfg(not(windows))]
fn replace_auth_file(tmp: &Path, path: &Path) -> Result<(), AuthError> {
    fs::rename(tmp, path)
        .map_err(|error| AuthError::OAuth(format!("failed to replace auth file: {error}")))
}

#[cfg(windows)]
fn replace_auth_file(tmp: &Path, path: &Path) -> Result<(), AuthError> {
    use windows_sys::Win32::Storage::FileSystem::{
        MoveFileExW, MOVEFILE_REPLACE_EXISTING, MOVEFILE_WRITE_THROUGH,
    };

    let source: Vec<u16> = tmp.as_os_str().encode_wide().chain(Some(0)).collect();
    let destination: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
    let replaced = unsafe {
        MoveFileExW(
            source.as_ptr(),
            destination.as_ptr(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH,
        )
    };
    if replaced == 0 {
        return Err(AuthError::OAuth(format!(
            "failed to replace auth file: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(())
}

#[cfg(unix)]
fn sync_auth_directory(path: &Path) -> Result<(), AuthError> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let directory = fs::File::open(parent)
        .map_err(|e| AuthError::OAuth(format!("failed to open auth directory for sync: {e}")))?;
    directory
        .sync_all()
        .map_err(|e| AuthError::OAuth(format!("failed to sync auth directory: {e}")))
}

#[cfg(not(unix))]
fn sync_auth_directory(_path: &Path) -> Result<(), AuthError> {
    Ok(())
}

fn finish_auth_write(
    primary: Result<(), AuthError>,
    cleanup: std::io::Result<()>,
) -> Result<(), AuthError> {
    let cleanup = match cleanup {
        Ok(()) => None,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => Some(error),
    };
    match (primary, cleanup) {
        (Ok(()), None) => Ok(()),
        (Err(primary), None) => Err(primary),
        (Ok(()), Some(cleanup)) => Err(AuthError::OAuth(format!(
            "failed to clean up temporary auth file: {cleanup}"
        ))),
        (Err(primary), Some(cleanup)) => Err(AuthError::WriteCleanup {
            primary: Box::new(primary),
            cleanup,
        }),
    }
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

fn refresh_url_from_environment() -> Result<String, AuthError> {
    let value = match std::env::var(REFRESH_URL_OVERRIDE_ENV) {
        Ok(value) if value.trim().is_empty() => {
            return Err(AuthError::Refresh(format!(
                "{REFRESH_URL_OVERRIDE_ENV} must not be empty when provided"
            )));
        }
        Ok(value) => value,
        Err(std::env::VarError::NotPresent) => DEFAULT_REFRESH_URL.to_string(),
        Err(std::env::VarError::NotUnicode(_)) => {
            return Err(AuthError::Refresh(format!(
                "{REFRESH_URL_OVERRIDE_ENV} must contain valid Unicode"
            )));
        }
    };
    validate_refresh_url(value)
}

fn validate_refresh_url(value: String) -> Result<String, AuthError> {
    if value != value.trim() {
        return Err(AuthError::Refresh(format!(
            "{REFRESH_URL_OVERRIDE_ENV} must not contain surrounding whitespace"
        )));
    }
    if value
        .chars()
        .any(|character| character.is_whitespace() || character.is_control())
    {
        return Err(AuthError::Refresh(format!(
            "{REFRESH_URL_OVERRIDE_ENV} must not contain whitespace or control characters"
        )));
    }
    let bytes = value.as_bytes();
    if (0..bytes.len()).any(|index| {
        bytes[index] == b'%'
            && (index + 2 >= bytes.len()
                || !bytes[index + 1].is_ascii_hexdigit()
                || !bytes[index + 2].is_ascii_hexdigit())
    }) {
        return Err(AuthError::Refresh(format!(
            "{REFRESH_URL_OVERRIDE_ENV} contains a malformed percent escape"
        )));
    }
    let has_raw_authority = value
        .split_once("://")
        .is_some_and(|(_, rest)| !rest.split('/').next().unwrap_or("").is_empty());
    let parsed = reqwest::Url::parse(&value).map_err(|_| {
        AuthError::Refresh(format!(
            "{REFRESH_URL_OVERRIDE_ENV} must be a valid HTTP(S) URL"
        ))
    })?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !has_raw_authority
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(AuthError::Refresh(format!(
            "{REFRESH_URL_OVERRIDE_ENV} must be an absolute HTTP(S) URL without credentials, query, or fragment"
        )));
    }
    Ok(value)
}

pub fn validate_auth_environment() -> Result<(), String> {
    refresh_url_from_environment()
        .map(|_| ())
        .map_err(|error| error.to_string())
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
    let lock = refresh_lock(&current.auth_path)?;
    let _guard = lock
        .lock()
        .map_err(|_| AuthError::Refresh("OAuth refresh lock is poisoned".to_string()))?;

    let latest = load_token_data(current.auth_path.to_str())?;
    fail_if_account_changed(&current, &latest)?;
    if access_token_changed(&current, &latest) {
        return Ok(latest);
    }
    if refresh_if_expiring && !latest.expires_within_refresh_window() {
        return Ok(latest);
    }
    let current = latest;
    let endpoint = refresh_url_from_environment()?;

    let body = serde_json::json!({
        "client_id": CHATGPT_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": current.refresh_token,
    });

    let client = reqwest::blocking::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|error| AuthError::Internal(error.to_string()))?;
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
            return Err(AuthError::RefreshTransport(format!(
                "ChatGPT OAuth token refresh failed: {}",
                redacted
            )));
        }
    };

    let status = response.status();
    if !status.is_success() {
        let body_text = response
            .text()
            .unwrap_or_else(|_| "could not read upstream error body".to_string());
        let redacted = redact_text(
            &body_text,
            &[
                &current.access_token,
                &current.refresh_token,
                &current.id_token,
                &current.account_id,
            ],
        );
        if matches!(status.as_u16(), 400 | 401) {
            return Err(AuthError::Refresh(format!(
                "ChatGPT OAuth refresh token is invalid; rerun codex login: {}",
                redacted
            )));
        }
        return Err(AuthError::RefreshUpstreamHttp {
            status: status.as_u16(),
            message: format!(
                "ChatGPT OAuth token refresh failed: HTTP {}: {}",
                status.as_u16(),
                redacted
            ),
        });
    }

    let response_bytes = response.bytes().map_err(|error| {
        AuthError::RefreshTransport(format!(
            "ChatGPT OAuth token refresh response read failed: {}",
            redact_text(
                &error.to_string(),
                &[
                    &current.access_token,
                    &current.refresh_token,
                    &current.id_token,
                    &current.account_id,
                ],
            )
        ))
    })?;
    let response_text = std::str::from_utf8(&response_bytes).map_err(|_| {
        AuthError::RefreshProtocol("ChatGPT OAuth token refresh returned invalid JSON".to_string())
    })?;
    let payload = strict_json::parse_str(response_text).map_err(|_| {
        AuthError::RefreshProtocol("ChatGPT OAuth token refresh returned invalid JSON".to_string())
    })?;

    if !payload.is_object() {
        return Err(AuthError::RefreshProtocol(
            "ChatGPT OAuth token refresh returned invalid JSON".to_string(),
        ));
    }
    let access_token = payload
        .get("access_token")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            AuthError::RefreshProtocol(
                "ChatGPT OAuth token refresh response is missing access_token".to_string(),
            )
        })?;
    if !is_ascii_visible_token(access_token) {
        return Err(AuthError::RefreshProtocol(
            "ChatGPT OAuth token refresh response has invalid access_token".to_string(),
        ));
    }
    for name in ["id_token", "refresh_token"] {
        if payload.get(name).is_some()
            && payload
                .get(name)
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .is_none()
        {
            return Err(AuthError::RefreshProtocol(format!(
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
    if !is_ascii_visible_token(access_token) {
        return Err(AuthError::Refresh(
            "ChatGPT OAuth token refresh response has invalid access_token".to_string(),
        ));
    }
    validate_refreshed_token_accounts(payload, &current.account_id)?;

    let auth_raw = fs::read_to_string(&current.auth_path)
        .map_err(|_| AuthError::Internal("failed to re-read auth file".to_string()))?;
    let mut data = strict_json::parse_str(&auth_raw)
        .map_err(|_| AuthError::Internal("failed to parse auth file".to_string()))?;

    let exact = token_data_from_document(&data, current.auth_path.clone()).map_err(|_| {
        AuthError::Internal("ChatGPT OAuth auth file changed to an invalid schema".to_string())
    })?;
    fail_if_account_changed(current, &exact)?;
    if access_token_changed(current, &exact) {
        return Ok(exact);
    }
    if other_credentials_changed(current, &exact) {
        return Err(AuthError::Refresh(
            "ChatGPT OAuth credentials changed while token refresh was in flight".to_string(),
        ));
    }

    apply_refresh_payload(&mut data, payload)?;
    debug_assert_eq!(
        data.get("tokens")
            .and_then(Value::as_object)
            .expect("auth tokens were validated as an object")
            .get("access_token")
            .and_then(Value::as_str),
        Some(access_token)
    );

    let now = Utc::now().format("%Y-%m-%dT%H:%M:%S%.fZ").to_string();
    if let Some(obj) = data.as_object_mut() {
        obj.insert("last_refresh".to_string(), Value::String(now));
    }

    write_auth_json(&current.auth_path, &data)
        .map_err(|error| AuthError::Internal(error.to_string()))?;
    load_token_data(current.auth_path.to_str())
        .map_err(|error| AuthError::Internal(error.to_string()))
}

fn apply_refresh_payload(data: &mut Value, payload: &Value) -> Result<(), AuthError> {
    let obj = data
        .as_object_mut()
        .ok_or_else(|| AuthError::Internal("auth file root must be an object".to_string()))?;
    let tokens = obj
        .get_mut("tokens")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| AuthError::Internal("auth file tokens must be an object".to_string()))?;

    for name in ["id_token", "access_token", "refresh_token"] {
        match payload.get(name) {
            None => {}
            Some(Value::String(value)) if !value.trim().is_empty() => {
                tokens.insert(name.to_string(), Value::String(value.to_string()));
            }
            Some(_) => {
                return Err(AuthError::RefreshProtocol(format!(
                    "ChatGPT OAuth token refresh response has invalid {name}"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    static AUTH_PATH_ENV_LOCK: Mutex<()> = Mutex::new(());

    fn restore_env(name: &str, value: Option<std::ffi::OsString>) {
        match value {
            Some(value) => std::env::set_var(name, value),
            None => std::env::remove_var(name),
        }
    }

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
    fn expiration_accepts_integral_json_numbers() {
        for payload in [r#"{"exp":1.0}"#, r#"{"exp":1e0}"#] {
            let jwt = format!(
                "header.{}.signature",
                URL_SAFE_NO_PAD.encode(payload.as_bytes())
            );
            assert_eq!(expiration(&jwt).unwrap().unwrap().timestamp(), 1);
        }
    }

    #[test]
    fn expiration_treats_missing_and_null_as_no_expiration() {
        for payload in [serde_json::json!({}), serde_json::json!({"exp": null})] {
            let jwt = format!(
                "header.{}.signature",
                URL_SAFE_NO_PAD.encode(serde_json::to_vec(&payload).unwrap())
            );
            assert_eq!(expiration(&jwt).unwrap(), None);
        }
    }

    #[test]
    fn expiration_rejects_nonintegral_unsafe_and_boolean_numbers() {
        for payload in [
            r#"{"exp":1.5}"#,
            r#"{"exp":true}"#,
            r#"{"exp":9007199254740992}"#,
        ] {
            let jwt = format!(
                "header.{}.signature",
                URL_SAFE_NO_PAD.encode(payload.as_bytes())
            );
            assert!(expiration(&jwt).is_err(), "{payload}");
        }
    }

    #[test]
    fn test_jwt_claims_too_few_parts() {
        assert!(jwt_claims("onlyonepart").is_err());
    }

    #[test]
    fn test_jwt_claims_empty_payload() {
        for token in [
            "header..signature",
            ".payload.sig",
            "header.payload.",
            "a.b.c.d",
        ] {
            let error = jwt_claims(token).unwrap_err();
            assert!(error.to_string().contains("JWT structure"));
        }
    }

    #[test]
    fn test_jwt_claims_invalid_base64_content() {
        let encoded = URL_SAFE_NO_PAD.encode(b"not json at all {{{");
        let jwt = format!("header.{}.signature", encoded);
        let result = jwt_claims(&jwt);
        assert!(result.is_err());
    }

    #[test]
    fn test_jwt_claims_rejects_noncanonical_base64url_payloads() {
        for payload in ["e30=", "e+0", "e/0", "a", "e31"] {
            let error = jwt_claims(&format!("header.{payload}.signature")).unwrap_err();
            assert!(error.to_string().contains("JWT payload"));
        }
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
    fn test_redact_text_handles_marker_secrets_and_replacement_boundaries() {
        assert_eq!(redact_text("*** token *", &["*"]), " token ");
        assert_eq!(redact_text("*** token ***", &["***"]), " token ");
        let boundary_safe = redact_text("ab", &["a*", "b"]);
        assert!(!boundary_safe.contains("a*"));
        assert!(!boundary_safe.contains('b'));
    }

    #[test]
    fn unsupported_auth_mode_diagnostic_does_not_reflect_value() {
        let secret = "access-token-sentinel";
        let path = std::env::temp_dir().join(format!(
            "codex-auth-mode-redaction-{}.json",
            uuid::Uuid::new_v4()
        ));
        fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({"auth_mode": secret})).unwrap(),
        )
        .unwrap();
        let error = load_token_data(Some(path.to_string_lossy().as_ref())).unwrap_err();
        assert!(!error.to_string().contains(secret));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn only_managed_official_auth_modes_are_accepted() {
        let dir = std::env::temp_dir().join(format!("codex-auth-modes-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("auth.json");

        for mode in [
            "Chatgpt",
            "ChatgptAuthTokens",
            "chatgpt_auth_tokens",
            "chatgptAuthTokens",
        ] {
            write_auth_json(&path, &serde_json::json!({"auth_mode": mode, "tokens": {}})).unwrap();
            let error = load_token_data(path.to_str()).unwrap_err();
            assert!(error.to_string().contains("auth_mode is unsupported"));
        }

        for mode in [Value::Null, Value::String("chatgpt".to_string())] {
            write_auth_json(&path, &serde_json::json!({"auth_mode": mode, "tokens": {}})).unwrap();
            let error = load_token_data(path.to_str()).unwrap_err();
            assert!(error.to_string().contains("access_token is missing"));
        }
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn external_auth_mode_is_rejected_before_managed_request_flow() {
        let dir =
            std::env::temp_dir().join(format!("codex-external-auth-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("auth.json");
        write_auth_json(
            &path,
            &serde_json::json!({
                "auth_mode": "chatgptAuthTokens",
                "tokens": {
                    "access_token": "header.e30.signature",
                    "refresh_token": "",
                    "id_token": "header.e30.signature"
                }
            }),
        )
        .unwrap();

        let error = token_for_request(path.to_str()).unwrap_err();
        assert!(error.to_string().contains("auth_mode is unsupported"));
        fs::remove_dir_all(dir).unwrap();
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
        assert!(fs::read_dir(&dir).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".tmp-")));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_write_auth_json_replaces_existing_destination() {
        let dir = std::env::temp_dir().join(format!(
            "codex_auth_replace_existing_{}",
            uuid::Uuid::new_v4()
        ));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        fs::write(&auth_path, b"stale contents").unwrap();

        let expected = serde_json::json!({"tokens": {"access_token": "replacement"}});
        write_auth_json(&auth_path, &expected).unwrap();

        let actual = strict_json::parse_slice(&fs::read(&auth_path).unwrap()).unwrap();
        assert_eq!(actual, expected);
        assert!(fs::read_dir(&dir).unwrap().all(|entry| !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".tmp-")));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_finish_auth_write_preserves_primary_and_cleanup_failures() {
        let result = finish_auth_write(
            Err(AuthError::OAuth("primary write failure".to_string())),
            Err(std::io::Error::other("cleanup failure")),
        );
        match result {
            Err(AuthError::WriteCleanup { primary, cleanup }) => {
                assert_eq!(primary.to_string(), "primary write failure");
                assert_eq!(cleanup.to_string(), "cleanup failure");
            }
            other => panic!("unexpected result: {other:?}"),
        }
    }

    #[test]
    fn test_load_token_data_unreadable_path_is_safe_missing_error() {
        let dir = std::env::temp_dir().join(format!("codex_unreadable_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();

        let result = load_token_data(dir.to_str());
        match result {
            Err(AuthError::Missing(message)) => {
                assert_eq!(message, "ChatGPT OAuth auth file is unavailable");
                assert!(!message.contains(&dir.display().to_string()));
            }
            other => panic!("unexpected result: {other:?}"),
        }
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

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn load_rejects_whitespace_only_refresh_and_id_tokens() {
        let dir =
            std::env::temp_dir().join(format!("codex-whitespace-auth-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");

        for name in ["refresh_token", "id_token"] {
            let mut tokens = serde_json::json!({
                "access_token": "header.e30.signature",
                "refresh_token": "refresh-token",
                "id_token": "header.e30.signature",
                "account_id": "account-id"
            });
            tokens[name] = Value::String(" \t ".to_string());
            fs::write(
                &auth_path,
                serde_json::to_vec(&serde_json::json!({"tokens": tokens})).unwrap(),
            )
            .unwrap();

            let error = load_token_data(auth_path.to_str()).unwrap_err();
            assert!(error.to_string().contains(&format!("{name} is missing")));
        }

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_load_token_data_rejects_root_fields() {
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
        });
        let auth_path = dir.join("auth.json");
        fs::write(&auth_path, serde_json::to_string(&auth_data).unwrap()).unwrap();

        let error = load_token_data(Some(auth_path.to_str().unwrap())).unwrap_err();

        assert!(error
            .to_string()
            .contains("file-backed ChatGPT OAuth tokens"));
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
    fn canonical_nested_tokens_reject_mixed_root_credentials() {
        let dir = std::env::temp_dir().join(format!("codex-mixed-auth-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");

        for name in [
            "access_token",
            "refresh_token",
            "id_token",
            "chatgptAuthTokens",
        ] {
            let mut auth = serde_json::json!({
                "tokens": {
                    "access_token": "header.e30.signature",
                    "refresh_token": "refresh-token",
                    "id_token": "header.e30.signature",
                    "account_id": "account-id"
                }
            });
            auth.as_object_mut().unwrap().insert(
                name.to_string(),
                Value::String("unsupported-root-value".to_string()),
            );
            fs::write(&auth_path, serde_json::to_vec(&auth).unwrap()).unwrap();

            let error = load_token_data(auth_path.to_str()).unwrap_err();
            assert!(error.to_string().contains("mixes canonical tokens"));
        }

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_root_auth_rejects_present_null_tokens_field() {
        let dir = std::env::temp_dir().join(format!("codex_test_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        fs::write(
            &auth_path,
            serde_json::to_vec(&serde_json::json!({
                "tokens": null,
                "access_token": "access-root",
                "refresh_token": "refresh-root",
                "id_token": "id-root",
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
    fn test_apply_refresh_payload_rejects_root_auth_layout() {
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

        let error = apply_refresh_payload(&mut data, &payload).unwrap_err();
        assert!(error.to_string().contains("tokens must be an object"));
    }

    #[test]
    fn refresh_payload_rejects_whitespace_only_optional_tokens_without_persisting_them() {
        for name in ["refresh_token", "id_token"] {
            let mut data = serde_json::json!({
                "tokens": {
                    "access_token": "old-access",
                    "refresh_token": "old-refresh",
                    "id_token": "old-id"
                }
            });
            let before = data.clone();
            let error =
                apply_refresh_payload(&mut data, &serde_json::json!({(name): " \t "})).unwrap_err();

            assert!(error.to_string().contains(&format!("invalid {name}")));
            assert_eq!(data, before);
        }
    }

    #[test]
    fn refresh_payload_preserves_nonblank_token_bytes() {
        let mut data = serde_json::json!({
            "tokens": {
                "access_token": "old-access",
                "refresh_token": "old-refresh",
                "id_token": "old-id"
            }
        });

        apply_refresh_payload(
            &mut data,
            &serde_json::json!({"refresh_token": " new-refresh "}),
        )
        .unwrap();

        assert_eq!(data["tokens"]["refresh_token"], " new-refresh ");
    }

    #[test]
    fn test_partial_refresh_rejects_root_auth_credentials() {
        let mut data = serde_json::json!({
            "access_token": "old-access",
            "refresh_token": "old-refresh",
            "id_token": "old-id",
        });
        let error = apply_refresh_payload(
            &mut data,
            &serde_json::json!({"access_token": "new-access"}),
        )
        .unwrap_err();
        assert!(error.to_string().contains("tokens must be an object"));
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
    fn test_load_rejects_non_boolean_fedramp_claims() {
        let dir = std::env::temp_dir().join(format!("codex_fedramp_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        let access = format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD
                .encode(serde_json::to_vec(&serde_json::json!({"exp": 9999999999i64})).unwrap())
        );
        let id = format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD.encode(
                serde_json::to_vec(&serde_json::json!({
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "acc",
                        "chatgpt_account_is_fedramp": "false"
                    }
                }))
                .unwrap()
            )
        );
        fs::write(
            &auth_path,
            serde_json::to_vec(&serde_json::json!({
                "tokens": {
                    "access_token": access,
                    "refresh_token": "refresh",
                    "id_token": id,
                    "account_id": "acc"
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let error = load_token_data(auth_path.to_str()).unwrap_err();
        assert_eq!(
            error.to_string(),
            "ChatGPT OAuth chatgpt_account_is_fedramp claim must be a boolean"
        );
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_load_rejects_conflicting_fedramp_claims() {
        let dir = std::env::temp_dir().join(format!("codex_fedramp_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        let make_token = |fedramp| {
            format!(
                "header.{}.signature",
                URL_SAFE_NO_PAD.encode(
                    serde_json::to_vec(&serde_json::json!({
                        "exp": 9999999999i64,
                        "https://api.openai.com/auth": {
                            "chatgpt_account_id": "acc",
                            "chatgpt_account_is_fedramp": fedramp
                        }
                    }))
                    .unwrap()
                )
            )
        };
        fs::write(
            &auth_path,
            serde_json::to_vec(&serde_json::json!({
                "tokens": {
                    "access_token": make_token(false),
                    "refresh_token": "refresh",
                    "id_token": make_token(true),
                    "account_id": "acc"
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let error = load_token_data(auth_path.to_str()).unwrap_err();
        assert!(error.to_string().contains("fedramp claims do not match"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_load_rejects_malformed_and_conflicting_plan_user_claims() {
        let dir = std::env::temp_dir().join(format!("codex_claims_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        let make_token = |claims: Value| {
            format!(
                "header.{}.signature",
                URL_SAFE_NO_PAD.encode(
                    serde_json::to_vec(&serde_json::json!({
                        "exp": 9999999999i64,
                        "https://api.openai.com/auth": claims,
                    }))
                    .unwrap()
                )
            )
        };
        let write_claims = |id_claims: Value, access_claims: Value| {
            fs::write(
                &auth_path,
                serde_json::to_vec(&serde_json::json!({
                    "tokens": {
                        "access_token": make_token(access_claims),
                        "refresh_token": "refresh",
                        "id_token": make_token(id_claims),
                        "account_id": "acc"
                    }
                }))
                .unwrap(),
            )
            .unwrap();
        };

        write_claims(
            serde_json::json!({"chatgpt_account_id": "acc", "chatgpt_plan_type": 42}),
            serde_json::json!({"chatgpt_account_id": "acc", "chatgpt_plan_type": "plus"}),
        );
        assert!(load_token_data(auth_path.to_str())
            .unwrap_err()
            .to_string()
            .contains("plan type claim must be a non-empty string"));

        write_claims(
            serde_json::json!({"chatgpt_account_id": "acc", "chatgpt_user_id": "user-a"}),
            serde_json::json!({"chatgpt_account_id": "acc", "chatgpt_user_id": "user-b"}),
        );
        assert!(load_token_data(auth_path.to_str())
            .unwrap_err()
            .to_string()
            .contains("user id claims do not match"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn test_load_rejects_whitespace_account_id() {
        let dir = std::env::temp_dir().join(format!("codex_account_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        let claims = URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&serde_json::json!({"exp": 9999999999i64})).unwrap());
        let token = format!("header.{claims}.signature");
        fs::write(
            &auth_path,
            serde_json::to_vec(&serde_json::json!({
                "tokens": {
                    "access_token": token,
                    "refresh_token": "refresh",
                    "id_token": token,
                    "account_id": "   "
                }
            }))
            .unwrap(),
        )
        .unwrap();

        let error = load_token_data(auth_path.to_str()).unwrap_err();
        assert!(error.to_string().contains("account id is invalid"));

        let mut document: Value = serde_json::from_slice(&fs::read(&auth_path).unwrap()).unwrap();
        document["tokens"]["account_id"] = serde_json::json!("bad account");
        fs::write(&auth_path, serde_json::to_vec(&document).unwrap()).unwrap();
        let error = load_token_data(auth_path.to_str()).unwrap_err().to_string();
        assert!(error.contains("account id is invalid"));
        assert!(!error.contains("bad account"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn header_bound_auth_values_require_visible_ascii_without_exposing_values() {
        let dir = std::env::temp_dir().join(format!("codex_header_auth_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let auth_path = dir.join("auth.json");
        fs::write(
            &auth_path,
            serde_json::to_vec(&serde_json::json!({
                "tokens": {
                    "access_token": "bad token",
                    "refresh_token": "refresh",
                    "id_token": "id",
                    "account_id": "acc"
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let error = load_token_data(auth_path.to_str()).unwrap_err().to_string();
        assert!(error.contains("access_token is invalid"));
        assert!(!error.contains("bad token"));

        let current = ChatGPTTokenData {
            auth_path: auth_path.clone(),
            access_token: "access".to_string(),
            refresh_token: "refresh".to_string(),
            id_token: "id".to_string(),
            account_id: "acc".to_string(),
            fedramp: false,
            access_expires_at: None,
        };
        let error = persist_refresh_payload(
            &current,
            &serde_json::json!({"access_token": "bad token"}),
            "bad token",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("invalid access_token"));
        assert!(!error.contains("bad token"));
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

        let replacement_id = format!(
            "header.{}.signature",
            URL_SAFE_NO_PAD.encode(
                serde_json::to_vec(&serde_json::json!({
                    "https://api.openai.com/auth": {"chatgpt_account_id": "acc-other"}
                }))
                .unwrap()
            )
        );
        let replacement = serde_json::to_vec(&serde_json::json!({
            "tokens": {
                "access_token": access("replacement", "acc-other"),
                "refresh_token": "refresh-other",
                "id_token": replacement_id
            }
        }))
        .unwrap();
        fs::write(&auth_path, &replacement).unwrap();
        let error = persist_refresh_payload(
            &reused,
            &serde_json::json!({"access_token": response_access}),
            &response_access,
        )
        .unwrap_err();
        assert!(error.to_string().contains("account changed"));
        assert_eq!(fs::read(&auth_path).unwrap(), replacement);
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
    fn test_refresh_endpoint_validation_is_strict_and_allows_path_and_port() {
        let endpoint = "http://127.0.0.1:18081/oauth/token".to_string();
        assert_eq!(validate_refresh_url(endpoint.clone()).unwrap(), endpoint);

        for unsafe_endpoint in [
            " ",
            " https://auth.openai.com/oauth/token",
            "ftp://auth.openai.com/oauth/token",
            "https://user:secret@auth.openai.com/oauth/token",
            "https://auth.openai.com/oauth/token?tenant=one",
            "https://auth.openai.com/oauth/token#fragment",
            "https:///oauth/token",
            "https://auth.openai.com:invalid/oauth/token",
            "http://auth.openai.com\n.evil/oauth/token",
            "https://auth.openai.com/a path",
            "https://auth.openai.com/%",
            "https://auth.openai.com/%zz",
            "https://auth.openai.com/%0G",
        ] {
            assert!(
                validate_refresh_url(unsafe_endpoint.to_string()).is_err(),
                "{unsafe_endpoint}"
            );
        }
        let encoded = "https://auth.openai.com/oauth%20token".to_string();
        assert_eq!(validate_refresh_url(encoded.clone()).unwrap(), encoded);
    }

    #[test]
    fn test_refresh_window_and_matching_account_change_detection() {
        let mut observed = ChatGPTTokenData {
            auth_path: PathBuf::from("auth.json"),
            access_token: "old-access".to_string(),
            refresh_token: "old-refresh".to_string(),
            id_token: "old-id".to_string(),
            account_id: "acc-shared".to_string(),
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
        let lock = refresh_lock(&auth_path).unwrap();
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
        let _guard = AUTH_PATH_ENV_LOCK.lock().unwrap();
        let previous = std::env::var_os("CODEX_HOME");
        std::env::remove_var("CODEX_HOME");
        let path = resolve_auth_path(None).unwrap();
        assert!(path.to_str().unwrap().ends_with(".codex/auth.json"));
        restore_env("CODEX_HOME", previous);
    }

    #[test]
    fn test_resolve_auth_path_explicit() {
        let path = resolve_auth_path(Some("/tmp/custom/auth.json")).unwrap();
        assert_eq!(path, PathBuf::from("/tmp/custom/auth.json"));
    }

    #[test]
    fn test_resolve_auth_path_expands_exact_tilde_for_explicit_and_codex_home() {
        let _guard = AUTH_PATH_ENV_LOCK.lock().unwrap();
        let home = PathBuf::from(std::env::var_os("HOME").expect("test requires HOME"));
        assert_eq!(resolve_auth_path(Some("~")).unwrap(), home);

        let previous = std::env::var_os("CODEX_HOME");
        std::env::set_var("CODEX_HOME", "~");
        assert_eq!(resolve_auth_path(None).unwrap(), home.join("auth.json"));
        restore_env("CODEX_HOME", previous);
    }

    #[cfg(windows)]
    #[test]
    fn windows_default_auth_path_uses_userprofile_without_home() {
        let _guard = AUTH_PATH_ENV_LOCK.lock().unwrap();
        let previous_codex_home = std::env::var_os("CODEX_HOME");
        let previous_home = std::env::var_os("HOME");
        let previous_profile = std::env::var_os("USERPROFILE");
        let profile =
            std::env::temp_dir().join(format!("codex_windows_profile_{}", uuid::Uuid::new_v4()));
        std::env::remove_var("CODEX_HOME");
        std::env::remove_var("HOME");
        std::env::set_var("USERPROFILE", &profile);

        assert_eq!(
            resolve_auth_path(None).unwrap(),
            profile.join(".codex").join("auth.json")
        );

        restore_env("CODEX_HOME", previous_codex_home);
        restore_env("HOME", previous_home);
        restore_env("USERPROFILE", previous_profile);
    }

    #[test]
    fn test_resolve_auth_path_rejects_empty_explicit_path() {
        assert!(resolve_auth_path(Some("")).is_err());
        assert!(resolve_auth_path(Some("  ")).is_err());
    }

    #[test]
    fn test_resolve_auth_path_preserves_nonempty_whitespace() {
        assert_eq!(
            resolve_auth_path(Some(" auth directory/auth.json ")).unwrap(),
            PathBuf::from(" auth directory/auth.json ")
        );
    }
}
