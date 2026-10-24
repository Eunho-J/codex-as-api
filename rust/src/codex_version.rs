use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Number, Value};
#[cfg(test)]
use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::watch;

pub const CODEX_NPM_LATEST_URL: &str = "https://registry.npmjs.org/@openai%2fcodex/latest";
pub const CODEX_NPM_PACKAGE_NAME: &str = "@openai/codex";
pub const CODEX_CLI_ORIGINATOR: &str = "codex_cli_rs";
pub const CODEX_VERSION_FETCH_TIMEOUT: Duration = Duration::from_secs(5);
pub const CODEX_VERSION_CACHE_TTL: Duration = Duration::from_secs(60 * 60);
pub const CODEX_VERSION_RETRY_BACKOFF: Duration = Duration::from_secs(60);
pub const MAX_CODEX_VERSION_LENGTH: usize = 64;
pub const JS_SAFE_INTEGER: u64 = 9_007_199_254_740_991;
const MAX_METADATA_RESPONSE_BYTES: usize = 1_048_576;

pub type CodexSemverTuple = (u64, u64, u64);
pub type CodexVersionFetchFuture =
    Pin<Box<dyn Future<Output = Result<Vec<u8>, CodexVersionError>> + Send + 'static>>;
pub type CodexVersionAsyncFetch =
    Arc<dyn Fn(String, Duration) -> CodexVersionFetchFuture + Send + Sync + 'static>;
pub type CodexVersionClock = Arc<dyn Fn() -> Instant + Send + Sync + 'static>;

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CodexVersionError {
    #[error("{0}")]
    Unavailable(String),
    #[error("{0}")]
    InvalidVersion(String),
}

#[cfg(test)]
impl CodexVersionError {
    pub const fn status_code(&self) -> u16 {
        match self {
            Self::Unavailable(_) => 503,
            Self::InvalidVersion(_) => 400,
        }
    }

    pub const fn is_unavailable(&self) -> bool {
        matches!(self, Self::Unavailable(_))
    }

    pub fn message(&self) -> &str {
        match self {
            Self::Unavailable(message) | Self::InvalidVersion(message) => message,
        }
    }
}

fn parse_nonnegative_semver_component(part: &str) -> Option<u64> {
    if part.is_empty() {
        return None;
    }
    if part.len() > 1 && part.starts_with('0') {
        return None;
    }
    if !part.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let parsed = part.parse::<u64>().ok()?;
    (parsed <= JS_SAFE_INTEGER).then_some(parsed)
}

pub fn parse_codex_semver(value: &str) -> Option<CodexSemverTuple> {
    if value.is_empty() || value.len() > MAX_CODEX_VERSION_LENGTH || value != value.trim() {
        return None;
    }
    if value
        .chars()
        .any(|ch| ch.is_whitespace() || ch.is_control())
    {
        return None;
    }
    let mut parts = value.split('.');
    let major = parse_nonnegative_semver_component(parts.next()?)?;
    let minor = parse_nonnegative_semver_component(parts.next()?)?;
    let patch = parse_nonnegative_semver_component(parts.next()?)?;
    if parts.next().is_some() {
        return None;
    }
    Some((major, minor, patch))
}

#[cfg(test)]
pub fn is_valid_codex_version(value: &str) -> bool {
    parse_codex_semver(value).is_some()
}

pub fn validate_codex_version(value: &str) -> Result<String, CodexVersionError> {
    if parse_codex_semver(value).is_none() {
        return Err(CodexVersionError::Unavailable(
            "npm @openai/codex version must be a strict stable three-component semantic version"
                .to_string(),
        ));
    }
    Ok(value.to_string())
}

pub fn require_explicit_codex_version(value: &str) -> Result<String, CodexVersionError> {
    if parse_codex_semver(value).is_none() {
        return Err(CodexVersionError::InvalidVersion(
            "Codex compatibility version must be a strict stable three-component semantic version"
                .to_string(),
        ));
    }
    Ok(value.to_string())
}

#[cfg(test)]
pub fn compare_codex_versions(left: &str, right: &str) -> Result<Ordering, CodexVersionError> {
    let left_semver = parse_codex_semver(left).ok_or_else(|| {
        CodexVersionError::InvalidVersion(
            "compared Codex versions must be strict stable three-component semantic versions"
                .to_string(),
        )
    })?;
    let right_semver = parse_codex_semver(right).ok_or_else(|| {
        CodexVersionError::InvalidVersion(
            "compared Codex versions must be strict stable three-component semantic versions"
                .to_string(),
        )
    })?;
    Ok(left_semver.cmp(&right_semver))
}

struct StrictJsonValue(Value);

impl StrictJsonValue {
    fn into_inner(self) -> Value {
        self.0
    }
}

impl<'de> Deserialize<'de> for StrictJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StrictValueVisitor;

        impl<'de> Visitor<'de> for StrictValueVisitor {
            type Value = StrictJsonValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("any valid strict JSON value without duplicate object keys")
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Bool(value)))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value.unsigned_abs() > JS_SAFE_INTEGER {
                    return Err(E::custom("JSON integers must be JavaScript-safe"));
                }
                Ok(StrictJsonValue(Value::Number(Number::from(value))))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if value > JS_SAFE_INTEGER {
                    return Err(E::custom("JSON integers must be JavaScript-safe"));
                }
                Ok(StrictJsonValue(Value::Number(Number::from(value))))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                if !value.is_finite()
                    || (value.fract() == 0.0 && value.abs() > JS_SAFE_INTEGER as f64)
                {
                    return Err(E::custom("JSON numbers must be finite JavaScript numbers"));
                }
                let number = Number::from_f64(value)
                    .ok_or_else(|| E::custom("JSON numbers must be finite"))?;
                Ok(StrictJsonValue(Value::Number(number)))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                Ok(StrictJsonValue(Value::String(value.to_owned())))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::String(value)))
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Null))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(StrictJsonValue(Value::Null))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut elements = Vec::new();
                while let Some(item) = seq.next_element::<StrictJsonValue>()? {
                    elements.push(item.into_inner());
                }
                Ok(StrictJsonValue(Value::Array(elements)))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut object = Map::new();
                while let Some((key, value)) = map.next_entry::<String, StrictJsonValue>()? {
                    if object.contains_key(&key) {
                        return Err(de::Error::custom(format!(
                            "JSON object contains duplicate key: {key:?}"
                        )));
                    }
                    object.insert(key, value.into_inner());
                }
                Ok(StrictJsonValue(Value::Object(object)))
            }
        }

        deserializer.deserialize_any(StrictValueVisitor)
    }
}

pub fn parse_json_strict_no_duplicate_keys(input: &[u8]) -> Result<Value, CodexVersionError> {
    let text = std::str::from_utf8(input).map_err(|err| {
        CodexVersionError::Unavailable(format!(
            "npm @openai/codex metadata response is not valid UTF-8: {err}"
        ))
    })?;
    let strict_value: StrictJsonValue = serde_json::from_str(text).map_err(|err| {
        CodexVersionError::Unavailable(format!(
            "npm @openai/codex metadata response is not valid strict JSON: {err}"
        ))
    })?;
    Ok(strict_value.into_inner())
}

pub fn parse_codex_npm_latest_payload(payload: &[u8]) -> Result<String, CodexVersionError> {
    let parsed = parse_json_strict_no_duplicate_keys(payload)?;
    let Some(object) = parsed.as_object() else {
        return Err(CodexVersionError::Unavailable(
            "npm @openai/codex metadata response must be a JSON object".to_string(),
        ));
    };
    let Some(name) = object.get("name").and_then(Value::as_str) else {
        return Err(CodexVersionError::Unavailable(
            "npm @openai/codex metadata response is missing package name".to_string(),
        ));
    };
    if name != CODEX_NPM_PACKAGE_NAME {
        return Err(CodexVersionError::Unavailable(
            "npm @openai/codex metadata response has unexpected package name".to_string(),
        ));
    }
    let Some(version) = object.get("version").and_then(Value::as_str) else {
        return Err(CodexVersionError::Unavailable(
            "npm @openai/codex metadata response is missing string version".to_string(),
        ));
    };
    validate_codex_version(version)
}

fn codex_os_name() -> &'static str {
    match std::env::consts::OS {
        "macos" => "Mac OS",
        "windows" => "Windows",
        "linux" => "Linux",
        other => other,
    }
}

fn codex_os_info() -> String {
    format!("{} unknown; {}", codex_os_name(), std::env::consts::ARCH)
}

fn sanitize_header_value(value: &str) -> String {
    value
        .chars()
        .map(|ch| if matches!(ch, ' '..='~') { ch } else { '_' })
        .collect()
}

pub fn codex_user_agent_for_versions(
    version: &str,
    codex_as_api_version: &str,
) -> Result<String, CodexVersionError> {
    let normalized_version = require_explicit_codex_version(version)?;
    let normalized_pkg_version = require_explicit_codex_version(codex_as_api_version)?;
    Ok(sanitize_header_value(&format!(
        "{CODEX_CLI_ORIGINATOR}/{normalized_version} ({}) codex-as-api/{normalized_pkg_version}",
        codex_os_info()
    )))
}

pub fn codex_cli_headers_for_version(
    version: &str,
) -> Result<HashMap<String, String>, CodexVersionError> {
    codex_cli_headers_for_versions(version, env!("CARGO_PKG_VERSION"))
}

pub fn codex_cli_headers_for_versions(
    version: &str,
    codex_as_api_version: &str,
) -> Result<HashMap<String, String>, CodexVersionError> {
    let user_agent = codex_user_agent_for_versions(version, codex_as_api_version)?;
    let mut headers = HashMap::with_capacity(2);
    headers.insert("originator".to_string(), CODEX_CLI_ORIGINATOR.to_string());
    headers.insert("User-Agent".to_string(), user_agent);
    Ok(headers)
}

fn validate_registry_url(url: &str) -> Result<String, CodexVersionError> {
    if url.is_empty() || url != url.trim() {
        return Err(CodexVersionError::InvalidVersion(
            "registry_url must be a non-empty HTTP(S) URL without surrounding whitespace"
                .to_string(),
        ));
    }
    if url.chars().any(|ch| ch.is_whitespace() || ch.is_control()) {
        return Err(CodexVersionError::InvalidVersion(
            "registry_url must not contain whitespace or control characters".to_string(),
        ));
    }
    let parsed = reqwest::Url::parse(url).map_err(|_| {
        CodexVersionError::InvalidVersion("registry_url must be a valid HTTP(S) URL".to_string())
    })?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
    {
        return Err(CodexVersionError::InvalidVersion(
            "registry_url must be an HTTP(S) URL without credentials".to_string(),
        ));
    }
    Ok(url.to_string())
}

fn validate_positive_duration(
    duration: Duration,
    field: &str,
) -> Result<Duration, CodexVersionError> {
    if duration.is_zero() {
        return Err(CodexVersionError::InvalidVersion(format!(
            "{field} must be greater than zero"
        )));
    }
    Ok(duration)
}

pub async fn default_fetch_codex_npm_latest(
    url: String,
    timeout: Duration,
) -> Result<Vec<u8>, CodexVersionError> {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .timeout(timeout)
        .build()
        .map_err(|err| {
            CodexVersionError::Unavailable(format!(
                "failed to build HTTP client for npm @openai/codex version lookup: {err}"
            ))
        })?;

    let send_future = client
        .get(&url)
        .header(reqwest::header::ACCEPT, "application/json")
        .timeout(timeout)
        .send();

    let response = match tokio::time::timeout(timeout, send_future).await {
        Ok(Ok(response)) => response,
        Ok(Err(err)) => {
            if err.is_timeout() {
                return Err(CodexVersionError::Unavailable(format!(
                    "npm @openai/codex version lookup timed out after {}ms",
                    timeout.as_millis()
                )));
            }
            return Err(CodexVersionError::Unavailable(format!(
                "npm @openai/codex version lookup request failed: {err}"
            )));
        }
        Err(_) => {
            return Err(CodexVersionError::Unavailable(format!(
                "npm @openai/codex version lookup timed out after {}ms",
                timeout.as_millis()
            )));
        }
    };

    let status = response.status();
    if status != reqwest::StatusCode::OK {
        return Err(CodexVersionError::Unavailable(format!(
            "npm @openai/codex version lookup failed with HTTP {}",
            status.as_u16()
        )));
    }

    let bytes = match tokio::time::timeout(timeout, response.bytes()).await {
        Ok(Ok(bytes)) => bytes,
        Ok(Err(err)) => {
            if err.is_timeout() {
                return Err(CodexVersionError::Unavailable(format!(
                    "npm @openai/codex version lookup timed out after {}ms",
                    timeout.as_millis()
                )));
            }
            return Err(CodexVersionError::Unavailable(format!(
                "failed to read npm @openai/codex version response body: {err}"
            )));
        }
        Err(_) => {
            return Err(CodexVersionError::Unavailable(format!(
                "npm @openai/codex version lookup timed out after {}ms",
                timeout.as_millis()
            )));
        }
    };

    if bytes.len() > MAX_METADATA_RESPONSE_BYTES {
        return Err(CodexVersionError::Unavailable(
            "npm @openai/codex version response body exceeds maximum allowed size".to_string(),
        ));
    }

    Ok(bytes.to_vec())
}

fn default_async_fetch() -> CodexVersionAsyncFetch {
    Arc::new(|url, timeout| Box::pin(default_fetch_codex_npm_latest(url, timeout)))
}

fn default_clock() -> CodexVersionClock {
    Arc::new(Instant::now)
}

#[derive(Clone)]
pub struct CodexVersionResolverOptions {
    pub fetch: Option<CodexVersionAsyncFetch>,
    pub clock: Option<CodexVersionClock>,
    pub ttl: Duration,
    pub retry_backoff: Duration,
    pub timeout: Duration,
    pub registry_url: String,
}

impl Default for CodexVersionResolverOptions {
    fn default() -> Self {
        Self {
            fetch: None,
            clock: None,
            ttl: CODEX_VERSION_CACHE_TTL,
            retry_backoff: CODEX_VERSION_RETRY_BACKOFF,
            timeout: CODEX_VERSION_FETCH_TIMEOUT,
            registry_url: CODEX_NPM_LATEST_URL.to_string(),
        }
    }
}

type InFlightWatchReceiver = watch::Receiver<Option<Result<String, CodexVersionError>>>;

struct ResolverState {
    fetch: CodexVersionAsyncFetch,
    clock: CodexVersionClock,
    ttl: Duration,
    retry_backoff: Duration,
    timeout: Duration,
    registry_url: String,
    last_good_version: Option<String>,
    last_good_semver: Option<CodexSemverTuple>,
    expires_at: Option<Instant>,
    retry_not_before: Option<Instant>,
    in_flight: Option<(u64, InFlightWatchReceiver)>,
    revision: u64,
}

impl ResolverState {
    fn from_options(options: CodexVersionResolverOptions) -> Result<Self, CodexVersionError> {
        let ttl = validate_positive_duration(options.ttl, "ttl")?;
        let retry_backoff = validate_positive_duration(options.retry_backoff, "retry_backoff")?;
        let timeout = validate_positive_duration(options.timeout, "timeout")?;
        let registry_url = validate_registry_url(&options.registry_url)?;
        Ok(Self {
            fetch: options.fetch.unwrap_or_else(default_async_fetch),
            clock: options.clock.unwrap_or_else(default_clock),
            ttl,
            retry_backoff,
            timeout,
            registry_url,
            last_good_version: None,
            last_good_semver: None,
            expires_at: None,
            retry_not_before: None,
            in_flight: None,
            revision: 0,
        })
    }
}

#[derive(Clone)]
pub struct CodexVersionResolver {
    state: Arc<Mutex<ResolverState>>,
}

impl Default for CodexVersionResolver {
    fn default() -> Self {
        Self::new()
    }
}

impl CodexVersionResolver {
    pub fn new() -> Self {
        Self::with_options(CodexVersionResolverOptions::default())
            .expect("default CodexVersionResolverOptions must be valid")
    }

    pub fn with_options(options: CodexVersionResolverOptions) -> Result<Self, CodexVersionError> {
        let state = ResolverState::from_options(options)?;
        Ok(Self {
            state: Arc::new(Mutex::new(state)),
        })
    }

    #[cfg(test)]
    pub fn new_for_test<F>(
        registry_url: String,
        ttl: Duration,
        retry_backoff: Duration,
        fetch: F,
    ) -> Self
    where
        F: Fn() -> Result<Vec<u8>, CodexVersionError> + Send + Sync + 'static,
    {
        let fetch = Arc::new(fetch);
        Self::with_options(CodexVersionResolverOptions {
            fetch: Some(Arc::new(move |_url, _timeout| {
                let result = fetch();
                Box::pin(async move { result })
            })),
            clock: None,
            ttl,
            retry_backoff,
            timeout: CODEX_VERSION_FETCH_TIMEOUT,
            registry_url,
        })
        .expect("test CodexVersionResolverOptions must be valid")
    }

    #[cfg(test)]
    pub fn configure(&self, options: CodexVersionResolverOptions) -> Result<(), CodexVersionError> {
        let ttl = validate_positive_duration(options.ttl, "ttl")?;
        let retry_backoff = validate_positive_duration(options.retry_backoff, "retry_backoff")?;
        let timeout = validate_positive_duration(options.timeout, "timeout")?;
        let registry_url = validate_registry_url(&options.registry_url)?;

        let mut state = self.state.lock().map_err(|_| {
            CodexVersionError::Unavailable("Codex version resolver lock is poisoned".to_string())
        })?;
        state.fetch = options.fetch.unwrap_or_else(default_async_fetch);
        state.clock = options.clock.unwrap_or_else(default_clock);
        state.ttl = ttl;
        state.retry_backoff = retry_backoff;
        state.timeout = timeout;
        state.registry_url = registry_url;
        state.revision = state.revision.wrapping_add(1);
        state.last_good_version = None;
        state.last_good_semver = None;
        state.expires_at = None;
        state.retry_not_before = None;
        state.in_flight = None;
        Ok(())
    }

    #[cfg(test)]
    pub fn cached_version(&self) -> Option<String> {
        self.state
            .lock()
            .ok()
            .and_then(|state| state.last_good_version.clone())
    }

    pub async fn resolve(&self) -> Result<String, CodexVersionError> {
        enum Action {
            ReturnCached(String),
            WaitInFlight(InFlightWatchReceiver),
            StartLookup {
                tx: watch::Sender<Option<Result<String, CodexVersionError>>>,
                observed_revision: u64,
                fetch_fn: CodexVersionAsyncFetch,
                registry_url: String,
                timeout: Duration,
            },
        }

        let (tx, observed_revision, fetch_fn, registry_url, timeout) = loop {
            let action = {
                let mut state = self.state.lock().map_err(|_| {
                    CodexVersionError::Unavailable(
                        "Codex version resolver lock is poisoned".to_string(),
                    )
                })?;
                let now = (state.clock)();
                let cached = match (&state.last_good_version, state.expires_at) {
                    (Some(version), Some(expires_at)) if now < expires_at => Some(version.clone()),
                    (Some(version), _)
                        if state
                            .retry_not_before
                            .is_some_and(|retry_not_before| now < retry_not_before) =>
                    {
                        Some(version.clone())
                    }
                    _ => None,
                };

                if let Some(version) = cached {
                    Action::ReturnCached(version)
                } else {
                    if let Some((_, rx)) = &state.in_flight {
                        if rx.borrow().is_none() && rx.has_changed().is_err() {
                            state.in_flight = None;
                        }
                    }
                    if let Some((_, rx)) = &state.in_flight {
                        Action::WaitInFlight(rx.clone())
                    } else {
                        let observed_revision = state.revision;
                        let (tx, rx) = watch::channel(None);
                        state.in_flight = Some((observed_revision, rx));
                        Action::StartLookup {
                            tx,
                            observed_revision,
                            fetch_fn: state.fetch.clone(),
                            registry_url: state.registry_url.clone(),
                            timeout: state.timeout,
                        }
                    }
                }
            };

            match action {
                Action::ReturnCached(version) => return Ok(version),
                Action::WaitInFlight(mut rx) => loop {
                    if let Some(result) = rx.borrow().clone() {
                        return result;
                    }
                    if rx.changed().await.is_err() {
                        if let Some(result) = rx.borrow().clone() {
                            return result;
                        }
                        break;
                    }
                },
                Action::StartLookup {
                    tx,
                    observed_revision,
                    fetch_fn,
                    registry_url,
                    timeout,
                } => break (tx, observed_revision, fetch_fn, registry_url, timeout),
            }
        };

        let fetched = Self::fetch_and_validate(fetch_fn, registry_url, timeout).await;

        let outcome = {
            let mut state = self.state.lock().map_err(|_| {
                CodexVersionError::Unavailable(
                    "Codex version resolver lock is poisoned".to_string(),
                )
            })?;

            let result = match fetched {
                Ok((fetched_version, fetched_semver)) => {
                    if state.revision != observed_revision {
                        Ok(fetched_version)
                    } else {
                        let completed_at = (state.clock)();
                        let should_update = match state.last_good_semver {
                            Some(current_semver) => fetched_semver >= current_semver,
                            None => true,
                        };
                        if should_update || state.last_good_version.is_none() {
                            state.last_good_version = Some(fetched_version.clone());
                            state.last_good_semver = Some(fetched_semver);
                        }
                        state.expires_at = Some(completed_at + state.ttl);
                        state.retry_not_before = None;
                        Ok(state.last_good_version.clone().unwrap_or(fetched_version))
                    }
                }
                Err(unavailable_err) => {
                    if state.revision == observed_revision {
                        if let Some(last_good) = state.last_good_version.clone() {
                            let failed_at = (state.clock)();
                            state.retry_not_before = Some(failed_at + state.retry_backoff);
                            Ok(last_good)
                        } else {
                            Err(unavailable_err)
                        }
                    } else {
                        Err(unavailable_err)
                    }
                }
            };

            if state
                .in_flight
                .as_ref()
                .is_some_and(|(rev, _)| *rev == observed_revision)
            {
                state.in_flight = None;
            }
            result
        };

        let _ = tx.send(Some(outcome.clone()));
        outcome
    }

    #[cfg(test)]
    pub async fn headers(&self) -> Result<HashMap<String, String>, CodexVersionError> {
        let version = self.resolve().await?;
        codex_cli_headers_for_version(&version)
    }

    pub fn resolve_blocking(&self) -> Result<String, CodexVersionError> {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            match handle.runtime_flavor() {
                tokio::runtime::RuntimeFlavor::MultiThread => {
                    tokio::task::block_in_place(|| handle.block_on(self.resolve()))
                }
                _ => {
                    let resolver = self.clone();
                    std::thread::scope(|scope| {
                        scope
                            .spawn(move || {
                                let runtime = tokio::runtime::Builder::new_current_thread()
                                    .enable_all()
                                    .build()
                                    .map_err(|err| {
                                        CodexVersionError::Unavailable(format!(
                                            "failed to initialize Tokio runtime for Codex version lookup: {err}"
                                        ))
                                    })?;
                                runtime.block_on(resolver.resolve())
                            })
                            .join()
                            .map_err(|_| {
                                CodexVersionError::Unavailable(
                                    "Codex version lookup worker thread panicked".to_string(),
                                )
                            })?
                    })
                }
            }
        } else {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .map_err(|err| {
                    CodexVersionError::Unavailable(format!(
                        "failed to initialize Tokio runtime for Codex version lookup: {err}"
                    ))
                })?;
            runtime.block_on(self.resolve())
        }
    }

    #[cfg(test)]
    pub fn headers_blocking(&self) -> Result<HashMap<String, String>, CodexVersionError> {
        let version = self.resolve_blocking()?;
        codex_cli_headers_for_version(&version)
    }

    async fn fetch_and_validate(
        fetch_fn: CodexVersionAsyncFetch,
        registry_url: String,
        timeout: Duration,
    ) -> Result<(String, CodexSemverTuple), CodexVersionError> {
        let bytes = match tokio::time::timeout(timeout, fetch_fn(registry_url, timeout)).await {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(err)) => return Err(err),
            Err(_) => {
                return Err(CodexVersionError::Unavailable(format!(
                    "npm @openai/codex version lookup timed out after {}ms",
                    timeout.as_millis()
                )))
            }
        };
        let version = parse_codex_npm_latest_payload(&bytes)?;
        let semver = parse_codex_semver(&version).ok_or_else(|| {
            CodexVersionError::Unavailable("npm @openai/codex metadata version is invalid".into())
        })?;
        Ok((version, semver))
    }
}

static DEFAULT_CODEX_VERSION_RESOLVER: LazyLock<CodexVersionResolver> =
    LazyLock::new(CodexVersionResolver::new);

pub fn default_codex_version_resolver() -> &'static CodexVersionResolver {
    &DEFAULT_CODEX_VERSION_RESOLVER
}

#[cfg(test)]
pub async fn resolve_codex_cli_version() -> Result<String, CodexVersionError> {
    DEFAULT_CODEX_VERSION_RESOLVER.resolve().await
}

#[cfg(test)]
pub async fn resolve_latest_codex_version() -> Result<String, CodexVersionError> {
    DEFAULT_CODEX_VERSION_RESOLVER.resolve().await
}

#[cfg(test)]
pub async fn resolve_codex_cli_headers() -> Result<HashMap<String, String>, CodexVersionError> {
    DEFAULT_CODEX_VERSION_RESOLVER.headers().await
}

#[cfg(test)]
pub fn resolve_codex_cli_version_blocking() -> Result<String, CodexVersionError> {
    DEFAULT_CODEX_VERSION_RESOLVER.resolve_blocking()
}

#[cfg(test)]
pub fn resolve_codex_cli_headers_blocking() -> Result<HashMap<String, String>, CodexVersionError> {
    DEFAULT_CODEX_VERSION_RESOLVER.headers_blocking()
}

#[cfg(test)]
pub fn get_cached_codex_cli_version() -> Option<String> {
    DEFAULT_CODEX_VERSION_RESOLVER.cached_version()
}

#[cfg(test)]
pub fn reset_codex_version_cache() {
    let _ = DEFAULT_CODEX_VERSION_RESOLVER.configure(CodexVersionResolverOptions::default());
}

#[cfg(test)]
pub fn configure_codex_version_cache(
    options: CodexVersionResolverOptions,
) -> Result<(), CodexVersionError> {
    DEFAULT_CODEX_VERSION_RESOLVER.configure(options)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::Notify;

    fn npm_payload(version: &str, name: &str) -> Vec<u8> {
        format!(r#"{{"name":"{name}","version":"{version}"}}"#).into_bytes()
    }

    #[test]
    fn parses_and_validates_strict_three_component_stable_semver() {
        assert_eq!(parse_codex_semver("0.0.0"), Some((0, 0, 0)));
        assert_eq!(parse_codex_semver("0.118.0"), Some((0, 118, 0)));
        assert_eq!(parse_codex_semver("1.23.456"), Some((1, 23, 456)));
        assert!(is_valid_codex_version("0.118.0"));
        assert_eq!(validate_codex_version("0.118.0").unwrap(), "0.118.0");
        assert_eq!(
            require_explicit_codex_version("0.118.0").unwrap(),
            "0.118.0"
        );

        let long_version = format!("1.2.{}", "9".repeat(62));
        let invalid_versions = [
            "",
            "0.118",
            "0.118.0.1",
            "v0.118.0",
            "0.118.0-alpha.1",
            "0.118.0-0",
            "0.118.0+build.1",
            "0.118.0-rc.1+build.1",
            "00.118.0",
            "0.0118.0",
            "0.118.00",
            "-1.0.0",
            "0.-1.0",
            "0.1.-1",
            " 0.118.0",
            "0.118.0 ",
            "0. 118.0",
            "0.118.0\n",
            "0.118.0\r\nX-Injected: 1",
            "0.118.0\0",
            long_version.as_str(),
            "9007199254740992.0.0",
        ];

        for candidate in invalid_versions {
            assert_eq!(
                parse_codex_semver(candidate),
                None,
                "expected {candidate:?} to be rejected"
            );
            assert!(!is_valid_codex_version(candidate));
            let err = validate_codex_version(candidate).unwrap_err();
            assert_eq!(err.status_code(), 503);
            assert!(err.is_unavailable());
            let explicit_err = require_explicit_codex_version(candidate).unwrap_err();
            assert_eq!(explicit_err.status_code(), 400);
        }
    }

    #[test]
    fn compares_codex_versions_numerically() {
        assert_eq!(
            compare_codex_versions("0.99.9", "0.100.0").unwrap(),
            Ordering::Less
        );
        assert_eq!(
            compare_codex_versions("0.100.0", "0.99.9").unwrap(),
            Ordering::Greater
        );
        assert_eq!(
            compare_codex_versions("0.118.0", "0.118.0").unwrap(),
            Ordering::Equal
        );
        assert_eq!(
            compare_codex_versions("1.0.0", "0.999.999").unwrap(),
            Ordering::Greater
        );
        assert!(compare_codex_versions("0.118.0-alpha.1", "0.118.0").is_err());
    }

    #[test]
    fn rejects_duplicate_json_keys_and_malformed_npm_payloads() {
        for duplicate_doc in [
            br#"{"name":"@openai/codex","version":"0.118.0","version":"0.119.0"}"#.as_slice(),
            br#"{"name":"@openai/codex","ver\u0073ion":"0.118.0","version":"0.119.0"}"#.as_slice(),
            br#"{"name":"@openai/codex","version":"0.118.0","dist":{"shasum":"a","shasum":"b"}}"#
                .as_slice(),
        ] {
            let err = parse_json_strict_no_duplicate_keys(duplicate_doc).unwrap_err();
            assert!(err.message().contains("duplicate key"), "{}", err.message());
            assert_eq!(err.status_code(), 503);
        }

        assert_eq!(
            parse_codex_npm_latest_payload(
                br#"{"name":"@openai/codex","version":"0.120.2","dist":{"tarball":"https://example.com/t.tgz"}}"#
            )
            .unwrap(),
            "0.120.2"
        );

        let invalid_payloads: &[&[u8]] = &[
            &[0xff, 0xfe, 0xfd],
            b"{malformed",
            b"null",
            b"[]",
            br#""0.118.0""#,
            br#"{"name":"codex","version":"0.118.0"}"#,
            br#"{"name":"@openai/other","version":"0.118.0"}"#,
            br#"{"version":"0.118.0"}"#,
            br#"{"name":"@openai/codex"}"#,
            br#"{"name":"@openai/codex","version":"0.118.0-beta.1"}"#,
            br#"{"name":"@openai/codex","version":"0.118.0","version":"0.119.0"}"#,
        ];

        for payload in invalid_payloads {
            let err = parse_codex_npm_latest_payload(payload).unwrap_err();
            assert_eq!(err.status_code(), 503);
        }
    }

    #[test]
    fn constructs_user_agent_and_cli_headers_for_explicit_version() {
        let ua = codex_user_agent_for_versions("0.118.0", "0.8.3").unwrap();
        assert!(ua.starts_with("codex_cli_rs/0.118.0 ("));
        assert!(ua.ends_with(") codex-as-api/0.8.3"));
        let headers = codex_cli_headers_for_versions("0.118.0", "0.8.3").unwrap();
        assert_eq!(
            headers.get("originator").map(String::as_str),
            Some(CODEX_CLI_ORIGINATOR)
        );
        assert_eq!(headers.get("User-Agent"), Some(&ua));
    }

    #[tokio::test]
    async fn first_lookup_failure_returns_503_unavailable_without_fallback_or_backoff() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let base_instant = Instant::now();
        let offset_ms = Arc::new(AtomicUsize::new(0));

        let attempts_clone = attempts.clone();
        let fetch: CodexVersionAsyncFetch = Arc::new(move |url, timeout| {
            let call = attempts_clone.fetch_add(1, AtomicOrdering::SeqCst) + 1;
            assert_eq!(url, CODEX_NPM_LATEST_URL);
            assert_eq!(timeout, CODEX_VERSION_FETCH_TIMEOUT);
            Box::pin(async move {
                match call {
                    1 => Err(CodexVersionError::Unavailable(
                        "network unreachable".to_string(),
                    )),
                    2 => Err(CodexVersionError::Unavailable(
                        "npm @openai/codex version lookup failed with HTTP 503".to_string(),
                    )),
                    _ => Ok(npm_payload("0.118.0", CODEX_NPM_PACKAGE_NAME)),
                }
            })
        });

        let offset_clone = offset_ms.clone();
        let clock: CodexVersionClock = Arc::new(move || {
            base_instant + Duration::from_millis(offset_clone.load(AtomicOrdering::SeqCst) as u64)
        });

        let resolver = CodexVersionResolver::with_options(CodexVersionResolverOptions {
            fetch: Some(fetch),
            clock: Some(clock),
            ..Default::default()
        })
        .unwrap();

        let err1 = resolver.resolve().await.unwrap_err();
        assert_eq!(err1.status_code(), 503);
        assert!(err1.message().contains("network unreachable"));
        assert_eq!(resolver.cached_version(), None);

        // Retry after 100ms (without waiting 60s) must hit fetch again because no last_good_version exists.
        offset_ms.store(100, AtomicOrdering::SeqCst);
        let err2 = resolver.resolve().await.unwrap_err();
        assert!(err2.message().contains("HTTP 503"));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 2);

        offset_ms.store(200, AtomicOrdering::SeqCst);
        let resolved = resolver.resolve().await.unwrap();
        assert_eq!(resolved, "0.118.0");
        assert_eq!(resolver.cached_version().as_deref(), Some("0.118.0"));
        assert_eq!(attempts.load(AtomicOrdering::SeqCst), 3);
    }

    #[tokio::test]
    async fn caches_for_one_hour_ttl_and_refreshes_after_expiry() {
        let calls = Arc::new(AtomicUsize::new(0));
        let base_instant = Instant::now();
        let offset_ms = Arc::new(AtomicUsize::new(0));

        let calls_clone = calls.clone();
        let fetch: CodexVersionAsyncFetch = Arc::new(move |_url, _timeout| {
            let idx = calls_clone.fetch_add(1, AtomicOrdering::SeqCst);
            let version = if idx == 0 { "0.118.0" } else { "0.119.0" };
            let payload = npm_payload(version, CODEX_NPM_PACKAGE_NAME);
            Box::pin(async move { Ok(payload) })
        });

        let offset_clone = offset_ms.clone();
        let clock: CodexVersionClock = Arc::new(move || {
            base_instant + Duration::from_millis(offset_clone.load(AtomicOrdering::SeqCst) as u64)
        });

        let resolver = CodexVersionResolver::with_options(CodexVersionResolverOptions {
            fetch: Some(fetch),
            clock: Some(clock),
            ..Default::default()
        })
        .unwrap();

        assert_eq!(resolver.resolve().await.unwrap(), "0.118.0");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);

        let ttl_ms = CODEX_VERSION_CACHE_TTL.as_millis() as usize;
        offset_ms.store(ttl_ms - 1, AtomicOrdering::SeqCst);
        assert_eq!(resolver.resolve().await.unwrap(), "0.118.0");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);

        offset_ms.store(ttl_ms, AtomicOrdering::SeqCst);
        assert_eq!(resolver.resolve().await.unwrap(), "0.119.0");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retains_last_good_version_on_expired_refresh_failure_and_backs_off_60s() {
        let calls = Arc::new(AtomicUsize::new(0));
        let base_instant = Instant::now();
        let offset_ms = Arc::new(AtomicUsize::new(0));

        let calls_clone = calls.clone();
        let fetch: CodexVersionAsyncFetch = Arc::new(move |_url, _timeout| {
            let call = calls_clone.fetch_add(1, AtomicOrdering::SeqCst) + 1;
            Box::pin(async move {
                match call {
                    1 => Ok(npm_payload("0.118.0", CODEX_NPM_PACKAGE_NAME)),
                    2 => Err(CodexVersionError::Unavailable("HTTP 502".to_string())),
                    3 => Ok(npm_payload("0.119.0-rc.1", CODEX_NPM_PACKAGE_NAME)),
                    _ => Ok(npm_payload("0.119.0", CODEX_NPM_PACKAGE_NAME)),
                }
            })
        });

        let offset_clone = offset_ms.clone();
        let clock: CodexVersionClock = Arc::new(move || {
            base_instant + Duration::from_millis(offset_clone.load(AtomicOrdering::SeqCst) as u64)
        });

        let resolver = CodexVersionResolver::with_options(CodexVersionResolverOptions {
            fetch: Some(fetch),
            clock: Some(clock),
            ..Default::default()
        })
        .unwrap();

        assert_eq!(resolver.resolve().await.unwrap(), "0.118.0");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);

        let mut now_ms = CODEX_VERSION_CACHE_TTL.as_millis() as usize;
        offset_ms.store(now_ms, AtomicOrdering::SeqCst);
        assert_eq!(resolver.resolve().await.unwrap(), "0.118.0");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);

        let backoff_ms = CODEX_VERSION_RETRY_BACKOFF.as_millis() as usize;
        now_ms += backoff_ms - 1;
        offset_ms.store(now_ms, AtomicOrdering::SeqCst);
        assert_eq!(resolver.resolve().await.unwrap(), "0.118.0");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);

        // At 60s boundary: retries (#3), gets invalid prerelease -> retains 0.118.0 and sets another 60s backoff.
        now_ms += 1;
        offset_ms.store(now_ms, AtomicOrdering::SeqCst);
        assert_eq!(resolver.resolve().await.unwrap(), "0.118.0");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 3);

        now_ms += backoff_ms - 1;
        offset_ms.store(now_ms, AtomicOrdering::SeqCst);
        assert_eq!(resolver.resolve().await.unwrap(), "0.118.0");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 3);

        now_ms += 1;
        offset_ms.store(now_ms, AtomicOrdering::SeqCst);
        assert_eq!(resolver.resolve().await.unwrap(), "0.119.0");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 4);
    }

    #[tokio::test]
    async fn enforces_monotonic_no_downgrade_when_stale_upstream_returns_older_semver() {
        let calls = Arc::new(AtomicUsize::new(0));
        let base_instant = Instant::now();
        let offset_ms = Arc::new(AtomicUsize::new(0));
        let versions = ["0.100.0", "0.99.9", "0.100.0", "0.101.2", "0.101.1"];

        let calls_clone = calls.clone();
        let fetch: CodexVersionAsyncFetch = Arc::new(move |_url, _timeout| {
            let idx = calls_clone.fetch_add(1, AtomicOrdering::SeqCst);
            let payload = npm_payload(versions[idx], CODEX_NPM_PACKAGE_NAME);
            Box::pin(async move { Ok(payload) })
        });

        let offset_clone = offset_ms.clone();
        let clock: CodexVersionClock = Arc::new(move || {
            base_instant + Duration::from_millis(offset_clone.load(AtomicOrdering::SeqCst) as u64)
        });

        let resolver = CodexVersionResolver::with_options(CodexVersionResolverOptions {
            fetch: Some(fetch),
            clock: Some(clock),
            ..Default::default()
        })
        .unwrap();

        let ttl_ms = CODEX_VERSION_CACHE_TTL.as_millis() as usize;
        let backoff_ms = CODEX_VERSION_RETRY_BACKOFF.as_millis() as usize;
        let mut now_ms = 0usize;

        assert_eq!(resolver.resolve().await.unwrap(), "0.100.0");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);

        // Stale response 0.99.9 -> retains 0.100.0 and extends TTL by 1h.
        now_ms += ttl_ms;
        offset_ms.store(now_ms, AtomicOrdering::SeqCst);
        assert_eq!(resolver.resolve().await.unwrap(), "0.100.0");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);

        // Not in backoff mode: after 61s, still cached without re-fetching.
        offset_ms.store(now_ms + backoff_ms + 1_000, AtomicOrdering::SeqCst);
        assert_eq!(resolver.resolve().await.unwrap(), "0.100.0");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 2);

        now_ms += ttl_ms;
        offset_ms.store(now_ms, AtomicOrdering::SeqCst);
        assert_eq!(resolver.resolve().await.unwrap(), "0.100.0");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 3);

        now_ms += ttl_ms;
        offset_ms.store(now_ms, AtomicOrdering::SeqCst);
        assert_eq!(resolver.resolve().await.unwrap(), "0.101.2");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 4);

        now_ms += ttl_ms;
        offset_ms.store(now_ms, AtomicOrdering::SeqCst);
        assert_eq!(resolver.resolve().await.unwrap(), "0.101.2");
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 5);
    }

    #[tokio::test]
    async fn coalesces_concurrent_requests_with_singleflight() {
        let calls = Arc::new(AtomicUsize::new(0));
        let entered = Arc::new(Notify::new());
        let gate = Arc::new(Notify::new());

        let calls_clone = calls.clone();
        let entered_clone = entered.clone();
        let gate_clone = gate.clone();
        let fetch: CodexVersionAsyncFetch = Arc::new(move |_url, _timeout| {
            calls_clone.fetch_add(1, AtomicOrdering::SeqCst);
            let entered_inner = entered_clone.clone();
            let gate_inner = gate_clone.clone();
            Box::pin(async move {
                entered_inner.notify_waiters();
                gate_inner.notified().await;
                Ok(npm_payload("0.118.0", CODEX_NPM_PACKAGE_NAME))
            })
        });

        let resolver = CodexVersionResolver::with_options(CodexVersionResolverOptions {
            fetch: Some(fetch),
            ..Default::default()
        })
        .unwrap();

        let mut handles = Vec::new();
        for _ in 0..8 {
            let r = resolver.clone();
            handles.push(tokio::spawn(async move { r.resolve().await }));
        }

        entered.notified().await;
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
        gate.notify_waiters();

        for handle in handles {
            assert_eq!(handle.await.unwrap().unwrap(), "0.118.0");
        }
        assert_eq!(calls.load(AtomicOrdering::SeqCst), 1);
    }

    #[tokio::test]
    async fn performs_real_loopback_http_io_and_enforces_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let request_count = Arc::new(AtomicUsize::new(0));
        let observed_request = Arc::new(Mutex::new(String::new()));

        let request_count_clone = request_count.clone();
        let observed_request_clone = observed_request.clone();
        let server_task = tokio::spawn(async move {
            if let Ok((mut stream, _)) = listener.accept().await {
                request_count_clone.fetch_add(1, AtomicOrdering::SeqCst);
                let mut buf = [0u8; 2048];
                let n = stream.read(&mut buf).await.unwrap_or(0);
                *observed_request_clone.lock().unwrap() =
                    String::from_utf8_lossy(&buf[..n]).to_string();
                let body = r#"{"name":"@openai/codex","version":"0.121.0"}"#;
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = stream.write_all(response.as_bytes()).await;
            }
        });

        let registry_url = format!("http://{addr}/@openai%2fcodex/latest");
        let resolver = CodexVersionResolver::with_options(CodexVersionResolverOptions {
            registry_url,
            ..Default::default()
        })
        .unwrap();

        let version = resolver.resolve().await.unwrap();
        assert_eq!(version, "0.121.0");
        assert_eq!(request_count.load(AtomicOrdering::SeqCst), 1);
        let raw_req = observed_request.lock().unwrap().clone();
        let raw_lower = raw_req.to_ascii_lowercase();
        assert!(raw_req.starts_with("GET /@openai%2fcodex/latest HTTP/1.1\r\n"));
        assert!(raw_lower.contains("accept: application/json\r\n"));
        assert!(!raw_lower.contains("authorization:"));

        let headers = resolver.headers().await.unwrap();
        assert_eq!(
            headers.get("originator").map(String::as_str),
            Some(CODEX_CLI_ORIGINATOR)
        );
        assert!(headers
            .get("User-Agent")
            .unwrap()
            .starts_with("codex_cli_rs/0.121.0 "));
        let _ = server_task.await;

        // Verify timeout against a stalled TCP listener.
        let stalled_listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let stalled_addr = stalled_listener.local_addr().unwrap();
        let hold_task = tokio::spawn(async move {
            let _accepted = stalled_listener.accept().await;
            tokio::time::sleep(Duration::from_millis(250)).await;
        });

        let timeout_resolver = CodexVersionResolver::with_options(CodexVersionResolverOptions {
            registry_url: format!("http://{stalled_addr}/@openai%2fcodex/latest"),
            timeout: Duration::from_millis(40),
            ..Default::default()
        })
        .unwrap();

        let err = timeout_resolver.resolve().await.unwrap_err();
        assert_eq!(err.status_code(), 503);
        assert!(err.message().contains("timed out"), "{}", err.message());
        hold_task.abort();

        // Verify global resolver helpers and blocking wrappers.
        let global_calls = Arc::new(AtomicUsize::new(0));
        let global_calls_clone = global_calls.clone();
        configure_codex_version_cache(CodexVersionResolverOptions {
            fetch: Some(Arc::new(move |_url, _timeout| {
                global_calls_clone.fetch_add(1, AtomicOrdering::SeqCst);
                Box::pin(async move { Ok(npm_payload("0.123.0", CODEX_NPM_PACKAGE_NAME)) })
            })),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(get_cached_codex_cli_version(), None);
        assert_eq!(resolve_codex_cli_version().await.unwrap(), "0.123.0");
        assert_eq!(resolve_latest_codex_version().await.unwrap(), "0.123.0");
        assert_eq!(get_cached_codex_cli_version().as_deref(), Some("0.123.0"));
        assert_eq!(resolve_codex_cli_version_blocking().unwrap(), "0.123.0");
        let global_headers = resolve_codex_cli_headers().await.unwrap();
        assert_eq!(
            global_headers,
            resolve_codex_cli_headers_blocking().unwrap()
        );
        assert_eq!(global_calls.load(AtomicOrdering::SeqCst), 1);
        reset_codex_version_cache();
        assert_eq!(get_cached_codex_cli_version(), None);
    }
}
