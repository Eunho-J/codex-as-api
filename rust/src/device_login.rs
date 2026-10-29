//! Modified port of OpenAI Codex `codex-rs/login/src/device_code_auth.rs`,
//! release rust-v0.164.0. Copyright OpenAI. Licensed under Apache License 2.0;
//! see THIRD_PARTY_NOTICES for attribution and license details.
//! No credentials are exposed by session APIs.
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;
use tokio::time::{timeout_at, Instant};
use uuid::Uuid;

const ISSUER: &str = "https://auth.openai.com";
pub(crate) const CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";
const LIFETIME: Duration = Duration::from_secs(900);
const MAX_RECORDS: usize = 32;

/// Call exactly once, while holding the store lock, around the atomic commit.
/// The fence rejects cancelled or expired sessions and marks the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommitError {
    Storage,
    PersistenceUncertain,
}
pub type SessionFence<'a> =
    dyn FnMut(&mut dyn FnMut() -> Result<(), CommitError>) -> Result<(), CommitError> + 'a;
/// Acquire the auth store lock first, then call the supplied session fence around
/// credential validation and durable atomic replacement. The store writer must
/// also check the deadline immediately before replacement. Never log secrets.
pub type Commit =
    Arc<dyn Fn(Value, Instant, &mut SessionFence<'_>) -> Result<(), CommitError> + Send + Sync>;

#[derive(Clone, Serialize, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoginStatus {
    Pending,
    Succeeded,
    Failed,
    Cancelled,
    Expired,
}

#[derive(Clone, Serialize, Debug)]
pub struct Session {
    pub id: String,
    pub method: &'static str,
    pub status: LoginStatus,
    pub expires_at: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_code: Option<&'static str>,
}

#[derive(Serialize)]
pub struct Started {
    #[serde(flatten)]
    pub session: Session,
    pub verification_url: String,
    pub user_code: String,
    pub interval: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StartError {
    InProgress,
    Unsupported,
    Upstream,
    InvalidResponse,
    Expired,
    Cancelled,
    CallbackPortUnavailable,
}
impl StartError {
    pub fn code(self) -> &'static str {
        match self {
            Self::InProgress => "login_in_progress",
            Self::Unsupported => "device_login_unsupported",
            Self::Upstream => "login_upstream_error",
            Self::InvalidResponse => "invalid_login_response",
            Self::Expired => "login_expired",
            Self::Cancelled => "login_cancelled",
            Self::CallbackPortUnavailable => "callback_port_unavailable",
        }
    }
}

struct Record {
    session: Session,
    cancel: Arc<Notify>,
}
#[derive(Default)]
struct State {
    active: Option<String>,
    records: HashMap<String, Record>,
    order: VecDeque<String>,
    closed: bool,
}
struct Inner {
    state: Mutex<State>,
    client: reqwest::Client,
    commit: Commit,
    issuer: String,
    lifetime: Duration,
}
#[derive(Clone)]
pub struct DeviceLogin {
    inner: Arc<Inner>,
}

impl DeviceLogin {
    /// Create one manager per credential store. The HTTP client deliberately
    /// refuses redirects, including redirects to attacker-controlled hosts.
    pub fn new(commit: Commit) -> Result<Self, reqwest::Error> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self::with_parts(
            client,
            commit,
            ISSUER.to_owned(),
            LIFETIME,
        ))
    }
    #[cfg(test)]
    pub(crate) fn test_with_issuer(commit: Commit, issuer: String) -> Result<Self, reqwest::Error> {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()?;
        Ok(Self::with_parts(client, commit, issuer, LIFETIME))
    }
    pub(crate) fn with_parts(
        client: reqwest::Client,
        commit: Commit,
        issuer: String,
        lifetime: Duration,
    ) -> Self {
        Self {
            inner: Arc::new(Inner {
                state: Mutex::new(State::default()),
                client,
                commit,
                issuer,
                lifetime,
            }),
        }
    }
    pub async fn start_browser(&self) -> Result<crate::browser_login::BrowserStarted, StartError> {
        crate::browser_login::start(self).await
    }

    pub(crate) fn client(&self) -> &reqwest::Client {
        &self.inner.client
    }
    pub(crate) fn issuer(&self) -> &str {
        &self.inner.issuer
    }

    pub(crate) fn reserve(
        &self,
        method: &'static str,
    ) -> Result<(Session, Instant, Arc<Notify>), StartError> {
        let deadline = Instant::now() + self.inner.lifetime;
        let expires_at = Utc::now()
            + ChronoDuration::from_std(self.inner.lifetime).map_err(|_| StartError::Upstream)?;
        let id = Uuid::new_v4().to_string();
        let cancel = Arc::new(Notify::new());
        let session = Session {
            id: id.clone(),
            method,
            status: LoginStatus::Pending,
            expires_at,
            error_code: None,
        };
        {
            let mut state = self.inner.state.lock().unwrap();
            if state.closed {
                return Err(StartError::Cancelled);
            }
            if state.active.is_some() {
                return Err(StartError::InProgress);
            }
            while state.order.len() >= MAX_RECORDS {
                if let Some(old) = state.order.pop_front() {
                    state.records.remove(&old);
                }
            }
            state.active = Some(id.clone());
            state.order.push_back(id.clone());
            state.records.insert(
                id.clone(),
                Record {
                    session: session.clone(),
                    cancel: cancel.clone(),
                },
            );
        }
        Ok((session, deadline, cancel))
    }

    /// Reserve the store before contacting the provider; dropping the future cancels it.
    pub async fn start(&self) -> Result<Started, StartError> {
        let (session, deadline, cancel) = self.reserve("device")?;
        let id = session.id;
        let mut guard = StartGuard {
            manager: self.clone(),
            id: id.clone(),
            armed: true,
        };
        let response = tokio::select! {
            _ = cancel.notified() => return Err(StartError::Cancelled),
            response = timeout_at(deadline, self.inner.client.post(format!("{}/api/accounts/deviceauth/usercode", self.inner.issuer)).json(&json!({"client_id": CLIENT_ID})).send()) => response,
        };
        let response = match response {
            Err(_) => {
                self.finish(&id, LoginStatus::Expired, Some("login_expired"));
                return Err(StartError::Expired);
            }
            Ok(Err(_)) => {
                self.finish(&id, LoginStatus::Failed, Some("login_upstream_error"));
                return Err(StartError::Upstream);
            }
            Ok(Ok(r)) => r,
        };
        if !response.status().is_success() {
            let error = if response.status().as_u16() == 404 {
                StartError::Unsupported
            } else {
                StartError::Upstream
            };
            self.finish(&id, LoginStatus::Failed, Some(error.code()));
            return Err(error);
        }
        let body = tokio::select! {
            _ = cancel.notified() => return Err(StartError::Cancelled),
            body = timeout_at(deadline, response.json::<Value>()) => body,
        };
        let body = match body {
            Err(_) => {
                self.finish(&id, LoginStatus::Expired, Some("login_expired"));
                return Err(StartError::Expired);
            }
            Ok(Err(_)) => {
                self.finish(&id, LoginStatus::Failed, Some("invalid_login_response"));
                return Err(StartError::InvalidResponse);
            }
            Ok(Ok(v)) => v,
        };
        let parsed = (|| {
            let auth_id = nonempty(body.get("device_auth_id")?)?;
            let user_code = nonempty(body.get("user_code").or_else(|| body.get("usercode"))?)?;
            let interval = match body.get("interval") {
                None => 5,
                Some(Value::Number(n)) => n.as_u64()?,
                Some(Value::String(s))
                    if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) =>
                {
                    s.parse().ok()?
                }
                _ => return None,
            };
            if interval == 0 {
                return None;
            }
            Some((auth_id.to_owned(), user_code.to_owned(), interval))
        })();
        let (auth_id, user_code, interval) = match parsed {
            Some(v) => v,
            None => {
                self.finish(&id, LoginStatus::Failed, Some("invalid_login_response"));
                return Err(StartError::InvalidResponse);
            }
        };
        let session = match self.get(&id) {
            Some(s) if s.status == LoginStatus::Pending => s,
            _ => return Err(StartError::Cancelled),
        };
        let manager = self.clone();
        let task_id = id.clone();
        let task_code = user_code.clone();
        tokio::spawn(async move {
            manager
                .run(task_id, auth_id, task_code, interval, deadline, cancel)
                .await;
        });
        guard.armed = false;
        Ok(Started {
            session,
            verification_url: format!("{}/codex/device", self.inner.issuer),
            user_code,
            interval,
        })
    }
    pub fn get(&self, id: &str) -> Option<Session> {
        self.inner
            .state
            .lock()
            .unwrap()
            .records
            .get(id)
            .map(|r| r.session.clone())
    }
    /// Idempotent; the lock also fences the final synchronous commit.
    pub fn cancel(&self, id: &str) -> Option<Session> {
        let mut state = self.inner.state.lock().unwrap();
        let record = state.records.get_mut(id)?;
        if record.session.status == LoginStatus::Pending {
            record.session.status = LoginStatus::Cancelled;
            record.cancel.notify_one();
            if state.active.as_deref() == Some(id) {
                state.active = None;
            }
        }
        state.records.get(id).map(|r| r.session.clone())
    }
    pub fn shutdown(&self) {
        let mut state = self.inner.state.lock().unwrap();
        state.closed = true;
        if let Some(id) = state.active.take() {
            if let Some(record) = state.records.get_mut(&id) {
                record.session.status = LoginStatus::Cancelled;
                record.cancel.notify_one();
            }
        }
    }
    pub(crate) fn finish(&self, id: &str, status: LoginStatus, code: Option<&'static str>) {
        let mut state = self.inner.state.lock().unwrap();
        if let Some(record) = state.records.get_mut(id) {
            if record.session.status != LoginStatus::Pending {
                return;
            }
            record.session.status = status;
            record.session.error_code = code;
            if state.active.as_deref() == Some(id) {
                state.active = None;
            }
        }
    }
    async fn run(
        &self,
        id: String,
        auth_id: String,
        user_code: String,
        interval: u64,
        deadline: Instant,
        cancel: Arc<Notify>,
    ) {
        let result = tokio::select! {
            _ = cancel.notified() => return,
            result = timeout_at(deadline, self.poll_and_exchange(&auth_id, &user_code, interval)) => result,
        };
        match result {
            Err(_) => self.finish(&id, LoginStatus::Expired, Some("login_expired")),
            Ok(Err(code)) => self.finish(&id, LoginStatus::Failed, Some(code)),
            Ok(Ok(tokens)) => {
                let manager = self.clone();
                let worker_id = id.clone();
                let worker = tokio::task::spawn_blocking(move || {
                    manager.commit_tokens(worker_id, tokens, deadline)
                });
                tokio::select! {
                    result = worker => if result.is_err() { self.finish(&id, LoginStatus::Failed, Some("login_storage_error")); },
                    _ = tokio::time::sleep_until(deadline) => self.finish(&id, LoginStatus::Expired, Some("login_expired")),
                }
            }
        }
    }
    pub(crate) fn commit_tokens(&self, id: String, tokens: Value, deadline: Instant) {
        let mut fence = |write: &mut dyn FnMut() -> Result<(), CommitError>| {
            let mut state = self.inner.state.lock().unwrap();
            if !state
                .records
                .get(&id)
                .is_some_and(|r| r.session.status == LoginStatus::Pending)
            {
                return Err(CommitError::Storage);
            }
            let result = if Instant::now() >= deadline {
                Err(CommitError::Storage)
            } else {
                write()
            };
            let record = state.records.get_mut(&id).unwrap();
            if result.is_ok() {
                record.session.status = LoginStatus::Succeeded;
            } else if result == Err(CommitError::PersistenceUncertain) {
                record.session.status = LoginStatus::Failed;
                record.session.error_code = Some("login_persistence_uncertain");
            } else if Instant::now() >= deadline {
                record.session.status = LoginStatus::Expired;
                record.session.error_code = Some("login_expired");
            } else {
                record.session.status = LoginStatus::Failed;
                record.session.error_code = Some("login_storage_error");
            }
            if state.active.as_deref() == Some(&id) {
                state.active = None;
            }
            result
        };
        let result = (self.inner.commit)(tokens, deadline, &mut fence);
        if result.is_err() {
            self.finish(
                &id,
                if Instant::now() >= deadline {
                    LoginStatus::Expired
                } else {
                    LoginStatus::Failed
                },
                Some(if Instant::now() >= deadline {
                    "login_expired"
                } else {
                    "login_storage_error"
                }),
            );
        } else {
            self.finish(&id, LoginStatus::Failed, Some("login_storage_error"));
        }
    }
    async fn poll_and_exchange(
        &self,
        auth_id: &str,
        user_code: &str,
        interval: u64,
    ) -> Result<Value, &'static str> {
        loop {
            let response = self
                .inner
                .client
                .post(format!(
                    "{}/api/accounts/deviceauth/token",
                    self.inner.issuer
                ))
                .json(&json!({"device_auth_id":auth_id,"user_code":user_code}))
                .send()
                .await
                .map_err(|_| "login_upstream_error")?;
            if response.status().as_u16() == 403 || response.status().as_u16() == 404 {
                tokio::time::sleep(Duration::from_secs(interval).min(self.inner.lifetime)).await;
                continue;
            }
            if !response.status().is_success() {
                return Err("login_upstream_error");
            }
            let data: Value = response
                .json()
                .await
                .map_err(|_| "invalid_login_response")?;
            let code = data
                .get("authorization_code")
                .and_then(nonempty)
                .ok_or("invalid_login_response")?;
            let verifier = data
                .get("code_verifier")
                .and_then(nonempty)
                .ok_or("invalid_login_response")?;
            data.get("code_challenge")
                .and_then(nonempty)
                .ok_or("invalid_login_response")?;
            let callback = format!("{}/deviceauth/callback", self.inner.issuer);
            let response = self
                .inner
                .client
                .post(format!("{}/oauth/token", self.inner.issuer))
                .form(&[
                    ("grant_type", "authorization_code"),
                    ("client_id", CLIENT_ID),
                    ("code", code),
                    ("redirect_uri", &callback),
                    ("code_verifier", verifier),
                ])
                .send()
                .await
                .map_err(|_| "login_upstream_error")?;
            if !response.status().is_success() {
                return Err("login_upstream_error");
            }
            let tokens: Value = response
                .json()
                .await
                .map_err(|_| "invalid_login_response")?;
            for key in ["access_token", "refresh_token", "id_token"] {
                tokens
                    .get(key)
                    .and_then(nonempty)
                    .ok_or("invalid_login_response")?;
            }
            return Ok(tokens);
        }
    }
}
fn nonempty(value: &Value) -> Option<&str> {
    value.as_str().filter(|s| !s.is_empty())
}
struct StartGuard {
    manager: DeviceLogin,
    id: String,
    armed: bool,
}
impl Drop for StartGuard {
    fn drop(&mut self) {
        if self.armed {
            self.manager.cancel(&self.id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn fake(responses: Vec<(u16, &'static str)>) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            for (status, body) in responses {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut data = Vec::new();
                loop {
                    let mut buf = [0; 4096];
                    let n = socket.read(&mut buf).await.unwrap();
                    if n == 0 {
                        break;
                    }
                    data.extend_from_slice(&buf[..n]);
                    if data.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                let response = format!("HTTP/1.1 {status} response\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{address}")
    }
    fn manager(issuer: String, lifetime: Duration, writes: Arc<AtomicUsize>) -> DeviceLogin {
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        DeviceLogin::with_parts(
            client,
            Arc::new(move |_, _, fence| {
                fence(&mut || {
                    writes.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
            }),
            issuer,
            lifetime,
        )
    }
    async fn terminal(manager: &DeviceLogin, id: &str) -> Session {
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                let session = manager.get(id).unwrap();
                if session.status != LoginStatus::Pending {
                    return session;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap()
    }
    // Run the real background operation with a join handle so cancellation tests
    // can prove it has stopped before the server releases its late response.
    fn spawn_pending_run(m: &DeviceLogin) -> (String, tokio::task::JoinHandle<()>) {
        let id = Uuid::new_v4().to_string();
        let cancel = Arc::new(Notify::new());
        {
            let mut state = m.inner.state.lock().unwrap();
            state.active = Some(id.clone());
            state.records.insert(
                id.clone(),
                Record {
                    session: Session {
                        id: id.clone(),
                        method: "device",
                        status: LoginStatus::Pending,
                        expires_at: Utc::now() + ChronoDuration::seconds(30),
                        error_code: None,
                    },
                    cancel: cancel.clone(),
                },
            );
        }
        let worker = m.clone();
        let worker_id = id.clone();
        let handle = tokio::spawn(async move {
            worker
                .run(
                    worker_id,
                    "x".into(),
                    "y".into(),
                    1,
                    Instant::now() + Duration::from_secs(30),
                    cancel,
                )
                .await;
        });
        (id, handle)
    }

    async fn gated_server(
        exchange: bool,
    ) -> (
        String,
        tokio::sync::oneshot::Receiver<()>,
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<()>,
    ) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (entered, entered_rx) = tokio::sync::oneshot::channel();
        let (release, release_rx) = tokio::sync::oneshot::channel();
        let handle = tokio::spawn(async move {
            let responses = if exchange {
                vec![
                    (
                        "/api/accounts/deviceauth/token",
                        r#"{"authorization_code":"c","code_challenge":"h","code_verifier":"v"}"#,
                    ),
                    (
                        "/oauth/token",
                        r#"{"access_token":"a","refresh_token":"r","id_token":"i"}"#,
                    ),
                ]
            } else {
                vec![(
                    "/api/accounts/deviceauth/token",
                    r#"{"authorization_code":"c","code_challenge":"h","code_verifier":"v"}"#,
                )]
            };
            let mut entered = Some(entered);
            let mut release_rx = Some(release_rx);
            let last = responses.len() - 1;
            for (index, (path, body)) in responses.into_iter().enumerate() {
                let (mut socket, _) = listener.accept().await.unwrap();
                let mut data = Vec::new();
                loop {
                    let mut buf = [0; 4096];
                    let n = socket.read(&mut buf).await.unwrap();
                    assert_ne!(n, 0);
                    data.extend_from_slice(&buf[..n]);
                    if data.windows(4).any(|w| w == b"\r\n\r\n") {
                        break;
                    }
                }
                assert!(String::from_utf8_lossy(&data).starts_with(&format!("POST {path} ")));
                if index == last {
                    entered.take().unwrap().send(()).unwrap();
                    release_rx.take().unwrap().await.unwrap();
                }
                let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        (format!("http://{address}"), entered_rx, release, handle)
    }

    async fn cancellation_with_late_response(exchange: bool) {
        let (issuer, entered, release, server) = gated_server(exchange).await;
        let writes = Arc::new(AtomicUsize::new(0));
        let m = manager(issuer, Duration::from_secs(30), writes.clone());
        let (id, worker) = spawn_pending_run(&m);
        tokio::time::timeout(Duration::from_secs(3), entered)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(m.cancel(&id).unwrap().status, LoginStatus::Cancelled);
        tokio::time::timeout(Duration::from_secs(3), worker)
            .await
            .unwrap()
            .unwrap();
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(3), server)
            .await
            .unwrap()
            .unwrap();
        let session = m.get(&id).unwrap();
        assert_eq!(session.status, LoginStatus::Cancelled);
        assert_eq!(session.error_code, None);
        assert_eq!(writes.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn cancellation_during_poll_ignores_late_response() {
        cancellation_with_late_response(false).await;
    }

    #[tokio::test]
    async fn cancellation_during_exchange_ignores_late_response() {
        cancellation_with_late_response(true).await;
    }

    #[tokio::test]
    async fn terminal_poll_denial_does_not_commit() {
        let writes = Arc::new(AtomicUsize::new(0));
        let m = manager(
            fake(vec![
                (
                    200,
                    r#"{"device_auth_id":"x","user_code":"y","interval":1}"#,
                ),
                (401, r#"{"error":"access_denied"}"#),
            ])
            .await,
            Duration::from_secs(3),
            writes.clone(),
        );
        let started = m.start().await.unwrap();
        let session = terminal(&m, &started.session.id).await;
        assert_eq!(session.status, LoginStatus::Failed);
        assert_eq!(session.error_code, Some("login_upstream_error"));
        assert_eq!(writes.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn malformed_exchange_token_does_not_commit() {
        for body in [
            "not json",
            r#"{"access_token":"a","refresh_token":"r"}"#,
            r#"{"access_token":"a","refresh_token":"","id_token":"i"}"#,
        ] {
            let writes = Arc::new(AtomicUsize::new(0));
            let m = manager(
                fake(vec![
                    (
                        200,
                        r#"{"device_auth_id":"x","user_code":"y","interval":1}"#,
                    ),
                    (
                        200,
                        r#"{"authorization_code":"c","code_challenge":"h","code_verifier":"v"}"#,
                    ),
                    (200, body),
                ])
                .await,
                Duration::from_secs(3),
                writes.clone(),
            );
            let started = m.start().await.unwrap();
            let session = terminal(&m, &started.session.id).await;
            assert_eq!(session.status, LoginStatus::Failed);
            assert_eq!(session.error_code, Some("invalid_login_response"));
            assert_eq!(writes.load(Ordering::SeqCst), 0);
        }
    }

    #[tokio::test]
    async fn ambiguous_commit_failure_remains_failed_after_cancel() {
        let issuer = fake(vec![
            (
                200,
                r#"{"device_auth_id":"x","user_code":"y","interval":1}"#,
            ),
            (
                200,
                r#"{"authorization_code":"c","code_challenge":"h","code_verifier":"v"}"#,
            ),
            (
                200,
                r#"{"access_token":"a","refresh_token":"r","id_token":"i"}"#,
            ),
        ])
        .await;
        let effects = Arc::new(AtomicUsize::new(0));
        let count = effects.clone();
        let m = DeviceLogin::test_with_issuer(
            Arc::new(move |_, _, fence| {
                fence(&mut || {
                    // Simulate replacement succeeding, then a subsequent I/O error.
                    count.fetch_add(1, Ordering::SeqCst);
                    Err(CommitError::PersistenceUncertain)
                })
            }),
            issuer,
        )
        .unwrap();
        let started = m.start().await.unwrap();
        let session = terminal(&m, &started.session.id).await;
        assert_eq!(session.status, LoginStatus::Failed);
        assert_eq!(session.error_code, Some("login_persistence_uncertain"));
        assert_eq!(effects.load(Ordering::SeqCst), 1);
        let cancelled = m.cancel(&session.id).unwrap();
        assert_eq!(cancelled.status, LoginStatus::Failed);
        assert_eq!(cancelled.error_code, Some("login_persistence_uncertain"));
        assert_eq!(effects.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn uncertain_write_crossing_deadline_stays_failed() {
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let manager = DeviceLogin::new(Arc::new(move |_, _, fence| {
            fence(&mut || {
                entered_tx.send(()).unwrap();
                release_rx.lock().unwrap().recv().unwrap();
                Err(CommitError::PersistenceUncertain)
            })
        }))
        .unwrap();
        let id = Uuid::new_v4().to_string();
        {
            let mut state = manager.inner.state.lock().unwrap();
            state.active = Some(id.clone());
            state.records.insert(
                id.clone(),
                Record {
                    session: Session {
                        id: id.clone(),
                        method: "device",
                        status: LoginStatus::Pending,
                        expires_at: Utc::now(),
                        error_code: None,
                    },
                    cancel: Arc::new(Notify::new()),
                },
            );
        }
        let deadline = Instant::now() + Duration::from_millis(200);
        let worker_manager = manager.clone();
        let worker_id = id.clone();
        let worker = std::thread::spawn(move || {
            worker_manager.commit_tokens(worker_id, json!({}), deadline)
        });
        entered_rx.recv_timeout(Duration::from_secs(3)).unwrap();
        while Instant::now() < deadline {
            std::thread::yield_now();
        }
        release_tx.send(()).unwrap();
        worker.join().unwrap();
        let session = manager.cancel(&id).unwrap();
        assert_eq!(session.status, LoginStatus::Failed);
        assert_eq!(session.error_code, Some("login_persistence_uncertain"));
    }

    #[tokio::test]
    async fn successful_exchange_and_private_records() {
        let issuer = fake(vec![
            (200, r#"{"device_auth_id":"secret-id","usercode":"ABCD","interval":"1"}"#),
            (403, "{}"),
            (200, r#"{"authorization_code":"secret-code","code_challenge":"challenge","code_verifier":"verifier"}"#),
            (200, r#"{"access_token":"access-secret","refresh_token":"refresh-secret","id_token":"id-secret"}"#),
        ]).await;
        let writes = Arc::new(AtomicUsize::new(0));
        let manager = manager(issuer, Duration::from_secs(4), writes.clone());
        let started = manager.start().await.unwrap();
        assert_eq!(started.user_code, "ABCD");
        assert!(matches!(manager.start().await, Err(StartError::InProgress)));
        let session = terminal(&manager, &started.session.id).await;
        assert_eq!(session.status, LoginStatus::Succeeded);
        assert_eq!(writes.load(Ordering::SeqCst), 1);
        let exposed = serde_json::to_string(&session).unwrap();
        for secret in [
            "ABCD",
            "secret-id",
            "secret-code",
            "verifier",
            "access-secret",
        ] {
            assert!(!exposed.contains(secret));
        }
        assert_eq!(
            manager.cancel(&session.id).unwrap().status,
            LoginStatus::Succeeded
        );
    }
    #[tokio::test]
    async fn huge_interval_does_not_delay_first_poll_or_overflow() {
        let writes = Arc::new(AtomicUsize::new(0));
        let m = manager(
            fake(vec![
                (
                    200,
                    r#"{"device_auth_id":"x","user_code":"y","interval":"18446744073709551615"}"#,
                ),
                (
                    200,
                    r#"{"authorization_code":"c","code_challenge":"h","code_verifier":"v"}"#,
                ),
                (
                    200,
                    r#"{"access_token":"a","refresh_token":"r","id_token":"i"}"#,
                ),
            ])
            .await,
            Duration::from_millis(500),
            writes.clone(),
        );
        let started = m.start().await.unwrap();
        assert_eq!(
            terminal(&m, &started.session.id).await.status,
            LoginStatus::Succeeded
        );
        assert_eq!(writes.load(Ordering::SeqCst), 1);
        let m = manager(
            fake(vec![
                (
                    200,
                    r#"{"device_auth_id":"x","user_code":"y","interval":"18446744073709551615"}"#,
                ),
                (404, "{}"),
            ])
            .await,
            Duration::from_millis(80),
            writes.clone(),
        );
        let started = m.start().await.unwrap();
        assert_eq!(
            terminal(&m, &started.session.id).await.status,
            LoginStatus::Expired
        );
        assert_eq!(writes.load(Ordering::SeqCst), 1);
    }
    #[tokio::test]
    async fn unsupported_and_malformed_are_safe() {
        let writes = Arc::new(AtomicUsize::new(0));
        let m = manager(
            fake(vec![(404, "credential-secret")]).await,
            Duration::from_secs(2),
            writes.clone(),
        );
        assert!(matches!(m.start().await, Err(StartError::Unsupported)));
        let m = manager(
            fake(vec![(
                200,
                r#"{"device_auth_id":"x","user_code":"y","interval":"0"}"#,
            )])
            .await,
            Duration::from_secs(2),
            writes.clone(),
        );
        assert!(matches!(m.start().await, Err(StartError::InvalidResponse)));
        assert_eq!(writes.load(Ordering::SeqCst), 0);
    }
    #[tokio::test]
    async fn expiry_while_waiting_for_store_lock_prevents_commit() {
        let issuer = fake(vec![
            (
                200,
                r#"{"device_auth_id":"x","user_code":"y","interval":1}"#,
            ),
            (
                200,
                r#"{"authorization_code":"c","code_challenge":"h","code_verifier":"v"}"#,
            ),
            (
                200,
                r#"{"access_token":"a","refresh_token":"r","id_token":"i"}"#,
            ),
        ])
        .await;
        let (release, wait) = std::sync::mpsc::channel::<()>();
        let wait = Mutex::new(wait);
        let writes = Arc::new(AtomicUsize::new(0));
        let count = writes.clone();
        let m = DeviceLogin::with_parts(
            reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap(),
            Arc::new(move |_, _, fence| {
                wait.lock().unwrap().recv().unwrap(); // Simulate a busy auth store lock.
                fence(&mut || {
                    count.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
            }),
            issuer,
            Duration::from_millis(120),
        );
        let started = m.start().await.unwrap();
        assert_eq!(
            terminal(&m, &started.session.id).await.status,
            LoginStatus::Expired
        );
        release.send(()).unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(writes.load(Ordering::SeqCst), 0);
    }
    #[tokio::test]
    async fn cancellation_while_waiting_for_store_lock_prevents_commit() {
        let issuer = fake(vec![
            (
                200,
                r#"{"device_auth_id":"x","user_code":"y","interval":1}"#,
            ),
            (
                200,
                r#"{"authorization_code":"c","code_challenge":"h","code_verifier":"v"}"#,
            ),
            (
                200,
                r#"{"access_token":"a","refresh_token":"r","id_token":"i"}"#,
            ),
        ])
        .await;
        let (release, wait) = std::sync::mpsc::channel::<()>();
        let (entered, entered_rx) = tokio::sync::oneshot::channel::<()>();
        let entered = Mutex::new(Some(entered));
        let wait = Mutex::new(wait);
        let writes = Arc::new(AtomicUsize::new(0));
        let count = writes.clone();
        let m = DeviceLogin::with_parts(
            reqwest::Client::builder()
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap(),
            Arc::new(move |_, _, fence| {
                if let Some(sender) = entered.lock().unwrap().take() {
                    let _ = sender.send(());
                }
                wait.lock().unwrap().recv().unwrap();
                fence(&mut || {
                    count.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })
            }),
            issuer,
            Duration::from_secs(3),
        );
        let started = m.start().await.unwrap();
        tokio::time::timeout(Duration::from_secs(1), entered_rx)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(
            m.cancel(&started.session.id).unwrap().status,
            LoginStatus::Cancelled
        );
        release.send(()).unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(writes.load(Ordering::SeqCst), 0);
        assert_eq!(
            m.get(&started.session.id).unwrap().status,
            LoginStatus::Cancelled
        );
    }
    #[tokio::test]
    async fn methods_share_one_slot_and_preserve_method() {
        let m = manager(
            "http://127.0.0.1:1".into(),
            Duration::from_secs(3),
            Arc::new(AtomicUsize::new(0)),
        );
        let (browser, _, _) = m.reserve("browser").unwrap();
        assert_eq!(browser.method, "browser");
        assert!(matches!(m.start().await, Err(StartError::InProgress)));
        assert_eq!(m.cancel(&browser.id).unwrap().method, "browser");
        let (device, _, _) = m.reserve("device").unwrap();
        assert_eq!(device.method, "device");
        assert!(matches!(
            m.start_browser().await,
            Err(StartError::InProgress)
        ));
        assert_eq!(m.cancel(&device.id).unwrap().method, "device");
    }
    #[tokio::test]
    async fn cancellation_and_deadline_prevent_writes() {
        let writes = Arc::new(AtomicUsize::new(0));
        let m = manager(
            fake(vec![
                (
                    200,
                    r#"{"device_auth_id":"x","user_code":"y","interval":1}"#,
                ),
                (404, "{}"),
            ])
            .await,
            Duration::from_secs(2),
            writes.clone(),
        );
        let s = m.start().await.unwrap();
        assert_eq!(
            m.cancel(&s.session.id).unwrap().status,
            LoginStatus::Cancelled
        );
        assert_eq!(
            m.cancel(&s.session.id).unwrap().status,
            LoginStatus::Cancelled
        );
        let m = manager(
            fake(vec![
                (
                    200,
                    r#"{"device_auth_id":"x","user_code":"y","interval":1}"#,
                ),
                (403, "{}"),
            ])
            .await,
            Duration::from_millis(80),
            writes.clone(),
        );
        let s = m.start().await.unwrap();
        assert_eq!(
            terminal(&m, &s.session.id).await.status,
            LoginStatus::Expired
        );
        assert_eq!(writes.load(Ordering::SeqCst), 0);
    }
}
