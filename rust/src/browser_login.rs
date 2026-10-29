//! Browser authorization-code flow based on OpenAI Codex rust-v0.164.0
//! login/src/server.rs and oauth/{authorization,pkce,client}.rs, Apache-2.0.
use crate::device_login::{DeviceLogin, LoginStatus, Session, StartError, CLIENT_ID};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, TcpStream},
    sync::Notify,
    time::{timeout, timeout_at, Instant},
};
use uuid::Uuid;

#[derive(Serialize)]
pub struct BrowserStarted {
    #[serde(flatten)]
    pub session: Session,
    pub auth_url: String,
}

// Two independent UUIDv4 values provide 244 bits of OS-backed randomness.
fn secret() -> String {
    format!("{}{}", Uuid::new_v4().simple(), Uuid::new_v4().simple())
}

pub(crate) async fn start(manager: &DeviceLogin) -> Result<BrowserStarted, StartError> {
    start_with_ports(manager, &[1455, 1457]).await
}

async fn start_with_ports(
    manager: &DeviceLogin,
    ports: &[u16],
) -> Result<BrowserStarted, StartError> {
    let (session, deadline, cancel) = manager.reserve("browser")?;
    // No await between reservation and spawning: dropping the start future cannot
    // strand an active session.
    let listener = ports.iter().find_map(|port| {
        let listener = std::net::TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, *port)).ok()?;
        listener.set_nonblocking(true).ok()?;
        TcpListener::from_std(listener).ok()
    });
    let Some(listener) = listener else {
        manager.finish(
            &session.id,
            LoginStatus::Failed,
            Some("callback_port_unavailable"),
        );
        return Err(StartError::CallbackPortUnavailable);
    };
    let port = match listener.local_addr() {
        Ok(addr) => addr.port(),
        Err(_) => {
            manager.finish(
                &session.id,
                LoginStatus::Failed,
                Some("login_upstream_error"),
            );
            return Err(StartError::Upstream);
        }
    };
    let redirect = format!("http://127.0.0.1:{port}/auth/callback");
    let state = secret();
    let verifier = secret();
    let challenge = URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()));
    let mut url = match reqwest::Url::parse(&format!("{}/oauth/authorize", manager.issuer())) {
        Ok(url) => url,
        Err(_) => {
            manager.finish(
                &session.id,
                LoginStatus::Failed,
                Some("login_upstream_error"),
            );
            return Err(StartError::Upstream);
        }
    };
    url.query_pairs_mut().extend_pairs([
        ("response_type", "code"),
        ("client_id", CLIENT_ID),
        ("redirect_uri", &redirect),
        ("state", &state),
        ("code_challenge", &challenge),
        ("code_challenge_method", "S256"),
        (
            "scope",
            "openid profile email offline_access api.connectors.read api.connectors.invoke",
        ),
        ("id_token_add_organizations", "true"),
        ("codex_cli_simplified_flow", "true"),
        ("originator", "codex_cli_rs"),
    ]);
    let task_manager = manager.clone();
    let id = session.id.clone();
    tokio::spawn(async move {
        worker(
            task_manager,
            id,
            listener,
            port,
            state,
            verifier,
            redirect,
            deadline,
            cancel,
        )
        .await
    });
    Ok(BrowserStarted {
        session,
        auth_url: url.into(),
    })
}

enum Callback {
    Code(String),
    Denied,
    Invalid,
}

fn parse_request(bytes: &[u8], port: u16, expected_state: &str) -> Callback {
    let Ok(text) = std::str::from_utf8(bytes) else {
        return Callback::Invalid;
    };
    let mut lines = text.split("\r\n");
    let Some(first) = lines.next() else {
        return Callback::Invalid;
    };
    let mut parts = first.split(' ');
    if parts.next() != Some("GET") {
        return Callback::Invalid;
    }
    let Some(target) = parts.next() else {
        return Callback::Invalid;
    };
    if !matches!(parts.next(), Some("HTTP/1.1") | Some("HTTP/1.0"))
        || parts.next().is_some()
        || target.split('?').next() != Some("/auth/callback")
        || target.contains('#')
    {
        return Callback::Invalid;
    }
    let mut hosts = lines
        .filter_map(|line| line.split_once(':'))
        .filter(|(name, _)| name.eq_ignore_ascii_case("host"))
        .map(|(_, value)| value.trim());
    let expected_host = format!("127.0.0.1:{port}");
    if hosts.next() != Some(expected_host.as_str()) || hosts.next().is_some() {
        return Callback::Invalid;
    }
    let Ok(url) = reqwest::Url::parse(&format!("http://127.0.0.1:{port}{target}")) else {
        return Callback::Invalid;
    };
    if url.path() != "/auth/callback" {
        return Callback::Invalid;
    }
    let (mut state, mut code, mut error) = (None, None, None);
    for (key, value) in url.query_pairs() {
        let slot = match key.as_ref() {
            "state" => &mut state,
            "code" => &mut code,
            "error" => &mut error,
            _ => continue,
        };
        if slot.is_some() {
            return Callback::Invalid;
        }
        *slot = Some(value.into_owned());
    }
    if state.as_deref() != Some(expected_state) || expected_state.is_empty() {
        return Callback::Invalid;
    }
    match (code, error) {
        (Some(code), None) if !code.is_empty() => Callback::Code(code),
        (None, Some(error)) if !error.is_empty() => Callback::Denied,
        _ => Callback::Invalid,
    }
}

async fn read_request(stream: &mut TcpStream) -> Option<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        let n = stream.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        bytes.extend_from_slice(&chunk[..n]);
        if bytes.len() > 8192 {
            return None;
        }
        if bytes.windows(4).any(|w| w == b"\r\n\r\n") {
            return Some(bytes);
        }
    }
}

async fn respond(stream: &mut TcpStream, success: bool) {
    let (status, body) = if success {
        ("200 OK", "Login complete. You may close this window.")
    } else {
        ("400 Bad Request", "Login could not be completed.")
    };
    let response = format!("HTTP/1.1 {status}\r\nContent-Type: text/plain; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nReferrer-Policy: no-referrer\r\nConnection: close\r\n\r\n{body}", body.len());
    let _ = timeout(
        Duration::from_secs(1),
        stream.write_all(response.as_bytes()),
    )
    .await;
}

#[allow(clippy::too_many_arguments)]
async fn worker(
    manager: DeviceLogin,
    id: String,
    listener: TcpListener,
    port: u16,
    state: String,
    verifier: String,
    redirect: String,
    deadline: Instant,
    cancel: Arc<Notify>,
) {
    loop {
        let accepted: Result<(TcpStream, SocketAddr), _> = tokio::select! {
            biased;
            _ = cancel.notified() => return,
            _ = tokio::time::sleep_until(deadline) => { manager.finish(&id, LoginStatus::Expired, Some("login_expired")); return; },
            result = listener.accept() => result,
        };
        let Ok((mut stream, _)) = accepted else {
            manager.finish(&id, LoginStatus::Failed, Some("login_upstream_error"));
            return;
        };
        let request = tokio::select! {
            biased;
            _ = cancel.notified() => return,
            _ = tokio::time::sleep_until(deadline) => { manager.finish(&id, LoginStatus::Expired, Some("login_expired")); return; },
            result = timeout(Duration::from_secs(3), read_request(&mut stream)) => result.ok().flatten(),
        };
        match request
            .as_deref()
            .map(|r| parse_request(r, port, &state))
            .unwrap_or(Callback::Invalid)
        {
            Callback::Invalid => {
                tokio::select! {
                    biased;
                    _ = cancel.notified() => return,
                    _ = tokio::time::sleep_until(deadline) => { manager.finish(&id, LoginStatus::Expired, Some("login_expired")); return; },
                    _ = respond(&mut stream, false) => {},
                }
                continue;
            }
            Callback::Denied => {
                drop(listener);
                manager.finish(&id, LoginStatus::Failed, Some("login_denied"));
                respond(&mut stream, false).await;
                return;
            }
            Callback::Code(code) => {
                // Stop accepting before exchanging, so a code can be consumed only once.
                drop(listener);
                let exchange = async {
                    let response = manager
                        .client()
                        .post(format!("{}/oauth/token", manager.issuer()))
                        .form(&[
                            ("grant_type", "authorization_code"),
                            ("client_id", CLIENT_ID),
                            ("code", &code),
                            ("redirect_uri", &redirect),
                            ("code_verifier", &verifier),
                        ])
                        .send()
                        .await
                        .map_err(|_| "login_upstream_error")?;
                    if !response.status().is_success() {
                        return Err("login_upstream_error");
                    }
                    let tokens = response
                        .json::<Value>()
                        .await
                        .map_err(|_| "invalid_login_response")?;
                    for key in ["access_token", "refresh_token", "id_token"] {
                        if tokens
                            .get(key)
                            .and_then(Value::as_str)
                            .is_none_or(|s| s.trim().is_empty())
                        {
                            return Err("invalid_login_response");
                        }
                    }
                    Ok(tokens)
                };
                let result = tokio::select! {
                    biased;
                    _ = cancel.notified() => return,
                    result = timeout_at(deadline, exchange) => result,
                };
                match result {
                    Err(_) => manager.finish(&id, LoginStatus::Expired, Some("login_expired")),
                    Ok(Err(code)) => manager.finish(&id, LoginStatus::Failed, Some(code)),
                    Ok(Ok(tokens)) => {
                        let m = manager.clone();
                        let worker_id = id.clone();
                        let task = tokio::task::spawn_blocking(move || {
                            m.commit_tokens(worker_id, tokens, deadline)
                        });
                        tokio::select! {
                            result = task => if result.is_err() { manager.finish(&id, LoginStatus::Failed, Some("login_storage_error")); },
                            _ = tokio::time::sleep_until(deadline) => manager.finish(&id, LoginStatus::Expired, Some("login_expired")),
                        }
                    }
                }
                let success = manager
                    .get(&id)
                    .is_some_and(|s| s.status == LoginStatus::Succeeded);
                respond(&mut stream, success).await;
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[tokio::test]
    async fn authorization_fields_and_cancellation_close_listener() {
        let commit = std::sync::Arc::new(
            |_: Value, _: Instant, _: &mut crate::device_login::SessionFence<'_>| Ok(()),
        );
        let manager = DeviceLogin::with_parts(
            reqwest::Client::new(),
            commit,
            "http://127.0.0.1:9999".into(),
            Duration::from_secs(30),
        );
        let started = start_with_ports(&manager, &[0]).await.unwrap();
        let url = reqwest::Url::parse(&started.auth_url).unwrap();
        let fields: std::collections::HashMap<_, _> = url.query_pairs().into_owned().collect();
        assert_eq!(url.path(), "/oauth/authorize");
        assert_eq!(fields["client_id"], CLIENT_ID);
        assert_eq!(fields["code_challenge_method"], "S256");
        assert_eq!(fields["originator"], "codex_cli_rs");
        assert!(fields["state"].len() >= 32);
        let redirect = reqwest::Url::parse(&fields["redirect_uri"]).unwrap();
        let address = format!("127.0.0.1:{}", redirect.port().unwrap());
        assert!(TcpStream::connect(&address).await.is_ok());
        manager.cancel(&started.session.id);
        for _ in 0..50 {
            if TcpStream::connect(&address).await.is_err() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("callback listener remained open after cancellation");
    }
    #[tokio::test]
    async fn real_callback_exchanges_pkce_and_durably_writes_store() {
        use crate::device_login::CommitError;
        let issuer = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let issuer_url = format!("http://{}", issuer.local_addr().unwrap());
        let jwt = format!(
            "h.{}.s",
            URL_SAFE_NO_PAD
                .encode(br#"{"https://api.openai.com/auth":{"chatgpt_account_id":"account"}}"#)
        );
        let body = serde_json::json!({"access_token":jwt,"id_token":jwt,"refresh_token":"refresh"})
            .to_string();
        let (sender, receiver) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let (mut stream, _) = issuer.accept().await.unwrap();
            let mut bytes = Vec::new();
            loop {
                let mut buf = [0; 4096];
                let n = stream.read(&mut buf).await.unwrap();
                if n == 0 {
                    break;
                }
                bytes.extend_from_slice(&buf[..n]);
                if let Some(end) = bytes.windows(4).position(|w| w == b"\r\n\r\n") {
                    let headers = String::from_utf8_lossy(&bytes[..end]);
                    let len = headers
                        .lines()
                        .find_map(|l| {
                            l.to_ascii_lowercase()
                                .strip_prefix("content-length:")
                                .and_then(|v| v.trim().parse::<usize>().ok())
                        })
                        .unwrap();
                    if bytes.len() >= end + 4 + len {
                        break;
                    }
                }
            }
            sender.send(String::from_utf8(bytes).unwrap()).unwrap();
            let response = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", body.len(), body);
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        let dir = std::env::temp_dir().join(format!("browser-login-{}", Uuid::new_v4()));
        let path = dir.join("auth.json");
        let path_clone = path.clone();
        let commit = Arc::new(
            move |tokens: Value,
                  deadline: Instant,
                  fence: &mut crate::device_login::SessionFence<'_>| {
                crate::auth::with_login_store_lock(&path_clone, |store| {
                    fence(&mut || {
                        store
                            .commit_before(&tokens, deadline)
                            .map_err(|_| CommitError::Storage)
                    })
                    .map_err(|_| crate::auth::AuthError::Internal("store failure".into()))
                })
                .map_err(|_| CommitError::Storage)
            },
        );
        let manager = DeviceLogin::with_parts(
            reqwest::Client::new(),
            commit,
            issuer_url,
            Duration::from_secs(30),
        );
        let started = start_with_ports(&manager, &[0]).await.unwrap();
        let url = reqwest::Url::parse(&started.auth_url).unwrap();
        let fields: std::collections::HashMap<_, _> = url.query_pairs().into_owned().collect();
        let redirect = reqwest::Url::parse(&fields["redirect_uri"]).unwrap();
        let callback = format!("{}?state={}&code=test-code", redirect, fields["state"]);
        let response = reqwest::get(callback).await.unwrap();
        assert_eq!(response.status(), 200);
        assert_eq!(
            manager.get(&started.session.id).unwrap().status,
            LoginStatus::Succeeded
        );
        assert_eq!(
            crate::auth::local_auth_status(path.to_str()),
            "authenticated"
        );
        let request = receiver.await.unwrap();
        let form = request.split_once("\r\n\r\n").unwrap().1;
        let params: std::collections::HashMap<_, _> =
            reqwest::Url::parse(&format!("http://localhost/?{form}"))
                .unwrap()
                .query_pairs()
                .into_owned()
                .collect();
        assert_eq!(params["code"], "test-code");
        assert_eq!(params["redirect_uri"], fields["redirect_uri"]);
        assert_eq!(
            URL_SAFE_NO_PAD.encode(Sha256::digest(params["code_verifier"].as_bytes())),
            fields["code_challenge"]
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
    #[tokio::test]
    async fn busy_ports_fall_back_and_report_exhaustion() {
        let first = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let second = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let first_port = first.local_addr().unwrap().port();
        let second_port = second.local_addr().unwrap().port();
        drop(second);
        let commit =
            Arc::new(|_: Value, _: Instant, _: &mut crate::device_login::SessionFence<'_>| Ok(()));
        let manager = DeviceLogin::with_parts(
            reqwest::Client::new(),
            commit,
            "http://127.0.0.1:9999".into(),
            Duration::from_secs(30),
        );
        let started = start_with_ports(&manager, &[first_port, second_port])
            .await
            .unwrap();
        let url = reqwest::Url::parse(&started.auth_url).unwrap();
        let redirect = url
            .query_pairs()
            .find(|(k, _)| k == "redirect_uri")
            .unwrap()
            .1;
        assert_eq!(
            reqwest::Url::parse(&redirect).unwrap().port(),
            Some(second_port)
        );
        manager.cancel(&started.session.id);
        // Keep a separately owned second port occupied after the worker closes.
        for _ in 0..50 {
            if let Ok(bound) = TcpListener::bind((std::net::Ipv4Addr::LOCALHOST, second_port)).await
            {
                assert!(matches!(
                    start_with_ports(&manager, &[first_port, second_port]).await,
                    Err(StartError::CallbackPortUnavailable)
                ));
                drop(bound);
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        panic!("callback listener did not close");
    }
    #[tokio::test]
    async fn cancel_during_exchange_prevents_late_commit() {
        let issuer = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let issuer_url = format!("http://{}", issuer.local_addr().unwrap());
        let (sent, received) = tokio::sync::oneshot::channel();
        let (release, wait) = tokio::sync::oneshot::channel();
        tokio::spawn(async move {
            let (mut stream, _) = issuer.accept().await.unwrap();
            let mut buf = [0; 4096];
            let _ = stream.read(&mut buf).await.unwrap();
            sent.send(()).unwrap();
            let _ = wait.await;
            let body = r#"{"access_token":"a","refresh_token":"r","id_token":"i"}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(response.as_bytes()).await;
        });
        let writes = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let count = writes.clone();
        let commit = Arc::new(
            move |_: Value, _: Instant, _: &mut crate::device_login::SessionFence<'_>| {
                count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            },
        );
        let manager = DeviceLogin::with_parts(
            reqwest::Client::new(),
            commit,
            issuer_url,
            Duration::from_secs(30),
        );
        let started = start_with_ports(&manager, &[0]).await.unwrap();
        let url = reqwest::Url::parse(&started.auth_url).unwrap();
        let fields: std::collections::HashMap<_, _> = url.query_pairs().into_owned().collect();
        let callback = format!(
            "{}?state={}&code=test-code",
            fields["redirect_uri"], fields["state"]
        );
        let task = tokio::spawn(async move {
            let _ = reqwest::get(callback).await;
        });
        received.await.unwrap();
        assert_eq!(
            manager.cancel(&started.session.id).unwrap().status,
            LoginStatus::Cancelled
        );
        let _ = release.send(());
        task.await.unwrap();
        assert_eq!(writes.load(std::sync::atomic::Ordering::SeqCst), 0);
        assert_eq!(
            manager.get(&started.session.id).unwrap().status,
            LoginStatus::Cancelled
        );
    }
    #[test]
    fn callback_validation() {
        let request = |target: &str, host: &str| {
            format!("GET {target} HTTP/1.1\r\nHost: {host}\r\n\r\n").into_bytes()
        };
        assert!(
            matches!(parse_request(&request("/auth/callback?state=s&code=c", "127.0.0.1:1455"), 1455, "s"), Callback::Code(c) if c == "c")
        );
        for target in [
            "/other?state=s&code=c",
            "/wrong/../auth/callback?state=s&code=c",
            "/auth/callback?state=x&code=c",
            "/auth/callback?state=s&state=s&code=c",
            "/auth/callback?state=s&code=c&code=d",
            "/auth/callback?state=s&code=c&error=e",
        ] {
            assert!(matches!(
                parse_request(&request(target, "127.0.0.1:1455"), 1455, "s"),
                Callback::Invalid
            ));
        }
        assert!(matches!(
            parse_request(
                &request("/auth/callback?state=s&error=denied", "127.0.0.1:1455"),
                1455,
                "s"
            ),
            Callback::Denied
        ));
        assert!(matches!(
            parse_request(
                &request("/auth/callback?state=s&code=c", "localhost:1455"),
                1455,
                "s"
            ),
            Callback::Invalid
        ));
    }
}
