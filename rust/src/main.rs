mod anthropic_adapter;
mod auth;
mod codex_config;
mod messages;
mod model_capabilities;
mod protocol;
mod provider;
mod server;

use provider::ChatGPTOAuthProvider;
use server::{create_router, AppState};
use std::sync::Arc;

fn env_str(name: &str, default: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| default.to_string())
}

fn env_int(name: &str, default: u16) -> u16 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[tokio::main]
async fn main() {
    provider::prime_codex_cli_version_cache();

    let host = env_str("CODEX_AS_API_HOST", "127.0.0.1");
    let port = env_int("CODEX_AS_API_PORT", 18080);
    let config = codex_config::load_codex_config(None);
    let model = std::env::var("CODEX_AS_API_MODEL")
        .ok()
        .or_else(|| config.model.clone())
        .unwrap_or_else(|| "gpt-5.5".to_string());
    let auth_path = std::env::var("CODEX_AS_API_AUTH_PATH").ok();

    let provider = ChatGPTOAuthProvider::new(
        model.clone(),
        provider::CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
        auth_path.clone(),
        None,
    );

    let state = AppState {
        model,
        auth_path,
        codex_config: config,
        provider: Arc::new(provider),
    };

    let app = create_router(state);
    let addr = format!("{}:{}", host, port);
    eprintln!("codex-as-api listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
