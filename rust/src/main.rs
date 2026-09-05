mod anthropic_adapter;
mod auth;
mod codex_config;
mod messages;
mod model_capabilities;
mod model_catalog;
mod o200k_tokenizer;
mod protocol;
mod provider;
mod server;
mod strict_json;

use provider::ChatGPTOAuthProvider;
use server::{create_router, AppState};
use std::sync::Arc;

fn optional_env(name: &str) -> Result<Option<String>, String> {
    match std::env::var(name) {
        Ok(value) if value.trim().is_empty() => {
            Err(format!("{name} must not be empty when provided"))
        }
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!("{name} must contain valid Unicode")),
    }
}

fn optional_identifier_env(name: &str) -> Result<Option<String>, String> {
    let value = optional_env(name)?;
    if value.as_deref().is_some_and(|value| value != value.trim()) {
        return Err(format!("{name} must not contain surrounding whitespace"));
    }
    Ok(value)
}

fn env_u16(name: &str, default: u16) -> Result<u16, String> {
    optional_env(name)?
        .map(|value| {
            if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                return Err(format!("{name} must be an integer from 1 to 65535"));
            }
            let parsed = value
                .parse::<u16>()
                .map_err(|_| format!("{name} must be an integer from 1 to 65535"))?;
            if parsed == 0 {
                return Err(format!("{name} must be an integer from 1 to 65535"));
            }
            Ok(parsed)
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

fn startup_value<T>(result: Result<T, String>) -> T {
    match result {
        Ok(value) => value,
        Err(message) => {
            eprintln!("{message}");
            std::process::exit(2);
        }
    }
}

fn reject_deprecated_environment(name: &str) -> Result<(), String> {
    match std::env::var_os(name) {
        None => Ok(()),
        Some(_) => Err(format!(
            "{name} is no longer supported; the Codex compatibility version is pinned by config/codex-upstream-contract.json"
        )),
    }
}

#[tokio::main]
async fn main() {
    let host =
        startup_value(optional_env("CODEX_AS_API_HOST")).unwrap_or_else(|| "127.0.0.1".to_string());
    let port = startup_value(env_u16("CODEX_AS_API_PORT", 18080));
    let config = codex_config::load_codex_config(None).unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(2);
    });
    let model = startup_value(optional_identifier_env("CODEX_AS_API_MODEL"))
        .or_else(|| config.model.clone())
        .unwrap_or_default();
    let auth_path = startup_value(optional_env("CODEX_AS_API_AUTH_PATH"));
    startup_value(reject_deprecated_environment(
        "CODEX_AS_API_CODEX_CLI_VERSION",
    ));
    startup_value(auth::validate_auth_environment());
    startup_value(model_capabilities::validate_model_capability_environment());

    let provider = startup_value(
        ChatGPTOAuthProvider::new(
            model.clone(),
            provider::CHATGPT_OAUTH_DEFAULT_BASE_URL.to_string(),
            auth_path.clone(),
            None,
        )
        .map_err(|error| error.to_string()),
    );

    let state = AppState {
        auth_path,
        codex_config: config,
        provider: Arc::new(provider),
    };

    let app = create_router(state);
    let addr = format!("{}:{}", host, port);
    eprintln!("codex-as-api listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|error| startup_value(Err(format!("failed to bind {addr}: {error}"))));
    axum::serve(listener, app)
        .await
        .unwrap_or_else(|error| startup_value(Err(format!("server failed: {error}"))));
}

#[cfg(test)]
mod tests {
    use super::{env_u16, optional_env};

    #[test]
    fn explicit_empty_environment_values_are_rejected() {
        let name = format!("CODEX_AS_API_TEST_EMPTY_{}", uuid::Uuid::new_v4());
        std::env::set_var(&name, "   ");
        let result = optional_env(&name);
        std::env::remove_var(&name);
        assert!(result.is_err());
    }

    #[test]
    fn nonempty_environment_values_are_preserved_verbatim() {
        let name = format!("CODEX_AS_API_TEST_PRESERVE_{}", uuid::Uuid::new_v4());
        std::env::set_var(&name, "  path with spaces  ");
        let result = optional_env(&name);
        std::env::remove_var(&name);
        assert_eq!(result.unwrap().as_deref(), Some("  path with spaces  "));
    }

    #[test]
    fn port_environment_requires_ascii_decimal_digits_and_valid_range() {
        for value in [" 80", "+80", "1_000", "１２", "0", "65536"] {
            let name = format!("CODEX_AS_API_TEST_PORT_{}", uuid::Uuid::new_v4());
            std::env::set_var(&name, value);
            let result = env_u16(&name, 18080);
            std::env::remove_var(&name);
            assert!(result.is_err(), "{value}");
        }
    }
}
