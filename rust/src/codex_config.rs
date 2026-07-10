use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct CodexConfig {
    pub codex_home: String,
    pub config_path: String,
    pub model: Option<String>,
    pub model_reasoning_effort: Option<String>,
    pub model_context_window: Option<i64>,
    pub model_auto_compact_token_limit: Option<i64>,
}

pub fn resolve_codex_home(raw: Option<&str>) -> String {
    expand_home(
        nonempty_trimmed(raw.map(|s| s.to_string()))
            .or_else(|| nonempty_trimmed(std::env::var("CODEX_HOME").ok()))
            .unwrap_or_else(|| {
                let mut home = home_dir();
                home.push(".codex");
                home.to_string_lossy().to_string()
            })
            .as_str(),
    )
}

fn nonempty_trimmed(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    })
}

pub fn load_codex_config(raw_codex_home: Option<&str>) -> Result<CodexConfig, String> {
    let codex_home = resolve_codex_home(raw_codex_home);
    let config_path = PathBuf::from(&codex_home).join("config.toml");
    let config_path_string = config_path.to_string_lossy().to_string();
    let text = match std::fs::read_to_string(&config_path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CodexConfig {
                codex_home,
                config_path: config_path_string,
                model: None,
                model_reasoning_effort: None,
                model_context_window: None,
                model_auto_compact_token_limit: None,
            });
        }
        Err(error) => {
            return Err(format!(
                "failed to read Codex config {}: {error}",
                config_path.display()
            ));
        }
    };

    config_from_toml(&text, codex_home, config_path_string)
}

fn config_from_toml(
    text: &str,
    codex_home: String,
    config_path: String,
) -> Result<CodexConfig, String> {
    let root = text
        .parse::<toml::Table>()
        .map_err(|error| format!("invalid Codex config {config_path}: {error}"))?;

    Ok(CodexConfig {
        codex_home,
        config_path,
        model: optional_model_string(&root)?,
        model_reasoning_effort: optional_nonempty_string(&root, "model_reasoning_effort")?,
        model_context_window: optional_positive_integer(&root, "model_context_window")?,
        model_auto_compact_token_limit: optional_positive_integer(
            &root,
            "model_auto_compact_token_limit",
        )?,
    })
}

fn expand_home(path: &str) -> String {
    if path == "~" {
        return home_dir().to_string_lossy().to_string();
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return home_dir().join(rest).to_string_lossy().to_string();
    }
    path.to_string()
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

fn optional_string(root: &toml::Table, key: &str) -> Result<Option<String>, String> {
    match root.get(key) {
        None => Ok(None),
        Some(toml::Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(format!("Codex config field {key} must be a string")),
    }
}

fn optional_model_string(root: &toml::Table) -> Result<Option<String>, String> {
    Ok(nonempty_trimmed(optional_string(root, "model")?))
}

fn optional_nonempty_string(root: &toml::Table, key: &str) -> Result<Option<String>, String> {
    let value = optional_string(root, key)?;
    if value.as_deref() == Some("") {
        return Err(format!("Codex config field {key} must be non-empty"));
    }
    Ok(value)
}

fn optional_positive_integer(root: &toml::Table, key: &str) -> Result<Option<i64>, String> {
    match root.get(key) {
        None => Ok(None),
        Some(toml::Value::Integer(value)) if *value > 0 => Ok(Some(*value)),
        Some(toml::Value::Integer(_)) => Err(format!("Codex config field {key} must be positive")),
        Some(_) => Err(format!("Codex config field {key} must be an integer")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_root_config_values_and_toml_escapes() {
        let text = r#"
model = "gpt-5.5-\u0063odex"
model_reasoning_effort = "future\tdeep#mode"
model_context_window = 200_000
model_auto_compact_token_limit = 160000 # comment
"#;
        let config =
            config_from_toml(text, "/tmp/codex".into(), "/tmp/config.toml".into()).unwrap();
        assert_eq!(config.model, Some("gpt-5.5-codex".to_string()));
        assert_eq!(
            config.model_reasoning_effort,
            Some("future\tdeep#mode".to_string())
        );
        assert_eq!(config.model_context_window, Some(200000));
        assert_eq!(config.model_auto_compact_token_limit, Some(160000));
    }

    #[test]
    fn inactive_profile_does_not_override_root_values() {
        let text = r#"
model = "gpt-5.6-sol"
model_reasoning_effort = "future#deep"

[profiles.expensive]
model_reasoning_effort = "ultra"
"#;
        let config =
            config_from_toml(text, "/tmp/codex".into(), "/tmp/config.toml".into()).unwrap();

        assert_eq!(
            config.model_reasoning_effort,
            Some("future#deep".to_string())
        );
        assert_eq!(config.model, Some("gpt-5.6-sol".to_string()));
    }

    #[test]
    fn wrong_types_and_nonpositive_context_fail_loudly() {
        for text in [
            "model_reasoning_effort = 42",
            "model_reasoning_effort = \"\"",
            "model = true",
            "model_context_window = \"large\"",
            "model_auto_compact_token_limit = 0",
        ] {
            assert!(
                config_from_toml(text, "/tmp/codex".into(), "/tmp/config.toml".into()).is_err()
            );
        }
    }

    #[test]
    fn empty_codex_home_value_is_absent() {
        assert_eq!(nonempty_trimmed(Some(String::new())), None);
        assert_eq!(nonempty_trimmed(Some("  ".to_string())), None);
        assert_eq!(
            nonempty_trimmed(Some(" ~/.codex-test ".to_string())),
            Some("~/.codex-test".to_string())
        );
    }

    #[test]
    fn empty_model_value_is_absent() {
        for text in ["model = \"\"", "model = \"   \""] {
            let config =
                config_from_toml(text, "/tmp/codex".into(), "/tmp/config.toml".into()).unwrap();
            assert_eq!(config.model, None);
        }
    }
}
