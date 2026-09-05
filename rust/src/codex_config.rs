use std::path::PathBuf;

const JS_SAFE_INTEGER: i64 = 9_007_199_254_740_991;

#[derive(Clone, Debug)]
pub struct CodexConfig {
    pub model: Option<String>,
    pub model_reasoning_effort: Option<String>,
    pub model_context_window: Option<i64>,
    pub model_auto_compact_token_limit: Option<i64>,
}

pub fn resolve_codex_home(raw: Option<&str>) -> Result<String, String> {
    let configured = match raw {
        Some(value) => Some(
            nonempty(Some(value.to_string()))
                .ok_or_else(|| "Codex home path must not be empty when provided".to_string())?,
        ),
        None => match std::env::var("CODEX_HOME") {
            Ok(value) => Some(
                nonempty(Some(value))
                    .ok_or_else(|| "CODEX_HOME must not be empty when provided".to_string())?,
            ),
            Err(std::env::VarError::NotPresent) => None,
            Err(std::env::VarError::NotUnicode(_)) => {
                return Err("CODEX_HOME must contain valid Unicode".to_string());
            }
        },
    };
    match configured {
        Some(path) => expand_home(&path),
        None => path_to_string(home_dir()?.join(".codex"), "default Codex home"),
    }
}

fn nonempty(value: Option<String>) -> Option<String> {
    value.and_then(|value| (!value.trim().is_empty()).then_some(value))
}

pub fn load_codex_config(raw_codex_home: Option<&str>) -> Result<CodexConfig, String> {
    let codex_home = resolve_codex_home(raw_codex_home)?;
    let config_path = PathBuf::from(&codex_home).join("config.toml");
    let config_path_string = path_to_string(config_path.clone(), "Codex config path")?;
    let text = match std::fs::read_to_string(&config_path) {
        Ok(text) => text,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(CodexConfig {
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

    config_from_toml(&text, config_path_string)
}

fn config_from_toml(text: &str, config_path: String) -> Result<CodexConfig, String> {
    let root = text
        .parse::<toml::Table>()
        .map_err(|error| format!("invalid Codex config {config_path}: {error}"))?;

    Ok(CodexConfig {
        model: optional_model_string(&root)?,
        model_reasoning_effort: optional_nonempty_string(&root, "model_reasoning_effort")?,
        model_context_window: optional_positive_integer(&root, "model_context_window")?,
        model_auto_compact_token_limit: optional_signed_integer(
            &root,
            "model_auto_compact_token_limit",
        )?,
    })
}

fn expand_home(path: &str) -> Result<String, String> {
    if path == "~" {
        return path_to_string(home_dir()?, "HOME");
    }
    if let Some(rest) = path.strip_prefix("~/") {
        return path_to_string(home_dir()?.join(rest), "expanded Codex home");
    }
    Ok(path.to_string())
}

#[cfg(unix)]
fn home_dir() -> Result<PathBuf, String> {
    unix_home_dir(std::env::var("HOME"))
}

#[cfg(windows)]
fn home_dir() -> Result<PathBuf, String> {
    windows_home_dir(
        std::env::var("USERPROFILE"),
        std::env::var("HOMEDRIVE"),
        std::env::var("HOMEPATH"),
    )
}

fn env_home_component(
    value: Result<String, std::env::VarError>,
    name: &str,
) -> Result<Option<String>, String> {
    match value {
        Ok(value) if value.trim().is_empty() => Err(format!("{name} must not be empty")),
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(format!("{name} must contain valid Unicode")),
    }
}

#[cfg(any(unix, test))]
fn unix_home_dir(home: Result<String, std::env::VarError>) -> Result<PathBuf, String> {
    env_home_component(home, "HOME")?
        .map(PathBuf::from)
        .ok_or_else(|| "HOME is required".to_string())
}

#[cfg(any(windows, test))]
fn windows_home_dir(
    user_profile: Result<String, std::env::VarError>,
    home_drive: Result<String, std::env::VarError>,
    home_path: Result<String, std::env::VarError>,
) -> Result<PathBuf, String> {
    if let Some(profile) = env_home_component(user_profile, "USERPROFILE")? {
        return Ok(PathBuf::from(profile));
    }
    let drive = env_home_component(home_drive, "HOMEDRIVE")?
        .ok_or_else(|| "USERPROFILE or HOMEDRIVE and HOMEPATH are required".to_string())?;
    let path = env_home_component(home_path, "HOMEPATH")?
        .ok_or_else(|| "USERPROFILE or HOMEDRIVE and HOMEPATH are required".to_string())?;
    Ok(PathBuf::from(format!("{drive}{path}")))
}

fn path_to_string(path: PathBuf, label: &str) -> Result<String, String> {
    path.into_os_string()
        .into_string()
        .map_err(|_| format!("{label} must contain valid Unicode"))
}

fn optional_string(root: &toml::Table, key: &str) -> Result<Option<String>, String> {
    match root.get(key) {
        None => Ok(None),
        Some(toml::Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(format!("Codex config field {key} must be a string")),
    }
}

fn optional_model_string(root: &toml::Table) -> Result<Option<String>, String> {
    match optional_string(root, "model")? {
        None => Ok(None),
        Some(value) if value.trim().is_empty() => {
            Err("Codex config field model must be non-empty when provided".to_string())
        }
        Some(value) if value != value.trim() => {
            Err("Codex config field model must not contain surrounding whitespace".to_string())
        }
        Some(value) => Ok(Some(value)),
    }
}

fn optional_nonempty_string(root: &toml::Table, key: &str) -> Result<Option<String>, String> {
    let value = optional_string(root, key)?;
    if value
        .as_deref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(format!(
            "Codex config field {key} must be non-empty when provided"
        ));
    }
    if value.as_deref().is_some_and(|value| value != value.trim()) {
        return Err(format!(
            "Codex config field {key} must not contain surrounding whitespace"
        ));
    }
    Ok(value)
}

fn optional_positive_integer(root: &toml::Table, key: &str) -> Result<Option<i64>, String> {
    match root.get(key) {
        None => Ok(None),
        Some(toml::Value::Integer(value)) if *value > JS_SAFE_INTEGER => Err(format!(
            "Codex config field {key} must be a JavaScript-safe integer"
        )),
        Some(toml::Value::Integer(value)) if *value > 0 => Ok(Some(*value)),
        Some(toml::Value::Integer(_)) => Err(format!("Codex config field {key} must be positive")),
        Some(_) => Err(format!("Codex config field {key} must be an integer")),
    }
}

fn optional_signed_integer(root: &toml::Table, key: &str) -> Result<Option<i64>, String> {
    match root.get(key) {
        None => Ok(None),
        Some(toml::Value::Integer(value))
            if *value > JS_SAFE_INTEGER || *value < -JS_SAFE_INTEGER =>
        {
            Err(format!(
                "Codex config field {key} must be a JavaScript-safe integer"
            ))
        }
        Some(toml::Value::Integer(value)) => Ok(Some(*value)),
        Some(_) => Err(format!("Codex config field {key} must be an integer")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_root_config_values_and_toml_escapes() {
        let text = r#"
model = "live-\u0063atalog-model"
model_reasoning_effort = "future\tdeep#mode"
model_context_window = 123_456
model_auto_compact_token_limit = 160000 # comment
"#;
        let config = config_from_toml(text, "/tmp/config.toml".into()).unwrap();
        assert_eq!(config.model, Some("live-catalog-model".to_string()));
        assert_eq!(
            config.model_reasoning_effort,
            Some("future\tdeep#mode".to_string())
        );
        assert_eq!(config.model_context_window, Some(123456));
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
        let config = config_from_toml(text, "/tmp/config.toml".into()).unwrap();

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
        ] {
            assert!(config_from_toml(text, "/tmp/config.toml".into()).is_err());
        }
    }

    #[test]
    fn context_limits_must_be_positive_javascript_safe_integers() {
        for field in ["model_context_window", "model_auto_compact_token_limit"] {
            let maximum = config_from_toml(
                &format!("{field} = 9007199254740991"),
                "/tmp/config.toml".into(),
            )
            .unwrap();
            assert_eq!(
                if field == "model_context_window" {
                    maximum.model_context_window
                } else {
                    maximum.model_auto_compact_token_limit
                },
                Some(JS_SAFE_INTEGER)
            );

            let error = config_from_toml(
                &format!("{field} = 9007199254740992"),
                "/tmp/config.toml".into(),
            )
            .unwrap_err();
            assert!(error.contains("JavaScript-safe integer"));
        }
    }

    #[test]
    fn zero_auto_compact_limit_is_preserved() {
        let config = config_from_toml(
            "model_auto_compact_token_limit = 0",
            "/tmp/config.toml".into(),
        )
        .unwrap();
        assert_eq!(config.model_auto_compact_token_limit, Some(0));
        let negative = config_from_toml(
            "model_auto_compact_token_limit = -1",
            "/tmp/config.toml".into(),
        )
        .unwrap();
        assert_eq!(negative.model_auto_compact_token_limit, Some(-1));
    }

    #[test]
    fn empty_explicit_codex_home_is_rejected() {
        assert!(resolve_codex_home(Some("")).is_err());
        assert!(resolve_codex_home(Some("  ")).is_err());
    }

    #[test]
    fn empty_model_value_is_rejected() {
        for text in ["model = \"\"", "model = \"   \""] {
            assert!(config_from_toml(text, "/tmp/config.toml".into()).is_err());
        }
    }

    #[test]
    fn rejects_model_reasoning_whitespace_but_preserves_codex_home_whitespace() {
        assert!(config_from_toml(
            "model = \"  live-model  \"\nmodel_reasoning_effort = \" high \"",
            "/tmp/config.toml".into(),
        )
        .is_err());
        assert_eq!(
            resolve_codex_home(Some(" /tmp/codex home ")).unwrap(),
            " /tmp/codex home "
        );
    }

    #[test]
    fn platform_home_helpers_resolve_without_fallbacks() {
        assert_eq!(
            unix_home_dir(Ok(" /home/user ".to_string())).unwrap(),
            PathBuf::from(" /home/user ")
        );
        assert!(unix_home_dir(Err(std::env::VarError::NotPresent)).is_err());

        assert_eq!(
            windows_home_dir(
                Ok(r"C:\Users\user".to_string()),
                Err(std::env::VarError::NotPresent),
                Err(std::env::VarError::NotPresent),
            )
            .unwrap(),
            PathBuf::from(r"C:\Users\user")
        );
        assert_eq!(
            windows_home_dir(
                Err(std::env::VarError::NotPresent),
                Ok("D:".to_string()),
                Ok(r"\Profiles\user".to_string()),
            )
            .unwrap(),
            PathBuf::from(r"D:\Profiles\user")
        );
        assert!(windows_home_dir(
            Err(std::env::VarError::NotPresent),
            Ok("D:".to_string()),
            Err(std::env::VarError::NotPresent),
        )
        .is_err());
    }
}
