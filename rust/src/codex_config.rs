use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct CodexConfig {
    pub codex_home: String,
    pub config_path: String,
    pub model: Option<String>,
    pub model_context_window: Option<i64>,
    pub model_auto_compact_token_limit: Option<i64>,
}

pub fn resolve_codex_home(raw: Option<&str>) -> String {
    expand_home(
        raw.map(|s| s.to_string())
            .or_else(|| std::env::var("CODEX_HOME").ok())
            .unwrap_or_else(|| {
                let mut home = home_dir();
                home.push(".codex");
                home.to_string_lossy().to_string()
            })
            .as_str(),
    )
}

pub fn load_codex_config(raw_codex_home: Option<&str>) -> CodexConfig {
    let codex_home = resolve_codex_home(raw_codex_home);
    let config_path = PathBuf::from(&codex_home).join("config.toml");
    let config_path_string = config_path.to_string_lossy().to_string();
    let text = match std::fs::read_to_string(&config_path) {
        Ok(text) => text,
        Err(_) => {
            return CodexConfig {
                codex_home,
                config_path: config_path_string,
                model: None,
                model_context_window: None,
                model_auto_compact_token_limit: None,
            };
        }
    };

    CodexConfig {
        codex_home,
        config_path: config_path_string,
        model: parse_toml_string(&text, "model"),
        model_context_window: parse_toml_integer(&text, "model_context_window"),
        model_auto_compact_token_limit: parse_toml_integer(&text, "model_auto_compact_token_limit"),
    }
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

fn parse_toml_string(text: &str, key: &str) -> Option<String> {
    for line in text.lines() {
        let line = strip_comment(line).trim();
        let Some((k, raw_value)) = line.split_once('=') else {
            continue;
        };
        if k.trim() != key {
            continue;
        }
        let value = raw_value.trim();
        if value.len() >= 2 {
            let first = value.as_bytes()[0] as char;
            let last = value.as_bytes()[value.len() - 1] as char;
            if (first == '"' && last == '"') || (first == '\'' && last == '\'') {
                return Some(value[1..value.len() - 1].to_string());
            }
        }
    }
    None
}

fn parse_toml_integer(text: &str, key: &str) -> Option<i64> {
    for line in text.lines() {
        let line = strip_comment(line).trim();
        let Some((k, raw_value)) = line.split_once('=') else {
            continue;
        };
        if k.trim() != key {
            continue;
        }
        let value = raw_value.trim().replace('_', "");
        if let Ok(parsed) = value.parse::<i64>() {
            if parsed > 0 {
                return Some(parsed);
            }
        }
    }
    None
}

fn strip_comment(line: &str) -> &str {
    line.split('#').next().unwrap_or(line)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_root_config_values() {
        let text = r#"
model = "gpt-5.5-codex"
model_context_window = 200_000
model_auto_compact_token_limit = 160000 # comment
"#;
        assert_eq!(
            parse_toml_string(text, "model"),
            Some("gpt-5.5-codex".to_string())
        );
        assert_eq!(
            parse_toml_integer(text, "model_context_window"),
            Some(200000)
        );
        assert_eq!(
            parse_toml_integer(text, "model_auto_compact_token_limit"),
            Some(160000)
        );
    }
}
