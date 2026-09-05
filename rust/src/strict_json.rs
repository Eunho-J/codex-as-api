use serde_json::Value;

pub const JS_SAFE_INTEGER: u64 = 9_007_199_254_740_991;

pub fn as_js_safe_integer(number: &serde_json::Number) -> Option<i64> {
    if let Some(value) = number.as_i64() {
        return (value.unsigned_abs() <= JS_SAFE_INTEGER).then_some(value);
    }
    if let Some(value) = number.as_u64() {
        return (value <= JS_SAFE_INTEGER).then_some(value as i64);
    }
    let value = number.as_f64()?;
    (value.is_finite() && value.fract() == 0.0 && value.abs() <= JS_SAFE_INTEGER as f64)
        .then_some(value as i64)
}

pub fn parse_slice(input: &[u8]) -> Result<Value, String> {
    let value: Value = serde_json::from_slice(input).map_err(|error| error.to_string())?;
    validate_value(&value)?;
    Ok(value)
}

pub fn parse_str(input: &str) -> Result<Value, String> {
    let value: Value = serde_json::from_str(input).map_err(|error| error.to_string())?;
    validate_value(&value)?;
    Ok(value)
}

pub fn validate_value(value: &Value) -> Result<(), String> {
    match value {
        Value::Array(values) => {
            for value in values {
                validate_value(value)?;
            }
        }
        Value::Object(values) => {
            for value in values.values() {
                validate_value(value)?;
            }
        }
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                if value.unsigned_abs() > JS_SAFE_INTEGER {
                    return Err("JSON integers must be JavaScript-safe".to_string());
                }
            } else if let Some(value) = number.as_u64() {
                if value > JS_SAFE_INTEGER {
                    return Err("JSON integers must be JavaScript-safe".to_string());
                }
            } else if let Some(value) = number.as_f64() {
                if !value.is_finite()
                    || (value.fract() == 0.0 && value.abs() > JS_SAFE_INTEGER as f64)
                {
                    return Err("JSON numbers must be finite JavaScript numbers".to_string());
                }
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unsafe_integer_forms_recursively() {
        for document in [
            r#"{"value":9007199254740992}"#,
            r#"{"value":-9007199254740992}"#,
            r#"{"value":9007199254740992.0}"#,
            r#"{"value":9007199254740992.5}"#,
            r#"{"nested":[1e16]}"#,
        ] {
            assert!(parse_str(document).is_err(), "{document}");
        }
    }

    #[test]
    fn accepts_safe_integer_boundaries() {
        for document in [
            r#"{"value":9007199254740991}"#,
            r#"{"value":-9007199254740991}"#,
        ] {
            assert!(parse_str(document).is_ok(), "{document}");
        }
    }
}
