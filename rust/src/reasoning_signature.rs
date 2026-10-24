use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::de::{Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::Serialize;
use serde_json::{Map, Value};
use std::fmt;

use crate::strict_json;

pub const REASONING_SIGNATURE_PREFIX: &str = "codex-reasoning-v1:";
const REASONING_SIGNATURE_RECOGNITION_PREFIX: &str = "codex-reasoning";
const REASONING_SIGNATURE_UNDERSCORE_PREFIX: &str = "codex_reasoning";

pub const REASONING_COMPATIBLE_MODELS: [&str; 5] = [
    "gpt-6.1-sol",
    "gpt-6-astra",
    "gpt-6-sol",
    "gpt-6-luna",
    "gpt-daybreak-blue-latest",
];

pub fn is_reasoning_model_compatible(source_model: &str, target_model: &str) -> bool {
    source_model == target_model
        || (REASONING_COMPATIBLE_MODELS.contains(&source_model)
            && REASONING_COMPATIBLE_MODELS.contains(&target_model))
}

fn has_recognized_codex_reasoning_prefix(signature: &str) -> bool {
    let trimmed = signature.trim_start();
    trimmed
        .get(..REASONING_SIGNATURE_RECOGNITION_PREFIX.len())
        .is_some_and(|prefix| {
            prefix.eq_ignore_ascii_case(REASONING_SIGNATURE_RECOGNITION_PREFIX)
                || prefix.eq_ignore_ascii_case(REASONING_SIGNATURE_UNDERSCORE_PREFIX)
        })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReasoningSummaryEntry {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReasoningSignatureItem {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub encrypted_content: String,
    pub summary: Vec<ReasoningSummaryEntry>,
}

impl ReasoningSignatureItem {
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("ReasoningSignatureItem serializes to Value")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReasoningSignatureEnvelope {
    pub v: u8,
    pub model: String,
    pub item: ReasoningSignatureItem,
}

impl ReasoningSignatureEnvelope {
    pub fn to_value(&self) -> Value {
        serde_json::to_value(self).expect("ReasoningSignatureEnvelope serializes to Value")
    }
}

pub fn encode_reasoning_signature(model: &str, item: &Value) -> Result<Option<String>, String> {
    let object = item
        .as_object()
        .ok_or_else(|| "reasoning item must be an object".to_string())?;
    if object.get("type").and_then(Value::as_str) != Some("reasoning") {
        return Err("reasoning item type must be 'reasoning'".to_string());
    }
    let encrypted_content = match object.get("encrypted_content") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(value)) if value.is_empty() => return Ok(None),
        Some(Value::String(value)) => value.clone(),
        Some(_) => {
            return Err("reasoning item encrypted_content must be a string or null".to_string());
        }
    };
    if model.trim().is_empty() {
        return Err("reasoning signature model must be a non-empty string".to_string());
    }
    let id = match object.get("id") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) if !value.is_empty() => Some(value.clone()),
        Some(Value::String(_)) => {
            return Err("reasoning item id must be a non-empty string when present".to_string());
        }
        Some(_) => return Err("reasoning item id must be a string or null".to_string()),
    };
    let summary_array = object
        .get("summary")
        .and_then(Value::as_array)
        .ok_or_else(|| "reasoning item requires a summary array".to_string())?;
    let mut summary = Vec::with_capacity(summary_array.len());
    for (index, entry) in summary_array.iter().enumerate() {
        let entry_obj = entry
            .as_object()
            .ok_or_else(|| format!("reasoning item summary[{index}] must be an object"))?;
        if entry_obj.get("type").and_then(Value::as_str) != Some("summary_text") {
            return Err(format!(
                "reasoning item summary[{index}] type must be 'summary_text'"
            ));
        }
        let text = entry_obj
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("reasoning item summary[{index}] requires string text"))?;
        summary.push(ReasoningSummaryEntry {
            kind: "summary_text".to_string(),
            text: text.to_string(),
        });
    }

    let envelope = ReasoningSignatureEnvelope {
        v: 1,
        model: model.to_string(),
        item: ReasoningSignatureItem {
            kind: "reasoning".to_string(),
            id,
            encrypted_content,
            summary,
        },
    };
    let json_bytes = serde_json::to_vec(&envelope)
        .map_err(|error| format!("failed to serialize reasoning signature envelope: {error}"))?;
    let payload = URL_SAFE_NO_PAD.encode(&json_bytes);
    Ok(Some(format!("{REASONING_SIGNATURE_PREFIX}{payload}")))
}

pub fn decode_reasoning_signature(
    signature: &str,
) -> Result<Option<ReasoningSignatureEnvelope>, String> {
    if signature.is_empty() {
        return Ok(None);
    }
    if !signature.starts_with(REASONING_SIGNATURE_PREFIX) {
        if has_recognized_codex_reasoning_prefix(signature) {
            return Err("thinking signature has an invalid codex-reasoning prefix".to_string());
        }
        return Ok(None);
    }

    let payload = &signature[REASONING_SIGNATURE_PREFIX.len()..];
    if payload.is_empty() || payload.len() % 4 == 1 {
        return Err("thinking signature payload is not valid unpadded base64url".to_string());
    }
    if !payload
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_')
    {
        return Err("thinking signature payload contains invalid base64url characters".to_string());
    }
    let decoded_bytes = URL_SAFE_NO_PAD
        .decode(payload)
        .map_err(|_| "thinking signature payload is not valid base64url".to_string())?;
    if URL_SAFE_NO_PAD.encode(&decoded_bytes) != payload {
        return Err("thinking signature payload has non-canonical base64url encoding".to_string());
    }
    let json_str = std::str::from_utf8(&decoded_bytes)
        .map_err(|_| "thinking signature payload is not valid UTF-8".to_string())?;
    let root = parse_strict_no_duplicate_keys(json_str)?;
    parse_envelope_value(&root).map(Some)
}

pub fn parse_envelope_value(root: &Value) -> Result<ReasoningSignatureEnvelope, String> {
    let root_obj = root
        .as_object()
        .ok_or_else(|| "thinking signature envelope must be a JSON object".to_string())?;
    for key in root_obj.keys() {
        if !matches!(key.as_str(), "v" | "model" | "item") {
            return Err(format!(
                "thinking signature envelope has unsupported field {key:?}"
            ));
        }
    }
    let v_val = root_obj
        .get("v")
        .ok_or_else(|| "thinking signature envelope requires field 'v'".to_string())?;
    let valid_v = v_val
        .as_number()
        .is_some_and(|num| !num.is_f64() && num.as_u64() == Some(1));
    if !valid_v {
        return Err("thinking signature envelope version 'v' must be 1".to_string());
    }

    let model = root_obj
        .get("model")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| {
            "thinking signature envelope requires a non-empty string 'model'".to_string()
        })?
        .to_string();

    let item_val = root_obj
        .get("item")
        .ok_or_else(|| "thinking signature envelope requires field 'item'".to_string())?;
    let item = parse_reasoning_signature_item(item_val)?;

    Ok(ReasoningSignatureEnvelope { v: 1, model, item })
}

pub fn parse_reasoning_signature_item(item_val: &Value) -> Result<ReasoningSignatureItem, String> {
    let item_obj = item_val
        .as_object()
        .ok_or_else(|| "thinking signature item must be a JSON object".to_string())?;
    for key in item_obj.keys() {
        if !matches!(
            key.as_str(),
            "type" | "id" | "encrypted_content" | "summary"
        ) {
            return Err(format!(
                "thinking signature item has unsupported field {key:?}"
            ));
        }
    }
    if item_obj.get("type").and_then(Value::as_str) != Some("reasoning") {
        return Err("thinking signature item.type must be 'reasoning'".to_string());
    }
    let id = match item_obj.get("id") {
        None => None,
        Some(Value::String(value)) if !value.is_empty() => Some(value.clone()),
        Some(_) => {
            return Err(
                "thinking signature item.id must be a non-empty string when present".to_string(),
            );
        }
    };
    let encrypted_content = item_obj
        .get("encrypted_content")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            "thinking signature item.encrypted_content must be a non-empty string".to_string()
        })?
        .to_string();
    let summary_arr = item_obj
        .get("summary")
        .and_then(Value::as_array)
        .ok_or_else(|| "thinking signature item.summary must be an array".to_string())?;
    let mut summary = Vec::with_capacity(summary_arr.len());
    for (index, entry) in summary_arr.iter().enumerate() {
        let entry_obj = entry
            .as_object()
            .ok_or_else(|| format!("thinking signature item.summary[{index}] must be an object"))?;
        for key in entry_obj.keys() {
            if !matches!(key.as_str(), "type" | "text") {
                return Err(format!(
                    "thinking signature item.summary[{index}] has unsupported field {key:?}"
                ));
            }
        }
        if entry_obj.get("type").and_then(Value::as_str) != Some("summary_text") {
            return Err(format!(
                "thinking signature item.summary[{index}].type must be 'summary_text'"
            ));
        }
        let text = entry_obj
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                format!("thinking signature item.summary[{index}].text must be a string")
            })?
            .to_string();
        summary.push(ReasoningSummaryEntry {
            kind: "summary_text".to_string(),
            text,
        });
    }

    Ok(ReasoningSignatureItem {
        kind: "reasoning".to_string(),
        id,
        encrypted_content,
        summary,
    })
}

pub fn normalize_reasoning_item_for_replay(
    raw_entry: &Value,
    target_model: Option<&str>,
) -> Result<Option<Value>, String> {
    let object = raw_entry
        .as_object()
        .ok_or_else(|| "reasoning_items entry must be an object".to_string())?;
    if object.contains_key("v") || object.contains_key("model") || object.contains_key("item") {
        let envelope = parse_envelope_value(raw_entry)?;
        if let Some(target) = target_model {
            if !is_reasoning_model_compatible(&envelope.model, target) {
                return Ok(None);
            }
        }
        return Ok(Some(envelope.item.to_value()));
    }

    if object.get("type").and_then(Value::as_str) != Some("reasoning") {
        return Err("reasoning_items entry must have type 'reasoning'".to_string());
    }
    let encrypted_content = match object.get("encrypted_content") {
        None | Some(Value::Null) => return Ok(None),
        Some(Value::String(value)) if value.is_empty() => return Ok(None),
        Some(Value::String(value)) => value.clone(),
        Some(_) => {
            return Err("reasoning item encrypted_content must be a string or null".to_string());
        }
    };
    let id = match object.get("id") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) if !value.is_empty() => Some(value.clone()),
        Some(Value::String(_)) => {
            return Err("reasoning item id must be a non-empty string when present".to_string());
        }
        Some(_) => return Err("reasoning item id must be a string or null".to_string()),
    };
    let summary_array = object
        .get("summary")
        .and_then(Value::as_array)
        .ok_or_else(|| "reasoning item requires a summary array".to_string())?;
    let mut summary = Vec::with_capacity(summary_array.len());
    for (index, entry) in summary_array.iter().enumerate() {
        let entry_obj = entry
            .as_object()
            .ok_or_else(|| format!("reasoning item summary[{index}] must be an object"))?;
        if entry_obj.get("type").and_then(Value::as_str) != Some("summary_text") {
            return Err(format!(
                "reasoning item summary[{index}] type must be 'summary_text'"
            ));
        }
        let text = entry_obj
            .get("text")
            .and_then(Value::as_str)
            .ok_or_else(|| format!("reasoning item summary[{index}] requires string text"))?;
        summary.push(ReasoningSummaryEntry {
            kind: "summary_text".to_string(),
            text: text.to_string(),
        });
    }
    let normalized = ReasoningSignatureItem {
        kind: "reasoning".to_string(),
        id,
        encrypted_content,
        summary,
    };
    Ok(Some(normalized.to_value()))
}

struct StrictNoDuplicateValue(Value);

impl<'de> Deserialize<'de> for StrictNoDuplicateValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ValueVisitor;

        impl<'de> Visitor<'de> for ValueVisitor {
            type Value = StrictNoDuplicateValue;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("any valid JSON value without duplicate object keys")
            }

            fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
                Ok(StrictNoDuplicateValue(Value::Bool(value)))
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
                Ok(StrictNoDuplicateValue(Value::Number(value.into())))
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
                Ok(StrictNoDuplicateValue(Value::Number(value.into())))
            }

            fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                let number = serde_json::Number::from_f64(value)
                    .ok_or_else(|| E::custom("non-finite JSON number"))?;
                Ok(StrictNoDuplicateValue(Value::Number(number)))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
                Ok(StrictNoDuplicateValue(Value::String(value.to_owned())))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(StrictNoDuplicateValue(Value::String(value)))
            }

            fn visit_none<E>(self) -> Result<Self::Value, E> {
                Ok(StrictNoDuplicateValue(Value::Null))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(StrictNoDuplicateValue(Value::Null))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut elements = Vec::new();
                while let Some(elem) = seq.next_element::<StrictNoDuplicateValue>()? {
                    elements.push(elem.0);
                }
                Ok(StrictNoDuplicateValue(Value::Array(elements)))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut values = Map::new();
                while let Some((key, value)) = map.next_entry::<String, StrictNoDuplicateValue>()? {
                    if values.insert(key.clone(), value.0).is_some() {
                        return Err(serde::de::Error::custom(format!(
                            "duplicate JSON object key {key:?}"
                        )));
                    }
                }
                Ok(StrictNoDuplicateValue(Value::Object(values)))
            }
        }

        deserializer.deserialize_any(ValueVisitor)
    }
}

fn parse_strict_no_duplicate_keys(raw: &str) -> Result<Value, String> {
    let parsed: StrictNoDuplicateValue = serde_json::from_str(raw)
        .map_err(|error| format!("thinking signature payload is not valid JSON: {error}"))?;
    strict_json::validate_value(&parsed.0).map_err(|error| {
        format!("thinking signature payload failed strict JSON validation: {error}")
    })?;
    Ok(parsed.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn encode_and_decode_roundtrip_preserves_exact_key_order_and_fields() {
        let item = json!({
            "type": "reasoning",
            "id": "rs_abc123",
            "status": "completed",
            "encrypted_content": "gAAAAABk_encrypted_blob",
            "summary": [
                {"type": "summary_text", "text": "Step 1"},
                {"type": "summary_text", "text": "Step 2"}
            ],
            "content": [
                {"type": "reasoning_text", "text": "raw text ignored"}
            ]
        });

        let signature = encode_reasoning_signature("gpt-5.4", &item)
            .expect("encode should succeed")
            .expect("signature should be present");
        assert!(signature.starts_with(REASONING_SIGNATURE_PREFIX));
        let payload = &signature[REASONING_SIGNATURE_PREFIX.len()..];
        assert!(!payload.contains('='));
        let decoded_json = String::from_utf8(URL_SAFE_NO_PAD.decode(payload).unwrap()).unwrap();
        assert_eq!(
            decoded_json,
            r#"{"v":1,"model":"gpt-5.4","item":{"type":"reasoning","id":"rs_abc123","encrypted_content":"gAAAAABk_encrypted_blob","summary":[{"type":"summary_text","text":"Step 1"},{"type":"summary_text","text":"Step 2"}]}}"#
        );

        let decoded = decode_reasoning_signature(&signature)
            .expect("decode should succeed")
            .expect("recognized signature should return envelope");
        assert_eq!(decoded.v, 1);
        assert_eq!(decoded.model, "gpt-5.4");
        assert_eq!(
            decoded.item.to_value(),
            json!({
                "type": "reasoning",
                "id": "rs_abc123",
                "encrypted_content": "gAAAAABk_encrypted_blob",
                "summary": [
                    {"type": "summary_text", "text": "Step 1"},
                    {"type": "summary_text", "text": "Step 2"}
                ]
            })
        );
    }

    #[test]
    fn encode_omits_absent_or_null_id_and_supports_empty_summary() {
        let item = json!({
            "type": "reasoning",
            "id": null,
            "encrypted_content": "gAAAAABk_encrypted_only",
            "summary": []
        });
        let signature = encode_reasoning_signature("gpt-5.4-mini", &item)
            .expect("encode should succeed")
            .expect("signature should be present");
        let payload = &signature[REASONING_SIGNATURE_PREFIX.len()..];
        let decoded_json = String::from_utf8(URL_SAFE_NO_PAD.decode(payload).unwrap()).unwrap();
        assert_eq!(
            decoded_json,
            r#"{"v":1,"model":"gpt-5.4-mini","item":{"type":"reasoning","encrypted_content":"gAAAAABk_encrypted_only","summary":[]}}"#
        );
        let decoded = decode_reasoning_signature(&signature)
            .expect("decode should succeed")
            .expect("envelope should be present");
        assert_eq!(decoded.item.id, None);
        assert!(decoded.item.summary.is_empty());
    }

    #[test]
    fn encode_returns_none_when_encrypted_content_missing_null_or_empty() {
        for item in [
            json!({"type": "reasoning", "summary": []}),
            json!({"type": "reasoning", "encrypted_content": null, "summary": []}),
            json!({"type": "reasoning", "encrypted_content": "", "summary": []}),
        ] {
            assert_eq!(encode_reasoning_signature("gpt-5.4", &item).unwrap(), None);
        }
    }

    #[test]
    fn decode_ignores_empty_and_foreign_signatures() {
        assert_eq!(decode_reasoning_signature("").unwrap(), None);
        assert_eq!(
            decode_reasoning_signature("ErUBCkYICBgCIkD_foreign_claude_signature").unwrap(),
            None
        );
        assert_eq!(
            decode_reasoning_signature("anthropic-signature-v1:abc").unwrap(),
            None
        );
    }

    #[test]
    fn decode_rejects_malformed_recognized_signatures() {
        let valid_json = r#"{"v":1,"model":"gpt-5.4","item":{"type":"reasoning","encrypted_content":"enc","summary":[]}}"#;
        let valid_b64 = URL_SAFE_NO_PAD.encode(valid_json.as_bytes());

        // Unsupported prefix version or malformed prefix
        assert!(decode_reasoning_signature(&format!("codex-reasoning-v2:{valid_b64}")).is_err());
        assert!(decode_reasoning_signature("codex-reasoning-v1").is_err());
        assert!(decode_reasoning_signature("codex-reasoning:abc").is_err());
        assert!(decode_reasoning_signature(&format!(" codex-reasoning-v1:{valid_b64}")).is_err());
        assert!(decode_reasoning_signature(&format!("CODEX-REASONING-v1:{valid_b64}")).is_err());
        assert!(decode_reasoning_signature(&format!("codex_reasoning_v1:{valid_b64}")).is_err());
        assert!(decode_reasoning_signature("  Codex-Reasoning:abc").is_err());

        // Empty payload, padding, whitespace, invalid base64url chars
        assert!(decode_reasoning_signature("codex-reasoning-v1:").is_err());
        assert!(decode_reasoning_signature(&format!("codex-reasoning-v1:{valid_b64}=")).is_err());
        assert!(decode_reasoning_signature(&format!("codex-reasoning-v1:{valid_b64} ")).is_err());
        assert!(decode_reasoning_signature("codex-reasoning-v1:ab+cd/ef").is_err());
        assert!(decode_reasoning_signature("codex-reasoning-v1:a").is_err());

        // Non-UTF8 payload
        let bad_utf8 = URL_SAFE_NO_PAD.encode([0xff, 0xfe, 0xfd]);
        assert!(decode_reasoning_signature(&format!("codex-reasoning-v1:{bad_utf8}")).is_err());

        // Duplicate keys in JSON
        let dup_json = URL_SAFE_NO_PAD.encode(
            r#"{"v":1,"v":1,"model":"gpt-5.4","item":{"type":"reasoning","encrypted_content":"enc","summary":[]}}"#
                .as_bytes(),
        );
        assert!(decode_reasoning_signature(&format!("codex-reasoning-v1:{dup_json}")).is_err());

        let nested_dup_json = URL_SAFE_NO_PAD.encode(
            r#"{"v":1,"model":"gpt-5.4","item":{"type":"reasoning","type":"reasoning","encrypted_content":"enc","summary":[]}}"#
                .as_bytes(),
        );
        assert!(
            decode_reasoning_signature(&format!("codex-reasoning-v1:{nested_dup_json}")).is_err()
        );

        // Wrong schema fields / types
        for invalid_payload in [
            r#"{"v":2,"model":"gpt-5.4","item":{"type":"reasoning","encrypted_content":"enc","summary":[]}}"#,
            r#"{"v":1,"model":"   ","item":{"type":"reasoning","encrypted_content":"enc","summary":[]}}"#,
            r#"{"v":1,"model":"gpt-5.4","extra":true,"item":{"type":"reasoning","encrypted_content":"enc","summary":[]}}"#,
            r#"{"v":1,"model":"gpt-5.4","item":{"type":"message","encrypted_content":"enc","summary":[]}}"#,
            r#"{"v":1,"model":"gpt-5.4","item":{"type":"reasoning","id":"","encrypted_content":"enc","summary":[]}}"#,
            r#"{"v":1,"model":"gpt-5.4","item":{"type":"reasoning","id":null,"encrypted_content":"enc","summary":[]}}"#,
            r#"{"v":1,"model":"gpt-5.4","item":{"type":"reasoning","encrypted_content":"","summary":[]}}"#,
            r#"{"v":1,"model":"gpt-5.4","item":{"type":"reasoning","encrypted_content":null,"summary":[]}}"#,
            r#"{"v":1,"model":"gpt-5.4","item":{"type":"reasoning","status":"completed","encrypted_content":"enc","summary":[]}}"#,
            r#"{"v":1,"model":"gpt-5.4","item":{"type":"reasoning","encrypted_content":"enc","summary":[{"type":"text","text":"x"}]}}"#,
            r#"{"v":1,"model":"gpt-5.4","item":{"type":"reasoning","encrypted_content":"enc","summary":[{"type":"summary_text","text":123}]}}"#,
            r#"{"v":1,"model":"gpt-5.4","item":{"type":"reasoning","encrypted_content":"enc","summary":[{"type":"summary_text","text":"x","extra":1}]}}"#,
        ] {
            let encoded = URL_SAFE_NO_PAD.encode(invalid_payload.as_bytes());
            assert!(
                decode_reasoning_signature(&format!("codex-reasoning-v1:{encoded}")).is_err(),
                "expected error for payload: {invalid_payload}"
            );
        }
    }

    #[test]
    fn five_model_compatibility_matrix_allows_all_25_ordered_pairs_and_blocks_outside_models() {
        let expected_item = json!({
            "type": "reasoning",
            "id": "rs_matrix",
            "encrypted_content": "enc_matrix",
            "summary": [{"type": "summary_text", "text": "Compatible reasoning"}],
        });

        let mut pair_count = 0usize;
        for source_model in super::REASONING_COMPATIBLE_MODELS {
            let envelope = json!({
                "v": 1,
                "model": source_model,
                "item": expected_item,
            });
            for target_model in super::REASONING_COMPATIBLE_MODELS {
                pair_count += 1;
                assert!(
                    super::is_reasoning_model_compatible(source_model, target_model),
                    "expected {source_model} -> {target_model} to be compatible"
                );
                let replayed =
                    normalize_reasoning_item_for_replay(&envelope, Some(target_model)).unwrap();
                assert_eq!(
                    replayed,
                    Some(expected_item.clone()),
                    "expected replay for {source_model} -> {target_model}"
                );
            }

            assert!(!super::is_reasoning_model_compatible(
                source_model,
                "gpt-5.4"
            ));
            assert_eq!(
                normalize_reasoning_item_for_replay(&envelope, Some("gpt-5.4")).unwrap(),
                None
            );

            let outside_envelope = json!({
                "v": 1,
                "model": "gpt-5.4",
                "item": expected_item,
            });
            assert!(!super::is_reasoning_model_compatible(
                "gpt-5.4",
                source_model
            ));
            assert_eq!(
                normalize_reasoning_item_for_replay(&outside_envelope, Some(source_model)).unwrap(),
                None
            );
        }
        assert_eq!(pair_count, 25);

        let exact_outside = json!({
            "v": 1,
            "model": "gpt-5.4",
            "item": expected_item,
        });
        assert!(super::is_reasoning_model_compatible("gpt-5.4", "gpt-5.4"));
        assert!(!super::is_reasoning_model_compatible(
            "gpt-5.4",
            "gpt-5.4-mini"
        ));
        assert_eq!(
            normalize_reasoning_item_for_replay(&exact_outside, Some("gpt-5.4")).unwrap(),
            Some(expected_item)
        );
        assert_eq!(
            normalize_reasoning_item_for_replay(&exact_outside, Some("gpt-5.4-mini")).unwrap(),
            None
        );
    }
}
