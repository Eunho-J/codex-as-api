use crate::auth::{self, AuthError, ChatGPTTokenData};
use crate::messages::{AssistantResponse, Message, MessageRole, ToolCall, ToolSchema, Usage};
use crate::protocol::{reasoning_from_response_items, response_failure_message};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::io::BufRead;

pub const CHATGPT_OAUTH_DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";
pub const CHATGPT_OAUTH_DEFAULT_MODEL: &str = "gpt-5.5";
pub const REMOTE_COMPACTION_MARKER: &str = "[Remote Responses compacted history]";
pub const REASONING_EFFORT_VALUES: &[&str] = &["high", "low", "medium", "minimal", "none", "xhigh"];

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("{0}")]
    Auth(#[from] AuthError),
    #[error("{0}")]
    Request(String),
}

pub struct ChatGPTOAuthProvider {
    pub model: String,
    pub base_url: String,
    pub auth_json_path: Option<String>,
    pub timeout: Option<std::time::Duration>,
}

impl ChatGPTOAuthProvider {
    pub fn new(
        model: String,
        base_url: String,
        auth_json_path: Option<String>,
        timeout: Option<std::time::Duration>,
    ) -> Self {
        Self {
            model,
            base_url: base_url.trim_end_matches('/').to_string(),
            auth_json_path,
            timeout,
        }
    }

    pub fn chat(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        temperature: Option<f64>,
        reasoning_effort: Option<&str>,
        max_tokens: Option<i64>,
        stop: Option<&[String]>,
        prompt_cache_key: Option<&str>,
        subagent: Option<&str>,
        memgen_request: Option<bool>,
        previous_response_id: Option<&str>,
        model: Option<&str>,
        tool_choice: Option<&Value>,
        service_tier: Option<&str>,
        text: Option<&Value>,
        client_metadata: Option<&HashMap<String, String>>,
    ) -> Result<AssistantResponse, ProviderError> {
        let mut content_parts: Vec<String> = Vec::new();
        let mut reasoning_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut finish_reason = "stop".to_string();
        let mut raw_events: Vec<Value> = Vec::new();
        let mut usage: Option<Usage> = None;

        let events = self.chat_stream(
            messages,
            tools,
            temperature,
            reasoning_effort,
            max_tokens,
            stop,
            prompt_cache_key,
            subagent,
            memgen_request,
            previous_response_id,
            model,
            tool_choice,
            service_tier,
            text,
            client_metadata,
        )?;

        for event in events {
            raw_events.push(event.clone());
            let typ = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match typ {
                "content" => {
                    if let Some(text) = event.get("text").and_then(|v| v.as_str()) {
                        content_parts.push(text.to_string());
                    }
                }
                "reasoning_delta" | "reasoning_raw_delta" => {
                    if let Some(text) = event.get("text").and_then(|v| v.as_str()) {
                        reasoning_parts.push(text.to_string());
                    }
                }
                "tool_call" => {
                    let id = event
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let name = event
                        .get("name")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let arguments = event
                        .get("arguments")
                        .and_then(|v| v.as_object())
                        .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
                        .unwrap_or_default();
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments,
                    });
                }
                "finish" => {
                    if let Some(fr) = event.get("finish_reason").and_then(|v| v.as_str()) {
                        finish_reason = fr.to_string();
                    }
                    if let Some(rc) = event.get("reasoning_content").and_then(|v| v.as_str()) {
                        reasoning_parts = vec![rc.to_string()];
                    }
                    if let Some(u) = usage_from_response(event.get("usage").unwrap_or(&Value::Null))
                    {
                        usage = Some(u);
                    }
                }
                _ => {}
            }
        }

        let reasoning_content = {
            let joined = reasoning_parts.join("");
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
        };

        let tail_events: Vec<Value> = if raw_events.len() > 20 {
            raw_events[raw_events.len() - 20..].to_vec()
        } else {
            raw_events
        };

        Ok(AssistantResponse {
            content: content_parts.join(""),
            tool_calls,
            finish_reason,
            usage,
            reasoning_content,
            raw: Some(json!({"events": tail_events})),
        })
    }

    pub fn chat_stream(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        temperature: Option<f64>,
        reasoning_effort: Option<&str>,
        _max_tokens: Option<i64>,
        stop: Option<&[String]>,
        prompt_cache_key: Option<&str>,
        subagent: Option<&str>,
        memgen_request: Option<bool>,
        previous_response_id: Option<&str>,
        model: Option<&str>,
        tool_choice: Option<&Value>,
        service_tier: Option<&str>,
        text: Option<&Value>,
        client_metadata: Option<&HashMap<String, String>>,
    ) -> Result<Vec<Value>, ProviderError> {
        let _ = temperature;
        let payload = self.responses_payload(
            messages,
            tools,
            reasoning_effort,
            stop,
            prompt_cache_key,
            previous_response_id,
            model,
            tool_choice,
            service_tier,
            text,
            client_metadata,
        )?;

        let mut extra_headers: HashMap<String, String> = HashMap::new();
        if let Some(sa) = subagent {
            extra_headers.insert("x-openai-subagent".to_string(), sa.to_string());
        }
        if let Some(mg) = memgen_request {
            extra_headers.insert(
                "x-openai-memgen-request".to_string(),
                if mg { "true" } else { "false" }.to_string(),
            );
        }

        let sse_events = self.post_sse("/responses", &payload, Some(&extra_headers))?;

        let mut result_events: Vec<Value> = Vec::new();
        let mut final_output: Vec<Value> = Vec::new();
        let mut reasoning_parts: Vec<String> = Vec::new();
        let mut saw_text_delta = false;
        let mut saw_reasoning_delta = false;

        for event in &sse_events {
            let typ = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match typ {
                "response.output_text.delta" => {
                    if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                        if !delta.is_empty() {
                            saw_text_delta = true;
                            result_events.push(json!({"type": "content", "text": delta}));
                        }
                    }
                }
                "response.output_item.done" => {
                    if let Some(item) = event.get("item") {
                        if item.is_object() {
                            final_output.push(item.clone());
                            if let Some(tc) = tool_call_from_response_item(item) {
                                result_events.push(json!({
                                    "type": "tool_call",
                                    "id": tc.id,
                                    "name": tc.name,
                                    "arguments": tc.arguments,
                                }));
                            }
                        }
                    }
                }
                "response.reasoning_summary_part.added" => {
                    result_events.push(json!({
                        "type": "reasoning_section_break",
                        "summary_index": event.get("summary_index"),
                        "part_index": event.get("part_index"),
                    }));
                }
                "response.reasoning_summary_text.delta" => {
                    if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                        if !delta.is_empty() {
                            saw_reasoning_delta = true;
                            reasoning_parts.push(delta.to_string());
                            result_events.push(json!({
                                "type": "reasoning_delta",
                                "text": delta,
                                "summary_index": event.get("summary_index"),
                            }));
                        }
                    }
                }
                "response.reasoning_text.delta" => {
                    if let Some(delta) = event.get("delta").and_then(|v| v.as_str()) {
                        if !delta.is_empty() {
                            saw_reasoning_delta = true;
                            reasoning_parts.push(delta.to_string());
                            result_events.push(json!({
                                "type": "reasoning_raw_delta",
                                "text": delta,
                                "summary_index": event.get("summary_index"),
                            }));
                        }
                    }
                }
                "response.failed" => {
                    return Err(ProviderError::Request(response_failure_message(
                        event, "failed",
                    )));
                }
                "response.incomplete" => {
                    return Err(ProviderError::Request(response_failure_message(
                        event,
                        "incomplete",
                    )));
                }
                "response.completed" => {
                    let mut usage_val = Value::Null;
                    if let Some(response) = event.get("response").and_then(|v| v.as_object()) {
                        usage_val = response.get("usage").cloned().unwrap_or(Value::Null);
                        if final_output.is_empty() {
                            if let Some(output) = response.get("output").and_then(|v| v.as_array())
                            {
                                for item in output {
                                    if item.is_object() {
                                        final_output.push(item.clone());
                                    }
                                }
                            }
                        }
                        if !saw_text_delta {
                            let final_text = text_from_response_items(&final_output);
                            if !final_text.is_empty() {
                                saw_text_delta = true;
                                result_events
                                    .push(json!({"type": "content", "text": final_text}));
                            }
                        }
                        if !saw_reasoning_delta {
                            let completed_reasoning = reasoning_from_response_items(
                                &final_output
                                    .iter()
                                    .filter(|i| i.is_object())
                                    .cloned()
                                    .collect::<Vec<_>>(),
                            );
                            if !completed_reasoning.is_empty() {
                                saw_reasoning_delta = true;
                                reasoning_parts.push(completed_reasoning.clone());
                                result_events.push(
                                    json!({"type": "reasoning_delta", "text": completed_reasoning}),
                                );
                            }
                        }
                    }
                    let reasoning_joined = reasoning_parts.join("");
                    result_events.push(json!({
                        "type": "finish",
                        "finish_reason": "stop",
                        "usage": usage_val,
                        "reasoning_content": if reasoning_joined.is_empty() { Value::Null } else { Value::String(reasoning_joined) },
                    }));
                }
                _ => {}
            }
        }

        Ok(result_events)
    }

    pub fn generate_image(
        &self,
        prompt: &str,
        reference_images: &[HashMap<String, String>],
        size: Option<&str>,
        reasoning_effort: Option<&str>,
        model: Option<&str>,
    ) -> Result<Vec<Value>, ProviderError> {
        if prompt.trim().is_empty() {
            return Err(ProviderError::Request(
                "image generation prompt is required".to_string(),
            ));
        }

        let mut content: Vec<Value> = vec![json!({"type": "input_text", "text": prompt})];
        let validated = validate_image_content_items(reference_images)?;
        content.extend(validated);

        if let Some(s) = size {
            if s != "auto" {
                let new_text = format!("{}\n\nRequested output size/aspect: {}", prompt, s);
                content[0] = json!({"type": "input_text", "text": new_text});
            }
        }

        let mut payload = json!({
            "model": model.unwrap_or(&self.model),
            "instructions": "Use the image_generation tool to create the requested image. Return the generated image through an image_generation_call result.",
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [{"type": "image_generation", "output_format": "png"}],
            "tool_choice": "auto",
            "parallel_tool_calls": false,
            "stream": true,
            "store": false,
            "include": [],
            "prompt_cache_key": uuid::Uuid::new_v4().to_string(),
        });

        set_reasoning_payload(payload.as_object_mut().unwrap(), reasoning_effort)?;
        let output_items = self.collect_response_output_items(&payload)?;

        let mut generated: Vec<Value> = Vec::new();
        for item in &output_items {
            if let Some(img) = image_generation_from_item(item)? {
                generated.push(img);
            }
        }

        if generated.is_empty() {
            return Err(ProviderError::Request(
                "image generation response returned no image_generation_call".to_string(),
            ));
        }

        Ok(generated)
    }

    pub fn inspect_images(
        &self,
        prompt: &str,
        images: &[HashMap<String, String>],
        reasoning_effort: Option<&str>,
        model: Option<&str>,
    ) -> Result<String, ProviderError> {
        if prompt.trim().is_empty() {
            return Err(ProviderError::Request(
                "image inspection prompt is required".to_string(),
            ));
        }

        let mut content: Vec<Value> = vec![json!({"type": "input_text", "text": prompt})];
        let validated = validate_image_content_items(images)?;
        content.extend(validated);

        let mut payload = json!({
            "model": model.unwrap_or(&self.model),
            "instructions": "Inspect the attached image(s) and answer the user's review prompt directly.",
            "input": [{"type": "message", "role": "user", "content": content}],
            "tools": [],
            "parallel_tool_calls": false,
            "stream": true,
            "store": false,
            "include": [],
            "prompt_cache_key": uuid::Uuid::new_v4().to_string(),
        });

        set_reasoning_payload(payload.as_object_mut().unwrap(), reasoning_effort)?;
        let output_items = self.collect_response_output_items(&payload)?;
        let text = text_from_response_items(&output_items).trim().to_string();

        if text.is_empty() {
            return Err(ProviderError::Request(
                "image inspection response returned empty content".to_string(),
            ));
        }

        Ok(text)
    }

    pub fn compact_messages(
        &self,
        messages: &[Message],
        reasoning_effort: Option<&str>,
        model: Option<&str>,
    ) -> Result<String, ProviderError> {
        let mut payload = json!({
            "model": model.unwrap_or(&self.model),
            "input": messages_to_response_items(messages),
            "instructions": "Create a compact checkpoint of this conversation for continuation.",
            "tools": [],
            "parallel_tool_calls": false,
        });

        set_reasoning_payload(payload.as_object_mut().unwrap(), reasoning_effort)?;
        let data = self.post_json("/responses/compact", &payload)?;

        let output = data
            .get("output")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                ProviderError::Request(
                    "remote compact response missing output array".to_string(),
                )
            })?;

        let serialized = serde_json::to_string(output).unwrap();
        Ok(format!("{}\n{}", REMOTE_COMPACTION_MARKER, serialized))
    }

    fn collect_response_output_items(
        &self,
        payload: &Value,
    ) -> Result<Vec<Value>, ProviderError> {
        let mut output_items: Vec<Value> = Vec::new();
        let mut completed_output_seen = false;
        let mut seen_keys: HashSet<String> = HashSet::new();

        let append_item = |item: &Value, items: &mut Vec<Value>, seen: &mut HashSet<String>| {
            let mut key_parts = vec![item
                .get("type")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string()];

            let mut found_id = false;
            for field in &["id", "call_id"] {
                if let Some(s) = item.get(field).and_then(|v| v.as_str()) {
                    if !s.is_empty() {
                        key_parts.push(s.to_string());
                        found_id = true;
                        break;
                    }
                }
            }
            if !found_id {
                key_parts.push(serde_json::to_string(item).unwrap_or_default());
            }

            let key = key_parts.join("\x1f");
            if seen.contains(&key) {
                return;
            }
            seen.insert(key);
            items.push(item.clone());
        };

        let events = self.post_sse("/responses", payload, None)?;
        for event in &events {
            let typ = event.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match typ {
                "response.output_item.done" => {
                    if let Some(item) = event.get("item") {
                        if item.is_object() {
                            append_item(item, &mut output_items, &mut seen_keys);
                        }
                    }
                }
                "response.failed" => {
                    return Err(ProviderError::Request(response_failure_message(
                        event, "failed",
                    )));
                }
                "response.incomplete" => {
                    return Err(ProviderError::Request(response_failure_message(
                        event,
                        "incomplete",
                    )));
                }
                "response.completed" => {
                    if let Some(response) = event.get("response").and_then(|v| v.as_object()) {
                        if let Some(output) = response.get("output").and_then(|v| v.as_array()) {
                            completed_output_seen = true;
                            for item in output {
                                if item.is_object() {
                                    append_item(item, &mut output_items, &mut seen_keys);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        if output_items.is_empty() && !completed_output_seen {
            return Err(ProviderError::Request(
                "ChatGPT OAuth response returned no output items".to_string(),
            ));
        }

        Ok(output_items)
    }

    fn responses_payload(
        &self,
        messages: &[Message],
        tools: Option<&[ToolSchema]>,
        reasoning_effort: Option<&str>,
        stop: Option<&[String]>,
        prompt_cache_key: Option<&str>,
        previous_response_id: Option<&str>,
        model: Option<&str>,
        tool_choice: Option<&Value>,
        service_tier: Option<&str>,
        text: Option<&Value>,
        client_metadata: Option<&HashMap<String, String>>,
    ) -> Result<Value, ProviderError> {
        let (instructions, input_items) = split_instructions_and_input(messages);

        if instructions.is_empty() {
            return Err(ProviderError::Request(
                "ChatGPT OAuth Responses request requires system instructions".to_string(),
            ));
        }

        let tools_array: Vec<Value> = match tools {
            Some(ts) => ts.iter().map(tool_schema_to_response_dict).collect(),
            None => vec![],
        };

        let mut payload = json!({
            "model": model.unwrap_or(&self.model),
            "instructions": instructions,
            "input": input_items,
            "tools": tools_array,
            "tool_choice": tool_choice.cloned().unwrap_or(json!("auto")),
            "parallel_tool_calls": false,
            "stream": true,
            "store": false,
            "include": [],
        });

        if let Some(key) = prompt_cache_key {
            if !key.is_empty() {
                payload
                    .as_object_mut()
                    .unwrap()
                    .insert("prompt_cache_key".to_string(), Value::String(key.to_string()));
            }
        }
        if let Some(s) = stop {
            payload.as_object_mut().unwrap().insert(
                "stop".to_string(),
                Value::Array(s.iter().map(|v| Value::String(v.clone())).collect()),
            );
        }
        if let Some(prid) = previous_response_id {
            payload.as_object_mut().unwrap().insert(
                "previous_response_id".to_string(),
                Value::String(prid.to_string()),
            );
        }
        if let Some(st) = service_tier {
            payload.as_object_mut().unwrap().insert(
                "service_tier".to_string(),
                Value::String(st.to_string()),
            );
        }
        if let Some(t) = text {
            payload.as_object_mut().unwrap().insert("text".to_string(), t.clone());
        }
        if let Some(cm) = client_metadata {
            let map: serde_json::Map<String, Value> = cm
                .iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect();
            payload
                .as_object_mut()
                .unwrap()
                .insert("client_metadata".to_string(), Value::Object(map));
        }

        set_reasoning_payload(payload.as_object_mut().unwrap(), reasoning_effort)?;

        Ok(payload)
    }

    fn headers(&self) -> Result<(HashMap<String, String>, ChatGPTTokenData), ProviderError> {
        let token = auth::load_token_data(self.auth_json_path.as_deref())?;
        let mut headers = HashMap::new();
        headers.insert(
            "Authorization".to_string(),
            format!("Bearer {}", token.access_token),
        );
        headers.insert(
            "ChatGPT-Account-Id".to_string(),
            token.account_id.clone(),
        );
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        if token.fedramp {
            headers.insert("X-OpenAI-Fedramp".to_string(), "true".to_string());
        }
        Ok((headers, token))
    }

    fn post_json(&self, path: &str, payload: &Value) -> Result<Value, ProviderError> {
        let raw = self.request_json(path, payload)?;
        let data: Value = serde_json::from_slice(&raw).map_err(|_| {
            ProviderError::Request("ChatGPT OAuth response must be a JSON object".to_string())
        })?;
        if !data.is_object() {
            return Err(ProviderError::Request(
                "ChatGPT OAuth response must be a JSON object".to_string(),
            ));
        }
        Ok(data)
    }

    fn post_sse(
        &self,
        path: &str,
        payload: &Value,
        extra_headers: Option<&HashMap<String, String>>,
    ) -> Result<Vec<Value>, ProviderError> {
        self.request_sse(path, payload, extra_headers)
    }

    fn request_sse(
        &self,
        path: &str,
        payload: &Value,
        extra_headers: Option<&HashMap<String, String>>,
    ) -> Result<Vec<Value>, ProviderError> {
        for attempt in 0..2 {
            let (mut headers, token) = self.headers()?;
            headers.insert("Accept".to_string(), "text/event-stream".to_string());
            if let Some(eh) = extra_headers {
                for (k, v) in eh {
                    headers.insert(k.clone(), v.clone());
                }
            }
            let token_values = [
                token.access_token.as_str(),
                token.refresh_token.as_str(),
                token.id_token.as_str(),
                token.account_id.as_str(),
            ];

            let url = format!("{}{}", self.base_url, path);
            let body = serde_json::to_vec(payload).unwrap();

            let mut builder = reqwest::blocking::Client::new().post(&url);
            for (k, v) in &headers {
                builder = builder.header(k.as_str(), v.as_str());
            }
            if let Some(t) = self.timeout {
                builder = builder.timeout(t);
            }
            builder = builder.body(body);

            match builder.send() {
                Ok(response) => {
                    let status = response.status();
                    if status == reqwest::StatusCode::UNAUTHORIZED && attempt == 0 {
                        let _ = auth::do_refresh_token(self.auth_json_path.as_deref());
                        continue;
                    }
                    if !status.is_success() {
                        let body_text = response.text().unwrap_or_default();
                        let redacted = auth::redact_text(&body_text, &token_values);
                        return Err(ProviderError::Request(format!(
                            "ChatGPT OAuth request failed: HTTP {}: {}",
                            status.as_u16(),
                            redacted
                        )));
                    }

                    let reader = std::io::BufReader::new(response);
                    let mut events: Vec<Value> = Vec::new();
                    let mut block: Vec<String> = Vec::new();

                    for line_result in reader.lines() {
                        let line = match line_result {
                            Ok(l) => l,
                            Err(_) => break,
                        };
                        if line.is_empty() {
                            if let Some(event) = decode_sse_block(&block) {
                                events.push(event);
                            }
                            block.clear();
                            continue;
                        }
                        block.push(line);
                    }
                    if !block.is_empty() {
                        if let Some(event) = decode_sse_block(&block) {
                            events.push(event);
                        }
                    }

                    return Ok(events);
                }
                Err(e) => {
                    let msg = auth::redact_text(&e.to_string(), &token_values);
                    return Err(ProviderError::Request(format!(
                        "ChatGPT OAuth request failed: {}",
                        msg
                    )));
                }
            }
        }

        unreachable!("ChatGPT OAuth request retry state")
    }

    fn request_json(&self, path: &str, payload: &Value) -> Result<Vec<u8>, ProviderError> {
        for attempt in 0..2 {
            let (headers, token) = self.headers()?;
            let token_values = [
                token.access_token.as_str(),
                token.refresh_token.as_str(),
                token.id_token.as_str(),
                token.account_id.as_str(),
            ];

            let url = format!("{}{}", self.base_url, path);
            let body = serde_json::to_vec(payload).unwrap();

            let mut builder = reqwest::blocking::Client::new().post(&url);
            for (k, v) in &headers {
                builder = builder.header(k.as_str(), v.as_str());
            }
            if let Some(t) = self.timeout {
                builder = builder.timeout(t);
            }
            builder = builder.body(body);

            match builder.send() {
                Ok(response) => {
                    let status = response.status();
                    if status == reqwest::StatusCode::UNAUTHORIZED && attempt == 0 {
                        let _ = auth::do_refresh_token(self.auth_json_path.as_deref());
                        continue;
                    }
                    if !status.is_success() {
                        let body_text = response.text().unwrap_or_default();
                        let redacted = auth::redact_text(&body_text, &token_values);
                        return Err(ProviderError::Request(format!(
                            "ChatGPT OAuth request failed: HTTP {}: {}",
                            status.as_u16(),
                            redacted
                        )));
                    }
                    let bytes = response.bytes().map_err(|e| {
                        ProviderError::Request(format!(
                            "ChatGPT OAuth request failed: {}",
                            auth::redact_text(&e.to_string(), &token_values)
                        ))
                    })?;
                    return Ok(bytes.to_vec());
                }
                Err(e) => {
                    let msg = auth::redact_text(&e.to_string(), &token_values);
                    return Err(ProviderError::Request(format!(
                        "ChatGPT OAuth request failed: {}",
                        msg
                    )));
                }
            }
        }

        unreachable!("ChatGPT OAuth request retry state")
    }
}

fn validate_image_content_items(
    images: &[HashMap<String, String>],
) -> Result<Vec<Value>, ProviderError> {
    let mut items: Vec<Value> = Vec::new();
    for (index, image) in images.iter().enumerate() {
        let image_url = image.get("image_url").map(|s| s.as_str()).unwrap_or("");
        if image_url.trim().is_empty() {
            return Err(ProviderError::Request(format!(
                "image reference {} requires image_url",
                index
            )));
        }
        if !image_url.starts_with("data:image/") {
            return Err(ProviderError::Request(format!(
                "image reference {} must be a data:image URL",
                index
            )));
        }
        items.push(json!({"type": "input_image", "image_url": image_url}));
    }
    Ok(items)
}

fn image_generation_from_item(item: &Value) -> Result<Option<Value>, ProviderError> {
    if item.get("type").and_then(|v| v.as_str()) != Some("image_generation_call") {
        return Ok(None);
    }
    let result = item
        .get("result")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if result.trim().is_empty() {
        return Err(ProviderError::Request(
            "image_generation_call returned empty result".to_string(),
        ));
    }
    let id = item
        .get("id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());
    let status = item
        .get("status")
        .and_then(|v| v.as_str())
        .unwrap_or("completed");
    let revised_prompt = item
        .get("revised_prompt")
        .and_then(|v| v.as_str())
        .map(|s| Value::String(s.to_string()))
        .unwrap_or(Value::Null);

    Ok(Some(json!({
        "id": id,
        "status": status,
        "revised_prompt": revised_prompt,
        "result": result,
    })))
}

fn decode_sse_block(lines: &[String]) -> Option<Value> {
    let data_lines: Vec<&str> = lines
        .iter()
        .filter_map(|line| {
            if line.starts_with("data:") {
                Some(line[5..].trim())
            } else {
                None
            }
        })
        .collect();

    if data_lines.is_empty() {
        return None;
    }

    let joined = data_lines.join("\n");
    if joined == "[DONE]" {
        return None;
    }

    let event: Value = serde_json::from_str(&joined).ok()?;
    if event.is_object() {
        Some(event)
    } else {
        None
    }
}

pub fn split_instructions_and_input(messages: &[Message]) -> (String, Vec<Value>) {
    let mut instructions: Vec<String> = Vec::new();
    let mut input_messages: Vec<&Message> = Vec::new();

    for msg in messages {
        if msg.role == MessageRole::System
            && !msg.content.starts_with(REMOTE_COMPACTION_MARKER)
        {
            instructions.push(msg.content.clone());
        } else {
            input_messages.push(msg);
        }
    }

    (
        instructions.join("\n\n"),
        messages_to_response_items_refs(&input_messages),
    )
}

pub fn messages_to_response_items(messages: &[Message]) -> Vec<Value> {
    let refs: Vec<&Message> = messages.iter().collect();
    messages_to_response_items_refs(&refs)
}

fn messages_to_response_items_refs(messages: &[&Message]) -> Vec<Value> {
    let mut items: Vec<Value> = Vec::new();

    for message in messages {
        if message.role == MessageRole::System
            && message.content.starts_with(REMOTE_COMPACTION_MARKER)
        {
            let raw = message.content[REMOTE_COMPACTION_MARKER.len()..].trim();
            if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(raw) {
                for item in arr {
                    if item.is_object() {
                        items.push(item);
                    }
                }
            }
            continue;
        }

        if message.role == MessageRole::Tool {
            let call_id = message
                .tool_call_id
                .as_deref()
                .or(message.name.as_deref())
                .unwrap_or("tool-call");
            items.push(json!({
                "type": "function_call_output",
                "call_id": call_id,
                "output": message.content,
            }));
            continue;
        }

        if message.role == MessageRole::Assistant && !message.tool_calls.is_empty() {
            if !message.content.is_empty() {
                items.push(message_item("assistant", &message.content, &[]));
            }
            for tc in &message.tool_calls {
                items.push(json!({
                    "type": "function_call",
                    "call_id": tc.id,
                    "name": tc.name,
                    "arguments": serde_json::to_string(&tc.arguments).unwrap_or_else(|_| "{}".to_string()),
                }));
            }
            continue;
        }

        let role = if message.role == MessageRole::Assistant {
            "assistant"
        } else {
            "user"
        };
        items.push(message_item(role, &message.content, &message.images));
    }

    items
}

fn message_item(role: &str, content: &str, images: &[String]) -> Value {
    let typ = if role == "assistant" {
        "output_text"
    } else {
        "input_text"
    };
    let mut content_items = vec![json!({"type": typ, "text": content})];
    for image_url in images {
        content_items.push(json!({"type": "input_image", "image_url": image_url}));
    }
    content_items[0] = json!({"type": typ, "text": if content.is_empty() { "" } else { content }});
    json!({
        "type": "message",
        "role": role,
        "content": content_items,
    })
}

fn tool_schema_to_response_dict(tool: &ToolSchema) -> Value {
    json!({
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
        "strict": false,
    })
}

fn set_reasoning_payload(
    payload: &mut serde_json::Map<String, Value>,
    reasoning_effort: Option<&str>,
) -> Result<(), ProviderError> {
    let effort = match reasoning_effort {
        Some(e) if !e.is_empty() => e,
        Some(_) => {
            return Err(ProviderError::Request(
                "reasoning_effort must be a non-empty string when provided".to_string(),
            ));
        }
        None => return Ok(()),
    };

    let lower = effort.to_lowercase();
    if !REASONING_EFFORT_VALUES.contains(&lower.as_str()) {
        return Err(ProviderError::Request(format!(
            "reasoning_effort must be one of: {}",
            {
                let mut sorted = REASONING_EFFORT_VALUES.to_vec();
                sorted.sort();
                sorted.join(", ")
            }
        )));
    }

    payload.insert(
        "reasoning".to_string(),
        json!({"effort": lower}),
    );

    Ok(())
}

fn tool_call_from_response_item(item: &Value) -> Option<ToolCall> {
    let typ = item.get("type").and_then(|v| v.as_str())?;
    if typ != "function_call" && typ != "custom_tool_call" {
        return None;
    }
    let name = item.get("name").and_then(|v| v.as_str())?;
    if name.is_empty() {
        return None;
    }
    let raw_args = item
        .get("arguments")
        .or_else(|| item.get("input"))
        .cloned()
        .unwrap_or(Value::String("{}".to_string()));

    let args: HashMap<String, Value> = match &raw_args {
        Value::String(s) => {
            if s.is_empty() {
                HashMap::new()
            } else {
                serde_json::from_str(s).unwrap_or_else(|_| {
                    let mut m = HashMap::new();
                    m.insert("input".to_string(), Value::String(s.clone()));
                    m
                })
            }
        }
        Value::Object(map) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        _ => HashMap::new(),
    };

    let call_id = item
        .get("call_id")
        .or_else(|| item.get("id"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| uuid::Uuid::new_v4().simple().to_string());

    Some(ToolCall {
        id: call_id,
        name: name.to_string(),
        arguments: args,
    })
}

pub fn text_from_response_items(items: &[Value]) -> String {
    let mut parts: Vec<String> = Vec::new();
    for item in items {
        let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if item_type == "output_text" || item_type == "text" {
            if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                if !text.is_empty() {
                    parts.push(text.to_string());
                }
            }
            continue;
        }
        if item_type != "message" {
            continue;
        }
        if let Some(content) = item.get("content").and_then(|v| v.as_array()) {
            for part in content {
                match part {
                    Value::String(s) if !s.is_empty() => {
                        parts.push(s.clone());
                    }
                    Value::Object(map) => {
                        let part_type = map.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        if part_type == "output_text" || part_type == "text" {
                            if let Some(text) = map.get("text").and_then(|v| v.as_str()) {
                                if !text.is_empty() {
                                    parts.push(text.to_string());
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    parts.join("")
}

pub fn usage_from_response(value: &Value) -> Option<Usage> {
    let obj = value.as_object()?;
    let prompt = obj
        .get("input_tokens")
        .or_else(|| obj.get("prompt_tokens"))
        .and_then(|v| v.as_i64())?;
    let completion = obj
        .get("output_tokens")
        .or_else(|| obj.get("completion_tokens"))
        .and_then(|v| v.as_i64())?;
    let total = obj.get("total_tokens").and_then(|v| v.as_i64());

    let mut cached_tokens: i64 = 0;
    let token_details = obj
        .get("input_tokens_details")
        .or_else(|| obj.get("prompt_tokens_details"));
    if let Some(details) = token_details.and_then(|v| v.as_object()) {
        if let Some(ct) = details.get("cached_tokens").and_then(|v| v.as_i64()) {
            cached_tokens = ct;
        }
    } else if let Some(ct) = obj.get("cached_input_tokens").and_then(|v| v.as_i64()) {
        cached_tokens = ct;
    } else if let Some(ct) = obj.get("cache_read_input_tokens").and_then(|v| v.as_i64()) {
        cached_tokens = ct;
    }

    Some(Usage::new(prompt, completion, total, cached_tokens))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_instructions_and_input() {
        let messages = vec![
            Message {
                role: MessageRole::System,
                content: "You are helpful.".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
            },
            Message {
                role: MessageRole::User,
                content: "Hello".to_string(),
                tool_calls: vec![],
                tool_call_id: None,
                name: None,
                reasoning_content: None,
                images: vec![],
            },
        ];
        let (instructions, input) = split_instructions_and_input(&messages);
        assert_eq!(instructions, "You are helpful.");
        assert_eq!(input.len(), 1);
    }

    #[test]
    fn test_message_item_user() {
        let item = message_item("user", "hello", &[]);
        assert_eq!(item["type"], "message");
        assert_eq!(item["role"], "user");
        let content = item["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "input_text");
        assert_eq!(content[0]["text"], "hello");
    }

    #[test]
    fn test_message_item_assistant() {
        let item = message_item("assistant", "response", &[]);
        let content = item["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "output_text");
    }

    #[test]
    fn test_tool_schema_to_response_dict() {
        let schema = ToolSchema {
            name: "my_tool".to_string(),
            description: "Does stuff".to_string(),
            parameters: json!({"type": "object"}),
        };
        let result = tool_schema_to_response_dict(&schema);
        assert_eq!(result["type"], "function");
        assert_eq!(result["name"], "my_tool");
        assert_eq!(result["strict"], false);
    }

    #[test]
    fn test_set_reasoning_payload_valid() {
        let mut payload = serde_json::Map::new();
        set_reasoning_payload(&mut payload, Some("high")).unwrap();
        assert_eq!(payload["reasoning"]["effort"], "high");
    }

    #[test]
    fn test_set_reasoning_payload_invalid() {
        let mut payload = serde_json::Map::new();
        let result = set_reasoning_payload(&mut payload, Some("ultra"));
        assert!(result.is_err());
    }

    #[test]
    fn test_set_reasoning_payload_none() {
        let mut payload = serde_json::Map::new();
        set_reasoning_payload(&mut payload, None).unwrap();
        assert!(!payload.contains_key("reasoning"));
    }

    #[test]
    fn test_text_from_response_items() {
        let items = vec![
            json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello "}]}),
            json!({"type": "output_text", "text": "world"}),
        ];
        assert_eq!(text_from_response_items(&items), "Hello world");
    }

    #[test]
    fn test_text_from_response_items_empty() {
        let items: Vec<Value> = vec![];
        assert_eq!(text_from_response_items(&items), "");
    }

    #[test]
    fn test_usage_from_response_valid() {
        let val = json!({
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
            "input_tokens_details": {"cached_tokens": 30}
        });
        let usage = usage_from_response(&val).unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
        assert_eq!(usage.cached_tokens, 30);
    }

    #[test]
    fn test_usage_from_response_alternative_keys() {
        let val = json!({
            "prompt_tokens": 200,
            "completion_tokens": 100,
            "cached_input_tokens": 50
        });
        let usage = usage_from_response(&val).unwrap();
        assert_eq!(usage.prompt_tokens, 200);
        assert_eq!(usage.cached_tokens, 50);
    }

    #[test]
    fn test_usage_from_response_invalid() {
        let val = json!({"foo": "bar"});
        assert!(usage_from_response(&val).is_none());
    }

    #[test]
    fn test_decode_sse_block_valid() {
        let lines = vec![
            "data: {\"type\": \"response.completed\"}".to_string(),
        ];
        let event = decode_sse_block(&lines).unwrap();
        assert_eq!(event["type"], "response.completed");
    }

    #[test]
    fn test_decode_sse_block_done() {
        let lines = vec!["data: [DONE]".to_string()];
        assert!(decode_sse_block(&lines).is_none());
    }

    #[test]
    fn test_decode_sse_block_no_data() {
        let lines = vec!["event: ping".to_string()];
        assert!(decode_sse_block(&lines).is_none());
    }

    #[test]
    fn test_tool_call_from_response_item_function() {
        let item = json!({
            "type": "function_call",
            "name": "read_file",
            "call_id": "call-1",
            "arguments": "{\"path\": \"/tmp/test\"}"
        });
        let tc = tool_call_from_response_item(&item).unwrap();
        assert_eq!(tc.name, "read_file");
        assert_eq!(tc.id, "call-1");
        assert_eq!(tc.arguments["path"], "/tmp/test");
    }

    #[test]
    fn test_tool_call_from_response_item_not_function() {
        let item = json!({"type": "message"});
        assert!(tool_call_from_response_item(&item).is_none());
    }

    #[test]
    fn test_messages_to_response_items_tool() {
        let msg = Message {
            role: MessageRole::Tool,
            content: "result".to_string(),
            tool_calls: vec![],
            tool_call_id: Some("call-1".to_string()),
            name: Some("my_tool".to_string()),
            reasoning_content: None,
                images: vec![],
        };
        let items = messages_to_response_items(&[msg]);
        assert_eq!(items.len(), 1);
        assert_eq!(items[0]["type"], "function_call_output");
        assert_eq!(items[0]["call_id"], "call-1");
    }

    #[test]
    fn test_validate_image_content_items_valid() {
        let mut img = HashMap::new();
        img.insert("image_url".to_string(), "data:image/png;base64,abc".to_string());
        let result = validate_image_content_items(&[img]).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0]["type"], "input_image");
    }

    #[test]
    fn test_validate_image_content_items_missing_url() {
        let img = HashMap::new();
        let result = validate_image_content_items(&[img]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_image_content_items_wrong_prefix() {
        let mut img = HashMap::new();
        img.insert("image_url".to_string(), "https://example.com/img.png".to_string());
        let result = validate_image_content_items(&[img]);
        assert!(result.is_err());
    }
}
