# Release Notes

## v0.6.0

### GPT-5.6 Codex compatibility

- Audit official `openai/codex` `main` at `6ad0e943cc727dc836d7c671f3377db30107f4d9` and keep public GPT-5.6 API extensions separate from the private Codex OAuth request contract.
- Add the public `gpt-5.6` alias (resolved outbound to Sol) plus official Codex capability metadata for `gpt-5.6-sol`, `gpt-5.6-terra`, and `gpt-5.6-luna`.
- Refresh context maxima and default reasoning effort for the existing GPT-5.5, GPT-5.4, GPT-5.4 Mini, and GPT-5.2 entries.
- Use the current Responses Lite input-item and header contract for chat, compact, and inspection requests; reject Lite image generation explicitly because this proxy has no standalone image-tool executor.
- Support `max`, map Codex's virtual `ultra` setting to backend `max`, and preserve non-empty model-defined effort values.
- Read `model_reasoning_effort` from Codex config and apply request, config, then model-default precedence.
- Preserve Anthropic `thinking.disabled` as explicit `none` so catalog defaults cannot re-enable reasoning.
- Use catalog context maxima for GPT-5.6 models and clamp configured context/compact limits to the official 372,000 / 334,800 bounds.

### GPT-5.6 request and response wiring

- Accept public Responses-shaped `reasoning.effort`, `reasoning.mode: "standard"`, and `reasoning.context` (`auto` / `current_turn` / `all_turns`) while retaining `reasoning_effort` compatibility and rejecting conflicting values. Omit `standard` from the private wire and reject Pro because official Codex has no equivalent request field or alias.
- Use the documented `medium` effort when a GPT-5.6 mode is explicit and neither the request nor Codex config selects an effort; keep mode and effort independent.
- Expose the upstream Responses ID as `response_id`, preserve encrypted reasoning state, and translate a known `previous_response_id` into bounded process-local full-history replay over the existing private HTTP transport. Capture replay state from the official `response.output_item.done` events; `response.completed` supplies the ID/usage and its private-rollout `output` array is empty.
- Preserve `prompt_cache_key`, `cache_write_tokens`, and standard Chat verbosity mapping. Treat explicit `null` as omitted and reject non-null values for public `prompt_cache_options`, explicit breakpoints, and `safety_identifier` before upstream because the private Codex HTTP/WS contracts provide no equivalent fields.
- Omit null or empty stop controls and reject non-empty OpenAI `stop` / Anthropic `stop_sequences` before upstream because the official Codex HTTP request has no stop field.
- Preserve multimodal Chat content and `auto` / `low` / `high` image detail on classic Responses, plus capability-gated `original` for GPT-5.6, GPT-5.5, GPT-5.4, and GPT-5.4 Mini; reject `original` for GPT-5.2 and conservative legacy entries, and keep the official Codex Lite behavior that removes only `detail` after capability validation.
- Resolve a known compact `previous_response_id` to full input locally; never forward that field or public cache options. Forward supported `prompt_cache_key`, service-tier, and text controls while retaining the official Lite private-wire `context: all_turns` default.
- Treat null optional controls as omitted; reject non-null values for compact `safety_identifier`, encrypted-reasoning `include`, and deprecated `prompt_cache_retention` instead of silently dropping unsupported fields.
- Reject hosted Multi-agent and Programmatic Tool Calling fields on Chat/Anthropic facades because their agent/program/caller lifecycle cannot be represented losslessly by those protocols.
- Canonicalize Codex `service_tier: "fast"` to the private wire value `priority`, omit explicit `default`, and reject tiers not advertised by the selected model.

### Validation

- Cover handler → provider → recording-upstream continuation with structured assertions in Python, TypeScript, and Rust, including encrypted reasoning history, branching, unknown IDs, Compact translation, and unsupported-field failures.
- Live Codex OAuth: `gpt-5.6-sol` completed both a direct private WebSocket continuation using the exact prior `response.id` and a private HTTP continuation using full input/output replay. Direct HTTP forwarding of Pro/cache/safety controls and `previous_response_id` remains rejected and is not used by the proxy.

## v0.5.2

### Codex upstream parity

- Support latest Codex root-level OAuth token files while keeping PAT-only, agent-identity-only, and Bedrock-only auth files explicitly unsupported.
- Add shared model capability gating for Responses Lite, parallel tool calls, verbosity, and service-tier behavior across Python, TypeScript, and Rust.
- Preserve encrypted reasoning state via `reasoning.encrypted_content` and add Codex metadata forwarding controls.

## v0.5.1

### Codex backend version-header compatibility
- Add the official Codex CLI `originator` header and versioned `User-Agent` header to ChatGPT/Codex OAuth requests across Python, TypeScript, and Rust.
- Resolve the latest `@openai/codex` version from npm during server startup and cache it for outgoing requests.
- Keep `CODEX_AS_API_CODEX_CLI_VERSION` as an explicit override for offline or pinned deployments.

### Validation
- Python: `CODEX_AS_API_AUTH_PATH=/tmp/codex-as-api-missing-auth.json .venv/bin/pytest -q`
- Rust: `cargo test`
- TypeScript: `npm test && npm run build && npm pack --dry-run`
- Python package dry run: `uv build --out-dir /tmp/codex-as-api-dist`

## v0.5.0

### Claude Code conversation-history compatibility
- Preserve Anthropic server-tool history blocks (`server_tool_use`, `web_search_tool_result`, and other `*_tool_result` blocks) as backend-readable context instead of silently dropping them on follow-up turns.
- Preserve `redacted_thinking` placeholders without exposing unavailable reasoning text.
- Preserve `document` and `search_result` content nested inside `tool_result` blocks.
- Keep Python Anthropic streaming routes aligned with provider defaults so `text.format` and omitted optional knobs can pass through consistently.

### Structured outputs
- Map Anthropic `output_format` / `output_config.format`-style JSON schema requests to OpenAI Responses `text.format` for Claude Code side queries.
- Keep JSON schema names OpenAI-compatible while preserving schema, description, and explicit `strict` settings.

### Web search version tolerance
- Accept unsuffixed `type: "web_search"` server tools in addition to versioned `web_search_*` tool types.

### Validation
- Python: `.venv/bin/pytest -q`
- Rust: `cargo test`
- TypeScript: `npm test && npm run build`

## v0.4.0

### Anthropic hosted web search compatibility
- Route Anthropic `web_search_*` server tools to the OpenAI Responses hosted `web_search` tool instead of treating them as function tools.
- Convert hosted web search calls back into Anthropic `server_tool_use` and `web_search_tool_result` blocks for Claude Code compatibility.
- Preserve or synthesize `usage.server_tool_use.web_search_requests` and cover the behavior across Python, TypeScript, and Rust.

### Package publishing preparation
- Add a GitHub Actions workflow that builds/tests the TypeScript package and can publish to npmjs (`codex-as-api`) and GitHub Packages (`@eunho-j/codex-as-api`) by manual dispatch.

### Validation
- Python: `.venv/bin/pytest -q`
- Rust: `cargo test`
- TypeScript: `npm test && npm run build`

## v0.3.3

### Claude Code retry fix
- Stop forwarding Anthropic/OpenAI `max_tokens` as Codex `max_output_tokens` because the Codex OAuth Responses backend rejects that parameter.
- Add Python, TypeScript, and Rust regression coverage to keep provider payloads free of `max_output_tokens` even when clients send `max_tokens`.

### Validation
- Python: `PYTHONPATH=src pytest -q`
- Rust: `cargo test`
- TypeScript: `npm test && npm run build`

## v0.3.2

### Claude Code compatibility fix
- Restored immediate Anthropic streaming so clients receive `message_start` before the backend response completes.
- `/v1/messages/count_tokens` now returns a conservative local estimate because Codex OAuth has no Anthropic-equivalent count-only endpoint.
- Token estimates use UTF-8 byte length as a conservative upper bound for GPT/Codex BPE text tokens, plus overhead for roles, message boundaries, tools, raw request metadata, and images.
- Keeps real final streaming usage in `message_delta` while avoiding stream buffering.

### Validation
- Python: `PYTHONPATH=src pytest -q`
- Rust: `cargo test`
- TypeScript: `npm test && npm run build`
- Live smoke: `/v1/messages/count_tokens` and streaming `/v1/messages` against local server

## v0.3.1

### Anthropic token accounting attempt
- `/v1/messages/count_tokens` asked the Codex backend for real input-token usage with `max_output_tokens: 0`; this is superseded by v0.3.2 because Codex OAuth rejects count-only requests.
- Token counting forwards Anthropic-converted tools, tool choice, stop sequences, and thinking/reasoning settings across Python, TypeScript, and Rust.
- Provider requests now pass `max_output_tokens` through to Codex where requested.

### Streaming usage parity
- Anthropic streaming now propagates real cumulative usage details from the backend, including cache creation/read fields, server tool use, and service tier metadata when present.
- `message_start` and final `message_delta` usage payloads now match the backend-reported accounting across all implementations.

### Validation
- Python: `PYTHONPATH=src pytest -q`
- Rust: `cargo test`
- TypeScript: `npm test && npm run build`

## v0.3.0

### Codex config parity
- Python, TypeScript, and Rust now read Codex CLI config from `CODEX_HOME` / `~/.codex/config.toml`.
- `model`, `model_context_window`, and `model_auto_compact_token_limit` are reflected consistently across implementations.
- `CODEX_AS_API_MODEL` still overrides the Codex config model when set.

### Claude Code / Anthropic compatibility
- `/v1/messages` preserves the client-supplied Anthropic model name in responses while using the configured Codex model for backend requests.
- Added `POST /v1/messages/count_tokens` with estimated `input_tokens`, `context_window`, and `auto_compact_token_limit`.
- Added `POST /v1/messages/compact` as an Anthropic-compatible alias for remote conversation compaction.

### Observability and error handling
- `/health` now reports `codex_config_path`, `context_window`, and `auto_compact_token_limit`.
- Context-window failures now map to Anthropic-style `400 invalid_request_error` responses.
- Streaming Anthropic requests now emit error SSE events when backend errors occur mid-stream.

### Validation
- Python: `PYTHONPATH=src pytest -q`
- Rust: `cargo test`
- TypeScript: `npm test && npm run build`
