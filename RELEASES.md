# Release Notes

## v0.3.2

### Claude Code compatibility fix
- Restored immediate Anthropic streaming so clients receive `message_start` before the backend response completes.
- `/v1/messages/count_tokens` now falls back to a local estimate when the Codex backend rejects zero-token counting with `Unsupported parameter: max_output_tokens`.
- Keeps real final streaming usage in `message_delta` while avoiding stream buffering.

### Validation
- Python: `PYTHONPATH=src pytest -q`
- Rust: `cargo test`
- TypeScript: `npm test && npm run build`
- Live smoke: `/v1/messages/count_tokens` and streaming `/v1/messages` against local server

## v0.3.1

### Real Anthropic token accounting
- `/v1/messages/count_tokens` now asks the Codex backend for real input-token usage with `max_output_tokens: 0` instead of returning a local estimate.
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
