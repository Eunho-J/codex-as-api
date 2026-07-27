# Release Notes

## v0.6.4

### Codex identity and cache affinity

- Remove process-global generated `session_id` and `thread_id` values from
  Python, TypeScript, and Rust. Codex metadata mode now requires an explicit
  non-empty `client_metadata.session_id`, defaults a root `thread_id` to that
  session, preserves an explicit child thread, and refreshes only turn-scoped
  identity on each request.
- Resolve Chat `prompt_cache_key` as explicit value, then non-empty
  `client_metadata.session_id`, then omission. This matches Codex 0.145.0 and
  current `main`, where the official default is Responses metadata
  `session_id`, not `thread_id`.
- Retain bounded process-local Chat `previous_response_id` history replay over
  private HTTP. The ID remains local and is never forwarded upstream.

### Claude Code session and cache compatibility

- Read `x-claude-code-session-id` on `/v1/messages` and derive a stable,
  privacy-safe SHA-256 `prompt_cache_key`; an explicit proxy extension
  `prompt_cache_key` takes precedence.
- Keep normal Anthropic Messages stateless: reject non-null
  `previous_response_id`, disable ambient Codex metadata, avoid fabricated
  Claude thread IDs, and remove Python's nonstandard Anthropic `response_id`.
- Validate Anthropic `cache_control` hints at request, system, message, content,
  and tool locations. Accept only `type: "ephemeral"` with optional TTL `5m` or
  `1h`, then strip the hint because the private Codex transport cannot preserve
  Anthropic cache breakpoints or TTL behavior.

### Validation

- Python: 515 tests, mypy, and changed-file Ruff checks.
- TypeScript: 261 tests and production build.
- Rust: 253 tests and rustfmt.
- Re-audited `@openai/codex` 0.145.0 plus `main` at
  `bd2de422aa287b97b06ca6425a10935bcf1b3731`, and reproduced the Claude Code
  2.1.220 request shape.

## v0.6.3

### Claude Code WebSearch/WebFetch auxiliary calls

- Fix HTTP 400 responses when a Claude Code session with ambient effort sends a WebSearch/WebFetch auxiliary request containing both `output_config.effort` and call-level `thinking: {"type":"disabled"}`.
- Give explicit disabled thinking precedence over valid ambient effort and forward Codex reasoning effort `none` across Python, TypeScript, and Rust.
- Require and document `CODEX_AS_API_RESPONSES_LITE=off` for Claude Code hosted WebSearch on GPT-5.6 because official Responses Lite relies on a standalone `web.run` executor; WebFetch does not require this override.
- Continue rejecting empty, non-string, or unsupported effort values and unsupported `output_config` fields rather than silently discarding malformed requests.

### Validation

- Adapter tests cover every valid Claude Code effort value paired with disabled thinking while retaining invalid-value failures.
- Streamed `/v1/messages` tests across all three runtimes and a local `/count_tokens` test reproduce the Claude Code 2.1.209 request shape and prove provider-backed requests reach the classic Codex wire with reasoning effort `none`.
- A real GPT-5.6 Sol Codex OAuth WebSearch completed with HTTP 200 and structured `server_tool_use`, `web_search_tool_result`, and terminal SSE events.

## v0.6.2

### Claude Code count_tokens autocompact fix

- Fix the Anthropic `/v1/messages/count_tokens` estimator treating every UTF-8 byte as a token and then adding the full raw request body a second time.
- Port official `tiktoken` `o200k_base` ordinary encoding into Python, TypeScript, and Rust: the same Unicode pre-tokenization regex, byte-pair merge algorithm, and merge ranks now count GPT-5-family text.
- Count tool calls, tool-result identifiers, reasoning content, and normalized tool schemas once while excluding model IDs, output limits, stream flags, metadata, and other non-model request controls.
- Keep the existing image estimate separate so URL and inline base64 images are counted once without tokenizing the base64 payload as text.
- Keep token counting local and OAuth-free. No `tiktoken` package dependency or first-run encoding download is introduced.
- Record the upstream synchronization point as `tiktoken` 0.13.0 commit `08a5f3b2c987ada4fc5aa1f16c643c203fa8acaa`, checked on 2026-07-14, with the official rank-file SHA-256 verified during release validation.

### Validation

- A 4,000-byte ASCII message returns `1,012` input tokens in all three implementations instead of roughly `8,000`.
- Endpoint tests cover UTF-8 text, normalized tools, tool/reasoning history, URL and base64 images, ignored request controls, effective model limits, and zero upstream provider calls.
- Official `tiktoken` 0.13.0 token IDs match the bundled ports across reference vectors, randomized multilingual text, code, whitespace, contractions, special-token-shaped literals, and long single-piece input.
- Python, TypeScript, and Rust return identical counts across 204 cross-runtime endpoint cases; Claude Code 2.1.209 real Codex OAuth chat remains compatible.

## v0.6.1

### Claude Code 2.1.208 compatibility

- Re-audit official `openai/codex` `main` at `393f64565ab46f09d99ca4d9bd973537e72a114b` after the `0.144.4` release; the private HTTP/WebSocket request split still has no Pro, public cache-policy, or safety-identifier aliases.
- Route bundled GPT IDs received through `/v1/messages` to the matching Codex backend model while preserving the configured Codex fallback for built-in Claude model names.
- Route the same model and Fast Mode controls through remote compaction, and report token-count context limits for the effective request model.
- Map Claude Code `output_config.effort` values through `max`, and translate Fast Mode's `speed: "fast"` into the Codex `priority` service tier.
- Accept the current no-op `clear_thinking_20251015` context cleanup, and reject context edits, task budgets, or enabled beta tool fields that cannot be represented by the Codex OAuth transport.
- Preserve Anthropic URL image sources and `tool_result.is_error` state instead of silently discarding them; reject malformed output formats and unsupported image source types.
- Move Python's blocking provider iterator off the ASGI event loop and replace Rust's buffered Anthropic path with end-to-end incremental SSE.
- Preserve Rust streaming 401, 429, and 529 semantics and surface OAuth refresh failures directly.
- Document the official custom model option that appends GPT to the `/model` picker alongside Fable, Opus, Sonnet, and Haiku, including effort controls and the process-wide gateway routing limitation.
- Restrict Python source distributions to the Python source, shared capability catalog, tests, and release documentation.

### Validation

- Latest Claude Code CLI `2.1.208`: real Codex OAuth chat, two-turn `Read` tool loop, explicit `gpt-5.6-sol`, and `--effort max`.
- Python concurrency and TypeScript/Rust gated-stream integration tests prove that a live stream does not block or buffer other Claude Code work.
- Python, TypeScript, and Rust unit and integration suites cover the captured latest request shape and unsupported-field failures.

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
