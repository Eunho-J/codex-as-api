# Release Notes

## v0.9.3

### Prompt cache routing headers

- Send the session headers the official Codex CLI sends on every Responses
  request: `session-id`, `thread-id` and `x-client-request-id`, derived from
  `client_metadata.session_id` / `thread_id` or, failing those, from
  `prompt_cache_key`. Applies to streamed `/responses` and `/responses/compact`
  requests in Python, TypeScript and Rust. Values containing CR or LF are not sent.
- Measured against the ChatGPT Codex backend: an identical 7k-token prefix sent
  with `prompt_cache_key` alone read 0 cached tokens on every request; with the
  headers, repeated requests read 7,296 cached tokens. The official Codex CLI on
  the same account read ~30k cached tokens per turn.
- No change when a request carries neither `prompt_cache_key` nor
  `client_metadata.session_id`.

## v0.9.2

### TypeScript version lookup deadline

- Keep the Codex version lookup timeout alive until the lookup settles, including
  fetch implementations without active I/O handles. Clear the timer on completion.
- Fix Node 22 resolver timeout cancellation that blocked the v0.9.0 and v0.9.1
  Release workflows. This release includes their reasoning and security updates.

## v0.9.1

### TypeScript dependency security updates

- Update Express to the 5.3 series and smol-toml to the 1.9.1 series.
- Refresh the lockfile to body-parser 2.4.0, proxy-addr 2.0.8 and qs 6.16.1,
  addressing the reported request-parser, IP-trust and TOML-parser advisories.
- Keep Python, TypeScript and Rust package versions aligned. No API or reasoning
  continuity behavior changes are included in this patch.

## v0.9.0

### Encrypted reasoning continuity

- Carry Codex reasoning items in a versioned proxy-specific `thinking.signature`
  envelope and restore their opaque encrypted content on the following request.
  The envelope is transport metadata, not a fabricated Anthropic signature.
- Preserve one thinking block per reasoning item, including multiple summary
  parts and encrypted-only items. Streaming signatures precede their block stops;
  signed reasoning history stays out of ordinary assistant text.
- Preserve reasoning across the verified GPT-6.1-Sol, GPT-6-Astra, GPT-6-Sol,
  GPT-6-Luna and Daybreak Blue model combinations; authenticated replay passed
  all 25 combinations. Daybreak availability remains account/authorization-gated.

### Current client identity and experimental models

- Resolve outbound Codex client identity from npm `@openai/codex`'s stable latest
  version instead of a hardcoded source-reference version. Use a consistent
  version for model-catalog lookup and request headers, with bounded lookup,
  in-memory caching and last-successful-version retry behavior.
- Explicitly include verified hidden models `gpt-6-chat-thinking`, `gpt-reserve`
  and `codex-auto-review` in the effective catalog without choosing them as an
  automatic default. Prefer current live capability metadata.
- Document their experimental status: upstream support may end at any time;
  users should report a discontinued model in an issue so it can be removed.
  Basic authenticated conversation, function-call continuation and generated-code
  execution checks passed for all three.

## v0.8.3

### Client-neutral Anthropic compatibility (all runtimes)

- Remove Claude Code identity gates around valid required `max_tokens` and
  ephemeral `cache_control` hints on Messages, counting and compact routes.
  These are compatibility hints, not native Codex output caps or cache TTLs;
  optional Claude session cache affinity remains available.
- Include provisional zero usage in every initial streaming `message_start`,
  followed by authoritative final usage with the 0.8.2 disjoint cache accounting.
- Support summarized thinking as real unsigned thinking blocks/deltas, retaining
  existing Codex reasoning extensions. Omitted and disabled thinking suppress
  reasoning output; never fabricate an Anthropic signature.
- Resolve disabled-derived effort to the model's lowest supported level when
  `none` is unavailable; explicit effort selection remains capability-checked.
- Validate and strip replayed assistant thinking history for multi-turn clients.
  Accept standard `metadata.user_id` and boolean tool streaming hints without
  inventing backend semantics, and use the live catalog default for unconfigured
  `claude-*` facade routing.

### Official test quality

- Replace regex diagnostic assertions with literal returned-error checks and
  remove source-presence, filesystem-permission and worker-liveness oracles.
  Retain actual input/output, persistence-content and cancellation-state cases.
- Repair stale client-gate expectations and positive controls, including valid
  null cache hints, live-default Claude facade routing and mixed thinking/tool
  replay. Validate all three runtimes with their standard regression runners.

## v0.8.2

### Hotfix: disjoint Anthropic cache usage (all runtimes)

- Correct Anthropic Messages usage in streaming and non-streaming responses:
  `input_tokens` now excludes both cache reads and reported cache creation,
  which remain in their separate `cache_read_input_tokens` and
  `cache_creation_input_tokens` counters. Previously, consumers adding those
  counters counted cached input twice.
- Keep OpenAI/Responses totals unchanged. Preserve unknown cache counters as
  null; subtract only reported cache buckets. Reject impossible cache sums
  exceeding the upstream input total instead of emitting negative/clamped usage.
- Add no-cache, partial/full-cache, cache-write, invalid-total and HTTP pipeline
  regressions across TypeScript, Python and Rust. Example: 100 total input,
  50 cache read and 25 cache write becomes 25 uncached + 50 read + 25 write.

## v0.8.1

### Explicit Ultra Fast selection (all runtimes)

- Accept `speed: "ultrafast"` as a proxy extension on Anthropic Messages and
  compact routes and forward the literal Codex `service_tier: "ultrafast"`.
  Direct `service_tier` selection remains supported on Chat and Messages.
- Preserve live account/model tier validation: unavailable Ultra Fast is HTTP
  400 before inference/streaming headers, never a silent downgrade to priority.
  Reject conflicting speed/tier inputs; keep Fast and standard behavior unchanged.
- Add cross-runtime mapping and capability regression tests, including HTTP
  recording-backend tests. Malformed Python speed objects/arrays now return
  invalid-request errors instead of causing an unhashable-type failure.
- Keep TOML config imports type-checkable on current Python while preserving
  Python 3.10 support; include `tomli` in dev dependencies for the 3.10 analysis target.
- Document Ultra Fast separately from priority and reasoning effort. No model
  support, account entitlement, pricing or config speed default is invented.

## v0.8.0

### ChatGPT Live realtime voice (TypeScript)

- Add `POST /v1/realtime/calls` to the TypeScript runtime. A client-owned WebRTC
  SDP offer plus an optional `session` object (`instructions`, `voice`, ...) is
  exchanged for the ChatGPT backend's SDP answer; the call id is returned both in
  the mirrored `Location: /v1/realtime/calls/{call_id}` header and as
  `x-realtime-call-id`, and the answer body is `text/plain`, never JSON.
- Add `WS /v1/realtime/sideband/{call_id}`, which relays text and binary realtime
  frames to `wss://api.openai.com/v1/live/{call_id}` with the account's OAuth
  headers so headless clients (bots, server-side agents, observers) can attach to
  an existing call. The proxy does not proxy RTP; the media plane stays client-side.
- Send the exact call contract the backend authorizes for OAuth: `version`,
  `openai-alpha: quicksilver=v2`, `session-id`/`x-session-id`/`thread-id` carrying
  one minted identity, and `intent=quicksilver&architecture=avas`. Reject unknown
  top-level request fields and any client-chosen voice-session id (`id`,
  `session_id`, `voice_session_id`) instead of forwarding a call the backend
  denies.
- Default `model` to `gpt-live-1-codex`, `voice` to `cove`, and `delegation` to
  `{"type": "client"}`.
- Realtime is TypeScript-only in this release. The Python and Rust runtimes keep
  their existing HTTP surface, and the runtime difference is stated in
  `README.md` and `ts/README.md` rather than presented as parity.

## v0.7.0

### Live authenticated model catalog

- Remove every bundled runtime model catalog and use the authenticated Codex
  `/models` response as the sole authority for model membership and advertised
  capabilities, including context bounds, reasoning levels, service tiers, and
  Responses Lite behavior. Explicit Codex config context values remain local
  settings and are clamped when the live row publishes a maximum.
- Add `GET /v1/models` across Python, TypeScript, and Rust. Cache only fresh
  account/base-URL/client-version snapshots for five minutes, coalesce refreshes,
  and invalidate on a changed Responses `X-Models-Etag`.
- Resolve the live Codex default when no model is configured, require exact live
  slugs without public GPT aliases, and require an explicit live backend for
  `claude-*` facade names. Unknown models now fail instead of receiving guessed
  metadata or silently routing elsewhere.
- Pin the private wire contract and request identity to Codex `0.153.3` commit
  `b1a547b1f73ce86205d9222ac19cff334b3b7a2e`.

### Fail-loud request and response semantics

- Validate the catalog and effective model before streaming headers; make
  `/health` a real readiness endpoint and return typed authentication, catalog,
  model, request, upstream HTTP, and upstream protocol errors.
- Reject unsupported controls and malformed request/upstream structures rather
  than dropping values, coercing roles, or inventing objects or IDs. Derive
  terminal status only from emitted tool calls and the authoritative
  `response.completed` event: missing, null, or true `end_turn` is terminal in
  pinned Codex, while explicit false remains nonterminal. Duplicate provider
  function/custom call IDs now fail
  as protocol errors instead of being merged or emitted ambiguously, and
  unrepresentable provider custom-tool call events fail instead of being folded
  into function calls.
- Remove fabricated Anthropic thinking blocks and signature deltas. Real Codex
  reasoning is exposed only as `codex_reasoning` or streamed
  `codex_reasoning_delta` proxy extensions.
- Reject hosted Anthropic WebSearch on Messages, token-counting, and compact
  requests because OpenAI hosted results cannot provide the complete Anthropic
  server-tool lifecycle. Unexpected provider web-search calls fail as upstream
  protocol errors instead of being translated into incomplete result blocks.
- Require authoritative final usage and completion semantics for Anthropic
  responses. Emit required nullable response fields from provider data, while
  retaining immediate Claude Code `message_start` delivery without guessed
  token counts.
- Scope process-local `previous_response_id` history to the authenticated account
  and reject known `comp_hash` mismatches instead of replaying incompatible
  history without Codex's automatic pre-turn compaction. Add finite provider
  timeouts while retaining the narrow documented
  managed-ChatGPT OAuth, Anthropic `max_tokens`, and validated `cache_control`
  compatibility behavior.
- Accept only the official nested `tokens` layout for managed ChatGPT auth.
  Remove unofficial root-level token fields and noncanonical `auth_mode` aliases,
  and reject external-host `chatgptAuthTokens` because the standalone proxy has
  no host refresh callback.

### Release integrity

- Make FastAPI, Pydantic, and Uvicorn regular Python dependencies so the
  advertised `pip install codex-as-api && codex-as-api` path installs a working
  server instead of an optional-dependency error shim.
- Build npmjs and scoped GitHub Packages tarballs in the tested TypeScript job,
  then publish those exact artifacts instead of rebuilding after the release
  gate. Require both tarballs to carry the repository Apache license and
  third-party notices.
- Package each platform Rust binary in a deterministic ZIP archive containing
  the binary, repository `LICENSE`, and generated notices covering bundled
  tokenizer data plus the actual shipped license files for the locked
  cross-platform runtime dependency graph.
- Reject catalog-like JSON assets and payloads from runtime source trees and
  package artifacts, and run the complete Python, TypeScript, and Rust test,
  lint, build, and package gates.

## v0.6.5

### Codex 0.147 compatibility

- Align GPT-5.6 alias, Sol, Terra, and Luna context limits with the stable
  Codex 0.147 catalog at 272,000 tokens. Automatic compact defaults now derive
  to 244,800 tokens.
- Pin a structured compatibility fixture to `openai/codex` 0.147.0 commit
  `be6e8eac029b183056b7e4402879f15d2c85f61b`, covering the model fields,
  Responses request path, Lite behavior, and originator header. The fractional
  `codex_rollout_budget_units` usage field is tracked as upstream-only because
  this facade has no approved public mapping for it.
- Replace the moving startup `npm view @openai/codex version` lookup with a
  deterministic Codex compatibility baseline and `codex-as-api` package
  version in the request `User-Agent`.

### Authentication and errors

- Refresh OAuth access tokens five minutes before expiry, coalesce concurrent
  refresh work, and reload credentials changed by another request before
  rotating a refresh token after HTTP 401.
- Preserve the existing root or nested auth-file layout, reject account
  switches and conflicting token claims, and compare the credential snapshot
  again before persisting a refresh response.
- Preserve upstream HTTP statuses, including rate-limit and overload responses,
  in Python and TypeScript to match Rust.

### CI and publishing

- Add pull-request and push CI for Python tests, Ruff, mypy, package builds,
  TypeScript tests/build/package checks, and Rust format, Clippy, tests, and
  release builds.
- Add a tag-triggered release workflow that requires exact tag/package version
  agreement and the full cross-runtime gate before publishing.
- Publish PyPI and npmjs through trusted-publisher OIDC, GitHub Packages and
  GitHub Releases through the automatic repository `GITHUB_TOKEN`, and attach
  Python/npm distributions plus Linux, macOS, and Windows Rust binaries.
- Remove the manual npm workflow that could report success after silently
  skipping npmjs publication when `NPM_TOKEN` was absent.

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
