# codex-as-api

[![GitHub Release](https://img.shields.io/github/v/release/Eunho-J/codex-as-api)](https://github.com/Eunho-J/codex-as-api/releases)
[![PyPI](https://img.shields.io/pypi/v/codex-as-api)](https://pypi.org/project/codex-as-api/)
[![npm](https://img.shields.io/npm/v/codex-as-api)](https://www.npmjs.com/package/codex-as-api)
[![License](https://img.shields.io/github/license/Eunho-J/codex-as-api)](LICENSE)

Use ChatGPT / Codex OAuth as a local OpenAI-compatible API server.

## Features

- **OpenAI & Anthropic compatible** — Chat Completions, Messages, and `GET /v1/models`
- **Claude Code ready** — use Codex models directly from Claude Code CLI
- **Streaming** — full SSE streaming for both OpenAI and Anthropic protocols
- **Tool calling** — function calls, tool results, and parallel tool calls
- **Image support** — generation, inspection, multimodal Chat input, and capability-gated `original` image detail in classic Responses
- **Reasoning** — capability-checked effort/context, persisted reasoning, and explicit Codex reasoning extensions without fabricated Anthropic signatures
- **Codex features** — live authenticated model discovery, session-aware `prompt_cache_key`, account-scoped Chat continuation, subagent headers, and remote compaction
- **Codex config aware** — reads `CODEX_HOME` / `~/.codex/config.toml` for model, reasoning-effort, and context-window settings
- **Token estimate & compaction helpers** — Anthropic-compatible `/v1/messages/count_tokens` and `/v1/messages/compact`
- **Standalone ChatGPT login** — ordinary browser OAuth by default, optional device-code login for remote servers, a private application-owned credential store, and managed refresh; no installed Codex CLI or existing Codex login required
- **3 implementations** — Python, TypeScript (npm), and Rust — identical behavior

## What it does

Runs a lightweight HTTP server on `localhost` that translates standard OpenAI API calls into authenticated requests against the ChatGPT / Codex backend. Sign in explicitly through this server's browser OAuth API; it stores and refreshes its own OAuth credentials without reading or changing your Codex login.

Python, Rust, and TypeScript (npm) implementations are provided — identical functionality, same endpoints, same behavior.

## Prerequisites

A ChatGPT account eligible for the Codex backend and a runtime for your chosen
implementation are required. The official Codex CLI is **not** required.

After starting the server, use [Standalone login](#standalone-login) to open
OpenAI's ordinary sign-in page in your browser and authorize this server. Credentials are saved privately to
`~/.codex-as-api/auth.json` by default, and the server handles refresh itself.
It does not discover, import or fall back to `~/.codex/auth.json`, `CODEX_HOME`
credentials or a Codex keychain entry. Upgrading from older versions therefore
requires a fresh login for codex-as-api; your existing Codex login is left alone.

Use one server process per credential store. To run independent instances,
assign each its own `CODEX_AS_API_HOME` or `CODEX_AS_API_AUTH_PATH`.
Model settings may still be read from `~/.codex/config.toml`; that separate
configuration behavior is unchanged and does not require installing Codex.

## Install & Run

### Python

Install from PyPI:

```bash
pip install codex-as-api
codex-as-api
```

Or with `uv`:

```bash
uv pip install codex-as-api
codex-as-api
```

Or from source:

```bash
git clone https://github.com/Eunho-J/codex-as-api.git
cd codex-as-api
pip install -e .
codex-as-api
```

### Rust

```bash
cd rust
cargo build --release
./target/release/codex-as-api
```

### TypeScript (npm)

Install from npm and run:

```bash
npm install -g codex-as-api
codex-as-api
```

Or use `npx` without installing:

```bash
npx codex-as-api
```

Or from source:

```bash
cd ts
npm install
npm run build
node dist/cli.js
```

Can also be used as a library:

```typescript
import { ChatGPTOAuthProvider, createApp } from "codex-as-api";

// Use the provider directly
const provider = new ChatGPTOAuthProvider({ model: "gpt-5.6-sol" });
const response = await provider.chat(
  [
    { role: "system", content: "You are helpful." },
    { role: "user", content: "Hello!" },
  ],
);
console.log(response.content);

// Or create an Express app
const app = createApp();
app.listen(18080);
```

All versions bind to `127.0.0.1:18080` (localhost only) by default.

## Standalone login

Login is an API operation; the `codex-as-api` command starts the server, not the
Codex CLI. Before starting, set `CODEX_AS_API_ADMIN_KEY` to a private random
administrator key. Supply that same key as a bearer credential on `/auth/*`.
This key is independent of both the OpenAI OAuth credentials and any inference
client authentication; never use your access or refresh token as the admin key.

1. Start the server with the key configured, then initiate login from a trusted
   local terminal or authenticated management client:

   ```bash
   curl -sS -X POST http://127.0.0.1:18080/auth/login \
     -H "Authorization: Bearer $CODEX_AS_API_ADMIN_KEY" \
     -H 'Content-Type: application/json' -d '{}'
   ```

2. The `202` response contains `id`, `method: "browser"`, `status: "pending"`,
   `auth_url` and `expires_at`. Your app opens `auth_url` in the system browser;
   sign in normally on OpenAI's page. No one-time device code or Codex CLI is
   needed. On approval, the browser returns to a separate loopback callback
   listener on the **same machine as the server**, and credentials are saved
   privately. The server tries registered callback port 1455, then 1457; if both
   are occupied it reports `409 callback_port_unavailable` instead of interfering
   with another app. The API itself never launches a browser on a remote server.
   The authorization URL contains temporary sensitive state; do not log or share
   it. Only start login on a server you control.

3. Poll the login ID returned by step 1 (the placeholder below is not a real ID):

   ```bash
   curl -sS http://127.0.0.1:18080/auth/login/LOGIN_ID \
     -H "Authorization: Bearer $CODEX_AS_API_ADMIN_KEY"
   ```

   Status becomes `succeeded` only after valid credentials are durably saved.
   Other terminal states are `failed`, `cancelled` and `expired`. Routine status
   responses include the method but never repeat the authorization URL or user
   code, or return OAuth tokens, authorization codes, verifiers, device IDs or
   storage paths.

4. To stop an outstanding attempt, send `DELETE /auth/login/LOGIN_ID` with the same
   administrator bearer. Repeating cancellation is safe. A login already saved
   remains `succeeded`; cancellation is not logout or provider-side revocation.

`GET /auth/status` with the administrator bearer returns only
`{"status":"authenticated"}`, `unauthenticated` or `reauthentication_required`.
This checks the local application-owned credential state, not provider-side
revocation or remote account validity; refresh is handled privately when needed.
Validation, cancellation and failures before credential replacement preserve a
prior valid login. If replacement succeeds but the final directory sync fails,
`login_persistence_uncertain` reports that durability could not be confirmed; a
new local credential may already exist. Cancellation is not a rollback.

### Explicit device login for remote/headless use

When the browser and API server are on different machines, a loopback browser
callback would reach the wrong machine. Explicitly send `{"method":"device"}`
to `POST /auth/login` instead. It returns `method: "device"`, `verification_url`,
`user_code`, `interval` and the common session fields. Open
`https://auth.openai.com/codex/device` in your own browser and enter the code.
Only approve a code from a login you initiated on a trusted server; never post
it in public channels or logs. Device-code login may need enabling in your
OpenAI account. No method silently falls back to another.

All management responses use `Cache-Control: no-store`. An unset admin key returns
`503 management_disabled`; a missing/wrong bearer returns `401 invalid_admin_key`.
An empty request body, `{}` or `{"method":"browser"}` selects browser login;
`{"method":"device"}` selects device login. No other body fields are accepted.
The API does not accept credentials to import, token export switches,
authentication endpoints or credential paths from a request. One active attempt per store is allowed; another
start returns `409 login_in_progress`. Unknown/retired IDs return `404`.

Keep the default loopback binding unless you deliberately deploy authenticated
HTTPS administration behind a trusted reverse proxy. A browser `Origin` must
match the request's own scheme and Host; the API does not blindly trust forwarded
headers. Configure proxy origins consistently. Origin checking supplements the
admin bearer and is not a replacement for authentication or network controls.
Each credential store must be owned by one server process, not shared between
independent workers. Choose separate store paths for separate instances.

## Configuration

Environment variables (Python, Rust, and TypeScript):

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEX_AS_API_HOST` | `127.0.0.1` | Bind address |
| `CODEX_AS_API_PORT` | `18080` | Listen port |
| `CODEX_AS_API_MODEL` | `~/.codex/config.toml` `model`, else live default | Model identifier passed to Codex backend |
| `CODEX_AS_API_AUTH_PATH` | `CODEX_AS_API_HOME/auth.json`, else `~/.codex-as-api/auth.json` | Explicit application-owned OAuth store path; not an import command |
| `CODEX_AS_API_HOME` | `~/.codex-as-api` | Application-owned credential directory |
| `CODEX_AS_API_ADMIN_KEY` | unset (management disabled) | Independent bearer key for `/auth/*`; never use an OAuth token as this key |
| `CODEX_AS_API_RESPONSES_LITE` | `auto` | Responses Lite mode: `auto`, `on`, or `off` |
| `CODEX_AS_API_CODEX_METADATA` | `off` | Add Codex-style per-turn `client_metadata` and related backend headers |
| `CODEX_HOME` | `~/.codex` | Optional Codex model/config discovery only; never used to find credentials |

The server also reads root-level Codex CLI settings from `~/.codex/config.toml`:

```toml
model = "gpt-5.6-sol"
model_reasoning_effort = "high"

# Optional explicit context values. A published live maximum still clamps them.
model_context_window = 272000
model_auto_compact_token_limit = 244800
```

`CODEX_AS_API_MODEL` overrides the Codex config model. Reasoning-effort precedence is request, Codex config, then the authenticated model catalog default when present. When no request or configured model exists, the proxy selects the catalog's first `visibility: "list"` model by ascending priority, matching Codex. The effective model, reasoning setting, catalog status, and context settings are exposed from `/health`; optional reasoning and context values are `null` when neither config nor the live catalog supplies them. Context settings are also returned by Anthropic token-count responses when available.

### Supported Models

Ordinary models and capabilities come from the authenticated Codex `/models` response. The proxy keeps each account/base-URL/client-version snapshot in memory for 300 seconds and invalidates it when a Responses `X-Models-Etag` changes. An unavailable or malformed refresh returns `503 catalog_unavailable` before streaming headers are sent; an upstream HTTP rejection after the one allowed OAuth refresh preserves its status as `upstream_error`. Model snapshots are never shared across accounts or served as an offline fallback.

Use `GET /v1/models` for the effective catalog. It includes exact slugs, visibility, `supported_in_api`, reasoning-summary controls, and opaque `comp_hash` compatibility identifiers when supplied. Unknown IDs return `404 model_not_found`. Automatic default selection uses only the upstream catalog's originally visible models, not the experimental additions below.

#### Experimental hidden models

These three upstream-hidden models returned real responses on October 24, 2026, and are explicitly included in codex-as-api's effective catalog:

| Model ID | Observed upstream purpose |
| --- | --- |
| `gpt-6-chat-thinking` | Server-described paid CCA experiment; hidden and marked unsupported in the upstream API catalog |
| `gpt-reserve` | Official Codex CLI's Luna Reserve routing for reserve usage |
| `codex-auto-review` | Official Codex CLI's Guardian automatic approval-review model |

A small authenticated smoke test also verified function-tool calling plus tool-result continuation for all three models, and executed a simple generated JavaScript function against four inputs per model. This is a basic functionality check, not a coding-quality benchmark.

**These are hidden models. Upstream support can stop at any time, and availability may differ by account. If a model stops responding, [open an issue](https://github.com/Eunho-J/codex-as-api/issues) with its model ID and a sanitized error; we will remove the discontinued model from this catalog.** Do not include access tokens or other credentials in the issue.

The proxy exposes these verified exceptions as selectable models and prefers their current live capability metadata. Bundled capability records keep only these three IDs available when a successful live catalog omits them; they do not bypass OAuth, upstream rejection, or a failed/malformed catalog fetch. A successful empty upstream catalog can therefore list these experimental entries, while health/default selection still fails until an ordinary upstream default is available. The proxy never selects a hidden experimental model as its automatic default.

To use a different port:

```bash
CODEX_AS_API_PORT=9000 codex-as-api
```

To expose on all interfaces (e.g. for remote access):

```bash
CODEX_AS_API_HOST=0.0.0.0 codex-as-api
```

## API Endpoints

### `GET /v1/models`

Returns the current account's authenticated Codex model catalog using an OpenAI-compatible list envelope. The response includes safe routing and capability metadata but omits backend instructions and model messages.

```bash
curl http://localhost:18080/v1/models
```

### `POST /v1/chat/completions`

Standard OpenAI chat completions. Supports streaming (`stream: true`) and non-streaming.

```bash
curl http://localhost:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello"}
    ]
  }'
```

Streaming:

```bash
curl http://localhost:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello"}
    ],
    "stream": true
  }'
```

With tools:

```bash
curl http://localhost:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [
      {"role": "system", "content": "You have access to tools."},
      {"role": "user", "content": "What is the weather in Seoul?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get current weather",
          "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
          }
        }
      }
    ]
  }'
```

`reasoning_content` is a response-only proxy extension. Chat request messages
cannot replay that field losslessly and reject it instead of silently dropping
it; continuation uses the complete provider Response items saved behind
`previous_response_id`.

### `POST /v1/messages`

Anthropic Messages compatibility endpoint for any client, including Claude Code and Scheherazade. It supports streaming (`stream: true`) and non-streaming without a client-identification header. A live Codex model ID routes directly. A `claude-*` facade ID uses the configured Codex backend model, or the authenticated catalog default when no backend model is configured. Other unknown IDs fail. The response preserves the client-supplied model name.

`max_tokens` is a required positive compatibility hint: the private Codex OAuth transport does not expose a matching output-token cap. Valid ephemeral `cache_control` hints are accepted and stripped for every client; their TTL and breakpoint semantics are not forwarded. An optional `x-claude-code-session-id` provides stable hashed cache affinity, not permission to use the endpoint. `metadata.user_id` is accepted as client metadata and is not forwarded.

```bash
curl http://localhost:18080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: unused" \
  -H "anthropic-version: 2023-06-01" \
  -H "x-claude-code-session-id: example-session" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 200,
    "system": "You are a helpful assistant.",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

Streaming:

```bash
curl -N http://localhost:18080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: unused" \
  -H "anthropic-version: 2023-06-01" \
  -H "x-claude-code-session-id: example-session" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 200,
    "stream": true,
    "system": "You are a helpful assistant.",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

With `thinking.display: "summarized"` (the enabled/adaptive default), real Codex reasoning summaries are returned as Anthropic `thinking` blocks and streamed `thinking_delta` content, alongside the existing `codex_reasoning` / `codex_reasoning_delta` extensions. When upstream supplies encrypted reasoning, each reasoning item has its own proxy-specific `signature`: `codex-reasoning-v1:` followed by base64url JSON containing the format version, actual backend model ID, and original reasoning item ID, encrypted content and summary. This is an opaque continuation envelope, not an Anthropic cryptographic signature.

On the following assistant-history request, recognized envelopes restore the original Responses reasoning items before that assistant turn's text/tool calls. Historical signed reasoning is accepted even when the current call uses `thinking: {"type":"disabled"}`, including compaction calls: current output-generation settings do not invalidate prior reasoning input. Ciphertext is preserved unchanged; summary text is never converted to ordinary assistant speech. Missing or foreign signatures retain the legacy ignored-thinking behavior, while malformed recognized envelopes return HTTP 400. Reasoning is retained for the same backend model and across the verified `gpt-6.1-sol` / `gpt-6-astra` / `gpt-6-sol` / `gpt-6-luna` / `gpt-daybreak-blue-latest` combinations (25 authenticated source/target pairs); other cross-model envelopes are discarded while ordinary history remains.

Multiple summary parts share one thinking block per reasoning item. Encrypted-only items use an empty thinking string with a non-empty signature. Streaming sends `signature_delta` before the matching `content_block_stop`. `display: "omitted"` hides the summary but can preserve an encrypted-only signed block for continuity; disabled thinking suppresses thinking output. Items without encrypted content retain unsigned-summary compatibility behavior.

Hosted Anthropic `web_search` declarations are rejected with HTTP 400 on
Messages, token-counting, and compact requests. OpenAI Responses cannot supply
the Anthropic server-tool provenance and encrypted lifecycle fields required to
translate a hosted result losslessly, so the gateway does not approximate that
protocol. Provider output containing an unexpected hosted web-search call is an
upstream protocol error instead of a fabricated Anthropic result block.

Completed responses require authoritative final token usage and a real
`response.completed` event. Emitted function calls map to `tool_calls` or
`tool_use`; otherwise a missing, `null`, or `true` `end_turn` maps to `stop` or
`end_turn`, matching pinned Codex, where only explicit `false` requests another
sampling turn. An explicit `false` completion without a representable client
tool call is an upstream protocol error because this proxy does not implement
Codex's internal follow-up loop. Non-streaming requests return HTTP 502 when
the completion is nonterminal or final usage is absent; a stream that has
already started emits an Anthropic `error` event and never emits a successful
`message_stop`. The final response includes every required usage field, using
`null` only where the Anthropic type permits it, plus required container,
citation, context-management, and direct-caller fields. `message_start` is sent immediately with numeric zero counters in `message.usage`; these are provisional, not measured counts. The final `message_delta.usage` replaces them with authoritative cumulative totals, including disjoint cache buckets.

Generated `tool_use` blocks with `caller: {"type":"direct"}` and empty
assistant content arrays can be replayed through the request adapter. Malformed,
null, or server-tool caller variants still fail before transport.

### Identifier and cache behavior

The two compatibility facades intentionally use the similarly named fields for
different jobs:

| Input or output | `/v1/chat/completions` | `/v1/messages` |
|---|---|---|
| `client_metadata.session_id` | Forwarded to Codex, used as the default `prompt_cache_key`, and required when Codex metadata mode is enabled | Not used or forwarded |
| `client_metadata.thread_id` | Codex metadata thread identity; defaults to `session_id` for a root request and preserves an explicit child thread | Not used or synthesized |
| `prompt_cache_key` | Explicit value wins; otherwise a non-empty `client_metadata.session_id` is used; otherwise omitted | Optional proxy extension; explicit value wins, otherwise the Claude Code session header is hashed |
| `x-claude-code-session-id` | Not used | Optional cache-affinity header; its exact value supplies a hashed default key |
| `previous_response_id` | Resolves a known entry from the 256-chain process-local history and replays full input/output over HTTP; never forwarded | Non-null values return Anthropic-style HTTP 400 because normal Messages is stateless |
| `cache_control` | Not an OpenAI Chat control | Accepted for every client, validated, then stripped before Codex transport |

Anthropic `cache_control` is a cache-boundary annotation, not an identifier.
This proxy accepts only `{"type":"ephemeral"}` with optional `ttl: "5m"` or
`"1h"` at the top level and on supported system/content blocks and tools. The
private Codex request has no Anthropic breakpoint or TTL fields, so accepted
hints do not reproduce Anthropic caching semantics. Normal `/v1/messages`
responses do not expose the Chat-only `response_id`. The custom
`/v1/messages/compact` endpoint retains its documented local
`previous_response_id` support.

The Anthropic type permits `cache_control: null`; the proxy treats that exact
value as omission. A non-null cache-control object is validated in full for every client.

### `POST /v1/messages/count_tokens`

Anthropic-compatible token counting helper. Codex OAuth does not expose a count-only endpoint equivalent to Anthropic's native API, so this route counts text locally and returns context-window metadata for the effective backend model. Normalized model-visible messages, tool calls, tool-result metadata, reasoning, and tool schemas are counted once. Request-envelope fields and the `max_tokens` compatibility hint are excluded; other unsupported generation controls fail instead of becoming no-ops. Image inputs use a separate fixed estimate so inline base64 data is not counted as text.

`max_tokens` and `cache_control` are optional client-neutral compatibility hints on this route; no Claude Code header is required.

Text estimation uses a bundled port of official `tiktoken` `o200k_base` `encode_ordinary`: the same Unicode pre-tokenization regex, byte-pair merge algorithm, and merge-rank data are implemented in Python, TypeScript, and Rust. It is a local estimator, not model metadata and not a substitute for a provider count API. The project does not depend on a `tiktoken` package or download encoding data at runtime. The last upstream synchronization check was **2026-07-14**, against [`tiktoken` 0.13.0 at `08a5f3b`](https://github.com/openai/tiktoken/tree/08a5f3b2c987ada4fc5aa1f16c643c203fa8acaa); the bundled rank file SHA-256 is `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`.

Official Codex also has a [`ceil(UTF-8 bytes / 4)` truncation helper](https://github.com/openai/codex/blob/bd2de422aa287b97b06ca6425a10935bcf1b3731/codex-rs/utils/string/src/truncate.rs#L4-L84), but Codex documents the history estimate using that helper as [a coarse lower bound rather than a tokenizer-accurate count](https://github.com/openai/codex/blob/bd2de422aa287b97b06ca6425a10935bcf1b3731/codex-rs/core/src/context_manager/history.rs#L162-L186). This endpoint therefore uses exact `o200k_base` ordinary text tokenization instead. The complete request count remains an estimate because protocol-wrapper overhead and image cost are local constants, but the former byte-as-token and raw-payload double count that could overstate ordinary Claude Code requests by about 8x is removed.

```bash
curl http://localhost:18080/v1/messages/count_tokens \
  -H "Content-Type: application/json" \
  -H "x-api-key: unused" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-sonnet-4-6",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### `POST /v1/messages/compact`

Anthropic-compatible alias for remote conversation compaction. Accepts Anthropic Messages-shaped bodies and returns compacted checkpoint content. Live GPT model IDs select the matching backend model, and Claude Code effort and Fast Mode controls use the same mappings as `/v1/messages`. Its required non-forwarded `max_tokens` and any valid `cache_control` hints are accepted for every client; the optional Claude Code session header only supplies cache affinity.

### `POST /v1/images/generations`

Generate images via the Codex image generation tool.

```bash
curl http://localhost:18080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "prompt": "a futuristic city at sunset",
    "size": "auto"
  }'
```

### `POST /v1/inspect`

Inspect images with a text prompt (custom endpoint).

```bash
curl http://localhost:18080/v1/inspect \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Describe what you see",
    "images": [{
      "image_url": "data:image/png;base64,iVBORw0KGgo...",
      "detail": "original"
    }],
    "responses_lite": false
  }'
```

### `POST /v1/compact`

Compact a conversation into a checkpoint for continuation (custom endpoint). `/v1/messages/compact` provides the Anthropic-compatible alias.

Compact accepts the existing private Codex `reasoning_effort` (or matching `reasoning.effort`) plus `prompt_cache_key`, supported `service_tier`, `text`, and top-level `verbosity`. A known process-local `previous_response_id` is resolved to its saved Response items and replayed as full compact input; the field is never forwarded to private Codex. Non-null caller-supplied `reasoning.mode` / `reasoning.context`, `prompt_cache_options`, cache breakpoints, encrypted-reasoning `include`, and deprecated `prompt_cache_retention` are rejected instead of being silently dropped; explicit `null` for those nullable compatibility controls is treated as omitted. Any supplied `safety_identifier`, including `null`, returns HTTP 400. When Responses Lite is active, the proxy follows the official private Codex builder and adds `reasoning.context: "all_turns"` on the compact wire.

```bash
curl http://localhost:18080/v1/compact \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Summarize our conversation so far."},
      {"role": "assistant", "content": "We discussed the project architecture."}
    ]
  }'
```

### `GET /health`

Readiness check. It returns `200` only when OAuth is available, a fresh catalog can be loaded, and the effective model resolves. Optional catalog metadata is not a readiness requirement: `reasoning_effort`, `context_window`, and `auto_compact_token_limit` are `null` when the selected live row and explicit config do not provide them. Otherwise the route returns `503` with a typed, sanitized error instead of reporting a false healthy state. The payload exposes catalog timing and effective model settings, but never access tokens, account IDs, auth paths, upstream ETags, or upstream response bodies.

```bash
curl -i http://localhost:18080/health
```

### `POST /v1/realtime/calls` (TypeScript)

Creates a ChatGPT Live (realtime voice) call on the OAuth-authenticated ChatGPT
backend. The client owns the media plane: it sends its own WebRTC SDP offer and
applies the returned answer, so no audio is relayed through this proxy.

The body is `{"sdp": "<offer>", "session": {...}}`. `session` may carry
`instructions`, a `voice` (default `cove`), and any additional ChatGPT Live
session field; `model` defaults to `gpt-live-1-codex`, and `delegation` defaults
to `{"type": "client"}`. The route rejects unknown top-level fields and any
client-chosen voice-session id (`id`, `session_id`, `voice_session_id`), because
the backend denies calls that set those itself. On success it returns `201` with
the SDP answer as `text/plain` and a mirror `Location: /v1/realtime/calls/{call_id}`.

A session with `"type": "realtime"` creates a standard Realtime call instead
(`gpt-realtime-1.5` by default, voice `marin`): it carries function `tools` and
`tool_choice` rather than `delegation`, and the proxy sends it to the backend's
bare `/realtime/calls` route without the ChatGPT Live intent, architecture or
`openai-alpha` header. Library callers can also pass `mode: "realtime"` to
`createRealtimeCall`. Tool calls and their outputs then follow the standard
Realtime events on the sideband (`response.output_item.done` with a
`function_call` item, `conversation.item.create` with a `function_call_output`).

```bash
curl -i http://localhost:18080/v1/realtime/calls \
  -H "Content-Type: application/json" \
  -d '{"sdp": "v=0\r\n...", "session": {"instructions": "Reply briefly."}}'
```

### `WS /v1/realtime/sideband/{call_id}` (TypeScript)

Relays a WebSocket to the OpenAI realtime sideband of an already-created call
(`wss://api.openai.com/v1/live/{call_id}`) with the account's OAuth headers, so a
headless client — a bot, a server-side agent, or an observer attached to the same
call — can read and write realtime events without running the media plane.
Frames, text and binary, are forwarded verbatim; the proxy does not rewrite or
invent session events. Browser clients that already hold the `oai-events` data
channel do not need this route.

## Codex-Specific Features

These features are extensions beyond the standard OpenAI API, designed for Codex CLI compatibility.

### Prompt caching

`prompt_cache_key` keeps related prefixes in the same backend cache family. Use one stable, privacy-safe key per conversation or application prefix.

Official Codex 0.153.3 [derives this field from Responses metadata](https://github.com/openai/codex/blob/b1a547b1f73ce86205d9222ac19cff334b3b7a2e/codex-rs/core/src/client.rs#L540-L550), unless an explicit override is present. This proxy follows the same precedence on Chat requests: explicit `prompt_cache_key`, then a non-empty `client_metadata.session_id`, then omission. It does not generate a process-wide session or reuse `thread_id` as the cache key.

The private Codex OAuth HTTP and WebSocket request structures do not contain public GPT-5.6 `prompt_cache_options` or content-block `prompt_cache_breakpoint` fields. This proxy treats explicit `null` as omitted and rejects non-null controls with HTTP 400 instead of forwarding a request that the private route rejects or silently pretending the requested cache policy was applied. Use a public Responses API client when explicit cache policy or breakpoints are required. See OpenAI's [public prompt caching guide](https://developers.openai.com/api/docs/guides/prompt-caching#prompt-cache-breakpoints).

```json
{
  "model": "gpt-5.6-sol",
  "prompt_cache_key": "tenant-hash:knowledge-v1",
  "messages": [
    {"role": "system", "content": "Answer from the supplied context."},
    {"role": "user", "content": "What changed today?"}
  ]
}
```

GPT-5.6 cache accounting is returned, when the backend supplies it, as `usage.prompt_tokens_details.cached_tokens` and `cache_write_tokens`. Treat those as cost/observability data, not as a conversation-truncation signal.

### `reasoning_effort` and `reasoning`

The legacy top-level `reasoning_effort` remains supported. Every effort is checked against the selected live model. Codex's virtual `ultra` setting resolves to the model's advertised multi-agent effort when valid, otherwise `max` when supported, otherwise the model's highest advertised non-`ultra` effort. If no valid wire effort exists, the request fails instead of inventing one.

Requests can instead use the Responses-shaped effort/context object:

```json
{
  "model": "gpt-5.6-sol",
  "messages": [
    {"role": "system", "content": "Review carefully."},
    {"role": "user", "content": "Find migration failure modes."}
  ],
  "reasoning": {
    "effort": "high",
    "context": "all_turns"
  }
}
```

- Non-null `mode`, including `"standard"` and `"pro"`, returns HTTP 400 because the private Codex request has no mode field or alternate model mapping. Codex `ultra` and `service_tier: "priority"` are different features.
- `context` is `auto`, `current_turn`, or `all_turns`. Encrypted reasoning content is included automatically on generation requests so full-history continuation can replay it.
- `reasoning_effort` and `reasoning.effort` may both be present only when equal. Conflicting values return HTTP 400.
- Responses Lite uses `all_turns` as the Codex wire default. An explicitly different context is rejected instead of silently overwritten; use `responses_lite: false` when the backend route supports classic Responses and another context is required.
- Remote compact keeps its existing private Codex `reasoning_effort` field but does not accept public `reasoning.mode` or `reasoning.context`.

Anthropic `thinking` values map as `enabled → high` and `adaptive → medium`. Disabled thinking selects `none` when available, otherwise the lowest supported catalog effort, and suppresses reasoning output; it does not claim to eliminate a model's internal reasoning. `output_config.effort` takes precedence over adaptive or enabled thinking and supports `low`, `medium`, `high`, `xhigh`, and `max`. Call-level disabled thinking takes precedence over ambient `output_config.effort` for auxiliary calls such as WebFetch. Explicit unsupported effort requests still fail. Hosted Anthropic WebSearch is rejected because its lifecycle cannot be translated losslessly. The supported reasoning context and verbosity extensions are also available on `/v1/messages`, image generation, and inspection requests; Pro, non-null public cache policy/breakpoints, and any supplied `safety_identifier` fail explicitly on the private Codex provider.

Anthropic's `JSONOutputFormat` carries a schema but no name. When Claude Code
sends that official shape, the proxy follows pinned Codex 0.153.3 and supplies
the transport-only name `codex_output_schema`; this label has no model-selection
or fallback role. An explicit proxy-extension name is preserved only when it is
already a valid Responses format name, and is never sanitized or truncated.

The pinned official Codex HTTP request has no `stop` field. Omitted OpenAI
`stop`, or explicit OpenAI `stop: null`, is omitted from the private request;
any other OpenAI `stop` value returns HTTP 400. Anthropic `stop_sequences`
follows its typed API contract: omission is allowed, while explicit `null` or
an array value returns HTTP 400 because the private transport cannot apply it.

The mapping is intentionally transport-aware:

| Public/facade input | Private Codex behavior |
|---|---|
| Non-null `reasoning.mode` | HTTP 400; no private Codex field or model alias |
| `prompt_cache_key` | Forward an explicit value; otherwise use non-empty `client_metadata.session_id` |
| Non-null `prompt_cache_options` / breakpoint | HTTP 400; `null` is omitted and there is no private field |
| Any supplied `safety_identifier` | HTTP 400; the optional field is non-nullable and OAuth account/thread IDs are not semantic aliases |
| Non-null `stop` / `stop_sequences` | HTTP 400; no private field |
| `previous_response_id` | Resolve locally and replay complete Response history over HTTP |
| `service_tier: "fast"` | Send `service_tier: "priority"` |
| `service_tier: "default"` | Omit the field |
| `service_tier: "ultrafast"` | Send `service_tier: "ultrafast"` only when the live model advertises it |
| Messages/compact `speed: "ultrafast"` (proxy extension) | Resolve to `service_tier: "ultrafast"`, with the same catalog check |

Ultra Fast is distinct from Fast (`priority`), not a reasoning effort. For OpenAI
Chat requests use `"service_tier": "ultrafast"`; for `/v1/messages` and
`/v1/messages/compact` either that field or the proxy extension
`"speed": "ultrafast"` is accepted. If both fields are supplied they must agree.
Unsupported tiers return HTTP 400 before an inference request, including streaming
requests; the proxy never silently downgrades Ultra Fast to priority. Check the
account-specific `service_tiers` returned by `GET /v1/models`. An official feature
flag does not grant account access or change pricing. `speed: "ultrafast"` is not
claimed as an official Anthropic field, and existing `speed: "fast"` behavior is
unchanged. The proxy does not inherit speed defaults from Codex config.

Official behavior reference: Codex [0.164.0 service-tier resolution](https://github.com/openai/codex/blob/rust-v0.164.0/codex-rs/tui/src/service_tier_resolution.rs)
and [stable, enabled-by-default ultrafast feature](https://github.com/openai/codex/blob/rust-v0.164.0/codex-rs/features/src/lib.rs).
The bundled private-protocol reference records the reviewed Codex source revision. Outbound client identity is separate: the proxy resolves the stable `latest` version of npm [`@openai/codex`](https://www.npmjs.com/package/@openai/codex) and uses that same version for the model-catalog `client_version` query and Codex User-Agent. It is not hardcoded to the bundled reference or inferred from a locally installed CLI. Successful version lookups are cached for one hour; a refresh failure retains the last successfully resolved version and retries after a short backoff. A cold lookup failure returns an explicit availability error rather than advertising an invented version.

Authenticated tests on July 10, 2026 verified both official continuation paths with `gpt-5.6-sol`: a direct private Responses WebSocket completed a second delta request using the exact prior `response.id`, and the private HTTP endpoint completed the same continuation when the full prior input/output history was replayed. The proxy implements the HTTP replay strategy. Direct HTTP forwarding of `previous_response_id`, Pro, public cache controls, and `safety_identifier` was rejected and is not used. See OpenAI's [reasoning mode documentation](https://developers.openai.com/api/docs/guides/reasoning#reasoning-mode) and the Codex 0.153.3 [HTTP/WebSocket request structures](https://github.com/openai/codex/blob/b1a547b1f73ce86205d9222ac19cff334b3b7a2e/codex-rs/codex-api/src/common.rs#L275-L350).

### `responses_lite`

Controls the Codex Responses Lite request shape. Accepted values are `true`, `false`, and `"auto"`. Request value takes precedence over `CODEX_AS_API_RESPONSES_LITE`; default is `"auto"`.

In `"auto"` mode, this package uses Lite only when the selected live catalog entry advertises it. Setting `responses_lite: true` forces the Lite request shape and moves tools/instructions into Lite-compatible developer input items.

Codex implements web search and image generation for Lite models through client-side standalone tools. This proxy has no standalone tool executor, so Lite requests containing hosted `web_search` or `image_generation` tools fail explicitly instead of silently dropping the tools. When a request does not override the mode, `CODEX_AS_API_RESPONSES_LITE=off` selects the existing classic request contract if the backend route supports it.

The official Codex Lite request builder has no lossless `input_image.detail` mapping. Classic Responses preserve `auto`, `low`, and `high`, plus `original` only when the live catalog advertises it. A caller-supplied image detail on Lite returns HTTP 400 instead of silently removing the requested behavior.

### Image detail

Chat multimodal blocks are preserved instead of being flattened to text:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Inspect this at native resolution."},
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/png;base64,...",
        "detail": "original"
      }
    }
  ]
}
```

`/v1/inspect` accepts the equivalent flat image object: `{"image_url":"data:image/...","detail":"original"}`. Recognized detail values are `auto`, `low`, `high`, and `original`; `original` additionally requires support in the live model catalog. See OpenAI's [image detail guide](https://developers.openai.com/api/docs/guides/images-vision#choose-an-image-detail-level).

### `safety_identifier` and verbosity

The public `safety_identifier` has no equivalent private Codex request field. `ChatGPT-Account-ID`, `thread-id`, and `session-id` identify different things and are not substituted. Because this optional OpenAI field is non-nullable, any supplied value, including explicit `null`, returns HTTP 400.

Standard Chat `verbosity: "low" | "medium" | "high"` maps to Responses `text.verbosity`. The existing `text` extension remains supported. Supplying both is allowed only when the values agree.

### `parallel_tool_calls`

Classic Responses preserves an explicit `parallel_tool_calls` value without relying on a model-catalog capability that Codex does not define. Lite requests fail rather than silently disabling an explicit `true` value.

### `client_metadata` and `codex_metadata`

`client_metadata` is forwarded to the Codex backend. Set `codex_metadata: true` or `CODEX_AS_API_CODEX_METADATA=on` to add Codex-style turn metadata. Metadata mode requires a non-empty caller-supplied `client_metadata.session_id`; it preserves that session, defaults a missing root `thread_id` to the session, and preserves an explicit child `thread_id`.

The installation ID and process window ID remain stable, while `turn_id` and `x-codex-turn-metadata` are regenerated for each request. Metadata `thread_id` is neither a `previous_response_id` alias nor a cache key. An explicit `prompt_cache_key` wins; otherwise the non-empty session ID supplies cache affinity.

### `previous_response_id`

Non-streaming responses and the final streaming finish chunk expose the real upstream Responses ID as `response_id`. The provider keeps up to 256 completed chains in a process-local LRU store scoped to the authenticated account. Passing a known ID as `previous_response_id` under the same account prepends the saved semantic input and exact prior `response.output_item.done` items, including encrypted reasoning and tool items, then sends one full private HTTP request. When both the saved source model and current live model publish `comp_hash`, unequal values fail before sampling because automatic pre-turn compaction is not implemented; equal or unavailable hashes follow Codex's compatibility rule. The ID itself is never forwarded and is never converted to `thread_id`; another account cannot replay it.

That event source is intentional: the Codex 0.153.3 SSE parser [takes semantic items from `response.output_item.done`](https://github.com/openai/codex/blob/b1a547b1f73ce86205d9222ac19cff334b3b7a2e/codex-rs/codex-api/src/sse/responses.rs#L357-L372), while its [`response.completed` shape contains response metadata and usage](https://github.com/openai/codex/blob/b1a547b1f73ce86205d9222ac19cff334b3b7a2e/codex-rs/codex-api/src/sse/responses.rs#L118-L166), not replay history. The authenticated private HTTP rollout likewise returned an empty `response.completed.output`; the proxy therefore commits the completed output-item events instead of treating that extra field as conversation state.

Only a real `response.completed` event commits a chain. Branches from an older retained ID are supported. Restarting the server or evicting an old entry removes that local state; an unknown ID returns HTTP 400 before any upstream request.

```bash
curl http://localhost:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Continue from where we left off."}
    ],
    "previous_response_id": "resp_abc123"
  }'
```

### Native Responses workflows not adapted by Chat

Programmatic Tool Calling and hosted Multi-agent beta introduce `program`, `program_output`, `caller`, agent-attributed items/events, beta headers, and replay rules that cannot be represented losslessly by Chat Completions tool messages. This facade therefore returns HTTP 400 for `programmatic_tool_calling`, `allowed_callers`, `output_schema`, or `multi_agent` instead of silently dropping lifecycle data. PDF `input_file.detail` is also Responses-only and is not accepted by the Chat content adapter.

Use a native Responses client/runtime for those workflows. Relevant OpenAI documentation: [Programmatic Tool Calling](https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling) and [Multi-agent beta](https://developers.openai.com/api/docs/guides/tools-multi-agent).

### `subagent` / `x-openai-subagent`

Identifies the request as coming from a specific subagent type. Values used by Codex CLI: `review`, `compact`, `memory_consolidation`, `collab_spawn`.

Can be passed as a body field or HTTP header:

```bash
# As body field
curl http://localhost:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [{"role": "system", "content": "Review this code."}, {"role": "user", "content": "..."}],
    "subagent": "review"
  }'

# As HTTP header
curl http://localhost:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-openai-subagent: review" \
  -d '{
    "model": "gpt-5.5",
    "messages": [{"role": "system", "content": "Review this code."}, {"role": "user", "content": "..."}]
  }'
```

### `memgen_request` / `x-openai-memgen-request`

Flags the request as a memory generation/consolidation request. Can be passed as a body field (`bool`) or HTTP header (`"true"/"false"`):

```bash
curl http://localhost:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-openai-memgen-request: true" \
  -d '{
    "model": "gpt-5.5",
    "messages": [{"role": "system", "content": "Consolidate memories."}, {"role": "user", "content": "..."}]
  }'
```

## Using with OpenAI SDKs

Point the base URL to your local server:

### Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:18080/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="gpt-5.5",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    extra_body={"prompt_cache_key": "my-session"},
)
print(response.choices[0].message.content)
```

### Node.js (openai SDK)

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://localhost:18080/v1",
  apiKey: "unused",
});

const response = await client.chat.completions.create({
  model: "gpt-5.5",
  messages: [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: "Hello!" },
  ],
});
console.log(response.choices[0].message.content);
```

### curl (streaming)

```bash
curl -N http://localhost:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Tell me a joke."}
    ],
    "stream": true,
    "prompt_cache_key": "joke-session"
  }'
```

## Using with Claude Code

The `/v1/messages` endpoint implements the Anthropic Messages gateway shape used by Claude Code. The current request shape was reproduced with Claude Code `2.1.220`; streaming, real Codex OAuth chat, request-level GPT model routing, adaptive thinking, `--effort max`, and Fast Mode remain covered.

Start the proxy first. `CODEX_AS_API_MODEL` is required when Claude Code sends a built-in `claude-*` facade model name; the configured backend must exist in the fresh authenticated catalog:

```bash
CODEX_AS_API_MODEL=gpt-5.6-terra \
codex-as-api
```

Claude Code hosted WebSearch is not supported through this gateway, regardless
of Responses Lite mode. Classic Responses can execute an OpenAI hosted search,
but its output lacks the Anthropic provenance and encrypted lifecycle data
needed for a lossless Messages response. WebFetch remains a client-side
auxiliary request and retains the disabled-thinking precedence behavior when
process-level effort is enabled.

To keep the built-in Fable, Opus, Sonnet, and Haiku rows and append one GPT row, launch the GPT-routed Claude Code process with these variables:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:18080 \
ANTHROPIC_AUTH_TOKEN=unused \
ANTHROPIC_CUSTOM_MODEL_OPTION=gpt-5.6-sol \
ANTHROPIC_CUSTOM_MODEL_OPTION_NAME='GPT-5.6 Sol' \
ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION='GPT-5.6 Sol through codex-as-api' \
CLAUDE_CODE_ATTRIBUTION_HEADER=0 \
claude
```

Run `/model` and select **GPT-5.6 Sol**. The custom entry is appended to the built-in rows rather than replacing them. Claude Code `2.1.220` recognizes this GPT model ID without a custom capability declaration: `/effort low|medium|high|xhigh|max`, the picker effort control, and `claude --effort ...` are translated from Claude Code's `output_config.effort` to the Codex reasoning effort. Claude Code Fast Mode's `speed: "fast"` is translated to the Codex `priority` service tier.

Claude Code sends `x-claude-code-session-id` for gateway request grouping. This
proxy hashes the exact header value into a stable `prompt_cache_key`; it does
not turn the header into Codex `session_id` or `thread_id` metadata. A
top-level `prompt_cache_key` is supported as a proxy extension and takes
precedence. Normal Messages requests remain stateless, so callers must send
full message history.

If managed settings define an `availableModels` allowlist, that list must include the exact `ANTHROPIC_CUSTOM_MODEL_OPTION` value, such as `gpt-5.6-sol`; otherwise Claude Code hides or rejects the custom row.

For a persistent gateway configuration, put the same variables in the `env` object in `~/.claude/settings.json`. Do that only when every Claude Code process using that config should route through codex-as-api.

`ANTHROPIC_CUSTOM_MODEL_OPTION` adds one picker row. To make another live model the visible GPT row, change that value and the display text. Any model returned by `/v1/models` can also be selected directly without changing the visible row:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:18080 ANTHROPIC_AUTH_TOKEN=unused \
  claude --model gpt-5.6-terra --effort high
ANTHROPIC_BASE_URL=http://127.0.0.1:18080 ANTHROPIC_AUTH_TOKEN=unused \
  claude --model gpt-5.6-luna --effort medium
```

Live Codex IDs are sent to the matching backend model. Built-in Claude/Fable/Opus/Sonnet/Haiku IDs are not Codex models, so recognized `claude-*` IDs use the configured Codex model or authenticated catalog default; the proxy preserves the selected client name in the Anthropic response.

`ANTHROPIC_BASE_URL` applies to the whole Claude Code process. Therefore the built-in rows remain available in the picker, but they also pass through codex-as-api and use the configured or catalog-default Codex backend; the rows do not call Anthropic models in that process. To use an actual Anthropic model and GPT in parallel, keep the gateway variables out of global settings, start the GPT process with the inline command above, and start a second process normally:

```bash
claude --model sonnet
```

If the gateway variables are already in `~/.claude/settings.json`, shell-level `env -u` does not override that settings file. Remove the persistent variables or use a separate `CLAUDE_CONFIG_DIR` for the direct-Anthropic process.

Claude Code gateway discovery cannot expose raw `gpt-*` IDs because Claude Code filters discovered IDs to `claude*` and `anthropic*`. The official single custom-model option avoids a misleading facade ID. This integration is an Anthropic Messages-compatible bridge; Anthropic explicitly does not support routing Claude Code to non-Claude models through third-party gateways. See the official [model configuration](https://code.claude.com/docs/en/model-config), [gateway connection](https://code.claude.com/docs/en/llm-gateway-connect), and [gateway protocol](https://code.claude.com/docs/en/llm-gateway-protocol) references.

The bridge accepts Claude Code's no-op thinking cleanup (`clear_thinking_20251015` with `keep: "all"`), validates and ignores boolean `eager_input_streaming`, and accepts `defer_loading: false`. Context edits that would change history, task budgets, enabled strict/deferred tool execution, malformed output formats, and unsupported image source types return HTTP 400. Valid base64 and URL image blocks plus `tool_result.is_error` are preserved during translation.

## Architecture

```
Client (OpenAI SDK / curl)
    |
    v
HTTP Server (FastAPI / Axum / Express)
    |
    +---> ChatGPTOAuthProvider
            |
            +---> ~/.codex-as-api/auth.json (own OAuth credentials, auto-refresh)
            +---> https://chatgpt.com/backend-api/codex/models
            +---> https://chatgpt.com/backend-api/codex/responses
```

The provider handles:
- Managed ChatGPT token loading, proactive refresh five minutes before expiry, and refresh on 401
- Fresh-only authenticated model discovery, validation, and ETag invalidation
- OpenAI Responses API over SSE
- `prompt_cache_key` and cache read/write accounting
- Live catalog-gated reasoning effort and context handling
- Real Codex reasoning extensions without fabricated Anthropic signatures
- Tool call streaming
- Codex-specific headers (`x-openai-subagent`, `x-openai-memgen-request`)
- Bounded, account-scoped `previous_response_id` history replay over private HTTP
- Multimodal Chat input, image generation/inspection, and capability-gated classic `original` detail
- Remote conversation compaction
- ChatGPT Live realtime call creation and a sideband event relay, in the TypeScript runtime only

## Release & package publishing

The `Release` GitHub Actions workflow runs on a pushed `v*` tag. It rejects a
tag that does not exactly match every Python, npm, and Rust version surface,
runs the complete cross-runtime gate, and then publishes PyPI, npmjs, the
scoped GitHub npm package, deterministic platform Rust ZIP archives, and one
GitHub Release. Each Rust archive contains the platform binary, `LICENSE`, and
the generated `THIRD_PARTY_NOTICES.md` covering both bundled tokenizer data and
the locked normal dependency graph across supported release targets.

The workflow uses registry-supported OIDC for PyPI and npmjs. No `PYPI_TOKEN`,
`NPM_TOKEN`, personal access token, or repository secret is required. One-time
registry setup is still required:

- Create GitHub environments named `pypi` and `npm`; add deployment protection
  rules if desired.
- On PyPI, add a trusted publisher for project `codex-as-api`, owner `Eunho-J`,
  repository `codex-as-api`, workflow `release.yml`, environment `pypi`.
- On npmjs, add a GitHub Actions trusted publisher for package `codex-as-api`,
  owner `Eunho-J`, repository `codex-as-api`, workflow `release.yml`,
  environment `npm`, with `npm publish` allowed.
- GitHub Packages and GitHub Releases use the job-scoped automatic
  `GITHUB_TOKEN`; repository policy must allow the workflow's declared
  `packages: write` and `contents: write` permissions.

After bumping every version and passing the local gate, push the matching tag:

```bash
python scripts/check_package_versions.py --tag v0.10.0
git tag v0.10.0
git push origin v0.10.0
```

## Tests

### Python

```bash
pip install -e ".[dev]"
pip install httpx
pytest tests/ -v
```

### Rust

```bash
cd rust
cargo test
```

### TypeScript

```bash
cd ts
npm install
npm test
```


## Release Notes

### v0.10.0

- Sign in directly with ordinary browser OAuth; no Codex installation or existing
  Codex credentials are required. Device-code login is an explicit remote option.
- Add independent admin-key-protected login start/status/cancel endpoints and a
  private application-owned credential store across Python, TypeScript and Rust.
- Upgrading requires a new codex-as-api login; existing Codex credentials remain
  untouched. See [Standalone login](#standalone-login) and [full release notes](RELEASES.md).

### v0.8.0

- Add ChatGPT Live realtime voice to the TypeScript runtime: `POST
  /v1/realtime/calls` creates an OAuth-authorized call from a client-owned WebRTC
  SDP offer and returns the SDP answer, and `WS /v1/realtime/sideband/{call_id}`
  relays the OpenAI realtime event sideband for headless clients. Audio stays on
  the client's media plane.
- Mirror the pinned Codex call contract exactly: one identity in
  `x-session-id`/`session-id`/`thread-id`, the `openai-alpha: quicksilver=v2`
  handshake, `intent=quicksilver&architecture=avas`, and a session payload that
  never chooses its own voice-session id.
- Document realtime as a TypeScript-only capability; Python and Rust remain
  HTTP-only in this release instead of claiming cross-runtime parity.

### v0.7.0

- Replace bundled model capability tables with a fresh authenticated Codex
  catalog and add `GET /v1/models` across all three runtimes.
- Remove stale/unknown-model fallbacks, fabricated Anthropic signatures, silent
  request downgrades, and malformed upstream recovery; return typed failures.
- Reject hosted Anthropic WebSearch instead of translating incomplete OpenAI
  search lifecycle data, and require authoritative completion and final usage.
- Keep immediate Claude Code `message_start` delivery without inventing token
  counts; final Anthropic fields are emitted only from authoritative provider data.
- Scope continuation history and catalog state to the authenticated account,
  validate before streaming, and make `/health` report actual readiness.
- Require the official nested `tokens` auth layout, remove root-level and
  noncanonical auth aliases, and reject external-host `chatgptAuthTokens`.
- Pin the validated private contract to Codex `0.153.3` and publish the exact
  tested artifacts to PyPI, npmjs, GitHub Packages, and GitHub Releases.
- Install Python server dependencies with the base package so the default
  `pip install codex-as-api` command produces a runnable server.

### v0.6.5

- Align GPT-5.6 alias, Sol, Terra, and Luna context limits with Codex `0.147.0`
  at 272,000 tokens and derive the 244,800-token automatic compact threshold.
- Pin the verified upstream request, Responses Lite, header, and model contract
  to `openai/codex` commit
  `be6e8eac029b183056b7e4402879f15d2c85f61b`; track its new fractional SSE
  rollout-budget field without claiming an unapproved facade mapping.
- Refresh OAuth credentials five minutes before expiry, coalesce concurrent
  refreshes, reload changed credentials after 401, and preserve upstream HTTP
  status codes across all three servers.
- Replace startup npm version discovery with deterministic compatibility and
  package-version identity in the Codex-style `User-Agent`.
- Add full cross-runtime CI plus tag-gated OIDC publishing to PyPI and npmjs,
  `GITHUB_TOKEN` publishing to GitHub Packages, and GitHub Release assets.

### v0.6.4

- Remove process-global generated Codex session/thread identities and require
  explicit session identity when Codex metadata mode is enabled.
- Resolve Chat cache affinity as explicit `prompt_cache_key`, then
  `client_metadata.session_id`, then omission.
- Derive Claude Code cache affinity from `x-claude-code-session-id` without
  fabricating Codex metadata; keep normal Anthropic Messages stateless.
- Validate and strip Anthropic `cache_control` compatibility hints without
  claiming unavailable breakpoint or TTL semantics.
- Update current compatibility evidence to Codex `0.145.0`, Codex `main` at
  `bd2de422aa287b97b06ca6425a10935bcf1b3731`, and Claude Code `2.1.220`.

### v0.6.3

- Accept Claude Code auxiliary requests that combine ambient `output_config.effort` with call-level `thinking.disabled`.
- Give explicit disabled thinking precedence and send Codex reasoning effort `none` instead of returning HTTP 400.
- Document and test `CODEX_AS_API_RESPONSES_LITE=off` for Claude Code hosted WebSearch on GPT-5.6; WebFetch needs no Responses Lite override.
- Preserve fail-loudly validation for invalid effort values and unsupported `output_config` fields.
- Add Python, TypeScript, and Rust adapter regressions, streamed `/v1/messages` coverage, and a `/count_tokens` regression for the WebSearch/WebFetch request shape.

### v0.6.2

- Fix `/v1/messages/count_tokens` overcounting that could trigger Claude Code autocompaction on every turn.
- Count normalized model-visible messages, tool calls, reasoning, tool-result metadata, and tool schemas once; remove the second full raw-request-body addition.
- Port official `tiktoken` `o200k_base` ordinary encoding into Python, TypeScript, and Rust without adding a `tiktoken` package dependency or runtime rank download.
- Match the upstream Unicode text split and BPE merge ranks last checked on 2026-07-14 against `tiktoken` 0.13.0 at `08a5f3b`.
- Keep image input cost separate and count inline base64 images once rather than adding both a fixed image estimate and the base64 request bytes.
- Add matching Python, TypeScript, and Rust endpoint regressions: a 4,000-byte ASCII message now returns `1,012` instead of roughly `8,000`, and non-model control fields do not change the count.
- Validate the release with Claude Code `2.1.209` and real Codex OAuth chat.

### v0.6.1

- Re-audit official Codex `0.144.4` and `main` at `393f64565ab46f09d99ca4d9bd973537e72a114b`, plus Claude Code `2.1.208`, before publishing.
- Validate Claude Code `2.1.208` with real Codex OAuth chat and tool loops, and document the official custom GPT picker entry alongside Fable, Opus, Sonnet, and Haiku.
- Route bundled GPT model IDs from Anthropic requests to the selected Codex backend model while retaining configured fallback behavior for built-in Claude model names.
- Map `output_config.effort` and Fast Mode to Codex effort and priority service tier, preserve URL images and tool-error state, and handle current context/tool beta fields without silent loss.
- Keep Messages, token counting, and remote compaction aligned on effective model limits and fail-loudly validation for output formats and image sources.
- Keep Python streaming work off the ASGI event loop and make Rust Anthropic SSE truly incremental so concurrent Claude Code requests are not stalled or buffered.
- Preserve Rust streaming authentication, rate-limit, and overload errors after OAuth refresh instead of collapsing them to generic HTTP 500 failures.
- Restrict the Python source distribution to release inputs so local agent state and unrelated runtime sources cannot enter PyPI artifacts.

### v0.6.0

- Add GPT-5.6 public alias plus Sol, Terra, and Luna capability metadata, model defaults, and Codex context-window behavior.
- Refresh bundled context and default-effort metadata for current GPT-5.5, GPT-5.4, GPT-5.4 Mini, and GPT-5.2 catalog entries.
- Update Responses Lite request bodies and headers across chat, compact, and inspection paths; Lite image generation is rejected explicitly because this proxy has no standalone image-tool executor.
- Add `max`, Codex-compatible `ultra` to `max` wire conversion, future model-defined efforts, and `model_reasoning_effort` config support.
- Add GPT-5.6 reasoning effort/context, `standard` mode compatibility, capability-gated image `original` detail, standard verbosity, cache accounting, and real backend `response_id` support across Python, TypeScript, and Rust.
- Translate known `previous_response_id` values into bounded local full-history replay; reject Pro and non-null public cache policy/breakpoints or `safety_identifier` because the private Codex request contract has no equivalent field.
- Reject hosted Multi-agent and Programmatic Tool Calling on the Chat facade until their native agent/program item lifecycle can be preserved.

### v0.5.2

- Support latest Codex root-level OAuth token files while keeping PAT-only, agent-identity-only, and Bedrock-only auth files explicitly unsupported.
- Add shared model capability gating for Responses Lite, parallel tool calls, verbosity, and service-tier behavior across Python, TypeScript, and Rust.
- Preserve encrypted reasoning state via top-level `reasoning.encrypted_content` include and add Codex metadata forwarding controls.

### v0.5.1

- Add official Codex CLI `originator` and versioned `User-Agent` headers for ChatGPT/Codex OAuth requests.
- Resolve the latest `@openai/codex` version from npm at server startup, with `CODEX_AS_API_CODEX_CLI_VERSION` as an explicit override.

### v0.5.0

- Preserve Claude Code server-tool history (`server_tool_use`, `web_search_tool_result`, MCP/advisor-like result blocks) as backend context instead of dropping it on the next turn.
- Support Anthropic `output_format` structured outputs by mapping JSON schema/object formats to OpenAI Responses `text.format`.
- Preserve `document` and `search_result` content blocks inside tool results, keep Python streaming defaults aligned, and accept unsuffixed `web_search` server-tool types across Python, TypeScript, and Rust.

### v0.4.0

- Add Claude Code-compatible Anthropic hosted web search support by mapping `web_search_*` server tools to OpenAI Responses `web_search`.
- Return `server_tool_use` and `web_search_tool_result` blocks so Claude Code can parse web search results reliably.
- Prepare TypeScript package publishing to npmjs and GitHub Packages via GitHub Actions.

### v0.3.3

- Stop forwarding client `max_tokens` as Codex `max_output_tokens`, restoring Claude Code compatibility with the Codex OAuth backend.
- Add Python, TypeScript, and Rust regression tests for the provider payload.

### v0.3.2

- Restore immediate Anthropic streaming so Claude Code receives events without waiting for the backend response to finish.
- Use conservative local token estimates for `/v1/messages/count_tokens`; Codex OAuth has no count-only backend endpoint.
- Keep real final streaming usage metadata in `message_delta`.

### v0.3.1

- Attempted real backend token counting for `/v1/messages/count_tokens` with `max_output_tokens: 0`; this is superseded by v0.3.2 because Codex OAuth rejects count-only requests.
- Forward converted Anthropic tools, tool choice, stop sequences, and thinking/reasoning settings during token-count requests.
- Propagate cumulative Anthropic streaming usage, including cache accounting, server tool use, and service tier metadata when available.
- Pass `max_output_tokens` through provider requests across Python, TypeScript, and Rust.

### v0.3.0

- Read Codex CLI config from `CODEX_HOME` / `~/.codex/config.toml` across Python, TypeScript, and Rust.
- Use the configured Codex backend model while preserving Anthropic client model names in `/v1/messages` responses.
- Expose `context_window` and `auto_compact_token_limit` through `/health` and `/v1/messages/count_tokens`.
- Add Anthropic-compatible `/v1/messages/count_tokens` and `/v1/messages/compact`.
- Map context-window failures to Anthropic-style `400 invalid_request_error` responses and stream error events.

## License

Apache License 2.0 — derived from [OpenAI Codex CLI](https://github.com/openai/codex) (Apache-2.0, Copyright 2025 OpenAI).
