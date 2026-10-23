# codex-as-api

Use ChatGPT / Codex OAuth as a local OpenAI-compatible API server.

## Prerequisite

Install the official Codex CLI and sign in once so `~/.codex/auth.json` exists:

```bash
npm install -g @openai/codex
codex login
```

The standalone proxy accepts only Codex-managed credentials: a nested `tokens`
object with `auth_mode` equal to `chatgpt`, `null`, or absent. External
`chatgptAuthTokens` files require a host refresh bridge and are rejected rather
than being refreshed or rewritten as managed credentials.

## Install and run

```bash
npm install -g codex-as-api
codex-as-api
```

The server listens on `127.0.0.1:18080` by default.

Model membership and capabilities come only from the authenticated Codex
`/models` response. `GET /v1/models` returns the current fresh catalog; no
bundled or stale model fallback is shipped. `/health` returns `503` unless OAuth,
catalog refresh, and effective-model resolution are ready.

`/v1/messages/count_tokens` bundles a dependency-free port of official
`tiktoken` `o200k_base` ordinary encoding, including the Unicode text split,
BPE merge logic, and rank data. No `tiktoken` package or runtime download is
required. The last upstream synchronization check was 2026-07-14 against
`tiktoken` 0.13.0 commit `08a5f3b2c987ada4fc5aa1f16c643c203fa8acaa`.

For configuration, supported endpoints, model behavior, and examples, see the [canonical GitHub documentation](https://github.com/Eunho-J/codex-as-api#readme).

`/v1/messages`, `/v1/messages/count_tokens`, and `/v1/messages/compact` accept Anthropic-shaped requests from any client. No Claude Code header is required. `max_tokens` and validated ephemeral `cache_control` are compatibility hints; Codex OAuth does not expose the matching output cap or cache TTL/breakpoints. The optional Claude Code session header supplies cache affinity only.

## Identifier and cache behavior

| Input | `/v1/chat/completions` | `/v1/messages` |
|---|---|---|
| `client_metadata.session_id` | Forwarded, used as default cache affinity, and required by Codex metadata mode | Not used |
| `client_metadata.thread_id` | Codex metadata identity; root defaults to the session and an explicit child thread is preserved | Not used or synthesized |
| `prompt_cache_key` | Explicit value, then non-empty session ID, then omission | Explicit proxy extension, then hashed Claude Code session header |
| `previous_response_id` | 256-chain process-local, account-scoped full-history replay; never forwarded | Non-null values return HTTP 400 |
| `cache_control` | Not a Chat control | Validated as an Anthropic hint and stripped before Codex |

Claude Code's exact `x-claude-code-session-id` value is hashed as
`SHA-256("codex-as-api:claude-code-session:" + sessionId)` for cache affinity
only. It is not converted to Codex session or thread metadata. Accepted
Anthropic cache hints use `type: "ephemeral"` with optional TTL `5m` or `1h`;
the private Codex transport cannot apply Anthropic breakpoint or TTL semantics.
This behavior was rechecked against Codex `0.153.3` at
`b1a547b1f73ce86205d9222ac19cff334b3b7a2e`; Claude Code compatibility was
last checked with `2.1.220`.

## Claude Code with GPT

Start `codex-as-api` with the desired live Codex model:

```bash
CODEX_AS_API_MODEL=gpt-5.6-terra codex-as-api
```

Then launch Claude Code with one live Codex model entry alongside the built-in Fable, Opus, Sonnet, and Haiku rows:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:18080 \
ANTHROPIC_AUTH_TOKEN=unused \
ANTHROPIC_CUSTOM_MODEL_OPTION=gpt-5.6-sol \
ANTHROPIC_CUSTOM_MODEL_OPTION_NAME='GPT-5.6 Sol' \
ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION='GPT-5.6 Sol through codex-as-api' \
CLAUDE_CODE_ATTRIBUTION_HEADER=0 \
claude
```

Run `/model` and select **GPT-5.6 Sol**. Change `ANTHROPIC_CUSTOM_MODEL_OPTION` to another ID returned by `/v1/models` when needed. Any live Codex ID can also be selected directly with the same base URL and token variables, for example `ANTHROPIC_BASE_URL=http://127.0.0.1:18080 ANTHROPIC_AUTH_TOKEN=unused claude --model gpt-5.6-terra --effort high`.

Hosted Anthropic WebSearch is rejected on Messages, token-counting, and compact
requests. OpenAI hosted search results do not carry the Anthropic provenance and
encrypted lifecycle data required for a lossless Messages response, and classic
Responses does not change that limitation. WebFetch remains a client-side
auxiliary request; call-level disabled thinking wins over ambient effort and uses the selected model's least reasoning level (`none` when supported, otherwise the lowest advertised level). Explicit unsupported OpenAI effort requests still return 400.

Final responses require authoritative provider usage and a real
`response.completed` event. Function calls map to `tool_use`; otherwise a
missing, `null`, or `true` `end_turn` maps to `end_turn`, matching pinned Codex,
where only explicit `false` requests another sampling turn. A nonterminal
completion without a representable client tool call fails; after streaming
headers it produces an Anthropic `error` event without a successful
`message_stop`. The immediate `message_start` includes provisional numeric zero
usage counters so standard stream consumers can initialize; final cumulative
usage supplies the authoritative counts.

Generated direct-caller tool blocks and empty assistant content arrays can be
replayed. Malformed, null, or server-tool caller variants fail before transport.

If managed settings define an `availableModels` allowlist, include the exact custom option ID, such as `gpt-5.6-sol`, or Claude Code will hide or reject the GPT row.

The base URL applies to the entire Claude Code process. Built-in `claude-*` rows use the configured `CODEX_AS_API_MODEL` or the fresh catalog default; they do not contact Anthropic while the gateway variables are active. Unknown non-facade names are not routed to a default. Keep these variables out of global settings when a second, direct-Anthropic Claude Code process is required. See the [full Claude Code guide](https://github.com/Eunho-J/codex-as-api#using-with-claude-code) for routing details and current compatibility limits.

Summarized thinking exposes actual Codex summaries as unsigned `thinking`
blocks / `thinking_delta` content and the existing `codex_reasoning` extensions.
Omitted and disabled thinking suppress reasoning output. No signature is
fabricated. Valid Anthropic thinking history is validated and stripped during
replay. OpenAI Chat `reasoning_content` remains response-only.

## Realtime voice (ChatGPT Live)

This runtime also proxies ChatGPT Live realtime voice over the same OAuth login.
`POST /v1/realtime/calls` takes a client-owned WebRTC SDP offer plus an optional
`session` object (`instructions`, `voice`, ...), returns the SDP answer as
`text/plain`, and mirrors the call id as `Location: /v1/realtime/calls/{call_id}`.
The client keeps the media plane; the proxy never relays audio.
`WS /v1/realtime/sideband/{call_id}` relays realtime events to
`wss://api.openai.com/v1/live/{call_id}` for headless clients such as a bot or an
observer attached to the same call.

The Python and Rust runtimes stay HTTP-only in this release. Realtime is a
TypeScript-only capability and is documented as such rather than claimed as
cross-runtime parity.
