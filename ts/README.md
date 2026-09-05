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

`/v1/messages` and `/v1/messages/compact` are Claude Code compatibility
endpoints using the Anthropic Messages shape, not general Anthropic API
proxies. Both require one valid `x-claude-code-session-id` header because the
private Codex transport cannot apply Anthropic's required `max_tokens` control.

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
auxiliary request; call-level `thinking.disabled` wins over ambient effort and is
forwarded as Codex reasoning effort `none`.

Final responses require authoritative provider usage and a real
`response.completed` event. Function calls map to `tool_use`; otherwise a
missing, `null`, or `true` `end_turn` maps to `end_turn`, matching pinned Codex,
where only explicit `false` requests another sampling turn. A nonterminal
completion without a representable client tool call fails; after streaming
headers it produces an Anthropic `error` event without a successful
`message_stop`. The gateway preserves Claude Code's immediate `message_start`
and does not invent input-token counts there, so this stream-start behavior is
not generic Anthropic API parity.

Generated direct-caller tool blocks and empty assistant content arrays can be
replayed. Malformed, null, or server-tool caller variants fail before transport.

If managed settings define an `availableModels` allowlist, include the exact custom option ID, such as `gpt-5.6-sol`, or Claude Code will hide or reject the GPT row.

The base URL applies to the entire Claude Code process. Built-in `claude-*` rows therefore require an explicit `CODEX_AS_API_MODEL` that exists in the fresh catalog; they do not contact Anthropic while the gateway variables are active. Unknown names are not routed to a default. Keep these variables out of global settings when a second, direct-Anthropic Claude Code process is required. See the [full Claude Code guide](https://github.com/Eunho-J/codex-as-api#using-with-claude-code) for routing details and current compatibility limits.

Real Codex reasoning is exposed as the top-level `codex_reasoning` extension or
the custom `codex_reasoning_delta` SSE event. The gateway does not fabricate
Anthropic `thinking` blocks or signatures. `reasoning_content` is response-only;
Chat request messages reject it rather than dropping an unrepresentable replay
field.
