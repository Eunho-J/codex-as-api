# codex-as-api

[![GitHub Release](https://img.shields.io/github/v/release/Eunho-J/codex-as-api)](https://github.com/Eunho-J/codex-as-api/releases)
[![PyPI](https://img.shields.io/pypi/v/codex-as-api)](https://pypi.org/project/codex-as-api/)
[![npm](https://img.shields.io/npm/v/codex-as-api)](https://www.npmjs.com/package/codex-as-api)
[![License](https://img.shields.io/github/license/Eunho-J/codex-as-api)](LICENSE)

Use ChatGPT / Codex OAuth as a local OpenAI-compatible API server.

## Features

- **OpenAI & Anthropic compatible** — `POST /v1/chat/completions` and `POST /v1/messages` endpoints
- **Claude Code ready** — use Codex models directly from Claude Code CLI
- **Streaming** — full SSE streaming for both OpenAI and Anthropic protocols
- **Tool calling** — function calls, tool results, and parallel tool calls
- **Image support** — generation, inspection, and base64 image passthrough (including tool result images)
- **Reasoning** — configurable reasoning effort with streaming thinking content
- **Codex features** — `prompt_cache_key`, `previous_response_id`, subagent headers, remote compaction
- **Codex config aware** — reads `CODEX_HOME` / `~/.codex/config.toml` for model and context-window settings
- **Token estimate & compaction helpers** — Anthropic-compatible `/v1/messages/count_tokens` and `/v1/messages/compact`
- **Auto auth** — reads `~/.codex/auth.json` and auto-refreshes OAuth tokens
- **3 implementations** — Python, TypeScript (npm), and Rust — identical behavior

## What it does

Runs a lightweight HTTP server on `localhost` that translates standard OpenAI API calls into authenticated requests against the ChatGPT / Codex backend using your existing `~/.codex/auth.json` OAuth credentials.

Python, Rust, and TypeScript (npm) implementations are provided — identical functionality, same endpoints, same behavior.

## Prerequisites

Install the official Codex CLI and log in so that `~/.codex/auth.json` exists:

```bash
npm install -g @openai/codex
codex login
```

The server reads that file to obtain and refresh ChatGPT OAuth tokens automatically.

`tokens` and latest root-level `access_token` / `refresh_token` / `id_token` auth files are supported. `personal_access_token`-only, `agent_identity`-only, and `bedrock_api_key`-only auth files are not supported for the ChatGPT OAuth backend; rerun `codex login` if you hit that diagnostic.

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
pip install -e ".[server]"
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
const provider = new ChatGPTOAuthProvider({ model: "gpt-5.5" });
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

## Configuration

Environment variables (Python, Rust, and TypeScript):

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEX_AS_API_HOST` | `127.0.0.1` | Bind address |
| `CODEX_AS_API_PORT` | `18080` | Listen port |
| `CODEX_AS_API_MODEL` | `~/.codex/config.toml` `model`, else `gpt-5.5` | Model identifier passed to Codex backend |
| `CODEX_AS_API_AUTH_PATH` | `~/.codex/auth.json` | Path to OAuth credentials file |
| `CODEX_AS_API_CODEX_CLI_VERSION` | latest `@openai/codex` from npm | Override the Codex CLI version used in backend request `User-Agent` headers |
| `CODEX_AS_API_RESPONSES_LITE` | `auto` | Responses Lite mode: `auto`, `on`, or `off` |
| `CODEX_AS_API_CODEX_METADATA` | `off` | Add Codex-style per-turn `client_metadata` and related backend headers |
| `CODEX_HOME` | `~/.codex` | Codex home directory used for `auth.json` and `config.toml` discovery |

The server also reads root-level Codex CLI settings from `~/.codex/config.toml`:

```toml
model = "gpt-5.5"
model_context_window = 200000
model_auto_compact_token_limit = 160000
```

`CODEX_AS_API_MODEL` overrides the Codex config model. The context settings are exposed from `/health` and returned by Anthropic token-count responses.

### Supported Models

| Model | Description |
|-------|-------------|
| `gpt-5.5` | Frontier model for complex coding, research, and real-world work |
| `gpt-5.4` | Strong model for everyday coding |
| `gpt-5.4-mini` | Small, fast, and cost-efficient model for simpler coding tasks |
| `gpt-5.3-codex` | Coding-optimized model |
| `gpt-5.3-codex-spark` | Ultra-fast coding model |
| `gpt-5.2` | Previous generation model |

Model capability behavior is driven by `config/model-capabilities.json` across Python, TypeScript, and Rust. Unknown models use conservative defaults: classic Responses payloads, no assumed parallel tool support, and no automatic verbosity or service-tier assumptions.

To use a different port:

```bash
CODEX_AS_API_PORT=9000 codex-as-api
```

To expose on all interfaces (e.g. for remote access):

```bash
CODEX_AS_API_HOST=0.0.0.0 codex-as-api
```

## API Endpoints

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

### `POST /v1/messages`

Anthropic Messages API compatible endpoint. Supports streaming (`stream: true`) and non-streaming. The client's model name is reflected in responses, but the server always uses the configured `CODEX_AS_API_MODEL` for the backend call.

```bash
curl http://localhost:18080/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: unused" \
  -H "anthropic-version: 2023-06-01" \
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

### `POST /v1/messages/count_tokens`

Anthropic-compatible token counting helper. Codex OAuth does not expose a count-only endpoint equivalent to Anthropic's native API, so this route returns a conservative local estimate plus the configured context-window metadata. The estimate uses UTF-8 byte length as an upper bound for GPT/Codex BPE text tokens, then adds protocol overhead for roles, message boundaries, tools, raw request metadata, and images.

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

Anthropic-compatible alias for remote conversation compaction. Accepts Anthropic Messages-shaped bodies and returns compacted checkpoint content.

### `POST /v1/images/generations`

Generate images via the Codex image generation tool.

```bash
curl http://localhost:18080/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "prompt": "a futuristic city at sunset",
    "size": "1024x1024"
  }'
```

### `POST /v1/inspect`

Inspect images with a text prompt (custom endpoint).

```bash
curl http://localhost:18080/v1/inspect \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Describe what you see",
    "images": [{"image_url": "data:image/png;base64,iVBORw0KGgo..."}]
  }'
```

### `POST /v1/compact`

Compact a conversation into a checkpoint for continuation (custom endpoint). `/v1/messages/compact` provides the Anthropic-compatible alias.

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

Health check. Returns auth availability, configured model, Codex config path, and context-window settings.

```bash
curl http://localhost:18080/health
# {"status":"ok","auth_available":true,"model":"gpt-5.5","codex_config_path":"/Users/me/.codex/config.toml","context_window":200000,"auto_compact_token_limit":160000}
```

## Codex-Specific Features

These features are extensions beyond the standard OpenAI API, designed for Codex CLI compatibility.

### `prompt_cache_key`

Enables prefix-cache stickiness on the Codex backend. When multiple requests share the same `prompt_cache_key`, the backend can reuse cached KV computations for the shared prefix, reducing latency and cost.

**When to use:** Set a stable key per conversation or session. All turns within the same session should share one key.

**Important:** Do not use `usage.prompt_tokens_details.cached_tokens` (or `usage.input_tokens_details.cached_tokens`) as a prompt or context-management signal. This server passes through the Codex backend usage payload when it is available, and current Codex OAuth responses may report `cached_tokens: 0` even when `prompt_cache_key` is used. Treat `prompt_cache_key` as a backend cache-affinity hint, not as a guarantee that cache-hit accounting will be exposed through the API response.

```bash
curl http://localhost:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello"}
    ],
    "prompt_cache_key": "session-abc-123"
  }'
```

### `reasoning_effort`

Controls how much compute the model spends on reasoning. Valid values: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`.

When reasoning is enabled, backend requests include `reasoning.encrypted_content` so Codex can preserve encrypted reasoning state when the backend supports it.

```bash
curl http://localhost:18080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [
      {"role": "system", "content": "Solve this step by step."},
      {"role": "user", "content": "Prove that sqrt(2) is irrational."}
    ],
    "reasoning_effort": "high"
  }'
```

### `responses_lite`

Controls the Codex Responses Lite request shape. Accepted values are `true`, `false`, and `"auto"`. Request value takes precedence over `CODEX_AS_API_RESPONSES_LITE`; default is `"auto"`.

In `"auto"` mode, this package only uses Lite when the shared model capability table says the selected model should use it. Current bundled model entries default to classic Responses payloads. Setting `responses_lite: true` forces Lite and moves tools/instructions into Lite-compatible developer input items.

### `parallel_tool_calls`

Set `parallel_tool_calls: true` to request parallel tool calls when the selected model capability allows it. The shared capability table gates this field, and Responses Lite always keeps `parallel_tool_calls` disabled.

### `client_metadata` and `codex_metadata`

`client_metadata` is forwarded to the Codex backend. Set `codex_metadata: true` or `CODEX_AS_API_CODEX_METADATA=on` to overlay Codex-style turn metadata keys such as `turn_id`, `session_id`, `thread_id`, and `x-codex-turn-metadata`.

Do not rely on user-supplied values for those reserved Codex metadata keys when metadata mode is enabled; this package regenerates them per turn.

### `previous_response_id`

Chains responses together on the backend. Pass the response ID from a previous turn to maintain server-side conversation state.

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

The `/v1/messages` endpoint is compatible with [Claude Code](https://claude.ai/code). Claude Code can send its normal Anthropic model names; responses preserve the client-supplied model name, while backend Codex requests use `CODEX_AS_API_MODEL` or the `model` from `~/.codex/config.toml`.

```bash
# Minimal setup
ANTHROPIC_BASE_URL=http://localhost:18080 \
ANTHROPIC_API_KEY=unused \
claude
```

```bash
# Optional: force the backend Codex model for all Claude Code requests
CODEX_AS_API_MODEL=gpt-5.5 codex-as-api

# In another shell
ANTHROPIC_BASE_URL=http://localhost:18080 \
ANTHROPIC_API_KEY=unused \
claude
```

## Architecture

```
Client (OpenAI SDK / curl)
    |
    v
HTTP Server (FastAPI / Axum / Express)
    |
    +---> ChatGPTOAuthProvider
            |
            +---> ~/.codex/auth.json (OAuth tokens, auto-refresh)
            +---> https://chatgpt.com/backend-api/codex/responses
```

The provider handles:
- Token loading and automatic refresh on 401
- OpenAI Responses API over SSE
- `prompt_cache_key` passthrough for prefix-cache stickiness
- Reasoning content streaming (`reasoning_content`, `reasoning`)
- Tool call streaming
- Codex-specific headers (`x-openai-subagent`, `x-openai-memgen-request`)
- `previous_response_id` for response chaining
- Image generation and inspection
- Remote conversation compaction

## Release & package publishing

- Bump versions in `pyproject.toml`, `ts/package.json`, `ts/package-lock.json`, `rust/Cargo.toml`, and `rust/Cargo.lock`.
- Publish a GitHub Release such as `v0.5.2` from the matching commit.
- The manually-dispatched `Publish npm packages` workflow builds/tests the TypeScript package, runs `npm pack --dry-run`, publishes `codex-as-api` to npmjs when `NPM_TOKEN` is configured, and publishes `@eunho-j/codex-as-api` to GitHub Packages with `GITHUB_TOKEN`.

## Tests

### Python

```bash
pip install -e ".[dev,server]"
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
