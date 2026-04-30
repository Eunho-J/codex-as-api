# codex-as-api

Use ChatGPT / Codex OAuth as a local OpenAI-compatible API server.

## What it does

Runs a lightweight HTTP server on `localhost` (or `0.0.0.0`) that translates standard OpenAI API calls into authenticated requests against the ChatGPT / Codex backend using your existing `~/.codex/auth.json` OAuth credentials. Supports streaming, tool calling, reasoning, image generation, and Codex-specific features like `prompt_cache_key` and subagent headers.

Both Python and Rust implementations are provided — identical functionality, same endpoints, same behavior.

## Prerequisites

You must be logged in via the official Codex CLI so that `~/.codex/auth.json` exists:

```bash
codex login
```

The server reads that file to obtain and refresh ChatGPT OAuth tokens automatically.

## Install & Run

### Python

```bash
git clone https://github.com/Eunho-J/codex-as-api.git
cd codex-as-api
pip install -e ".[server]"
codex-as-api
```

Or with `uv`:

```bash
uv pip install -e ".[server]"
codex-as-api
```

### Rust

```bash
cd rust
cargo build --release
./target/release/codex-as-api
```

Both versions bind to `0.0.0.0:8000` by default.

## Configuration

Environment variables (both Python and Rust):

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEX_AS_API_HOST` | `0.0.0.0` | Bind address |
| `CODEX_AS_API_PORT` | `8000` | Listen port |
| `CODEX_AS_API_MODEL` | `gpt-5.5` | Model identifier passed to Codex backend |
| `CODEX_AS_API_AUTH_PATH` | `~/.codex/auth.json` | Path to OAuth credentials file |

To bind to localhost only:

```bash
CODEX_AS_API_HOST=127.0.0.1 codex-as-api
```

## API Endpoints

### `POST /v1/chat/completions`

Standard OpenAI chat completions. Supports streaming (`stream: true`) and non-streaming.

```bash
curl http://localhost:8000/v1/chat/completions \
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
curl http://localhost:8000/v1/chat/completions \
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
curl http://localhost:8000/v1/chat/completions \
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

### `POST /v1/images/generations`

Generate images via the Codex image generation tool.

```bash
curl http://localhost:8000/v1/images/generations \
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
curl http://localhost:8000/v1/inspect \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Describe what you see",
    "images": [{"image_url": "data:image/png;base64,iVBORw0KGgo..."}]
  }'
```

### `POST /v1/compact`

Compact a conversation into a checkpoint for continuation (custom endpoint).

```bash
curl http://localhost:8000/v1/compact \
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

Health check. Returns auth availability and configured model.

```bash
curl http://localhost:8000/health
# {"status":"ok","auth_available":true,"model":"gpt-5.5"}
```

## Codex-Specific Features

These features are extensions beyond the standard OpenAI API, designed for Codex CLI compatibility.

### `prompt_cache_key`

Enables prefix-cache stickiness on the Codex backend. When multiple requests share the same `prompt_cache_key`, the backend can reuse cached KV computations for the shared prefix, reducing latency and cost.

**When to use:** Set a stable key per conversation or session. All turns within the same session should share one key.

```bash
curl http://localhost:8000/v1/chat/completions \
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

```bash
curl http://localhost:8000/v1/chat/completions \
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

### `previous_response_id`

Chains responses together on the backend. Pass the response ID from a previous turn to maintain server-side conversation state.

```bash
curl http://localhost:8000/v1/chat/completions \
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
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [{"role": "system", "content": "Review this code."}, {"role": "user", "content": "..."}],
    "subagent": "review"
  }'

# As HTTP header
curl http://localhost:8000/v1/chat/completions \
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
curl http://localhost:8000/v1/chat/completions \
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
    base_url="http://localhost:8000/v1",
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
  baseURL: "http://localhost:8000/v1",
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
curl -N http://localhost:8000/v1/chat/completions \
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

## Architecture

```
Client (OpenAI SDK / curl)
    |
    v
HTTP Server (FastAPI or Axum)
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

## License

Apache License 2.0 — derived from [OpenAI Codex CLI](https://github.com/openai/codex) (Apache-2.0, Copyright 2025 OpenAI).
