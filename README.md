# codex-as-api

Use ChatGPT / Codex OAuth as a local OpenAI-compatible API server.

## What it does

Runs a lightweight HTTP server on `localhost` that translates standard OpenAI API calls into authenticated requests against the ChatGPT / Codex backend using your existing `~/.codex/auth.json` OAuth credentials.

## Install

```bash
git clone https://github.com/Eunho-J/codex-as-api.git
cd codex-as-api
pip install -e ".[server]"
```

Or with `uv`:

```bash
uv pip install -e ".[server]"
```

## Prerequisites

You must be logged in via the official Codex CLI so that `~/.codex/auth.json` exists:

```bash
codex login
```

The server reads that file to obtain and refresh ChatGPT OAuth tokens automatically.

## Run

```bash
codex-as-api
```

By default the server binds to `0.0.0.0:8000`. Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CODEX_AS_API_HOST` | `0.0.0.0` | Bind address |
| `CODEX_AS_API_PORT` | `8000` | Listen port |
| `CODEX_AS_API_MODEL` | `gpt-5.5` | Model identifier passed to Codex |
| `CODEX_AS_API_AUTH_PATH` | (auto) | Path to `auth.json` |

## API Endpoints

### `POST /v1/chat/completions`

Standard OpenAI chat completions. Supports streaming (`stream: true`) and non-streaming.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "prompt_cache_key": "my-session-42"
  }'
```

Extra body fields supported by this server (not in the standard OpenAI API):

- `prompt_cache_key` (string) — Sent to the Codex backend for prefix-cache stickiness.
- `reasoning_effort` (string) — One of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`.

### `POST /v1/images/generations`

Generate images via Codex image generation tool.

```bash
curl -s http://localhost:8000/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-5.5",
    "prompt": "a futuristic city at sunset",
    "size": "1024x1024"
  }'
```

### `POST /v1/inspect`

Inspect images with a text prompt (custom endpoint, not standard OpenAI).

```bash
curl -s http://localhost:8000/v1/inspect \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Describe what you see",
    "images": [{"image_url": "data:image/png;base64,iVBORw0KGgo..."}]
  }'
```

### `POST /v1/compact`

Compact a conversation into a checkpoint for continuation (custom endpoint).

```bash
curl -s http://localhost:8000/v1/compact \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hi"},
      {"role": "assistant", "content": "Hello!"}
    ]
  }'
```

### `GET /health`

Health check + auth availability.

```bash
curl -s http://localhost:8000/health
```

## Architecture

```
Client (OpenAI SDK / curl)
    |
    v
FastAPI server (codex-as-api)
    |
    +---> ChatGPTOAuthProvider
            |
            +---> ~/.codex/auth.json (OAuth tokens)
            +---> https://chatgpt.com/backend-api/codex
```

The provider layer (`ChatGPTOAuthProvider`) handles:

- Token loading and automatic refresh on 401
- OpenAI Responses API over SSE
- `prompt_cache_key` passthrough for prefix-cache stickiness
- Reasoning content streaming (`reasoning_content`, `reasoning`)
- Tool call streaming
- Image generation and inspection
- Remote conversation compaction

## License

MIT
