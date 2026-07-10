# codex-as-api

Use ChatGPT / Codex OAuth as a local OpenAI-compatible API server.

## Prerequisite

Install the official Codex CLI and sign in once so `~/.codex/auth.json` exists:

```bash
npm install -g @openai/codex
codex login
```

## Install and run

```bash
npm install -g codex-as-api
codex-as-api
```

The server listens on `127.0.0.1:18080` by default.

For configuration, supported endpoints, model behavior, and examples, see the [canonical GitHub documentation](https://github.com/Eunho-J/codex-as-api#readme).
