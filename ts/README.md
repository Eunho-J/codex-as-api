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

`/v1/messages/count_tokens` bundles a dependency-free port of official
`tiktoken` `o200k_base` ordinary encoding, including the Unicode text split,
BPE merge logic, and rank data. No `tiktoken` package or runtime download is
required. The last upstream synchronization check was 2026-07-14 against
`tiktoken` 0.13.0 commit `08a5f3b2c987ada4fc5aa1f16c643c203fa8acaa`.

For configuration, supported endpoints, model behavior, and examples, see the [canonical GitHub documentation](https://github.com/Eunho-J/codex-as-api#readme).

## Claude Code with GPT

Start `codex-as-api` with the classic Responses transport so Claude Code can use hosted WebSearch:

```bash
CODEX_AS_API_RESPONSES_LITE=off codex-as-api
```

Then launch Claude Code with one GPT entry alongside the built-in Fable, Opus, Sonnet, and Haiku rows:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:18080 \
ANTHROPIC_AUTH_TOKEN=unused \
ANTHROPIC_CUSTOM_MODEL_OPTION=gpt-5.6-sol \
ANTHROPIC_CUSTOM_MODEL_OPTION_NAME='GPT-5.6 Sol' \
ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION='GPT-5.6 Sol through codex-as-api' \
CLAUDE_CODE_ATTRIBUTION_HEADER=0 \
claude
```

Run `/model` and select **GPT-5.6 Sol**. Change `ANTHROPIC_CUSTOM_MODEL_OPTION` to show Terra or Luna instead. Any bundled GPT ID can also be selected directly when the same base URL and token variables are present, for example `ANTHROPIC_BASE_URL=http://127.0.0.1:18080 ANTHROPIC_AUTH_TOKEN=unused claude --model gpt-5.6-terra --effort high`.

Claude Code auxiliary WebSearch/WebFetch requests can combine process-level
`output_config.effort` with call-level `thinking.disabled`. The explicit
disabled setting wins and is forwarded as Codex reasoning effort `none`.
`CODEX_AS_API_RESPONSES_LITE=off` is required for hosted WebSearch because the
official Lite path uses a standalone `web.run` executor that this gateway does
not implement. WebFetch is a client-side auxiliary request, so WebFetch only
needs the disabled-thinking precedence fix and does not require the classic
transport.

If managed settings define an `availableModels` allowlist, include the exact custom option ID, such as `gpt-5.6-sol`, or Claude Code will hide or reject the GPT row.

The base URL applies to the entire Claude Code process. Built-in model rows therefore use the configured `CODEX_AS_API_MODEL` fallback through this proxy; they do not contact Anthropic while the gateway variables are active. Keep these variables out of global settings when a second, direct-Anthropic Claude Code process is required. See the [full Claude Code guide](https://github.com/Eunho-J/codex-as-api#using-with-claude-code) for routing details and current compatibility limits.
