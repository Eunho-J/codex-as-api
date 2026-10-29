# Third-Party Notices

## OpenAI Codex — standalone OAuth login protocol port

The standalone browser OAuth and device-code login implementations are adapted
from OpenAI Codex (`openai/codex`, Apache License 2.0), pinned to
`rust-v0.164.0`, commit `6aaac694803004a63c6f393d153c96d22521a6b9`, in particular
`codex-rs/login/src/server.rs`, `codex-rs/login/src/device_code_auth.rs`, and
`codex-rs/login/src/oauth/{authorization,pkce,client}.rs`.
The Apache License 2.0 is included in this distribution's `LICENSE` file.

Local modifications implement equivalent Python, TypeScript and Rust login
managers and authenticated REST start/status/cancel endpoints; use a dedicated
codex-as-api credential store with no discovery/import of existing Codex
credentials; add bounded deadlines, cancellation, atomic storage and private
refresh ownership. This is not an unmodified distribution of the Codex CLI.

The upstream NOTICE is retained below. The login port does not include the
Codex terminal UI or use Ratatui; its attribution is retained as part of the
upstream notice rather than a claim that the terminal UI is bundled here.

> OpenAI Codex
> Copyright 2025 OpenAI
>
> This project includes code derived from [Ratatui](https://github.com/ratatui/ratatui), licensed under the MIT license.
> Copyright (c) 2016-2022 Florian Dehau
> Copyright (c) 2023-2025 The Ratatui Developers

## OpenAI tiktoken

This project includes a port of the `o200k_base` ordinary-encoding logic and
the corresponding merge-rank data from
[openai/tiktoken](https://github.com/openai/tiktoken).

Synchronized with tiktoken 0.13.0 at commit
`08a5f3b2c987ada4fc5aa1f16c643c203fa8acaa` on 2026-07-14.
The bundled rank data has SHA-256
`446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`.

MIT License

Copyright (c) 2022 OpenAI, Shantanu Jain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
