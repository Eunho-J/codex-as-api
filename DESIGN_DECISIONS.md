# Design Decision Agenda

This is the fixed repository-local agenda for design-affecting work.

## D-001: Identifier and cache semantics by facade

Status: APPROVED

### Evidence

User words, verbatim:

> thread_id는 어디서 쓴느거고, session_id, prompt_cache_key, previous_response_id 는 어디서 쓰는건데? codex 랑 claude 각각 어디야? 지금 구현기준으로만 말해

> 이건 뭔데. cache_control이 애초에 뭔데

> 그럼 다시 수정계획. 특히 낡은거 고치는거까지

> 작업 후 새 버전 릴리즈 업데이트, pypi/npm/github 배포 모두 수행한다. 인증은 [$aside-browser](/Users/cayde/.agents/skills/aside-browser/SKILL.md) 에 등록된 인증 수단을 활용한다.

### Decision

- Remove process-global generated Codex session and thread identities.
- Use explicit `client_metadata.session_id` as the Codex session and default
  prompt-cache affinity. Preserve an explicit thread ID and otherwise use the
  session ID for the root thread.
- Keep Chat `previous_response_id` as bounded process-local HTTP history replay.
  Do not forward it to the private Codex HTTP request.
- Keep normal Anthropic Messages stateless. Reject a non-null
  `previous_response_id` and do not expose the nonstandard Chat `response_id`.
- Use Claude Code's session header only to derive a privacy-safe prompt cache
  key. Do not synthesize Codex metadata for Anthropic Messages.
- Accept Anthropic `cache_control` only as a validated compatibility hint and
  strip it before the Codex request. Do not claim Anthropic breakpoint or TTL
  semantics on a transport that cannot represent them.
- Update active documentation and tests to the current Codex and Claude Code
  behavior, then release the change to PyPI, npm, and GitHub.

### Rejected Alternatives

- Process-wide generated session/thread IDs: unrelated callers share identity.
- Treating `thread_id`, `previous_response_id`, and `prompt_cache_key` as aliases:
  the current Codex implementation assigns them different roles.
- Claiming that stripped Anthropic `cache_control` applies cache boundaries or
  TTLs: the private Codex request has no equivalent fields.

### Affected Components

- Python, TypeScript, and Rust request providers and Anthropic routes
- Cross-runtime request/transport tests
- `README.md`, `ts/README.md`, and `RELEASES.md`
- Python, npm, Rust, and GitHub release metadata

## D-002: Codex 0.147 compatibility and automated releases

Status: APPROVED

### Evidence

User words, verbatim:

> 바로고칠항목 다 고치자. 그리고 배포도 자동화 가능한가? 깃허브 actions로? 거기서 토큰 필요하거나 인증필요한거 정리해서 알려주고, 너는 작업 들어가

### Decision

- Align the bundled GPT-5.6 capability limits with the stable Codex 0.147
  catalog and derive the matching compact limit.
- Preserve upstream HTTP status semantics across Python, TypeScript, and Rust,
  including rate-limit and overload responses.
- Refresh OAuth credentials proactively and coalesce concurrent refreshes.
  Before refreshing after an unauthorized response, reload the credential file
  and reuse a matching credential update made by another request.
- Replace the startup npm lookup and moving CLI-version impersonation with a
  deterministic User-Agent that identifies both the validated Codex
  compatibility baseline and the `codex-as-api` package version.
- Add pinned, structured upstream contract fixtures and automated drift checks
  for the model catalog, Responses request surface, Responses Lite behavior,
  SSE usage, and relevant headers.
- Require a cross-runtime GitHub Actions test and package-build gate.
- Automate tag-based PyPI, npm, GitHub Packages, and GitHub Release publishing.
  Prefer registry-supported GitHub OIDC trusted publishing, use the repository
  `GITHUB_TOKEN` for same-repository GitHub publishing, and fail loudly when a
  required publisher or credential is not configured.
- Publish only after the tag and every package version agree and all tests and
  package checks pass.

### Rejected Alternatives

- Discover the newest npm Codex version at server startup: that version is not
  evidence that the private request contract was validated.
- Silently skip a registry publish when credentials are absent: a green release
  workflow must mean every configured release target completed.
- Replace the approved private HTTP facade with App Server, native Responses,
  or WebSocket lifecycle behavior as part of this compatibility update.
- Add live runtime model-catalog state before a separate design decision.

### Affected Components

- Python, TypeScript, and Rust authentication, providers, and HTTP servers
- Shared model capability data and version-pinned contract fixtures
- Cross-runtime tests and release metadata
- GitHub Actions CI and release workflows
- `README.md`, `ts/README.md`, and `RELEASES.md`
