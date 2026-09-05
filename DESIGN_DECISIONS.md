# Design Decision Agenda

This is the fixed repository-local agenda for design-affecting work.

## D-001: Identifier and cache semantics by facade

Status: APPROVED; CONTINUATION SCOPE AMENDED BY D-003

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

Status: APPROVED; MODEL-CATALOG CLAUSES SUPERSEDED AND AUTH SCOPE AMENDED BY D-003

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

## D-003: Live model authority and fail-loud proxy semantics

Status: APPROVED

### Evidence

User words, verbatim:

> 수동 업데이트 안하게 가능한가? 모델 카탈로그 받는 API도 제공해야하지 않아?

> 정적 `model-capabilities.json`은 시작용 bootstrap과 보수적 fallback으로만 유지 이거 필요 한거야? 그냥 다 없애는게 낫지 않나?

> 불필요한 fallback이 있지 않은지, 필요없는데 과하게 동작하는 예외처리가 있는건 아닌지.

> 이거 다 고쳐야해. 앞선 내용 포함해서 모두 작업해줘. 배포까지

### Decision

- Treat the authenticated ChatGPT Codex `GET /models` response as the only
  runtime authority for model membership and model capabilities.
- Remove bundled model catalogs, hard-coded model and context defaults, unknown
  model metadata, stale-cache recovery, and bootstrap catalog data from every
  runtime package. Test-only fixtures are not runtime fallback data.
- Cache only fresh in-memory catalog snapshots for five minutes. Scope each
  snapshot to the authenticated account, backend base URL, and validated Codex
  client version; coalesce concurrent refreshes and invalidate on a changed
  `X-Models-Etag` response header.
- Resolve and validate the effective model before opening streaming response
  headers. Return explicit authentication, catalog-unavailable, model-not-found,
  request-validation, and upstream-protocol errors without substituting data.
- Expose every model in the authenticated fresh snapshot through `GET /v1/models`.
  Preserve upstream visibility and API-support metadata instead of inferring an
  additional local filter.
- Do not maintain public GPT model aliases. Requests use exact slugs from the
  fresh authenticated snapshot.
  Recognized `claude-*` compatibility names require an explicitly configured
  backend model from the same snapshot; arbitrary unknown names and missing
  configured models fail.
- Namespace process-local `previous_response_id` history by authenticated account
  so one account's content cannot be replayed under another account.
- Accept only the official nested `tokens` layout for managed ChatGPT auth.
  Reject root-level token aliases, noncanonical `auth_mode` aliases, and
  external-host `chatgptAuthTokens`; the standalone proxy has no host callback
  that can resolve or refresh externally owned credentials.
- Reject unsupported or malformed caller controls before transport and reject
  malformed upstream protocol data instead of dropping fields, inventing IDs,
  synthesizing values, or emitting placeholder signatures.
- For Anthropic JSON schema output, use pinned Codex's deterministic
  `codex_output_schema` transport label when the official Anthropic shape omits
  a name. This required wire label is not model data, an alias, or a recovery
  fallback; caller-supplied names are validated and never rewritten.
- Preserve the Anthropic typed distinction between an omitted field and
  explicit `null`. Accept `null` only for fields whose supported Anthropic type
  is nullable or for a separately documented private-transport no-op; reject
  it for non-nullable optional fields before transport.
- Keep the narrowly documented Claude Code compatibility handling for Anthropic
  `max_tokens` and validated `cache_control`, managed-ChatGPT OAuth refresh-once
  behavior, and protocol-correct in-stream error events after response headers
  have begun.
- Reject Anthropic hosted WebSearch instead of fabricating its server-tool
  provenance from OpenAI hosted-search output. Require authoritative final
  usage and a real `response.completed`; derive terminal status from emitted
  tool calls and pinned Codex's `end_turn` semantics, where only explicit
  `false` is nonterminal. Preserve immediate Claude Code `message_start`
  delivery without guessing token counts.
- Implement and verify the same behavior in Python, TypeScript, and Rust, then
  publish one versioned release to every configured registry and GitHub target.

### Rejected Alternatives

- Bundled or stale model data as a fallback when the authenticated catalog is
  unavailable.
- Routing arbitrary unknown model IDs to a configured default model.
- Reporting healthy readiness without valid authentication, a fresh catalog, and
  a valid effective model.
- Accepting unsupported controls as no-ops outside the explicitly approved
  Anthropic compatibility surface.
- Treating externally hosted ChatGPT auth tokens as locally refreshable OAuth
  credentials without the required external-host callback.
- Recovering malformed request or upstream data with fabricated IDs, empty
  objects, default roles, default completion reasons, or placeholder signatures.

### Affected Components

- Python, TypeScript, and Rust authentication, catalog, provider, and HTTP server
  modules
- Cross-runtime integration, error, package-content, and account-isolation tests
- `README.md`, `ts/README.md`, `RELEASES.md`, package metadata, and release
  workflows
