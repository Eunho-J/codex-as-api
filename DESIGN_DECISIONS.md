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
