import * as assert from "node:assert/strict";
import * as http from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, beforeEach, describe, it } from "node:test";
import {
  ChatGPTOAuthCatalogUnavailableError,
  ChatGPTOAuthInvalidRequestError,
} from "../auth.js";
import {
  CODEX_CLI_ORIGINATOR,
  CODEX_NPM_LATEST_URL,
  CODEX_NPM_PACKAGE_NAME,
  CODEX_VERSION_CACHE_TTL_MS,
  CODEX_VERSION_FETCH_TIMEOUT_MS,
  CODEX_VERSION_RETRY_BACKOFF_MS,
  ChatGPTOAuthCodexVersionUnavailableError,
  CodexVersionResolver,
  compareCodexVersions,
  codexCliHeadersForVersion,
  codexUserAgentForVersion,
  getCachedCodexCliVersion,
  isValidCodexVersion,
  parseCodexNpmLatestPayload,
  parseCodexSemver,
  parseJsonStrictNoDuplicateKeys,
  requireExplicitCodexVersion,
  resetCodexVersionCache,
  resolveCodexCliHeaders,
  resolveCodexCliVersion,
  resolveLatestCodexVersion,
  validateCodexVersion,
} from "../codex-version.js";

function jsonResponse(
  body: string,
  status = 200,
  headers: Record<string, string> = { "Content-Type": "application/json" },
): Response {
  return new Response(body, { status, headers });
}

function npmLatestPayload(
  version = "0.118.0",
  name = CODEX_NPM_PACKAGE_NAME,
): string {
  return JSON.stringify({ name, version });
}

describe("codex-version semver and payload validation", () => {
  it("accepts strict three-component nonnegative stable versions", () => {
    assert.deepEqual(parseCodexSemver("0.0.0"), [0, 0, 0]);
    assert.deepEqual(parseCodexSemver("0.118.0"), [0, 118, 0]);
    assert.deepEqual(parseCodexSemver("1.23.456"), [1, 23, 456]);
    assert.equal(isValidCodexVersion("0.118.0"), true);
    assert.equal(validateCodexVersion("0.118.0"), "0.118.0");
    assert.equal(requireExplicitCodexVersion("0.118.0"), "0.118.0");
  });

  it("rejects prerelease, build metadata, non-3-component, leading-zero, whitespace, and injection versions", () => {
    const invalidVersions: unknown[] = [
      "",
      "0.118",
      "0.118.0.1",
      "v0.118.0",
      "0.118.0-alpha.1",
      "0.118.0-0",
      "0.118.0+build.1",
      "0.118.0-rc.1+build.1",
      "00.118.0",
      "0.0118.0",
      "0.118.00",
      "-1.0.0",
      "0.-1.0",
      "0.1.-1",
      " 0.118.0",
      "0.118.0 ",
      "0. 118.0",
      "0.118.0\n",
      "0.118.0\r\nX-Injected: true",
      "0.118.0\u0000",
      `1.2.${"9".repeat(62)}`,
      "9007199254740992.0.0",
      null,
      undefined,
      118,
      { version: "0.118.0" },
    ];

    for (const candidate of invalidVersions) {
      assert.equal(
        parseCodexSemver(candidate),
        null,
        `expected ${String(candidate)} to be rejected`,
      );
      assert.equal(isValidCodexVersion(candidate), false);
      assert.throws(
        () => validateCodexVersion(candidate),
        ChatGPTOAuthCodexVersionUnavailableError,
      );
      assert.throws(
        () => requireExplicitCodexVersion(candidate),
        ChatGPTOAuthInvalidRequestError,
      );
    }
  });

  it("compares semver tuples numerically without lexicographic regression", () => {
    assert.equal(compareCodexVersions("0.99.9", "0.100.0"), -1);
    assert.equal(compareCodexVersions("0.100.0", "0.99.9"), 1);
    assert.equal(compareCodexVersions("0.118.0", "0.118.0"), 0);
    assert.equal(compareCodexVersions("1.0.0", "0.999.999"), 1);
    assert.throws(
      () => compareCodexVersions("0.118.0-alpha.1", "0.118.0"),
      ChatGPTOAuthInvalidRequestError,
    );
  });

  it("rejects duplicate JSON keys at root and nested levels, including escaped keys", () => {
    assert.throws(
      () =>
        parseJsonStrictNoDuplicateKeys(
          '{"name":"@openai/codex","version":"0.118.0","version":"0.119.0"}',
        ),
      SyntaxError,
    );
    assert.throws(
      () =>
        parseJsonStrictNoDuplicateKeys(
          '{"name":"@openai/codex","ver\\u0073ion":"0.118.0","version":"0.119.0"}',
        ),
      SyntaxError,
    );
    assert.throws(
      () =>
        parseJsonStrictNoDuplicateKeys(
          '{"name":"@openai/codex","version":"0.118.0","dist":{"shasum":"a","shasum":"b"}}',
        ),
      SyntaxError,
    );
  });

  it("validates npm @openai/codex latest metadata payload strictly", () => {
    assert.equal(
      parseCodexNpmLatestPayload(
        '{"name":"@openai/codex","version":"0.120.1","dist":{"tarball":"https://example.com/t.tgz"}}',
      ),
      "0.120.1",
    );

    const invalidPayloads: Array<string | Uint8Array> = [
      new Uint8Array([0xff, 0xfe, 0xfd]),
      "{malformed",
      "null",
      "[]",
      '"0.118.0"',
      '{"name":"codex","version":"0.118.0"}',
      '{"name":"@openai/other","version":"0.118.0"}',
      '{"version":"0.118.0"}',
      '{"name":"@openai/codex"}',
      '{"name":"@openai/codex","version":"0.118.0-beta.1"}',
      '{"name":"@openai/codex","version":"0.118.0","version":"0.119.0"}',
    ];

    for (const payload of invalidPayloads) {
      assert.throws(
        () => parseCodexNpmLatestPayload(payload),
        (err: unknown) =>
          err instanceof ChatGPTOAuthCodexVersionUnavailableError
          && err instanceof ChatGPTOAuthCatalogUnavailableError,
      );
    }
  });

  it("constructs User-Agent and CLI headers from explicit or resolved version", () => {
    const ua = codexUserAgentForVersion("0.118.0", "0.1.3");
    assert.match(
      ua,
      /^codex_cli_rs\/0\.118\.0 \([^)]+\) codex-as-api\/0\.1\.3$/,
    );
    const headers = codexCliHeadersForVersion("0.118.0", "0.1.3");
    assert.equal(headers.originator, CODEX_CLI_ORIGINATOR);
    assert.equal(headers["User-Agent"], ua);
  });
});

describe("CodexVersionResolver lifecycle, caching, backoff, singleflight, and monotonicity", () => {
  beforeEach(() => {
    resetCodexVersionCache();
  });

  afterEach(() => {
    resetCodexVersionCache();
  });

  it("fetches from npm latest endpoint with GET, Accept: application/json, manual redirect, and no auth headers", async () => {
    let seenUrl: string | undefined;
    let seenInit: RequestInit | undefined;

    const resolver = new CodexVersionResolver({
      fetchImpl: async (input, init) => {
        seenUrl = input;
        seenInit = init;
        return jsonResponse(npmLatestPayload("0.118.0"));
      },
    });

    const version = await resolver.resolve();
    assert.equal(version, "0.118.0");
    assert.equal(resolver.cachedVersion(), "0.118.0");
    assert.equal(seenUrl, CODEX_NPM_LATEST_URL);
    assert.equal(seenInit?.method, "GET");
    assert.equal(seenInit?.redirect, "manual");
    const headers = seenInit?.headers as Record<string, string>;
    assert.deepEqual(headers, { Accept: "application/json" });
    assert.equal("Authorization" in headers, false);
    assert.equal("authorization" in headers, false);
  });

  it("fails with ChatGPTOAuthCodexVersionUnavailableError on first lookup error without fallback or backoff", async () => {
    let attempts = 0;
    let clockMs = 1_000;

    const resolver = new CodexVersionResolver({
      now: () => clockMs,
      fetchImpl: async () => {
        attempts++;
        if (attempts === 1) {
          throw new TypeError("network unreachable");
        }
        if (attempts === 2) {
          return jsonResponse("upstream unavailable", 503);
        }
        return jsonResponse(npmLatestPayload("0.118.0"));
      },
    });

    await assert.rejects(
      () => resolver.resolve(),
      (err: unknown) =>
        err instanceof ChatGPTOAuthCodexVersionUnavailableError
        && err instanceof ChatGPTOAuthCatalogUnavailableError
        && err.message.includes("network unreachable"),
    );
    assert.equal(resolver.cachedVersion(), null);

    // Without advancing past 60s backoff, second call immediately attempts lookup because no lastGoodVersion exists.
    clockMs += 100;
    await assert.rejects(
      () => resolver.resolve(),
      (err: unknown) =>
        err instanceof ChatGPTOAuthCodexVersionUnavailableError
        && err.message.includes("HTTP 503"),
    );
    assert.equal(attempts, 2);

    clockMs += 100;
    const resolved = await resolver.resolve();
    assert.equal(resolved, "0.118.0");
    assert.equal(attempts, 3);
  });

  it("enforces 5s timeout and aborts hung first lookup", async () => {
    let abortObserved = false;

    const resolver = new CodexVersionResolver({
      timeoutMs: 30,
      fetchImpl: async (_url, init) =>
        await new Promise<Response>((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            abortObserved = true;
            reject(init.signal?.reason ?? new Error("aborted"));
          });
        }),
    });

    await assert.rejects(
      () => resolver.resolve(),
      (err: unknown) =>
        err instanceof ChatGPTOAuthCodexVersionUnavailableError
        && err.message.includes("timed out after 30ms"),
    );
    assert.equal(abortObserved, true);
    assert.equal(CODEX_VERSION_FETCH_TIMEOUT_MS, 5_000);
  });

  it("caches resolved version for 1 hour TTL and refreshes after expiry", async () => {
    let clockMs = 10_000;
    let calls = 0;
    const versions = ["0.118.0", "0.119.0"];

    const resolver = new CodexVersionResolver({
      now: () => clockMs,
      fetchImpl: async () => {
        const version = versions[calls] ?? "0.119.0";
        calls++;
        return jsonResponse(npmLatestPayload(version));
      },
    });

    assert.equal(await resolver.resolve(), "0.118.0");
    assert.equal(calls, 1);

    clockMs += CODEX_VERSION_CACHE_TTL_MS - 1;
    assert.equal(await resolver.resolve(), "0.118.0");
    assert.equal(calls, 1);

    clockMs += 1;
    assert.equal(await resolver.resolve(), "0.119.0");
    assert.equal(calls, 2);
  });

  it("retains last good version on expired refresh failure and backs off for 60s before retrying", async () => {
    let clockMs = 50_000;
    let calls = 0;

    const resolver = new CodexVersionResolver({
      now: () => clockMs,
      fetchImpl: async () => {
        calls++;
        if (calls === 1) {
          return jsonResponse(npmLatestPayload("0.118.0"));
        }
        if (calls === 2) {
          return jsonResponse("bad gateway", 502);
        }
        if (calls === 3) {
          return jsonResponse('{"name":"@openai/codex","version":"0.119.0-rc.1"}');
        }
        return jsonResponse(npmLatestPayload("0.119.0"));
      },
    });

    assert.equal(await resolver.resolve(), "0.118.0");
    assert.equal(calls, 1);

    // Advance to TTL expiration; refresh #2 fails with 502 -> retains 0.118.0 and sets 60s backoff.
    clockMs += CODEX_VERSION_CACHE_TTL_MS;
    assert.equal(await resolver.resolve(), "0.118.0");
    assert.equal(calls, 2);

    // Within 60s backoff window: no network fetch is attempted.
    clockMs += CODEX_VERSION_RETRY_BACKOFF_MS - 1;
    assert.equal(await resolver.resolve(), "0.118.0");
    assert.equal(calls, 2);

    // At 60s boundary: retries (#3), gets invalid prerelease version -> retains 0.118.0 and backs off another 60s.
    clockMs += 1;
    assert.equal(await resolver.resolve(), "0.118.0");
    assert.equal(calls, 3);

    clockMs += CODEX_VERSION_RETRY_BACKOFF_MS - 1;
    assert.equal(await resolver.resolve(), "0.118.0");
    assert.equal(calls, 3);

    // After second 60s backoff: retries (#4) and upgrades to 0.119.0.
    clockMs += 1;
    assert.equal(await resolver.resolve(), "0.119.0");
    assert.equal(calls, 4);
  });

  it("deduplicates concurrent lookups via singleflight for both cold start and expired refresh", async () => {
    let clockMs = 1_000;
    let calls = 0;
    let releaseFetch!: (response: Response) => void;

    const resolver = new CodexVersionResolver({
      now: () => clockMs,
      fetchImpl: async () => {
        calls++;
        return await new Promise<Response>((resolve) => {
          releaseFetch = resolve;
        });
      },
    });

    const p1 = resolver.resolve();
    const p2 = resolver.resolve();
    const p3 = resolver.resolve();
    assert.equal(calls, 1);

    releaseFetch(jsonResponse(npmLatestPayload("0.118.0")));
    const results = await Promise.all([p1, p2, p3]);
    assert.deepEqual(results, ["0.118.0", "0.118.0", "0.118.0"]);
    assert.equal(calls, 1);

    // Expire TTL and verify singleflight during refresh as well.
    clockMs += CODEX_VERSION_CACHE_TTL_MS;
    const r1 = resolver.resolve();
    const r2 = resolver.resolve();
    assert.equal(calls, 2);

    releaseFetch(jsonResponse(npmLatestPayload("0.120.0")));
    assert.deepEqual(await Promise.all([r1, r2]), ["0.120.0", "0.120.0"]);
    assert.equal(calls, 2);
  });

  it("never regresses semver when a stale upstream response reports an older version", async () => {
    let clockMs = 1_000;
    let calls = 0;
    const sequence = ["0.100.0", "0.99.9", "0.100.0", "0.101.2", "0.101.1"];

    const resolver = new CodexVersionResolver({
      now: () => clockMs,
      fetchImpl: async () => {
        const version = sequence[calls] ?? "0.101.2";
        calls++;
        return jsonResponse(npmLatestPayload(version));
      },
    });

    assert.equal(await resolver.resolve(), "0.100.0");

    // Stale mirror returns 0.99.9 after TTL expiry -> must retain 0.100.0 and extend TTL by 1h.
    clockMs += CODEX_VERSION_CACHE_TTL_MS;
    assert.equal(await resolver.resolve(), "0.100.0");
    assert.equal(calls, 2);

    // Cached for full TTL after stale response without re-fetching after 60s.
    clockMs += CODEX_VERSION_RETRY_BACKOFF_MS + 1_000;
    assert.equal(await resolver.resolve(), "0.100.0");
    assert.equal(calls, 2);

    // Next TTL expiry returns 0.100.0 -> stays 0.100.0.
    clockMs += CODEX_VERSION_CACHE_TTL_MS;
    assert.equal(await resolver.resolve(), "0.100.0");
    assert.equal(calls, 3);

    // Next TTL expiry returns 0.101.2 -> upgrades to 0.101.2.
    clockMs += CODEX_VERSION_CACHE_TTL_MS;
    assert.equal(await resolver.resolve(), "0.101.2");
    assert.equal(calls, 4);

    // Next TTL expiry returns 0.101.1 -> retains 0.101.2.
    clockMs += CODEX_VERSION_CACHE_TTL_MS;
    assert.equal(await resolver.resolve(), "0.101.2");
    assert.equal(calls, 5);
  });

  it("performs real HTTP I/O against a local loopback server and resets global resolver state", async () => {
    let requestCount = 0;
    let observedHeaders: http.IncomingHttpHeaders = {};
    let observedPath = "";
    let serverVersion = "0.121.0";

    const server = http.createServer((req, res) => {
      requestCount++;
      observedHeaders = req.headers;
      observedPath = req.url ?? "";
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ name: CODEX_NPM_PACKAGE_NAME, version: serverVersion }));
    });

    await new Promise<void>((resolve) => {
      server.listen(0, "127.0.0.1", () => resolve());
    });

    try {
      const address = server.address() as AddressInfo;
      const registryUrl = `http://127.0.0.1:${address.port}/@openai%2fcodex/latest`;

      resetCodexVersionCache({ registryUrl });
      assert.equal(getCachedCodexCliVersion(), null);

      const resolved = await resolveCodexCliVersion();
      assert.equal(resolved, "0.121.0");
      assert.equal(await resolveLatestCodexVersion(), "0.121.0");
      assert.equal(getCachedCodexCliVersion(), "0.121.0");
      assert.equal(requestCount, 1);
      assert.equal(observedPath, "/@openai%2fcodex/latest");
      assert.equal(observedHeaders.accept, "application/json");
      assert.equal(observedHeaders.authorization, undefined);

      const headers = await resolveCodexCliHeaders();
      assert.equal(headers.originator, CODEX_CLI_ORIGINATOR);
      assert.ok(headers["User-Agent"]?.startsWith("codex_cli_rs/0.121.0 "));

      serverVersion = "0.122.0";
      resetCodexVersionCache({ registryUrl });
      assert.equal(getCachedCodexCliVersion(), null);
      assert.equal(await resolveCodexCliVersion(), "0.122.0");
      assert.equal(requestCount, 2);
    } finally {
      await new Promise<void>((resolve, reject) => {
        server.close((err) => (err ? reject(err) : resolve()));
      });
    }
  });
});
