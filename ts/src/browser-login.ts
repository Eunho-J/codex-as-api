/*
 * Modified TypeScript port of OpenAI Codex rust-v0.164.0 browser OAuth
 * (commit 6aaac694803004a63c6f393d153c96d22521a6b9):
 * codex-rs/login/src/server.rs and oauth/{authorization,pkce,client}.rs.
 * Copyright 2025 OpenAI; Apache-2.0. Modified for codex-as-api.
 * SPDX-License-Identifier: Apache-2.0
 */
import { createHash, randomBytes, randomUUID } from "node:crypto";
import { createServer, type Server, type ServerResponse } from "node:http";
import { resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { commitLoginCredentials, LoginCommitExpiredError, LoginPersistenceUncertainError, resolveAuthPath } from "./auth.js";
import { reserveLogin, releaseLogin } from "./login-coordination.js";
import { parseJsonResponseStrict } from "./utf8-json.js";
import type { DeviceLoginClock } from "./device-login.js";

const ISSUER = "https://auth.openai.com";
const CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const LIFETIME_MS = 900_000;
const MAX_RECORDS = 32;
export type BrowserLoginStatus = "pending" | "succeeded" | "failed" | "cancelled" | "expired";
export interface BrowserLoginSnapshot {
  id: string;
  method: "browser";
  status: BrowserLoginStatus;
  expires_at: string;
  error_code?: string;
}
export interface BrowserLoginStart extends BrowserLoginSnapshot {
  status: "pending";
  auth_url: string;
}
export interface BrowserLoginOptions {
  fetch?: typeof fetch;
  clock?: DeviceLoginClock;
  ports?: number[];
}
export class BrowserLoginError extends Error {
  constructor(readonly code: string, readonly status: number) {
    super(code === "login_in_progress" ? "A login is already in progress" : "Browser login could not be started");
    this.name = "BrowserLoginError";
  }
}
const defaultClock: DeviceLoginClock = {
  now: () => performance.now(), wallNow: () => Date.now(),
  setTimeout: (callback, ms) => setTimeout(callback, ms),
  clearTimeout: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
};
interface Attempt {
  snapshot: BrowserLoginSnapshot;
  deadline: number;
  controller: AbortController;
  timer: unknown;
  stop: Promise<void>;
  signalStop: () => void;
  server?: Server;
  response?: ServerResponse;
  claimed: boolean;
  state: string;
  verifier: string;
  redirect: string;
}
function page(res: ServerResponse, status: number, text: string): void {
  if (res.writableEnded || res.destroyed) return;
  res.writeHead(status, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "X-Content-Type-Options": "nosniff", "Connection": "close" });
  res.end(`<!doctype html><title>Login</title><p>${text}</p>`);
}
function text(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) throw new Error("invalid");
  return value;
}

export class BrowserLoginManager {
  private readonly path: string;
  private readonly transport: typeof fetch;
  private readonly clock: DeviceLoginClock;
  private readonly ports: number[];
  private readonly records = new Map<string, Attempt>();
  private closed = false;
  constructor(authPath?: string | null, options: BrowserLoginOptions = {}) {
    this.path = resolve(resolveAuthPath(authPath));
    this.transport = options.fetch ?? fetch;
    this.clock = options.clock ?? defaultClock;
    this.ports = options.ports ?? [1455, 1457];
  }
  private finish(a: Attempt, status: Exclude<BrowserLoginStatus, "pending">, errorCode?: string): void {
    if (a.snapshot.status !== "pending") return;
    a.snapshot.status = status;
    if (errorCode) a.snapshot.error_code = errorCode;
    this.clock.clearTimeout(a.timer);
    a.controller.abort();
    a.signalStop();
    if (a.server) a.server.close();
    if (a.response) page(a.response, status === "succeeded" ? 200 : 400, status === "succeeded" ? "Login completed. You may close this window." : "Login did not complete. You may close this window.");
    releaseLogin(this.path, a);
    while (this.records.size > MAX_RECORDS) {
      const old = [...this.records].find(([, item]) => item.snapshot.status !== "pending");
      if (!old) break;
      this.records.delete(old[0]);
    }
  }
  private check(a: Attempt): void {
    if (a.snapshot.status === "pending" && this.clock.now() >= a.deadline) this.finish(a, "expired");
    if (a.snapshot.status !== "pending") throw new Error("stopped");
  }
  private async guarded<T>(a: Attempt, operation: Promise<T>): Promise<T> {
    this.check(a);
    try {
      const value = await Promise.race([operation, a.stop.then(() => { throw new Error("stopped"); })]);
      this.check(a);
      return value;
    } catch (error) {
      this.check(a);
      throw error;
    }
  }
  async start(): Promise<BrowserLoginStart> {
    if (this.closed) throw new BrowserLoginError("login_unavailable", 503);
    let signalStop!: () => void;
    const stop = new Promise<void>((done) => { signalStop = done; });
    const a: Attempt = {
      snapshot: { id: randomUUID(), method: "browser", status: "pending", expires_at: new Date(this.clock.wallNow() + LIFETIME_MS).toISOString() },
      deadline: this.clock.now() + LIFETIME_MS, controller: new AbortController(), timer: undefined, stop, signalStop,
      claimed: false, state: randomBytes(32).toString("base64url"), verifier: randomBytes(32).toString("base64url"), redirect: "",
    };
    if (!reserveLogin(this.path, a)) throw new BrowserLoginError("login_in_progress", 409);
    this.records.set(a.snapshot.id, a);
    a.timer = this.clock.setTimeout(() => this.finish(a, "expired"), LIFETIME_MS);
    for (const port of this.ports) {
      try {
        this.check(a);
        const server = createServer({ maxHeaderSize: 8192 }, (req, res) => {
          const hosts = req.rawHeaders.filter((_, i) => i % 2 === 0 && req.rawHeaders[i]?.toLowerCase() === "host");
          if (a.snapshot.status !== "pending" || a.claimed || req.method !== "GET" || hosts.length !== 1 || req.headers.host !== `127.0.0.1:${(server.address() as { port: number } | null)?.port}` || !req.url || req.url.length > 8192 || !req.url.startsWith("/auth/callback?")) {
            page(res, 400, "Invalid login callback."); return;
          }
          let url: URL;
          try { url = new URL(req.url, "http://127.0.0.1"); } catch { page(res, 400, "Invalid login callback."); return; }
          const params = url.searchParams;
          const states = params.getAll("state"), codes = params.getAll("code"), errors = params.getAll("error");
          if (url.pathname !== "/auth/callback" || states.length !== 1 || !states[0] || states[0] !== a.state || codes.length > 1 || errors.length > 1 || (codes.length && errors.length) || (!codes.length && !errors.length) || (codes.length && !codes[0]) || (errors.length && !errors[0])) {
            page(res, 400, "Invalid login callback."); return;
          }
          try { this.check(a); } catch { page(res, 400, "Login did not complete."); return; }
          a.claimed = true;
          a.response = res;
          if (errors.length) { this.finish(a, "failed", "login_denied"); return; }
          void this.exchange(a, codes[0]!);
        });
        server.on("clientError", (_error, socket) => {
          if (socket.writable) socket.end("HTTP/1.1 400 Bad Request\r\nCache-Control: no-store\r\nReferrer-Policy: no-referrer\r\nConnection: close\r\nContent-Length: 0\r\n\r\n");
          else socket.destroy();
        });
        server.requestTimeout = 10_000;
        server.headersTimeout = 10_000;
        a.server = server;
        const listening = new Promise<void>((ok, fail) => {
          server.once("error", fail);
          server.listen(port, "127.0.0.1", () => {
            server.removeListener("error", fail);
            if (a.snapshot.status !== "pending") server.close();
            ok();
          });
        });
        try { await this.guarded(a, listening); } catch (err) { server.close(); throw err; }
        const address = server.address();
        if (!address || typeof address === "string") throw new Error("listen");
        a.redirect = `http://127.0.0.1:${address.port}/auth/callback`;
        const query = new URLSearchParams({ response_type: "code", client_id: CLIENT_ID, redirect_uri: a.redirect,
          state: a.state, code_challenge: createHash("sha256").update(a.verifier).digest("base64url"), code_challenge_method: "S256",
          scope: "openid profile email offline_access api.connectors.read api.connectors.invoke",
          id_token_add_organizations: "true", codex_cli_simplified_flow: "true", originator: "codex_cli_rs" });
        return { ...a.snapshot, status: "pending", auth_url: `${ISSUER}/oauth/authorize?${query}` };
      } catch (err) {
        if (a.snapshot.status !== "pending") throw new BrowserLoginError("login_unavailable", 503);
        if ((err as NodeJS.ErrnoException).code !== "EADDRINUSE") {
          this.finish(a, "failed", "login_unavailable");
          throw new BrowserLoginError("login_unavailable", 503);
        }
      }
    }
    this.finish(a, "failed", "callback_port_unavailable");
    throw new BrowserLoginError("callback_port_unavailable", 409);
  }
  private async exchange(a: Attempt, code: string): Promise<void> {
    try {
      const form = new URLSearchParams({ grant_type: "authorization_code", client_id: CLIENT_ID, code, redirect_uri: a.redirect, code_verifier: a.verifier });
      const response = await this.guarded(a, this.transport(`${ISSUER}/oauth/token`, { method: "POST", redirect: "manual", headers: { "Content-Type": "application/x-www-form-urlencoded" }, body: form.toString(), signal: a.controller.signal }));
      if (!response.ok) { this.finish(a, "failed", "login_upstream_error"); return; }
      let tokens: Record<string, unknown>;
      try {
        const data = await this.guarded(a, parseJsonResponseStrict(response));
        if (!data || typeof data !== "object" || Array.isArray(data)) throw new Error("invalid");
        const obj = data as Record<string, unknown>;
        tokens = { access_token: text(obj["access_token"]), refresh_token: text(obj["refresh_token"]), id_token: text(obj["id_token"]) };
      } catch { this.finish(a, "failed", "invalid_login_response"); return; }
      this.check(a);
      try { commitLoginCredentials(this.path, tokens, () => {
        if (a.snapshot.status !== "pending" || this.clock.now() >= a.deadline) throw new LoginCommitExpiredError();
      }); }
      catch (error) {
        if (error instanceof LoginCommitExpiredError) this.finish(a, "expired");
        else this.finish(a, "failed", error instanceof LoginPersistenceUncertainError ? "login_persistence_uncertain" : "login_storage_error");
        return;
      }
      this.finish(a, "succeeded");
    } catch { this.finish(a, "failed", "login_upstream_error"); }
  }
  get(id: string): BrowserLoginSnapshot | undefined {
    const a = this.records.get(id); if (!a) return undefined;
    if (a.snapshot.status === "pending" && this.clock.now() >= a.deadline) this.finish(a, "expired");
    return { ...a.snapshot };
  }
  cancel(id: string): BrowserLoginSnapshot | undefined {
    const a = this.records.get(id); if (!a) return undefined;
    if (a.snapshot.status === "pending" && this.clock.now() >= a.deadline) this.finish(a, "expired");
    else this.finish(a, "cancelled");
    return { ...a.snapshot };
  }
  close(): void { this.closed = true; for (const a of this.records.values()) this.finish(a, "cancelled"); }
}
