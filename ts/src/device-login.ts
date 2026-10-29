/*
 * Modified TypeScript port of the device authorization protocol in OpenAI
 * Codex rust-v0.164.0 (commit 6aaac694803004a63c6f393d153c96d22521a6b9),
 * codex-rs/login/src/device_code_auth.rs.
 * Original source: Copyright 2025 OpenAI; licensed under the Apache License 2.0.
 * This implementation has been modified for codex-as-api.
 * SPDX-License-Identifier: Apache-2.0
 */
import { randomUUID } from "node:crypto";
import { resolve } from "node:path";
import { performance } from "node:perf_hooks";
import { commitLoginCredentials, LoginCommitExpiredError, LoginPersistenceUncertainError, resolveAuthPath } from "./auth.js";
import { reserveLogin, releaseLogin } from "./login-coordination.js";
import { parseJsonResponseStrict } from "./utf8-json.js";

// Protocol derived from OpenAI Codex rust-v0.164.0 device_code_auth.rs.
const ISSUER = "https://auth.openai.com";
const CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann";
const LIFETIME_MS = 900_000;
const MAX_RECORDS = 32;

export type DeviceLoginStatus = "pending" | "succeeded" | "failed" | "cancelled" | "expired";
export interface DeviceLoginSnapshot {
  id: string;
  method: "device";
  status: DeviceLoginStatus;
  expires_at: string;
  error_code?: string;
}
export interface DeviceLoginStart extends DeviceLoginSnapshot {
  status: "pending";
  verification_url: string;
  user_code: string;
  interval: number;
}
export interface DeviceLoginClock {
  now(): number;
  wallNow(): number;
  setTimeout(callback: () => void, ms: number): unknown;
  clearTimeout(handle: unknown): void;
}
export interface DeviceLoginOptions {
  fetch?: typeof fetch;
  clock?: DeviceLoginClock;
  commit?: (path: string, tokens: { access_token: string; refresh_token: string; id_token: string }, beforeReplace?: () => void) => void;
}
export class DeviceLoginError extends Error {
  constructor(readonly code: string, readonly status: number) {
    super(code === "login_in_progress" ? "A login is already in progress" : "Device login could not be started");
    this.name = "DeviceLoginError";
  }
}
const defaultClock: DeviceLoginClock = {
  now: () => performance.now(),
  wallNow: () => Date.now(),
  setTimeout: (callback, ms) => setTimeout(callback, ms),
  clearTimeout: (handle) => clearTimeout(handle as ReturnType<typeof setTimeout>),
};
interface Attempt {
  snapshot: DeviceLoginSnapshot;
  deadline: number;
  controller: AbortController;
  timer: unknown;
  stop: Promise<void>;
  signalStop: () => void;
}

function object(value: unknown): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("invalid");
  return value as Record<string, unknown>;
}
function text(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) throw new Error("invalid");
  return value;
}
function intervalSeconds(value: unknown): number {
  if (value === undefined) return 5;
  if (typeof value !== "number" && (typeof value !== "string" || !/^\d+$/.test(value))) throw new Error("invalid");
  const n = Number(value);
  if (!Number.isSafeInteger(n) || n <= 0) throw new Error("invalid");
  return n;
}

export class DeviceLoginManager {
  private readonly path: string;
  private readonly transport: typeof fetch;
  private readonly clock: DeviceLoginClock;
  private readonly commit: NonNullable<DeviceLoginOptions["commit"]>;
  private readonly records = new Map<string, Attempt>();
  private closed = false;

  constructor(authPath?: string | null, options: DeviceLoginOptions = {}) {
    this.path = resolve(resolveAuthPath(authPath));
    this.transport = options.fetch ?? fetch;
    this.clock = options.clock ?? defaultClock;
    this.commit = options.commit ?? commitLoginCredentials;
  }

  private finish(attempt: Attempt, status: Exclude<DeviceLoginStatus, "pending">, errorCode?: string): void {
    if (attempt.snapshot.status !== "pending") return;
    attempt.snapshot.status = status;
    if (errorCode) attempt.snapshot.error_code = errorCode;
    this.clock.clearTimeout(attempt.timer);
    attempt.controller.abort();
    attempt.signalStop();
    releaseLogin(this.path, attempt);
    this.trim();
  }
  private trim(): void {
    while (this.records.size > MAX_RECORDS) {
      const old = [...this.records].find(([, a]) => a.snapshot.status !== "pending");
      if (!old) break;
      this.records.delete(old[0]);
    }
  }
  private check(attempt: Attempt): void {
    if (attempt.snapshot.status === "pending" && this.clock.now() >= attempt.deadline) this.finish(attempt, "expired");
    if (attempt.snapshot.status !== "pending") throw new Error("stopped");
  }
  private async guarded<T>(attempt: Attempt, operation: Promise<T>): Promise<T> {
    this.check(attempt);
    const result = await Promise.race([operation, attempt.stop.then(() => { throw new Error("stopped"); })]);
    this.check(attempt);
    return result;
  }
  private async request(attempt: Attempt, url: string, body: string, contentType: string): Promise<Response> {
    this.check(attempt);
    return this.guarded(attempt, this.transport(url, {
      method: "POST", redirect: "manual", headers: { "Content-Type": contentType },
      body, signal: attempt.controller.signal,
    }));
  }
  private async json(attempt: Attempt, response: Response): Promise<Record<string, unknown>> {
    return object(await this.guarded(attempt, parseJsonResponseStrict(response)));
  }
  private async wait(attempt: Attempt, ms: number): Promise<void> {
    this.check(attempt);
    let timer: unknown;
    try {
      await this.guarded(attempt, new Promise<void>((resolveWait) => {
        timer = this.clock.setTimeout(resolveWait, Math.min(ms, Math.max(0, attempt.deadline - this.clock.now())));
      }));
    } finally {
      if (timer !== undefined) this.clock.clearTimeout(timer);
    }
  }

  async start(): Promise<DeviceLoginStart> {
    if (this.closed) throw new DeviceLoginError("login_unavailable", 503);
    const id = randomUUID();
    let signalStop!: () => void;
    const stop = new Promise<void>((resolveStop) => { signalStop = resolveStop; });
    const attempt: Attempt = {
      snapshot: { id, method: "device", status: "pending", expires_at: new Date(this.clock.wallNow() + LIFETIME_MS).toISOString() },
      deadline: this.clock.now() + LIFETIME_MS,
      controller: new AbortController(), timer: undefined, stop, signalStop,
    };
    if (!reserveLogin(this.path, attempt)) throw new DeviceLoginError("login_in_progress", 409);
    this.records.set(id, attempt);
    attempt.timer = this.clock.setTimeout(() => this.finish(attempt, "expired"), LIFETIME_MS);
    let response: Response;
    try {
      response = await this.request(attempt, `${ISSUER}/api/accounts/deviceauth/usercode`, JSON.stringify({ client_id: CLIENT_ID }), "application/json");
    } catch {
      this.finish(attempt, "failed", "login_upstream_error");
      throw new DeviceLoginError("login_upstream_error", 502);
    }
    if (!response.ok) {
      const code = response.status === 404 ? "device_login_unsupported" : "login_upstream_error";
      this.finish(attempt, "failed", code);
      throw new DeviceLoginError(code, 502);
    }
    let userCode: string, deviceId: string, interval: number;
    try {
      const data = await this.json(attempt, response);
      userCode = text(data["user_code"] ?? data["usercode"]);
      deviceId = text(data["device_auth_id"]);
      interval = intervalSeconds(data["interval"]);
      this.check(attempt);
    } catch {
      this.finish(attempt, "failed", "invalid_login_response");
      throw new DeviceLoginError("invalid_login_response", 502);
    }
    void this.poll(attempt, deviceId, userCode, interval);
    return { ...attempt.snapshot, status: "pending", verification_url: `${ISSUER}/codex/device`, user_code: userCode, interval };
  }

  private async poll(attempt: Attempt, deviceId: string, userCode: string, interval: number): Promise<void> {
    try {
      for (;;) {
        const response = await this.request(attempt, `${ISSUER}/api/accounts/deviceauth/token`, JSON.stringify({ device_auth_id: deviceId, user_code: userCode }), "application/json");
        if (response.status === 403 || response.status === 404) {
          await this.wait(attempt, interval * 1000);
          continue;
        }
        if (!response.ok) { this.finish(attempt, "failed", "login_upstream_error"); return; }
        let authorizationCode: string, verifier: string;
        try {
          const data = await this.json(attempt, response);
          authorizationCode = text(data["authorization_code"]);
          verifier = text(data["code_verifier"]);
          text(data["code_challenge"]);
        } catch {
          this.finish(attempt, "failed", "invalid_login_response"); return;
        }
        const form = new URLSearchParams({ grant_type: "authorization_code", client_id: CLIENT_ID,
          code: authorizationCode, redirect_uri: `${ISSUER}/deviceauth/callback`, code_verifier: verifier });
        const exchange = await this.request(attempt, `${ISSUER}/oauth/token`, form.toString(), "application/x-www-form-urlencoded");
        if (!exchange.ok) { this.finish(attempt, "failed", "login_upstream_error"); return; }
        let tokens: { access_token: string; refresh_token: string; id_token: string };
        try {
          const data = await this.json(attempt, exchange);
          tokens = { access_token: text(data["access_token"]), refresh_token: text(data["refresh_token"]), id_token: text(data["id_token"]) };
        } catch { this.finish(attempt, "failed", "invalid_login_response"); return; }
        this.check(attempt);
        try {
          this.commit(this.path, tokens, () => {
            if (attempt.snapshot.status !== "pending" || this.clock.now() >= attempt.deadline) {
              if (attempt.snapshot.status === "pending") this.finish(attempt, "expired");
              throw new LoginCommitExpiredError();
            }
          });
        }
        catch (error) {
          this.finish(attempt, "failed", error instanceof LoginPersistenceUncertainError
            ? "login_persistence_uncertain" : "login_storage_error");
          return;
        }
        this.finish(attempt, "succeeded");
        return;
      }
    } catch {
      this.finish(attempt, "failed", "login_upstream_error");
    }
  }
  get(id: string): DeviceLoginSnapshot | undefined {
    const attempt = this.records.get(id);
    if (!attempt) return undefined;
    if (attempt.snapshot.status === "pending" && this.clock.now() >= attempt.deadline) this.finish(attempt, "expired");
    return { ...attempt.snapshot };
  }
  cancel(id: string): DeviceLoginSnapshot | undefined {
    const attempt = this.records.get(id);
    if (!attempt) return undefined;
    if (attempt.snapshot.status === "pending" && this.clock.now() >= attempt.deadline) this.finish(attempt, "expired");
    else this.finish(attempt, "cancelled");
    return { ...attempt.snapshot };
  }
  close(): void {
    this.closed = true;
    for (const attempt of this.records.values()) this.finish(attempt, "cancelled");
  }
}
