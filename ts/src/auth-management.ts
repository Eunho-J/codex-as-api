import * as crypto from "node:crypto";
import express, { type Request, type Response } from "express";
import { decodeUtf8Strict, parseJsonStrict } from "./utf8-json.js";

export type LoginMethod = "browser" | "device";
export interface LoginRecord {
  id: string;
  method?: LoginMethod;
  status: "pending" | "succeeded" | "failed" | "cancelled" | "expired";
  expires_at: string;
  error_code?: string;
}
export interface LoginStart extends LoginRecord {
  auth_url?: string;
  verification_url?: string;
  user_code?: string;
  interval?: number;
}
export interface LoginManager {
  start(method?: LoginMethod): Promise<LoginStart>;
  get(id: string): LoginRecord | undefined | Promise<LoginRecord | undefined>;
  cancel(id: string): LoginRecord | undefined | Promise<LoginRecord | undefined>;
  close(): void;
}

// Each underlying manager retains its own bounded records and coordinates the
// per-store active slot with the other method. No authorization material is
// copied into this dispatcher.
export class CombinedLoginManager implements LoginManager {
  constructor(private readonly browser: LoginManager, private readonly device: LoginManager) {}

  async start(method: LoginMethod = "browser"): Promise<LoginStart> {
    const result = await (method === "browser" ? this.browser : this.device).start();
    return { ...result, method };
  }

  async get(id: string): Promise<LoginRecord | undefined> {
    const browser = await this.browser.get(id);
    if (browser) return { ...browser, method: "browser" };
    const device = await this.device.get(id);
    return device ? { ...device, method: "device" } : undefined;
  }

  async cancel(id: string): Promise<LoginRecord | undefined> {
    // Look up the owner before cancellation so an unknown id is never sent to
    // both managers as a mutating operation.
    if (await this.browser.get(id)) {
      const record = await this.browser.cancel(id);
      return record ? { ...record, method: "browser" } : undefined;
    }
    const record = await this.device.cancel(id);
    return record ? { ...record, method: "device" } : undefined;
  }

  close(): void {
    try { this.browser.close(); } finally { this.device.close(); }
  }
}

type LocalStatus = "authenticated" | "unauthenticated" | "reauthentication_required";
const messages: Record<string, string> = {
  management_disabled: "Management API is disabled.",
  invalid_admin_key: "Invalid admin key.",
  origin_not_allowed: "Origin is not allowed.",
  invalid_request: "Invalid request body.",
  login_not_found: "Login not found.",
  login_in_progress: "A login is already in progress.",
  callback_port_unavailable: "Browser login callback ports are unavailable.",
  device_login_unsupported: "Device login is not supported.",
  login_upstream_error: "Login could not be started.",
  invalid_login_response: "Invalid login response.",
};
function error(res: Response, status: number, code: string): void {
  res.status(status).json({ error: { code, message: messages[code] } });
}
function sameKey(actual: string, expected: string): boolean {
  const a = Buffer.from(actual);
  const b = Buffer.from(expected);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}
function publicRecord(record: LoginRecord, method: LoginMethod): LoginRecord {
  return { id: record.id, method, status: record.status, expires_at: record.expires_at,
    ...(record.error_code ? { error_code: record.error_code } : {}) };
}

// This router is mounted before the inference transport middleware so management
// authentication and no-store apply even to malformed or empty request bodies.
export function createAuthManagementRouter(manager: LoginManager, status: () => LocalStatus): express.Router {
  const router = express.Router();
  // Compatibility with injected legacy managers that do not include a method.
  const methods = new Map<string, LoginMethod>();
  router.use((req, res, next) => {
    res.set("Cache-Control", "no-store");
    const key = process.env.CODEX_AS_API_ADMIN_KEY;
    if (!key) return error(res, 503, "management_disabled");
    const header = req.headers.authorization;
    const match = typeof header === "string" ? /^Bearer ([^\s]+)$/i.exec(header) : null;
    if (!match || !sameKey(match[1], key)) return error(res, 401, "invalid_admin_key");
    const origin = req.headers.origin;
    if (origin !== undefined) {
      const host = req.headers.host;
      if (!host || origin !== `${req.protocol}://${host}`) return error(res, 403, "origin_not_allowed");
    }
    next();
  });
  router.post("/login", express.raw({ type: () => true, limit: "4kb" }), async (req: Request, res: Response) => {
    try {
      const body = req.body as Buffer | undefined;
      let method: LoginMethod = "browser";
      if (body && body.length) {
        if (!/^application\/(?:json|[\w.+-]+\+json)(?:\s*;|\s*$)/i.test(req.headers["content-type"] ?? "")) {
          return error(res, 400, "invalid_request");
        }
        let value: unknown;
        try { value = parseJsonStrict(decodeUtf8Strict(body)); }
        catch { return error(res, 400, "invalid_request"); }
        if (value === null || typeof value !== "object" || Array.isArray(value)) {
          return error(res, 400, "invalid_request");
        }
        const fields = value as Record<string, unknown>;
        if (Object.keys(fields).some((key) => key !== "method")
          || (Object.hasOwn(fields, "method") && fields.method !== "browser" && fields.method !== "device")) {
          return error(res, 400, "invalid_request");
        }
        if (fields.method === "device") method = "device";
      }
      const result = await manager.start(method);
      // Old injected device-only managers have no method property.
      const actualMethod = result.method ?? (result.verification_url !== undefined ? "device" : method);
      methods.set(result.id, actualMethod);
      if (methods.size > 64) methods.delete(methods.keys().next().value!);
      const details = actualMethod === "browser"
        ? { auth_url: result.auth_url }
        : { verification_url: result.verification_url, user_code: result.user_code, interval: result.interval };
      res.status(202).json({ ...publicRecord(result, actualMethod), ...details });
    } catch (err) {
      const code = typeof err === "object" && err !== null && "code" in err ? (err as { code: unknown }).code : undefined;
      if (code === "login_in_progress" || code === "callback_port_unavailable") return error(res, 409, code);
      if (code === "device_login_unsupported" || code === "invalid_login_response" || code === "login_upstream_error") {
        return error(res, 502, code);
      }
      // Parsing failures and upstream exceptions must never expose their details.
      return error(res, 502, "login_upstream_error");
    }
  });
  router.get("/login/:id", async (req, res) => {
    try {
      const record = await manager.get(req.params.id);
      if (!record) return error(res, 404, "login_not_found");
      res.json(publicRecord(record, record.method ?? methods.get(record.id) ?? "browser"));
    } catch { error(res, 502, "login_upstream_error"); }
  });
  router.delete("/login/:id", async (req, res) => {
    try {
      const record = await manager.cancel(req.params.id);
      if (!record) return error(res, 404, "login_not_found");
      res.json(publicRecord(record, record.method ?? methods.get(record.id) ?? "browser"));
    } catch { error(res, 502, "login_upstream_error"); }
  });
  router.get("/status", (_req, res) => {
    try { res.json({ status: status() }); }
    catch { res.json({ status: "reauthentication_required" }); }
  });
  router.use((_req, res) => error(res, 404, "login_not_found"));
  router.use((_err: unknown, _req: Request, res: Response, _next: express.NextFunction) => error(res, 400, "invalid_request"));
  return router;
}
