import { describe, it } from "node:test";
import * as assert from "node:assert/strict";
import fs from "node:fs";
import { syncBuiltinESMExports } from "node:module";
import * as os from "node:os";
import * as path from "node:path";
import { createServer } from "node:http";
import type { Socket } from "node:net";
import { Duplex } from "node:stream";
import express from "express";
import { createAuthManagementRouter } from "../auth-management.js";
import { DeviceLoginManager } from "../device-login.js";
import { loadTokenData } from "../auth.js";

// A real Express/HTTP parser over an in-memory duplex, with no listening socket.
async function request(app: express.Express, method: string, url: string) {
  const output: Buffer[] = [];
  const socket = new Duplex({ read() {}, write(chunk, _encoding, callback) { output.push(Buffer.from(chunk)); callback(); } });
  createServer(app).emit("connection", socket as unknown as Socket);
  const finished = new Promise<void>((resolve, reject) => { socket.once("finish", resolve); socket.once("error", reject); });
  const timeout = setTimeout(() => socket.destroy(new Error("HTTP request timed out")), 2000);
  try {
    socket.push(`${method} ${url} HTTP/1.1\r\nhost: local.test\r\nauthorization: Bearer fake-admin\r\nconnection: close\r\ncontent-length: 0\r\n\r\n`);
    await finished;
  } finally { clearTimeout(timeout); socket.destroy(); }
  const raw = Buffer.concat(output).toString("utf8");
  const split = raw.indexOf("\r\n\r\n");
  assert.notEqual(split, -1);
  const [statusLine, ...headerLines] = raw.slice(0, split).split("\r\n");
  const headers = new Map(headerLines.map((line) => {
    const colon = line.indexOf(":");
    assert.notEqual(colon, -1);
    return [line.slice(0, colon).toLowerCase(), line.slice(colon + 1).trim()] as const;
  }));
  let body = raw.slice(split + 4);
  if (headers.get("transfer-encoding")?.toLowerCase() === "chunked") {
    const end = body.indexOf("\r\n");
    body = body.slice(end + 2, end + 2 + parseInt(body.slice(0, end), 16));
  }
  assert.equal(statusLine.split(" ")[0], "HTTP/1.1");
  return { status: Number(statusLine.split(" ")[1]), cache: headers.get("cache-control"), body: JSON.parse(body) as Record<string, unknown> };
}

function replaceFs(name: string, replacement: unknown): () => void {
  const mutable = fs as unknown as Record<string, unknown>;
  const previous = mutable[name];
  mutable[name] = replacement;
  syncBuiltinESMExports();
  return () => { mutable[name] = previous; syncBuiltinESMExports(); };
}
const jwt = (account: string) => `e30.${Buffer.from(JSON.stringify({ "https://api.openai.com/auth": { chatgpt_account_id: account } })).toString("base64url")}.signature`;
const json = (value: unknown) => new Response(JSON.stringify(value), { headers: { "content-type": "application/json" } });

describe("device login persistence failures through management HTTP", () => {
  for (const stage of ["write", "directory-fsync"] as const) {
    it(`reports ${stage} failure without claiming cancellation`, async () => {
      if (stage === "directory-fsync" && process.platform === "win32") return;
      const dir = fs.mkdtempSync(path.join(os.tmpdir(), "device-http-persistence-"));
      const file = path.join(dir, "auth.json");
      const old = JSON.stringify({ tokens: { access_token: jwt("old-account"), id_token: jwt("old-account"), refresh_token: "old-refresh" } });
      fs.writeFileSync(file, old);
      const previousKey = process.env.CODEX_AS_API_ADMIN_KEY;
      process.env.CODEX_AS_API_ADMIN_KEY = "fake-admin";
      const replies = [json({ device_auth_id: "fake-device-secret", user_code: "FAKE-CODE", interval: 1 }), json({ authorization_code: "fake-authorization-secret", code_verifier: "fake-verifier-secret", code_challenge: "fake-challenge-secret" }), json({ access_token: jwt("new-account"), id_token: jwt("new-account"), refresh_token: "new-refresh-secret" })];
      let calls = 0, injected = 0, renamed = false;
      const manager = new DeviceLoginManager(file, { fetch: async () => { calls++; const response = replies.shift(); if (!response) throw Error("unexpected transport request"); return response; } });
      const app = express();
      app.use("/auth", createAuthManagementRouter(manager, () => "authenticated"));
      const originalOpen = fs.openSync, originalWrite = fs.writeFileSync, originalFsync = fs.fsyncSync, originalRename = fs.renameSync;
      let tempFd: number | undefined, directoryFd: number | undefined;
      const restoreOpen = replaceFs("openSync", (...args: Parameters<typeof fs.openSync>) => {
        const fd = originalOpen(...args);
        if (typeof args[0] === "string" && args[0].startsWith(path.join(dir, ".auth.json.tmp-")) && args[1] === "wx") tempFd = fd;
        if (args[0] === dir && args[1] === "r") directoryFd = fd;
        return fd;
      });
      const restoreWrite = replaceFs("writeFileSync", (...args: Parameters<typeof fs.writeFileSync>) => {
        if (stage === "write" && args[0] === tempFd && tempFd !== undefined) { injected++; throw Error("private simulated write failure"); }
        return originalWrite(...args);
      });
      const restoreRename = replaceFs("renameSync", (...args: Parameters<typeof fs.renameSync>) => {
        originalRename(...args);
        if (args[1] === file) renamed = true;
      });
      const restoreFsync = replaceFs("fsyncSync", (fd: number) => {
        if (stage === "directory-fsync" && fd === directoryFd && directoryFd !== undefined) { injected++; assert.equal(renamed, true); throw Error("private simulated directory failure"); }
        originalFsync(fd);
      });
      try {
        const start = await request(app, "POST", "/auth/login");
        assert.equal(start.status, 202); assert.equal(start.cache, "no-store");
        assert.equal(typeof start.body.id, "string");
        const endpoint = `/auth/login/${start.body.id}`;
        let result = await request(app, "GET", endpoint);
        for (let i = 0; i < 50 && result.body.status === "pending"; i++) {
          await new Promise<void>((resolve) => setImmediate(resolve));
          result = await request(app, "GET", endpoint);
        }
        const expected = stage === "write" ? "login_storage_error" : "login_persistence_uncertain";
        assert.equal(result.status, 200); assert.equal(result.cache, "no-store");
        assert.equal(result.body.status, "failed"); assert.equal(result.body.error_code, expected);
        assert.deepEqual(Object.keys(result.body).sort(), ["error_code", "expires_at", "id", "method", "status"]);
        assert.equal(result.body.method, "device");
        for (const secret of ["fake-device-secret", "FAKE-CODE", "fake-authorization-secret", "fake-verifier-secret", "new-refresh-secret", "private simulated", "old-refresh"]) {
          assert.equal(JSON.stringify(result.body).includes(secret), false);
        }
        const cancelled = await request(app, "DELETE", endpoint);
        assert.equal(cancelled.status, 200); assert.equal(cancelled.cache, "no-store");
        assert.deepEqual(cancelled.body, result.body);
        assert.equal(calls, 3); assert.equal(injected, 1);
        assert.equal(renamed, stage === "directory-fsync");
        if (stage === "write") { assert.equal(fs.readFileSync(file, "utf8"), old); assert.equal(loadTokenData(file).account_id, "old-account"); }
        else { assert.equal(loadTokenData(file).account_id, "new-account"); assert.equal(loadTokenData(file).refresh_token, "new-refresh-secret"); }
        assert.deepEqual(fs.readdirSync(dir).filter((name) => name.includes(".tmp-")), []);
      } finally {
        restoreFsync(); restoreRename(); restoreWrite(); restoreOpen(); manager.close();
        if (previousKey === undefined) delete process.env.CODEX_AS_API_ADMIN_KEY; else process.env.CODEX_AS_API_ADMIN_KEY = previousKey;
        fs.rmSync(dir, { recursive: true, force: true });
      }
    });
  }
});
