import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { defineConfig } from "tsup";

const rankPath = fileURLToPath(new URL("../config/o200k_base.tiktoken", import.meta.url));
const rankData = readFileSync(rankPath, "utf8");
const rankSha256 = createHash("sha256").update(rankData).digest("hex");
const expectedRankSha256 =
  "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d";
if (rankSha256 !== expectedRankSha256) {
  throw new Error(
    `o200k_base.tiktoken SHA-256 mismatch: expected ${expectedRankSha256}, got ${rankSha256}`,
  );
}
const embeddedRankData = {
  __O200K_RANK_DATA__: JSON.stringify(rankData),
};

export default defineConfig([
  {
    entry: ["src/index.ts"],
    format: ["esm", "cjs"],
    dts: true,
    clean: true,
    splitting: false,
    define: embeddedRankData,
  },
  {
    entry: ["src/cli.ts"],
    format: ["esm"],
    clean: false,
    splitting: false,
    banner: { js: "#!/usr/bin/env node" },
    define: embeddedRankData,
  },
]);
