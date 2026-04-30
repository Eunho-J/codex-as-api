import { defineConfig } from "tsup";

export default defineConfig([
  {
    entry: ["src/index.ts"],
    format: ["esm", "cjs"],
    dts: true,
    clean: true,
    splitting: false,
  },
  {
    entry: ["src/cli.ts"],
    format: ["esm"],
    clean: false,
    splitting: false,
    banner: { js: "#!/usr/bin/env node" },
  },
]);
