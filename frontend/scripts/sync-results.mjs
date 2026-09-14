#!/usr/bin/env node
/**
 * Copies real measured results from the Python project's results/ directory
 * into src/data/generated/ so the frontend can `import` them directly
 * (Vite + TS resolveJsonModule handles plain JSON imports natively).
 *
 * This is the ONLY path by which numbers enter the bundle from results/ —
 * nothing in src/ should hand-type a benchmark number. Run automatically via
 * `predev`/`prebuild` (see package.json), or manually:
 *   node scripts/sync-results.mjs
 *
 * Missing source files are not an error: they emit `null` into the
 * generated index so a scenario that hasn't been run yet (S4-S9 as of this
 * writing) renders its "not yet run" empty state instead of failing the
 * build.
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const FRONTEND_ROOT = join(__dirname, "..");
const REPO_ROOT = join(FRONTEND_ROOT, "..");
const RESULTS_DIR = join(REPO_ROOT, "results");
const OUT_DIR = join(FRONTEND_ROOT, "src", "data", "generated");

mkdirSync(OUT_DIR, { recursive: true });

/** @type {{ src: string, out: string }[]} */
const FILES = [
  { src: "scenarios/s1.json", out: "s1.json" },
  { src: "scenarios/s2.json", out: "s2.json" },
  { src: "scenarios/s3.json", out: "s3.json" },
  { src: "scenarios/s7.json", out: "s7.json" },
  { src: "scenarios/s10.json", out: "s10.json" },
  { src: "qwen14b_live_metrics.json", out: "qwen14b_live_metrics.json" },
  { src: "qwen14b_benchmark_results.json", out: "qwen14b_benchmark_results.json" },
  { src: "benchmark_results.json", out: "benchmark_results.json" },
];

const manifest = { generatedAt: new Date().toISOString(), files: {} };

for (const { src, out } of FILES) {
  const srcPath = join(RESULTS_DIR, src);
  const outPath = join(OUT_DIR, out);
  const key = out.replace(/\.json$/, "");
  if (existsSync(srcPath)) {
    const raw = readFileSync(srcPath, "utf-8");
    // Round-trip through JSON.parse/stringify to fail loudly on malformed
    // source rather than shipping broken JSON into the bundle silently.
    const parsed = JSON.parse(raw);
    writeFileSync(outPath, JSON.stringify(parsed, null, 2) + "\n");
    manifest.files[key] = { present: true, source: `results/${src}` };
    console.log(`[sync-results] ${src} -> src/data/generated/${out}`);
  } else {
    writeFileSync(outPath, "null\n");
    manifest.files[key] = { present: false, source: `results/${src}` };
    console.log(`[sync-results] ${src} not found -> src/data/generated/${out} = null`);
  }
}

writeFileSync(join(OUT_DIR, "manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
console.log(`[sync-results] wrote manifest.json (${Object.values(manifest.files).filter(f => f.present).length}/${FILES.length} present)`);
