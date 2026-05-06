#!/usr/bin/env node
// Stage runtime DLLs for the Tauri bundler.
//
// On Windows we link sherpa-onnx in `shared` mode (see Cargo.toml
// for the MSVC ABI rationale), which means the final installer
// has to ship `sherpa-onnx-c-api.dll`, `onnxruntime.dll`, and a
// few siblings next to `vibe-to-text.exe`.
//
// `sherpa-onnx-sys`'s build script copies them into
// `src-tauri/target/<profile>/` so `cargo run` works in dev. For
// production bundling we need them at a STABLE path so
// `tauri.conf.json -> bundle.resources` can reference them; the
// bundler validates resource paths at build-script time and
// `target/release/` doesn't exist before the link step.
//
// This script runs as a Tauri `beforeBundleCommand` and copies the
// DLLs from `target/<profile>/` into `src-tauri/bundle-resources/`,
// where the resources glob picks them up.
//
// Cross-platform: on macOS / Linux we use sherpa-onnx in `static`
// mode — there's nothing to stage and this script exits quietly.

const fs = require("fs");
const path = require("path");

const repoRoot = path.resolve(__dirname, "..");
const tauriDir = path.join(repoRoot, "src-tauri");
const stageDir = path.join(tauriDir, "bundle-resources");

// Ensure the dir exists so Tauri's resources glob doesn't trip on
// missing-dir validation.
fs.mkdirSync(stageDir, { recursive: true });

// macOS uses `static` sherpa-onnx (clean ABI, single-file .app
// bundle). Windows + Linux use `shared` because sherpa-onnx-sys's
// prebuilt static libs ship their own copy of protobuf which
// collides with sentencepiece-sys's protobuf at link time. On
// shared builds, sherpa-onnx-sys's build script copies the runtime
// libs into target/<profile>/; we re-stage them here so Tauri's
// `bundle.resources` glob can pick them up alongside the executable
// in the installer.
if (process.platform === "darwin") {
  console.log(
    "stage-bundle-resources: skipping on darwin (sherpa-onnx is static here)."
  );
  process.exit(0);
}

// Tauri sets TAURI_ENV_DEBUG=true for `tauri dev` and false (or
// unset) for `tauri build`. Match that to the cargo profile.
const profile = process.env.TAURI_ENV_DEBUG === "true" ? "debug" : "release";
const targetDir = path.join(tauriDir, "target", profile);

// The list sherpa-onnx-sys's build script emits when the `shared`
// feature is on. The actual file extension + naming differs per
// platform: `.dll` on Windows, `.so` (with optional version
// suffixes like `.so.1.13.0`) on Linux. We glob for sensible name
// patterns rather than hard-code each file.
const PLATFORM_PATTERNS = {
  win32: [
    /^sherpa-onnx.*\.dll$/i,
    /^onnxruntime.*\.dll$/i,
  ],
  linux: [
    /^libsherpa-onnx.*\.so(\.\d+)*$/,
    /^libonnxruntime.*\.so(\.\d+)*$/,
  ],
};

const patterns = PLATFORM_PATTERNS[process.platform];
if (!patterns) {
  console.log(
    `stage-bundle-resources: no staging rules for ${process.platform}; skipping.`
  );
  process.exit(0);
}

if (!fs.existsSync(targetDir)) {
  console.error(
    `stage-bundle-resources: targetDir ${targetDir} doesn't exist.\n` +
      `  Run \`cargo build --release\` first.`
  );
  process.exit(1);
}

const entries = fs.readdirSync(targetDir);
let staged = 0;
for (const name of entries) {
  if (!patterns.some((re) => re.test(name))) continue;
  const src = path.join(targetDir, name);
  // Skip if it's a directory that happens to match.
  if (!fs.statSync(src).isFile()) continue;
  fs.copyFileSync(src, path.join(stageDir, name));
  staged += 1;
}

if (staged === 0) {
  console.error(
    `stage-bundle-resources: no matching shared libs found in ${targetDir}.\n` +
      `  Expected sherpa-onnx + onnxruntime ${
        process.platform === "win32" ? "DLLs" : ".so files"
      }.\n` +
      `  Confirm the sherpa-onnx \`shared\` feature is enabled.`
  );
  process.exit(1);
}
console.log(
  `stage-bundle-resources: staged ${staged} shared lib(s) from ${targetDir} → ${stageDir}`
);
