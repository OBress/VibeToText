#!/usr/bin/env node
// Stage runtime shared libraries for the Tauri bundler.
//
// On Windows we link sherpa-onnx in `shared` mode (see Cargo.toml
// for the MSVC ABI rationale), which means the final installer
// has to ship `sherpa-onnx-c-api.dll`, `onnxruntime.dll`, and a
// few siblings next to `vibe-to-text.exe`. macOS ships the matching
// `.dylib` files next to the app executable.
//
// `sherpa-onnx-sys`'s build script copies them into
// `src-tauri/target/<profile>/` so `cargo run` works in dev. For
// production bundling we need them at a STABLE path so
// `tauri.conf.json -> bundle.resources` can reference them; the
// bundler validates resource paths at build-script time and
// `target/release/` doesn't exist before the link step.
//
// This script runs as a Tauri `beforeBundleCommand` and copies the
// shared libraries from `target/<profile>/` into
// `src-tauri/bundle-resources/`, where the bundle config picks them up.
//
// Cross-platform: on macOS, Linux, and Windows we use sherpa-onnx
// in `shared` mode and stage the platform runtime libraries here.

const fs = require("fs");
const path = require("path");
const { spawnSync } = require("child_process");

const repoRoot = path.resolve(__dirname, "..");
const tauriDir = path.join(repoRoot, "src-tauri");
const stageDir = path.join(tauriDir, "bundle-resources");

// Ensure the dir exists so Tauri's resources glob doesn't trip on
// missing-dir validation.
fs.mkdirSync(stageDir, { recursive: true });

// sherpa-onnx-sys's build script copies the runtime libs into
// target/<profile>/; we re-stage them here so Tauri can pick them
// up alongside the executable in the installer.

// Tauri sets TAURI_ENV_DEBUG=true for `tauri dev` and false (or
// unset) for `tauri build`. Match that to the cargo profile.
const profile = process.env.TAURI_ENV_DEBUG === "true" ? "debug" : "release";
// When `tauri build --target <triple>` is invoked (which our CI
// matrix does), cargo writes outputs to `target/<triple>/<profile>/`
// rather than `target/<profile>/`. Tauri exposes the active triple
// as TAURI_ENV_TARGET_TRIPLE.
const triple = process.env.TAURI_ENV_TARGET_TRIPLE || "";
const preferredTargetDir = triple
  ? path.join(tauriDir, "target", triple, profile)
  : path.join(tauriDir, "target", profile);
const fallbackTargetDir = path.join(tauriDir, "target", profile);
const targetDir = fs.existsSync(preferredTargetDir)
  ? preferredTargetDir
  : fallbackTargetDir;

// The list sherpa-onnx-sys's build script emits when the `shared`
// feature is on. The actual file extension + naming differs per
// platform: `.dll` on Windows, `.dylib` on macOS, `.so` (with
// optional version suffixes like `.so.1.13.0`) on Linux. We glob for
// sensible name patterns rather than hard-code each file.
const PLATFORM_PATTERNS = {
  darwin: [
    /^libsherpa-onnx.*\.dylib$/i,
    /^libonnxruntime.*\.dylib$/i,
  ],
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
  const cargoBuildHint =
    profile === "release" ? "cargo build --release" : "cargo build";
  console.error(
    `stage-bundle-resources: targetDir ${targetDir} doesn't exist.\n` +
      `  Run \`${cargoBuildHint}\` first.`
  );
  process.exit(1);
}

const entries = fs.readdirSync(targetDir);
let staged = 0;
for (const name of fs.readdirSync(stageDir)) {
  if (!patterns.some((re) => re.test(name))) continue;
  fs.rmSync(path.join(stageDir, name), { force: true });
}
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
        process.platform === "win32"
          ? "DLLs"
          : process.platform === "darwin"
            ? ".dylib files"
            : ".so files"
      }.\n` +
      `  Confirm the sherpa-onnx \`shared\` feature is enabled.`
  );
  process.exit(1);
}
console.log(
  `stage-bundle-resources: staged ${staged} shared lib(s) from ${targetDir} → ${stageDir}`
);

if (process.platform === "darwin") {
  const executable = path.join(targetDir, "vibe-to-text");
  if (fs.existsSync(executable)) {
    const result = spawnSync(
      "install_name_tool",
      ["-add_rpath", "@executable_path", executable],
      { encoding: "utf8" }
    );
    if (result.status !== 0) {
      const stderr = result.stderr || "";
      if (!stderr.includes("would duplicate path")) {
        console.error(stderr.trim());
        process.exit(result.status || 1);
      }
    }
    console.log(
      `stage-bundle-resources: ensured @executable_path rpath on ${executable}`
    );
  }
}
