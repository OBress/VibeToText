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

// On non-Windows platforms sherpa-onnx links statically — nothing
// to do here. Still ensure the dir exists so Tauri's resources
// glob doesn't complain.
fs.mkdirSync(stageDir, { recursive: true });
if (process.platform !== "win32") {
  console.log(
    `stage-bundle-resources: skipping on ${process.platform} (sherpa-onnx is static here).`
  );
  process.exit(0);
}

// Pick the profile based on the BUILD_PROFILE env var that Tauri
// passes to beforeBundleCommand, falling back to "release" since
// that's what `tauri build` uses by default.
const profile = process.env.TAURI_ENV_DEBUG === "true" ? "debug" : "release";
const targetDir = path.join(tauriDir, "target", profile);

// The full list sherpa-onnx-sys's build script emits when the
// `shared` feature is on. Some files are optional depending on the
// release archive contents (`onnxruntime_providers_shared.dll` is
// only present when ONNX Runtime was built with provider plugins),
// so we skip-with-a-warning for missing-but-optional files.
const REQUIRED_DLLS = [
  "sherpa-onnx-c-api.dll",
  "sherpa-onnx-cxx-api.dll",
  "onnxruntime.dll",
];
const OPTIONAL_DLLS = ["onnxruntime_providers_shared.dll"];

let staged = 0;
for (const name of REQUIRED_DLLS) {
  const src = path.join(targetDir, name);
  if (!fs.existsSync(src)) {
    console.error(
      `stage-bundle-resources: required DLL missing at ${src}.\n` +
        `  Run \`cargo build --release\` first, or check that the sherpa-onnx \`shared\` feature is enabled.`
    );
    process.exit(1);
  }
  fs.copyFileSync(src, path.join(stageDir, name));
  staged += 1;
}
for (const name of OPTIONAL_DLLS) {
  const src = path.join(targetDir, name);
  if (fs.existsSync(src)) {
    fs.copyFileSync(src, path.join(stageDir, name));
    staged += 1;
  }
}
console.log(
  `stage-bundle-resources: staged ${staged} DLL(s) from ${targetDir} → ${stageDir}`
);
