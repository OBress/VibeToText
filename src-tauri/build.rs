use std::{env, fs, path::PathBuf};

// Tauri validates every bundle.resources source path while compiling
// this build script, before `beforeBundleCommand` can stage the real
// shared libraries. Keep zero-byte placeholders in place so local
// macOS / Linux `cargo check`, `tauri dev`, and `tauri build` follow
// the same path as CI. scripts/stage-bundle-resources.cjs overwrites
// these files with the runtime libraries before the installer is
// created.
const BUNDLE_RESOURCE_STUBS: &[&str] = &[
    "sherpa-onnx-c-api.dll",
    "sherpa-onnx-cxx-api.dll",
    "onnxruntime.dll",
    "onnxruntime_providers_shared.dll",
    "libsherpa-onnx-c-api.dylib",
    "libsherpa-onnx-cxx-api.dylib",
    "libonnxruntime.dylib",
    "libonnxruntime.1.24.4.dylib",
];

fn main() {
    ensure_bundle_resource_stubs();
    tauri_build::build();
}

fn ensure_bundle_resource_stubs() {
    let manifest_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is set by cargo"));
    let resource_dir = manifest_dir.join("bundle-resources");

    fs::create_dir_all(&resource_dir).expect("create bundle resource staging dir");
    for name in BUNDLE_RESOURCE_STUBS {
        let path = resource_dir.join(name);
        if !path.exists() {
            fs::File::create(&path).expect("create bundle resource validation stub");
        }
    }
}
