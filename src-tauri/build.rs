// Tauri's standard build hook. Whisper-rs statically links whisper.cpp
// into the binary at build time (via CMake), so we no longer need to
// shuffle external DLLs (onnxruntime.dll, moonshine.dll) next to the
// exe — those were Moonshine-era artifacts and have been removed.
fn main() {
    tauri_build::build();
}
