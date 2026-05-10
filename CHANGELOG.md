# Changelog

All notable changes to VibeToText.

## [0.1.1] — 2026-05-10

macOS fixes — the app now actually runs on macOS, and Intel Macs
get their own native DMG.

### Fixed

- **macOS Moonshine crash on first dictation.** sherpa-onnx was
  statically linked on macOS and threw an uncaught `Ort::Exception`
  while creating the ONNX Runtime session. Switched macOS to the
  `shared` build (matching Windows + Linux); the `.app` now ships
  `lib{sherpa-onnx,onnxruntime}*.dylib` next to the executable in
  `Contents/MacOS/`, with an `@executable_path` rpath added at
  bundle time so dyld resolves them.
- **Missing Intel Mac DMG.** v0.1.0 only published the arm64 DMG,
  with a README claim that it would run on Intel under Rosetta 2 —
  but Rosetta translates Intel → Apple Silicon, not the other
  direction. The release matrix now includes `macos-13` building
  `x86_64-apple-darwin`, producing `VibeToText_<version>_x64.dmg`.
- **CI `cargo check` failing on every PR.** `tauri-build` validates
  every `bundle.resources` source path at compile time on every
  platform; build.rs now pre-creates zero-byte stubs for the
  Windows DLL + macOS dylib paths so local + CI builds don't trip
  on missing files.

## [0.1.0] — 2026-05-07

First public release. Local push-to-talk dictation with two STT
backends, cross-platform installers.

### Added

- **Whisper backend** via CTranslate2 + ct2rs. CUDA path with
  INT8_FLOAT16 compute type, CPU fallback path with INT8. Default
  GPU model: `Systran/faster-whisper-medium.en` (~770 MB,
  ~5.5 % WER). Beam search at width 5 + 1500 ms trailing-silence
  pad to prevent end-of-sentence truncation.
- **Moonshine backend** via sherpa-onnx 1.13. Default CPU choice;
  v1 base-en INT8 (~250 MB, ~6.65 % WER, RTFx 25–40× on AVX2).
- **Auto / CPU-only compute device picker** with runtime CUDA
  detection via `cuda-dynamic-loading`. Same binary runs with or
  without an NVIDIA card.
- **Custom vocabulary** for Whisper — initial-prompt + word list
  fed to the decoder via `<|startofprev|>` for jargon biasing.
- **Energy-based VAD** trims leading + trailing silence on both
  backends.
- **Clipboard-restore paste**: stash existing clipboard → write
  transcript → Ctrl+V → restore previous contents.
- **Modifier-only hotkeys** on Windows (Ctrl+Shift, etc.) via a
  Raw Input hook, since `RegisterHotKey` can't reliably deliver
  those.
- **System tray + always-on-top listening overlay** with live audio
  waveform.
- **Per-utterance analytics dashboard**: time talking, words
  spoken, sessions, vocabulary, hourly + weekly heatmaps, streaks,
  word cloud.
- **Self-test button** in the Dashboard footer that runs Moonshine
  against the bundled `test_wavs/` fixtures and reports per-file
  WER + RTFx.
- **Auto-start at login** via `tauri-plugin-autostart` (registry on
  Windows, LaunchAgent on macOS, .desktop on Linux).
- **Cross-platform installers** built by GitHub Actions on tag
  push: NSIS + MSI on Windows, DMG on macOS (arm64 + x86_64),
  .deb on Linux.

### Tech notes

- Final binary size: ~80 MB Linux, ~100 MB Windows and macOS
  (sherpa-onnx ships shared runtime libraries alongside due to
  platform-specific static-link issues — see Cargo.toml comments).
- First clean build: ~25–35 min (CT2 C++ compile dominates).
  Incremental: ~30 s. CI uses `Swatinem/rust-cache` to skip the
  cold path on subsequent runs.
- Release profile: `opt-level = "s"`, fat LTO, single codegen unit,
  panic = abort, symbols stripped. ~30 % size reduction vs default
  release at ~5 % runtime cost.
