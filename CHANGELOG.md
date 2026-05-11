# Changelog

All notable changes to VibeToText.

## [0.1.2] — 2026-05-10

macOS usability follow-up to v0.1.1: microphone selection,
permission strings, and paste injection fixes.

### Added

- **Microphone picker** in the settings UI. Replaces the
  free-text input field with a populated dropdown driven by a
  new `list_input_devices` Tauri command (CPAL enumeration).
  Default device is labeled and pinned to the top; previously-
  saved devices that are no longer plugged in show as "— not
  currently available" rather than silently failing.
- **macOS `NSMicrophoneUsageDescription`** via a bundled
  `src-tauri/Info.plist`. VibeToText now appears in
  System Settings → Privacy & Security → Microphone so users
  can grant + revoke the permission cleanly.

### Fixed

- **macOS paste crash in Cursor (and other Electron apps).**
  Enigo's `Key::Unicode('v')` path calls keyboard-layout APIs
  that must run on the main dispatch queue; invoking it from a
  Tokio worker crashed the process. macOS paste now uses raw
  HID keycodes (`55` = Cmd, `9` = V) which skip the layout
  lookup entirely.
- **Repeated macOS Accessibility prompts** every dictation.
  Enigo's default settings re-request the permission each time
  an injector is constructed. We now construct with
  `open_prompt_to_get_permissions = false` — the app already
  prompts for Accessibility separately on first use.

## [0.1.1] — 2026-05-10 [skipped]

Tagged but not released — the `macos-13` Intel build sat in
GitHub Actions' runner queue for 90+ min and never started, so
the draft release was abandoned in favor of v0.1.2 which rolls
up both releases' changes.

### Fixed (carried forward into v0.1.2)

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
