# Changelog

All notable changes to VibeToText.

## [0.1.0] — Unreleased

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
  .deb + .AppImage on Linux.

### Tech notes

- Final binary size: ~80 MB Linux, ~100 MB Windows (sherpa-onnx
  ships its DLLs alongside on Windows due to a prebuilt-vs-local
  MSVC ABI mismatch — see Cargo.toml comment), ~60 MB macOS.
- First clean build: ~25–35 min (CT2 C++ compile dominates).
  Incremental: ~30 s. CI uses `Swatinem/rust-cache` to skip the
  cold path on subsequent runs.
- Release profile: `opt-level = "s"`, fat LTO, single codegen unit,
  panic = abort, symbols stripped. ~30 % size reduction vs default
  release at ~5 % runtime cost.
