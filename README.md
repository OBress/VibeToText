# VibeToText

Local push-to-talk dictation for Windows / macOS / Linux. Hold a global hotkey, speak, release — your transcript is pasted at the cursor, wherever you are (Slack, VS Code, browser, terminal, anywhere that accepts text). All inference is on-device — no API key, no network round trip.

## Speed (real measurements on a Lenovo Legion + RTX 30/40-class GPU)

| Mode | RTFx | Latency on 2.6 s clip |
|---|---|---|
| GPU (CUDA + INT8_FLOAT16) | 5-9× | 0.3-0.5 s |
| CPU (INT8 + 16 threads) | 1-2× | 1-3 s |

The app is designed around **push-to-talk** — you hold the hotkey while speaking and release to send.

## Architecture

- **Tauri 2** desktop shell — Rust backend, WebView2/WebKit frontend, persistent settings + analytics windows.
- **Hotkey capture** — `tauri-plugin-global-shortcut` for normal combos, plus a Windows Raw Input listener for modifier-only hotkeys like `Ctrl+Shift` (which `RegisterHotKey` can't reliably deliver).
- **Two STT backends, picked at dictation time**:
  - **Whisper** (GPU path, and an optional CPU path) via [CTranslate2](https://github.com/OpenNMT/CTranslate2) + [ct2rs](https://codeberg.org/tazz4843/ct2rs) — the same engine that powers `faster-whisper`. Used unconditionally on CUDA. We drive the lower-level `sys::Whisper` API so we own prompt construction (needed for the custom-vocabulary feature).
  - **Moonshine base-en** (default CPU path) via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — RTFx 25–40× on AVX2 CPUs at ~6.65 % WER, beating Whisper small.en's ~3–5× RTFx at the same accuracy tier. Loaded once and reused.
- **Per-device model + compute_type pairs**:
  - GPU (CUDA, Whisper): default `Systran/faster-whisper-medium.en`, `INT8_FLOAT16`
  - CPU (Moonshine, default): k2-fsa's prebuilt v1 base-en, INT8
  - CPU (Whisper, opt-in): default `Systran/faster-whisper-base.en`, `INT8`
  - All user-overridable in settings; the active backend is named in the tray + settings UI.
- **Energy-based VAD** — trims leading + trailing silence before encoding. Cuts encoder time on partial-30 s clips and prevents Whisper's silence-tail hallucination loop. Applied to both backends.
- **Custom dictionary (Whisper only)** — a free-form initial prompt + a vocabulary list get tokenized and prepended to Whisper's decoder prompt with a `<|startofprev|>` marker, biasing the model toward project-specific jargon.
- **Hallucination filter (Whisper only)** — drops known training-residue patterns (`[BLANK_AUDIO]`, `Thanks for watching`, etc.) so they never paste into your editor. Moonshine doesn't have these training-residue issues, so its output is forwarded as-is.
- **Cross-platform CUDA detection** via `cuda-dynamic-loading` — same binary works with or without an NVIDIA card; falls back to the CPU path (Moonshine by default) silently.

## Distribution

End users get **one self-contained .exe** (or `.app` on macOS, `.AppImage` on Linux). CTranslate2's static library AND sherpa-onnx's ONNX runtime statics are linked into the binary at build time — no separate DLLs to ship. Model weights download on first dictation:

- Whisper: ~150–770 MB from HuggingFace, cached under `~/.cache/huggingface/hub/`.
- Moonshine base-en (INT8): ~250 MB from k2-fsa's GitHub releases, extracted into `<app_data_dir>/models/moonshine-base-en-int8/`.

## Build prerequisites

Building from source requires:

- **Rust** (1.75+)
- **CMake** (≥3.28; ct2rs builds CTranslate2 from C++ source)
- **MSVC C++ toolchain** on Windows (Visual Studio 2022 with C++ workload)
- **Node.js** + npm (Tauri's frontend tooling)
- **CUDA Toolkit** (only required if you want GPU support; the binary works without it)

First clean build takes ~20–30 minutes — the C++ compile of CTranslate2 dominates; sherpa-onnx adds ~30 s by downloading its prebuilt static libs from k2-fsa's release artifacts. Subsequent incremental builds are ~30 s.

```bash
npm install
npm run dev      # debug build with hot reload
npm run build    # release build with installer
```

End users do NOT need any of these — they just run the installed `.exe`.

## Default hotkeys

| Action | Default |
|---|---|
| Show/hide settings | `Ctrl+Alt+V` |
| Push-to-talk dictation (hold) | `Ctrl+Alt+D` |

Configure both in settings. Modifier-only combos like `Ctrl+Shift` work on Windows via a separate Raw Input hook.

## Layout

- `src-tauri/` — Rust backend
  - `src/lib.rs` — Tauri app entry, hotkey routing, IPC commands
  - `src/stt/whisper.rs` — CT2 inference path with VAD + initial_prompt
  - `src/models.rs` — HuggingFace model download + cache management
  - `src/vad.rs` — energy-based silence trimmer
  - `src/audio.rs` — `cpal` capture, resampling to 16 kHz mono f32
  - `src/inject.rs` — clipboard-paste injection at cursor
  - `src/modifier_hook.rs` — Windows Raw Input for modifier-only hotkeys
  - `src/analytics.rs` — per-utterance stats persisted to `analytics.json`
  - `.cargo/config.toml` — Windows build workarounds (`+crt-static`, CUDA path forward-slash fix)
- `src/` — frontend (vanilla HTML/JS/CSS, no framework)
  - `index.html` — settings + analytics dashboard
  - `overlay.html` — listening overlay (transparent always-on-top window with live waveform)
  - `main.js`, `overlay.js`, `styles.css`, `overlay.css`
- `package.json`, `Cargo.toml` — dependency manifests

## License

MIT — do whatever.
