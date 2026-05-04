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
- **Speech-to-text via [CTranslate2](https://github.com/OpenNMT/CTranslate2) + [ct2rs](https://codeberg.org/tazz4843/ct2rs)** — the same engine that powers `faster-whisper`. We use the lower-level `sys::Whisper` API directly so we own the prompt construction (needed for the custom-vocabulary feature).
- **Per-device model + compute_type pairs**:
  - GPU (CUDA): default `Systran/faster-whisper-medium.en`, `INT8_FLOAT16`
  - CPU: default `Systran/faster-whisper-base.en`, `INT8`
  - User-overridable via dropdowns in settings.
- **Energy-based VAD** — trims leading + trailing silence before the mel spectrogram. Cuts encoder time on partial-30 s clips and prevents the silence-tail hallucination loop.
- **Custom dictionary** — a free-form initial prompt + a vocabulary list get tokenized and prepended to Whisper's decoder prompt with a `<|startofprev|>` marker, biasing the model toward project-specific jargon.
- **Hallucination filter** — drops known training-residue patterns (`[BLANK_AUDIO]`, `Thanks for watching`, etc.) so they never paste into your editor.
- **Cross-platform CUDA detection** via `cuda-dynamic-loading` — same binary works with or without an NVIDIA card; falls back to CPU silently.

## Distribution

End users get **one self-contained .exe** (or `.app` on macOS, `.AppImage` on Linux). CTranslate2's static library is linked into the binary at build time. Whisper model weights download from HuggingFace on first dictation (~150-770 MB depending on the model picked) and cache forever after under `~/.cache/huggingface/hub/`.

## Build prerequisites

Building from source requires:

- **Rust** (1.75+)
- **CMake** (≥3.28; ct2rs builds CTranslate2 from C++ source)
- **MSVC C++ toolchain** on Windows (Visual Studio 2022 with C++ workload)
- **Node.js** + npm (Tauri's frontend tooling)
- **CUDA Toolkit** (only required if you want GPU support; the binary works without it)

First clean build takes ~20-30 minutes (the C++ compile of CTranslate2 dominates). Subsequent incremental builds are ~30 s.

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
