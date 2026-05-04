# VibeToText

A small Tauri 2 starter for **local streaming dictation**. Press a hotkey, speak, watch your words appear at the cursor — wherever you are (Slack, VS Code, browser, anywhere that accepts text).

## What you get

- Tauri 2 app with a hidden-by-default settings window
- **Two configurable global hotkeys**: one toggles settings, one toggles dictation
- **Cross-platform cursor injection** via `enigo` (streaming) or clipboard-paste (one-shot)
- **Two output modes** you can flip in settings:
  - `stream` — partial transcripts type live as you speak
  - `oneshot` — nothing types until you stop, then full transcript pastes at once
- **Two transcription backends with automatic fallback**:
  - **Voxtral** (GPU) via WebSocket to a local vLLM / voxmlx server — high quality, low latency, multilingual
  - **Moonshine Base** (CPU) embedded via ONNX Runtime — runs offline, English only, used as a battery-friendly fallback when no NVIDIA GPU is available
- Settings persisted to disk (JSON in the OS app config dir)
- System tray icon

## Layout

```
vibe-to-text/
├── package.json              # Tauri CLI + JS deps
├── src/                      # Frontend (vanilla HTML/JS — no framework)
│   ├── index.html
│   ├── main.js
│   └── styles.css
└── src-tauri/
    ├── Cargo.toml
    ├── tauri.conf.json
    ├── capabilities/default.json
    └── src/
        ├── main.rs           # entry
        ├── lib.rs            # Tauri builder, hotkey wiring, IPC commands
        ├── config.rs         # settings struct + JSON persistence
        ├── audio.rs          # cpal capture → 16 kHz mono f32 frames
        ├── inject.rs         # enigo type / clipboard-paste
        ├── models.rs         # lazy download + cache for Moonshine weights
        ├── stt.rs            # backend trait, DictationSession, select_backend()
        └── stt/
            ├── voxtral.rs    # WebSocket Realtime client (GPU path)
            ├── moonshine.rs  # ONNX Runtime greedy decode (CPU path)
            └── gpu.rs        # NVML + endpoint reachability checks
```

## Prereqs

- Rust (stable). Install via `rustup`.
- Node 20+. (Just used for the Tauri CLI.)
- Platform extras:
  - **Windows**: nothing special.
  - **macOS**: Xcode CLT (`xcode-select --install`).
  - **Linux**: `webkit2gtk-4.1-dev`, `libgtk-3-dev`, `libayatana-appindicator3-dev`, `librsvg2-dev`, plus ALSA dev headers (`libasound2-dev`).

## First run

```bash
cd vibe-to-text
npm install
npm run dev
```

Tauri will compile the Rust backend (slow first time, ~5 min), open the settings window, and register your hotkeys. Default hotkeys:

| Action | Default |
|---|---|
| Show/hide settings | `Ctrl+Alt+V` |
| Toggle dictation | `Ctrl+Alt+Space` |

## Backend selection

VibeToText picks a transcription backend automatically each time you press the dictate hotkey. You can override this in settings (Auto / Voxtral / Moonshine).

| Condition | Selected | Why |
|---|---|---|
| macOS (any) | **Voxtral** | Apple Silicon's unified memory makes voxmlx fast and power-efficient. Moonshine isn't bundled on macOS. |
| Windows + active NVIDIA GPU + reachable WS endpoint | **Voxtral** | Best quality and latency. |
| Windows on battery (dGPU parked in P8) | **Moonshine** | Embedded, CPU-only, no external server. |
| Windows + GPU present but vLLM not running | **Moonshine** | TCP probe fails → fallback. |
| No NVIDIA GPU at all | **Moonshine** | Only viable option. |

Selection is re-evaluated at the start of every dictation, so unplugging a laptop transparently switches the next press from Voxtral to Moonshine.

### Moonshine model files

The first time Moonshine is selected, the app downloads ~62 MB of ONNX weights (encoder + decoder + tokenizer) from `huggingface.co/onnx-community/moonshine-base-ONNX` into your OS app-data directory. You can prefetch them via the **Download Moonshine model now** button in settings.

### ONNX Runtime library

Moonshine uses the `ort` crate with `load-dynamic`, which means an `onnxruntime` shared library must be available at runtime. `lib.rs::init_ort_runtime` searches in this order:

1. `ORT_DYLIB_PATH` env var (explicit override)
2. Same directory as the binary (dev builds, Windows MSI, Linux deb)
3. `../Resources/` relative to the binary (macOS `.app` bundle layout)
4. `../lib/` relative to the binary (Linux split bin/lib layout)
5. `/opt/homebrew/lib`, `/usr/local/lib`, `/usr/lib` (macOS / Linux system paths)
6. ORT's default `dlopen` search path

For production bundles, **drop the right lib for each target into `src-tauri/resources/`** — it's already wired into `bundle.resources` in `tauri.conf.json` and Tauri ships it next to the binary on Windows/Linux and into `Contents/Resources/` on macOS:

| Platform | File to drop in `src-tauri/resources/` | Where to get it |
|---|---|---|
| Windows | `onnxruntime.dll` (~14 MB) | [microsoft/onnxruntime releases](https://github.com/microsoft/onnxruntime/releases) — `onnxruntime-win-x64-1.24.x.zip`. Already included for the current Windows build. |
| macOS | `libonnxruntime.dylib` | `brew install onnxruntime` then copy from `/opt/homebrew/lib/`, or grab `onnxruntime-osx-arm64-1.24.x.tgz` from the same releases page |
| Linux | `libonnxruntime.so` | `apt install libonnxruntime-dev` (Debian/Ubuntu) or `onnxruntime-linux-x64-1.24.x.tgz` |

The version must be in the **1.24.x** line to match the `ort = 2.0.0-rc.12` Rust crate's ABI.

## Wiring up the Voxtral backend

The Voxtral backend talks to a local WebSocket endpoint that speaks an OpenAI-Realtime-style protocol. Pick one:

### NVIDIA GPU on Linux

```bash
pip install vllm
VLLM_DISABLE_COMPILE_CACHE=1 vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 \
  --quantization fp8 \
  --port 8000
```

In settings, set:
- Endpoint: `ws://127.0.0.1:8000/v1/realtime`
- Model: `voxtral-mini-4b-realtime-2602`

### NVIDIA GPU on Windows (via WSL2)

vLLM doesn't run natively on Windows — only Linux. WSL2 with Ubuntu is the supported path, and Windows 11 + a recent NVIDIA driver give you native GPU passthrough into WSL with no extra setup.

1. Install WSL: from an **admin** PowerShell run `wsl --install`. Reboot.
2. Verify GPU passthrough works: `wsl -- nvidia-smi` should list your card.
3. Install Python tooling inside WSL: `wsl -- sudo apt update && sudo apt install -y python3-pip python3-venv`
4. Create a venv and install vLLM:
   ```bash
   wsl -- bash -lc "python3 -m venv ~/vllm_env && ~/vllm_env/bin/pip install vllm"
   ```
5. Start the server (downloads ~8 GB of weights first time, ~1 min for warm starts):
   ```bash
   wsl -- bash -lc "~/vllm_env/bin/vllm serve mistralai/Voxtral-Mini-4B-Realtime-2602 --quantization fp8 --port 8000"
   ```

WSL2 forwards `localhost:8000` to the Windows host automatically, so the VibeToText app on Windows connects to `ws://127.0.0.1:8000/v1/realtime` without any tunnel config. Voxtral picks up automatically when the app's auto-select sees both an active NVIDIA GPU (via NVML) and a reachable endpoint.

### Apple Silicon Mac

```bash
uvx --from "git+https://github.com/T0mSIlver/voxmlx.git[server]" \
  voxmlx-serve --model T0mSIlver/Voxtral-Mini-4B-Realtime-2602-MLX-4bit
```

(See the `localvoxtral` repo for a known-good config — VibeToText uses the same protocol.)

### Want Whisper instead?

Swap the WebSocket URL to a `faster-whisper` or `whisper.cpp` server that exposes the Realtime API. The protocol assumed by `stt.rs` only uses four message types:
`session.update`, `input_audio_buffer.append`, `input_audio_buffer.commit`, `response.create` — and listens for `response.audio_transcript.delta` and `response.audio_transcript.done`. If your backend speaks a different protocol, edit `stt.rs::run_session_inner`.

## Building production binaries

```bash
npm run build
```

Output:
- Windows → `src-tauri/target/release/bundle/msi/*.msi` and `nsis/*.exe`
- macOS → `src-tauri/target/release/bundle/dmg/*.dmg`
- Linux → `src-tauri/target/release/bundle/{appimage,deb}/`

For multi-platform CI, add a GitHub Actions matrix using `tauri-apps/tauri-action@v0` over `[macos-latest, ubuntu-22.04, windows-latest]`.

## Things to wire up next

- **Replace the linear resampler** in `audio.rs` with `rubato` for production-quality 16 kHz conversion.
- **VAD / silence detection** so dictation auto-stops when you pause for ~1.5 s instead of you having to release the hotkey.
- **ONNX Runtime auto-bundle for Mac/Linux** — Windows already ships `resources/onnxruntime.dll`. Mac/Linux production builds need the equivalent dropped in `src-tauri/resources/` (see ONNX Runtime library section above).
- **macOS + Linux smoke test** — the build is wired up cross-platform but only Windows has been smoke-tested end-to-end at the moment.

> **Why no streaming for Moonshine?** Moonshine is a whole-utterance encoder-decoder — streaming isn't part of the model architecture, only external scaffolding (sliding-window re-decoding with locked prefixes). Quality at chunk boundaries is consistently mediocre and it's not worth the complexity for a CPU fallback. Use Voxtral if you want true live partials.

### Already wired up

- **Push-to-talk hotkey** ([lib.rs](src-tauri/src/lib.rs)) — hold dictate hotkey to talk, release to send. Pressed/Released both handled, idempotent on auto-repeat.
- **Live waveform overlay** ([overlay.html](src/overlay.html), [overlay.js](src/overlay.js)) — frameless, transparent, click-through, always-on-top window with log-amplitude (dBFS) RMS waveform driven by `audio-level` events from a `~50Hz` event pump in [stt.rs](src-tauri/src/stt.rs). Auto-positions at the bottom-center of the primary monitor.
- **Per-utterance analytics dashboard** ([analytics.rs](src-tauri/src/analytics.rs), [main.js](src/main.js)) — every transcript is recorded with timestamp/duration/backend, persisted to `analytics.json`, and surfaced as KPIs, hour-of-day heatmap, 7-day bar chart, log-scaled word cloud, backend split, streaks, and generated insights.
- **KV-cached Moonshine decode** ([stt/moonshine.rs:run_cached_step](src-tauri/src/stt/moonshine.rs)) — uses `decoder_with_past_model_quantized.onnx` for ~50× speedup on long utterances. Falls back to the naive O(n²) prefill loop if the cached model is missing.
- **Background Moonshine warm-up** ([stt::warm_moonshine](src-tauri/src/stt.rs)) — preloads ONNX sessions at app start when weights are already on disk, eliminating the ~2–5 s cold-start latency.
- **Hash-verified manifest** ([models.rs](src-tauri/src/models.rs)) — every model file has a baked-in SHA256. A mismatch triggers automatic re-download.
- **Cross-platform ORT lookup** ([lib.rs::init_ort_runtime](src-tauri/src/lib.rs)) — searches `ORT_DYLIB_PATH`, then next-to-binary, `../Resources/` (macOS .app), `../lib/` (Linux), and the standard system lib paths.
- **Automatic backend selection** ([stt.rs::select_backend](src-tauri/src/stt.rs)) — re-evaluated on every dictation: NVIDIA GPU + reachable WS endpoint → Voxtral; otherwise Moonshine. macOS short-circuits to Voxtral.
- **Custom titlebar + window controls** — frameless dashboard with `data-tauri-drag-region`, custom min/close buttons, thin scrollbar.
- **Hotkey capture UI** — click a hotkey field, press the combo, it captures it. Esc cancels, Backspace reverts to default, blanks fall back to defaults at registration.

## License

MIT — do whatever.
