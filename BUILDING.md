# Building VibeToText from source

If you just want to **use** VibeToText, grab the latest installer from
the [releases page](https://github.com/OBress/VibeToText/releases) — see
the README for install instructions. You only need the steps below if
you want to modify the code or build a custom binary.

## Prerequisites

| Platform | What to install | Notes |
| --- | --- | --- |
| **Windows** | Visual Studio 2022 with the "Desktop development with C++" workload, [CMake ≥ 3.28](https://cmake.org/download/), [Rust stable](https://rustup.rs/), [Node.js 20+](https://nodejs.org) | The C++ workload installs MSVC + the Windows SDK that the CT2 + sherpa-onnx C++ code needs. |
| **macOS** | Xcode Command Line Tools (`xcode-select --install`), `brew install cmake`, [Rust stable](https://rustup.rs/), [Node.js 20+](https://nodejs.org) | Apple Silicon and Intel both supported. |
| **Linux (Ubuntu/Debian)** | `sudo apt install build-essential cmake libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev libasound2-dev`, [Rust stable](https://rustup.rs/), [Node.js 20+](https://nodejs.org) | Other distros: install the equivalents of webkit2gtk-4.1, gtk-3, libayatana-appindicator-3, and librsvg-2 dev packages. |

**Optional: NVIDIA CUDA Toolkit 12.x** — required only if you want to
test the GPU path during development. End-user binaries already work
on machines with or without CUDA via dynamic loading, so you don't
need it just to build a release.

## First clone

```bash
git clone https://github.com/OBress/VibeToText.git
cd VibeToText
npm install
```

## Development run (hot reload)

```bash
npm run tauri dev
```

The first run takes **20–35 minutes** because Cargo compiles
CTranslate2 (~10 min) and sherpa-onnx (~5–10 min downloads + links)
from C++ source. Both build outputs are cached after that — incremental
rebuilds are ~30 s.

You'll see the settings window appear once the build finishes. Press
the dictation hotkey (default `Ctrl+Alt+D`) to verify everything works.
Models are downloaded lazily on first dictation — the first press in
CPU mode pulls the ~250 MB Moonshine archive; the first press in GPU
mode pulls the ~770 MB Whisper medium.en model.

### Self-test the CPU path without a microphone

Set an environment variable before starting the dev server and the
app will run Moonshine against the bundled `test_wavs/` fixtures
once the model is downloaded, logging WER + RTFx for each:

```bash
# macOS / Linux
VIBE_TO_TEXT_SELF_TEST=1 npm run tauri dev

# Windows (PowerShell)
$env:VIBE_TO_TEXT_SELF_TEST = "1"; npm run tauri dev
```

Or click "Test Moonshine on bundled wavs" in the Dashboard footer
once the app is open.

## Production build

```bash
npm run tauri build
```

The bundler drops installers into
`src-tauri/target/release/bundle/<format>/`:

- **Windows**: `nsis/VibeToText_<version>_x64-setup.exe` and
  `msi/VibeToText_<version>_x64_en-US.msi`
- **macOS**: `dmg/VibeToText_<version>_<arch>.dmg`
- **Linux**: `deb/vibe-to-text_<version>_amd64.deb` and
  `appimage/vibe-to-text_<version>_amd64.AppImage`

The build profile is tuned for size + speed:

```toml
[profile.release]
opt-level = "s"      # size-optimized; ~5 % runtime cost vs -O3
lto = "fat"          # cross-crate inline + dead-code strip
strip = "symbols"    # drop debug info from final binary
codegen-units = 1    # better LTO at cost of build time
panic = "abort"      # smaller binary, no unwind tables
```

Final binary size: ~80 MB on Linux, ~100 MB on Windows (sherpa-onnx
ships its DLLs alongside, see the Cargo.toml comment for why), ~60 MB
on macOS. About half of that is the statically-linked CT2 runtime.

## Cross-compilation

We don't currently cross-compile in the repo workflow — each platform
runs its own GitHub Actions runner because the C++ build chain is
finicky enough across hosts. The release workflow
(`.github/workflows/release.yml`) is the source of truth for the
production build matrix.

## Where things live

- `src-tauri/src/lib.rs` — Tauri setup, IPC commands, hotkey wiring,
  tray + window plumbing.
- `src-tauri/src/audio.rs` — cpal audio capture, resampled to 16 kHz mono.
- `src-tauri/src/stt.rs` — backend dispatch, channel-drain helper, warm-up.
- `src-tauri/src/stt/whisper.rs` — CTranslate2 path. Owns the mel
  spectrogram + tokenizer, calls into ct2rs's `sys::Whisper`.
- `src-tauri/src/stt/moonshine.rs` — sherpa-onnx path. Wraps the
  `OfflineRecognizer` API.
- `src-tauri/src/vad.rs` — energy-based silence trimmer.
- `src-tauri/src/inject.rs` — clipboard-based paste with restore-after.
- `src-tauri/src/modifier_hook.rs` — Windows Raw Input listener for
  modifier-only push-to-talk combos like Ctrl+Shift.
- `src-tauri/src/analytics.rs` — per-utterance stats + the dashboard summary.
- `src-tauri/src/models.rs` — HF + sherpa-onnx model downloads.
- `src/index.html`, `src/main.js`, `src/styles.css` — settings + dashboard UI.
- `src/overlay.html`, `src/overlay.js` — listening waveform overlay.
- `scripts/stage-bundle-resources.cjs` — staging script the Tauri
  bundler invokes via `beforeBundleCommand` to copy Windows DLLs into
  the resources dir.

## Releasing

1. Bump the version in `src-tauri/Cargo.toml` and
   `src-tauri/tauri.conf.json` (must match).
2. Update `CHANGELOG.md` with what's new since the last tag.
3. Commit and push to main.
4. Tag the commit: `git tag v0.X.Y && git push --tags`.
5. The `release.yml` workflow runs against all four target platforms
   (~30–60 min total wall time) and creates a **draft** release with
   the installers attached.
6. Smoke-test the Windows + macOS installers on real hardware, then
   publish the draft from the GitHub UI.
