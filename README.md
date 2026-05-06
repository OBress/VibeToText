# VibeToText

Local push-to-talk dictation for **Windows / macOS / Linux**. Hold a
global hotkey, speak, release — your transcript pastes at the cursor,
wherever you are (Slack, VS Code, browser, terminal, anywhere that
accepts text). Inference is 100 % on-device — no API key, no network
round-trip, no audio leaves your machine.

[**Download the latest release →**](https://github.com/OBress/VibeToText/releases/latest)

---

## Install

### Windows

1. Grab `VibeToText_<version>_x64-setup.exe` (NSIS installer, smaller)
   or `VibeToText_<version>_x64_en-US.msi` (MSI for managed installs)
   from the [latest release](https://github.com/OBress/VibeToText/releases/latest).
2. Run the installer. It installs to `%LOCALAPPDATA%\Programs\VibeToText\`
   (current-user, no admin prompt).
3. Launch **VibeToText** from the Start menu. The settings window
   opens; minimize it — the app lives in the system tray.
4. **Press your push-to-talk hotkey (default `Ctrl+Alt+D`), speak,
   release.** The transcript pastes at your cursor.

### macOS

1. Grab `VibeToText_<version>_aarch64.dmg` (Apple silicon) or
   `VibeToText_<version>_x64.dmg` (Intel) from the [latest release](https://github.com/OBress/VibeToText/releases/latest).
2. Mount the DMG and drag **VibeToText.app** to `/Applications`.
3. **First launch:** right-click the app → **Open** → confirm in the
   dialog. macOS Gatekeeper blocks unsigned apps on double-click; the
   right-click path adds it to your allowlist. (We don't yet sign +
   notarize — see [#issue](https://github.com/OBress/VibeToText/issues)
   if this is a blocker for you.)
4. Grant **Microphone** and **Accessibility** permissions when macOS
   prompts (System Settings → Privacy & Security). The Accessibility
   permission is what lets VibeToText paste at your cursor.
5. **Press the hotkey, speak, release.**

### Linux

```bash
# Debian / Ubuntu / Mint / Pop!_OS / etc.
sudo dpkg -i vibe-to-text_<version>_amd64.deb
# Resolves any missing system deps:
sudo apt -f install
```

Then launch **VibeToText** from your app launcher and press the hotkey.

The `.deb` declares `libwebkit2gtk-4.1-0` + `libgtk-3-0` as
dependencies; both are present on most desktops out of the box.

Other distros: extract the `.deb` with `ar x` (it's a tar archive
under the hood) or build from source per [BUILDING.md](BUILDING.md).

---

## First-run experience

| Step | What happens |
| --- | --- |
| First launch | Settings window opens. Configure your hotkeys, choose Auto / CPU-only mode. |
| First dictation press | If you're on CPU mode, the ~250 MB Moonshine model downloads (one-time, cached afterwards). On GPU mode, the ~770 MB Whisper medium.en model downloads from HuggingFace. |
| Subsequent presses | Hotkey-to-paste latency is **300–500 ms on GPU**, **0.7–1.5 s on CPU** (Moonshine). Both well under most users' thinking pauses. |

Models live under:

- **Whisper**: `~/.cache/huggingface/hub/`
- **Moonshine**: `<app_data_dir>/models/moonshine-base-en-int8/`
  - Windows: `%APPDATA%\dev.vibetotext.app\models\`
  - macOS: `~/Library/Application Support/dev.vibetotext.app/models/`
  - Linux: `~/.local/share/dev.vibetotext.app/models/`

You can pre-download from the **Settings → Model files → Download
model** button if you'd rather not wait on the first dictation.

---

## Hotkeys

| Action | Default |
| --- | --- |
| Show / hide settings | `Ctrl+Alt+V` |
| Push-to-talk dictation (hold) | `Ctrl+Alt+D` |
| Toggle STT on / off | (unset; configurable) |

All three are configurable in Settings. Modifier-only combos like
`Ctrl+Shift` are supported on Windows via a Raw Input hook (the
standard `RegisterHotKey` API can't reliably deliver modifier-only
events).

---

## What's under the hood

Two STT backends, picked at dictation time based on your "Compute
device" setting and runtime CUDA detection:

- **Whisper** via [CTranslate2](https://github.com/OpenNMT/CTranslate2)
  + [ct2rs](https://codeberg.org/tazz4843/ct2rs) — the same engine
  that powers `faster-whisper`. Default on GPU. Custom-vocabulary
  bias via Whisper's `initial_prompt`. Hallucination filter for
  training-set residue (`[BLANK_AUDIO]`, "Thanks for watching", etc.).
- **Moonshine base-en (INT8)** via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
  — RTFx 25–40× on AVX2 CPUs at ~6.65 % WER. Default on CPU. Faster
  than Whisper small.en on the same hardware at comparable accuracy.

Energy-based VAD trims leading + trailing silence before encoding.
Cross-platform CUDA detection via dynamic loading — same binary
works with or without an NVIDIA card.

Output goes through a **clipboard-restore paste**: stash existing
clipboard → write transcript → send Ctrl+V → restore previous
clipboard contents. Fast on long transcripts, doesn't lose your
copied text.

For the full architectural detail see [BUILDING.md](BUILDING.md).

---

## Privacy

Everything runs locally:

- Audio is captured by [cpal](https://github.com/RustAudio/cpal),
  resampled to 16 kHz mono, fed to the model in-process. It's never
  written to disk; never sent over the network.
- Models download once from HuggingFace (Whisper) or the sherpa-onnx
  GitHub releases page (Moonshine). After that, no further network
  traffic happens during dictation.
- The only data persisted is `analytics.json` (per-utterance counts +
  short transcript previews, used by the dashboard widgets). Reset
  any time from Dashboard → "Reset analytics".

---

## Building from source

If you want to modify VibeToText or build a custom binary, see
[BUILDING.md](BUILDING.md). End users don't need any of this — the
installer is self-contained.

---

## License

[MIT](LICENSE) — do whatever.
