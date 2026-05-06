// VibeToText — local push-to-talk dictation app.
//
// Architecture:
//   - User presses TOGGLE_SETTINGS hotkey → show/hide settings window.
//   - User presses DICTATE hotkey → start capturing mic audio. On
//     release, the buffered audio is fed to whisper.cpp (via
//     whisper-rs, statically linked) running locally on CPU
//     (Windows/Linux) or Metal (macOS). The transcript is pasted at
//     the user's cursor.
//
// All inference is on-device. No network round-trip, no external
// model server, no cloud key.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod analytics;
mod audio;
mod config;
mod inject;
mod models;
#[cfg(target_os = "windows")]
mod modifier_hook;
mod stt;
mod vad;

use std::sync::Arc;
use tauri::{
    menu::{MenuBuilder, MenuItemBuilder, PredefinedMenuItem},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    Emitter, Manager,
};
use tauri_plugin_global_shortcut::{
    Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState,
};
use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
use tokio::sync::Mutex;

use config::AppConfig;

/// Shared state held by Tauri.
pub struct AppState {
    pub config: Mutex<AppConfig>,
    pub session: Mutex<Option<stt::DictationSession>>,
    /// Lazy-loaded transcription backend (Whisper for GPU/Whisper-CPU
    /// path, Moonshine for default-CPU path). Stored as a trait
    /// object so the cache can hold either. Cached across dictations
    /// so we only pay the model-load cost once. Becomes `None` again
    /// when the user changes anything that would resolve to a
    /// different backend (mode, model id, cpu_engine) — the next
    /// dictation will rebuild against the new choice.
    pub backend: Mutex<Option<Arc<dyn stt::SttBackend>>>,
    /// Name of the active backend ("whisper"), surfaced to the UI for
    /// status display. Kept as `Option` because nothing fills it in
    /// until the first dictation completes the lazy init.
    pub current_backend: Mutex<Option<&'static str>>,
    /// Per-utterance analytics, persisted to disk on each update.
    pub analytics: Mutex<analytics::Analytics>,
    /// Flips to true when the user releases the dictate hotkey while
    /// `start_dictation` is still mid-await (e.g. waiting on Whisper
    /// model load). Lets a slow start abort cleanly instead of leaving
    /// the overlay stuck on "Preparing model".
    pub start_cancel: AtomicBool,
    /// True while a `start_dictation` call is in progress. Lets us
    /// avoid taking the session lock during the long model load while
    /// still suppressing duplicate starts from auto-repeat.
    pub starting: AtomicBool,
}

#[tauri::command]
async fn get_config(state: tauri::State<'_, Arc<AppState>>) -> Result<AppConfig, String> {
    Ok(state.config.lock().await.clone())
}

#[tauri::command]
async fn save_config(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
    new_config: AppConfig,
) -> Result<(), String> {
    // Reject same-combo collisions early. Both hotkeys go to the same
    // global-shortcut handler; if they're equal, the settings-toggle
    // branch matches first and dictation can never fire. Canonical
    // comparison so e.g. "Ctrl+Shift" and "Shift+Ctrl" are caught.
    let toggle_keys = canonical_combo_str(new_config.toggle_settings_hotkey.trim());
    let dictate_keys = canonical_combo_str(new_config.dictate_hotkey.trim());
    if !toggle_keys.is_empty() && toggle_keys == dictate_keys {
        return Err(format!(
            "Settings hotkey and dictation hotkey can't be the same ({}). Pick a different combo for one of them.",
            new_config.dictate_hotkey
        ));
    }

    let mut cfg = state.config.lock().await;
    let old = cfg.clone();
    *cfg = new_config.clone();
    drop(cfg);

    config::save(&app, &new_config).map_err(|e| e.to_string())?;

    // Drop the cached backend if any of the runtime-relevant
    // settings changed: the device mode, EITHER per-device model
    // pick, or the CPU engine choice. The next dictation rebuilds
    // with the new choice. (If the field that changed isn't currently
    // active — e.g. user edited the GPU model while running in CPU
    // mode — we still drop it; cheap to rebuild and avoids stale
    // mode-mismatch surprises.)
    let mode_changed = old.backend_mode != new_config.backend_mode;
    let cpu_model_changed = old.whisper_model_cpu != new_config.whisper_model_cpu;
    let gpu_model_changed = old.whisper_model_gpu != new_config.whisper_model_gpu;
    let cpu_engine_changed = old.cpu_engine != new_config.cpu_engine;
    if mode_changed || cpu_model_changed || gpu_model_changed || cpu_engine_changed {
        let mut w = state.backend.lock().await;
        if w.is_some() {
            log::info!(
                "backend config changed (mode {} → {}, cpu_engine {} → {}, cpu {} → {}, gpu {} → {}); dropping cached backend",
                old.backend_mode,
                new_config.backend_mode,
                old.cpu_engine,
                new_config.cpu_engine,
                old.whisper_model_cpu,
                new_config.whisper_model_cpu,
                old.whisper_model_gpu,
                new_config.whisper_model_gpu,
            );
            *w = None;
        }
        drop(w);
        // Also clear the surfaced backend name so the UI doesn't
        // keep showing a stale "current backend" until next dictation.
        *state.current_backend.lock().await = None;

        // Kick a background warm-up of the NEW backend so the next
        // dictation press doesn't pay the 1-3 s model-load cost.
        // Without this, switching from "GPU" to "CPU only" silently
        // leaves the cache empty until the user actually triggers
        // dictation, at which point they wait through a model load
        // they didn't expect.
        let app_for_warm = app.clone();
        let cfg_for_warm = new_config.clone();
        tauri::async_runtime::spawn(async move {
            stt::warm_whisper(app_for_warm, cfg_for_warm).await;
        });
    }

    // Re-register hotkeys if any changed.
    if old.toggle_settings_hotkey != new_config.toggle_settings_hotkey
        || old.dictate_hotkey != new_config.dictate_hotkey
    {
        register_hotkeys(&app, &new_config).map_err(|e| e.to_string())?;
    }

    // Reflect autostart toggle at the OS level when it flips. Logging-only
    // failure: we don't want a registry hiccup to abort the whole save,
    // and the user can flip it again next time.
    if old.auto_start != new_config.auto_start {
        use tauri_plugin_autostart::ManagerExt;
        let mgr = app.autolaunch();
        let r = if new_config.auto_start {
            mgr.enable()
        } else {
            mgr.disable()
        };
        if let Err(e) = r {
            log::warn!("autostart toggle failed: {e}");
        }
    }
    Ok(())
}

/// Click-driven toggle from the "Test dictation" button. Goes through
/// the same start/stop helpers as the push-to-talk hotkey so the
/// overlay shows/hides consistently. Returns true if now listening.
#[tauri::command]
async fn toggle_dictation(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<bool, String> {
    let listening = state.session.lock().await.is_some();
    if listening {
        stop_dictation(&app).await.map_err(|e| e.to_string())?;
        Ok(false)
    } else {
        start_dictation(&app).await.map_err(|e| e.to_string())?;
        Ok(true)
    }
}

#[tauri::command]
async fn current_backend(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<Option<&'static str>, String> {
    Ok(*state.current_backend.lock().await)
}

#[tauri::command]
async fn whisper_model_present(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<bool, String> {
    let cfg = state.config.lock().await.clone();
    // "Model present" now means: every asset the current backend_mode
    // + cpu_engine combination needs is on disk. For `auto` we require
    // both GPU and CPU paths to be ready (otherwise unplugging the
    // laptop would silently force a download mid-press).
    let mode = cfg.backend_mode.to_ascii_lowercase();
    let cpu_uses_moonshine = cfg.cpu_engine.eq_ignore_ascii_case("moonshine");
    let cpu_ready = if cpu_uses_moonshine {
        models::moonshine_already_downloaded(&app)
    } else {
        // Reuse the helper, which already handles the "any candidate
        // present" semantics — close enough for the CPU-Whisper case.
        models::whisper_already_downloaded(&app, &cfg)
    };
    let gpu_ready = models::whisper_already_downloaded(&app, &cfg);
    let ready = match mode.as_str() {
        "gpu" | "cuda" => gpu_ready,
        "cpu" => cpu_ready,
        _ => cpu_ready && gpu_ready,
    };
    Ok(ready)
}

/// Force-download (or re-download) the model assets the current
/// `backend_mode` + `cpu_engine` could pick at runtime. Backs the
/// "Download model" button in settings.
///
///   - gpu  → Whisper GPU model only.
///   - cpu  → Whisper CPU model OR Moonshine archive, depending on
///            `cpu_engine`.
///   - auto → BOTH the GPU model AND the CPU asset (Moonshine or
///            Whisper-CPU), so the user is prepared for either
///            device after `auto` re-resolves at next dictation.
#[tauri::command]
async fn download_whisper_model(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let cfg = state.config.lock().await.clone();
    let mode = cfg.backend_mode.to_ascii_lowercase();
    let cpu_uses_moonshine = cfg.cpu_engine.eq_ignore_ascii_case("moonshine");

    let mut whisper_ids: Vec<String> = Vec::new();
    let mut need_moonshine = false;

    match mode.as_str() {
        "gpu" | "cuda" => {
            whisper_ids.push(cfg.whisper_model_gpu.clone());
        }
        "cpu" => {
            if cpu_uses_moonshine {
                need_moonshine = true;
            } else {
                whisper_ids.push(cfg.whisper_model_cpu.clone());
            }
        }
        _ => {
            // auto: prepare both the GPU asset and whichever CPU asset
            // would win on this machine.
            whisper_ids.push(cfg.whisper_model_gpu.clone());
            if cpu_uses_moonshine {
                need_moonshine = true;
            } else if cfg.whisper_model_cpu != cfg.whisper_model_gpu {
                whisper_ids.push(cfg.whisper_model_cpu.clone());
            }
        }
    }

    for id in whisper_ids {
        models::ensure_whisper_ready(&app, &cfg, &id)
            .await
            .map_err(|e| e.to_string())?;
    }
    if need_moonshine {
        models::ensure_moonshine_ready(&app)
            .await
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

// (`download_whisper_cli` removed — the CT2-based backend has no
// separate CLI/runtime archive. The model itself is the only
// downloaded artifact, handled by `download_whisper_model`.)

/// Self-test the Moonshine backend by running it against the .wav
/// fixtures shipped inside k2-fsa's model archive (`test_wavs/0.wav`,
/// `1.wav`, `8k.wav`) and comparing each transcript against the
/// ground-truth strings in `trans.txt`. Returns one entry per fixture:
/// `{ file, expected, actual, ok }`.
///
/// Lives behind an IPC so we can drive it from the Dashboard without
/// needing real microphone input. Match comparison is case- and
/// punctuation-insensitive (Moonshine emits lower-case un-punctuated
/// transcripts; the ground truth is upper-case).
#[tauri::command]
async fn run_moonshine_self_test(
    app: tauri::AppHandle,
) -> Result<Vec<serde_json::Value>, String> {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    let dir = models::moonshine_dir(&app).map_err(|e| e.to_string())?;
    let test_dir = dir.join("test_wavs");
    if !test_dir.is_dir() {
        return Err(format!(
            "Moonshine test_wavs not found at {} — download the model first",
            test_dir.display()
        ));
    }

    // Force a Moonshine load even if config currently picks Whisper —
    // we don't want the test to silently no-op when the user is in
    // GPU mode. We bypass select_backend's caching here on purpose.
    let backend = stt::MoonshineBackend::new(dir.clone())
        .await
        .map_err(|e| e.to_string())?;

    // Parse trans.txt. Lines look like: `0.wav SOME UPPER CASE TEXT`.
    let trans_path = test_dir.join("trans.txt");
    let f = File::open(&trans_path)
        .map_err(|e| format!("open {}: {e}", trans_path.display()))?;
    let mut expected: HashMap<String, String> = HashMap::new();
    for line in BufReader::new(f).lines().map_while(|l| l.ok()) {
        if let Some((file, text)) = line.split_once(' ') {
            expected.insert(file.to_string(), text.trim().to_string());
        }
    }

    // Decode each .wav as f32 16 kHz mono and run the recognizer.
    let mut results: Vec<serde_json::Value> = Vec::new();
    for fname in ["0.wav", "1.wav", "8k.wav"] {
        let wav_path = test_dir.join(fname);
        if !wav_path.is_file() {
            continue;
        }
        let samples = decode_wav_to_f32(&wav_path).map_err(|e| e.to_string())?;
        let n_audio_samples = samples.len();
        let audio_seconds = n_audio_samples as f32 / 16_000.0;
        let started = std::time::Instant::now();
        let actual = backend
            .transcribe_samples(samples)
            .await
            .map_err(|e| e.to_string())?;
        let elapsed = started.elapsed();

        let exp = expected.get(fname).cloned().unwrap_or_default();
        let wer = word_error_rate(&exp, &actual);
        // Pass at WER ≤ 10 % — the bundled fixtures contain proper
        // nouns ("Prynne") and archaic spellings ("for ever") where
        // a small mismatch is expected even from a perfect model.
        // 10 % is the "this clearly transcribed real English"
        // threshold, not a strict accuracy claim.
        let ok = wer <= 0.10;
        results.push(serde_json::json!({
            "file": fname,
            "expected": exp,
            "actual": actual.trim(),
            "ok": ok,
            "wer": wer,
            "elapsed_ms": elapsed.as_millis() as u64,
            "rtfx": (audio_seconds / elapsed.as_secs_f32().max(0.001)),
        }));
        log::info!(
            "Moonshine self-test {fname}: ok={ok} wer={:.3} elapsed={:?} rtfx={:.1}× actual={:?}",
            wer,
            elapsed,
            audio_seconds / elapsed.as_secs_f32().max(0.001),
            actual.trim()
        );
    }
    Ok(results)
}

/// Standard ASR word error rate over a Levenshtein-on-words distance.
/// Returns a fraction (0.0 = perfect, 1.0 = every word wrong).
/// Comparison is case-insensitive and punctuation-insensitive.
fn word_error_rate(reference: &str, hypothesis: &str) -> f32 {
    let r = canonicalize(reference);
    let h = canonicalize(hypothesis);
    let r_words: Vec<&str> = r.split_whitespace().collect();
    let h_words: Vec<&str> = h.split_whitespace().collect();
    if r_words.is_empty() {
        return if h_words.is_empty() { 0.0 } else { 1.0 };
    }
    let n = r_words.len();
    let m = h_words.len();
    // Classic DP edit distance over word tokens. O(n*m) memory, fine
    // for our short fixtures.
    let mut prev: Vec<usize> = (0..=m).collect();
    let mut cur: Vec<usize> = vec![0; m + 1];
    for i in 1..=n {
        cur[0] = i;
        for j in 1..=m {
            let cost = if r_words[i - 1] == h_words[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1)
                .min(cur[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[m] as f32 / n as f32
}

/// Decode a 16-bit (or 32-bit float) PCM WAV to mono 16 kHz f32 in
/// [-1, 1]. We trust Moonshine's bundled fixtures to be 16 kHz mono;
/// the 8k.wav is 8 kHz mono and we naive-upsample by 2× linear
/// interpolation so the recognizer's 16 kHz expectation is met.
fn decode_wav_to_f32(path: &std::path::Path) -> anyhow::Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| anyhow::anyhow!("open wav {}: {e}", path.display()))?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .filter_map(Result::ok)
                .map(|s| s as f32 / max)
                .collect()
        }
        hound::SampleFormat::Float => reader.samples::<f32>().filter_map(Result::ok).collect(),
    };
    let mono: Vec<f32> = if channels > 1 {
        samples
            .chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };
    let resampled = if spec.sample_rate == 16_000 {
        mono
    } else if spec.sample_rate == 8_000 {
        // Linear 2× up-sample: between every pair of samples insert
        // their midpoint. Good enough for the self-test fixture.
        let mut out = Vec::with_capacity(mono.len() * 2);
        for w in mono.windows(2) {
            out.push(w[0]);
            out.push((w[0] + w[1]) * 0.5);
        }
        if let Some(last) = mono.last() {
            out.push(*last);
        }
        out
    } else {
        return Err(anyhow::anyhow!(
            "self-test wav has unsupported sample rate {}",
            spec.sample_rate
        ));
    };
    Ok(resampled)
}

/// Lower-case + strip non-alphanumerics for transcript comparison.
fn canonicalize(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .flat_map(|c| c.to_lowercase())
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

#[tauri::command]
async fn get_analytics(
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<analytics::AnalyticsSummary, String> {
    let a = state.analytics.lock().await;
    Ok(a.summary())
}

#[tauri::command]
async fn reset_analytics(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let mut a = state.analytics.lock().await;
    a.reset();
    analytics::save(&app, &a).map_err(|e| e.to_string())?;
    let _ = app.emit("analytics-updated", ());
    Ok(())
}

/// Temporarily unregister every global shortcut. The settings UI calls
/// this when a hotkey input is focused so the user can pick a new combo
/// without the OS-level RegisterHotKey eating the keys before the
/// focused webview sees them. Pair with `resume_hotkeys` on blur.
#[tauri::command]
fn pause_hotkeys(app: tauri::AppHandle) -> Result<(), String> {
    app.global_shortcut()
        .unregister_all()
        .map_err(|e| e.to_string())?;
    log::info!("hotkeys paused (capture mode)");
    Ok(())
}

/// Re-register hotkeys from the current in-memory config. Called when a
/// hotkey input loses focus, undoing whatever `pause_hotkeys` did.
#[tauri::command]
async fn resume_hotkeys(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
) -> Result<(), String> {
    let cfg = state.config.lock().await.clone();
    register_hotkeys(&app, &cfg).map_err(|e| e.to_string())
}

/// Register / unregister the app from launching at user login. Backed by
/// `tauri-plugin-autostart` (registry on Windows, LaunchAgent on macOS,
/// .desktop file on Linux).
#[tauri::command]
async fn set_auto_start(app: tauri::AppHandle, enabled: bool) -> Result<(), String> {
    use tauri_plugin_autostart::ManagerExt;
    let mgr = app.autolaunch();
    if enabled {
        mgr.enable().map_err(|e| e.to_string())?;
    } else {
        mgr.disable().map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Returns whether autostart is currently enabled at the OS level. The
/// in-memory config flag may drift if the user disables autostart from
/// the OS (Task Manager → Startup), so this is the source of truth.
#[tauri::command]
async fn is_auto_start_enabled(app: tauri::AppHandle) -> Result<bool, String> {
    use tauri_plugin_autostart::ManagerExt;
    app.autolaunch().is_enabled().map_err(|e| e.to_string())
}

fn register_hotkeys(app: &tauri::AppHandle, cfg: &AppConfig) -> anyhow::Result<()> {
    let gs = app.global_shortcut();
    gs.unregister_all()?;

    // Defensive: an empty-string hotkey (e.g. from a UI bug that wiped
    // the saved config) silently disables the hotkey entirely. Fall
    // back to defaults in that case so the app is never deaf.
    let toggle_str = if cfg.toggle_settings_hotkey.trim().is_empty() {
        log::warn!("toggle_settings_hotkey is empty; falling back to Ctrl+Alt+V");
        "Ctrl+Alt+V"
    } else {
        cfg.toggle_settings_hotkey.as_str()
    };
    let dictate_str = if cfg.dictate_hotkey.trim().is_empty() {
        log::warn!("dictate_hotkey is empty; falling back to Ctrl+Alt+D");
        "Ctrl+Alt+D"
    } else {
        cfg.dictate_hotkey.as_str()
    };

    // The "stt_toggle" entry is optional — empty string disables it
    // entirely (no hotkey for the master switch).
    let stt_toggle_str = cfg.stt_toggle_hotkey.trim();
    let mut entries: Vec<(&str, &str)> = vec![
        ("settings", toggle_str),
        ("dictate", dictate_str),
    ];
    if !stt_toggle_str.is_empty() {
        entries.push(("stt_toggle", stt_toggle_str));
    }

    for (label, combo) in entries {
        let canonical = canonical_combo_str(combo);

        // Modifier-only combos (e.g. "Ctrl+Shift") can't be reliably
        // delivered by RegisterHotKey on Windows, so route them through
        // the WH_KEYBOARD_LL hook instead. Only the dictate hotkey uses
        // the hook (push-to-talk semantics); other actions need a
        // non-modifier key.
        #[cfg(target_os = "windows")]
        {
            if let Some(mask) = modifier_hook::mask_for_modifier_only(&canonical) {
                if label == "dictate" {
                    modifier_hook::set_watch_mask(mask);
                    log::info!(
                        "{label} hotkey '{combo}' is modifier-only — using keyboard hook"
                    );
                    continue;
                } else {
                    log::warn!(
                        "{label} hotkey '{combo}' is modifier-only; that hotkey requires a non-modifier key, ignoring"
                    );
                    continue;
                }
            }
        }
        // Non-Windows or combo isn't modifier-only → fall through to
        // RegisterHotKey path. Make sure the hook is disarmed for dictate.
        #[cfg(target_os = "windows")]
        if label == "dictate" {
            modifier_hook::set_watch_mask(0);
        }

        match shortcut_variants(combo) {
            Ok(variants) if variants.is_empty() => {
                log::warn!("{label} hotkey '{combo}' produced no variants");
            }
            Ok(variants) => {
                for v in &variants {
                    if let Err(e) = gs.register(v.clone()) {
                        // Don't bail — duplicate or OS-rejected variants
                        // shouldn't prevent the rest from registering.
                        log::warn!("{label} variant register failed: {e}");
                    }
                }
                log::info!("{label} hotkey registered: {combo} ({} variants)", variants.len());
            }
            Err(e) => {
                log::warn!("{label} hotkey '{combo}' parse failed: {e}");
            }
        }
    }
    Ok(())
}

/// Parse "Ctrl+Alt+Space" or "Ctrl+Shift" style strings into a Shortcut.
/// Modifier-only combos use the LAST modifier as the trigger key (the
/// caller may then expand variants for press-order independence).
fn parse_shortcut(s: &str) -> anyhow::Result<Shortcut> {
    let parts: Vec<&str> = s
        .split('+')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    if parts.is_empty() {
        return Err(anyhow::anyhow!("empty shortcut"));
    }
    let last = parts.last().unwrap();
    let head = &parts[..parts.len() - 1];

    let mut mods = Modifiers::empty();
    for part in head {
        match part.to_ascii_lowercase().as_str() {
            "ctrl" | "control" => mods |= Modifiers::CONTROL,
            "alt" | "option" => mods |= Modifiers::ALT,
            "shift" => mods |= Modifiers::SHIFT,
            "cmd" | "super" | "meta" | "win" => mods |= Modifiers::SUPER,
            other => {
                return Err(anyhow::anyhow!("'{}' is not a modifier in '{}'", other, s));
            }
        }
    }

    // Trigger: real key first, then fall back to "modifier-as-key".
    let key = if let Some(code) = code_from_str(last) {
        code
    } else {
        match last.to_ascii_lowercase().as_str() {
            "ctrl" | "control" => Code::ControlLeft,
            "shift" => Code::ShiftLeft,
            "alt" | "option" => Code::AltLeft,
            "cmd" | "super" | "meta" | "win" => Code::MetaLeft,
            _ => {
                return Err(anyhow::anyhow!(
                    "unknown key '{}' in shortcut '{}'",
                    last,
                    s
                ))
            }
        }
    };
    Ok(Shortcut::new(Some(mods), key))
}

/// Expand a combo string into every Shortcut variant we should register
/// so it fires regardless of which modifier is the "trigger" or whether
/// the left or right side of a modifier was pressed.
///
/// Examples:
///   "Ctrl+Alt+D"  → 1 variant: (Ctrl|Alt, KeyD)
///   "Ctrl+Shift"  → 4 variants:
///       (Ctrl,  ShiftLeft)   (Ctrl,  ShiftRight)
///       (Shift, ControlLeft) (Shift, ControlRight)
///   "Ctrl+Shift+Alt" → 6 variants (each modifier gets the trigger role,
///                                  both L/R per modifier).
fn shortcut_variants(s: &str) -> anyhow::Result<Vec<Shortcut>> {
    let primary = parse_shortcut(s)?;

    // Non-modifier trigger: nothing more to expand.
    let trigger_as_mod = match primary.key {
        Code::ControlLeft | Code::ControlRight => Some(Modifiers::CONTROL),
        Code::ShiftLeft | Code::ShiftRight => Some(Modifiers::SHIFT),
        Code::AltLeft | Code::AltRight => Some(Modifiers::ALT),
        Code::MetaLeft | Code::MetaRight => Some(Modifiers::SUPER),
        _ => None,
    };
    let Some(trigger_mod) = trigger_as_mod else {
        return Ok(vec![primary]);
    };

    // Modifier-only combo. Build the full set of held modifiers and
    // generate one Shortcut per (held_mod_chosen_as_trigger, side).
    let all_mods = primary.mods | trigger_mod;
    let mut variants = Vec::new();
    for m in [
        Modifiers::CONTROL,
        Modifiers::ALT,
        Modifiers::SHIFT,
        Modifiers::SUPER,
    ] {
        if !all_mods.contains(m) {
            continue;
        }
        let other_mods = all_mods - m;
        let (l, r) = match m {
            Modifiers::CONTROL => (Code::ControlLeft, Code::ControlRight),
            Modifiers::ALT => (Code::AltLeft, Code::AltRight),
            Modifiers::SHIFT => (Code::ShiftLeft, Code::ShiftRight),
            Modifiers::SUPER => (Code::MetaLeft, Code::MetaRight),
            _ => continue,
        };
        variants.push(Shortcut::new(Some(other_mods), l));
        variants.push(Shortcut::new(Some(other_mods), r));
    }
    Ok(variants)
}

/// Canonical key set for a combo string ("Ctrl+Shift" → {"ctrl","shift"}).
/// Used for set-equality matching so press-order and L/R variants all
/// resolve to the same configured combo.
fn canonical_combo_str(s: &str) -> std::collections::BTreeSet<String> {
    s.split('+')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| match p.to_ascii_lowercase().as_str() {
            "ctrl" | "control" => "ctrl".to_string(),
            "alt" | "option" => "alt".to_string(),
            "shift" => "shift".to_string(),
            "cmd" | "super" | "meta" | "win" => "meta".to_string(),
            other => other.to_string(),
        })
        .collect()
}

/// Canonical key set for a fired Shortcut.
fn canonical_shortcut(s: &Shortcut) -> std::collections::BTreeSet<String> {
    let mut set = std::collections::BTreeSet::new();
    if s.mods.contains(Modifiers::CONTROL) {
        set.insert("ctrl".into());
    }
    if s.mods.contains(Modifiers::SHIFT) {
        set.insert("shift".into());
    }
    if s.mods.contains(Modifiers::ALT) {
        set.insert("alt".into());
    }
    if s.mods.contains(Modifiers::SUPER) {
        set.insert("meta".into());
    }
    set.insert(code_canonical_name(s.key));
    set
}

fn code_canonical_name(c: Code) -> String {
    match c {
        Code::ControlLeft | Code::ControlRight => "ctrl".into(),
        Code::ShiftLeft | Code::ShiftRight => "shift".into(),
        Code::AltLeft | Code::AltRight => "alt".into(),
        Code::MetaLeft | Code::MetaRight => "meta".into(),
        Code::KeyA => "a".into(), Code::KeyB => "b".into(), Code::KeyC => "c".into(),
        Code::KeyD => "d".into(), Code::KeyE => "e".into(), Code::KeyF => "f".into(),
        Code::KeyG => "g".into(), Code::KeyH => "h".into(), Code::KeyI => "i".into(),
        Code::KeyJ => "j".into(), Code::KeyK => "k".into(), Code::KeyL => "l".into(),
        Code::KeyM => "m".into(), Code::KeyN => "n".into(), Code::KeyO => "o".into(),
        Code::KeyP => "p".into(), Code::KeyQ => "q".into(), Code::KeyR => "r".into(),
        Code::KeyS => "s".into(), Code::KeyT => "t".into(), Code::KeyU => "u".into(),
        Code::KeyV => "v".into(), Code::KeyW => "w".into(), Code::KeyX => "x".into(),
        Code::KeyY => "y".into(), Code::KeyZ => "z".into(),
        Code::Digit0 => "0".into(), Code::Digit1 => "1".into(), Code::Digit2 => "2".into(),
        Code::Digit3 => "3".into(), Code::Digit4 => "4".into(), Code::Digit5 => "5".into(),
        Code::Digit6 => "6".into(), Code::Digit7 => "7".into(), Code::Digit8 => "8".into(),
        Code::Digit9 => "9".into(),
        Code::F1 => "f1".into(), Code::F2 => "f2".into(), Code::F3 => "f3".into(),
        Code::F4 => "f4".into(), Code::F5 => "f5".into(), Code::F6 => "f6".into(),
        Code::F7 => "f7".into(), Code::F8 => "f8".into(), Code::F9 => "f9".into(),
        Code::F10 => "f10".into(), Code::F11 => "f11".into(), Code::F12 => "f12".into(),
        Code::Space => "space".into(),
        Code::Enter => "enter".into(),
        Code::Escape => "escape".into(),
        Code::Tab => "tab".into(),
        Code::Backspace => "backspace".into(),
        Code::Delete => "delete".into(),
        Code::Insert => "insert".into(),
        Code::Home => "home".into(),
        Code::End => "end".into(),
        Code::PageUp => "pageup".into(),
        Code::PageDown => "pagedown".into(),
        Code::ArrowUp => "up".into(),
        Code::ArrowDown => "down".into(),
        Code::ArrowLeft => "left".into(),
        Code::ArrowRight => "right".into(),
        Code::Backquote => "`".into(),
        Code::Minus => "-".into(),
        Code::Equal => "=".into(),
        Code::BracketLeft => "[".into(),
        Code::BracketRight => "]".into(),
        Code::Backslash => "\\".into(),
        Code::Semicolon => ";".into(),
        Code::Quote => "'".into(),
        Code::Comma => ",".into(),
        Code::Period => ".".into(),
        Code::Slash => "/".into(),
        _ => format!("{:?}", c).to_lowercase(),
    }
}

fn code_from_str(s: &str) -> Option<Code> {
    Some(match s {
        // Whitespace / control
        "space" => Code::Space,
        "enter" | "return" => Code::Enter,
        "esc" | "escape" => Code::Escape,
        "tab" => Code::Tab,
        "backspace" => Code::Backspace,
        "delete" | "del" => Code::Delete,
        "insert" | "ins" => Code::Insert,
        "home" => Code::Home,
        "end" => Code::End,
        "pageup" | "pgup" => Code::PageUp,
        "pagedown" | "pgdn" => Code::PageDown,

        // Arrows
        "up" | "arrowup" => Code::ArrowUp,
        "down" | "arrowdown" => Code::ArrowDown,
        "left" | "arrowleft" => Code::ArrowLeft,
        "right" | "arrowright" => Code::ArrowRight,

        // Function keys
        "f1" => Code::F1, "f2" => Code::F2, "f3" => Code::F3, "f4" => Code::F4,
        "f5" => Code::F5, "f6" => Code::F6, "f7" => Code::F7, "f8" => Code::F8,
        "f9" => Code::F9, "f10" => Code::F10, "f11" => Code::F11, "f12" => Code::F12,

        // Punctuation / symbols (US-layout names)
        "backtick" | "`" => Code::Backquote,
        "-" | "minus" => Code::Minus,
        "=" | "equal" | "equals" => Code::Equal,
        "[" | "leftbracket" => Code::BracketLeft,
        "]" | "rightbracket" => Code::BracketRight,
        "\\" | "backslash" => Code::Backslash,
        ";" | "semicolon" => Code::Semicolon,
        "'" | "quote" => Code::Quote,
        "," | "comma" => Code::Comma,
        "." | "period" => Code::Period,
        "/" | "slash" => Code::Slash,

        // Digits
        c if c.len() == 1 && c.chars().next().unwrap().is_ascii_digit() => {
            match c {
                "0" => Code::Digit0, "1" => Code::Digit1, "2" => Code::Digit2,
                "3" => Code::Digit3, "4" => Code::Digit4, "5" => Code::Digit5,
                "6" => Code::Digit6, "7" => Code::Digit7, "8" => Code::Digit8,
                "9" => Code::Digit9,
                _ => return None,
            }
        }

        // Letters
        c if c.len() == 1 && c.chars().next().unwrap().is_ascii_alphabetic() => {
            let up = c.to_ascii_uppercase();
            match up.as_str() {
                "A" => Code::KeyA, "B" => Code::KeyB, "C" => Code::KeyC, "D" => Code::KeyD,
                "E" => Code::KeyE, "F" => Code::KeyF, "G" => Code::KeyG, "H" => Code::KeyH,
                "I" => Code::KeyI, "J" => Code::KeyJ, "K" => Code::KeyK, "L" => Code::KeyL,
                "M" => Code::KeyM, "N" => Code::KeyN, "O" => Code::KeyO, "P" => Code::KeyP,
                "Q" => Code::KeyQ, "R" => Code::KeyR, "S" => Code::KeyS, "T" => Code::KeyT,
                "U" => Code::KeyU, "V" => Code::KeyV, "W" => Code::KeyW, "X" => Code::KeyX,
                "Y" => Code::KeyY, "Z" => Code::KeyZ,
                _ => return None,
            }
        }
        _ => return None,
    })
}

fn show_settings(app: &tauri::AppHandle) {
    if let Some(win) = app.get_webview_window("settings") {
        let _ = win.show();
        let _ = win.set_focus();
        // JS listens for this so it can replay the fade-in animation
        // each time the window is brought back from hidden.
        let _ = app.emit("settings-shown", ());
    }
}

fn hide_settings(app: &tauri::AppHandle) {
    if let Some(win) = app.get_webview_window("settings") {
        let _ = win.hide();
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // env_logger removed in favor of tauri-plugin-log below — only one
    // global log backend can be installed and the plugin handles both
    // stderr and file output.

    tauri::Builder::default()
        // File + stderr logging. Logs land in
        //   %APPDATA%\dev.vibetotext.app\logs\VibeToText.log
        // on Windows, so we can actually diagnose STT failures from the
        // user's machine instead of guessing blind.
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .level_for("vibe_to_text_lib", log::LevelFilter::Debug)
                .targets([
                    tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::Stdout),
                    tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::LogDir {
                        file_name: None,
                    }),
                ])
                .build(),
        )
        .plugin(tauri_plugin_clipboard_manager::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_autostart::init(
            tauri_plugin_autostart::MacosLauncher::LaunchAgent,
            // Pass `--minimized` so the app starts hidden in the tray on
            // login instead of popping the settings window in the user's face.
            Some(vec!["--minimized"]),
        ))
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(|app, shortcut, event| {
                    let app = app.clone();
                    // Snapshot the fired shortcut's canonical key set so we
                    // can match it against the user's configured combos
                    // regardless of L/R variant or press order.
                    let fired_keys = canonical_shortcut(shortcut);
                    let is_press = event.state() == ShortcutState::Pressed;
                    log::debug!(
                        "hotkey fired: state={:?} mods={:?} key={:?} canonical={:?}",
                        event.state(),
                        shortcut.mods,
                        shortcut.key,
                        fired_keys
                    );
                    tauri::async_runtime::spawn(async move {
                        let state: tauri::State<Arc<AppState>> = app.state();
                        let cfg = state.config.lock().await.clone();

                        let toggle_keys = canonical_combo_str(&cfg.toggle_settings_hotkey);
                        let dictate_keys = canonical_combo_str(&cfg.dictate_hotkey);
                        let stt_toggle_keys =
                            canonical_combo_str(&cfg.stt_toggle_hotkey);

                        if !toggle_keys.is_empty() && fired_keys == toggle_keys {
                            if !is_press { return; }
                            // Settings hotkey: classic toggle on press.
                            if let Some(win) = app.get_webview_window("settings") {
                                if win.is_visible().unwrap_or(false) {
                                    let _ = win.hide();
                                } else {
                                    let _ = win.show();
                                    let _ = win.set_focus();
                                    let _ = app.emit("settings-shown", ());
                                }
                            }
                        } else if !dictate_keys.is_empty() && fired_keys == dictate_keys {
                            // Dictation hotkey: push-to-talk.
                            //   Press   → start session + show overlay
                            //   Release → stop session + hide overlay
                            if is_press {
                                let _ = start_dictation(&app).await;
                            } else {
                                let _ = stop_dictation(&app).await;
                            }
                        } else if !stt_toggle_keys.is_empty()
                            && fired_keys == stt_toggle_keys
                        {
                            if !is_press { return; }
                            // STT master kill-switch: flip on press.
                            let _ = toggle_stt_enabled(&app).await;
                        }
                    });
                })
                .build(),
        )
        .setup(|app| {
            // Load (or initialize) config from disk.
            let cfg = config::load(app.handle()).unwrap_or_default();
            let analytics_state = analytics::load(app.handle());
            let state = Arc::new(AppState {
                config: Mutex::new(cfg.clone()),
                session: Mutex::new(None),
                backend: Mutex::new(None),
                current_backend: Mutex::new(None),
                analytics: Mutex::new(analytics_state),
                start_cancel: AtomicBool::new(false),
                starting: AtomicBool::new(false),
            });
            app.manage(state);

            // Windows-only: install the low-level keyboard hook used to
            // detect modifier-only push-to-talk hotkeys (Ctrl+Shift etc.)
            // that RegisterHotKey can't reliably deliver. The hook is
            // dormant until `register_hotkeys` arms it via set_watch_mask.
            #[cfg(target_os = "windows")]
            modifier_hook::install(app.handle().clone());

            // Warm the Whisper model in the background so the first
            // dictation doesn't pay a 1-3 s cold-start penalty (GGML
            // mmap + Metal buffer alloc on macOS).
            let app_for_warm = app.handle().clone();
            let cfg_for_warm = cfg.clone();
            tauri::async_runtime::spawn(async move {
                stt::warm_whisper(app_for_warm, cfg_for_warm).await;
            });

            // Self-test escape hatch: when `VIBE_TO_TEXT_SELF_TEST=1`
            // is set in the environment, run the bundled-wav self-
            // test once at startup and log the result. Used by the
            // automated verification in CI / dev to confirm that the
            // Moonshine pipeline (download → extract → load →
            // recognize) is wired correctly without needing a human
            // to press the dictate hotkey. NO-OP otherwise — adds
            // zero cost for normal users.
            if std::env::var("VIBE_TO_TEXT_SELF_TEST").as_deref() == Ok("1") {
                let app_for_test = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    // Brief delay so the warm-up gets a head start;
                    // running a parallel Moonshine load fights for
                    // CPU and just makes both slower.
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    log::info!("VIBE_TO_TEXT_SELF_TEST=1: running Moonshine self-test…");
                    match run_moonshine_self_test(app_for_test).await {
                        Ok(rows) => {
                            for row in &rows {
                                log::info!("self-test: {row}");
                            }
                            let passed = rows
                                .iter()
                                .filter(|r| r.get("ok").and_then(|b| b.as_bool()) == Some(true))
                                .count();
                            log::info!(
                                "self-test summary: {}/{} passed",
                                passed,
                                rows.len()
                            );
                        }
                        Err(e) => log::warn!("self-test failed: {e}"),
                    }
                });
            }

            // Register hotkeys.
            register_hotkeys(app.handle(), &cfg)?;

            // System tray. Left-click → open settings, right-click → menu
            // with "Show settings" and "Quit". Without an explicit
            // `.icon(...)` Tauri 2 paints a blank tray icon, and without
            // `.menu(...)` right-click does nothing — both were the
            // bugs we just hit.
            let show_item = MenuItemBuilder::with_id("show", "Show settings").build(app)?;
            let quit_item = MenuItemBuilder::with_id("quit", "Quit VibeToText").build(app)?;
            let sep = PredefinedMenuItem::separator(app)?;
            let tray_menu = MenuBuilder::new(app)
                .item(&show_item)
                .item(&sep)
                .item(&quit_item)
                .build()?;

            let mut tray_builder = TrayIconBuilder::with_id("main")
                .tooltip("VibeToText — push to talk for dictation")
                .menu(&tray_menu)
                .show_menu_on_left_click(false)
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "show" => show_settings(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, event| {
                    // Open the settings window only on left-button RELEASE,
                    // so we don't double-fire on the press.
                    if let TrayIconEvent::Click {
                        button: MouseButton::Left,
                        button_state: MouseButtonState::Up,
                        ..
                    } = event
                    {
                        show_settings(tray.app_handle());
                    }
                });
            // Use the bundled window icon as the tray icon. Falls back
            // gracefully if for some reason it's missing.
            if let Some(icon) = app.default_window_icon().cloned() {
                tray_builder = tray_builder.icon(icon);
            }
            let _tray = tray_builder.build(app)?;

            // Show settings window unless we're being autostarted with
            // `--minimized` (in which case the user just expects us in
            // the tray quietly). First-run always shows so a brand-new
            // user is guided to settings even on autostart.
            let started_minimized = std::env::args().any(|a| a == "--minimized");
            if cfg.first_run || !started_minimized {
                show_settings(app.handle());
            }

            // Make settings hide instead of quitting on close.
            if let Some(win) = app.get_webview_window("settings") {
                let app_handle = app.handle().clone();
                win.on_window_event(move |e| {
                    if let tauri::WindowEvent::CloseRequested { api, .. } = e {
                        api.prevent_close();
                        hide_settings(&app_handle);
                    }
                });
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_config,
            save_config,
            toggle_dictation,
            current_backend,
            whisper_model_present,
            download_whisper_model,
            run_moonshine_self_test,
            get_analytics,
            reset_analytics,
            pause_hotkeys,
            resume_hotkeys,
            set_auto_start,
            is_auto_start_enabled,
            set_stt_enabled,
        ])
        .run(tauri::generate_context!())
        .expect("error while running VibeToText");
}

/// Push-to-talk: start (idempotent — held key auto-repeats Press, ignore re-entry).
///
/// Critical: this function does NOT hold the session lock while it
/// awaits backend selection or DictationSession::start. Both of those
/// can block for seconds (Whisper model load on first use, plus the
/// initial ~547 MB download if it hasn't run yet), and if we held the
/// lock the whole time, `stop_dictation` would block too — the overlay
/// would stay stuck on "Preparing model" until the slow load finished,
/// even after the user already released the hotkey.
///
/// Instead we track a separate `starting` flag (bool, lock-free) for
/// duplicate-start suppression, and `start_cancel` (also lock-free)
/// for the "user released mid-load" case. Only the brief moment of
/// installing the freshly-built session takes the session lock.
pub(crate) async fn start_dictation(app: &tauri::AppHandle) -> anyhow::Result<()> {
    let state: tauri::State<Arc<AppState>> = app.state();

    // Master kill-switch. If the user has flipped STT off in settings
    // (or via the toggle hotkey), the dictate hotkey still fires but
    // we no-op silently. No overlay, no audio init.
    if !state.config.lock().await.stt_enabled {
        log::debug!("start_dictation: STT disabled, ignoring press");
        return Ok(());
    }

    // Suppress duplicate starts (auto-repeat keydowns from Windows
    // while the dictate hotkey is still held). swap returns the
    // PREVIOUS value — true means a start is already in flight.
    //
    // When the user presses again WHILE a previous start is still
    // loading the model (e.g., they tapped the hotkey, released
    // before the model loaded, then tapped again), we don't try to
    // start a second instance — but we still re-show the overlay
    // and re-emit the "warming" state so the user gets visual
    // feedback that their press was heard. Without this they see a
    // popup the first time and then nothing on subsequent presses
    // until the in-flight start finishes cancelling.
    if state.starting.swap(true, AtomicOrdering::SeqCst) {
        log::debug!("start_dictation: duplicate press while loading; refreshing overlay");
        // Cancel the previous in-flight start so the user's new
        // press takes priority instead of getting silently dropped
        // when the in-flight one finishes. The in-flight task polls
        // start_cancel at every await point and bails cleanly.
        // (We deliberately do NOT call stop_dictation here — there
        // may not be a session to stop yet.)
        state.start_cancel.store(true, AtomicOrdering::SeqCst);
        // Surface UI again so the user knows the press registered.
        let _ = app.emit("dictation-state", "warming");
        show_overlay(app);
        return Ok(());
    }
    // Reset the cancel flag for this new press cycle.
    state.start_cancel.store(false, AtomicOrdering::SeqCst);

    // Quick non-blocking check that no session exists yet. This lock
    // is held for nanoseconds.
    {
        let session = state.session.lock().await;
        if session.is_some() {
            state.starting.store(false, AtomicOrdering::SeqCst);
            return Ok(());
        }
    }

    // Fire visual feedback immediately. Even if backend init is slow,
    // the user sees the overlay right away.
    let _ = app.emit("dictation-state", "listening");
    show_overlay(app);

    // First-dictation race: if the user hits the hotkey before the
    // background Whisper warm-up finishes, `select_backend` will
    // block for the duration of CT2 model load + CUDA init (~1-3 s
    // warm, plus the 150 MB download on a truly cold machine).
    // Surface this with a "warming" event so the overlay shows
    // what's happening instead of looking frozen.
    let warming_emitted = state.backend.lock().await.is_none();
    if warming_emitted {
        let _ = app.emit("dictation-state", "warming");
    }

    log::info!("start_dictation: selecting backend");
    let cfg = state.config.lock().await.clone();
    let backend = match stt::select_backend(app, &cfg).await {
        Ok(b) => b,
        Err(e) => {
            log::error!("start_dictation: backend selection failed: {e:#}");
            let _ = app.emit("dictation-error", format!("Backend unavailable: {e}"));
            let _ = app.emit("dictation-state", "idle");
            hide_overlay(app);
            state.starting.store(false, AtomicOrdering::SeqCst);
            return Err(e);
        }
    };

    // Did the user release while we were loading? If so, abort — they
    // don't want to dictate anymore. This is the path that fixes the
    // "overlay stuck on Preparing model" bug.
    if state.start_cancel.load(AtomicOrdering::SeqCst) {
        log::info!("start_dictation: cancelled by user release during backend load");
        let _ = app.emit("dictation-state", "idle");
        hide_overlay(app);
        state.starting.store(false, AtomicOrdering::SeqCst);
        return Ok(());
    }

    let backend_name = backend.name();
    *state.current_backend.lock().await = Some(backend_name);
    log::info!("start_dictation: selected backend = {backend_name}");
    let _ = app.emit("backend-selected", backend_name);

    // Backend is ready — drop the warming UI state if we showed it.
    if warming_emitted {
        let _ = app.emit("dictation-state", "listening");
    }

    let s = match stt::DictationSession::start(app.clone(), cfg, backend).await {
        Ok(s) => s,
        Err(e) => {
            log::error!("start_dictation: session start failed: {e:#}");
            let _ = app.emit("dictation-error", format!("Couldn't start audio: {e}"));
            let _ = app.emit("dictation-state", "idle");
            hide_overlay(app);
            state.starting.store(false, AtomicOrdering::SeqCst);
            return Err(e);
        }
    };

    // One more cancel check before installing — user might have
    // released between session start and now.
    if state.start_cancel.load(AtomicOrdering::SeqCst) {
        log::info!("start_dictation: cancelled by user release during audio init");
        // Fire-and-forget stop on the session we just built so cpal
        // releases the mic.
        let _ = s.stop().await;
        let _ = app.emit("dictation-state", "idle");
        hide_overlay(app);
        state.starting.store(false, AtomicOrdering::SeqCst);
        return Ok(());
    }

    {
        let mut session = state.session.lock().await;
        *session = Some(s);
    }
    state.starting.store(false, AtomicOrdering::SeqCst);
    log::info!("start_dictation: session started OK");
    Ok(())
}

/// Flip the master STT enable switch. Persists to disk, emits an
/// `stt-enabled-changed` event so the settings UI can reflect the new
/// value, and stops any in-flight dictation if we're turning OFF.
pub(crate) async fn toggle_stt_enabled(app: &tauri::AppHandle) -> anyhow::Result<bool> {
    let state: tauri::State<Arc<AppState>> = app.state();
    let new_value = {
        let mut cfg = state.config.lock().await;
        cfg.stt_enabled = !cfg.stt_enabled;
        let v = cfg.stt_enabled;
        let _ = config::save(app, &cfg);
        v
    };
    log::info!("toggle_stt_enabled: STT is now {}", if new_value { "ON" } else { "OFF" });
    let _ = app.emit("stt-enabled-changed", new_value);
    // If we're turning STT off, terminate any current session.
    if !new_value {
        let _ = stop_dictation(app).await;
    }
    Ok(new_value)
}

/// IPC: explicit set (used by the settings UI toggle switch).
#[tauri::command]
async fn set_stt_enabled(
    app: tauri::AppHandle,
    state: tauri::State<'_, Arc<AppState>>,
    enabled: bool,
) -> Result<(), String> {
    {
        let mut cfg = state.config.lock().await;
        if cfg.stt_enabled == enabled {
            return Ok(());
        }
        cfg.stt_enabled = enabled;
        config::save(&app, &cfg).map_err(|e| e.to_string())?;
    }
    log::info!("set_stt_enabled: {}", if enabled { "ON" } else { "OFF" });
    let _ = app.emit("stt-enabled-changed", enabled);
    if !enabled {
        let _ = stop_dictation(&app).await;
    }
    Ok(())
}

/// Push-to-talk: stop (idempotent — Released without an active session is a no-op).
///
/// Flow: emit "transcribing" the moment the user releases (so the overlay
/// can swap the waveform for a spinner), then await `s.stop()` which
/// blocks until the backend has finalized + injected the transcript,
/// then emit "idle" and hide the overlay. Without the "transcribing"
/// beat the overlay would freeze on a dead waveform for the duration of
/// inference and feel broken.
///
/// We ALSO set `start_cancel` so any in-flight `start_dictation`
/// (e.g. one that's currently blocked on Whisper model load) bails
/// out instead of installing a session the user no longer wants.
pub(crate) async fn stop_dictation(app: &tauri::AppHandle) -> anyhow::Result<()> {
    let state: tauri::State<Arc<AppState>> = app.state();

    // Signal any concurrent start_dictation to abort (it polls this
    // flag at await boundaries).
    state.start_cancel.store(true, AtomicOrdering::SeqCst);

    let session = state.session.lock().await.take();
    if let Some(s) = session {
        let _ = app.emit("dictation-state", "transcribing");
        s.stop().await?;
    }
    // Always emit idle + hide, even if there was no session (e.g.
    // start_dictation was still warming up). Otherwise the overlay
    // can stay stuck on "Preparing model" forever.
    let _ = app.emit("dictation-state", "idle");
    hide_overlay(app);
    Ok(())
}

fn show_overlay(app: &tauri::AppHandle) {
    let Some(win) = app.get_webview_window("overlay") else {
        log::warn!("show_overlay: overlay window not found");
        return;
    };
    // Re-center horizontally on the primary monitor each time so the
    // overlay still looks right after monitor changes.
    if let Ok(Some(monitor)) = win.primary_monitor() {
        let size = monitor.size();
        let scale = monitor.scale_factor();
        // Window logical size from tauri.conf.json. Convert from physical px.
        let win_w_logical = 460u32;
        let win_h_logical = 130u32;
        let target_x_phys =
            ((size.width as f64) - (win_w_logical as f64) * scale) / 2.0;
        // Anchor near the bottom of the screen, taskbar-aware-ish (60px gap).
        let target_y_phys =
            (size.height as f64) - (win_h_logical as f64) * scale - 80.0 * scale;
        let _ = win.set_position(tauri::PhysicalPosition::new(
            target_x_phys.max(0.0),
            target_y_phys.max(0.0),
        ));
        log::debug!(
            "show_overlay: positioned at ({}, {}) on monitor {}x{} @{}x",
            target_x_phys,
            target_y_phys,
            size.width,
            size.height,
            scale
        );
    } else {
        log::warn!("show_overlay: primary_monitor query failed; using last position");
    }
    if let Err(e) = win.set_ignore_cursor_events(true) {
        log::warn!("show_overlay: set_ignore_cursor_events: {e}");
    }
    // Set always-on-top BEFORE show so the window pops in already
    // raised — avoids a frame where it appears under the focused app.
    if let Err(e) = win.set_always_on_top(true) {
        log::warn!("show_overlay: set_always_on_top: {e}");
    }
    match win.show() {
        Ok(()) => log::debug!("show_overlay: win.show() OK"),
        Err(e) => log::error!("show_overlay: win.show() failed: {e}"),
    }
}

fn hide_overlay(app: &tauri::AppHandle) {
    if let Some(win) = app.get_webview_window("overlay") {
        let _ = win.hide();
    }
}

// (Whisper-rs statically links whisper.cpp into the app binary at
// build time via CMake, so there are no runtime DLLs / dylibs to
// resolve from Rust. The previous Moonshine + ONNX Runtime DLL-search
// dance has been removed entirely.)
