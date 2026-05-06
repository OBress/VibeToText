// STT facade: a `DictationSession` owns audio capture + a backend task,
// and dispatches frames into a pluggable `SttBackend` impl.
//
// Two backends today:
//   - `whisper` (CT2 + faster-whisper-* HF repos): used for GPU
//     mode and as a Whisper fallback for CPU. Higher quality
//     ceiling, multilingual-capable.
//   - `moonshine` (sherpa-onnx + Moonshine v1 base-en INT8):
//     the default CPU choice. RTFx 25-40× on AVX2 CPUs (vs
//     Whisper small.en at ~3-5×), 6.65 % WER, English only.
//
// Dispatch happens in `select_backend` based on the resolved
// device + the user's `cpu_engine` preference (Moonshine or
// Whisper). GPU always uses Whisper.

mod moonshine;
mod whisper;

use crate::audio::AudioCapture;
use crate::config::AppConfig;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager};
use tokio::{sync::Mutex, task::JoinHandle};

pub use moonshine::MoonshineBackend;
pub use whisper::WhisperBackend;

/// Drain frames the audio thread is delivering during cpal's
/// tail-flush window. cpal's WASAPI/CoreAudio backend has
/// 10-50 ms of internal latency between a physical mic sample and
/// our callback firing, so the FINAL frames the user spoke arrive
/// AFTER `cancel` has been set. We poll the channel for a fixed
/// duration, sleeping briefly when it's empty, so we collect frames
/// as cpal delivers them rather than only the ones already queued
/// at the moment we entered drain.
///
/// Pairs with the 250 ms keep-alive in DictationSession::stop —
/// cpal stays running for at least that long so it can deliver,
/// while we sit here pulling.
pub(crate) async fn drain_remaining_audio(
    audio: &Arc<Mutex<Option<crate::audio::AudioCapture>>>,
    buf: &mut Vec<f32>,
    max_samples: usize,
) -> usize {
    use std::time::Instant;
    let deadline = Instant::now() + Duration::from_millis(220);
    let mut drained = 0usize;
    // We need the channel handle but don't want to hold the audio
    // mutex for the whole drain (DictationSession::stop's audio.stop()
    // path also takes that lock). Snapshot the receiver and release.
    let frames = {
        let guard = audio.lock().await;
        guard.as_ref().map(|a| a.frames.clone())
    };
    let Some(frames) = frames else {
        return 0;
    };
    while Instant::now() < deadline {
        match frames.try_recv() {
            Ok(frame) => {
                let take = max_samples.saturating_sub(buf.len()).min(frame.0.len());
                if take == 0 {
                    break;
                }
                buf.extend_from_slice(&frame.0[..take]);
                drained += take;
            }
            Err(_) => {
                // Channel empty for now — wait for cpal's next callback.
                tokio::time::sleep(Duration::from_millis(15)).await;
            }
        }
    }
    drained
}

/// Trait every transcription backend implements. `run` consumes audio
/// frames until `cancel` flips, then finalizes and injects the transcript.
#[async_trait]
pub trait SttBackend: Send + Sync {
    fn name(&self) -> &'static str;

    async fn run(
        &self,
        app: AppHandle,
        cfg: AppConfig,
        audio: Arc<Mutex<Option<AudioCapture>>>,
        cancel: Arc<AtomicBool>,
    ) -> Result<()>;
}

pub struct DictationSession {
    cancel: Arc<AtomicBool>,
    audio: Arc<Mutex<Option<AudioCapture>>>,
    handle: Mutex<Option<JoinHandle<()>>>,
}

impl DictationSession {
    pub async fn start(
        app: AppHandle,
        cfg: AppConfig,
        backend: Arc<dyn SttBackend>,
    ) -> Result<Self> {
        let cancel = Arc::new(AtomicBool::new(false));
        let audio = AudioCapture::start(&cfg.mic_device)?;
        let level_rx = audio.levels.clone();
        let audio_arc = Arc::new(Mutex::new(Some(audio)));

        let app_run = app.clone();
        let cfg_run = cfg.clone();
        let audio_run = audio_arc.clone();
        let cancel_run = cancel.clone();
        let backend_run = backend.clone();

        let handle = tokio::spawn(async move {
            if let Err(e) = backend_run.run(app_run, cfg_run, audio_run, cancel_run).await {
                log::error!("STT backend error: {e:#}");
            }
        });

        // Pump audio levels to the overlay window. ~50Hz of float events.
        // Drains the channel, drops backlog, sends one event per drain so
        // the UI never sees stale data.
        let app_levels = app.clone();
        let cancel_levels = cancel.clone();
        tokio::spawn(async move {
            loop {
                if cancel_levels.load(Ordering::SeqCst) {
                    break;
                }
                let mut latest: Option<f32> = None;
                while let Ok(l) = level_rx.try_recv() {
                    latest = Some(l.0);
                }
                if let Some(rms) = latest {
                    let _ = app_levels.emit_to(
                        "overlay",
                        "audio-level",
                        rms,
                    );
                }
                tokio::time::sleep(Duration::from_millis(20)).await;
            }
        });

        Ok(Self {
            cancel,
            audio: audio_arc,
            handle: Mutex::new(Some(handle)),
        })
    }

    pub async fn stop(self) -> Result<()> {
        // Set cancel BEFORE killing cpal so the backend's main loop
        // exits on the next iteration. The backend then enters its
        // post-cancel drain phase.
        self.cancel.store(true, Ordering::SeqCst);

        // CRITICAL: keep cpal alive for ~250 ms after cancel so its
        // hardware audio buffer has time to flush into our channel.
        // Without this delay, the user's last word gets cut off:
        // cpal's WASAPI/CoreAudio backend has 10-50 ms of internal
        // latency before a sample physically captured at the mic
        // shows up in our callback. If we kill the stream the
        // moment the user releases the hotkey, those samples are
        // discarded by the OS audio driver before they ever reach
        // us. 250 ms is safely past the worst-case latency on
        // every audio backend I tested without being user-
        // perceivable.
        //
        // The backend's drain loop (which runs concurrently with
        // this sleep, in a separate task) reads the frames as cpal
        // delivers them.
        tokio::time::sleep(Duration::from_millis(250)).await;

        if let Some(a) = self.audio.lock().await.take() {
            a.stop();
        }
        if let Some(h) = self.handle.lock().await.take() {
            let _ = h.await;
        }
        Ok(())
    }
}

/// Decide which concrete backend the current config + runtime should
/// produce, BEFORE building it. Pure function over the config + the
/// CT2 device probe — no model load, no network. The result feeds two
/// places: the cache-staleness check (does the cached backend match
/// what we'd build now?) and the actual lazy-init.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BackendKind {
    /// Whisper via CT2 + faster-whisper-* HF repos. Used for GPU mode
    /// always, and CPU mode when the user picks "whisper" in settings.
    Whisper,
    /// Moonshine v1 base-en INT8 via sherpa-onnx. CPU-only path;
    /// the default CPU choice (RTFx 25-40× vs Whisper small.en at
    /// ~3-5×).
    Moonshine,
}

fn pick_backend_kind(cfg: &AppConfig) -> Result<BackendKind> {
    let choice = whisper::resolve_runtime(cfg)
        .map_err(|e| anyhow!("failed to resolve runtime: {e:#}"))?;
    use ct2rs::sys::Device;
    // GPU mode always means Whisper (Moonshine has no CUDA path).
    // CPU mode honors `cpu_engine`; default is Moonshine. The Device
    // enum is `#[non_exhaustive]` (or has a sentinel repr range) per
    // ct2rs 0.9, so we need a wildcard for forward-compat with future
    // backends like Metal/Vulkan.
    let kind = if choice.device == Device::CUDA {
        BackendKind::Whisper
    } else {
        // Treat everything non-CUDA as CPU for dispatch purposes.
        match cfg.cpu_engine.to_ascii_lowercase().as_str() {
            "whisper" => BackendKind::Whisper,
            // anything else → Moonshine. "moonshine" is the canonical
            // value but we accept "" / unknown defensively because
            // older config.json files won't have the field.
            _ => BackendKind::Moonshine,
        }
    };
    Ok(kind)
}

/// Backend names emitted to the UI / cache check. Must stay in sync
/// with the `name()` returns of each impl below.
const WHISPER_PREFIX: &str = "whisper:";
const MOONSHINE_PREFIX: &str = "moonshine:";

fn cached_matches(kind: BackendKind, name: &str) -> bool {
    match kind {
        BackendKind::Whisper => name.starts_with(WHISPER_PREFIX),
        BackendKind::Moonshine => name.starts_with(MOONSHINE_PREFIX),
    }
}

/// Resolve the backend to use. Lazy-init + cache: the backend is built
/// on first dictation and reused thereafter, until config changes
/// invalidate the cache (handled in `lib.rs::save_config`).
pub async fn select_backend(app: &AppHandle, cfg: &AppConfig) -> Result<Arc<dyn SttBackend>> {
    let state: tauri::State<Arc<crate::AppState>> = app.state();
    let want = pick_backend_kind(cfg)?;

    // Fast path: cached backend AND it's the kind we want.
    {
        let guard = state.backend.lock().await;
        if let Some(b) = guard.as_ref() {
            if cached_matches(want, b.name()) {
                return Ok(b.clone());
            }
            // Otherwise the cached backend is stale (user flipped
            // mode or cpu_engine). Drop the lock and rebuild below;
            // the actual cache replacement happens after the new
            // backend is constructed.
            log::info!(
                "cached backend {} doesn't match desired kind {:?}; rebuilding",
                b.name(),
                want
            );
        }
    }

    // Slow path: build the requested backend OUTSIDE the lock so
    // model load (1-3 s for Whisper, ~500 ms for Moonshine warm) does
    // not block other awaiters reading `state.backend`. If two
    // dictations race in here only one wins the install; the loser
    // drops its built object — mildly wasteful but safe.
    let new_backend = match want {
        BackendKind::Whisper => build_whisper(app, cfg).await?,
        BackendKind::Moonshine => build_moonshine(app).await?,
    };

    let mut guard = state.backend.lock().await;
    // Re-check under the lock: another caller may have populated it
    // with the same kind in the gap. If so, prefer theirs to avoid
    // discarding their work; otherwise install ours.
    if guard
        .as_ref()
        .map(|b| !cached_matches(want, b.name()))
        .unwrap_or(true)
    {
        *guard = Some(new_backend.clone());
    }
    Ok(guard.as_ref().unwrap().clone())
}

async fn build_whisper(app: &AppHandle, cfg: &AppConfig) -> Result<Arc<dyn SttBackend>> {
    let _ = app.emit("backend-status", "whisper:loading");

    // Resolve the device + model PAIR. The model id depends on the
    // resolved device (base.en for CPU, medium.en for GPU by default
    // — user-overridable in settings).
    let choice = whisper::resolve_runtime(cfg)
        .map_err(|e| anyhow!("failed to resolve Whisper runtime: {e:#}"))?;
    log::info!(
        "resolved Whisper runtime: device={} compute={} model={}",
        choice.device,
        choice.compute_type,
        choice.model_id
    );

    let model_dir = crate::models::ensure_whisper_ready(app, cfg, &choice.model_id).await?;
    let backend: Arc<dyn SttBackend> = Arc::new(
        WhisperBackend::new(model_dir, choice)
            .await
            .map_err(|e| anyhow!("failed to initialize Whisper backend: {e:#}"))?,
    );
    let _ = app.emit("backend-status", "whisper:ready");
    Ok(backend)
}

async fn build_moonshine(app: &AppHandle) -> Result<Arc<dyn SttBackend>> {
    let _ = app.emit("backend-status", "moonshine:loading");
    log::info!("resolved CPU runtime: backend=moonshine base-en (INT8)");
    let model_dir = crate::models::ensure_moonshine_ready(app).await?;
    let backend: Arc<dyn SttBackend> = Arc::new(
        MoonshineBackend::new(model_dir)
            .await
            .map_err(|e| anyhow!("failed to initialize Moonshine backend: {e:#}"))?,
    );
    let _ = app.emit("backend-status", "moonshine:ready");
    Ok(backend)
}

/// Best-effort: pre-load the active backend in the background at app
/// start so the first dictation doesn't pay the cold-start cost
/// (Whisper: model mmap + GPU buffer alloc + CUDA init, ~1-3 s;
/// Moonshine: ONNX session create, ~500 ms).
///
/// For backends with assets already on disk this is the warm-up
/// itself. For backends whose assets are missing the behavior splits:
///   - Whisper: skip silently. The default model varies a lot in
///     size (40 MB tiny → 1.5 GB large-v3) and we don't want to
///     surprise a metered-link user with a 1.5 GB download at app
///     launch. They'll get an explicit-consent flow via the
///     "Download model" button or the first dictation press.
///   - Moonshine: same skip-silent behavior; the user has to opt
///     in via "Download model" before Moonshine works. (We also
///     trigger download lazily on first dictation.)
pub async fn warm_whisper(app: AppHandle, cfg: AppConfig) {
    let want = match pick_backend_kind(&cfg) {
        Ok(w) => w,
        Err(e) => {
            log::warn!("warm-up skipped: backend kind resolution failed: {e:#}");
            return;
        }
    };
    let assets_present = match want {
        BackendKind::Whisper => crate::models::whisper_already_downloaded(&app, &cfg),
        BackendKind::Moonshine => crate::models::moonshine_already_downloaded(&app),
    };
    if !assets_present {
        // Moonshine is small (~250 MB) and is pinned to a single
        // file set, so we proactively fetch it on first launch when
        // it's the active backend — the alternative (silently
        // skip) would block the very first dictation for ~30 s on
        // download. Whisper still skips because variant size ranges
        // 40 MB → 1.5 GB and we don't want to surprise a metered-
        // link user.
        match want {
            BackendKind::Moonshine => {
                log::info!("warm-up: Moonshine assets missing, fetching now…");
                match crate::models::ensure_moonshine_ready(&app).await {
                    Ok(_) => log::info!("warm-up: Moonshine download complete"),
                    Err(e) => {
                        log::warn!("warm-up: Moonshine download failed: {e:#}");
                        return;
                    }
                }
            }
            BackendKind::Whisper => {
                log::info!(
                    "warm-up skipped: Whisper model not yet downloaded (use the Download button in settings)",
                );
                return;
            }
        }
    }
    log::info!("warming {:?} in background…", want);
    match select_backend(&app, &cfg).await {
        Ok(_) => log::info!("warm-up complete"),
        Err(e) => log::warn!("warm-up failed: {e:#}"),
    }
}
