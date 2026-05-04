// STT facade: a `DictationSession` owns audio capture + a backend task,
// and dispatches frames into a pluggable `SttBackend` impl.
//
// Today there's exactly one backend — Whisper via whisper.cpp — but
// the trait stays so we can drop in alternatives (online APIs, future
// Voxtral revival, etc.) without surgery on the call sites in
// `lib.rs`.

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

pub use whisper::WhisperBackend;

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
        self.cancel.store(true, Ordering::SeqCst);
        if let Some(a) = self.audio.lock().await.take() {
            a.stop();
        }
        if let Some(h) = self.handle.lock().await.take() {
            let _ = h.await;
        }
        Ok(())
    }
}

/// Resolve the backend to use. With Whisper as the only backend, this is
/// just a lazy-init wrapper around `WhisperBackend::load`. The result is
/// cached on `AppState` so we don't re-pay the ~547 MB GGML load on each
/// dictation.
pub async fn select_backend(app: &AppHandle, cfg: &AppConfig) -> Result<Arc<dyn SttBackend>> {
    Ok(get_or_init_whisper(app, cfg).await?)
}

async fn get_or_init_whisper(
    app: &AppHandle,
    cfg: &AppConfig,
) -> Result<Arc<dyn SttBackend>> {
    let state: tauri::State<Arc<crate::AppState>> = app.state();

    // Fast path: if a backend is already cached, just hand it back.
    // We hold the lock for nanoseconds; the actual heavy work
    // happens outside it.
    {
        let guard = state.whisper.lock().await;
        if let Some(b) = guard.as_ref() {
            return Ok(b.clone() as Arc<dyn SttBackend>);
        }
    }

    // Slow path: build the backend. We do this OUTSIDE the lock so
    // model load (1-3 s) doesn't block other awaiters checking
    // `state.whisper`. If two dictations race in here at the same
    // time, only one wins the install — the loser drops its built
    // backend, which is mildly wasteful but safe.
    let _ = app.emit("backend-status", "whisper:loading");

    // Resolve the device + model PAIR up front. The model id we
    // download depends on which device CT2 will run on (base.en
    // for CPU, small.en for GPU by default — user-overridable in
    // settings).
    let choice = whisper::resolve_runtime(cfg)
        .map_err(|e| anyhow!("failed to resolve Whisper runtime: {e:#}"))?;
    log::info!(
        "resolved runtime: device={} compute={} model={}",
        choice.device,
        choice.compute_type,
        choice.model_id
    );

    let model_dir =
        crate::models::ensure_whisper_ready(app, cfg, &choice.model_id).await?;
    let backend = Arc::new(
        WhisperBackend::new(model_dir, choice)
            .await
            .map_err(|e| anyhow!("failed to initialize Whisper backend: {e:#}"))?,
    );
    let _ = app.emit("backend-status", "whisper:ready");

    let mut guard = state.whisper.lock().await;
    if guard.is_none() {
        *guard = Some(backend.clone());
    }
    let installed = guard.as_ref().unwrap().clone();
    Ok(installed as Arc<dyn SttBackend>)
}

/// Best-effort: pre-load the CT2 Whisper model in the background at
/// app start so the first dictation doesn't pay the cold-start cost
/// (model file mmap + GPU buffer alloc + CUDA init, ~1-3 s).
/// Skips silently when the model isn't downloaded yet — auto-
/// downloading 150 MB at every app launch would be surprising on
/// metered links. The "Download offline assets" button in settings
/// is the explicit-consent path.
pub async fn warm_whisper(app: AppHandle, cfg: AppConfig) {
    if !crate::models::whisper_already_downloaded(&app, &cfg) {
        log::info!("Whisper warm-up skipped: model not yet downloaded");
        return;
    }
    log::info!("warming Whisper in background…");
    match get_or_init_whisper(&app, &cfg).await {
        Ok(_) => log::info!("Whisper warm-up complete"),
        Err(e) => log::warn!("Whisper warm-up failed: {e:#}"),
    }
}
