// Moonshine base-en backend, powered by sherpa-onnx.
//
// Why a separate backend from Whisper:
//
//   - Moonshine has a fundamentally different architecture from
//     Whisper (sliding-window encoder vs Whisper's fixed 30s mel
//     padding). On a modern AVX2 CPU, Moonshine base hits RTFx
//     25-40× vs Whisper small.en at ~3-5× on the same hardware
//     — big win for the CPU-only path. WER is comparable: 6.65%
//     base / 7% small.en.
//
//   - Moonshine isn't supported by ct2rs/CTranslate2; we use
//     sherpa-onnx (the k2-fsa C++ ASR runtime that ships
//     prebuilt binaries via GitHub releases).
//
//   - We use Moonshine v1 base (split across 4 ONNX files:
//     preprocess, encode, cached_decode, uncached_decode) because
//     that's what k2-fsa publishes as `moonshine-base-en-int8`.
//     v2's single-file `merged_decoder` exists in sherpa-onnx but
//     k2-fsa hasn't shipped a base/medium English INT8 v2 build
//     in their `asr-models` release.
//
// We ship Moonshine as the DEFAULT CPU choice while keeping
// Whisper available for both CPU + GPU. The dispatch lives in
// `stt.rs::pick_backend_kind` — for `backend_mode = "cpu"` (or
// "auto" with no GPU) the runtime picks Moonshine; "gpu" mode
// always picks Whisper.

use crate::audio::AudioCapture;
use crate::config::AppConfig;
use crate::inject;
use crate::stt::SttBackend;
use crate::vad;
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use sherpa_onnx::{
    OfflineModelConfig, OfflineMoonshineModelConfig, OfflineRecognizer,
    OfflineRecognizerConfig,
};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex as StdMutex,
};
use std::time::Duration;
use tauri::AppHandle;
use tokio::sync::Mutex;

const SAMPLE_RATE_I32: i32 = 16_000;
const SAMPLE_RATE_USIZE: usize = 16_000;
/// Moonshine handles long audio internally via its sliding-window
/// encoder, so we don't need Whisper's strict 30 s cap. We still
/// bound here at 60 s to keep the in-memory audio buffer small.
const MAX_SECONDS: f32 = 60.0;

pub struct MoonshineBackend {
    inner: Arc<MoonshineInner>,
}

struct MoonshineInner {
    recognizer: OfflineRecognizer,
    /// sherpa-onnx's OfflineRecognizer creates fresh streams per
    /// utterance, but the underlying ONNX session has internal
    /// state we serialize through this mutex. In practice we only
    /// run one dictation at a time.
    inflight: StdMutex<()>,
}

// SAFETY: sherpa-onnx's OfflineRecognizer holds a *const pointer
// to a heap-allocated C++ object. The C++ side is thread-safe per
// the upstream docs (the sherpa-onnx-rs README explicitly notes
// it can be shared across threads as long as decode calls are
// serialized). We serialize via `inflight`.
unsafe impl Send for MoonshineInner {}
unsafe impl Sync for MoonshineInner {}

impl MoonshineBackend {
    /// Load the Moonshine v1 base-en model directory. Expects:
    ///   model_dir/preprocess.onnx
    ///   model_dir/encode.int8.onnx
    ///   model_dir/cached_decode.int8.onnx
    ///   model_dir/uncached_decode.int8.onnx
    ///   model_dir/tokens.txt
    pub async fn new(model_dir: PathBuf) -> Result<Self> {
        let preprocess = model_dir.join("preprocess.onnx");
        let encoder = model_dir.join("encode.int8.onnx");
        let cached_decoder = model_dir.join("cached_decode.int8.onnx");
        let uncached_decoder = model_dir.join("uncached_decode.int8.onnx");
        let tokens = model_dir.join("tokens.txt");

        for (label, p) in [
            ("preprocess.onnx", &preprocess),
            ("encode.int8.onnx", &encoder),
            ("cached_decode.int8.onnx", &cached_decoder),
            ("uncached_decode.int8.onnx", &uncached_decoder),
            ("tokens.txt", &tokens),
        ] {
            if !p.exists() {
                return Err(anyhow!(
                    "Moonshine model missing {}: {}",
                    label,
                    p.display()
                ));
            }
        }

        log::info!(
            "Moonshine: loading v1 base-en from {} (preprocess + encode + cached_decode + uncached_decode)",
            model_dir.display()
        );

        let threads = pick_thread_count();
        let recognizer = tokio::task::spawn_blocking(move || -> Result<OfflineRecognizer> {
            let mut config = OfflineRecognizerConfig::default();
            config.model_config = OfflineModelConfig {
                moonshine: OfflineMoonshineModelConfig {
                    preprocessor: Some(path_to_string(&preprocess)),
                    encoder: Some(path_to_string(&encoder)),
                    cached_decoder: Some(path_to_string(&cached_decoder)),
                    uncached_decoder: Some(path_to_string(&uncached_decoder)),
                    merged_decoder: None,
                },
                tokens: Some(path_to_string(&tokens)),
                num_threads: threads as i32,
                debug: false,
                ..Default::default()
            };
            OfflineRecognizer::create(&config)
                .ok_or_else(|| anyhow!("sherpa-onnx OfflineRecognizer::create returned None"))
        })
        .await
        .map_err(|e| anyhow!("Moonshine load task: {e}"))??;

        log::info!("MoonshineBackend ready: threads={threads}");
        Ok(Self {
            inner: Arc::new(MoonshineInner {
                recognizer,
                inflight: StdMutex::new(()),
            }),
        })
    }

    /// Run the recognizer over an in-memory PCM-f32 16 kHz buffer
    /// and return the raw transcript. No VAD trim, no analytics, no
    /// paste — used by the self-test IPC and any future "transcribe
    /// a file" feature. Heavy work runs on the blocking pool to
    /// keep Tokio responsive.
    pub async fn transcribe_samples(&self, samples: Vec<f32>) -> Result<String> {
        let inner = self.inner.clone();
        tokio::task::spawn_blocking(move || -> Result<String> {
            let _g = inner
                .inflight
                .lock()
                .map_err(|_| anyhow!("Moonshine inflight mutex poisoned"))?;
            let stream = inner.recognizer.create_stream();
            stream.accept_waveform(SAMPLE_RATE_I32, &samples);
            inner.recognizer.decode(&stream);
            let result = stream
                .get_result()
                .ok_or_else(|| anyhow!("Moonshine returned no result"))?;
            Ok(result.text)
        })
        .await
        .map_err(|e| anyhow!("Moonshine transcribe task: {e}"))?
    }
}

/// Same heuristic as the Whisper backend: physical-core
/// approximation, capped at 16. Moonshine's encoder is small enough
/// that going wider has diminishing returns, but it parallelizes
/// well up to that cap.
fn pick_thread_count() -> usize {
    let logical = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    (logical / 2).clamp(2, 16)
}

fn path_to_string(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

#[async_trait]
impl SttBackend for MoonshineBackend {
    fn name(&self) -> &'static str {
        "moonshine:base-en-int8"
    }

    async fn run(
        &self,
        app: AppHandle,
        cfg: AppConfig,
        audio: Arc<Mutex<Option<AudioCapture>>>,
        cancel: Arc<AtomicBool>,
    ) -> Result<()> {
        // 1) Buffer audio until the user releases the hotkey.
        // Identical loop to the Whisper backend — same audio source,
        // same cancellation semantics, same overflow behavior.
        let max_samples = (MAX_SECONDS * SAMPLE_RATE_USIZE as f32) as usize;
        let mut buf: Vec<f32> = Vec::with_capacity(max_samples);
        let mut warned_overflow = false;

        loop {
            if cancel.load(Ordering::SeqCst) {
                break;
            }
            let frame = {
                let guard = audio.lock().await;
                let Some(a) = guard.as_ref() else { break };
                a.frames.try_recv().ok()
            };
            match frame {
                Some(f) => {
                    if buf.len() + f.0.len() > max_samples {
                        if !warned_overflow {
                            log::warn!(
                                "Moonshine: audio exceeds {MAX_SECONDS}s cap; truncating"
                            );
                            warned_overflow = true;
                        }
                        let take = max_samples.saturating_sub(buf.len());
                        buf.extend_from_slice(&f.0[..take]);
                    } else {
                        buf.extend_from_slice(&f.0);
                    }
                }
                None => tokio::time::sleep(Duration::from_millis(10)).await,
            }
        }

        // Drain post-cancel frames so we don't lose the user's last
        // word or two. See the comment on `drain_remaining_audio` in
        // stt.rs for the full reasoning.
        let drained = crate::stt::drain_remaining_audio(&audio, &mut buf, max_samples).await;
        if drained > 0 {
            log::debug!(
                "Moonshine: drained {} samples ({:.0} ms) after cancel",
                drained,
                drained as f32 * 1000.0 / SAMPLE_RATE_USIZE as f32
            );
        }

        if buf.len() < SAMPLE_RATE_USIZE / 4 {
            log::info!(
                "Moonshine: too little audio ({} samples), skipping",
                buf.len()
            );
            return Ok(());
        }

        // 2) VAD trim — same module the Whisper backend uses.
        let trimmed = vad::trim_silence(&buf);
        let trimmed_samples: Vec<f32> = trimmed.to_vec();
        let audio_seconds = trimmed_samples.len() as f32 / SAMPLE_RATE_USIZE as f32;
        log::info!("Moonshine: transcribing {:.2}s of audio", audio_seconds);

        // 3) Run inference on the blocking pool. sherpa-onnx's
        //    decode call is synchronous CPU work; offloading keeps
        //    the Tokio runtime responsive.
        let inner = self.inner.clone();
        let started = std::time::Instant::now();
        let text = tokio::task::spawn_blocking(move || -> Result<String> {
            let _g = inner
                .inflight
                .lock()
                .map_err(|_| anyhow!("Moonshine inflight mutex poisoned"))?;
            let stream = inner.recognizer.create_stream();
            stream.accept_waveform(SAMPLE_RATE_I32, &trimmed_samples);
            inner.recognizer.decode(&stream);
            let result = stream
                .get_result()
                .ok_or_else(|| anyhow!("Moonshine returned no result"))?;
            Ok(result.text)
        })
        .await
        .map_err(|e| anyhow!("Moonshine inference task: {e}"))??;
        let elapsed = started.elapsed();
        log::info!(
            "Moonshine: transcription took {:?} (RTFx {:.1}x)",
            elapsed,
            audio_seconds / elapsed.as_secs_f32().max(0.001)
        );

        let mut text = text.trim().to_string();
        if text.is_empty() {
            log::info!("Moonshine: empty transcript");
            return Ok(());
        }

        // Moonshine doesn't suffer from Whisper's training-residue
        // hallucinations (no "Thanks for watching" pulled from
        // YouTube subtitles), so we skip the hallucination filter.
        // If real-world output ever needs filtering, lift the same
        // function from stt::whisper.

        if cfg.trailing_space && !text.ends_with(' ') {
            text.push(' ');
        }

        crate::analytics::record_from_backend(&app, text.trim(), elapsed, "moonshine").await;
        let _ = inject::paste_text(&app, &text);
        Ok(())
    }
}
