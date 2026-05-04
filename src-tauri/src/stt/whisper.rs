// Whisper backend powered by `ct2rs` (Rust bindings to OpenNMT's
// CTranslate2). This is the same inference engine that
// `faster-whisper` uses under the hood — and the reason that
// project hits RTFx ~3-5× on CPU and ~30-50× on CUDA where
// whisper.cpp on the same hardware tops out around 1× CPU.
//
// Architecture: a `WhisperBackend` owns one loaded `ct2rs::Whisper`
// instance for the lifetime of the app. Audio buffered while the
// user holds the dictate hotkey gets handed to it directly via a
// `tokio::task::spawn_blocking` call — no sidecar process, no HTTP,
// no WAV temp files. Transcript comes back as a `Vec<String>` and
// we paste it.
//
// Device + compute-type selection follows `cfg.backend_mode`:
//   - "auto"  (default) → CUDA if `get_device_count(CUDA) > 0`,
//                          else CPU.
//   - "gpu"            → CUDA, error if no device.
//   - "cpu"            → CPU.
//
// Compute type per device:
//   - CUDA: FLOAT16 — what faster-whisper's CUDA path uses.
//   - CPU:  INT8    — what faster-whisper's CPU fallback uses.

use crate::audio::AudioCapture;
use crate::config::AppConfig;
use crate::inject;
use crate::stt::SttBackend;
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use ct2rs::sys::{get_device_count, ComputeType, Device, WhisperOptions};
use ct2rs::{Config as CtConfig, Whisper};
use std::path::PathBuf;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex as StdMutex,
};
use std::time::Duration;
use tauri::AppHandle;
use tokio::sync::Mutex;

/// Patterns Whisper emits when it has nothing real to transcribe —
/// the model was trained on subtitle data, so it produces these
/// "fillers" on silent / noisy / music-only input. faster-whisper's
/// reference implementation strips them; we do the same.
///
/// Match is case-insensitive on the trimmed transcript. We do
/// EXACT-string match (after trim+lowercase) for the bracketed
/// markers, and SUBSTRING match for the longer phrases — Whisper
/// often surrounds them with extra punctuation.
const HALLUCINATION_EXACT: &[&str] = &[
    "[blank_audio]",
    "[ blank_audio ]",
    "[blank audio]",
    "[ blank audio ]",
    "[silence]",
    "[ silence ]",
    "(silence)",
    "( silence )",
    "[no speech]",
    "[ no speech ]",
    "(no speech)",
    "( no speech )",
    "[inaudible]",
    "[ inaudible ]",
    "(inaudible)",
    "( inaudible )",
    "[music]",
    "[ music ]",
    "(music)",
    "[applause]",
    "[laughter]",
    "♪",
    "♪♪",
    "♪♪♪",
];

/// Substring-match patterns. These are YouTube/TED training residue
/// that Whisper falls back to on quiet audio.
const HALLUCINATION_CONTAINS: &[&str] = &[
    "thanks for watching",
    "thank you for watching",
    "subtitles by",
    "subscribe to",
    "translated by",
];

// (We requested `return_no_speech_prob=true` in WhisperOptions
// hoping to filter on it, but ct2rs's high-level `Whisper::generate`
// only returns the transcript strings — the no-speech probability
// is computed but not surfaced. Falling back to text-pattern
// hallucination filter only. To get the probability back we'd need
// to drop down to `sys::Whisper::generate` and read
// `WhisperGenerationResult.no_speech_prob` directly. Tracked as
// future work alongside `initial_prompt` plumbing.)

const SAMPLE_RATE: u32 = 16_000;
// CTranslate2's Whisper transparently chunks audio longer than 30 s
// internally (n_samples loop in ct2rs::Whisper::generate), so we no
// longer need a hard cap here, but we keep one to bound the
// in-memory buffer and the user's worst-case wait. 60 s is well
// past any reasonable single-utterance dictation.
const MAX_SECONDS: f32 = 60.0;

pub struct WhisperBackend {
    inner: Arc<WhisperInner>,
    /// Stable name surfaced to the UI / analytics. Encodes both the
    /// engine and the active device so we can tell at a glance which
    /// path a given dictation took.
    backend_label: &'static str,
}

struct WhisperInner {
    whisper: Whisper,
    /// CT2's Whisper is `Send + Sync` (verified in upstream
    /// `unsafe impl`s), but `generate` performs internal mutable
    /// state changes that aren't safe across true parallel calls.
    /// Serialize them through a mutex. In practice we only ever
    /// run one dictation at a time, so this is uncontended.
    inflight: StdMutex<()>,
}

/// Resolved per-dictation device + compute + model triple. Returned
/// by `resolve_runtime` so that stt.rs can download the right model
/// before constructing the backend.
pub struct RuntimeChoice {
    pub device: Device,
    pub compute_type: ComputeType,
    pub model_id: String,
    pub label: &'static str,
}

/// Look at `cfg.backend_mode` plus the runtime CUDA probe and decide
/// (a) which device we'll run on, (b) which compute_type matches,
/// (c) which HF model id to download. Errors out for "gpu" mode
/// when no CUDA device is visible — silently downgrading would
/// surprise users who explicitly forced GPU.
pub fn resolve_runtime(cfg: &AppConfig) -> Result<RuntimeChoice> {
    let cuda_count = get_device_count(Device::CUDA);
    let cuda_ok = cuda_count > 0;
    log::info!("CT2 device probe: CUDA device_count={cuda_count}");

    // GPU compute type: INT8_FLOAT16 (int8 weights, fp16 matmul).
    // ~10-20% faster than plain FLOAT16 on small Whisper models
    // because matmul is memory-bandwidth-bound; halving weight
    // size halves the bandwidth needed. Quality cost: <0.5% WER.
    let gpu_compute = ComputeType::INT8_FLOAT16;

    let pick_gpu = || RuntimeChoice {
        device: Device::CUDA,
        compute_type: gpu_compute,
        model_id: cfg.whisper_model_gpu.clone(),
        label: "ct2:cuda-int8-fp16",
    };
    let pick_cpu = || RuntimeChoice {
        device: Device::CPU,
        compute_type: ComputeType::INT8,
        model_id: cfg.whisper_model_cpu.clone(),
        label: "ct2:cpu-int8",
    };

    match cfg.backend_mode.to_ascii_lowercase().as_str() {
        "gpu" | "cuda" => {
            if !cuda_ok {
                return Err(anyhow!(
                    "backend_mode=\"gpu\" but no CUDA device is visible. \
                     Plug in / wake the dGPU, or switch to \"auto\" / \"cpu\"."
                ));
            }
            Ok(pick_gpu())
        }
        "cpu" => Ok(pick_cpu()),
        // "auto" or anything unrecognized → auto.
        _ => Ok(if cuda_ok { pick_gpu() } else { pick_cpu() }),
    }
}

impl WhisperBackend {
    /// Load a CT2 Whisper model from `model_dir` using the device +
    /// compute_type + label from a `RuntimeChoice`. Caller is
    /// responsible for matching `model_dir` to `choice.model_id`.
    pub async fn new(model_dir: PathBuf, choice: RuntimeChoice) -> Result<Self> {
        let RuntimeChoice {
            device,
            compute_type,
            label,
            ..
        } = choice;
        let threads = pick_thread_count(device);
        log::info!(
            "Whisper: loading model from {} (device={}, compute={}, threads={})",
            model_dir.display(),
            device,
            compute_type,
            threads
        );

        // Model load is heavy — runs on the blocking pool so we
        // don't stall the Tokio runtime.
        let model_dir_for_load = model_dir.clone();
        let whisper = tokio::task::spawn_blocking(move || -> Result<Whisper> {
            let cfg = CtConfig {
                device,
                compute_type,
                num_threads_per_replica: threads,
                ..Default::default()
            };
            Whisper::new(&model_dir_for_load, cfg).with_context(|| {
                format!(
                    "loading CT2 Whisper model from {}",
                    model_dir_for_load.display()
                )
            })
        })
        .await
        .map_err(|e| anyhow!("model load task: {e}"))??;

        log::info!(
            "WhisperBackend ready: label={} sampling_rate={}",
            label,
            whisper.sampling_rate()
        );

        Ok(Self {
            inner: Arc::new(WhisperInner {
                whisper,
                inflight: StdMutex::new(()),
            }),
            backend_label: label,
        })
    }
}

/// Pick CT2's `num_threads_per_replica`. On CPU we want physical
/// cores (HT siblings fight over the same SIMD pipeline; Intel
/// hybrid CPUs put threads on slow E-cores). On GPU the value
/// barely matters — CT2 mostly defers to cuBLAS — so we set it
/// low to keep the host side responsive.
fn pick_thread_count(device: Device) -> usize {
    match device {
        Device::CUDA => 1,
        _ => {
            let logical = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            // Halve to approximate physical-core count (handles HT),
            // then clamp to [2, 16]. The 16 ceiling matters because
            // (a) past 16 threads the matmul has parallelism cliffs
            // — synchronization overhead starts dominating speedup —
            // and (b) the user wants the rest of their machine to
            // stay responsive while transcription is in flight.
            (logical / 2).clamp(2, 16)
        }
    }
}

// (`pick_device` was replaced by `resolve_runtime` above — the
// new API also returns the resolved HF model id so stt.rs can
// download the right weights before constructing the backend.)

#[async_trait]
impl SttBackend for WhisperBackend {
    fn name(&self) -> &'static str {
        self.backend_label
    }

    async fn run(
        &self,
        app: AppHandle,
        cfg: AppConfig,
        audio: Arc<Mutex<Option<AudioCapture>>>,
        cancel: Arc<AtomicBool>,
    ) -> Result<()> {
        // 1) Buffer audio until the user releases the hotkey.
        let max_samples = (MAX_SECONDS * SAMPLE_RATE as f32) as usize;
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
                                "Whisper: audio exceeds {MAX_SECONDS}s cap; truncating"
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

        if buf.len() < SAMPLE_RATE as usize / 4 {
            log::info!(
                "Whisper: too little audio ({} samples), skipping",
                buf.len()
            );
            return Ok(());
        }

        let audio_seconds = buf.len() as f32 / SAMPLE_RATE as f32;
        log::info!("Whisper: transcribing {:.2}s of audio", audio_seconds);

        // 2) Run CT2 inference on the blocking pool — generate() is
        //    a synchronous CPU/GPU call that would otherwise block
        //    the Tokio worker.
        let inner = self.inner.clone();
        let started = std::time::Instant::now();
        let backend_label = self.backend_label;
        let text = tokio::task::spawn_blocking(move || -> Result<String> {
            let _g = inner
                .inflight
                .lock()
                .map_err(|_| anyhow!("Whisper inflight mutex poisoned"))?;

            // Dictation-tuned WhisperOptions. Defaults are calibrated
            // for transcribing podcasts; we want low-latency single
            // utterances. See the constants section above for the
            // reasoning behind each value.
            let opts = WhisperOptions {
                // Greedy decode: ~3x faster than beam_size=5, with
                // <0.3% WER cost on short clips. Standard tradeoff
                // for dictation.
                beam_size: 1,
                // Bumped to 1.2 (from 1.1) because 1.1 sometimes
                // wasn't enough to break Whisper's infinite-loop
                // hallucinations on dictation tail-silence — the
                // decoder would emit "the the the the…" all the way
                // to max_length, taking 100+ seconds on CPU.
                repetition_penalty: 1.2,
                // Hard-block repeated 3-grams. Belt-and-suspenders
                // against the same loop-hallucination pattern.
                no_repeat_ngram_size: 3,
                // Cap output at 224 tokens (default 448) so even if
                // the decoder DOES get stuck in a loop, the worst-
                // case wall time is bounded. 224 tokens is ~40 s of
                // speech at typical dictation speed — well above any
                // single utterance the user is likely to record.
                max_length: 224,
                ..WhisperOptions::default()
            };

            // Force English; saves the language-detect pass and
            // matches the comparison project's behavior. If we ever
            // expose multilingual support, plumb a config option
            // through here.
            let segments = inner
                .whisper
                .generate(&buf, Some("en"), false, &opts)
                .map_err(|e| anyhow!("ct2 whisper generate: {e}"))?;
            log::debug!(
                "{backend_label}: {} segment(s) returned",
                segments.len()
            );
            Ok(segments.join(" "))
        })
        .await
        .map_err(|e| anyhow!("inference task: {e}"))??;
        let elapsed = started.elapsed();
        log::info!(
            "Whisper: transcription took {:?} (RTFx {:.1}x, backend={})",
            elapsed,
            audio_seconds / elapsed.as_secs_f32().max(0.001),
            self.backend_label
        );

        let mut text = text.trim().to_string();
        if text.is_empty() {
            log::info!("Whisper: empty transcript");
            return Ok(());
        }

        // Hallucination filter. Whisper has well-known training
        // residue patterns it emits on silent / music / noisy
        // input; faster-whisper strips these by default and so do
        // we. Note: we DON'T paste these into the user's editor —
        // pasting "[BLANK_AUDIO]" mid-sentence is one of the most
        // annoying STT bugs a user can hit.
        if is_hallucination(&text) {
            log::info!("Whisper: dropping hallucinated output: {text:?}");
            return Ok(());
        }

        if cfg.trailing_space && !text.ends_with(' ') {
            text.push(' ');
        }

        crate::analytics::record_from_backend(&app, text.trim(), elapsed, "whisper").await;

        let _ = inject::paste_text(&app, &text);
        Ok(())
    }
}

/// True if the transcript matches one of Whisper's known
/// hallucination patterns. Conservative on purpose — only filters
/// patterns that are essentially never legitimate dictation.
fn is_hallucination(text: &str) -> bool {
    let trimmed = text.trim().to_lowercase();
    if trimmed.is_empty() {
        return true;
    }
    if HALLUCINATION_EXACT.iter().any(|p| trimmed == *p) {
        return true;
    }
    // For substring matches we want the WHOLE transcript to be
    // basically just the hallucination — otherwise we'd drop a
    // legitimate "thanks for watching the demo" mid-sentence.
    // Heuristic: if a pattern is present AND the transcript is
    // <60 chars, treat as hallucination. Real dictation longer
    // than 60 chars containing "thanks for watching" is almost
    // certainly intentional.
    if trimmed.len() < 60
        && HALLUCINATION_CONTAINS.iter().any(|p| trimmed.contains(p))
    {
        return true;
    }
    false
}
