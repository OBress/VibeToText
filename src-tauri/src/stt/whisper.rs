// Whisper backend powered by `ct2rs::sys::Whisper` directly (the
// LOW-level ct2rs API). We dropped down from ct2rs's high-level
// `Whisper` wrapper because:
//
//   1. The high-level wrapper hides the prompt-token vector that
//      gets passed to CTranslate2's C++ inference, so there's no
//      way to inject `initial_prompt` for vocabulary biasing — the
//      mechanism the reference Python project uses to teach
//      Whisper project-specific jargon and proper nouns.
//
//   2. We want to run our own VAD before the mel spectrogram, and
//      the high-level wrapper doesn't let us hook into the audio
//      pipeline either.
//
// Trading complexity for control: we now own the mel spectrogram
// computation (via the same `mel_spec` crate ct2rs uses), the
// HuggingFace tokenizer (via `tokenizers`), and the prompt
// construction. The `sys::Whisper` does the actual encoder +
// decoder math, which is what we wanted from CTranslate2 anyway.
//
// Audio pipeline per dictation:
//   raw f32 samples → VAD trim → mel spectrogram → encoder →
//   decoder (with prompt = optional initial-prompt tokens +
//   standard `<|sot|> <|en|> <|transcribe|> <|notimestamps|>`)
//   → token ids → tokenizer.decode → text → hallucination
//   filter → paste.

use crate::audio::AudioCapture;
use crate::config::AppConfig;
use crate::inject;
use crate::stt::SttBackend;
use crate::vad;
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use ct2rs::sys::{
    get_device_count, ComputeType, Config as CtConfig, Device, StorageView, Whisper as SysWhisper,
    WhisperOptions,
};
use mel_spec::mel::{log_mel_spectrogram, mel, norm_mel};
use mel_spec::stft::Spectrogram;
use ndarray::{s, stack, Array2, Axis};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex as StdMutex,
};
use std::time::Duration;
use tauri::AppHandle;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

const SAMPLE_RATE: u32 = 16_000;
/// CTranslate2's Whisper handles audio longer than 30 s by
/// splitting it into chunks, but we cap the user's hold at 60 s
/// to keep the in-memory buffer bounded.
const MAX_SECONDS: f32 = 60.0;

/// Hallucination patterns Whisper emits on quiet/non-speech input.
/// Same list the previous implementation had — kept identical.
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
const HALLUCINATION_CONTAINS: &[&str] = &[
    "thanks for watching",
    "thank you for watching",
    "subtitles by",
    "subscribe to",
    "translated by",
];

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
/// (c) which HF model id to download.
pub fn resolve_runtime(cfg: &AppConfig) -> Result<RuntimeChoice> {
    let cuda_count = get_device_count(Device::CUDA);
    let cuda_ok = cuda_count > 0;
    log::info!("CT2 device probe: CUDA device_count={cuda_count}");

    // INT8_FLOAT16 on GPU: int8 weights, fp16 matmul. Faster than
    // plain FLOAT16 on small Whisper models because matmul is
    // memory-bandwidth-bound. <0.5% WER cost. Tensor Cores on
    // Ampere+ NVIDIA accept this combo natively.
    let gpu_compute = ComputeType::INT8_FLOAT16;

    let pick_gpu = || RuntimeChoice {
        device: Device::CUDA,
        compute_type: gpu_compute,
        model_id: cfg.whisper_model_gpu.clone(),
        label: "whisper:cuda-int8-fp16",
    };
    let pick_cpu = || RuntimeChoice {
        device: Device::CPU,
        compute_type: ComputeType::INT8,
        model_id: cfg.whisper_model_cpu.clone(),
        label: "whisper:cpu-int8",
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
        _ => Ok(if cuda_ok { pick_gpu() } else { pick_cpu() }),
    }
}

// ---------------------------------------------------------------------------
// PreprocessorConfig — same schema ct2rs uses, deserialized from
// the synthesized `preprocessor_config.json` we wrote in models.rs.
// We keep our own copy of this struct + parser because ct2rs's is
// private to its `whisper` module.
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct PreprocessorConfig {
    feature_size: usize,
    hop_length: usize,
    n_fft: usize,
    n_samples: usize,
    nb_max_frames: usize,
    sampling_rate: usize,
    /// Pre-computed mel filter bank. Shape: (feature_size, n_fft/2+1).
    /// We compute this from the simple integer params above on
    /// load — `mel_spec::mel::mel(...)` is the same function ct2rs
    /// calls internally.
    mel_filters: Array2<f64>,
}

impl PreprocessorConfig {
    fn read(path: &Path) -> Result<Self> {
        #[derive(Deserialize)]
        struct Aux {
            feature_size: usize,
            hop_length: usize,
            n_fft: usize,
            n_samples: usize,
            nb_max_frames: usize,
            sampling_rate: usize,
        }
        let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let aux: Aux = serde_json::from_reader(BufReader::new(file))
            .with_context(|| format!("parse {}", path.display()))?;

        // mel_spec::mel::mel computes the standard Whisper mel
        // filter bank. Args: sample_rate, n_fft, n_mels, fmin (None
        // = 0), fmax (None = sample_rate/2), htk (false = use slaney
        // formula), norm (true = slaney normalization). These
        // defaults match what ct2rs uses internally.
        let mel_filters = mel(
            aux.sampling_rate as f64,
            aux.n_fft,
            aux.feature_size,
            None,
            None,
            false,
            true,
        );

        Ok(Self {
            feature_size: aux.feature_size,
            hop_length: aux.hop_length,
            n_fft: aux.n_fft,
            n_samples: aux.n_samples,
            nb_max_frames: aux.nb_max_frames,
            sampling_rate: aux.sampling_rate,
            mel_filters,
        })
    }
}

// ---------------------------------------------------------------------------
// WhisperBackend — owns the loaded sys::Whisper, tokenizer, and
// preprocessor config. Built once per app session and cached on
// AppState.
// ---------------------------------------------------------------------------

pub struct WhisperBackend {
    inner: Arc<WhisperInner>,
    backend_label: &'static str,
}

struct WhisperInner {
    whisper: SysWhisper,
    tokenizer: Tokenizer,
    config: PreprocessorConfig,
    /// CTranslate2's per-decode state isn't safe to share across
    /// concurrent generate() calls. In practice we only run one
    /// dictation at a time, but the mutex is belt-and-suspenders.
    inflight: StdMutex<()>,
}

// SAFETY: SysWhisper is `Send + Sync` (verified upstream).
// Tokenizer is `Send + Sync`. PreprocessorConfig holds plain data.
// StdMutex provides interior mutability for serialization.
unsafe impl Send for WhisperInner {}
unsafe impl Sync for WhisperInner {}

impl WhisperBackend {
    /// Load a CT2 Whisper model from `model_dir`. Heavy work —
    /// reads model.bin (~150 MB+), constructs the encoder graph,
    /// allocates GPU buffers if device=CUDA. Caller should run
    /// this on a blocking pool.
    pub async fn new(model_dir: PathBuf, choice: RuntimeChoice) -> Result<Self> {
        let RuntimeChoice {
            device,
            compute_type,
            label,
            model_id: _,
        } = choice;
        let threads = pick_thread_count(device);
        log::info!(
            "Whisper: loading model from {} (device={}, compute={}, threads={})",
            model_dir.display(),
            device,
            compute_type,
            threads
        );

        let model_dir_clone = model_dir.clone();
        let inner = tokio::task::spawn_blocking(move || -> Result<WhisperInner> {
            let cfg = CtConfig {
                device,
                compute_type,
                num_threads_per_replica: threads,
                ..Default::default()
            };
            let whisper = SysWhisper::new(&model_dir_clone, cfg).with_context(|| {
                format!("loading sys::Whisper from {}", model_dir_clone.display())
            })?;

            let tokenizer = Tokenizer::from_file(model_dir_clone.join("tokenizer.json"))
                .map_err(|e| anyhow!("loading tokenizer.json: {e}"))?;

            let config =
                PreprocessorConfig::read(&model_dir_clone.join("preprocessor_config.json"))?;

            Ok(WhisperInner {
                whisper,
                tokenizer,
                config,
                inflight: StdMutex::new(()),
            })
        })
        .await
        .map_err(|e| anyhow!("model load task: {e}"))??;

        log::info!(
            "WhisperBackend ready: label={} sampling_rate={}",
            label,
            inner.config.sampling_rate
        );

        Ok(Self {
            inner: Arc::new(inner),
            backend_label: label,
        })
    }
}

/// Pick CT2's `num_threads_per_replica`. CPU: physical-core
/// approximation, capped at 16. GPU: 1 (cuBLAS does the work,
/// host threads barely matter).
fn pick_thread_count(device: Device) -> usize {
    match device {
        Device::CUDA => 1,
        _ => {
            let logical = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            (logical / 2).clamp(2, 16)
        }
    }
}

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
                            log::warn!("Whisper: audio exceeds {MAX_SECONDS}s cap; truncating");
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

        // Drain frames cpal delivers AFTER we saw cancel. cpal has
        // 10-50 ms of internal latency between a physical mic
        // sample and our callback firing, so the user's final word
        // arrives in the channel only after they've already
        // released the hotkey. The drain helper polls for ~220 ms
        // (paired with the 250 ms cpal keep-alive in
        // DictationSession::stop) and pulls each frame as it lands.
        let drained = crate::stt::drain_remaining_audio(&audio, &mut buf, max_samples).await;
        if drained > 0 {
            log::debug!(
                "Whisper: drained {} samples ({:.0} ms) after cancel",
                drained,
                drained as f32 * 1000.0 / SAMPLE_RATE as f32
            );
        }

        if buf.len() < SAMPLE_RATE as usize / 4 {
            log::info!(
                "Whisper: too little audio ({} samples), skipping",
                buf.len()
            );
            return Ok(());
        }

        // 2) VAD: trim leading/trailing silence. Cuts encoder time
        //    and prevents tail-silence hallucinations. Returns a
        //    borrowed slice — no copy.
        let trimmed = vad::trim_silence(&buf);
        let trimmed_samples: Vec<f32> = trimmed.to_vec();
        let audio_seconds = trimmed_samples.len() as f32 / SAMPLE_RATE as f32;
        log::info!("Whisper: transcribing {:.2}s of audio", audio_seconds);

        // 3) Build the optional initial_prompt text from
        //    user-configured fields. Whisper conditions the decoder
        //    on these tokens, biasing it toward project-specific
        //    vocabulary the model wouldn't otherwise know how to
        //    spell.
        let initial_prompt = build_initial_prompt(&cfg);

        // 4) Run inference on the blocking pool.
        let inner = self.inner.clone();
        let backend_label = self.backend_label;
        let started = std::time::Instant::now();
        let text = tokio::task::spawn_blocking(move || -> Result<String> {
            let _g = inner
                .inflight
                .lock()
                .map_err(|_| anyhow!("Whisper inflight mutex poisoned"))?;
            transcribe_one(
                &inner,
                &trimmed_samples,
                initial_prompt.as_deref(),
                backend_label,
            )
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

/// Synchronous inference path: mel spectrogram → encoder →
/// decoder → text. Runs entirely on the calling thread (caller
/// should already be inside `tokio::task::spawn_blocking`).
fn transcribe_one(
    inner: &WhisperInner,
    samples: &[f32],
    initial_prompt: Option<&str>,
    backend_label: &str,
) -> Result<String> {
    let cfg = &inner.config;

    // === Mel spectrogram ===
    // Same algorithm ct2rs uses internally. For each `n_samples`
    // chunk (30 s at 16 kHz = 480 000 samples), compute STFT then
    // log-mel-spectrogram, normalize, and pack into an Array2 of
    // shape (feature_size, nb_max_frames). Pad with zeros if the
    // chunk is shorter than `n_samples`.
    //
    // BEFORE chunking, we append ~600 ms of trailing silence to the
    // raw samples for two distinct reasons that BOTH bite Whisper
    // (Moonshine's sliding-window encoder doesn't have either, which
    // is why Moonshine never drops the user's last word):
    //
    //   1. The STFT only emits an FFT frame when its rolling window
    //      fills (n_fft samples = 25 ms). Without padding, samples in
    //      the final partial window get stuck in STFT's internal
    //      buffer and never produce a mel column. ~10-25 ms loss
    //      per utterance.
    //
    //   2. More importantly: Whisper's decoder is trained on
    //      30 s mel windows where a CLEAR run of trailing silence
    //      signals "speech ended, emit EOT." With no trailing
    //      silence, the decoder sees the user's last word right at
    //      the end of the meaningful mel frames and the rest as
    //      zero-padding (silence). The transition is so abrupt that
    //      the decoder can fire EOT BEFORE finishing the last word,
    //      truncating it. Padding with real silence smooths the
    //      transition and matches the training distribution.
    //
    // 1500 ms gives the decoder a generous "speech is over" signal.
    // Empirically, 600 ms wasn't enough — the user reported Whisper
    // still cut off the final word. The longer silence run more
    // closely matches what Whisper saw at training time (clean
    // 30 s clips with mostly-silence tails) and gives the EOT-vs-
    // continue decision a clearer margin.
    const TRAILING_PAD_MS: usize = 1500;
    let pad_samples = (cfg.sampling_rate as usize) * TRAILING_PAD_MS / 1000;
    let mut padded: Vec<f32> = Vec::with_capacity(samples.len() + pad_samples);
    padded.extend_from_slice(samples);
    padded.extend(std::iter::repeat(0.0_f32).take(pad_samples));
    let samples: &[f32] = &padded;

    let mut stft = Spectrogram::new(cfg.n_fft, cfg.hop_length);
    let mut mel_per_chunk: Vec<Array2<f32>> = Vec::new();

    for chunk in samples.chunks(cfg.n_samples) {
        let mut mel_chunk = Array2::<f32>::zeros((cfg.feature_size, cfg.nb_max_frames));
        for (i, frame) in chunk.chunks(cfg.hop_length).enumerate() {
            if let Some(fft_frame) = stft.add(frame) {
                let mel =
                    norm_mel(&log_mel_spectrogram(&fft_frame, &cfg.mel_filters)).mapv(|v| v as f32);
                if i < cfg.nb_max_frames {
                    mel_chunk.slice_mut(s![.., i]).assign(&mel.slice(s![.., 0]));
                }
            }
        }
        mel_per_chunk.push(mel_chunk);
    }

    if mel_per_chunk.is_empty() {
        return Ok(String::new());
    }

    // Stack chunk-mels into a 3D array (n_chunks, feature_size,
    // nb_max_frames). Ensure standard memory layout — CTranslate2's
    // C++ side expects contiguous row-major.
    let views: Vec<_> = mel_per_chunk.iter().map(|a| a.view()).collect();
    let mut mel_3d = stack(Axis(0), &views).context("stacking mel spectrogram chunks")?;
    if !mel_3d.is_standard_layout() {
        mel_3d = mel_3d.as_standard_layout().into_owned();
    }
    let shape = mel_3d.shape().to_vec();

    // === Encoder input as StorageView ===
    let storage_view = StorageView::new(
        &shape,
        mel_3d
            .as_slice_mut()
            .ok_or_else(|| anyhow!("mel array not contiguous"))?,
        Default::default(),
    )
    .map_err(|e| anyhow!("StorageView::new: {e}"))?;

    // === Build prompt ===
    // If we have an initial_prompt, tokenize it and prepend with
    // the `<|startofprev|>` marker. Whisper's decoder treats this
    // as "previous text" — it conditions on the tokens but doesn't
    // re-emit them in the output.
    //
    // Standard Whisper prompt suffix: <|sot|> <|en|> <|transcribe|> <|notimestamps|>
    let prompt_strings: Vec<String> =
        build_prompt_tokens(&inner.tokenizer, initial_prompt, backend_label)?;

    // === Decoder options ===
    // beam_size=5 matches OpenAI Whisper's reference default. With
    // greedy decoding (beam_size=1) the user reported Whisper
    // dropping the final word(s) of long sentences while Moonshine
    // — which uses a fundamentally different sliding-window
    // architecture — never had the issue. Greedy decoding commits
    // to the highest-probability next token at every step, and
    // when the EOT token edges out the continuation token by even
    // a tiny margin Whisper just stops mid-sentence. Beam search
    // explores multiple paths and the path that DOES finish the
    // sentence ends up with a higher total log-prob, so the model
    // commits to it. ~3-4× slower than greedy on the same
    // hardware but on GPU + INT8_FLOAT16 we're still well under
    // 1 s for a 10 s utterance.
    //
    // max_length=448 matches Whisper's default decoder context so
    // a long dictation (~30 s / ~300 words) doesn't get cut off
    // mid-sentence by the OUTPUT cap. We previously had this at
    // 224 to bound hallucination loops, but the real fix for those
    // is repetition_penalty=1.2 + no_repeat_ngram_size=3, not the
    // artificial cap.
    let opts = WhisperOptions {
        beam_size: 5,
        repetition_penalty: 1.2,
        no_repeat_ngram_size: 3,
        max_length: 448,
        ..WhisperOptions::default()
    };

    // === Run ===
    // sys::Whisper::generate takes one prompt per chunk in the mel
    // batch. We use the same prompt for all chunks (clipped audio
    // never reaches multi-chunk territory at our 60 s cap, but the
    // API requires the vec).
    let prompts: Vec<Vec<&str>> = (0..mel_per_chunk.len())
        .map(|_| prompt_strings.iter().map(|s| s.as_str()).collect())
        .collect();
    let results = inner
        .whisper
        .generate(&storage_view, &prompts, &opts)
        .map_err(|e| anyhow!("sys::Whisper::generate: {e}"))?;
    log::debug!(
        "{backend_label}: {} result chunk(s) returned",
        results.len()
    );

    // === Decode tokens to text ===
    // Each result has `sequences_ids[0]` = generated token IDs
    // (just the new tokens, NOT the prompt). Concatenate all chunks.
    let mut full_text = String::new();
    for res in results {
        let Some(ids) = res.sequences_ids.into_iter().next() else {
            continue;
        };
        let ids_u32: Vec<u32> = ids.into_iter().map(|x| x as u32).collect();
        let text = inner
            .tokenizer
            .decode(&ids_u32, true)
            .map_err(|e| anyhow!("tokenizer decode: {e}"))?;
        if !full_text.is_empty() {
            full_text.push(' ');
        }
        full_text.push_str(text.trim());
    }
    Ok(full_text)
}

/// Construct the prompt token-string vector that gets passed to
/// `sys::Whisper::generate`. Format:
///
///   [<|startofprev|>, ...prev_tokens, <|sot|>, <|en|>, <|transcribe|>, <|notimestamps|>]
///
/// where `prev_tokens` are HF-tokenized BPE pieces of
/// `initial_prompt`, omitted entirely if no initial_prompt was
/// configured.
fn build_prompt_tokens(
    tokenizer: &Tokenizer,
    initial_prompt: Option<&str>,
    backend_label: &str,
) -> Result<Vec<String>> {
    let mut prompt: Vec<String> = Vec::new();

    if let Some(text) = initial_prompt {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            // Encode without adding the model's own special tokens
            // (we add `<|startofprev|>` etc. ourselves below).
            let encoded = tokenizer
                .encode(trimmed, false)
                .map_err(|e| anyhow!("tokenize initial_prompt: {e}"))?;
            let pieces: Vec<String> = encoded.get_tokens().iter().map(|s| s.to_string()).collect();

            // Whisper's prompt is capped at 224 tokens (n_text_ctx
            // / 2). Truncate if the user dumped a wall of text —
            // keep the LAST N tokens since recent context tends to
            // bias more strongly than earlier context.
            const MAX_PROMPT_TOKENS: usize = 200;
            let pieces = if pieces.len() > MAX_PROMPT_TOKENS {
                let start = pieces.len() - MAX_PROMPT_TOKENS;
                log::debug!(
                    "{backend_label}: initial_prompt truncated from {} to {} tokens",
                    pieces.len(),
                    MAX_PROMPT_TOKENS
                );
                pieces.into_iter().skip(start).collect::<Vec<_>>()
            } else {
                pieces
            };

            log::debug!(
                "{backend_label}: prepending {} prompt tokens for vocab biasing",
                pieces.len()
            );
            prompt.push("<|startofprev|>".into());
            prompt.extend(pieces);
        }
    }

    prompt.push("<|startoftranscript|>".into());
    prompt.push("<|en|>".into());
    prompt.push("<|transcribe|>".into());
    prompt.push("<|notimestamps|>".into());
    Ok(prompt)
}

/// Combine `whisper_initial_prompt` (free-form text) with
/// `custom_dictionary` (one word per line) into a single prompt
/// string. Returns None if both are empty — saves us from
/// tokenizing nothing.
fn build_initial_prompt(cfg: &AppConfig) -> Option<String> {
    let raw = cfg.whisper_initial_prompt.trim();
    let dict = cfg
        .custom_dictionary
        .iter()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>();
    if raw.is_empty() && dict.is_empty() {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    if !raw.is_empty() {
        parts.push(raw.to_string());
    }
    if !dict.is_empty() {
        // Comma-separate the vocab list and embed it in a sentence
        // that nudges Whisper to treat them as proper-noun
        // vocabulary. Same shape the reference Python project uses.
        parts.push(format!("Important vocabulary: {}.", dict.join(", ")));
    }
    Some(parts.join(" "))
}

/// True if the transcript is one of Whisper's known training-set
/// hallucination patterns we should silently drop.
fn is_hallucination(text: &str) -> bool {
    let trimmed = text.trim().to_lowercase();
    if trimmed.is_empty() {
        return true;
    }
    if HALLUCINATION_EXACT.iter().any(|p| trimmed == *p) {
        return true;
    }
    if trimmed.len() < 60 && HALLUCINATION_CONTAINS.iter().any(|p| trimmed.contains(p)) {
        return true;
    }
    false
}
