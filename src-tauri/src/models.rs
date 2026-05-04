// Whisper model fetching, CT2-format edition.
//
// We download the multi-file CTranslate2 representation of Whisper
// from HuggingFace via `hf-hub`. The reference model is
// `Systran/faster-whisper-base.en` — same one the Python
// `faster-whisper` project picks by default.
//
// **Storage strategy**: don't mirror to app_data_dir. The HF cache
// at `~/.cache/huggingface/hub/<repo>/snapshots/<commit>/` is
// already content-addressable and resumable; copying files out of
// it just creates Windows symlink/junction edge cases (we burned
// hours on those before settling here). Instead, ct2rs loads the
// model directly from the HF snapshot directory.
//
// We DO write one file into the snapshot dir post-download:
// `preprocessor_config.json`. The Systran/faster-whisper-* repos
// don't ship it (CTranslate2's C++ has Whisper's mel filter banks
// baked in), but ct2rs's high-level Whisper wrapper requires it
// for its Rust-side mel spectrogram computation. We synthesize one
// with the standard Whisper preprocessor values for the variant.

use anyhow::{anyhow, Context, Result};
use hf_hub::api::tokio::{Api, ApiError};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tauri::{AppHandle, Emitter};
use tokio::sync::Mutex;

/// Files that must be in the snapshot directory for ct2rs to load
/// the model. Everything else (.gitattributes, README, vocabulary
/// variant files) is bonus.
const REQUIRED_FILES: &[&str] = &[
    "model.bin",
    "config.json",
    "tokenizer.json",
];

/// Process-wide download lock: serializes ensure_whisper_ready so
/// concurrent callers (start_dictation race + the settings UI's
/// "Download model" button) don't both try to populate the cache
/// at the same time.
fn model_download_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Skip files we don't need. Saves bandwidth and works around
/// hf-hub edge cases (.gitattributes occasionally returned a
/// malformed URL on 0.3.x).
fn should_download_file(name: &str) -> bool {
    if name.starts_with('.') {
        return false;
    }
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".md") || lower == "license" || lower == "notice" {
        return false;
    }
    if lower.ends_with(".safetensors")
        || lower.ends_with(".pt")
        || lower.ends_with(".pth")
        || lower.ends_with(".onnx")
    {
        return false;
    }
    true
}

/// Resolve where hf-hub stores its cache. Honors `HF_HOME`,
/// otherwise the platform-standard `~/.cache/huggingface/hub`.
fn hf_cache_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("HF_HOME") {
        return Some(PathBuf::from(p).join("hub"));
    }
    let home = if cfg!(target_os = "windows") {
        std::env::var("USERPROFILE").ok()
    } else {
        std::env::var("HOME").ok()
    }?;
    Some(
        PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("hub"),
    )
}

/// Look for an already-downloaded snapshot of `model_id` in the HF
/// cache. Returns the snapshot directory path if every file in
/// `REQUIRED_FILES` is present (which means ct2rs can load it
/// without us hitting the network).
fn find_cached_snapshot(model_id: &str) -> Option<PathBuf> {
    let cache = hf_cache_dir()?;
    let dir_name = format!("models--{}", model_id.replace('/', "--"));
    let snapshots = cache.join(&dir_name).join("snapshots");
    if !snapshots.is_dir() {
        return None;
    }
    // hf-hub creates one snapshot subdirectory per commit hash.
    // Pick the first one that contains the required files. (In
    // practice there's only ever one, since hf-hub doesn't keep
    // multiple commits unless the user specifically pins them.)
    for entry in std::fs::read_dir(&snapshots).ok()? {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if REQUIRED_FILES.iter().all(|f| path.join(f).exists()) {
            return Some(path);
        }
    }
    None
}

/// Returns the directory path containing a complete CT2 Whisper
/// model, downloading from HuggingFace first if needed. The
/// `model_id` is passed in explicitly because the active model
/// depends on which device we resolved to (CPU vs GPU may want
/// different sizes — see `resolve_runtime` in stt/whisper.rs).
pub async fn ensure_whisper_ready(
    app: &AppHandle,
    cfg: &crate::config::AppConfig,
    model_id: &str,
) -> Result<PathBuf> {
    let _guard = model_download_lock().lock().await;

    // Override path: user pre-downloaded somewhere. Trust them.
    // Still synthesize the preprocessor config in case it's missing.
    if let Some(p) = cfg.whisper_model_dir.as_ref().filter(|s| !s.is_empty()) {
        let dir = PathBuf::from(p);
        if dir.is_dir() {
            log::info!("using explicit whisper_model_dir at {}", dir.display());
            synthesize_preprocessor_config(&dir, model_id)?;
            return Ok(dir);
        }
        log::warn!(
            "configured whisper_model_dir {} doesn't exist; falling back to HF download",
            dir.display()
        );
    }

    // Cached snapshot fast-path: ct2rs can load directly from
    // ~/.cache/huggingface/hub/.../snapshots/<commit>/ — no need to
    // re-download or mirror.
    if let Some(snap) = find_cached_snapshot(model_id) {
        log::info!(
            "Whisper model {model_id} already cached at {}",
            snap.display()
        );
        synthesize_preprocessor_config(&snap, model_id)?;
        return Ok(snap);
    }

    // Slow path: download via hf-hub.
    log::info!("downloading CT2 Whisper model {model_id} from HuggingFace");
    let api = Api::new()
        .map_err(api_err)
        .context("hf_hub Api init")?;
    let repo = api.model(model_id.to_string());
    let info = repo.info().await.map_err(api_err).with_context(|| {
        format!("HF repo info for {model_id}")
    })?;

    let siblings: Vec<_> = info
        .siblings
        .into_iter()
        .filter(|s| should_download_file(&s.rfilename))
        .collect();
    let total_files = siblings.len() as u32;
    let mut completed: u32 = 0;
    let mut snapshot_dir: Option<PathBuf> = None;

    for sib in siblings {
        let _ = app.emit(
            "model-download",
            serde_json::json!({
                "file": sib.rfilename,
                "phase": "starting",
                "bytes": 0u64,
                "total": serde_json::Value::Null,
            }),
        );
        log::info!("fetching {} from HF...", sib.rfilename);

        let cached = repo
            .get(&sib.rfilename)
            .await
            .map_err(api_err)
            .with_context(|| format!("downloading {}", sib.rfilename))?;

        if snapshot_dir.is_none() {
            snapshot_dir = cached.parent().map(PathBuf::from);
        }
        completed += 1;

        let bytes = std::fs::metadata(&cached)
            .map(|m| m.len())
            .unwrap_or(0);
        let _ = app.emit(
            "model-download",
            serde_json::json!({
                "file": sib.rfilename,
                "phase": "done",
                "bytes": bytes,
                "total": bytes,
                "files_done": completed,
                "files_total": total_files,
            }),
        );
        log::info!(
            "fetched {} ({} of {})",
            sib.rfilename,
            completed,
            total_files
        );
    }

    let dst = snapshot_dir
        .ok_or_else(|| anyhow!("hf-hub returned no files for {model_id}"))?;

    // Verify ct2rs's required files are present.
    let missing: Vec<_> = REQUIRED_FILES
        .iter()
        .filter(|f| !dst.join(f).exists())
        .collect();
    if !missing.is_empty() {
        return Err(anyhow!(
            "download finished but required files are missing in {}: {:?}",
            dst.display(),
            missing
        ));
    }

    // Synthesize preprocessor_config.json (Systran repos don't ship
    // it; ct2rs requires it).
    synthesize_preprocessor_config(&dst, model_id)?;

    Ok(dst)
}

/// Write a `preprocessor_config.json` next to the model weights if
/// it isn't already there. Values are the standard Whisper
/// preprocessor defaults: 16 kHz mono, 30 s chunks, 400-bin FFT,
/// 160-sample hop length. Mel-filter count varies by model
/// generation — large-v3 family uses 128 mel bins; everything else
/// uses 80.
fn synthesize_preprocessor_config(model_dir: &Path, model_id: &str) -> Result<()> {
    let pcfg = model_dir.join("preprocessor_config.json");
    if pcfg.exists() {
        log::debug!(
            "preprocessor_config.json already present at {}",
            pcfg.display()
        );
        return Ok(());
    }

    let lower = model_id.to_ascii_lowercase();
    let feature_size: usize = if lower.contains("large-v3") || lower.contains("turbo") {
        128
    } else {
        80
    };

    // Schema matches what ct2rs::whisper::PreprocessorConfig::read
    // deserializes. mel_filters omitted → ct2rs computes them from
    // (sampling_rate, n_fft, feature_size).
    let json = serde_json::json!({
        "chunk_length": 30,
        "feature_extractor_type": "WhisperFeatureExtractor",
        "feature_size": feature_size,
        "hop_length": 160,
        "n_fft": 400,
        "n_samples": 480_000,
        "nb_max_frames": 3000,
        "padding_side": "right",
        "padding_value": 0.0,
        "processor_class": "WhisperProcessor",
        "return_attention_mask": false,
        "sampling_rate": 16_000,
    });
    let json_str = serde_json::to_string_pretty(&json)?;
    std::fs::write(&pcfg, json_str)
        .with_context(|| format!("write {}", pcfg.display()))?;
    log::info!(
        "synthesized preprocessor_config.json (feature_size={feature_size}) for {model_id} at {}",
        pcfg.display()
    );
    Ok(())
}

/// True when at least one of the models the current `backend_mode`
/// might pick is on disk. Cheap: no network, just stat a few file
/// paths. Used by `warm_whisper` at app startup to decide whether
/// to preload — we don't auto-download in the background, so this
/// returns false only when EVERY candidate model is missing.
pub fn whisper_already_downloaded(
    _app: &AppHandle,
    cfg: &crate::config::AppConfig,
) -> bool {
    if let Some(p) = cfg.whisper_model_dir.as_ref().filter(|s| !s.is_empty()) {
        let dir = PathBuf::from(p);
        if REQUIRED_FILES.iter().all(|f| dir.join(f).exists()) {
            return true;
        }
    }
    // Which model(s) are candidates depends on the user's backend_mode:
    //   gpu     → only the GPU model
    //   cpu     → only the CPU model
    //   auto/_  → either could be picked at runtime, so any one
    //             being on disk is enough to skip the warm-up
    //             download prompt.
    let candidates: Vec<&str> = match cfg.backend_mode.to_ascii_lowercase().as_str() {
        "gpu" | "cuda" => vec![&cfg.whisper_model_gpu],
        "cpu" => vec![&cfg.whisper_model_cpu],
        _ => vec![&cfg.whisper_model_gpu, &cfg.whisper_model_cpu],
    };
    candidates
        .iter()
        .any(|id| find_cached_snapshot(id).is_some())
}

/// Translate `hf_hub::api::tokio::ApiError` into `anyhow::Error`
/// so `?` works at the call sites.
fn api_err(e: ApiError) -> anyhow::Error {
    anyhow!("hf-hub error: {e}")
}
