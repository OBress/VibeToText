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
// `Manager` brings the `path()` method into scope on AppHandle —
// without it, `app.path().app_data_dir()` won't resolve.
use tauri::{AppHandle, Emitter, Manager};
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

// ---------------------------------------------------------------------------
// Moonshine model fetcher.
//
// Moonshine (useful-sensors) ships in two sizes — `tiny` (27M params,
// 12.7% WER) and `base` (61M params, 6.65% WER). We pick `base`
// because the WER advantage is huge and the model is still ~250 MB
// — small enough that downloading it on first dictation is fine.
//
// k2-fsa's sherpa-onnx ships prebuilt INT8 ONNX exports of v1 base
// at this archive URL on their `asr-models` GitHub release. The v1
// model is split across four ONNX files (preprocess, encode,
// cached_decode, uncached_decode) — the same layout the official
// Moonshine reference inference loop expects. v2 uses a single
// `merged_decoder` file but isn't shipped in `medium-en` form by
// sherpa-onnx as of 1.13.
// ---------------------------------------------------------------------------

/// Pinned URL for the Moonshine v1 base-en INT8 release archive. If
/// the upstream k2-fsa team renames or moves the artifact, update
/// this string.
const MOONSHINE_ARCHIVE_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-base-en-int8.tar.bz2";

/// Files that must exist inside the extracted directory for
/// sherpa-onnx's `OfflineMoonshineModelConfig` to load. v1 needs
/// the four ONNX components plus the tokens table.
const MOONSHINE_REQUIRED_FILES: &[&str] = &[
    "preprocess.onnx",
    "encode.int8.onnx",
    "cached_decode.int8.onnx",
    "uncached_decode.int8.onnx",
    "tokens.txt",
];

/// Resolved directory where we keep the extracted Moonshine model
/// files. Lives under the OS app-data dir so an uninstall can clean
/// it up cleanly.
pub fn moonshine_dir(app: &AppHandle) -> Result<PathBuf> {
    let base = app.path().app_data_dir().context("app_data_dir")?;
    Ok(base.join("models").join("moonshine-base-en-int8"))
}

/// True if every required Moonshine file is on disk.
pub fn moonshine_already_downloaded(app: &AppHandle) -> bool {
    let Ok(dir) = moonshine_dir(app) else {
        return false;
    };
    MOONSHINE_REQUIRED_FILES.iter().all(|f| dir.join(f).exists())
}

/// Process-wide download lock for Moonshine. Same reasoning as
/// `model_download_lock` for Whisper — concurrent callers would
/// otherwise race on the same `.part` archive.
fn moonshine_download_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Returns the path to the Moonshine model directory, downloading
/// + extracting from the upstream sherpa-onnx release archive
/// first if needed.
pub async fn ensure_moonshine_ready(app: &AppHandle) -> Result<PathBuf> {
    let _guard = moonshine_download_lock().lock().await;
    let dir = moonshine_dir(app)?;
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create moonshine dir {}", dir.display()))?;

    // Fast path: every required file already on disk.
    if MOONSHINE_REQUIRED_FILES.iter().all(|f| dir.join(f).exists()) {
        log::debug!("Moonshine model already on disk at {}", dir.display());
        return Ok(dir);
    }

    log::info!(
        "downloading Moonshine base-en INT8 archive from {}",
        MOONSHINE_ARCHIVE_URL
    );
    let archive_label = "sherpa-onnx-moonshine-base-en-int8.tar.bz2";
    let _ = app.emit(
        "model-download",
        serde_json::json!({
            "file": archive_label,
            "phase": "starting",
            "bytes": 0u64,
            "total": serde_json::Value::Null,
        }),
    );

    // Download → temp file in dir.
    let archive_path = dir.join("download.tar.bz2");
    download_to_file(app, MOONSHINE_ARCHIVE_URL, &archive_path).await?;

    // Extract .tar.bz2 → flatten any wrapping directory.
    let _ = app.emit(
        "model-download",
        serde_json::json!({
            "file": archive_label,
            "phase": "extracting",
            "bytes": 0u64,
            "total": serde_json::Value::Null,
        }),
    );
    extract_tar_bz2(&archive_path, &dir)
        .with_context(|| format!("extracting {}", archive_path.display()))?;
    let _ = std::fs::remove_file(&archive_path);

    // Verify required files landed on disk.
    let missing: Vec<_> = MOONSHINE_REQUIRED_FILES
        .iter()
        .filter(|f| !dir.join(f).exists())
        .collect();
    if !missing.is_empty() {
        return Err(anyhow!(
            "Moonshine extraction finished but required files are missing: {:?}",
            missing
        ));
    }

    log::info!("Moonshine ready at {}", dir.display());
    let _ = app.emit(
        "model-download",
        serde_json::json!({
            "file": archive_label,
            "phase": "done",
        }),
    );
    Ok(dir)
}

/// Stream-download `url` to `dst`, writing through a `.part` temp
/// file and atomically renaming on success. Skips if the dst
/// already exists.
async fn download_to_file(app: &AppHandle, url: &str, dst: &Path) -> Result<()> {
    use tokio::io::AsyncWriteExt;

    if dst.exists() {
        return Ok(());
    }
    let tmp = dst.with_extension("part");
    let resp = reqwest::Client::builder()
        .user_agent("VibeToText/0.1 (+model-download)")
        .build()?
        .get(url)
        .send()
        .await
        .with_context(|| format!("GET {url}"))?;
    if !resp.status().is_success() {
        return Err(anyhow!("download {url} failed: HTTP {}", resp.status()));
    }
    let total = resp.content_length();
    let mut stream = resp.bytes_stream();
    let mut file = tokio::fs::File::create(&tmp)
        .await
        .with_context(|| format!("create {}", tmp.display()))?;
    let mut written: u64 = 0;
    let mut last_emit: u64 = 0;
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk).await?;
        written += chunk.len() as u64;
        if written - last_emit > 1024 * 1024 {
            last_emit = written;
            let _ = app.emit(
                "model-download",
                serde_json::json!({
                    "file": dst.file_name().and_then(|s| s.to_str()).unwrap_or(""),
                    "phase": "progress",
                    "bytes": written,
                    "total": total,
                }),
            );
        }
    }
    file.flush().await?;
    drop(file);
    std::fs::rename(&tmp, dst)
        .with_context(|| format!("rename {} -> {}", tmp.display(), dst.display()))?;
    Ok(())
}

/// Extract `archive` (a `.tar.bz2` file) into `dst`. If the archive
/// has a single top-level directory wrapping all files, strip it
/// — sherpa-onnx ships archives with one extra level of nesting.
fn extract_tar_bz2(archive: &Path, dst: &Path) -> Result<()> {
    use bzip2::read::BzDecoder;
    use std::io::{BufReader, Read};
    use tar::Archive;

    // First pass: probe for a shared top-level directory we'll strip.
    // We can't iterate the same Archive twice, so we open the file
    // separately for the probe and again for the real extraction.
    // Each `Archive` owns its own decoder + file handle.
    let mut shared_root: Option<String> = None;
    {
        let probe_file = std::fs::File::open(archive)
            .with_context(|| format!("open {}", archive.display()))?;
        let probe_bz = BzDecoder::new(BufReader::new(probe_file));
        let mut probe = Archive::new(probe_bz);
        let mut detect_init = false;
        for entry in probe.entries()? {
            let entry = entry?;
            let path = entry.path()?;
            let first = path
                .components()
                .next()
                .and_then(|c| c.as_os_str().to_str())
                .unwrap_or("");
            if first.is_empty() {
                continue;
            }
            if !detect_init {
                shared_root = Some(first.to_string());
                detect_init = true;
            } else if shared_root.as_deref() != Some(first) {
                shared_root = None;
                break;
            }
        }
    } // probe dropped here, file handle closed

    // Second pass: actually extract, stripping shared_root if any.
    let file = std::fs::File::open(archive)
        .with_context(|| format!("open {}", archive.display()))?;
    let bz = BzDecoder::new(BufReader::new(file));
    let mut tar = Archive::new(bz);
    for entry in tar.entries()? {
        let mut entry = entry?;
        let raw_path = entry.path()?.to_path_buf();
        let stripped = if let Some(root) = &shared_root {
            match raw_path.strip_prefix(root) {
                Ok(p) => p.to_path_buf(),
                Err(_) => raw_path.clone(),
            }
        } else {
            raw_path.clone()
        };
        if stripped.as_os_str().is_empty() {
            continue;
        }
        // Reject path traversal — never trust archive paths.
        if stripped.components().any(|c| c.as_os_str() == "..") {
            log::warn!("skipping tar entry with .. traversal: {}", raw_path.display());
            continue;
        }
        let out = dst.join(&stripped);
        if entry.header().entry_type().is_dir() {
            std::fs::create_dir_all(&out).ok();
            continue;
        }
        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let mut f = std::fs::File::create(&out)
            .with_context(|| format!("create {}", out.display()))?;
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = entry.read(&mut buf)?;
            if n == 0 {
                break;
            }
            std::io::Write::write_all(&mut f, &buf[..n])?;
        }
    }
    Ok(())
}
