use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::{AppHandle, Manager};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppConfig {
    /// Hotkey that opens/closes the settings window.
    pub toggle_settings_hotkey: String,
    /// Hotkey that toggles dictation on/off.
    pub dictate_hotkey: String,
    /// Reserved for future modes. Whisper currently always behaves
    /// as oneshot (paste full transcript at end of utterance) — true
    /// streaming partials would require a chunked-decode pass that's
    /// out of scope for v1.
    pub output_mode: String,
    /// Microphone device name; empty = system default.
    pub mic_device: String,
    /// Insert a trailing space after each utterance.
    pub trailing_space: bool,
    /// Has the app shown its first-run UI yet?
    pub first_run: bool,

    /// Override path for the Whisper model directory. None = app data dir.
    #[serde(default)]
    pub whisper_model_dir: Option<String>,

    /// Compute device selection. One of:
    ///   - "auto" (default): use CUDA if a device is visible,
    ///     otherwise CPU.
    ///   - "gpu": force CUDA. Errors out if no device is found,
    ///     so the user knows their GPU isn't being used.
    ///   - "cpu": force CPU even if a GPU is present. Useful for
    ///     A/B testing latency or saving battery on Optimus
    ///     laptops.
    #[serde(default = "default_backend_mode")]
    pub backend_mode: String,

    /// Engine to use when the resolved device is CPU. One of:
    ///   - "moonshine" (default): Moonshine v2 medium via
    ///     sherpa-onnx. ~6.65% WER, RTFx 25-40× on AVX2 CPUs.
    ///     The Better CPU choice for an English-only dictation
    ///     workflow.
    ///   - "whisper": faster-whisper-* via ct2rs. Slower on CPU
    ///     (~3-5× RTFx for small.en) but offers wider model
    ///     selection + multilingual.
    /// GPU mode always uses Whisper regardless of this setting —
    /// Moonshine doesn't have a CUDA path.
    #[serde(default = "default_cpu_engine")]
    pub cpu_engine: String,

    /// Legacy single-model field. Kept on the struct for forward
    /// compatibility with old config.json files; new code reads
    /// `whisper_model_cpu` / `whisper_model_gpu` instead.
    #[serde(default = "default_whisper_model_cpu")]
    pub whisper_model_id: String,

    /// HF model ID used when the active device is CPU. Defaults to
    /// `Systran/faster-whisper-base.en` — same model the reference
    /// Python project uses for its CPU fallback. Smaller encoder
    /// (6 layers) than small.en (12) → ~2× faster on CPU.
    #[serde(default = "default_whisper_model_cpu")]
    pub whisper_model_cpu: String,

    /// HF model ID used when the active device is CUDA. Defaults to
    /// `Systran/faster-whisper-small.en` — better WER than base.en
    /// (~7% vs ~10%), and on GPU the speed difference is negligible
    /// because the encoder runs in <100 ms either way.
    #[serde(default = "default_whisper_model_gpu")]
    pub whisper_model_gpu: String,

    /// Free-form text prompt that gets prepended to every
    /// transcription, biasing the decoder toward terms it sees
    /// here. Whisper's standard "initial_prompt" mechanism (used by
    /// the reference faster-whisper project for the technical-vocab
    /// preamble). Currently a NO-OP — the high-level ct2rs Whisper
    /// API doesn't expose initial_prompt. Plumbing TBD; field is
    /// here so the config schema doesn't churn when we wire it up.
    #[serde(default)]
    pub whisper_initial_prompt: String,

    /// User-supplied vocabulary list for biasing transcription
    /// toward jargon / proper nouns / project-specific terms. Will
    /// be appended to whatever's in `whisper_initial_prompt` when
    /// the prompt is built. Same NO-OP caveat as above for now.
    #[serde(default)]
    pub custom_dictionary: Vec<String>,

    /// Launch the app at user login (registry / LaunchAgent / .desktop).
    /// The actual OS-level state is managed by `tauri-plugin-autostart`;
    /// this flag mirrors it so the settings UI can render correctly.
    #[serde(default)]
    pub auto_start: bool,

    /// Master enable for the dictation hotkey. When false the hotkey
    /// fires but `start_dictation` no-ops — useful for letting the
    /// user temporarily silence the app without changing their bind.
    #[serde(default = "default_true")]
    pub stt_enabled: bool,

    /// Optional hotkey that toggles `stt_enabled`. Empty string = no
    /// toggle hotkey (user has to flip the switch in settings).
    #[serde(default)]
    pub stt_toggle_hotkey: String,
}

fn default_true() -> bool {
    true
}

fn default_backend_mode() -> String {
    "auto".into()
}

fn default_cpu_engine() -> String {
    "moonshine".into()
}

fn default_whisper_model_cpu() -> String {
    // base.en: 150 MB, ~10% WER, the comparison Python project's
    // default. Sweet spot for CPU dictation latency.
    "Systran/faster-whisper-base.en".into()
}

fn default_whisper_model_gpu() -> String {
    // medium.en: 770 MB, ~5.5% WER. On a modern GPU the latency
    // difference vs small.en (~7% WER) is sub-100 ms, well under
    // user-perceptible — we trade VRAM/bandwidth for accuracy.
    // On NVIDIA Tensor Cores the int8_float16 path makes this
    // basically free vs the smaller models.
    "Systran/faster-whisper-medium.en".into()
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            toggle_settings_hotkey: "Ctrl+Alt+V".into(),
            dictate_hotkey: "Ctrl+Alt+D".into(),
            output_mode: "oneshot".into(),
            mic_device: String::new(),
            trailing_space: true,
            first_run: true,
            whisper_model_dir: None,
            backend_mode: default_backend_mode(),
            cpu_engine: default_cpu_engine(),
            whisper_model_id: default_whisper_model_cpu(),
            whisper_model_cpu: default_whisper_model_cpu(),
            whisper_model_gpu: default_whisper_model_gpu(),
            whisper_initial_prompt: String::new(),
            custom_dictionary: Vec::new(),
            auto_start: false,
            stt_enabled: true,
            stt_toggle_hotkey: String::new(),
        }
    }
}

fn config_path(app: &AppHandle) -> Result<PathBuf> {
    let dir = app.path().app_config_dir()?;
    std::fs::create_dir_all(&dir).ok();
    Ok(dir.join("config.json"))
}

pub fn load(app: &AppHandle) -> Result<AppConfig> {
    let p = config_path(app)?;
    if !p.exists() {
        return Ok(AppConfig::default());
    }
    let raw = std::fs::read_to_string(p)?;
    Ok(serde_json::from_str(&raw)?)
}

pub fn save(app: &AppHandle, cfg: &AppConfig) -> Result<()> {
    let p = config_path(app)?;
    let mut to_save = cfg.clone();
    to_save.first_run = false;
    let json = serde_json::to_string_pretty(&to_save)?;
    std::fs::write(p, json)?;
    Ok(())
}
