const { invoke } = window.__TAURI__.core;
const { listen } = window.__TAURI__.event;
const { getCurrentWindow } = window.__TAURI__.window;

// ============================ window controls ============================
const appWindow = getCurrentWindow();
document.getElementById("win_min")?.addEventListener("click", () => appWindow.minimize());
document.getElementById("win_close")?.addEventListener("click", () => appWindow.close()); // hides via prevent-close handler in Rust

// ============================ tabs ============================
function activateTab(name) {
  for (const t of document.querySelectorAll(".tab")) {
    t.classList.toggle("active", t.dataset.tab === name);
  }
  document.getElementById("view-dashboard").classList.toggle("hidden", name !== "dashboard");
  document.getElementById("view-settings").classList.toggle("hidden", name !== "settings");
  if (name === "dashboard") refreshAnalytics();
}
for (const t of document.querySelectorAll(".tab")) {
  t.addEventListener("click", () => activateTab(t.dataset.tab));
}

// ============================ settings ============================
const fields = [
  "toggle_settings_hotkey",
  "dictate_hotkey",
  "stt_toggle_hotkey",
  "mic_device",
];

async function loadConfig() {
  const cfg = await invoke("get_config");
  for (const k of fields) {
    const el = document.getElementById(k);
    if (el) el.value = cfg[toCamel(k)] ?? "";
  }
  document.getElementById("trailing_space").checked = !!cfg.trailingSpace;
  // STT master toggle. cfg.sttEnabled defaults to true if missing on
  // older configs that didn't have the field.
  const sttEl = document.getElementById("stt_enabled");
  if (sttEl) sttEl.checked = cfg.sttEnabled !== false;
  // Backend mode (auto / gpu / cpu). Default to "auto" if missing.
  const mode = cfg.backendMode || "auto";
  for (const r of document.querySelectorAll('input[name="backend_mode"]')) {
    r.checked = r.value === mode;
  }
  // CPU engine (moonshine / whisper). Defaults to moonshine because
  // it's much faster on CPU at comparable WER. GPU mode always uses
  // Whisper regardless of this setting.
  const cpuEngine = (cfg.cpuEngine || "moonshine").toLowerCase();
  const cpuEngineEl = document.getElementById("cpu_engine");
  if (cpuEngineEl) cpuEngineEl.value = cpuEngine;
  // Per-device Whisper model dropdowns. Defaults match the Rust
  // side: base.en for CPU (speed), small.en for GPU (quality).
  const modelCpu = cfg.whisperModelCpu || "Systran/faster-whisper-base.en";
  const modelGpu = cfg.whisperModelGpu || "Systran/faster-whisper-small.en";
  const cpuSelect = document.getElementById("whisper_model_cpu");
  const gpuSelect = document.getElementById("whisper_model_gpu");
  if (cpuSelect) cpuSelect.value = modelCpu;
  if (gpuSelect) gpuSelect.value = modelGpu;
  // Wire dropdown visibility to the backend_mode + cpu_engine combo.
  applyModelPickerVisibility(mode, cpuEngine);
  // Custom-vocab fields (Whisper-only — Moonshine ignores them).
  const promptEl = document.getElementById("whisper_initial_prompt");
  if (promptEl) promptEl.value = cfg.whisperInitialPrompt || "";
  const dictEl = document.getElementById("custom_dictionary");
  if (dictEl) dictEl.value = (cfg.customDictionary || []).join("\n");
  // Autostart: read OS-level state as the source of truth (the user
  // might have disabled it from Task Manager → Startup since we last
  // saved). Falls back to the saved cfg flag if the IPC errors.
  const autoStartEl = document.getElementById("auto_start");
  if (autoStartEl) {
    try {
      autoStartEl.checked = await invoke("is_auto_start_enabled");
    } catch (_e) {
      autoStartEl.checked = !!cfg.autoStart;
    }
  }
  // Surface any pre-existing hotkey collision (e.g. config edited by
  // hand).
  if (typeof updateCollisionWarning === "function") updateCollisionWarning();
}

/// Show only the dropdowns that matter for the current backend_mode +
/// cpu_engine combo.
///   - GPU dropdown:    visible if mode != "cpu" (auto / gpu)
///   - CPU dropdown:    visible if mode != "gpu" AND cpu_engine == "whisper"
///   - cpu_engine:      visible if mode != "gpu" (auto / cpu)
function applyModelPickerVisibility(mode, cpuEngine) {
  const gpuWrap = document.getElementById("model_picker_gpu_wrap");
  const cpuWrap = document.getElementById("model_picker_cpu_wrap");
  const engineWrap = document.getElementById("cpu_engine_wrap");
  const showGpu = mode !== "cpu";
  const showEngine = mode !== "gpu";
  const showCpuModel =
    mode !== "gpu" && (cpuEngine || "moonshine").toLowerCase() === "whisper";
  if (gpuWrap) gpuWrap.style.display = showGpu ? "" : "none";
  if (cpuWrap) cpuWrap.style.display = showCpuModel ? "" : "none";
  if (engineWrap) engineWrap.style.display = showEngine ? "" : "none";
}

function toCamel(s) {
  return s.replace(/_([a-z])/g, (_, c) => c.toUpperCase());
}

const HOTKEY_DEFAULTS = {
  toggle_settings_hotkey: "Ctrl+Alt+V",
  dictate_hotkey: "Ctrl+Alt+D",
};

// =================== auto-save ===================
// Every change persists immediately (radios / checkboxes / hotkey commits)
// or after a short debounce (text inputs). The header status pill shows
// a brief animated green checkmark on success.

const CHECK_SVG = `<svg viewBox="0 0 16 16" width="12" height="12" xmlns="http://www.w3.org/2000/svg"><path d="M3 8.4 L6.7 12 L13 4.5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

let _saveTimer = null;
let _saveInFlight = false;
let _saveDirty = false;
let _flashTimer = null;

function scheduleSave(opts = {}) {
  if (opts.immediate) {
    clearTimeout(_saveTimer);
    _saveTimer = null;
    return doSave();
  }
  clearTimeout(_saveTimer);
  _saveTimer = setTimeout(doSave, 350);
}

async function doSave() {
  if (_saveInFlight) {
    // Mark dirty — the running save will loop and pick up the latest
    // values when it finishes. Avoids both lost edits AND retry storms.
    _saveDirty = true;
    return;
  }
  _saveInFlight = true;
  _saveDirty = false;
  try {
    do {
      _saveDirty = false;
      await runSaveOnce();
    } while (_saveDirty);
  } finally {
    _saveInFlight = false;
  }
}

async function runSaveOnce() {
  // Defensive: never send an empty hotkey — that silently disables the key.
  const toggle = val("toggle_settings_hotkey") || HOTKEY_DEFAULTS.toggle_settings_hotkey;
  const dictate = val("dictate_hotkey") || HOTKEY_DEFAULTS.dictate_hotkey;
  // Custom-vocab textarea → array of trimmed non-empty lines.
  const dictEl = document.getElementById("custom_dictionary");
  const customDictionary = dictEl
    ? dictEl.value
        .split("\n")
        .map((s) => s.trim())
        .filter((s) => s.length > 0)
    : [];

  const payload = {
    toggleSettingsHotkey: toggle,
    dictateHotkey: dictate,
    outputMode: "oneshot",
    micDevice: val("mic_device"),
    trailingSpace: document.getElementById("trailing_space").checked,
    firstRun: false,
    whisperModelDir: null,
    backendMode:
      document.querySelector('input[name="backend_mode"]:checked')?.value || "auto",
    // Legacy single-model field — kept synced with the CPU pick so
    // older code paths that still reference it stay sane. Real
    // model selection happens via per-device fields below.
    whisperModelId:
      document.getElementById("whisper_model_cpu")?.value
        || "Systran/faster-whisper-base.en",
    whisperModelCpu:
      document.getElementById("whisper_model_cpu")?.value
        || "Systran/faster-whisper-base.en",
    whisperModelGpu:
      document.getElementById("whisper_model_gpu")?.value
        || "Systran/faster-whisper-small.en",
    cpuEngine:
      document.getElementById("cpu_engine")?.value
        || "moonshine",
    whisperInitialPrompt:
      document.getElementById("whisper_initial_prompt")?.value || "",
    customDictionary,
    autoStart: !!document.getElementById("auto_start")?.checked,
    sttEnabled: !!document.getElementById("stt_enabled")?.checked,
    sttToggleHotkey: val("stt_toggle_hotkey"),
  };
  try {
    await invoke("save_config", { newConfig: payload });
    document.getElementById("toggle_settings_hotkey").value = toggle;
    document.getElementById("dictate_hotkey").value = dictate;
    flashSaved();
  } catch (e) {
    flashError(e);
  }
  // Note: _saveInFlight is owned by doSave's outer try/finally.
}

function flashSaved() {
  const status = document.getElementById("status");
  if (!status) return;
  // Remove and re-add classes to retrigger the animation even on rapid
  // saves (the keyframes fire once per class-application cycle).
  status.classList.remove("status-saved", "status-error");
  void status.offsetWidth; // force reflow so the next add re-runs the keyframes
  status.classList.add("status-saved");
  status.innerHTML = `<span class="check">${CHECK_SVG}</span><span>saved</span>`;
  clearTimeout(_flashTimer);
  _flashTimer = setTimeout(() => {
    status.classList.remove("status-saved");
    status.textContent = "idle";
  }, 1400);
}

function flashError(e) {
  const status = document.getElementById("status");
  if (status) {
    status.classList.remove("status-saved");
    status.classList.add("status-error");
    status.textContent = "save failed";
  }
  console.error("save failed:", e);
  // Surface real Rust errors (e.g. same-combo collision) to the user.
  if (typeof e === "string" && e.length < 200) alert(e);
}

function wireAutoSave() {
  // Backend-mode radios — save AND toggle which model dropdown(s)
  // are visible in the UI based on the new mode + current cpu_engine.
  document
    .querySelectorAll('input[name="backend_mode"]')
    .forEach((el) =>
      el.addEventListener("change", () => {
        const engine = document.getElementById("cpu_engine")?.value || "moonshine";
        applyModelPickerVisibility(el.value, engine);
        scheduleSave({ immediate: true });
      })
    );
  // CPU engine select — flips between Moonshine (no model picker) and
  // Whisper (model picker visible). Triggers save immediately.
  const engineEl = document.getElementById("cpu_engine");
  if (engineEl) {
    engineEl.addEventListener("change", () => {
      const mode =
        document.querySelector('input[name="backend_mode"]:checked')?.value || "auto";
      applyModelPickerVisibility(mode, engineEl.value);
      scheduleSave({ immediate: true });
    });
  }
  // Model dropdowns — save the moment the user picks a new model
  // for either device.
  ["whisper_model_cpu", "whisper_model_gpu"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("change", () => scheduleSave({ immediate: true }));
  });
  // Checkboxes.
  const ts = document.getElementById("trailing_space");
  if (ts) ts.addEventListener("change", () => scheduleSave({ immediate: true }));
  const auto = document.getElementById("auto_start");
  if (auto) auto.addEventListener("change", () => scheduleSave({ immediate: true }));
  const stt = document.getElementById("stt_enabled");
  if (stt) stt.addEventListener("change", () => scheduleSave({ immediate: true }));
  // Free-text fields — debounce so typing isn't one save per keystroke.
  ["mic_device"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("input", () => scheduleSave());
    // Commit immediately when focus leaves so a debounced save in flight
    // doesn't get stranded.
    el.addEventListener("blur", () => scheduleSave({ immediate: true }));
  });
  // Multi-line text fields (initial prompt, custom dictionary).
  // Same debounce-on-input + commit-on-blur pattern.
  ["whisper_initial_prompt", "custom_dictionary"].forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener("input", () => scheduleSave());
    el.addEventListener("blur", () => scheduleSave({ immediate: true }));
  });
  // Hotkey fields are handled inside `setupHotkeyCapture` — they call
  // scheduleSave({immediate:true}) right after a successful capture.
}

// ============================ hotkey capture ============================
// Click a hotkey input → it goes into "capturing" state. The next
// non-modifier-only keydown is captured as the new combo, formatted as
// e.g. "Ctrl+Alt+D", and committed. Esc cancels; Backspace reverts to
// the default. Inputs are readonly so users can't accidentally type
// garbage or wipe them blank.

const MODIFIER_KEYS = new Set(["Control", "Shift", "Alt", "Meta"]);

// Map e.code (physical key, layout-independent, ignores Shift) to the
// names the Rust parser in `lib.rs::code_from_str` understands. Using
// e.code instead of e.key means Ctrl+Shift+1 maps to "Ctrl+Shift+1" not
// "Ctrl+Shift+!", and AZERTY/QWERTY layouts produce the same string.
const CODE_MAP = {
  Space: "Space",
  Enter: "Enter",
  Escape: "Escape",
  Tab: "Tab",
  Backspace: "Backspace",
  Delete: "Delete",
  Insert: "Insert",
  Home: "Home",
  End: "End",
  PageUp: "PageUp",
  PageDown: "PageDown",
  ArrowUp: "Up",
  ArrowDown: "Down",
  ArrowLeft: "Left",
  ArrowRight: "Right",
  Backquote: "`",
  Minus: "-",
  Equal: "=",
  BracketLeft: "[",
  BracketRight: "]",
  Backslash: "\\",
  Semicolon: ";",
  Quote: "'",
  Comma: ",",
  Period: ".",
  Slash: "/",
};

// Convert a keydown event into a hotkey combo string.
// Treats modifier keys (Ctrl/Shift/Alt/Cmd) as VALID trigger keys for
// modifier-only combos like "Ctrl+Shift". The matching modifier flag is
// excluded from the modifier list when the trigger IS a modifier so we
// don't emit "Ctrl+Ctrl".
function keyEventToCombo(e) {
  if (e.key === "Dead") return null;

  let ctrl = e.ctrlKey, alt = e.altKey, shift = e.shiftKey, meta = e.metaKey;
  const k = e.key;
  // If the key BEING PRESSED is a modifier, treat it as the trigger and
  // remove it from the held-modifiers set.
  if (k === "Control") ctrl = false;
  else if (k === "Shift") shift = false;
  else if (k === "Alt") alt = false;
  else if (k === "Meta") meta = false;

  const parts = [];
  if (ctrl)  parts.push("Ctrl");
  if (alt)   parts.push("Alt");
  if (shift) parts.push("Shift");
  if (meta)  parts.push("Cmd");

  let key = null;
  if (k === "Control") key = "Ctrl";
  else if (k === "Shift") key = "Shift";
  else if (k === "Alt") key = "Alt";
  else if (k === "Meta") key = "Cmd";
  else {
    const code = e.code || "";
    if (/^Key[A-Z]$/.test(code)) key = code.slice(3);
    else if (/^Digit[0-9]$/.test(code)) key = code.slice(5);
    else if (/^Numpad[0-9]$/.test(code)) key = code.slice(6);
    else if (/^F\d{1,2}$/.test(code)) key = code;
    else if (CODE_MAP[code]) key = CODE_MAP[code];
    else if (e.key && e.key.length === 1) key = e.key.toUpperCase();
  }

  if (!key) return null;
  parts.push(key);
  return parts.join("+");
}

// Canonical sorted-key form for collision detection. Mirrors
// `canonical_combo_str` on the Rust side so "Ctrl+Shift" and "Shift+Ctrl"
// hash to the same value.
function canonicalCombo(combo) {
  if (!combo) return "";
  return combo
    .split("+")
    .map((p) => p.trim().toLowerCase())
    .map((p) => {
      if (p === "control") return "ctrl";
      if (p === "option") return "alt";
      if (p === "cmd" || p === "super" || p === "win") return "meta";
      return p;
    })
    .filter((p) => p)
    .sort()
    .join("+");
}

function setupHotkeyCapture(inputId, defaultCombo) {
  const el = document.getElementById(inputId);
  if (!el) return;
  let original = el.value;
  let paused = false;
  let ready = false;

  // We commit on key-RELEASE, not on first key-down. That lets users
  // build up a multi-key combo like Ctrl→Shift→D and have it captured
  // in one shot once they let go. The "best" combo seen during the
  // press is what we ultimately commit; nothing happens until all keys
  // are released. This also makes modifier-only combos (Ctrl+Shift)
  // work — when the user releases everything we read whatever was
  // captured at peak.
  let bestCombo = null;
  let pressActive = false;
  let cancelled = false;

  function resetCapture() {
    bestCombo = null;
    pressActive = false;
    cancelled = false;
  }

  el.addEventListener("focus", async () => {
    original = el.value;
    el.value = "";
    el.placeholder = "preparing…";
    el.classList.add("capturing");
    el.classList.remove("ready");
    ready = false;
    resetCapture();
    try {
      await invoke("pause_hotkeys");
      paused = true;
      // Win32 message-queue flush cushion — enough for any in-flight
      // WM_HOTKEY events from the registrations we just dropped to
      // drain before we start accepting key input.
      await new Promise((r) => setTimeout(r, 60));
      if (document.activeElement !== el) return;
      ready = true;
      el.classList.add("ready");
      el.placeholder = "press a combo… (Esc cancels, Backspace = default)";
    } catch (e) {
      console.error("pause_hotkeys failed:", e);
      el.placeholder = "couldn't pause hotkeys — check console";
    }
  });

  el.addEventListener("blur", async () => {
    el.classList.remove("capturing", "ready");
    ready = false;
    if (!el.value) el.value = original;
    el.placeholder = "click and press a combo";
    if (paused) {
      paused = false;
      try {
        await invoke("resume_hotkeys");
      } catch (e) {
        console.error("resume_hotkeys failed:", e);
      }
    }
    resetCapture();
    updateCollisionWarning();
  });

  el.addEventListener("keydown", (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!ready) return;
    if (e.key === "Escape") {
      cancelled = true;
      el.value = original;
      el.blur();
      return;
    }
    if (e.key === "Backspace" || e.key === "Delete") {
      cancelled = true;
      el.value = defaultCombo;
      el.blur();
      // Commit the default immediately.
      if (typeof scheduleSave === "function") scheduleSave({ immediate: true });
      return;
    }
    const combo = keyEventToCombo(e);
    if (combo) {
      bestCombo = combo;
      pressActive = true;
      // Live preview while the user is still holding keys.
      el.value = combo;
    }
  });

  el.addEventListener("keyup", (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!ready || cancelled) return;
    // Once every modifier flag is false AND the key being released isn't
    // anything we care about, we consider the press fully released.
    const stillHeld = e.ctrlKey || e.altKey || e.shiftKey || e.metaKey;
    if (!stillHeld && pressActive) {
      pressActive = false;
      if (bestCombo) {
        el.value = bestCombo;
        el.blur();
        if (typeof scheduleSave === "function") scheduleSave({ immediate: true });
      } else {
        el.value = original;
        el.blur();
      }
    }
  });
}

// Show / hide inline warnings under the hotkey inputs:
//   - exact-equal collision: HARD warning, can't be dismissed (the Rust
//     backstop will reject the save anyway)
//   - prefix collision: SOFT warning, the user can dismiss it via the X
//     button. The dismissal is per-pair-of-strings so changing either
//     hotkey resurfaces the warning if a NEW prefix collision arises.
function dismissedKeyFor(t, d) {
  return `dismissedHotkeyWarn:${t}::${d}`;
}
function isWarningDismissed(t, d) {
  try {
    return localStorage.getItem(dismissedKeyFor(t, d)) === "1";
  } catch (_) {
    return false;
  }
}
function setWarningDismissed(t, d) {
  try {
    localStorage.setItem(dismissedKeyFor(t, d), "1");
  } catch (_) {}
}

function updateCollisionWarning() {
  const tStr = val("toggle_settings_hotkey");
  const dStr = val("dictate_hotkey");
  const t = canonicalCombo(tStr);
  const d = canonicalCombo(dStr);

  const a = document.getElementById("hotkey_warn_settings");
  const b = document.getElementById("hotkey_warn_dictate");

  let msg = null;
  let dismissable = false;
  if (t && t === d) {
    msg = "⚠️ Same combo as the other hotkey — pick a different one.";
  } else if (t && d) {
    const ts = new Set(t.split("+"));
    const ds = new Set(d.split("+"));
    const tSubD = [...ts].every((k) => ds.has(k)) && ts.size < ds.size;
    const dSubT = [...ds].every((k) => ts.has(k)) && ds.size < ts.size;
    if (tSubD || dSubT) {
      const shorter = tSubD ? tStr : dStr;
      const longer = tSubD ? dStr : tStr;
      msg = `⚠️ "${shorter}" is a prefix of "${longer}". Pressing the longer one will briefly fire the shorter one too.`;
      dismissable = true;
    }
  }

  // Suppress dismissable warnings the user has already X-ed away for
  // this exact pair of combos.
  if (dismissable && isWarningDismissed(tStr, dStr)) {
    msg = null;
  }

  for (const el of [a, b]) {
    if (!el) continue;
    if (msg) {
      el.style.display = "";
      const txt = el.querySelector(".warn-text");
      if (txt) txt.textContent = msg;
      const btn = el.querySelector(".warn-close");
      if (btn) btn.style.display = dismissable ? "" : "none";
    } else {
      el.style.display = "none";
    }
  }
}

function dismissCurrentWarning() {
  setWarningDismissed(val("toggle_settings_hotkey"), val("dictate_hotkey"));
  updateCollisionWarning();
}

setupHotkeyCapture("toggle_settings_hotkey", HOTKEY_DEFAULTS.toggle_settings_hotkey);
setupHotkeyCapture("dictate_hotkey", HOTKEY_DEFAULTS.dictate_hotkey);
// stt_toggle_hotkey is optional — Backspace clears it (default = empty).
setupHotkeyCapture("stt_toggle_hotkey", "");

// ============================ shared tooltip ============================
// One floating tooltip element follows the cursor for any element with a
// `data-tip` attribute. Native HTML `title` attributes show the default
// Windows hover popup which (a) looks ugly, (b) disappears on screenshot,
// and (c) gets clipped by overflow:hidden parents. This replaces all of
// that with a styled, in-page element that lives at the document root.
(() => {
  const tip = document.createElement("div");
  tip.id = "tooltip";
  tip.className = "tooltip";
  document.body.appendChild(tip);

  let current = null;
  let raf = 0;
  let mouseX = 0;
  let mouseY = 0;

  function positionTip() {
    raf = 0;
    if (!current) return;
    const pad = 14;
    const rect = tip.getBoundingClientRect();
    let x = mouseX + pad;
    let y = mouseY - rect.height - 12;
    // Keep on-screen.
    if (x + rect.width + 8 > window.innerWidth) x = mouseX - rect.width - pad;
    if (y < 6) y = mouseY + 22;
    tip.style.transform = `translate(${x}px, ${y}px)`;
  }

  function show(target) {
    const text = target.dataset.tip;
    if (!text) return;
    current = target;
    tip.textContent = text;
    tip.classList.add("on");
    if (!raf) raf = requestAnimationFrame(positionTip);
  }

  function hide() {
    current = null;
    tip.classList.remove("on");
  }

  document.addEventListener("mousemove", (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    if (current && !raf) raf = requestAnimationFrame(positionTip);
  }, { passive: true });

  document.addEventListener("mouseover", (e) => {
    const t = e.target.closest("[data-tip]");
    if (t) show(t);
    else if (current && !current.contains(e.target)) hide();
  });

  document.addEventListener("mouseout", (e) => {
    const t = e.target.closest("[data-tip]");
    if (t && !t.contains(e.relatedTarget)) hide();
  });

  // Hide on scroll / blur / window leave so a stale tip never lingers.
  window.addEventListener("scroll", hide, { passive: true });
  window.addEventListener("blur", hide);
  document.addEventListener("mouseleave", hide);
})();

function val(id) { return document.getElementById(id).value.trim(); }

async function refreshWhisperStatus() {
  try {
    const present = await invoke("whisper_model_present");
    const el = document.getElementById("whisper_status");
    el.innerHTML = present
      ? `Status: <strong>downloaded ✓</strong>`
      : `Status: <strong>not downloaded yet</strong> — fetched automatically on first dictation`;
  } catch (e) { console.error(e); }
}

async function downloadWhisper() {
  const btn = document.getElementById("download_whisper");
  const wrap = document.getElementById("whisper_progress_wrap");
  btn.disabled = true;
  wrap.style.display = "block";
  try {
    // Single multi-file model download — the CT2 inference engine
    // is statically linked into the app binary, so there's no
    // separate runtime archive anymore.
    await invoke("download_whisper_model");
    document.getElementById("whisper_progress_label").textContent = "complete";
    await refreshWhisperStatus();
  } catch (e) {
    document.getElementById("whisper_progress_label").textContent = "error: " + e;
  } finally {
    btn.disabled = false;
  }
}

function fmtBytes(n) {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB"];
  let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return n.toFixed(1) + " " + u[i];
}

// ============================ analytics ============================

const palette = ["#ff7849", "#ffa86b", "#6aa9ff", "#b58bff", "#ff6ab8", "#6ee7b7", "#ffd166"];

function fmtDuration(ms) {
  const s = Math.floor(ms / 1000);
  if (s < 60) return { num: s, unit: "s" };
  const m = s / 60;
  if (m < 60) return { num: m.toFixed(1), unit: "min" };
  const h = m / 60;
  return { num: h.toFixed(1), unit: "h" };
}

function fmtClock(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  const now = new Date();
  const sameDay = d.toDateString() === now.toDateString();
  const yesterday = new Date(now);
  yesterday.setDate(yesterday.getDate() - 1);
  const isYesterday = d.toDateString() === yesterday.toDateString();
  const hh = d.getHours().toString().padStart(2, "0");
  const mm = d.getMinutes().toString().padStart(2, "0");
  if (sameDay) return `${hh}:${mm}`;
  if (isYesterday) return `y ${hh}:${mm}`;
  return `${d.getMonth() + 1}/${d.getDate()} ${hh}:${mm}`;
}

function animateCount(el, target, opts = {}) {
  const dur = opts.duration ?? 900;
  const decimals = opts.decimals ?? 0;
  const unit = opts.unit ?? "";
  const start = performance.now();
  const from = parseFloat(el.dataset.last || "0");
  el.dataset.last = String(target);
  const easeOut = (t) => 1 - Math.pow(1 - t, 3);
  function step(now) {
    const t = Math.min(1, (now - start) / dur);
    const v = from + (target - from) * easeOut(t);
    const num = decimals > 0 ? v.toFixed(decimals) : Math.round(v).toLocaleString();
    el.innerHTML = unit
      ? `${num}<span class="kpi-unit">${unit}</span>`
      : `${num}`;
    if (t < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

function renderHourly(hourly) {
  const max = Math.max(...hourly, 1);
  const wrap = document.getElementById("hour_heatmap");
  wrap.innerHTML = "";
  for (let h = 0; h < 24; h++) {
    const v = hourly[h];
    const ratio = v / max;
    const cell = document.createElement("div");
    cell.className = "hcell";
    cell.style.height = `${Math.max(6, ratio * 100)}%`;
    cell.style.setProperty("--alpha", String(0.12 + ratio * 0.85));
    cell.style.animationDelay = `${h * 18}ms`;
    cell.dataset.tip = `${formatHour(h)} · ${v} dictation${v === 1 ? "" : "s"}`;
    wrap.appendChild(cell);
  }
  const total = hourly.reduce((a, b) => a + b, 0);
  document.getElementById("hourly_meta").textContent =
    total ? `${total} dictation${total === 1 ? "" : "s"} · local time` : "";
}

function formatHour(h) {
  const suffix = h < 12 ? "a" : "p";
  const h12 = h % 12 === 0 ? 12 : h % 12;
  return `${h12}${suffix}`;
}

function renderWeek(days) {
  const max = Math.max(...days.map(d => d.utterances), 1);
  const wrap = document.getElementById("week_bars");
  wrap.innerHTML = "";
  for (let i = 0; i < days.length; i++) {
    const d = days[i];
    const bar = document.createElement("div");
    bar.className = "bar" + (d.utterances === 0 ? " empty" : "");
    const ratio = d.utterances / max;
    bar.style.height = `${Math.max(8, ratio * 100)}%`;
    bar.style.animationDelay = `${i * 60}ms`;
    bar.dataset.tip = `${d.date} · ${d.utterances} dictation${d.utterances === 1 ? "" : "s"} · ${d.words} words`;
    if (d.utterances > 0) {
      const lbl = document.createElement("div");
      lbl.className = "bar-label";
      lbl.textContent = d.utterances;
      bar.appendChild(lbl);
    }
    const day = document.createElement("div");
    day.className = "bar-day";
    day.textContent = d.label;
    bar.appendChild(day);
    wrap.appendChild(bar);
  }
  const total = days.reduce((s, d) => s + d.utterances, 0);
  document.getElementById("week_meta").textContent =
    total ? `${total} dictation${total === 1 ? "" : "s"}` : "";
}

function renderCloud(words) {
  const wrap = document.getElementById("word_cloud");
  wrap.innerHTML = "";
  if (!words.length) {
    wrap.innerHTML = `<div style="color:var(--muted-2);font-size:11px">no words yet — start dictating</div>`;
    return;
  }
  const max = words[0].count;
  const min = words[words.length - 1].count;
  words.forEach((w, i) => {
    const ratio = max === min ? 1 : (w.count - min) / (max - min);
    const size = 13 + ratio * 32;
    const tilt = (Math.random() - 0.5) * 6;
    const z = ratio * 80;
    const span = document.createElement("span");
    span.className = "cw";
    span.textContent = w.word;
    span.style.fontSize = `${size}px`;
    span.style.color = palette[i % palette.length];
    span.style.transform = `rotate(${tilt}deg) translateZ(${z}px)`;
    span.style.opacity = String(0.55 + ratio * 0.45);
    span.style.animationDelay = `${i * 35}ms`;
    span.dataset.tip = `${w.word} — ${w.count} time${w.count === 1 ? "" : "s"}`;
    wrap.appendChild(span);
  });
}

function renderRecent(items) {
  const wrap = document.getElementById("recent_list");
  wrap.innerHTML = "";
  if (!items.length) {
    wrap.innerHTML = `<div style="color:var(--muted-2);font-size:11px;padding:8px 4px">No dictations yet.</div>`;
    document.getElementById("recent_meta").textContent = "";
    return;
  }
  document.getElementById("recent_meta").textContent =
    `${items.length} most recent`;
  items.forEach((r, i) => {
    const row = document.createElement("div");
    row.className = "recent-item";
    row.style.animationDelay = `${i * 30}ms`;
    const time = document.createElement("div");
    time.className = "recent-time";
    time.textContent = fmtClock(r.timestamp);
    const back = document.createElement("div");
    back.className = `recent-backend ${r.backend}`;
    back.textContent = r.backend;
    const text = document.createElement("div");
    text.className = "recent-text";
    text.textContent = r.preview || "(empty)";
    text.dataset.tip = r.preview;
    const words = document.createElement("div");
    words.className = "recent-words";
    words.textContent = `${r.words}w`;
    row.appendChild(time);
    row.appendChild(back);
    row.appendChild(text);
    row.appendChild(words);
    wrap.appendChild(row);
  });
}

function renderInsights(insights) {
  const wrap = document.getElementById("insights_list");
  wrap.innerHTML = "";
  insights.forEach((s, i) => {
    const li = document.createElement("li");
    li.textContent = s;
    li.style.animationDelay = `${i * 60}ms`;
    wrap.appendChild(li);
  });
}

function renderHero(s) {
  const empty = s.totalUtterances === 0;
  document.getElementById("hero_empty").style.display = empty ? "" : "none";
  document.getElementById("hero_stats").style.display = empty ? "none" : "";
  if (empty) return;

  const dur = fmtDuration(s.totalDurationMs);
  animateCount(document.querySelector('[data-counter="duration"]'),
    parseFloat(dur.num), { unit: dur.unit, decimals: dur.unit === "s" ? 0 : 1 });
  document.getElementById("kpi_duration_sub").textContent =
    `across ${s.totalUtterances} session${s.totalUtterances === 1 ? "" : "s"}`;

  animateCount(document.querySelector('[data-counter="words"]'), s.totalWords);
  document.getElementById("kpi_words_sub").textContent =
    `${s.avgWordsPerMinute.toFixed(0)} WPM avg · ${s.avgWordsPerSession.toFixed(0)} per session`;

  animateCount(document.querySelector('[data-counter="sessions"]'), s.totalUtterances);
  document.getElementById("kpi_sessions_sub").textContent =
    `avg ${s.avgSessionSeconds.toFixed(1)}s · longest ${(s.longestSessionMs / 1000).toFixed(1)}s`;

  animateCount(document.querySelector('[data-counter="vocab"]'), s.uniqueWords);
  document.getElementById("kpi_vocab_sub").textContent =
    `${(s.vocabRichness * 100).toFixed(0)}% richness · avg ${s.avgWordLength.toFixed(1)} chars`;
}

function renderStreaks(s) {
  document.getElementById("streak_current").textContent = s.currentStreakDays;
  document.getElementById("streak_longest").textContent = s.longestStreakDays;
  document.getElementById("streak_active").textContent = s.daysActive;
}

let analyticsRefreshInFlight = false;
async function refreshAnalytics() {
  if (analyticsRefreshInFlight) return;
  analyticsRefreshInFlight = true;
  try {
    const s = await invoke("get_analytics");
    renderHero(s);
    renderHourly(s.hourly);
    renderWeek(s.last7Days);
    renderCloud(s.topWords);
    renderRecent(s.recent);
    renderInsights(s.insights);
    renderStreaks(s);
    document.getElementById("updated_at").textContent =
      `updated ${new Date().toLocaleTimeString()}`;
  } catch (e) {
    console.error("analytics fetch failed:", e);
  } finally {
    analyticsRefreshInFlight = false;
  }
}

async function resetAnalytics() {
  if (!confirm("Reset all analytics? This deletes the local analytics.json file.")) return;
  try {
    await invoke("reset_analytics");
    await refreshAnalytics();
  } catch (e) {
    alert("Reset failed: " + e);
  }
}

// ============================ wiring ============================

// Save button is gone — every settings change now auto-saves.
// CRITICAL: load the config BEFORE wiring up auto-save listeners.
// Otherwise a 'change' event fired while the inputs are still empty
// would persist blank values over the user's saved settings.
document.getElementById("download_whisper").addEventListener("click", downloadWhisper);
document.getElementById("reset_analytics").addEventListener("click", resetAnalytics);

// Wire the X buttons on the soft (prefix-collision) warnings so the
// user can dismiss them. Both warnings reflect the same per-pair state,
// so dismissing either hides both.
for (const id of ["hotkey_warn_settings", "hotkey_warn_dictate"]) {
  const el = document.getElementById(id);
  const btn = el && el.querySelector(".warn-close");
  if (btn) {
    btn.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      dismissCurrentWarning();
    });
  }
}

(async () => {
  try {
    await loadConfig();
  } catch (e) {
    console.error("loadConfig failed:", e);
  } finally {
    // Wire auto-save AFTER values are populated so the first save we
    // ever ship to disk reflects what the user sees, not blank inputs.
    wireAutoSave();
  }
})();

// Replay the window fade-in animation each time the settings window
// is shown again (Tauri's win.show() doesn't reload the page, so the
// CSS animation that runs on initial paint won't naturally re-fire).
listen("settings-shown", () => {
  const body = document.body;
  body.classList.remove("show-pop");
  // Force reflow so the next class-add re-runs the keyframes.
  void body.offsetWidth;
  body.classList.add("show-pop");
});

// Kept for forward-compatibility — currently we always select "whisper".
listen("backend-selected", (_e) => {});

// When the master STT switch is flipped via the toggle hotkey, the
// backend emits this so the settings UI checkbox stays in sync.
listen("stt-enabled-changed", (e) => {
  const el = document.getElementById("stt_enabled");
  if (el) el.checked = !!e.payload;
});

listen("dictation-state", (e) => {
  const status = document.getElementById("status");
  if (status) status.textContent = e.payload === "listening" ? "listening…" : "idle";
});

listen("analytics-updated", () => {
  refreshAnalytics();
});

listen("model-download", (e) => {
  const { file, phase, bytes, total } = e.payload || {};
  const label = document.getElementById("whisper_progress_label");
  const bar = document.getElementById("whisper_progress_bar");
  if (!label || !bar) return;
  if (phase === "starting") {
    label.textContent = `${file} — starting…`;
    bar.style.width = "0%";
  } else if (phase === "progress") {
    const pct = total ? ((bytes / total) * 100).toFixed(0) : null;
    label.textContent = pct
      ? `${file} — ${pct}% (${fmtBytes(bytes)} / ${fmtBytes(total)})`
      : `${file} — ${fmtBytes(bytes)}`;
    if (pct) bar.style.width = pct + "%";
  } else if (phase === "done") {
    label.textContent = `${file} — done`;
    bar.style.width = "100%";
  }
});

// Surface the active acceleration backend in the settings UI. The
// detection uses navigator.platform as a quick, render-time hint —
// authoritative info is in the Rust logs ("gpu=true/false").
(() => {
  const accelEl = document.getElementById("accel_label");
  if (!accelEl) return;
  const isMac = /Mac|Darwin/i.test(navigator.platform || navigator.userAgent || "");
  accelEl.textContent = isMac ? "Metal (Apple GPU)" : "CPU (AVX2 + OpenMP)";
})();

// loadConfig is invoked above (before wireAutoSave). Don't double-call
// here — that would clobber any value the user just typed if the second
// load resolved later than the first.
refreshWhisperStatus();
refreshAnalytics();
