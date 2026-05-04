// Live audio waveform overlay for push-to-talk dictation.
//
// Behavior:
//   - When dictation starts the buffer is empty. New samples are written
//     left-to-right. The waveform GROWS across the canvas until it
//     reaches the right edge, then it scrolls (push-left, newest on
//     the right) like a classic oscilloscope. Older bars fade out near
//     the left edge.
//   - Audio levels arrive as RMS scalars from Rust at ~50Hz via the
//     "audio-level" event. We log-scale to dBFS for perceptual sanity.
//   - Talking vs silence contrast is amplified by raising amplitudes
//     above a soft floor and biasing the noise floor downward — so
//     silence looks visibly flat and speech visibly tall.

const { listen } = window.__TAURI__.event;

const canvas = document.getElementById("viz");
const ctx = canvas.getContext("2d");
const elapsed = document.getElementById("elapsed");

// Ring buffer of normalized [0..1] amplitudes, log-scaled.
const HISTORY = 220;            // ~4.4s at 50Hz
const buf = new Float32Array(HISTORY);
let count = 0;                  // number of valid samples (caps at HISTORY)
let head = 0;                   // index of *next* write
let startedAt = performance.now();

// Pixel ratio — overlay window has fixed pixel size, so handle DPR.
function fitCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  if (canvas.width !== cssW * dpr || canvas.height !== cssH * dpr) {
    canvas.width = cssW * dpr;
    canvas.height = cssH * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}

// Map RMS → [0..1] with an aggressive contrast curve so speech looks
// dramatically taller than ambient room tone.
//   1. dB-scale around a -50dB noise floor → 0..1
//   2. Two-zone shape: noise gate hard-clamps silence near zero, and
//      above the gate we use a sqrt-style curve so even modest speech
//      reaches high amplitudes (visually punchy peaks).
//   3. Allow occasional peaks above 1.0; the draw loop clips them.
function rmsToNorm(rms) {
  const safe = Math.max(rms, 1e-5);
  const db = 20 * Math.log10(safe);
  const min = -50;
  const max = 0;
  let v = (db - min) / (max - min);
  v = Math.max(0, Math.min(1.0, v));

  // Hard noise gate — silence is FLAT, not a low hum. The contrast
  // between flat-line silence and speech-time peaks is what makes the
  // waveform feel alive instead of always wiggling.
  const gate = 0.22;
  if (v <= gate) {
    // Square the input below the gate so even soft room tone settles
    // to ~0.005, basically invisible.
    v = Math.pow(v / gate, 2) * 0.03;
  } else {
    // Above the gate: sqrt curve so even a half-volume "hello" pushes
    // close to 0.7-0.8. Loud speech easily hits 1.0.
    const t = (v - gate) / (1 - gate);
    v = 0.03 + 0.97 * Math.pow(t, 0.5);
  }
  return v;
}

listen("audio-level", (e) => {
  const rms = typeof e.payload === "number" ? e.payload : 0;
  buf[head] = rmsToNorm(rms);
  head = (head + 1) % HISTORY;
  if (count < HISTORY) count++;
});

function draw() {
  fitCanvas();
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  ctx.clearRect(0, 0, W, H);

  const mid = H / 2;
  const barW = W / HISTORY;

  // --- Fill-from-left phase ---
  // While count < HISTORY the bars grow from x=0 outward to x=count*barW.
  // Once full we scroll: oldest sample on the left, newest on the right.
  // Sample order: oldest sample is at (head - count) mod HISTORY when
  // full, or at index 0 when growing (since we always start from 0).
  const filling = count < HISTORY;
  const samplesToDraw = filling ? count : HISTORY;
  const startBufIdx = filling ? 0 : head; // oldest visible sample

  // Visual easing: an extra 0.85-power curve on the rendered HEIGHT.
  // Combined with the sqrt curve already in `rmsToNorm`, this means
  // softer sounds get a small visual boost too — but the dominant
  // effect is still that loud peaks tower over silence-flat baseline.
  const maxHalf = H / 2 - 2;
  for (let i = 0; i < samplesToDraw; i++) {
    const idx = (startBufIdx + i) % HISTORY;
    const v = buf[idx];

    // Only fade the leading edge while scrolling (so old data dissolves
    // off the left). While filling, show the full crisp range from x=0.
    let alpha = 1.0;
    if (!filling) {
      const leadFade = Math.min(1, i / 14);
      alpha = leadFade;
    }
    // Trailing-edge fade keeps the right-most bars from looking like a
    // hard cutoff while they're being added.
    const trail = Math.min(1, (samplesToDraw - i) / 4);
    alpha *= trail;

    // Height: easing curve + headroom clipping. Allow the curve to push
    // up to ~95% of the canvas while leaving 2px breathing room.
    const eased = Math.pow(v, 0.85);
    const halfH = Math.min(eased * maxHalf, maxHalf);
    const x = i * barW;

    // Color by amplitude. At silence: deep, low-saturation orange-brown
    // (almost invisible). At speech peaks: near-white peach with a hint
    // of orange. The dramatic luminance ramp makes peaks visually pop
    // even at the same physical bar height.
    const t = Math.min(1, v);
    const r = 255;
    const g = Math.floor(95 + t * 155);
    const b = Math.floor(40 + t * 175);
    const aBase = 0.45 + 0.55 * Math.pow(t, 0.7);
    ctx.fillStyle = `rgba(${r},${g},${b},${aBase * alpha})`;

    // Symmetric bar — classic waveform shape. We pad a 1px minimum so
    // even silence has a hairline trace (looks alive, not crashed).
    const barH = Math.max(1.5, halfH * 2);
    ctx.fillRect(x, mid - barH / 2, Math.max(1, barW * 0.72), barH);
  }

  // Subtle right-edge highlight only while scrolling. Drawing it during
  // the fill phase looks weird because the "edge" keeps moving.
  if (!filling) {
    const grd = ctx.createLinearGradient(W - 50, 0, W, 0);
    grd.addColorStop(0, "rgba(255,150,100,0)");
    grd.addColorStop(1, "rgba(255,150,100,0.14)");
    ctx.fillStyle = grd;
    ctx.fillRect(W - 50, 0, 50, H);
  }

  // Center line.
  ctx.fillStyle = "rgba(255,255,255,0.05)";
  ctx.fillRect(0, mid - 0.5, W, 1);

  // Update the elapsed counter.
  if (card.classList.contains("warming")) {
    // While warming, prefer concrete download progress over a vague
    // "loading…". Falls back to the generic message until the first
    // model-download event arrives (e.g. during whisper-cli's local
    // disk-load phase, after both downloads have finished).
    const dl = currentDownload;
    if (dl && dl.phase === "extracting") {
      title.textContent = "Extracting runtime";
      elapsed.textContent = `unpacking ${dl.file}…`;
    } else if (dl && dl.phase === "done") {
      title.textContent = "Preparing model";
      elapsed.textContent = `${dl.file} ✓ — finishing up…`;
    } else if (dl && dl.bytes != null) {
      const friendly = dlLabelFor(dl.file);
      title.textContent = `Downloading ${friendly}`;
      const mb = (dl.bytes / 1_048_576).toFixed(1);
      if (dl.total) {
        const totalMb = (dl.total / 1_048_576).toFixed(1);
        const pct = Math.min(100, Math.floor((dl.bytes / dl.total) * 100));
        elapsed.textContent = `${pct}% · ${mb} / ${totalMb} MB`;
      } else {
        elapsed.textContent = `${mb} MB downloaded…`;
      }
    } else {
      title.textContent = "Preparing model";
      elapsed.textContent = "loading speech model…";
    }
  } else if (card.classList.contains("transcribing")) {
    const ts = (performance.now() - transcribingStartedAt) / 1000;
    elapsed.textContent = `transcribing… ${ts.toFixed(1)}s`;
  } else if (!card.classList.contains("errored")) {
    const sec = (performance.now() - startedAt) / 1000;
    elapsed.textContent = `${sec.toFixed(1)}s · release to send`;
  }

  requestAnimationFrame(draw);
}

// State machine driven by the Rust side:
//   "listening"    → live waveform + "release to send" hint
//   "transcribing" → equalizer spinner + "transcribing…" hint
//   "idle"         → (overlay window gets hidden by Rust; nothing to draw)
const card = document.getElementById("card");
const title = document.getElementById("title");
let transcribingStartedAt = 0;

// Latest model-download event from Rust, kept module-level so the
// draw loop can render it into the warming-state hint without racing
// against the event arrival. Survives across press/release cycles —
// when the user re-presses the dictate hotkey while a background
// download is still in flight, the overlay picks up where it left off.
let currentDownload = null; // { file, phase, bytes, total }

// Translate the raw filename into something readable for the user.
// Hides our internal "ggml-large-v3-turbo-q5_0.bin" naming.
function dlLabelFor(file) {
  if (!file) return "model";
  if (/whisper-bin|whisper-cli|\.zip$/i.test(file)) return "runtime";
  if (/ggml-.*\.bin$/i.test(file)) return "model";
  return file;
}

// Listen for download progress from Rust. Same `model-download` event
// the settings UI reads — both surfaces share one stream.
listen("model-download", (e) => {
  currentDownload = e.payload || null;
  // When a download finishes, hold onto the "done" state briefly so
  // the user sees the ✓ flash, then clear so the next download starts
  // from a clean slate. The draw loop will pick this up.
  if (currentDownload && currentDownload.phase === "done") {
    setTimeout(() => {
      // Only clear if the latest event is still this same "done" —
      // otherwise a new "starting" event already replaced it.
      if (currentDownload && currentDownload.phase === "done") {
        currentDownload = null;
      }
    }, 600);
  }
});

listen("dictation-state", (e) => {
  if (e.payload === "warming") {
    // Cold-start race: model is still loading. Show a "Preparing…" hint
    // so the user knows the silence is intentional, not a frozen UI.
    card.classList.remove("transcribing", "leaving", "errored");
    card.classList.add("warming");
    card.style.animation = "none";
    void card.offsetWidth;
    card.style.animation = "";
    title.textContent = "Preparing model";
    startedAt = performance.now();
    for (let i = 0; i < HISTORY; i++) buf[i] = 0;
    head = 0;
    count = 0;
  } else if (e.payload === "listening") {
    card.classList.remove("transcribing", "leaving", "errored", "warming");
    // CSS keyframes don't auto-replay just because we removed the
    // .leaving class — without this dance the card stays stuck at
    // opacity:0/translateY (the final state of card-out) on the
    // second-and-later dictations, which is exactly the "popup
    // doesn't show" bug. Strip the animation, force a reflow, restore
    // it: the browser then treats it as a fresh animation and plays
    // card-in from frame 0.
    card.style.animation = "none";
    void card.offsetWidth;
    card.style.animation = "";
    title.textContent = "Listening";
    startedAt = performance.now();
    // Reset to fill-from-left state.
    for (let i = 0; i < HISTORY; i++) buf[i] = 0;
    head = 0;
    count = 0;
  } else if (e.payload === "transcribing") {
    card.classList.add("transcribing");
    title.textContent = "Transcribing";
    transcribingStartedAt = performance.now();
  } else if (e.payload === "idle") {
    // Play the leave animation. Rust hides the window shortly after; the
    // animation finishing in time is best-effort.
    card.classList.remove("transcribing");
    card.classList.add("leaving");
  }
});

// Surface STT failures (Whisper model download failed, audio init
// failed, etc.) so the user sees what's wrong instead of a
// silently-broken hotkey.
listen("dictation-error", (e) => {
  card.classList.remove("transcribing", "leaving");
  card.classList.add("errored");
  title.textContent = "Couldn't start";
  const hint = e.payload || "unknown error";
  // The overlay uses #elapsed for the secondary line — repurpose it.
  if (elapsed) elapsed.textContent = String(hint).slice(0, 80);
});

requestAnimationFrame(draw);
