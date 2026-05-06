// Voice Activity Detection (VAD) — energy/RMS-threshold edition.
//
// Why energy-based instead of Silero-via-ONNX-Runtime:
//
//   - Push-to-talk dictation has unambiguous start/stop boundaries
//     (the user holds the hotkey). The hard problem VAD usually
//     solves — segmenting continuous audio into utterances —
//     doesn't apply. We just need to clean the EDGES.
//   - The dominant pain points are:
//       (a) leading silence: user takes ~300 ms to start speaking
//           after pressing the hotkey, which becomes wasted encoder
//           work,
//       (b) trailing silence: user stops speaking, takes ~500 ms
//           to release the hotkey, and Whisper hallucinates into
//           the silent tail ("the the the…", "Thanks for watching").
//     Both are solved by simple amplitude-based trimming.
//   - Silero VAD via `ort` re-introduces an ONNX Runtime dependency
//     we just spent significant effort removing. Big complexity
//     ask for marginal benefit on push-to-talk.
//
// Algorithm: chunk the audio into 20 ms frames, compute RMS per
// frame, classify as speech if `rms > THRESHOLD`. To tolerate
// brief sub-threshold dips inside real speech (consonants, breath
// pauses), we use HYSTERESIS — once we've seen N consecutive
// speech frames we're "in speech" until we see M consecutive
// silent frames.
//
// We trim ONLY edges. We don't try to remove silence in the middle
// of an utterance — Whisper handles that fine internally, and
// stitching speech segments back together would risk garbled
// phoneme boundaries.

const SAMPLE_RATE: usize = 16_000;
/// 20 ms frame at 16 kHz.
const FRAME_SAMPLES: usize = SAMPLE_RATE / 50;
/// RMS amplitude above which a frame counts as speech. Empirically
/// tuned for a push-to-talk dictation app: 0.005 (-46 dBFS) is well
/// above typical room noise (mic preamp hiss is around -60 dBFS)
/// while staying low enough not to clip the quietest plosives.
const SPEECH_THRESHOLD: f32 = 0.005;
/// How many consecutive above-threshold frames before we declare
/// "speech started". 1 frame is too jumpy (single mouse click),
/// 3+ frames is too sluggish (chops the start of words). 2 = 40 ms
/// works well.
const SPEECH_ON_FRAMES: usize = 2;
/// How many consecutive silent frames before we declare "speech
/// ended". Higher = more tolerant of pauses inside words/sentences;
/// at 30 frames (600 ms) we keep natural sentence pauses intact
/// without trapping training-set hallucination silence.
const SPEECH_OFF_FRAMES: usize = 30;
/// Pad the trimmed segment with a small amount of silence on each
/// side so we don't clip the very start/end of the first/last
/// phoneme. 5 frames = 100 ms.
const EDGE_PAD_FRAMES: usize = 5;
/// Don't bother running VAD on clips this short — they're shorter
/// than the hysteresis windows would meaningfully resolve.
const MIN_SAMPLES_TO_TRIM: usize = SAMPLE_RATE / 4; // 250 ms

/// Trim leading and trailing silence from `samples`. Returns the
/// trimmed slice as a borrowed view (no allocation). If the input
/// is too short, contains no detected speech, or VAD analysis
/// fails for any reason, returns the original slice unchanged —
/// the philosophy is "VAD is a perf optimization, never a
/// correctness one; on uncertainty pass through".
pub fn trim_silence(samples: &[f32]) -> &[f32] {
    if samples.len() < MIN_SAMPLES_TO_TRIM {
        return samples;
    }

    // 1. Compute RMS for every 20 ms frame.
    let n_frames = samples.len() / FRAME_SAMPLES;
    if n_frames == 0 {
        return samples;
    }
    let mut rms = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * FRAME_SAMPLES;
        let end = start + FRAME_SAMPLES;
        let sum_sq: f32 = samples[start..end].iter().map(|&s| s * s).sum();
        rms.push((sum_sq / FRAME_SAMPLES as f32).sqrt());
    }

    // 2. Walk frames forward applying hysteresis to find the
    //    first frame that's confidently inside speech.
    let mut consecutive_speech = 0usize;
    let mut first_speech_frame: Option<usize> = None;
    for (i, &r) in rms.iter().enumerate() {
        if r > SPEECH_THRESHOLD {
            consecutive_speech += 1;
            if consecutive_speech >= SPEECH_ON_FRAMES {
                // The confident-speech onset is `SPEECH_ON_FRAMES - 1`
                // frames BEFORE the current index — that's where the
                // run started.
                first_speech_frame = Some(i + 1 - SPEECH_ON_FRAMES);
                break;
            }
        } else {
            consecutive_speech = 0;
        }
    }
    let Some(first) = first_speech_frame else {
        // Pure silence detected. Don't return empty — let Whisper
        // see the audio so the hallucination filter handles the
        // result. (Returning &[] would crash mel spectrogram.)
        log::debug!("VAD: no speech detected, passing through unchanged");
        return samples;
    };

    // 3. Walk backwards looking for the LAST speech onset run.
    //    We use the same hysteresis but reversed — find the last
    //    point where we have SPEECH_OFF_FRAMES consecutive silent
    //    frames AFTER speech.
    let mut consecutive_silent = 0usize;
    let mut last_speech_frame = n_frames - 1;
    for (i, &r) in rms.iter().enumerate().rev() {
        if r > SPEECH_THRESHOLD {
            // Found the last speech frame walking backwards.
            last_speech_frame = i;
            break;
        }
        consecutive_silent += 1;
        if consecutive_silent > SPEECH_OFF_FRAMES * 4 {
            // Way past any reasonable trailing pause; stop searching
            // and use what we have.
            break;
        }
    }

    // 4. Apply edge padding so we don't clip phoneme boundaries.
    let first_padded = first.saturating_sub(EDGE_PAD_FRAMES);
    let last_padded = (last_speech_frame + EDGE_PAD_FRAMES).min(n_frames - 1);

    let start_sample = first_padded * FRAME_SAMPLES;
    let end_sample = ((last_padded + 1) * FRAME_SAMPLES).min(samples.len());

    if start_sample >= end_sample {
        // Pathological: nothing left after trimming. Fall back to
        // the original samples so Whisper at least gets something.
        return samples;
    }

    let trimmed = &samples[start_sample..end_sample];
    log::debug!(
        "VAD: trimmed {} → {} samples ({:.2}s → {:.2}s, {:.0}% reduction)",
        samples.len(),
        trimmed.len(),
        samples.len() as f32 / SAMPLE_RATE as f32,
        trimmed.len() as f32 / SAMPLE_RATE as f32,
        100.0 * (1.0 - trimmed.len() as f32 / samples.len() as f32)
    );
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(seconds: f32) -> Vec<f32> {
        vec![0.0; (seconds * SAMPLE_RATE as f32) as usize]
    }

    fn tone(seconds: f32, amplitude: f32) -> Vec<f32> {
        let n = (seconds * SAMPLE_RATE as f32) as usize;
        // 440 Hz sine — clearly above the speech threshold.
        (0..n)
            .map(|i| {
                amplitude
                    * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / SAMPLE_RATE as f32).sin()
            })
            .collect()
    }

    #[test]
    fn trims_leading_and_trailing_silence() {
        let mut audio = silence(0.5);
        audio.extend(tone(1.0, 0.3));
        audio.extend(silence(2.0));

        let trimmed = trim_silence(&audio);
        let trimmed_secs = trimmed.len() as f32 / SAMPLE_RATE as f32;
        // We should get back ~1 s of tone plus the edge padding,
        // not the original 3.5 s.
        assert!(
            trimmed_secs > 0.8 && trimmed_secs < 1.5,
            "expected ~1s after trim, got {:.2}s",
            trimmed_secs
        );
    }

    #[test]
    fn passes_through_pure_silence() {
        let audio = silence(2.0);
        // Pure silence: pass through unchanged (don't crash).
        let trimmed = trim_silence(&audio);
        assert_eq!(trimmed.len(), audio.len());
    }

    #[test]
    fn passes_through_short_clips() {
        let audio = silence(0.1);
        let trimmed = trim_silence(&audio);
        assert_eq!(trimmed.len(), audio.len());
    }
}
