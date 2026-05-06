// Per-utterance dictation analytics: persistent record of every
// finalized transcript plus a derived summary view consumed by the
// dashboard UI. Stored as JSON next to config.json in the app config
// dir; the file is small (a few hundred KB at most thanks to MAX_UTTERANCES).

use anyhow::Result;
use chrono::{DateTime, Datelike, Local, NaiveDate, TimeZone, Timelike};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Manager};

const MAX_UTTERANCES: usize = 5000;
const TOP_WORDS: usize = 30;
const RECENT_LIMIT: usize = 12;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Utterance {
    /// Unix epoch seconds (UTC) at finalize.
    pub timestamp: i64,
    /// Wall-clock duration of the dictation session in milliseconds.
    pub duration_ms: u64,
    /// Backend name — currently always "whisper". Kept as a string
    /// for forward compatibility (older entries on disk may say
    /// "voxtral" or "moonshine"; we still display them in recents).
    pub backend: String,
    /// Final transcript text.
    pub text: String,
    pub word_count: u32,
    pub char_count: u32,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Analytics {
    #[serde(default)]
    pub utterances: Vec<Utterance>,
    #[serde(default)]
    pub created_at: i64,
}

/// What the dashboard renders.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyticsSummary {
    // Headline counters.
    pub total_utterances: u64,
    pub total_words: u64,
    pub total_chars: u64,
    pub total_duration_ms: u64,
    pub avg_words_per_session: f64,
    pub avg_words_per_minute: f64,
    pub avg_session_seconds: f64,
    pub longest_session_ms: u64,
    pub longest_transcript_words: u32,

    // Time-of-day patterns.
    pub hourly: [u32; 24],
    pub weekday: [u32; 7], // Mon=0 ... Sun=6
    pub last_7_days: Vec<DayBucket>,
    pub current_streak_days: u32,
    pub longest_streak_days: u32,
    pub days_active: u32,
    pub first_used: Option<i64>,
    pub last_used: Option<i64>,
    pub peak_hour: Option<u32>,
    pub peak_weekday: Option<u32>,

    // Vocabulary.
    pub unique_words: u32,
    pub top_words: Vec<WordCount>,
    pub avg_word_length: f64,
    pub longest_word: Option<String>,
    pub vocab_richness: f64, // unique / total (0..1)

    // Recent activity.
    pub recent: Vec<RecentEntry>,

    // Generated text bullets.
    pub insights: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DayBucket {
    pub date: String,
    pub utterances: u32,
    pub words: u32,
    pub duration_ms: u64,
    pub label: String, // short "Mon", "Tue", ...
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WordCount {
    pub word: String,
    pub count: u32,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RecentEntry {
    pub timestamp: i64,
    pub backend: String,
    pub words: u32,
    pub preview: String,
}

impl Analytics {
    /// Append a new utterance; trims to MAX_UTTERANCES if needed.
    pub fn record(&mut self, text: &str, duration: Duration, backend: &str) {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        if self.created_at == 0 {
            self.created_at = now;
        }

        let trimmed = text.trim();
        let word_count = trimmed.split_whitespace().count() as u32;
        let char_count = trimmed.chars().count() as u32;

        self.utterances.push(Utterance {
            timestamp: now,
            duration_ms: duration.as_millis() as u64,
            backend: backend.to_string(),
            text: trimmed.to_string(),
            word_count,
            char_count,
        });

        if self.utterances.len() > MAX_UTTERANCES {
            let drop = self.utterances.len() - MAX_UTTERANCES;
            self.utterances.drain(0..drop);
        }
    }

    pub fn summary(&self) -> AnalyticsSummary {
        let n = self.utterances.len();

        let total_utterances = n as u64;
        let total_words: u64 = self.utterances.iter().map(|u| u.word_count as u64).sum();
        let total_chars: u64 = self.utterances.iter().map(|u| u.char_count as u64).sum();
        let total_duration_ms: u64 = self.utterances.iter().map(|u| u.duration_ms).sum();

        let avg_words_per_session = if n > 0 {
            total_words as f64 / n as f64
        } else {
            0.0
        };
        let total_minutes = total_duration_ms as f64 / 60_000.0;
        let avg_words_per_minute = if total_minutes > 0.0 {
            total_words as f64 / total_minutes
        } else {
            0.0
        };
        let avg_session_seconds = if n > 0 {
            (total_duration_ms as f64 / 1000.0) / n as f64
        } else {
            0.0
        };
        let longest_session_ms = self
            .utterances
            .iter()
            .map(|u| u.duration_ms)
            .max()
            .unwrap_or(0);
        let longest_transcript_words = self
            .utterances
            .iter()
            .map(|u| u.word_count)
            .max()
            .unwrap_or(0);

        // Hour-of-day + weekday distributions (local time).
        let mut hourly = [0u32; 24];
        let mut weekday = [0u32; 7];
        for u in &self.utterances {
            let dt = local_dt(u.timestamp);
            hourly[dt.hour() as usize] += 1;
            // Datelike::weekday() — Mon=0 with .num_days_from_monday()
            weekday[dt.weekday().num_days_from_monday() as usize] += 1;
        }
        let peak_hour = (0..24)
            .max_by_key(|&h| hourly[h as usize])
            .filter(|&h| hourly[h as usize] > 0);
        let peak_weekday = (0..7)
            .max_by_key(|&d| weekday[d as usize])
            .filter(|&d| weekday[d as usize] > 0);

        // Last 7 days bucketed by local date.
        let last_7_days = build_last_7_days(&self.utterances);

        // Streaks based on unique active local dates.
        let unique_dates: std::collections::BTreeSet<NaiveDate> = self
            .utterances
            .iter()
            .map(|u| local_dt(u.timestamp).date_naive())
            .collect();
        let days_active = unique_dates.len() as u32;
        let (current_streak_days, longest_streak_days) = compute_streaks(&unique_dates);

        let first_used = self.utterances.first().map(|u| u.timestamp);
        let last_used = self.utterances.last().map(|u| u.timestamp);

        // Vocabulary.
        let (unique_words, top_words, avg_word_length, longest_word, vocab_richness) =
            vocab_stats(&self.utterances);

        // Recent.
        let recent: Vec<RecentEntry> = self
            .utterances
            .iter()
            .rev()
            .take(RECENT_LIMIT)
            .map(|u| {
                let preview: String = u.text.chars().take(90).collect();
                let preview = if u.text.chars().count() > 90 {
                    format!("{preview}…")
                } else {
                    preview
                };
                RecentEntry {
                    timestamp: u.timestamp,
                    backend: u.backend.clone(),
                    words: u.word_count,
                    preview,
                }
            })
            .collect();

        let insights = build_insights(
            n,
            total_words,
            total_duration_ms,
            avg_words_per_minute,
            avg_session_seconds,
            longest_session_ms,
            peak_hour,
            peak_weekday,
            current_streak_days,
            days_active,
            &top_words,
            vocab_richness,
            longest_transcript_words,
        );

        AnalyticsSummary {
            total_utterances,
            total_words,
            total_chars,
            total_duration_ms,
            avg_words_per_session,
            avg_words_per_minute,
            avg_session_seconds,
            longest_session_ms,
            longest_transcript_words,
            hourly,
            weekday,
            last_7_days,
            current_streak_days,
            longest_streak_days,
            days_active,
            first_used,
            last_used,
            peak_hour,
            peak_weekday,
            unique_words,
            top_words,
            avg_word_length,
            longest_word,
            vocab_richness,
            recent,
            insights,
        }
    }

    pub fn reset(&mut self) {
        self.utterances.clear();
        self.created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
    }
}

fn local_dt(ts: i64) -> DateTime<Local> {
    Local
        .timestamp_opt(ts, 0)
        .single()
        .unwrap_or_else(|| Local::now())
}

fn build_last_7_days(utterances: &[Utterance]) -> Vec<DayBucket> {
    let today = Local::now().date_naive();
    let mut by_date: HashMap<NaiveDate, (u32, u32, u64)> = HashMap::new();
    for u in utterances {
        let d = local_dt(u.timestamp).date_naive();
        let entry = by_date.entry(d).or_insert((0, 0, 0));
        entry.0 += 1;
        entry.1 += u.word_count;
        entry.2 += u.duration_ms;
    }
    let mut out = Vec::with_capacity(7);
    for offset in (0..7).rev() {
        let d = today - chrono::Duration::days(offset);
        let (utts, wds, dur) = by_date.get(&d).copied().unwrap_or((0, 0, 0));
        let label = match d.weekday().num_days_from_monday() {
            0 => "Mon",
            1 => "Tue",
            2 => "Wed",
            3 => "Thu",
            4 => "Fri",
            5 => "Sat",
            _ => "Sun",
        };
        out.push(DayBucket {
            date: d.format("%Y-%m-%d").to_string(),
            utterances: utts,
            words: wds,
            duration_ms: dur,
            label: label.to_string(),
        });
    }
    out
}

fn compute_streaks(dates: &std::collections::BTreeSet<NaiveDate>) -> (u32, u32) {
    if dates.is_empty() {
        return (0, 0);
    }
    let mut sorted: Vec<NaiveDate> = dates.iter().copied().collect();
    sorted.sort();

    let mut longest = 1u32;
    let mut run = 1u32;
    for w in sorted.windows(2) {
        if (w[1] - w[0]).num_days() == 1 {
            run += 1;
            if run > longest {
                longest = run;
            }
        } else {
            run = 1;
        }
    }

    // Current streak: count back from today (or yesterday if today not active).
    let today = Local::now().date_naive();
    let mut cursor = if dates.contains(&today) {
        today
    } else if dates.contains(&(today - chrono::Duration::days(1))) {
        today - chrono::Duration::days(1)
    } else {
        return (0, longest);
    };
    let mut current = 0u32;
    while dates.contains(&cursor) {
        current += 1;
        cursor -= chrono::Duration::days(1);
    }
    (current, longest)
}

fn vocab_stats(utterances: &[Utterance]) -> (u32, Vec<WordCount>, f64, Option<String>, f64) {
    let stop = stopwords();
    let mut counts: HashMap<String, u32> = HashMap::new();
    let mut total_word_chars: u64 = 0;
    let mut total_word_tokens: u64 = 0;
    let mut longest: Option<String> = None;

    for u in utterances {
        for raw in u.text.split_whitespace() {
            // Strip leading/trailing punctuation, lowercase.
            let cleaned: String = raw
                .trim_matches(|c: char| !c.is_alphanumeric() && c != '\'')
                .to_lowercase();
            if cleaned.is_empty() {
                continue;
            }
            total_word_tokens += 1;
            total_word_chars += cleaned.chars().count() as u64;
            match &longest {
                Some(l) if l.chars().count() >= cleaned.chars().count() => {}
                _ => longest = Some(cleaned.clone()),
            }
            if stop.contains(cleaned.as_str()) {
                continue;
            }
            // Skip pure-numeric tokens — usually noise in word clouds.
            if cleaned.chars().all(|c| c.is_ascii_digit()) {
                continue;
            }
            *counts.entry(cleaned).or_insert(0) += 1;
        }
    }

    let unique_words = counts.len() as u32;
    let mut top: Vec<WordCount> = counts
        .into_iter()
        .map(|(word, count)| WordCount { word, count })
        .collect();
    top.sort_by(|a, b| b.count.cmp(&a.count).then(a.word.cmp(&b.word)));
    top.truncate(TOP_WORDS);

    let avg_word_length = if total_word_tokens > 0 {
        total_word_chars as f64 / total_word_tokens as f64
    } else {
        0.0
    };
    let vocab_richness = if total_word_tokens > 0 {
        unique_words as f64 / total_word_tokens as f64
    } else {
        0.0
    };
    (unique_words, top, avg_word_length, longest, vocab_richness)
}

fn stopwords() -> std::collections::HashSet<&'static str> {
    [
        "a", "an", "and", "or", "but", "the", "of", "to", "in", "on", "at", "by", "for", "with",
        "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "doing",
        "have", "has", "had", "having", "i", "you", "he", "she", "it", "we", "they", "me", "him",
        "her", "us", "them", "my", "your", "his", "their", "our", "this", "that", "these", "those",
        "as", "if", "so", "not", "no", "yes", "from", "up", "down", "out", "over", "under", "then",
        "than", "just", "very", "really", "kind", "sort", "like", "well", "okay", "ok", "so",
        "now", "also", "yeah", "yep", "hmm", "uh", "um", "er", "ah", "oh", "what", "when", "where",
        "who", "why", "how", "which", "would", "could", "should", "will", "shall", "can", "may",
        "might", "must", "into", "about", "after", "before", "again", "any", "all", "some", "more",
        "most", "other", "such", "only", "own", "same", "too", "s", "t", "d", "ll", "m", "ve",
        "re", "isn't", "don't", "i'm", "it's", "that's", "you're", "we're", "they're", "there",
        "here",
    ]
    .into_iter()
    .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_insights(
    n: usize,
    total_words: u64,
    total_duration_ms: u64,
    wpm: f64,
    avg_session_seconds: f64,
    longest_session_ms: u64,
    peak_hour: Option<u32>,
    peak_weekday: Option<u32>,
    current_streak: u32,
    days_active: u32,
    top_words: &[WordCount],
    vocab_richness: f64,
    longest_transcript_words: u32,
) -> Vec<String> {
    let mut out = Vec::new();
    if n == 0 {
        out.push(
            "No dictations recorded yet. Press the dictate hotkey and speak — your stats will populate here."
                .into(),
        );
        return out;
    }

    let total_minutes = total_duration_ms as f64 / 60_000.0;
    out.push(format!(
        "You've spent {} talking to VibeToText across {} sessions.",
        humanize_minutes(total_minutes),
        n
    ));

    if total_words > 0 {
        out.push(format!(
            "You've dictated {} words — averaging {:.0} per session at {:.0} WPM.",
            thousands(total_words),
            (total_words as f64) / (n as f64),
            wpm
        ));
    }

    if let Some(h) = peak_hour {
        out.push(format!(
            "You're most active around {}. {}",
            format_hour(h),
            hour_vibe(h)
        ));
    }

    if let Some(d) = peak_weekday {
        out.push(format!(
            "Your most productive day of the week is {}.",
            weekday_name(d)
        ));
    }

    if current_streak >= 2 {
        out.push(format!(
            "🔥 You're on a {}-day streak. Keep it up!",
            current_streak
        ));
    } else if days_active >= 1 {
        out.push(format!(
            "You've used VibeToText on {} unique day(s).",
            days_active
        ));
    }

    if avg_session_seconds > 0.0 {
        out.push(format!(
            "Average session length: {:.1}s. Longest: {:.1}s.",
            avg_session_seconds,
            longest_session_ms as f64 / 1000.0
        ));
    }

    if let Some(top) = top_words.first() {
        out.push(format!(
            "Your favourite word is \"{}\" — used {} times.",
            top.word, top.count
        ));
    }

    if vocab_richness > 0.0 {
        let label = if vocab_richness > 0.55 {
            "extremely varied"
        } else if vocab_richness > 0.4 {
            "rich and varied"
        } else if vocab_richness > 0.25 {
            "fairly diverse"
        } else {
            "focused / repetitive"
        };
        out.push(format!(
            "Vocabulary richness: {:.0}% unique tokens — {}.",
            vocab_richness * 100.0,
            label
        ));
    }

    if longest_transcript_words >= 50 {
        out.push(format!(
            "Your longest single dictation was {} words.",
            longest_transcript_words
        ));
    }

    out
}

fn humanize_minutes(m: f64) -> String {
    if m < 1.0 {
        format!("{:.0}s", m * 60.0)
    } else if m < 60.0 {
        format!("{:.1} min", m)
    } else {
        let h = (m / 60.0).floor();
        let rem = m - h * 60.0;
        format!("{:.0}h {:.0}m", h, rem)
    }
}

fn thousands(n: u64) -> String {
    let s = n.to_string();
    let chars: Vec<char> = s.chars().rev().collect();
    let mut out = String::new();
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && i % 3 == 0 {
            out.push(',');
        }
        out.push(*c);
    }
    out.chars().rev().collect()
}

fn format_hour(h: u32) -> String {
    let suffix = if h < 12 { "AM" } else { "PM" };
    let h12 = match h % 12 {
        0 => 12,
        x => x,
    };
    format!("{}:00 {}", h12, suffix)
}

fn hour_vibe(h: u32) -> &'static str {
    match h {
        5..=8 => "An early bird, eh?",
        9..=11 => "Morning focus mode.",
        12..=13 => "Lunch-break productivity.",
        14..=17 => "Classic afternoon flow.",
        18..=21 => "Evening grinder.",
        22..=23 | 0..=4 => "Night owl 🦉.",
        _ => "",
    }
}

fn weekday_name(d: u32) -> &'static str {
    match d {
        0 => "Monday",
        1 => "Tuesday",
        2 => "Wednesday",
        3 => "Thursday",
        4 => "Friday",
        5 => "Saturday",
        _ => "Sunday",
    }
}

// ---------------- Persistence ----------------

fn analytics_path(app: &AppHandle) -> Result<PathBuf> {
    let dir = app.path().app_config_dir()?;
    std::fs::create_dir_all(&dir).ok();
    Ok(dir.join("analytics.json"))
}

pub fn load(app: &AppHandle) -> Analytics {
    let p = match analytics_path(app) {
        Ok(p) => p,
        Err(_) => return Analytics::default(),
    };
    if !p.exists() {
        return Analytics::default();
    }
    match std::fs::read_to_string(&p) {
        Ok(raw) => serde_json::from_str(&raw).unwrap_or_default(),
        Err(_) => Analytics::default(),
    }
}

pub fn save(app: &AppHandle, a: &Analytics) -> Result<()> {
    let p = analytics_path(app)?;
    let json = serde_json::to_string(a)?;
    std::fs::write(p, json)?;
    Ok(())
}

/// Convenience: record from an STT backend. Loads the AppState, mutates
/// the in-memory analytics, and persists. Errors are logged but don't
/// surface — analytics shouldn't break dictation.
pub async fn record_from_backend(app: &AppHandle, text: &str, duration: Duration, backend: &str) {
    if text.trim().is_empty() {
        return;
    }
    let state: tauri::State<std::sync::Arc<crate::AppState>> = app.state();
    let mut a = state.analytics.lock().await;
    a.record(text, duration, backend);
    if let Err(e) = save(app, &a) {
        log::warn!("failed to persist analytics: {e:#}");
    }
    let _ = app.emit("analytics-updated", ());
}

// re-export Emitter so the macro line above type-checks without
// extra imports in callers.
use tauri::Emitter;
