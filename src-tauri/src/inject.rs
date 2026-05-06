use anyhow::Result;
use enigo::{Enigo, Key, Keyboard, Settings, Direction};
use tauri::AppHandle;
use tauri_plugin_clipboard_manager::ClipboardExt;

/// Type a string at the current cursor as if the user typed it.
/// Used in stream mode for incremental partials.
#[allow(dead_code)]
pub fn type_text(s: &str) -> Result<()> {
    let mut enigo = Enigo::new(&Settings::default())?;
    enigo.text(s)?;
    Ok(())
}

/// One-shot paste: stash existing clipboard, copy `s`, send paste, restore.
/// This is what production dictation tools do — it's faster and more reliable
/// than typing each character on long transcripts.
///
/// Race protection: after writing, we read the clipboard back with an
/// exponential-backoff retry until our exact text is observed. Adaptive:
/// fast clipboard managers confirm in <10 ms; slow ones (Ditto, ClipX,
/// some accessibility tools) get all the time they need. There's a
/// 10-second hard cap as a safety net — if confirmation never lands the
/// clipboard subsystem is genuinely broken, and we paste anyway with a
/// loud warning rather than hang forever.
pub fn paste_text(app: &AppHandle, s: &str) -> Result<()> {
    let cb = app.clipboard();
    let backup = cb.read_text().ok();

    cb.write_text(s.to_string())?;

    // Adaptive verification loop: exponential backoff from 2 ms, capped
    // at 80 ms per poll. Each iteration we re-read; we exit the moment
    // the clipboard mirrors our exact target. Every fifth iteration we
    // re-write, in case some clipboard-manager app dropped the first
    // write while it had ownership.
    let target = s;
    let mut delay = std::time::Duration::from_millis(2);
    let max_delay = std::time::Duration::from_millis(80);
    let started = std::time::Instant::now();
    let safety_cap = std::time::Duration::from_secs(10);
    let mut iter = 0u32;
    let confirmed = loop {
        std::thread::sleep(delay);
        match cb.read_text() {
            Ok(current) if current == target => break true,
            Ok(_) | Err(_) => {
                // Either still stale or transient read failure. Retry.
            }
        }
        iter += 1;
        if iter % 5 == 0 {
            let _ = cb.write_text(target.to_string());
        }
        // Backoff: double up to the cap. Keeps tight loops cheap on
        // fast machines and gives slow clipboard chains time to settle
        // without burning cycles polling.
        delay = (delay * 2).min(max_delay);

        if started.elapsed() > safety_cap {
            break false;
        }
    };
    if confirmed {
        log::debug!(
            "paste_text: clipboard confirmed after {:?} ({} iters)",
            started.elapsed(),
            iter
        );
    } else {
        log::warn!(
            "paste_text: clipboard never confirmed in {:?}; pasting anyway",
            started.elapsed()
        );
    }

    let mut enigo = Enigo::new(&Settings::default())?;
    #[cfg(target_os = "macos")]
    let mod_key = Key::Meta;
    #[cfg(not(target_os = "macos"))]
    let mod_key = Key::Control;

    enigo.key(mod_key, Direction::Press)?;
    enigo.key(Key::Unicode('v'), Direction::Click)?;
    enigo.key(mod_key, Direction::Release)?;

    // Restore the user's previous clipboard contents. CRITICAL: there's
    // no API for "the foreground app finished consuming this paste,"
    // so we have to give the OS + target app enough time to deliver
    // the synthesized Ctrl+V keystroke AND for the target's clipboard
    // read to complete BEFORE we overwrite. If we restore too early,
    // the target reads the BACKUP we just put back and pastes that
    // instead of the transcript — the exact "old clipboard content"
    // bug. 500 ms is safely beyond every well-behaved app's keyboard
    // → clipboard latency on a healthy machine.
    //
    // Belt-and-braces: re-read the clipboard right before restoring;
    // if some other app changed it in the meantime (rare but real
    // with clipboard managers), leave their value alone.
    if let Some(prev) = backup {
        std::thread::sleep(std::time::Duration::from_millis(500));
        match cb.read_text() {
            Ok(current) if current == target => {
                let _ = cb.write_text(prev);
                log::debug!("paste_text: clipboard restored to previous contents");
            }
            Ok(_other) => {
                log::debug!(
                    "paste_text: clipboard changed under us during paste — leaving it alone"
                );
            }
            Err(_) => {
                // Read failed — best-effort restore.
                let _ = cb.write_text(prev);
            }
        }
    }
    Ok(())
}
