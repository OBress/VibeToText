use anyhow::Result;
#[cfg(not(target_os = "macos"))]
use enigo::Key;
use enigo::{Direction, Enigo, Keyboard, Settings};
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
        if iter.is_multiple_of(5) {
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

    send_paste_shortcut()?;

    // Restore the user's previous clipboard contents. CRITICAL: there's
    // no API for "the foreground app finished consuming this paste,"
    // so we have to give the OS + target app enough time to deliver
    // the synthesized Ctrl+V keystroke AND for the target's clipboard
    // read to complete BEFORE we overwrite. If we restore too early,
    // the target reads the BACKUP we just put back and pastes that
    // instead of the transcript — the exact "old clipboard content"
    // bug. 700 ms covers slow text editors + clipboard managers on
    // every machine I've tested.
    //
    // We restore UNCONDITIONALLY: previous versions of this code
    // re-read the clipboard before restoring and skipped restore if
    // the contents had changed, on the theory that "some other app
    // overwrote our paste, so respect their value." In practice this
    // mis-fired on Windows machines with clipboard history (Win+V),
    // password managers, and similar — those tools sometimes echo
    // our paste back into the clipboard via their own hook, which
    // looked like an unrelated change to our read and made us LEAVE
    // the dictation as the active clipboard. Result: the user's
    // pre-dictation clipboard got silently lost. Restoring without
    // a guard is the right trade-off; if the user truly wanted the
    // dictation in their clipboard they can copy it from the target
    // editor.
    if let Some(prev) = backup {
        std::thread::sleep(std::time::Duration::from_millis(700));
        if let Err(e) = cb.write_text(prev.clone()) {
            log::warn!("paste_text: clipboard restore write failed: {e}");
        } else {
            // Quick verify pass: read once to confirm we successfully
            // overwrote the dictation. If something is fighting us,
            // make one re-write attempt and accept whatever lands.
            std::thread::sleep(std::time::Duration::from_millis(40));
            match cb.read_text() {
                Ok(current) if current == prev => {
                    log::debug!("paste_text: clipboard restored to previous contents");
                }
                Ok(_other) => {
                    log::debug!(
                        "paste_text: restore landed but clipboard differs (manager hook?) — re-writing once"
                    );
                    let _ = cb.write_text(prev);
                }
                Err(e) => {
                    log::debug!("paste_text: post-restore read failed: {e}");
                }
            }
        }
    } else {
        log::debug!("paste_text: no previous clipboard text to restore (was image / file / empty)");
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn send_paste_shortcut() -> Result<()> {
    let mut enigo = Enigo::new(&Settings::default())?;
    // macOS keycode 55 = Command, 9 = V. Using raw keycodes avoids Enigo's
    // layout lookup path, which must run on the main dispatch queue.
    enigo.raw(55, Direction::Press)?;
    enigo.raw(9, Direction::Click)?;
    enigo.raw(55, Direction::Release)?;
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn send_paste_shortcut() -> Result<()> {
    let mut enigo = Enigo::new(&Settings::default())?;
    enigo.key(Key::Control, Direction::Press)?;
    enigo.key(Key::Unicode('v'), Direction::Click)?;
    enigo.key(Key::Control, Direction::Release)?;
    Ok(())
}
