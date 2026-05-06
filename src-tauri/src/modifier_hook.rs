// Global keyboard input listener — Raw Input edition.
//
// Earlier versions used WH_KEYBOARD_LL, which has a hard timeout
// (LowLevelHooksTimeout, default 300 ms): if the callback ever takes
// too long, Windows silently removes the hook with no notification
// and your app loses its global hotkey forever. We caught this with
// a 30 s reinstall watchdog, but the right fix is to use an API that
// has no timeout in the first place.
//
// Raw Input (RegisterRawInputDevices + WM_INPUT) is exactly that:
// the OS posts keyboard events as window messages with no timeout.
// We get every key system-wide (RIDEV_INPUTSINK = even when not
// focused), and there's no per-callback budget for Windows to
// enforce.
//
// Architecture:
//   1. A worker thread creates a hidden message-only window.
//   2. It registers for raw keyboard input on that window.
//   3. The window proc handles WM_INPUT, parses RAWKEYBOARD, updates
//      atomic state, and runs the same state machine as before.
//   4. The state machine fires a debounced press/release through an
//      mpsc channel to a Tauri-side dispatcher thread.
//
// Compatible with the rest of the crate: same WATCH_MASK API,
// same start_dictation/stop_dictation events, same canonical
// modifier names. Only the source-of-truth for "did a key change"
// is different.

#![cfg(target_os = "windows")]

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU32, AtomicU8, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Mutex, OnceLock};
use tauri::AppHandle;

use windows_sys::Win32::Foundation::{HINSTANCE, HWND, LPARAM, LRESULT, WPARAM};
use windows_sys::Win32::System::LibraryLoader::GetModuleHandleW;
use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, VK_CONTROL, VK_LCONTROL, VK_LMENU, VK_LSHIFT, VK_LWIN, VK_MENU,
    VK_RCONTROL, VK_RMENU, VK_RSHIFT, VK_RWIN, VK_SHIFT,
};
use windows_sys::Win32::UI::Input::{
    GetRawInputData, RegisterRawInputDevices, HRAWINPUT, RAWINPUT, RAWINPUTDEVICE,
    RAWINPUTHEADER, RID_INPUT, RIDEV_INPUTSINK, RIM_TYPEKEYBOARD,
};
use windows_sys::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DispatchMessageW, GetMessageW, RegisterClassExW,
    TranslateMessage, CW_USEDEFAULT, HWND_MESSAGE, MSG, WM_INPUT, WNDCLASSEXW,
};

// Raw keyboard event flags.
const RI_KEY_BREAK: u16 = 0x01; // 0 = make (down), 1 = break (up)
const RI_KEY_E0: u16 = 0x02; // extended scancode (right-side modifiers, etc.)

// Public canonical-modifier mask values.
pub const MOD_CTRL: u32 = 1 << 0;
pub const MOD_SHIFT: u32 = 1 << 1;
pub const MOD_ALT: u32 = 1 << 2;
pub const MOD_WIN: u32 = 1 << 3;

// Per-side "is this exact key held" bits, tracked from Raw Input events.
const D_LCTRL: u8 = 1 << 0;
const D_RCTRL: u8 = 1 << 1;
const D_LSHIFT: u8 = 1 << 2;
const D_RSHIFT: u8 = 1 << 3;
const D_LALT: u8 = 1 << 4;
const D_RALT: u8 = 1 << 5;
const D_LWIN: u8 = 1 << 6;
const D_RWIN: u8 = 1 << 7;

static WATCH_MASK: AtomicU32 = AtomicU32::new(0);
static HELD_DETAIL: AtomicU8 = AtomicU8::new(0);
static OTHER_KEYS_HELD: AtomicU32 = AtomicU32::new(0);
static INVALIDATED: AtomicBool = AtomicBool::new(false);
static STARTED: AtomicBool = AtomicBool::new(false);
static WAS_MATCHING: AtomicBool = AtomicBool::new(false);
static START_GEN: AtomicU32 = AtomicU32::new(0);
static SENDER: OnceLock<Mutex<Sender<bool>>> = OnceLock::new();
static LAST_EVENT_MS: AtomicI64 = AtomicI64::new(0);

const START_DEBOUNCE_MS: u64 = 45;

fn now_millis() -> i64 {
    use std::time::SystemTime;
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

pub fn mask_for_modifier_only(keys: &std::collections::BTreeSet<String>) -> Option<u32> {
    let mut mask = 0u32;
    for k in keys {
        match k.as_str() {
            "ctrl" => mask |= MOD_CTRL,
            "shift" => mask |= MOD_SHIFT,
            "alt" => mask |= MOD_ALT,
            "meta" => mask |= MOD_WIN,
            _ => return None,
        }
    }
    if mask == 0 {
        None
    } else {
        Some(mask)
    }
}

pub fn set_watch_mask(mask: u32) {
    let prev = WATCH_MASK.swap(mask, Ordering::SeqCst);
    if prev != mask {
        log::info!("modifier_hook: watch_mask = {mask:#x}");
    }
    INVALIDATED.store(false, Ordering::SeqCst);
    STARTED.store(false, Ordering::SeqCst);
    WAS_MATCHING.store(false, Ordering::SeqCst);
    START_GEN.fetch_add(1, Ordering::SeqCst);
}

/// Install the global Raw Input listener. Idempotent.
pub fn install(app: AppHandle) {
    static INSTALLED: AtomicBool = AtomicBool::new(false);
    if INSTALLED.swap(true, Ordering::SeqCst) {
        return;
    }

    let (tx, rx) = channel::<bool>();
    if SENDER.set(Mutex::new(tx)).is_err() {
        log::warn!("modifier_hook: sender already initialized");
        return;
    }

    // Worker that talks to Tauri's async runtime.
    std::thread::spawn(move || {
        for is_press in rx.iter() {
            let app = app.clone();
            tauri::async_runtime::spawn(async move {
                if is_press {
                    let _ = crate::start_dictation(&app).await;
                } else {
                    let _ = crate::stop_dictation(&app).await;
                }
            });
        }
    });

    // Raw Input thread: create a message-only window, register the
    // keyboard device, and pump WM_INPUT messages.
    std::thread::spawn(|| unsafe {
        let h_instance: HINSTANCE = GetModuleHandleW(std::ptr::null());

        // Class name (UTF-16 + null terminator).
        let class_name: Vec<u16> = "VibeToTextRawInput\0".encode_utf16().collect();

        let wc = WNDCLASSEXW {
            cbSize: std::mem::size_of::<WNDCLASSEXW>() as u32,
            style: 0,
            lpfnWndProc: Some(wnd_proc),
            cbClsExtra: 0,
            cbWndExtra: 0,
            hInstance: h_instance,
            hIcon: std::ptr::null_mut(),
            hCursor: std::ptr::null_mut(),
            hbrBackground: std::ptr::null_mut(),
            lpszMenuName: std::ptr::null(),
            lpszClassName: class_name.as_ptr(),
            hIconSm: std::ptr::null_mut(),
        };
        if RegisterClassExW(&wc) == 0 {
            log::error!("modifier_hook: RegisterClassExW failed");
            return;
        }

        let title: Vec<u16> = "VibeToText raw input\0".encode_utf16().collect();
        let hwnd: HWND = CreateWindowExW(
            0,
            class_name.as_ptr(),
            title.as_ptr(),
            0,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            0,
            0,
            HWND_MESSAGE, // message-only window — never visible, no taskbar entry
            std::ptr::null_mut(),
            h_instance,
            std::ptr::null(),
        );
        if hwnd.is_null() {
            log::error!("modifier_hook: CreateWindowExW failed");
            return;
        }

        // Register for raw keyboard input. Usage page 0x01 (generic
        // desktop), usage 0x06 (keyboard). RIDEV_INPUTSINK delivers
        // events to us even when our window doesn't have focus —
        // critical for global hotkeys.
        let rid = RAWINPUTDEVICE {
            usUsagePage: 0x01,
            usUsage: 0x06,
            dwFlags: RIDEV_INPUTSINK,
            hwndTarget: hwnd,
        };
        let ok = RegisterRawInputDevices(
            &rid,
            1,
            std::mem::size_of::<RAWINPUTDEVICE>() as u32,
        );
        if ok == 0 {
            log::error!("modifier_hook: RegisterRawInputDevices failed");
            return;
        }
        log::info!("modifier_hook: raw input listener installed");
        LAST_EVENT_MS.store(now_millis(), Ordering::SeqCst);

        // Standard message pump. WM_INPUT messages are dispatched to
        // wnd_proc which does the actual work.
        let mut msg: MSG = std::mem::zeroed();
        while GetMessageW(&mut msg, std::ptr::null_mut(), 0, 0) > 0 {
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    });
}

unsafe extern "system" fn wnd_proc(
    hwnd: HWND,
    msg: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if msg == WM_INPUT {
        process_raw_input(lparam as HRAWINPUT);
        // Per docs, apps should still call DefWindowProcW for WM_INPUT
        // so the OS can clean up.
        return DefWindowProcW(hwnd, msg, wparam, lparam);
    }
    DefWindowProcW(hwnd, msg, wparam, lparam)
}

unsafe fn process_raw_input(h: HRAWINPUT) {
    let mut size: u32 = 0;
    let header_size = std::mem::size_of::<RAWINPUTHEADER>() as u32;
    // First call with NULL buffer to learn the size.
    let r = GetRawInputData(h, RID_INPUT, std::ptr::null_mut(), &mut size, header_size);
    if r == u32::MAX || size == 0 {
        return;
    }
    let mut buf: Vec<u8> = vec![0; size as usize];
    let r = GetRawInputData(
        h,
        RID_INPUT,
        buf.as_mut_ptr() as *mut _,
        &mut size,
        header_size,
    );
    if r == u32::MAX {
        return;
    }
    let raw = &*(buf.as_ptr() as *const RAWINPUT);
    if raw.header.dwType != RIM_TYPEKEYBOARD {
        return;
    }
    let kb = &raw.data.keyboard;

    LAST_EVENT_MS.store(now_millis(), Ordering::Relaxed);

    if WATCH_MASK.load(Ordering::Relaxed) == 0 {
        return;
    }

    // Determine which side of which modifier (or non-modifier).
    let is_down = (kb.Flags & RI_KEY_BREAK) == 0;
    let has_e0 = (kb.Flags & RI_KEY_E0) != 0;
    let mod_bit: Option<u8> = match kb.VKey {
        v if v == VK_LCONTROL => Some(D_LCTRL),
        v if v == VK_RCONTROL => Some(D_RCTRL),
        v if v == VK_LSHIFT => Some(D_LSHIFT),
        v if v == VK_RSHIFT => Some(D_RSHIFT),
        v if v == VK_LMENU => Some(D_LALT),
        v if v == VK_RMENU => Some(D_RALT),
        v if v == VK_LWIN => Some(D_LWIN),
        v if v == VK_RWIN => Some(D_RWIN),
        // Generic shift/ctrl/alt — distinguish L/R via scancode + E0.
        v if v == VK_SHIFT => {
            // LShift make code = 0x2A, RShift = 0x36.
            if kb.MakeCode == 0x36 {
                Some(D_RSHIFT)
            } else {
                Some(D_LSHIFT)
            }
        }
        v if v == VK_CONTROL => {
            if has_e0 {
                Some(D_RCTRL)
            } else {
                Some(D_LCTRL)
            }
        }
        v if v == VK_MENU => {
            if has_e0 {
                Some(D_RALT)
            } else {
                Some(D_LALT)
            }
        }
        _ => None,
    };

    if let Some(b) = mod_bit {
        let prev = HELD_DETAIL.load(Ordering::Relaxed);
        let next = if is_down { prev | b } else { prev & !b };
        HELD_DETAIL.store(next, Ordering::Relaxed);
    } else if is_down {
        OTHER_KEYS_HELD.fetch_add(1, Ordering::Relaxed);
    } else if OTHER_KEYS_HELD.load(Ordering::Relaxed) > 0 {
        OTHER_KEYS_HELD.fetch_sub(1, Ordering::Relaxed);
    }

    recompute_state();
}

fn detail_to_mask(d: u8) -> u32 {
    let mut m = 0u32;
    if d & (D_LCTRL | D_RCTRL) != 0 {
        m |= MOD_CTRL;
    }
    if d & (D_LSHIFT | D_RSHIFT) != 0 {
        m |= MOD_SHIFT;
    }
    if d & (D_LALT | D_RALT) != 0 {
        m |= MOD_ALT;
    }
    if d & (D_LWIN | D_RWIN) != 0 {
        m |= MOD_WIN;
    }
    m
}

/// Press-cycle state machine. Driven on every relevant key event,
/// acts only on FALSE↔TRUE transitions of the matching state so
/// auto-repeat keydowns can't reset the debounce timer.
fn recompute_state() {
    let want = WATCH_MASK.load(Ordering::Relaxed);
    if want == 0 {
        return;
    }
    let detail = HELD_DETAIL.load(Ordering::Relaxed);
    let held = detail_to_mask(detail);
    let other = OTHER_KEYS_HELD.load(Ordering::Relaxed);

    if detail == 0 && other == 0 {
        INVALIDATED.store(false, Ordering::Relaxed);
    }

    let now_matching = (held == want) && other == 0;
    let prefix_collision = (held & want) == want && other > 0;
    let was = WAS_MATCHING.load(Ordering::Relaxed);
    let inv = INVALIDATED.load(Ordering::Relaxed);
    log::trace!(
        "modifier_hook: recompute want={want:#x} held={held:#x} other={other} now_matching={now_matching} prefix_collision={prefix_collision} was={was} invalidated={inv}"
    );

    if prefix_collision {
        if !inv {
            log::debug!(
                "modifier_hook: prefix collision (held={held:#x} other={other}), invalidating"
            );
        }
        INVALIDATED.store(true, Ordering::Relaxed);
        WAS_MATCHING.store(false, Ordering::Relaxed);
        START_GEN.fetch_add(1, Ordering::Relaxed);
        if STARTED.swap(false, Ordering::Relaxed) {
            send(false);
        }
        return;
    }

    if now_matching && !was {
        WAS_MATCHING.store(true, Ordering::Relaxed);
        if INVALIDATED.load(Ordering::Relaxed) {
            log::debug!("modifier_hook: matching but INVALIDATED — skipping press");
            return;
        }
        log::debug!("modifier_hook: combo matched, scheduling press after debounce");
        let my_gen = START_GEN.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(START_DEBOUNCE_MS));
            if START_GEN.load(Ordering::Relaxed) != my_gen {
                log::debug!("modifier_hook: debounce: gen changed, dropping press");
                return;
            }
            if INVALIDATED.load(Ordering::Relaxed) {
                log::debug!("modifier_hook: debounce: invalidated mid-debounce, dropping press");
                return;
            }
            let still_held = detail_to_mask(HELD_DETAIL.load(Ordering::Relaxed));
            let still_other = OTHER_KEYS_HELD.load(Ordering::Relaxed);
            if still_held != WATCH_MASK.load(Ordering::Relaxed) || still_other != 0 {
                log::debug!(
                    "modifier_hook: debounce: state changed (held={still_held:#x} other={still_other}), dropping press"
                );
                return;
            }
            if !STARTED.swap(true, Ordering::Relaxed) {
                log::debug!("modifier_hook: debounce: firing press");
                send(true);
            }
        });
    } else if !now_matching && was {
        WAS_MATCHING.store(false, Ordering::Relaxed);
        START_GEN.fetch_add(1, Ordering::Relaxed);
        if STARTED.swap(false, Ordering::Relaxed) {
            log::debug!("modifier_hook: combo released, firing release");
            send(false);
        }
    }
}

fn send(is_press: bool) {
    if let Some(s) = SENDER.get() {
        let _ = s.lock().unwrap().send(is_press);
    }
}

/// Belt-and-braces: query GetAsyncKeyState once on demand so a future
/// "is the key actually held?" diagnostic doesn't lie. Currently
/// unused but kept for future re-sync calls.
#[allow(dead_code)]
fn resync_from_async_keystate() {
    unsafe {
        let mut detail: u8 = 0;
        let pairs: [(i32, u8); 8] = [
            (VK_LCONTROL as i32, D_LCTRL),
            (VK_RCONTROL as i32, D_RCTRL),
            (VK_LSHIFT as i32, D_LSHIFT),
            (VK_RSHIFT as i32, D_RSHIFT),
            (VK_LMENU as i32, D_LALT),
            (VK_RMENU as i32, D_RALT),
            (VK_LWIN as i32, D_LWIN),
            (VK_RWIN as i32, D_RWIN),
        ];
        for (vk, bit) in pairs {
            if (GetAsyncKeyState(vk) as u16 & 0x8000) != 0 {
                detail |= bit;
            }
        }
        HELD_DETAIL.store(detail, Ordering::SeqCst);
    }
}
