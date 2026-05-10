use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossbeam_channel::{bounded, Receiver, Sender};
use serde::Serialize;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// One audio frame (resampled to 16 kHz mono f32 in [-1, 1]).
pub struct Frame(pub Vec<f32>);

/// One short slice of "loudness" — RMS amplitude in [0, 1]. Used to
/// drive the live waveform overlay; produced at the same cadence as
/// frames (~50 Hz). Cheap to compute and cheap to drop on the floor
/// if the UI consumer is slow.
#[derive(Clone, Copy, Debug)]
pub struct Level(pub f32);

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InputDeviceInfo {
    pub name: String,
    pub is_default: bool,
}

pub struct AudioCapture {
    stop: Arc<AtomicBool>,
    pub frames: Receiver<Frame>,
    pub levels: Receiver<Level>,
    _thread: std::thread::JoinHandle<()>,
}

impl AudioCapture {
    /// Start capturing from `device_name` (empty = default). Frames arrive
    /// at ~16 kHz mono in chunks of ~20ms (320 samples). Levels arrive
    /// at the same rate, one per frame.
    pub fn start(device_name: &str) -> Result<Self> {
        let host = cpal::default_host();
        let device = if device_name.is_empty() {
            host.default_input_device()
                .ok_or_else(|| anyhow!("no default input device"))?
        } else {
            host.input_devices()?
                .find(|d| d.name().map(|n| n == device_name).unwrap_or(false))
                .ok_or_else(|| anyhow!("input device '{}' not found", device_name))?
        };
        let config = device.default_input_config()?;
        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;

        let (frame_tx, frame_rx) = bounded::<Frame>(256);
        // Levels channel is small + lossy by design. The overlay only
        // needs the latest few; backpressure on this would be a UI bug.
        let (level_tx, level_rx) = bounded::<Level>(64);
        let stop = Arc::new(AtomicBool::new(false));
        let stop_t = stop.clone();

        let thread = std::thread::spawn(move || {
            if let Err(e) = run_stream(
                device,
                config,
                channels,
                sample_rate,
                frame_tx,
                level_tx,
                stop_t,
            ) {
                log::error!("audio capture error: {e:#}");
            }
        });

        Ok(Self {
            stop,
            frames: frame_rx,
            levels: level_rx,
            _thread: thread,
        })
    }

    pub fn stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }
}

pub fn input_devices() -> Result<Vec<InputDeviceInfo>> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());
    let mut devices = Vec::new();

    for device in host.input_devices()? {
        let Ok(name) = device.name() else {
            continue;
        };
        let is_default = default_name.as_ref() == Some(&name);
        devices.push(InputDeviceInfo { name, is_default });
    }

    devices.sort_by(|a, b| match (a.is_default, b.is_default) {
        (true, false) => std::cmp::Ordering::Less,
        (false, true) => std::cmp::Ordering::Greater,
        _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
    });
    Ok(devices)
}

fn run_stream(
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
    channels: usize,
    sample_rate: u32,
    frame_tx: Sender<Frame>,
    level_tx: Sender<Level>,
    stop: Arc<AtomicBool>,
) -> Result<()> {
    let target_rate = 16_000u32;
    let stream_config: cpal::StreamConfig = config.clone().into();
    let sample_format = config.sample_format();

    let frame_tx_cb = frame_tx.clone();
    let level_tx_cb = level_tx.clone();
    let stop_cb = stop.clone();

    let err_fn = |e| log::error!("input stream error: {e}");

    let mut buf: Vec<f32> = Vec::with_capacity(8192);

    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &stream_config,
            move |data: &[f32], _| {
                handle(
                    data,
                    channels,
                    sample_rate,
                    target_rate,
                    &mut buf,
                    &frame_tx_cb,
                    &level_tx_cb,
                    &stop_cb,
                )
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &stream_config,
            move |data: &[i16], _| {
                let f: Vec<f32> = data.iter().map(|&s| s as f32 / i16::MAX as f32).collect();
                handle(
                    &f,
                    channels,
                    sample_rate,
                    target_rate,
                    &mut buf,
                    &frame_tx_cb,
                    &level_tx_cb,
                    &stop_cb,
                )
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::U16 => device.build_input_stream(
            &stream_config,
            move |data: &[u16], _| {
                let f: Vec<f32> = data
                    .iter()
                    .map(|&s| (s as f32 - 32_768.0) / 32_768.0)
                    .collect();
                handle(
                    &f,
                    channels,
                    sample_rate,
                    target_rate,
                    &mut buf,
                    &frame_tx_cb,
                    &level_tx_cb,
                    &stop_cb,
                )
            },
            err_fn,
            None,
        )?,
        fmt => return Err(anyhow!("unsupported sample format: {fmt:?}")),
    };

    stream.play()?;
    while !stop.load(Ordering::SeqCst) {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    drop(stream);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn handle(
    data: &[f32],
    channels: usize,
    src_rate: u32,
    dst_rate: u32,
    buf: &mut Vec<f32>,
    frame_tx: &Sender<Frame>,
    level_tx: &Sender<Level>,
    stop: &Arc<AtomicBool>,
) {
    if stop.load(Ordering::SeqCst) {
        return;
    }
    let mono: Vec<f32> = if channels == 1 {
        data.to_vec()
    } else {
        data.chunks(channels)
            .map(|c| c.iter().sum::<f32>() / channels as f32)
            .collect()
    };
    // Naive linear resample to 16 kHz.
    let ratio = dst_rate as f32 / src_rate as f32;
    let out_len = (mono.len() as f32 * ratio) as usize;
    for i in 0..out_len {
        let src_idx = i as f32 / ratio;
        let i0 = src_idx.floor() as usize;
        let i1 = (i0 + 1).min(mono.len().saturating_sub(1));
        let t = src_idx - i0 as f32;
        let v = mono[i0] * (1.0 - t) + mono[i1] * t;
        buf.push(v);
    }
    while buf.len() >= 320 {
        let frame: Vec<f32> = buf.drain(..320).collect();

        // RMS for the visualizer. Computed in the audio thread (cheap)
        // and dropped if nobody's listening.
        let mut sum_sq = 0.0f32;
        for &s in &frame {
            sum_sq += s * s;
        }
        let rms = (sum_sq / frame.len() as f32).sqrt();
        let _ = level_tx.try_send(Level(rms));

        if frame_tx.try_send(Frame(frame)).is_err() {
            log::warn!("STT consumer is behind; dropping frame");
        }
    }
}
