use std::io::{BufReader, Read};
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use anyhow::{bail, Context, Result};
use image::ExtendedColorType;

/// How much of ffmpeg's complaining to keep for the error message. A clip
/// whose video rots emits one line per corrupt packet, which can run to
/// megabytes; the last few kilobytes say everything useful.
const STDERR_KEEP: usize = 8 << 10;

/// Pixel layout requested from ffmpeg. Grayscale is enough for OCR and moves a
/// third of the bytes; the ONNX models need colour.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    Gray,
    Rgb,
}

impl PixelFormat {
    fn ffmpeg(self) -> &'static str {
        match self {
            PixelFormat::Gray => "gray",
            PixelFormat::Rgb => "rgb24",
        }
    }

    pub fn channels(self) -> usize {
        match self {
            PixelFormat::Gray => 1,
            PixelFormat::Rgb => 3,
        }
    }
}

/// One sampled, ROI-cropped frame.
pub struct Frame {
    /// Seconds from the start of the clip.
    pub offset: f64,
    pub width: u32,
    pub height: u32,
    /// Top-left of this crop within the full source frame.
    pub origin: (u32, u32),
    pub data: Vec<u8>,
    pub format: PixelFormat,
}

impl Frame {
    /// Copy a window out of the frame as 8-bit grayscale.
    pub fn crop_gray(&self, ox: u32, oy: u32, w: u32, h: u32) -> image::GrayImage {
        let ch = self.format.channels();
        let stride = self.width as usize * ch;
        let mut buf = Vec::with_capacity((w * h) as usize);
        for row in 0..h as usize {
            let start = (oy as usize + row) * stride + ox as usize * ch;
            match self.format {
                PixelFormat::Gray => buf.extend_from_slice(&self.data[start..start + w as usize]),
                PixelFormat::Rgb => buf.extend(
                    self.data[start..start + w as usize * 3]
                        .chunks_exact(3)
                        // Rec. 601 luma, the same weighting ffmpeg would apply.
                        .map(|p| {
                            ((p[0] as u32 * 299 + p[1] as u32 * 587 + p[2] as u32 * 114) / 1000)
                                as u8
                        }),
                ),
            }
        }
        image::GrayImage::from_raw(w, h, buf).expect("buffer matches window dimensions")
    }

    /// Copy a window out of the frame as RGB, clamped to the frame bounds.
    pub fn crop_rgb(&self, ox: u32, oy: u32, w: u32, h: u32) -> image::RgbImage {
        let ox = ox.min(self.width.saturating_sub(1));
        let oy = oy.min(self.height.saturating_sub(1));
        let w = w.min(self.width - ox).max(1);
        let h = h.min(self.height - oy).max(1);
        let ch = self.format.channels();
        let stride = self.width as usize * ch;
        let mut buf = Vec::with_capacity((w * h * 3) as usize);
        for row in 0..h as usize {
            let start = (oy as usize + row) * stride + ox as usize * ch;
            match self.format {
                PixelFormat::Rgb => {
                    buf.extend_from_slice(&self.data[start..start + w as usize * 3])
                }
                PixelFormat::Gray => {
                    for &g in &self.data[start..start + w as usize] {
                        buf.extend_from_slice(&[g, g, g]);
                    }
                }
            }
        }
        image::RgbImage::from_raw(w, h, buf).expect("buffer matches window dimensions")
    }

    /// Encode a padded still around a detection, taken from this very frame.
    ///
    /// The crop has to come from the frame that was analysed. Re-decoding the
    /// clip afterwards with a seek lands on a neighbouring frame, and at road
    /// speeds a nearby vehicle moves far enough in that gap to put the box on
    /// its tail light instead of its plate.
    pub fn still_jpeg(&self, rect_in_source: (u32, u32, u32, u32)) -> Option<Vec<u8>> {
        let (sx, sy, w, h) = rect_in_source;
        // Detections are reported in full-frame coordinates; our pixels are
        // only the region of interest.
        let local = (
            sx.checked_sub(self.origin.0)?,
            sy.checked_sub(self.origin.1)?,
            w,
            h,
        );
        let (x, y, pw, ph) = pad_rect(local, (self.width, self.height), 2.4, 3.0);
        let crop = self.crop_rgb(x, y, pw, ph);
        let mut out = Vec::new();
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 92)
            .encode(
                crop.as_raw(),
                crop.width(),
                crop.height(),
                ExtendedColorType::Rgb8,
            )
            .ok()?;
        Some(out)
    }
}

/// Grow a box by the given factors, keeping it inside the frame.
fn pad_rect(
    (x, y, w, h): (u32, u32, u32, u32),
    (fw, fh): (u32, u32),
    fx: f64,
    fy: f64,
) -> (u32, u32, u32, u32) {
    let cx = x as f64 + w as f64 / 2.0;
    let cy = y as f64 + h as f64 / 2.0;
    let nw = (w as f64 * fx).max(240.0).min(fw as f64);
    let nh = (h as f64 * fy).max(120.0).min(fh as f64);
    let nx = (cx - nw / 2.0).clamp(0.0, fw as f64 - nw);
    let ny = (cy - nh / 2.0).clamp(0.0, fh as f64 - nh);
    let round = |v: f64| v.round() as u32;
    (round(nx), round(ny), round(nw).max(1), round(nh).max(1))
}

pub struct SampleSpec {
    pub start: f64,
    pub end: Option<f64>,
    pub fps: f64,
    /// x, y, w, h in source pixels.
    pub roi: (u32, u32, u32, u32),
    pub format: PixelFormat,
}

/// Streams decoded frames out of ffmpeg one at a time so a 4K clip never needs
/// to be held in memory.
pub struct FrameStream {
    child: Child,
    stdout: BufReader<ChildStdout>,
    /// ffmpeg's stderr, drained by a thread for as long as it runs. Reading it
    /// only at the end would deadlock: a pipe holds 64KB, and a clip with a
    /// corrupt tail writes past that, at which point ffmpeg blocks on the write
    /// and stops producing frames while we wait for a frame that never comes.
    /// Both processes then sit at no CPU indefinitely.
    stderr: Arc<Mutex<String>>,
    drain: Option<JoinHandle<()>>,
    spec_start: f64,
    fps: f64,
    origin: (u32, u32),
    width: u32,
    height: u32,
    format: PixelFormat,
    index: u64,
    done: bool,
}

/// Read a child's stderr to exhaustion on a thread, keeping the tail.
fn drain_stderr(mut stderr: impl Read + Send + 'static, into: Arc<Mutex<String>>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buf = [0u8; 8 << 10];
        loop {
            match stderr.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let mut held = into.lock().unwrap_or_else(|e| e.into_inner());
                    held.push_str(&String::from_utf8_lossy(&buf[..n]));
                    if held.len() > STDERR_KEEP {
                        // keep the tail, on a character boundary
                        let cut = held.len() - STDERR_KEEP;
                        let cut = (cut..held.len())
                            .find(|i| held.is_char_boundary(*i))
                            .unwrap_or(held.len());
                        *held = held[cut..].to_string();
                    }
                }
            }
        }
    })
}

impl FrameStream {
    pub fn open(path: &Path, spec: &SampleSpec) -> Result<FrameStream> {
        let (x, y, w, h) = spec.roi;
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-hide_banner", "-nostdin", "-v", "error"]);
        // Hardware decode via the media engine; ffmpeg falls back to
        // software on its own for codecs VideoToolbox does not take.
        #[cfg(target_os = "macos")]
        cmd.args(["-hwaccel", "videotoolbox"]);
        cmd.args(["-ss", &format!("{:.3}", spec.start)]);
        cmd.arg("-i").arg(path);
        if let Some(end) = spec.end {
            cmd.args(["-t", &format!("{:.3}", (end - spec.start).max(0.0))]);
        }
        cmd.args(["-an", "-sn", "-dn"]);
        cmd.args([
            "-vf",
            &format!("fps={},crop={w}:{h}:{x}:{y}", spec.fps),
            "-f",
            "rawvideo",
            "-pix_fmt",
            spec.format.ffmpeg(),
            "-",
        ]);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

        let mut child = cmd.spawn().context("failed to run ffmpeg")?;
        let stdout = child.stdout.take().expect("stdout was piped");
        let stderr = Arc::new(Mutex::new(String::new()));
        // Start draining immediately; ffmpeg must never block writing errors.
        let drain = child
            .stderr
            .take()
            .map(|e| drain_stderr(e, Arc::clone(&stderr)));
        Ok(FrameStream {
            child,
            stdout: BufReader::with_capacity(1 << 20, stdout),
            stderr,
            drain,
            spec_start: spec.start,
            fps: spec.fps,
            origin: (x, y),
            width: w,
            height: h,
            format: spec.format,
            index: 0,
            done: false,
        })
    }

    /// Number of frames this stream is expected to yield, for progress display.
    pub fn expected(spec: &SampleSpec, duration: f64) -> u64 {
        let end = spec.end.unwrap_or(duration).min(duration);
        (((end - spec.start).max(0.0)) * spec.fps).round() as u64
    }

    pub fn next_frame(&mut self) -> Result<Option<Frame>> {
        if self.done {
            return Ok(None);
        }
        let mut data =
            vec![0u8; (self.width as usize) * (self.height as usize) * self.format.channels()];
        match self.stdout.read_exact(&mut data) {
            Ok(()) => {
                let frame = Frame {
                    offset: self.spec_start + self.index as f64 / self.fps,
                    width: self.width,
                    height: self.height,
                    origin: self.origin,
                    data,
                    format: self.format,
                };
                self.index += 1;
                Ok(Some(frame))
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                self.done = true;
                self.finish()?;
                Ok(None)
            }
            Err(e) => Err(e).context("reading frames from ffmpeg"),
        }
    }

    fn finish(&mut self) -> Result<()> {
        let status = self.child.wait()?;
        // The drain ends when the pipe closes, which wait() has just ensured.
        if let Some(h) = self.drain.take() {
            let _ = h.join();
        }
        if !status.success() {
            let err = self
                .stderr
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .trim()
                .to_string();
            bail!("ffmpeg exited with {status}: {err}");
        }
        Ok(())
    }
}

impl Drop for FrameStream {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
        // Killing the child closes the pipe, so the drain returns on its own.
        if let Some(h) = self.drain.take() {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: u32, h: u32, origin: (u32, u32)) -> Frame {
        Frame {
            offset: 0.0,
            width: w,
            height: h,
            origin,
            data: vec![128u8; (w * h * 3) as usize],
            format: PixelFormat::Rgb,
        }
    }

    #[test]
    fn pad_stays_inside_frame() {
        let (x, y, w, h) = pad_rect((3800, 2100, 100, 40), (3840, 2160), 2.4, 3.0);
        assert!(x + w <= 3840, "x={x} w={w}");
        assert!(y + h <= 2160, "y={y} h={h}");
    }

    #[test]
    fn pad_enforces_minimum_size() {
        let (_, _, w, h) = pad_rect((100, 100, 20, 8), (3840, 2160), 2.4, 3.0);
        assert!(w >= 240 && h >= 120);
    }

    #[test]
    fn still_is_taken_from_the_analysed_frame() {
        // ROI starting at (192, 950) in the source.
        let f = frame(3456, 540, (192, 950));
        let jpeg = f.still_jpeg((1000, 1200, 80, 40)).expect("encodes");
        assert_eq!(&jpeg[..2], &[0xFF, 0xD8], "JPEG magic");
    }

    #[test]
    fn detections_outside_the_roi_yield_no_still() {
        let f = frame(3456, 540, (192, 950));
        // Above the region of interest, so it cannot have come from this frame.
        assert!(f.still_jpeg((1000, 100, 80, 40)).is_none());
    }

    #[test]
    fn gray_frames_still_produce_rgb_crops() {
        let f = Frame {
            offset: 0.0,
            width: 640,
            height: 480,
            origin: (0, 0),
            data: vec![200u8; 640 * 480],
            format: PixelFormat::Gray,
        };
        let crop = f.crop_rgb(10, 10, 4, 2);
        assert_eq!(crop.as_raw().len(), 4 * 2 * 3);
        assert!(crop.as_raw().iter().all(|&v| v == 200));
    }

    /// A clip whose video rots writes one complaint per corrupt packet, easily
    /// more than the 64KB a pipe holds. Reading stderr only at the end would
    /// wedge ffmpeg mid-write and stall the frame stream forever, so the drain
    /// has to keep up while the process runs.
    #[test]
    fn a_flood_of_stderr_does_not_block_the_writer() {
        use std::io::Write;
        use std::time::{Duration, Instant};

        const FLOOD: usize = 512 << 10; // 8x a pipe buffer

        let mut child = Command::new("sh")
            .arg("-c")
            .arg(format!(
                "i=0; while [ $i -lt {} ]; do                    printf '[h264] Invalid NAL unit size (2034819075 > 98454).\n' >&2;                    i=$((i+1)); done; printf 'done'",
                FLOOD / 64
            ))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn");

        let held = Arc::new(Mutex::new(String::new()));
        let drain = drain_stderr(child.stderr.take().unwrap(), Arc::clone(&held));

        // Without a concurrent drain the child blocks and this never returns.
        let start = Instant::now();
        let mut out = String::new();
        child.stdout.take().unwrap().read_to_string(&mut out).expect("read stdout");
        let status = child.wait().expect("wait");
        drain.join().expect("drain");

        assert!(status.success(), "child should exit cleanly");
        assert_eq!(out, "done", "stdout must still be readable past the flood");
        assert!(start.elapsed() < Duration::from_secs(60), "took too long, likely blocked");

        let kept = held.lock().unwrap();
        assert!(kept.contains("Invalid NAL unit size"), "the tail is kept for the error message");
        assert!(kept.len() <= STDERR_KEEP + 8192, "and it is bounded, got {}", kept.len());
    }
}
