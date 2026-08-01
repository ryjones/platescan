use std::io::{BufReader, Read};
use std::path::Path;
use std::process::{Child, ChildStdout, Command, Stdio};

use anyhow::{bail, Context, Result};
use image::ExtendedColorType;

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
    spec_start: f64,
    fps: f64,
    origin: (u32, u32),
    width: u32,
    height: u32,
    format: PixelFormat,
    index: u64,
    done: bool,
}

impl FrameStream {
    pub fn open(path: &Path, spec: &SampleSpec) -> Result<FrameStream> {
        let (x, y, w, h) = spec.roi;
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-hide_banner", "-nostdin", "-v", "error"]);
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
        Ok(FrameStream {
            child,
            stdout: BufReader::with_capacity(1 << 20, stdout),
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
        if !status.success() {
            let mut err = String::new();
            if let Some(mut stderr) = self.child.stderr.take() {
                let _ = stderr.read_to_string(&mut err);
            }
            bail!("ffmpeg exited with {status}: {}", err.trim());
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
}
